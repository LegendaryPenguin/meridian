#include "meridian/index.hpp"

#include <algorithm>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <unordered_set>

#include "../storage/file.hpp"
#include "builder.hpp"
#include "segment.hpp"

namespace meridian {

// ---------------------------------------------------------------------------
// Where the Index's state lives, and why it lives there
// ---------------------------------------------------------------------------
//
// The state below lives in `Index::Impl`, declared opaquely in the header and defined
// here, because every member of it is spelled in terms of types private to this directory
// — `SegmentBuilder`, `SegmentReaderImpl` — and the public header has no business naming
// the storage format.
//
// It was briefly a side table keyed by the `Index*`, to avoid touching a header that was
// supposed to be frozen. That is worth recording as a thing not to do: it put the object's
// own state outside the object, so lifetime stopped being the destructor's business, and
// it added a process-wide mutex that every construction and destruction serialised on in
// order to express a plain member. The pimpl deletes the table, the lock and the lookup.

namespace {

using index::SegmentBuilder;
using index::SegmentReaderImpl;

struct PendingRemoval {
    std::shared_ptr<SegmentReaderImpl> reader;
    std::string path;
};


std::string SegmentPath(const std::string& directory, uint64_t id) {
    std::ostringstream os;
    os << directory << "/segment-" << id << ".mrd";
    return os.str();
}

std::string ManifestPath(const std::string& directory) { return directory + "/MANIFEST"; }

SegmentBuilder::Options BuilderOptions(const Index::Options& options) {
    SegmentBuilder::Options builder_options;
    builder_options.positions[static_cast<size_t>(Field::kTitle)] = options.positions_for_title;
    builder_options.positions[static_cast<size_t>(Field::kBody)] = options.positions_for_body;
    builder_options.positions[static_cast<size_t>(Field::kTags)] = options.positions_for_tags;
    return builder_options;
}

/// Size tier of a segment. Tier boundaries are powers of merge_factor, so a tier holds
/// segments within a factor of merge_factor of each other and merging merge_factor of
/// them produces exactly one segment of the next tier up. That is what makes the policy
/// size-tiered rather than "merge the smallest few": merging segments of wildly different
/// sizes rewrites the large one to absorb the small one, which is where leveled
/// compaction's write cost comes from.
size_t TierOf(uint64_t bytes, uint32_t merge_factor) {
    constexpr uint64_t kBaseBytes = 1u << 16;
    const uint64_t factor = std::max<uint32_t>(2, merge_factor);
    size_t tier = 0;
    uint64_t limit = kBaseBytes;
    while (bytes > limit && tier < 32) {
        limit *= factor;
        ++tier;
    }
    return tier;
}

}  // namespace

struct Index::Impl {
    bool opened = false;

    /// Documents added but not yet written to a segment.
    std::unique_ptr<SegmentBuilder> buffer;
    std::unordered_set<std::string> buffered_ids;
    /// Deletes that named a buffered document. They cannot be applied until the document
    /// has a segment-local id, so they wait for the flush that gives it one.
    std::unordered_set<std::string> deferred_deletes;

    /// Ordered by segment id, which is also creation order.
    std::vector<std::shared_ptr<SegmentReaderImpl>> segments;

    /// external id -> (segment id, segment-local DocId). Deletion needs it, and nothing
    /// else does, which is why it is here and not in the segment file: a segment would
    /// have to store a sorted term-like structure over external ids to answer the same
    /// question, and deletes are rare next to queries.
    std::unordered_map<std::string, std::pair<uint64_t, DocId>> locator;

    /// Segments replaced by a merge, kept alive until the last reader lets go.
    std::vector<PendingRemoval> pending_removals;

    uint64_t next_segment_id = 1;

    /// Set while a merge is building its output outside the lock, so two concurrent
    /// MaybeMerge calls cannot pick the same inputs.
    bool merging = false;
};

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

Generation::Generation(uint64_t version, std::vector<std::shared_ptr<SegmentReader>> segments,
                       CorpusStats stats)
    : version_(version), segments_(std::move(segments)), stats_(stats) {}

Result<std::pair<const SegmentReader*, DocId>> Generation::Resolve(DocId global) const {
    // Segments are in ascending doc_base order with contiguous ranges, so this is a
    // binary search over the bases rather than a scan. It is on the hot path of every
    // scored hit: ranking produces global ids and needs the segment-local id to read the
    // document's length and metadata.
    size_t lo = 0, hi = segments_.size();
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        const SegmentMeta& meta = segments_[mid]->meta();
        if (global < meta.doc_base) {
            hi = mid;
        } else if (global >= meta.doc_base + meta.document_count) {
            lo = mid + 1;
        } else {
            return std::pair<const SegmentReader*, DocId>(
                segments_[mid].get(), static_cast<DocId>(global - meta.doc_base));
        }
    }
    std::ostringstream os;
    os << "document " << global << " is not in generation " << version_;
    return Status::NotFound(os.str());
}

// ---------------------------------------------------------------------------
// Index::Stats
// ---------------------------------------------------------------------------

double Index::Stats::WriteAmplification() const {
    if (documents_indexed == 0) return 0.0;
    // Bytes written per document indexed.
    //
    // The header asks for bytes written divided by bytes of live data, and Stats has no
    // live-byte counter to divide by. Normalising by documents instead gives the same
    // shape: an index that never merges sits at the bytes one segment write costs per
    // document, and every merge round that rewrites a document pushes the figure up by
    // roughly that amount again. Dividing two of these numbers from two merge policies
    // gives exactly the ratio the measure is for, which is why this is the useful half to
    // keep.
    return static_cast<double>(bytes_written) / static_cast<double>(documents_indexed);
}

// ---------------------------------------------------------------------------
// Index
// ---------------------------------------------------------------------------

Index::Index(Options options)
    : impl_(std::make_unique<Impl>()), options_(std::move(options)) {}

Index::~Index() {
    Index::Impl& state = *impl_;
    {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        current_.reset();
        state.segments.clear();
        // Best effort: a segment whose files are still held by a leaked generation stays
        // on disk, and the next Open will find it unreferenced by the manifest.
        for (PendingRemoval& pending : state.pending_removals) {
            if (pending.reader.use_count() == 1) {
                pending.reader.reset();
                (void)storage::RemoveFile(pending.path);
            }
        }
        state.pending_removals.clear();
    }
}

namespace {

/// Publishes a new generation over `state.segments`.
///
/// The two steps that matter are both here and both are single assignments. doc_base is
/// recomputed first, because a merge changes how many documents precede each segment and
/// a stale base would resolve a hit to the wrong document. A segment whose base moved is
/// *cloned*, never mutated: an older generation may still be handing that base out to a
/// query that is mid-flight.
///
/// Then `current_` is replaced. One pointer assignment under the write lock, so a reader
/// sees either the old set or the new one — never a segment list that has had its merge
/// inputs removed but not its output added. That is the whole reason a generation exists.
Status PublishLocked(const Index::Options& options, Index::Impl& state,
                     std::shared_ptr<Generation>& current, std::atomic<uint64_t>& next_version,
                     Index::Stats& stats);

std::string EncodeManifest(const Index::Impl& state, const Index::Stats& stats) {
    std::ostringstream os;
    os << "meridian-manifest 1\n";
    os << "next_segment " << state.next_segment_id << "\n";
    os << "stats " << stats.generations_published << " " << stats.merges_run << " "
       << stats.segments_merged << " " << stats.bytes_written << " " << stats.documents_indexed
       << " " << stats.documents_deleted << "\n";
    for (const auto& segment : state.segments) {
        const SegmentMeta& meta = segment->meta();
        os << "segment " << meta.id << " " << meta.document_count << " " << meta.deleted_count;
        for (DocId local = 0; local < meta.document_count; ++local) {
            if (segment->IsDeleted(local)) os << " " << local;
        }
        os << "\n";
    }
    return os.str();
}

/// The manifest is the durable form of a generation, and the rename that replaces it is
/// the on-disk equivalent of the pointer swap in PublishLocked: a reader starting up sees
/// one complete segment list or the previous one. Writing it as part of every publication
/// is what makes a crash during a merge recoverable — the old manifest still names the
/// inputs, and the orphaned output is simply never referenced.
Status WriteManifest(const std::string& directory, const Index::Impl& state,
                     const Index::Stats& stats) {
    const std::string text = EncodeManifest(state, stats);
    const std::span<const uint8_t> bytes(reinterpret_cast<const uint8_t*>(text.data()),
                                         text.size());
    return storage::WriteFileAtomic(ManifestPath(directory), bytes);
}

Status PublishLocked(const Index::Options& options, Index::Impl& state,
                     std::shared_ptr<Generation>& current, std::atomic<uint64_t>& next_version,
                     Index::Stats& stats) {
    DocId base = 0;
    std::vector<std::shared_ptr<SegmentReader>> readers;
    readers.reserve(state.segments.size());
    CorpusStats corpus;

    for (auto& segment : state.segments) {
        if (segment->meta().doc_base != base) {
            segment = segment->Clone(segment->deleted_bitmap(), segment->meta().deleted_count,
                                     base);
        }
        corpus.document_count += segment->meta().live_count();
        corpus.total_tokens += segment->live_tokens();
        base += segment->meta().document_count;
        readers.push_back(segment);
    }

    corpus.average_document_length =
        corpus.document_count == 0
            ? 0.0
            : static_cast<double>(corpus.total_tokens) / static_cast<double>(corpus.document_count);

    current = std::make_shared<Generation>(next_version.fetch_add(1), std::move(readers), corpus);
    ++stats.generations_published;
    return WriteManifest(options.directory, state, stats);
}

/// Unlinks the files of merged-away segments whose last reader has gone.
///
/// `use_count() == 1` means the only remaining owner is this list, so no Generation and
/// therefore no query still holds it. This is the load-bearing use of shared_ptr in the
/// whole design: without it a merge would unlink a file that an in-flight query is
/// part-way through reading, and on a platform where that is not merely rude the query
/// gets a short read rather than a result.
void CollectGarbageLocked(Index::Impl& state) {
    std::vector<PendingRemoval> still_held;
    for (PendingRemoval& pending : state.pending_removals) {
        if (pending.reader.use_count() == 1) {
            pending.reader.reset();
            (void)storage::RemoveFile(pending.path);
        } else {
            still_held.push_back(std::move(pending));
        }
    }
    state.pending_removals = std::move(still_held);
}

}  // namespace

Status Index::Open() {
    Index::Impl& state = *impl_;
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (state.opened) return Status::Internal("Index::Open called twice");

    MERIDIAN_TRY(storage::EnsureDirectory(options_.directory));
    state.buffer = std::make_unique<SegmentBuilder>(BuilderOptions(options_));

    const std::string manifest = ManifestPath(options_.directory);
    if (storage::FileExists(manifest)) {
        MERIDIAN_ASSIGN(const std::shared_ptr<storage::ReadableFile> file,
                        storage::ReadableFile::Open(manifest));
        std::string text(static_cast<size_t>(file->size()), '\0');
        MERIDIAN_TRY(file->ReadAt(0, text.size(), reinterpret_cast<uint8_t*>(text.data())));

        std::istringstream in(text);
        std::string line;
        std::vector<std::pair<uint64_t, std::vector<DocId>>> segment_records;
        while (std::getline(in, line)) {
            std::istringstream fields(line);
            std::string keyword;
            fields >> keyword;
            if (keyword == "meridian-manifest") {
                uint32_t version = 0;
                fields >> version;
                if (version != 1) {
                    return Status::Corruption("unsupported manifest version");
                }
            } else if (keyword == "next_segment") {
                fields >> state.next_segment_id;
            } else if (keyword == "stats") {
                fields >> stats_.generations_published >> stats_.merges_run >>
                    stats_.segments_merged >> stats_.bytes_written >> stats_.documents_indexed >>
                    stats_.documents_deleted;
            } else if (keyword == "segment") {
                uint64_t id = 0;
                uint32_t document_count = 0, deleted_count = 0;
                fields >> id >> document_count >> deleted_count;
                std::vector<DocId> deleted;
                DocId local = 0;
                while (fields >> local) deleted.push_back(local);
                segment_records.emplace_back(id, std::move(deleted));
            } else if (!keyword.empty()) {
                return Status::Corruption("unrecognised manifest line: " + keyword);
            }
        }

        for (const auto& [id, deleted] : segment_records) {
            MERIDIAN_ASSIGN(std::shared_ptr<SegmentReaderImpl> reader,
                            SegmentReaderImpl::Open(SegmentPath(options_.directory, id), id, 0));
            if (!deleted.empty()) {
                std::vector<uint8_t> bitmap;
                for (DocId local : deleted) index::SetDeletedBit(bitmap, local);
                reader = reader->Clone(std::move(bitmap), static_cast<uint32_t>(deleted.size()), 0);
            }
            state.segments.push_back(std::move(reader));
        }
        std::sort(state.segments.begin(), state.segments.end(),
                  [](const auto& a, const auto& b) { return a->meta().id < b->meta().id; });

        // The external-id locator is rebuilt by reading every document record once.
        // That is O(documents) positioned reads at open, paid so that Delete is a hash
        // probe instead of a scan. The alternative — a per-segment sorted index over
        // external ids — is a second dictionary to build, verify and merge, for an
        // operation that is rare compared with querying.
        for (const auto& segment : state.segments) {
            const uint64_t id = segment->meta().id;
            for (DocId local = 0; local < segment->meta().document_count; ++local) {
                // Tombstoned documents are deliberately left out: the locator answers
                // "which document does this external id name", and a deleted document
                // does not name one any more. Including them would make a second Delete
                // of the same id succeed silently.
                if (segment->IsDeleted(local)) continue;
                MERIDIAN_ASSIGN(const DocumentMeta meta, segment->Document(local));
                state.locator[meta.external_id] = {id, local};
            }
        }
    }

    state.opened = true;
    return PublishLocked(options_, state, current_, next_version_, stats_);
}

Status Index::Close() {
    Index::Impl& state = *impl_;
    {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        if (!state.opened) return Status::Ok();
    }
    MERIDIAN_TRY(Flush());

    std::unique_lock<std::shared_mutex> lock(mutex_);
    MERIDIAN_TRY(WriteManifest(options_.directory, state, stats_));
    state.opened = false;
    current_.reset();
    state.segments.clear();
    CollectGarbageLocked(state);
    return Status::Ok();
}

Status Index::Add(const DocumentMeta& meta,
                  const std::unordered_map<Field, std::vector<std::string>>& tokens_by_field) {
    bool flush_needed = false;
    {
        Index::Impl& state = *impl_;
        std::unique_lock<std::shared_mutex> lock(mutex_);
        if (!state.opened) return Status::Internal("Index::Add before Open");
        if (meta.external_id.empty()) {
            return Status::InvalidArgument("document has no external id");
        }
        if (state.locator.count(meta.external_id) != 0 ||
            state.buffered_ids.count(meta.external_id) != 0) {
            return Status::InvalidArgument("duplicate external id: " + meta.external_id);
        }

        MERIDIAN_TRY(state.buffer->AddTokenizedDocument(meta, tokens_by_field, nullptr));
        state.buffered_ids.insert(meta.external_id);
        ++stats_.documents_indexed;
        flush_needed = state.buffer->document_count() >= options_.docs_per_segment;
    }
    // Flushed outside the lock's scope rather than inline, because Flush takes the same
    // lock. The buffer threshold is a soft bound anyway: a concurrent Add may push it one
    // document further before this call gets there.
    if (flush_needed) return Flush();
    return Status::Ok();
}

Status Index::Flush() {
    Index::Impl& state = *impl_;
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (!state.opened) return Status::Internal("Index::Flush before Open");
    if (state.buffer == nullptr || state.buffer->empty()) return Status::Ok();

    const uint64_t id = state.next_segment_id++;
    const std::string path = SegmentPath(options_.directory, id);

    SegmentMeta meta;
    MERIDIAN_TRY(state.buffer->Build(path, meta));
    stats_.bytes_written += meta.bytes;

    MERIDIAN_ASSIGN(std::shared_ptr<SegmentReaderImpl> reader,
                    SegmentReaderImpl::Open(path, id, 0));

    const std::vector<std::string>& external_ids = state.buffer->external_ids();

    // Deletes that arrived while these documents were still in the buffer are applied
    // now, as part of the same publication that makes the segment visible. A document
    // must never be searchable in the window between its flush and its tombstone.
    std::vector<uint8_t> bitmap;
    uint32_t deleted_count = 0;
    if (!state.deferred_deletes.empty()) {
        for (DocId local = 0; local < external_ids.size(); ++local) {
            if (state.deferred_deletes.count(external_ids[local]) != 0) {
                index::SetDeletedBit(bitmap, local);
                ++deleted_count;
            }
        }
    }
    if (deleted_count > 0) {
        reader = reader->Clone(std::move(bitmap), deleted_count, 0);
    }

    for (DocId local = 0; local < external_ids.size(); ++local) {
        if (state.deferred_deletes.count(external_ids[local]) != 0) continue;
        state.locator[external_ids[local]] = {id, local};
    }
    state.deferred_deletes.clear();
    state.buffered_ids.clear();
    state.segments.push_back(std::move(reader));
    state.buffer = std::make_unique<SegmentBuilder>(BuilderOptions(options_));

    return PublishLocked(options_, state, current_, next_version_, stats_);
}

Status Index::Delete(std::string_view external_id) {
    Index::Impl& state = *impl_;
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (!state.opened) return Status::Internal("Index::Delete before Open");

    const std::string key(external_id);
    if (state.buffered_ids.count(key) != 0) {
        state.deferred_deletes.insert(key);
        ++stats_.documents_deleted;
        return Status::Ok();
    }

    const auto found = state.locator.find(key);
    if (found == state.locator.end()) {
        return Status::NotFound("no document with external id " + key);
    }
    const uint64_t segment_id = found->second.first;
    const DocId local = found->second.second;

    for (auto& segment : state.segments) {
        if (segment->meta().id != segment_id) continue;
        if (segment->IsDeleted(local)) return Status::Ok();  // idempotent

        std::vector<uint8_t> bitmap = segment->deleted_bitmap();
        index::SetDeletedBit(bitmap, local);
        segment = segment->Clone(std::move(bitmap), segment->meta().deleted_count + 1,
                                 segment->meta().doc_base);
        ++stats_.documents_deleted;
        state.locator.erase(found);
        return PublishLocked(options_, state, current_, next_version_, stats_);
    }
    return Status::Internal("locator names a segment the index does not hold");
}

std::shared_ptr<Generation> Index::Acquire() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    // A copy of the pointer, not a reference to it. The caller's generation stays alive
    // for as long as it holds this, which is what lets a query run to completion across
    // any number of merges.
    return current_;
}

namespace {

/// Builds one segment out of several, dropping deleted documents.
///
/// Runs without the index lock held: every input is immutable, so there is nothing to
/// race with. What the caller must check afterwards is whether any input gained a
/// deletion in the meantime — see MergeLocked's second phase.
Status MergeInto(const std::vector<std::shared_ptr<SegmentReaderImpl>>& inputs,
                 const SegmentBuilder::Options& options, const std::string& path,
                 SegmentMeta& meta_out, std::vector<std::string>& external_ids_out) {
    SegmentBuilder builder(options);

    // Documents first, in input order, so new local ids ascend with (input, old local).
    // That is what keeps each term's postings sorted when they are added input by input
    // below: every posting from input k precedes every posting from input k+1.
    std::vector<std::vector<DocId>> remap(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        const SegmentReaderImpl& input = *inputs[i];
        remap[i].assign(input.meta().document_count, kNoMoreDocs);
        for (DocId local = 0; local < input.meta().document_count; ++local) {
            if (input.IsDeleted(local)) continue;
            MERIDIAN_ASSIGN(const DocumentMeta meta, input.Document(local));
            remap[i][local] = builder.AddDocument(meta);
        }
    }

    for (size_t i = 0; i < inputs.size(); ++i) {
        const SegmentReaderImpl& input = *inputs[i];
        const std::vector<DocId>& mapping = remap[i];
        MERIDIAN_TRY(input.ForEachTerm([&](Field field, std::string_view term,
                                           const index::TermLocation& location) -> Status {
            MERIDIAN_ASSIGN(const std::unique_ptr<postings::ListIterator> it,
                            input.PostingsAt(location, postings::IteratorMode::kSequential));
            for (DocId doc = it->doc(); doc != kNoMoreDocs; doc = it->next()) {
                if (doc >= mapping.size()) {
                    return Status::Corruption("posting list names a document outside the segment");
                }
                const DocId mapped = mapping[doc];
                if (mapped == kNoMoreDocs) continue;  // the document was deleted
                MERIDIAN_TRY(builder.AddTerm(field, term, mapped, it->term_frequency(),
                                             it->positions()));
            }
            // A decode failure mid-walk leaves the iterator exhausted, which would
            // otherwise look exactly like the end of a short list and silently drop
            // postings from the merged segment.
            return it->error();
        }));
    }

    MERIDIAN_TRY(builder.Build(path, meta_out));
    external_ids_out = builder.external_ids();
    return Status::Ok();
}

}  // namespace

namespace {

/// Shared body of MaybeMerge and ForceMerge. `select` picks the inputs under the lock.
Status RunMerge(const Index::Options& options, std::shared_mutex& mutex, Index::Impl& state,
                std::shared_ptr<Generation>& current, std::atomic<uint64_t>& next_version,
                Index::Stats& stats,
                const std::function<std::vector<std::shared_ptr<SegmentReaderImpl>>(
                    const Index::Impl&)>& select) {
    std::vector<std::shared_ptr<SegmentReaderImpl>> inputs;
    uint64_t output_id = 0;

    {
        std::unique_lock<std::shared_mutex> lock(mutex);
        if (!state.opened) return Status::Internal("merge before Open");
        CollectGarbageLocked(state);
        if (state.merging) return Status::Ok();  // another merge owns the candidates
        inputs = select(state);
        if (inputs.empty()) return Status::Ok();
        output_id = state.next_segment_id++;
        state.merging = true;
    }

    const std::string path = SegmentPath(options.directory, output_id);
    SegmentMeta output_meta;
    std::vector<std::string> external_ids;
    Status build = MergeInto(inputs, BuilderOptions(options), path, output_meta, external_ids);

    std::unique_lock<std::shared_mutex> lock(mutex);
    state.merging = false;
    if (!build.ok()) {
        (void)storage::RemoveFile(path);
        return build;
    }

    // Optimistic concurrency, resolved by checking rather than by locking for the whole
    // build. A Delete that landed on an input while the merge was running produced a new
    // reader for that segment, and the merged output does not have the tombstone. Rather
    // than publish a segment that resurrects a deleted document, the merge is abandoned;
    // the next MaybeMerge will pick the same tier up again with the deletion included.
    for (const auto& input : inputs) {
        const auto current_reader =
            std::find_if(state.segments.begin(), state.segments.end(),
                         [&input](const std::shared_ptr<SegmentReaderImpl>& candidate) {
                             return candidate->meta().id == input->meta().id;
                         });
        if (current_reader == state.segments.end() ||
            (*current_reader)->meta().deleted_count != input->meta().deleted_count) {
            (void)storage::RemoveFile(path);
            return Status::Ok();
        }
    }

    MERIDIAN_ASSIGN(std::shared_ptr<SegmentReaderImpl> output,
                    SegmentReaderImpl::Open(path, output_id, 0));
    stats.bytes_written += output_meta.bytes;

    std::set<uint64_t> consumed;
    for (const auto& input : inputs) consumed.insert(input->meta().id);

    std::vector<std::shared_ptr<SegmentReaderImpl>> retained;
    for (auto& segment : state.segments) {
        if (consumed.count(segment->meta().id) != 0) {
            // The reader goes on the pending list, not out of scope: an in-flight query
            // may still be reading it, and the file must outlive that query.
            state.pending_removals.push_back(PendingRemoval{segment, segment->meta().path});
        } else {
            retained.push_back(segment);
        }
    }
    retained.push_back(output);
    state.segments = std::move(retained);

    for (const auto& input : inputs) {
        for (DocId local = 0; local < input->meta().document_count; ++local) {
            if (input->IsDeleted(local)) continue;
            MERIDIAN_ASSIGN(const DocumentMeta meta, input->Document(local));
            state.locator.erase(meta.external_id);
        }
    }
    for (DocId local = 0; local < external_ids.size(); ++local) {
        state.locator[external_ids[local]] = {output_id, local};
    }

    ++stats.merges_run;
    stats.segments_merged += inputs.size();
    MERIDIAN_TRY(PublishLocked(options, state, current, next_version, stats));

    // The inputs are dropped from state.segments above, so their only owners now are the
    // pending list and whatever generations are still alive. If no query holds them this
    // unlinks immediately; if one does, the files survive until it releases.
    CollectGarbageLocked(state);
    return Status::Ok();
}

}  // namespace

Status Index::MaybeMerge() {
    Index::Impl& state = *impl_;
    const uint32_t merge_factor = std::max<uint32_t>(2, options_.merge_factor);
    return RunMerge(
        options_, mutex_, state, current_, next_version_, stats_,
        [merge_factor](const Index::Impl& s) {
            std::map<size_t, std::vector<std::shared_ptr<SegmentReaderImpl>>> tiers;
            for (const auto& segment : s.segments) {
                tiers[TierOf(segment->meta().bytes, merge_factor)].push_back(segment);
            }
            // Smallest qualifying tier first. Small segments are the ones a query pays
            // for most — every one of them is another dictionary probe and another
            // iterator — and merging them is also the cheapest work available.
            for (auto& [tier, segments] : tiers) {
                (void)tier;
                if (segments.size() >= merge_factor) return segments;
            }
            return std::vector<std::shared_ptr<SegmentReaderImpl>>();
        });
}

Status Index::ForceMerge() {
    MERIDIAN_TRY(Flush());
    Index::Impl& state = *impl_;
    return RunMerge(options_, mutex_, state, current_, next_version_, stats_,
                    [](const Index::Impl& s) {
                        // One segment with tombstones in it is still worth rewriting:
                        // that is how the space comes back and how the deleted postings
                        // stop being decoded on every query.
                        if (s.segments.size() >= 2) return s.segments;
                        if (s.segments.size() == 1 && s.segments[0]->meta().deleted_count > 0) {
                            return s.segments;
                        }
                        return std::vector<std::shared_ptr<SegmentReaderImpl>>();
                    });
}

Index::Stats Index::stats() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return stats_;
}

}  // namespace meridian
