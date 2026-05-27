// Stand-in corpus and stages. See reference_index.hpp for why this file exists.
//
// This translation unit touches no engine symbol that is only declared: everything it
// needs (BM25, filter evaluation, fusion) it computes itself. That is what keeps the
// tests and the CLIs linkable while src/index/, src/query/ and src/rank/ are still
// being written. The adapters that *do* call into those owners live in stages_engine.cpp
// for exactly that reason.

#include "reference_index.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace meridian::serve {
namespace {

/// Positions are indexed for title and body but not tags, mirroring the defaults in
/// Index::Options: positions roughly triple a field's size and only phrase scoring reads
/// them, and nobody phrase-searches a tag list.
bool IndexesPositions(Field field) { return field != Field::kTags; }

double FieldWeightFor(const FieldWeights& weights, Field field) {
    switch (field) {
        case Field::kTitle: return weights.title;
        case Field::kBody: return weights.body;
        case Field::kTags: return weights.tags;
        case Field::kFieldCount: break;
    }
    return 0.0;
}

/// Filters are evaluated through the engine's own `Filters::Matches` rather than
/// reimplemented here. A second copy of the predicate would be a place for the two to
/// disagree about an edge case — whether the date bounds are closed, for one — and a
/// filter that behaves differently in the serving path than in the retrieval path is a
/// wrong-answer bug that no test of either component alone would catch.
bool FilterMatches(const Filters& filters, const DocumentMeta& doc) {
    return filters.Matches(doc);
}

double Bm25Score(const Bm25Params& params, uint32_t document_count, uint32_t document_frequency,
                 double average_length, uint32_t tf, uint32_t doc_length) {
    if (tf == 0 || document_count == 0) return 0.0;
    const double n = static_cast<double>(document_count);
    const double df = static_cast<double>(document_frequency);
    const double idf = std::log(1.0 + (n - df + 0.5) / (df + 0.5));
    const double length_ratio =
        average_length > 0.0 ? static_cast<double>(doc_length) / average_length : 1.0;
    const double norm = params.k1 * (1.0 - params.b + params.b * length_ratio);
    return idf * (static_cast<double>(tf) * (params.k1 + 1.0)) / (static_cast<double>(tf) + norm);
}

void TruncateByScore(std::vector<Candidate>& candidates, uint32_t k) {
    const size_t limit = std::min<size_t>(k, candidates.size());
    // Ties broken by document id so a result list is reproducible run to run. Without
    // it, two documents with identical scores swap places depending on hash iteration
    // order, and every evaluation diff is full of noise.
    auto better = [](const Candidate& a, const Candidate& b) {
        if (a.score != b.score) return a.score > b.score;
        return a.doc < b.doc;
    };
    std::partial_sort(candidates.begin(), candidates.begin() + static_cast<ptrdiff_t>(limit),
                      candidates.end(), better);
    candidates.resize(limit);
}

}  // namespace

std::vector<std::string> Tokenize(std::string_view text) {
    std::vector<std::string> tokens;
    std::string current;
    for (char c : text) {
        const unsigned char uc = static_cast<unsigned char>(c);
        if ((uc >= 'a' && uc <= 'z') || (uc >= '0' && uc <= '9')) {
            current.push_back(static_cast<char>(uc));
        } else if (uc >= 'A' && uc <= 'Z') {
            current.push_back(static_cast<char>(uc - 'A' + 'a'));
        } else if (!current.empty()) {
            tokens.push_back(std::move(current));
            current.clear();
        }
    }
    if (!current.empty()) tokens.push_back(std::move(current));
    return tokens;
}

double CosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    const size_t n = std::min(a.size(), b.size());
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (size_t i = 0; i < n; ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        norm_a += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        norm_b += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    if (norm_a == 0.0 || norm_b == 0.0) return 0.0;
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

// ---------------------------------------------------------------------------
// ReferenceIndex
// ---------------------------------------------------------------------------

uint64_t ReferenceIndex::Segment::term_count() const {
    uint64_t total = 0;
    for (const auto& field : terms) total += field.size();
    return total;
}

uint64_t ReferenceIndex::Segment::EstimateBytes() const {
    uint64_t bytes = docs.size() * sizeof(Entry);
    for (const auto& doc : docs) {
        bytes += doc.meta.external_id.capacity() + doc.meta.language.capacity() +
                 doc.meta.category.capacity() + doc.embedding.capacity() * sizeof(float);
    }
    for (const auto& field : terms) {
        for (const auto& [term, list] : field) {
            bytes += term.capacity() + sizeof(PostingList) + list.postings.size() * sizeof(Posting);
            for (const auto& posting : list.postings) {
                bytes += posting.positions.capacity() * sizeof(Position);
            }
        }
    }
    return bytes;
}

const ReferenceIndex::Entry* ReferenceIndex::Snapshot::Resolve(DocId global) const {
    for (const auto& segment : segments) {
        if (global >= segment->doc_base &&
            global < segment->doc_base + static_cast<DocId>(segment->docs.size())) {
            return &segment->docs[global - segment->doc_base];
        }
    }
    return nullptr;
}

uint64_t ReferenceIndex::Snapshot::term_count() const {
    uint64_t total = 0;
    for (const auto& segment : segments) total += segment->term_count();
    return total;
}

uint64_t ReferenceIndex::Snapshot::EstimateBytes() const {
    uint64_t total = 0;
    for (const auto& segment : segments) total += segment->EstimateBytes();
    return total;
}

ReferenceIndex::ReferenceIndex() { current_ = std::make_shared<Snapshot>(); }

void ReferenceIndex::Add(Document doc) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    buffer_.push_back(std::move(doc));
}

uint64_t ReferenceIndex::Flush() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (buffer_.empty()) return current_->version;

    auto segment = std::make_shared<Segment>();
    segment->id = next_segment_++;
    segment->doc_base = next_doc_base_;
    segment->docs.reserve(buffer_.size());

    for (size_t local = 0; local < buffer_.size(); ++local) {
        Document& doc = buffer_[local];
        uint32_t total_tokens = 0;
        for (auto& [field, tokens] : doc.tokens) {
            if (field == Field::kFieldCount) continue;
            total_tokens += static_cast<uint32_t>(tokens.size());
            auto& dictionary = segment->terms[static_cast<size_t>(field)];
            for (size_t position = 0; position < tokens.size(); ++position) {
                PostingList& list = dictionary[tokens[position]];
                if (list.postings.empty() || list.postings.back().doc != static_cast<DocId>(local)) {
                    list.postings.push_back(Posting{static_cast<DocId>(local), 0, {}});
                    ++list.stats.document_frequency;
                }
                Posting& posting = list.postings.back();
                ++posting.term_frequency;
                ++list.stats.collection_frequency;
                if (IndexesPositions(field)) {
                    posting.positions.push_back(static_cast<Position>(position));
                }
            }
        }
        // The caller may supply a length; when it does not, the indexed token count is
        // the authority, because BM25's length normalisation has to agree with what was
        // actually indexed.
        if (doc.meta.length == 0) doc.meta.length = total_tokens;
        segment->docs.push_back(Entry{doc.meta, std::move(doc.embedding)});
    }

    next_doc_base_ += static_cast<DocId>(segment->docs.size());
    buffer_.clear();

    auto segments = current_->segments;
    segments.push_back(std::move(segment));
    Publish(std::move(segments), current_->deleted);
    return current_->version;
}

bool ReferenceIndex::Delete(const std::string& external_id) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    for (const auto& segment : current_->segments) {
        for (size_t local = 0; local < segment->docs.size(); ++local) {
            if (segment->docs[local].meta.external_id != external_id) continue;
            auto deleted = current_->deleted;
            deleted.insert(segment->doc_base + static_cast<DocId>(local));
            auto segments = current_->segments;
            Publish(std::move(segments), std::move(deleted));
            return true;
        }
    }
    return false;
}

uint64_t ReferenceIndex::Merge() {
    for (;;) {
        std::shared_ptr<const Snapshot> base = Current();
        if (base->segments.size() < 2 && base->deleted.empty()) return base->version;

        // Built outside the lock. Queries keep running against `base` the whole time —
        // they hold it by reference, so nothing it points at can go away — and only the
        // pointer swap below needs exclusive access. A merge that held the write lock
        // for its whole duration would block every Acquire and make the design's central
        // claim false.
        auto merged = std::make_shared<Segment>();

        // Renumbering map, built first. Segments are visited in order and documents
        // within a segment in ascending local id, so the target ids come out ascending —
        // which is what keeps the merged posting lists sorted without a sort, and sorted
        // order is the precondition every downstream merge loop relies on.
        std::unordered_map<const Segment*, std::unordered_map<DocId, DocId>> renumber;
        for (const auto& segment : base->segments) {
            auto& table = renumber[segment.get()];
            for (size_t local = 0; local < segment->docs.size(); ++local) {
                const DocId global = segment->doc_base + static_cast<DocId>(local);
                if (base->IsDeleted(global)) continue;
                table[static_cast<DocId>(local)] = static_cast<DocId>(merged->docs.size());
                merged->docs.push_back(segment->docs[local]);
            }
        }

        for (const auto& segment : base->segments) {
            const auto& table = renumber[segment.get()];
            for (size_t field = 0; field < kFieldCount; ++field) {
                for (const auto& [term, list] : segment->terms[field]) {
                    // The output entry is created lazily: a term whose every posting was
                    // tombstoned must disappear from the merged dictionary, not survive
                    // as an empty list that still costs a lookup on every query.
                    PostingList* out = nullptr;
                    for (const auto& posting : list.postings) {
                        const auto it = table.find(posting.doc);
                        if (it == table.end()) continue;  // tombstoned, so dropped here
                        if (out == nullptr) out = &merged->terms[field][term];
                        Posting copy = posting;
                        copy.doc = it->second;
                        out->stats.collection_frequency += copy.term_frequency;
                        ++out->stats.document_frequency;
                        out->postings.push_back(std::move(copy));
                    }
                }
            }
        }

        {
            std::unique_lock<std::shared_mutex> lock(mutex_);
            // Optimistic: if another thread published while this merge was running, the
            // merged segment describes a stale segment set and must be rebuilt. Retrying
            // is correct and cheap compared to serialising merges against writers.
            if (current_ != base) continue;
            merged->id = next_segment_++;
            // A fresh doc_base rather than zero: ids from the retired segments must never
            // be reused, or a cached candidate list from the old generation would resolve
            // to the wrong document in the new one.
            merged->doc_base = next_doc_base_;
            next_doc_base_ += static_cast<DocId>(merged->docs.size());
            ++merges_;
            std::vector<std::shared_ptr<const Segment>> segments{std::move(merged)};
            Publish(std::move(segments), {});
            return current_->version;
        }
    }
}

void ReferenceIndex::Publish(std::vector<std::shared_ptr<const Segment>> segments,
                             std::unordered_set<DocId> deleted) {
    auto snapshot = std::make_shared<Snapshot>();
    snapshot->version = next_version_++;
    snapshot->segments = std::move(segments);
    snapshot->deleted = std::move(deleted);

    uint64_t total_tokens = 0;
    uint32_t live = 0;
    for (const auto& segment : snapshot->segments) {
        for (size_t local = 0; local < segment->docs.size(); ++local) {
            const DocId global = segment->doc_base + static_cast<DocId>(local);
            if (snapshot->IsDeleted(global)) continue;
            ++live;
            total_tokens += segment->docs[local].meta.length;
        }
        for (size_t field = 0; field < kFieldCount; ++field) {
            for (const auto& [term, list] : segment->terms[field]) {
                snapshot->document_frequency[field][term] += list.stats.document_frequency;
            }
        }
    }
    snapshot->stats.document_count = live;
    snapshot->stats.total_tokens = total_tokens;
    snapshot->stats.average_document_length =
        live == 0 ? 0.0 : static_cast<double>(total_tokens) / static_cast<double>(live);

    ++generations_;
    // The single store that makes everything above visible. Readers either see the whole
    // new generation or the whole old one.
    current_ = std::move(snapshot);
}

std::shared_ptr<const ReferenceIndex::Snapshot> ReferenceIndex::Current() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return current_;
}

GenerationPin ReferenceIndex::Acquire() const {
    GenerationPin pin;
    pin.attachment = Current();
    pin.version = static_cast<const Snapshot*>(pin.attachment.get())->version;
    return pin;
}

ReferenceIndex::Stats ReferenceIndex::stats() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    Stats stats;
    stats.documents = current_->stats.document_count;
    stats.segments = current_->segments.size();
    stats.terms = current_->term_count();
    stats.bytes = current_->EstimateBytes();
    stats.generations = generations_;
    stats.merges = merges_;
    return stats;
}

std::shared_ptr<const ReferenceIndex::Snapshot> SnapshotFrom(const GenerationPin& pin) {
    return std::static_pointer_cast<const ReferenceIndex::Snapshot>(pin.attachment);
}

// ---------------------------------------------------------------------------
// Stages
// ---------------------------------------------------------------------------

Result<Query> ReferenceParseStage::Parse(const Query& request) const {
    Query query = request;
    if (!query.raw.empty()) {
        query.terms = Tokenize(query.raw);
    }
    if (query.terms.empty() && !query.embedding) {
        return Status::QueryError("query has no terms and no embedding");
    }
    if (query.top_k == 0) return Status::InvalidArgument("top_k must be positive");
    if (query.candidate_k < query.top_k) {
        // Not an error: a caller asking for more results than candidates is asking for
        // the retriever's whole output, and silently returning fewer than top_k would be
        // the short-result-list failure this project refuses to have.
        query.candidate_k = query.top_k;
    }
    return query;
}

Result<std::vector<Candidate>> ReferenceLexicalStage::Retrieve(const StageContext& ctx,
                                                              const Query& query) const {
    auto snapshot = SnapshotFrom(*ctx.pin);
    if (!snapshot) return Status::Internal("lexical stage received a pin it cannot read");
    if (query.terms.empty()) return std::vector<Candidate>{};

    std::unordered_map<DocId, double> scores;
    std::unordered_map<DocId, uint32_t> terms_matched;

    for (const auto& term : query.terms) {
        std::unordered_map<DocId, double> term_scores;

        for (const auto& segment : snapshot->segments) {
            for (size_t field_index = 0; field_index < kFieldCount; ++field_index) {
                const Field field = static_cast<Field>(field_index);
                const double weight = FieldWeightFor(weights_, field);
                if (weight == 0.0) continue;

                CachedPostingList postings;
                if (ctx.posting_cache) {
                    PostingCacheKey key{term, field, segment->id};
                    if (auto cached = ctx.posting_cache->Lookup(key); cached.has_value()) {
                        postings = *cached;
                        if (ctx.counters) ctx.counters->posting_cache_hit = true;
                    } else {
                        auto it = segment->terms[field_index].find(term);
                        // A term absent from this segment is not cached. Most terms are
                        // absent from most segments, so caching absence would fill the
                        // budget with entries that hold nothing and evict the hot lists
                        // that do.
                        if (it == segment->terms[field_index].end()) continue;
                        auto entry = std::make_shared<CachedPostings>();
                        entry->postings = it->second.postings;
                        entry->stats = it->second.stats;
                        postings = entry;
                        ctx.posting_cache->Insert(key, postings);
                    }
                } else {
                    auto it = segment->terms[field_index].find(term);
                    if (it == segment->terms[field_index].end()) continue;
                    auto entry = std::make_shared<CachedPostings>();
                    entry->postings = it->second.postings;
                    entry->stats = it->second.stats;
                    postings = entry;
                }

                // Document frequency is taken per field rather than pooled across
                // fields. A term that is common in bodies and rare in titles genuinely
                // carries different information in each, and the per-field dictionary
                // already has the number.
                const auto& df_map = snapshot->document_frequency[field_index];
                const auto df_it = df_map.find(term);
                const uint32_t df = df_it == df_map.end() ? 0 : df_it->second;

                for (const auto& posting : postings->postings) {
                    const DocId global = segment->doc_base + posting.doc;
                    if (snapshot->IsDeleted(global)) continue;
                    const ReferenceIndex::Entry& doc = segment->docs[posting.doc];
                    if (!query.filters.empty() && !FilterMatches(query.filters, doc.meta)) continue;

                    const double score =
                        weight * Bm25Score(params_, snapshot->stats.document_count, df,
                                           snapshot->stats.average_document_length,
                                           posting.term_frequency, doc.meta.length);
                    term_scores[global] += score;
                    if (ctx.counters) ++ctx.counters->postings_decoded;
                }
            }
        }

        for (const auto& [doc, score] : term_scores) {
            scores[doc] += score;
            ++terms_matched[doc];
        }
    }

    // Conjunctive and phrase queries both require every term. Phrase additionally
    // requires adjacency, which this stand-in does not check; the engine's
    // LexicalRetriever is where positional verification belongs, and approximating it
    // here would hide its absence.
    const bool require_all =
        query.mode == QueryMode::kConjunctive || query.mode == QueryMode::kPhrase;

    std::vector<Candidate> candidates;
    candidates.reserve(scores.size());
    for (const auto& [doc, score] : scores) {
        if (require_all && terms_matched[doc] != query.terms.size()) continue;
        Candidate candidate;
        candidate.doc = doc;
        candidate.score = score;
        candidate.bm25 = score;
        candidate.source = Candidate::Source::kLexical;
        candidates.push_back(candidate);
    }
    TruncateByScore(candidates, query.candidate_k);
    return candidates;
}

Result<std::vector<Candidate>> ReferenceSemanticStage::Search(const StageContext& ctx,
                                                             const Query& query) const {
    auto snapshot = SnapshotFrom(*ctx.pin);
    if (!snapshot) return Status::Internal("semantic stage received a pin it cannot read");
    if (!query.embedding) return std::vector<Candidate>{};

    std::vector<Candidate> candidates;
    for (const auto& segment : snapshot->segments) {
        for (size_t local = 0; local < segment->docs.size(); ++local) {
            const DocId global = segment->doc_base + static_cast<DocId>(local);
            if (snapshot->IsDeleted(global)) continue;
            const ReferenceIndex::Entry& doc = segment->docs[local];
            if (doc.embedding.empty()) continue;
            if (!query.filters.empty() && !FilterMatches(query.filters, doc.meta)) continue;

            Candidate candidate;
            candidate.doc = global;
            candidate.vector_similarity = CosineSimilarity(*query.embedding, doc.embedding);
            candidate.score = candidate.vector_similarity;
            candidate.source = Candidate::Source::kSemantic;
            candidates.push_back(candidate);
        }
    }
    TruncateByScore(candidates, query.candidate_k);
    return candidates;
}

std::vector<Candidate> ReferenceMergeStage::Merge(const std::vector<Candidate>& lexical,
                                                 const std::vector<Candidate>& semantic,
                                                 uint32_t candidate_k) const {
    struct Accumulator {
        double score = 0.0;
        double bm25 = 0.0;
        double similarity = 0.0;
        bool from_lexical = false;
        bool from_semantic = false;
    };
    std::unordered_map<DocId, Accumulator> merged;

    auto fuse = [&](const std::vector<Candidate>& list, double weight, bool is_lexical) {
        for (size_t rank = 0; rank < list.size(); ++rank) {
            Accumulator& acc = merged[list[rank].doc];
            acc.score += weight / (k_ + static_cast<double>(rank) + 1.0);
            if (is_lexical) {
                acc.bm25 = list[rank].bm25;
                acc.from_lexical = true;
            } else {
                acc.similarity = list[rank].vector_similarity;
                acc.from_semantic = true;
            }
        }
    };
    fuse(lexical, lexical_weight_, true);
    fuse(semantic, semantic_weight_, false);

    std::vector<Candidate> candidates;
    candidates.reserve(merged.size());
    for (const auto& [doc, acc] : merged) {
        Candidate candidate;
        candidate.doc = doc;
        candidate.score = acc.score;
        candidate.bm25 = acc.bm25;
        candidate.vector_similarity = acc.similarity;
        candidate.source = acc.from_lexical && acc.from_semantic ? Candidate::Source::kBoth
                           : acc.from_lexical                   ? Candidate::Source::kLexical
                                                                : Candidate::Source::kSemantic;
        candidates.push_back(candidate);
    }
    TruncateByScore(candidates, candidate_k);
    return candidates;
}

std::vector<std::string> ReferenceRerankStage::FeatureNames() const {
    return {"fusion", "bm25", "vector_similarity", "popularity", "freshness"};
}

Result<std::vector<Hit>> ReferenceRerankStage::Rank(const StageContext& ctx, const Query& query,
                                                   const std::vector<Candidate>& candidates) const {
    auto snapshot = SnapshotFrom(*ctx.pin);
    if (!snapshot) return Status::Internal("rerank stage received a pin it cannot read");

    std::vector<Hit> hits;
    hits.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        const ReferenceIndex::Entry* doc = snapshot->Resolve(candidate.doc);
        if (doc == nullptr) {
            // A candidate the pinned generation cannot resolve means the pin and the
            // candidate came from different index states, which is the bug this whole
            // pinning scheme exists to prevent. Reporting it beats dropping the document
            // and returning a list one short.
            return Status::Internal("candidate " + std::to_string(candidate.doc) +
                                   " does not resolve in generation " +
                                   std::to_string(snapshot->version));
        }

        double freshness = 0.0;
        if (now_seconds_ > 0 && doc->meta.published_at > 0) {
            const double age_days =
                static_cast<double>(now_seconds_ - doc->meta.published_at) / 86400.0;
            freshness = std::exp(-std::max(0.0, age_days) / 30.0);
        }

        Hit hit;
        hit.doc = candidate.doc;
        hit.features["fusion"] = candidate.score;
        hit.features["bm25"] = candidate.bm25;
        hit.features["vector_similarity"] = candidate.vector_similarity;
        hit.features["popularity"] = doc->meta.popularity;
        hit.features["freshness"] = freshness;
        hit.score = candidate.score + 0.3 * doc->meta.popularity + 0.2 * freshness;
        hits.push_back(std::move(hit));
    }

    std::stable_sort(hits.begin(), hits.end(), [](const Hit& a, const Hit& b) {
        if (a.score != b.score) return a.score > b.score;
        return a.doc < b.doc;
    });
    (void)query;
    return hits;
}

Status ReferenceMaterializeStage::Materialize(const StageContext& ctx, const Query& query,
                                              std::vector<Hit>& hits) const {
    auto snapshot = SnapshotFrom(*ctx.pin);
    if (!snapshot) return Status::Internal("materialize stage received a pin it cannot read");
    for (Hit& hit : hits) {
        const ReferenceIndex::Entry* doc = snapshot->Resolve(hit.doc);
        if (doc == nullptr) {
            return Status::Internal("hit " + std::to_string(hit.doc) +
                                   " does not resolve in generation " +
                                   std::to_string(snapshot->version));
        }
        hit.external_id = doc->meta.external_id;
    }
    (void)query;
    return Status::Ok();
}

ReferenceEngine::ReferenceEngine(PipelineOptions options, size_t result_cache_bytes,
                                 size_t posting_cache_bytes)
    : result_cache(result_cache_bytes),
      posting_cache(posting_cache_bytes),
      pipeline(
          PipelineStages{&parse, &lexical, &semantic, &merge, &rerank, &materialize},
          options, &result_cache, &posting_cache) {}

}  // namespace meridian::serve
