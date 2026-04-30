#pragma once

// Index organisation: immutable segments, and the generation that publishes them.
//
// The shape of this file is driven by one requirement: **new documents must become
// searchable without the query path stopping.** That rules out a single mutable
// index, and it is what forces everything below.
//
// Segments are immutable once written. A write goes into a small, recent segment;
// large historical segments are never touched. Background merging compacts small
// segments into larger ones, and because the inputs are immutable, a merge can run
// while queries are reading them.
//
// The subtle part is publication. A query must see a *consistent* set of segments —
// never half of a merge's inputs plus half of its output, which would return
// duplicate documents. So the segment set is versioned as a `Generation`, published
// atomically, and a query pins one generation for its whole execution. That is also
// why a generation is reference-counted: a merge cannot delete its input files while
// an in-flight query is still reading them.

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "meridian/postings.hpp"
#include "meridian/status.hpp"

namespace meridian {

/// A document's external identity and the structured fields a filter can test.
///
/// Filters are evaluated against these rather than against the inverted index,
/// because a filter on `language` has a selectivity measured in whole-corpus
/// fractions and putting it in the term dictionary would make it a posting list with
/// a third of the corpus in it.
struct DocumentMeta {
    std::string external_id;
    std::string language;
    std::string category;
    int64_t published_at = 0;
    uint32_t length = 0;  // tokens, for BM25 length normalisation
    double popularity = 0.0;
};

/// Which fields a term matched in. Field-weighted scoring and title-match features
/// both need this, and a single untyped text blob cannot express it.
enum class Field : uint8_t {
    kTitle = 0,
    kBody = 1,
    kTags = 2,
    kFieldCount = 3,
};

const char* FieldName(Field f);

/// One immutable index segment.
///
/// `doc_base` exists because each segment numbers its documents from zero. A
/// segment-local DocId plus the base gives a generation-global one, which is what
/// lets segments be built independently and merged without renumbering their posting
/// lists.
struct SegmentMeta {
    uint64_t id = 0;
    std::string path;
    DocId doc_base = 0;
    uint32_t document_count = 0;
    uint32_t deleted_count = 0;
    uint64_t term_count = 0;
    uint64_t bytes = 0;
    int64_t created_at = 0;

    /// Documents still live. A merge is worth running when this falls well below
    /// document_count, because deleted postings are still being decoded on every
    /// query that touches them.
    uint32_t live_count() const { return document_count - deleted_count; }
};

/// A segment open for reading.
class SegmentReader {
public:
    virtual ~SegmentReader() = default;

    /// Posting iterator for a term in a field, or nullptr when the term is absent.
    /// Absence is not an error: most terms are absent from most segments, and
    /// returning a status for the common case would make the merge loops unreadable.
    virtual std::unique_ptr<PostingIterator> Postings(std::string_view term, Field field) const = 0;

    virtual std::optional<TermStats> Term(std::string_view term, Field field) const = 0;
    virtual Result<DocumentMeta> Document(DocId local) const = 0;
    virtual bool IsDeleted(DocId local) const = 0;
    virtual const SegmentMeta& meta() const = 0;

    /// Bytes this reader has actually read, so projection and skipping claims are
    /// measurable rather than asserted.
    virtual uint64_t bytes_read() const = 0;
};

/// A consistent, immutable view of the whole index at one point in time.
///
/// Held by reference for the duration of a query. The counter is what makes a
/// concurrent merge safe: the merge publishes a new generation and then waits for the
/// old one's readers to drain before unlinking files.
class Generation {
public:
    Generation(uint64_t version, std::vector<std::shared_ptr<SegmentReader>> segments,
               CorpusStats stats);

    uint64_t version() const { return version_; }
    const std::vector<std::shared_ptr<SegmentReader>>& segments() const { return segments_; }

    /// Corpus statistics across all segments. Computed once at publication rather
    /// than per query, and worth noting: these are *generation-wide*, so two queries
    /// against different generations can legitimately produce slightly different
    /// scores for the same document. That is correct, and surprising enough to say
    /// out loud.
    const CorpusStats& stats() const { return stats_; }

    /// Resolves a global DocId to its segment and the segment-local id.
    Result<std::pair<const SegmentReader*, DocId>> Resolve(DocId global) const;

private:
    uint64_t version_;
    std::vector<std::shared_ptr<SegmentReader>> segments_;
    CorpusStats stats_;
};

/// The index: owns segments, publishes generations, runs merges.
class Index {
public:
    struct Options {
        std::string directory;

        /// Documents buffered before a new segment is flushed. Small enough that
        /// recent documents become searchable quickly, large enough that the segment
        /// count does not explode.
        uint32_t docs_per_segment = 10'000;

        /// Merge when this many segments share a size tier. Size-tiered rather than
        /// leveled, because an append-mostly corpus does not need leveled's read
        /// amplification guarantees and pays several times the write cost for them.
        uint32_t merge_factor = 4;

        /// Index positions for these fields only. Positions roughly triple size and
        /// only phrase scoring reads them, so this is a per-field decision.
        bool positions_for_title = true;
        bool positions_for_body = true;
        bool positions_for_tags = false;
    };

    explicit Index(Options options);
    ~Index();

    Index(const Index&) = delete;
    Index& operator=(const Index&) = delete;

    Status Open();
    Status Close();

    /// Adds a document to the in-memory buffer. Not searchable until Flush.
    Status Add(const DocumentMeta& meta,
               const std::unordered_map<Field, std::vector<std::string>>& tokens_by_field);

    /// Writes the buffer as a new segment and publishes a new generation.
    Status Flush();

    /// Marks a document deleted. Deletion is a tombstone rather than a rewrite,
    /// because the segment containing it is immutable; the space returns at merge.
    Status Delete(std::string_view external_id);

    /// Acquires the current generation. Hold it for the whole query.
    std::shared_ptr<Generation> Acquire() const;

    /// Runs one merge round if any tier qualifies. Safe to call concurrently with
    /// queries; that is the property the whole design exists for.
    Status MaybeMerge();

    /// Forces a merge of everything into one segment. For benchmarks and for the
    /// "what does a fully compacted index cost" question.
    Status ForceMerge();

    struct Stats {
        uint64_t generations_published = 0;
        uint64_t merges_run = 0;
        uint64_t segments_merged = 0;
        uint64_t bytes_written = 0;
        uint64_t documents_indexed = 0;
        uint64_t documents_deleted = 0;

        /// Bytes written divided by bytes of live data. The merge policy's cost,
        /// measured the same way the LSM comparison in another project measures it.
        double WriteAmplification() const;
    };
    Stats stats() const;

public:
    /// Declared but never defined here, so it names nothing and exposes nothing; the
    /// definition is private to src/index/index.cpp. Public only because the free
    /// functions in that file take it by reference, and a private nested type cannot be
    /// named at namespace scope even inside its own translation unit.
    struct Impl;

private:
    /// Everything the implementation needs that this header has no business naming: the
    /// write buffer, the open segment readers, the external-id locator, the segment-id
    /// counter and the files waiting to be unlinked.
    ///
    /// Opaque on purpose. Those members are all spelled in terms of types private to
    /// src/index/ (`SegmentBuilder`, `SegmentReaderImpl`), so naming them here would drag
    /// the storage format's headers into the public interface and make every consumer of
    /// an index recompile whenever the segment layout changed. Declaring the struct
    /// without defining it keeps the dependency pointing one way.
    std::unique_ptr<Impl> impl_;

    Options options_;
    mutable std::shared_mutex mutex_;
    std::shared_ptr<Generation> current_;
    std::atomic<uint64_t> next_version_{1};
    Stats stats_;
};

}  // namespace meridian
