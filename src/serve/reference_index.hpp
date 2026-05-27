#pragma once

// An in-memory corpus and a complete set of stage implementations on top of it.
//
// Why this exists: the serving layer's job is admission control, caching, deadlines and
// tracing, and none of that can be exercised — or demonstrated to a reader — without
// something that actually retrieves documents. The engine's own index and retrieval
// (`Index`, `Generation`, `LexicalRetriever`) live in translation units this layer does
// not own and were not yet written when the serving layer was built, so this file provides
// the smallest honest stand-in: a segmented, generation-versioned, in-memory index with
// BM25 scoring and brute-force vector search.
//
// It stays after the engine's own index lands, for two reasons that are not about
// expedience. It needs no filesystem, so a test can publish a generation or run a merge at
// an exact point in a query's execution; and it has no persistence format, so a serving
// test cannot fail because of a storage change. The adapters in stages_engine.cpp are how
// the same pipeline runs on the real index.
//
// It is deliberately *structurally* faithful rather than fast, because the structure is
// what the serving layer depends on:
//
//  - Segments are immutable once published. This is what makes posting cache entries
//    valid across generations, so a stand-in that mutated segments in place would make
//    the cache tests prove the wrong thing.
//  - Generations are versioned and published atomically, and a reader pins one with a
//    shared_ptr. A merge publishes a replacement and drops its reference; in-flight
//    readers keep the old snapshot alive.
//  - Deletes are tombstones held by the *generation*, not by the segment, which is what
//    lets a delete happen without touching immutable data.
//
// When src/index/ and src/rank/ land, `stages_engine.cpp` swaps in for this file with no
// change to pipeline.cpp, server.cpp, or the tests' structure.

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "meridian/index.hpp"
#include "meridian/postings.hpp"
#include "meridian/retrieval.hpp"
#include "pipeline.hpp"

namespace meridian::serve {

constexpr size_t kFieldCount = static_cast<size_t>(Field::kFieldCount);

/// Lowercases and splits on anything that is not a letter or digit.
///
/// Exposed as a free function because the indexer and the query parser must agree
/// exactly: a query term normalized differently from the indexed term matches nothing,
/// and that failure looks like "the engine found no results" rather than like a bug.
std::vector<std::string> Tokenize(std::string_view text);

/// Cosine similarity, 0 for a zero-length vector rather than NaN. A NaN score
/// propagates into the merge and sorts unpredictably, which is a much harder failure to
/// read than a zero.
double CosineSimilarity(const std::vector<float>& a, const std::vector<float>& b);

class ReferenceIndex : public GenerationSource {
public:
    struct Document {
        DocumentMeta meta;
        std::unordered_map<Field, std::vector<std::string>> tokens;
        std::vector<float> embedding;
    };

    struct PostingList {
        std::vector<Posting> postings;
        TermStats stats;
    };

    struct Entry {
        DocumentMeta meta;
        std::vector<float> embedding;
    };

    /// One immutable segment. Documents are numbered from `doc_base`, matching
    /// SegmentMeta's reason for having one: segments are built independently and a
    /// merge must not have to renumber anything it copies.
    struct Segment {
        uint64_t id = 0;
        DocId doc_base = 0;
        std::vector<Entry> docs;
        std::array<std::unordered_map<std::string, PostingList>, kFieldCount> terms;

        uint64_t term_count() const;
        uint64_t EstimateBytes() const;
    };

    /// The stand-in for `Generation`: a consistent, immutable view of the whole index.
    struct Snapshot {
        uint64_t version = 0;
        std::vector<std::shared_ptr<const Segment>> segments;
        CorpusStats stats;

        /// Corpus-wide document frequency per field, aggregated at publication. BM25's
        /// IDF needs a number that spans segments, and computing it per query would mean
        /// touching every segment's dictionary on every term.
        std::array<std::unordered_map<std::string, uint32_t>, kFieldCount> document_frequency;

        /// Generation-level tombstones. Outside the segments on purpose: a delete must
        /// not modify immutable data, and keeping tombstones here is what preserves the
        /// posting cache's freedom from invalidation.
        std::unordered_set<DocId> deleted;

        const Entry* Resolve(DocId global) const;
        bool IsDeleted(DocId global) const { return deleted.count(global) != 0; }
        uint64_t term_count() const;
        uint64_t EstimateBytes() const;
    };

    ReferenceIndex();

    /// Buffers a document. Not searchable until Flush, exactly as Index::Add specifies.
    void Add(Document doc);

    /// Publishes the buffer as a new segment in a new generation. Returns the version.
    uint64_t Flush();

    /// Tombstones a document in a new generation. False when no such document exists.
    bool Delete(const std::string& external_id);

    /// Merges every segment into one, dropping tombstoned documents, and publishes the
    /// result. Safe to run while queries are reading: it builds the new segment from
    /// snapshots it holds by reference and only then swaps the generation pointer.
    uint64_t Merge();

    std::shared_ptr<const Snapshot> Current() const;
    GenerationPin Acquire() const override;

    struct Stats {
        uint64_t documents = 0;
        uint64_t segments = 0;
        uint64_t terms = 0;
        uint64_t bytes = 0;
        uint64_t generations = 0;
        uint64_t merges = 0;
    };
    Stats stats() const;

private:
    void Publish(std::vector<std::shared_ptr<const Segment>> segments,
                 std::unordered_set<DocId> deleted);

    mutable std::shared_mutex mutex_;
    std::vector<Document> buffer_;
    std::shared_ptr<const Snapshot> current_;
    uint64_t next_version_ = 1;
    uint64_t next_segment_ = 1;
    DocId next_doc_base_ = 0;
    uint64_t generations_ = 0;
    uint64_t merges_ = 0;
};

/// Recovers the snapshot a pin refers to. Fails loudly rather than returning null on a
/// pin that came from a different source: a silent null here would surface as an empty
/// result set, which is the failure mode this project treats as the worst one.
std::shared_ptr<const ReferenceIndex::Snapshot> SnapshotFrom(const GenerationPin& pin);

// ---------------------------------------------------------------------------
// Stages
// ---------------------------------------------------------------------------

class ReferenceParseStage : public ParseStage {
public:
    Result<Query> Parse(const Query& request) const override;
};

/// BM25 retrieval over the pinned snapshot, through the posting list cache.
class ReferenceLexicalStage : public LexicalStage {
public:
    ReferenceLexicalStage(Bm25Params params = {}, FieldWeights weights = {})
        : params_(params), weights_(weights) {}

    Result<std::vector<Candidate>> Retrieve(const StageContext& ctx,
                                            const Query& query) const override;

private:
    Bm25Params params_;
    FieldWeights weights_;
};

/// Brute-force vector search. Exact, which is the right choice for a stand-in: an
/// approximate stand-in would make recall numbers meaningless without adding anything
/// the serving layer needs.
///
/// It stayed after the HNSW path was wired up (see semantic_ann.hpp) because recall is
/// *defined* as agreement with it. Replacing it with the graph would leave the approximate
/// path's quality not merely unmeasured but unmeasurable, and the tools select between the
/// two rather than choosing for the reader.
class ReferenceSemanticStage : public SemanticStage {
public:
    Result<std::vector<Candidate>> Search(const StageContext& ctx,
                                          const Query& query) const override;
};

/// Reciprocal rank fusion. Chosen over score addition for the reason retrieval.hpp
/// gives: a BM25 score and a cosine similarity are not on the same scale, and RRF is
/// insensitive to scale by construction.
class ReferenceMergeStage : public MergeStage {
public:
    explicit ReferenceMergeStage(double lexical_weight = 1.0, double semantic_weight = 1.0,
                                 double k = 60.0)
        : lexical_weight_(lexical_weight), semantic_weight_(semantic_weight), k_(k) {}

    std::vector<Candidate> Merge(const std::vector<Candidate>& lexical,
                                const std::vector<Candidate>& semantic,
                                uint32_t candidate_k) const override;

private:
    double lexical_weight_;
    double semantic_weight_;
    double k_;
};

/// Second-stage scoring with features kept in the Hit, so a ranking change can be
/// explained instead of merely observed.
class ReferenceRerankStage : public RerankStage {
public:
    explicit ReferenceRerankStage(int64_t now_seconds = 0) : now_seconds_(now_seconds) {}

    Result<std::vector<Hit>> Rank(const StageContext& ctx, const Query& query,
                                  const std::vector<Candidate>& candidates) const override;
    std::vector<std::string> FeatureNames() const;

private:
    int64_t now_seconds_;
};

class ReferenceMaterializeStage : public MaterializeStage {
public:
    Status Materialize(const StageContext& ctx, const Query& query,
                       std::vector<Hit>& hits) const override;
};

/// The five stages plus the two caches, assembled. A convenience for the CLIs and the
/// tests, which otherwise repeat the same wiring.
struct ReferenceEngine {
    explicit ReferenceEngine(PipelineOptions options = {}, size_t result_cache_bytes = 8u << 20,
                             size_t posting_cache_bytes = 64u << 20);

    ReferenceIndex index;
    QueryResultCache result_cache;
    PostingListCache posting_cache;
    ReferenceParseStage parse;
    ReferenceLexicalStage lexical;
    ReferenceSemanticStage semantic;
    ReferenceMergeStage merge;
    ReferenceRerankStage rerank;
    ReferenceMaterializeStage materialize;
    Pipeline pipeline;
};

}  // namespace meridian::serve
