#pragma once

// Retrieval and ranking, which are deliberately two different things.
//
// The split is the project's organising idea. **Retrieval optimises recall** — it
// produces a small candidate set that probably contains the good documents, and it
// must be cheap enough to run over the whole corpus. **Ranking optimises precision**
// — it orders those candidates well, and it may be expensive precisely because it
// only ever sees a few hundred of them.
//
// Collapsing the two is what makes a search system either slow or bad: a cheap scorer
// applied to everything ranks poorly, and an expensive scorer applied to everything
// cannot meet a latency budget. Every feature in the reranker below would be
// unaffordable over a million documents and is trivial over 500.

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "meridian/index.hpp"
#include "meridian/postings.hpp"
#include "meridian/status.hpp"

namespace meridian {

/// How a query's terms combine.
enum class QueryMode : uint8_t {
    kDisjunctive,  // OR: any term matches. High recall, large candidate set.
    kConjunctive,  // AND: all terms. Far cheaper, and where skip lists pay off.
    kPhrase,       // adjacent positions required; needs positional postings
};

/// Structured predicates applied alongside text relevance.
///
/// Evaluated against DocumentMeta rather than the term dictionary. A filter on a
/// low-cardinality field would otherwise be a posting list containing a third of the
/// corpus, which costs more to decode than the documents it eliminates.
struct Filters {
    std::optional<std::string> language;
    std::optional<std::string> category;
    std::optional<int64_t> published_after;
    std::optional<int64_t> published_before;

    bool empty() const {
        return !language && !category && !published_after && !published_before;
    }
    bool Matches(const DocumentMeta& doc) const;
};

/// BM25 parameters.
///
/// k1 controls how quickly term frequency saturates — the fifth occurrence of a word
/// says much less than the second, and k1 is how much less. b controls length
/// normalisation. The defaults are the conventional ones; the benchmarks sweep them
/// rather than assuming the convention is right for this corpus.
struct Bm25Params {
    double k1 = 1.2;
    double b = 0.75;
};

/// Per-field weights, so a title match counts for more than a body match.
struct FieldWeights {
    double title = 3.0;
    double body = 1.0;
    double tags = 1.5;
};

/// A parsed query.
struct Query {
    std::string raw;
    std::vector<std::string> terms;   // tokenized and normalized
    QueryMode mode = QueryMode::kDisjunctive;
    Filters filters;

    /// Embedding for the semantic path. Absent means lexical-only.
    std::optional<std::vector<float>> embedding;

    /// How many results the caller wants. Distinct from the candidate-set size: the
    /// reranker needs more candidates than it returns or it has nothing to reorder.
    uint32_t top_k = 10;

    /// Candidates to generate before reranking. The recall/latency dial, and the
    /// benchmarks sweep it against NDCG.
    uint32_t candidate_k = 200;

    /// Latency budget. A query that cannot finish in time is refused rather than
    /// allowed to consume unbounded resources; see serve/.
    uint32_t deadline_ms = 250;
};

/// A candidate from one retrieval path, before merging.
struct Candidate {
    DocId doc = 0;
    double score = 0.0;

    /// Which path produced it. Kept because the scores from different paths are not
    /// comparable and the merge has to know which scale each came from.
    enum class Source : uint8_t { kLexical, kSemantic, kBoth } source = Source::kLexical;

    double bm25 = 0.0;
    double vector_similarity = 0.0;
};

/// A final result.
struct Hit {
    DocId doc = 0;
    std::string external_id;
    double score = 0.0;

    /// The feature values that produced the score. Carried so a ranking change can be
    /// explained rather than merely observed — without this, "why did result 3 move"
    /// is unanswerable.
    std::unordered_map<std::string, double> features;
};

/// Lexical retrieval over the inverted index.
class LexicalRetriever {
public:
    LexicalRetriever(Bm25Params params, FieldWeights weights);

    /// Produces up to `candidate_k` candidates scored by BM25.
    ///
    /// For conjunctive queries the merge orders iterators by `size_hint` and advances
    /// the rarest first, which is the single largest win available here: the cost
    /// becomes proportional to the rarest term's posting count rather than the
    /// commonest.
    Result<std::vector<Candidate>> Retrieve(const Generation& gen, const Query& query) const;

    /// Scores one document against one term. Exposed so the evaluation harness can
    /// decompose a score rather than treat it as opaque.
    double Bm25(const CorpusStats& corpus, const TermStats& term, uint32_t tf,
                uint32_t doc_length) const;

    struct Stats {
        uint64_t postings_decoded = 0;
        uint64_t skips_taken = 0;
        uint64_t candidates_scored = 0;
        uint64_t segments_searched = 0;
    };
    Stats stats() const { return stats_; }

private:
    Bm25Params params_;
    FieldWeights weights_;
    mutable Stats stats_;
};

/// Approximate nearest-neighbour search for the semantic path.
///
/// The interface is narrow on purpose: build, search, and report what the search
/// actually cost. The interesting parameter is `ef_search`, which trades recall
/// against latency continuously — the benchmarks plot that curve, because "ANN is
/// fast" without a recall number beside it is not a claim about anything.
class AnnIndex {
public:
    virtual ~AnnIndex() = default;

    virtual Status Add(DocId doc, std::span<const float> vector) = 0;
    virtual Status Build() = 0;

    /// Searches for `k` neighbours with the given search effort.
    virtual Result<std::vector<Candidate>> Search(std::span<const float> query, uint32_t k,
                                                 uint32_t ef_search) const = 0;

    /// Exact brute-force search. Not for production use — it exists so recall can be
    /// measured against ground truth, which is the only way an approximate index's
    /// quality is knowable at all.
    virtual Result<std::vector<Candidate>> SearchExact(std::span<const float> query,
                                                       uint32_t k) const = 0;

    virtual size_t size() const = 0;

    struct Stats {
        uint64_t distance_computations = 0;
        uint64_t nodes_visited = 0;
        uint64_t hops = 0;
    };
    virtual Stats stats() const = 0;
};

/// Merges candidates from heterogeneous retrieval paths.
///
/// This is harder than it looks and the reason is worth stating: a BM25 score and a
/// cosine similarity are not on the same scale, are not bounded alike, and are not
/// even monotone in the same way across queries. Adding them directly lets whichever
/// has the larger numeric range dominate. So the merge normalises per query and per
/// path before combining, and the strategy is selectable because which one is right
/// is an empirical question the evaluation harness answers.
class HybridMerger {
public:
    enum class Strategy : uint8_t {
        /// Rank-based: combine 1/(k + rank) from each path. Insensitive to score
        /// scale by construction, which is its main virtue.
        kReciprocalRankFusion,
        /// Normalise each path's scores to [0,1] within this query, then weight.
        /// Sensitive to outliers — one very high score compresses everything else.
        kNormalizedLinear,
        /// Z-score each path, then weight. Handles outliers better and assumes a
        /// distribution shape that short candidate lists may not have.
        kZScoreLinear,
    };

    HybridMerger(Strategy strategy, double lexical_weight, double semantic_weight);

    std::vector<Candidate> Merge(const std::vector<Candidate>& lexical,
                                 const std::vector<Candidate>& semantic,
                                 uint32_t candidate_k) const;

private:
    Strategy strategy_;
    double lexical_weight_;
    double semantic_weight_;
};

/// Second-stage ranking over the candidate set.
///
/// Features are computed only for candidates, which is the entire reason the retrieval
/// and ranking split exists — phrase matching and freshness decay over a million
/// documents would be unaffordable, and over 500 they are free.
class Reranker {
public:
    virtual ~Reranker() = default;

    /// Reorders candidates and fills each Hit's feature map.
    virtual Result<std::vector<Hit>> Rank(const Generation& gen, const Query& query,
                                          const std::vector<Candidate>& candidates) const = 0;

    /// The feature names this reranker produces, so the evaluation harness can report
    /// per-feature contribution without guessing.
    virtual std::vector<std::string> FeatureNames() const = 0;
};

/// Where a query's time went.
///
/// Instrumented per stage because an end-to-end latency number cannot be acted on. A
/// p99 regression is a different problem depending on whether it is in posting
/// retrieval, the ANN hop count, or reranking, and this is how that is told apart.
struct QueryTrace {
    uint64_t parse_ns = 0;
    uint64_t lexical_ns = 0;
    uint64_t ann_ns = 0;
    uint64_t merge_ns = 0;
    uint64_t rerank_ns = 0;
    uint64_t materialize_ns = 0;
    uint64_t total_ns = 0;

    uint32_t lexical_candidates = 0;
    uint32_t semantic_candidates = 0;
    uint32_t merged_candidates = 0;

    bool cache_hit_results = false;
    bool cache_hit_postings = false;
    uint64_t generation_version = 0;
};

struct SearchResponse {
    std::vector<Hit> hits;
    QueryTrace trace;
    uint32_t total_candidates = 0;
};

}  // namespace meridian
