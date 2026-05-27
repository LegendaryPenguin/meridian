#pragma once

// Second-stage ranking.
//
// Everything here is affordable only because it runs over a few hundred candidates. The
// cost shape is worth being explicit about, because the whole retrieval/ranking split is
// justified by it:
//
//   - Per *query*, per segment: 3 term-dictionary lookups per query term, to open the
//     posting cursors the match features read.
//   - Per *candidate*: one `advance` per (term, field) cursor and one `DocumentMeta`
//     read. The candidates are sorted by document id first, so the cursors only ever
//     move forward — the whole candidate set costs one monotone pass over the relevant
//     portion of each posting list, not one seek per candidate.
//   - Per candidate, zero document-text reads. This is the important one.
//
// **What a feature needing full document text would cost.** None of the features below
// read it, and that is a deliberate restriction rather than an oversight. The stored
// document body is not in the posting lists; fetching it is a random read into the
// segment's stored-fields region per candidate. At 200 candidates that is 200 random
// reads, which on a cold page cache is comfortably the dominant term in the whole query
// — more than posting decode, ANN traversal and scoring combined — and it does not
// amortise, because candidate sets differ per query. So a feature that genuinely needs
// text (a snippet-quality score, a readability measure, a cross-encoder) has to be
// treated as a third stage over the top ~20, not as a reranker feature over the top 200.
// Where a text-derived signal is wanted cheaply, the positional postings already in the
// index give it: `phrase_match` below is exactly a text feature computed without the
// text.
//
// Note what is *absent*: this file does not tokenize anything. The query's terms arrive
// already analysed and the index stores analysed tokens, so matching is a dictionary
// lookup. That is a direct dividend of the index and query paths sharing one tokenizer —
// if they did not, every feature here would silently read zero.

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "meridian/index.hpp"
#include "meridian/retrieval.hpp"

namespace meridian::rank {

/// Feature names, as constants rather than string literals at use sites, because the
/// name is the join key between the extractor, the weight table and the `Hit::features`
/// map and a typo in any one of them is a feature that silently weighs nothing.
namespace features {
inline constexpr const char* kBm25 = "bm25";
inline constexpr const char* kVectorSimilarity = "vector_similarity";
inline constexpr const char* kTitleMatch = "title_match";
inline constexpr const char* kBodyMatch = "body_match";
inline constexpr const char* kTagsMatch = "tags_match";
inline constexpr const char* kPhraseMatch = "phrase_match";
inline constexpr const char* kFreshness = "freshness";
inline constexpr const char* kPopularity = "popularity";
}  // namespace features

/// Prefix for the weighted contribution entries added alongside each raw feature value.
///
/// The raw value answers "what was this document like"; the contribution answers "what
/// moved it", which is the question the feature map exists for. Keeping both costs eight
/// doubles per hit and removes the need to reconstruct the weights from a config file to
/// read a ranking explanation.
inline constexpr const char* kContributionPrefix = "contribution:";

struct FeatureConfig {
    /// Reference time for freshness, in epoch seconds. Zero means "ask the clock".
    ///
    /// Explicit rather than always-the-clock because a freshness feature read from the
    /// wall clock makes every ranking test time-dependent, and a test that passes in
    /// March and fails in April is worse than no test.
    int64_t now_epoch_seconds = 0;

    /// Freshness half-life.
    ///
    /// The decay is exponential: f = 2^(-age_days / half_life). Chosen over the two
    /// obvious alternatives for reasons that are about shape, not fit:
    ///
    ///  - Reciprocal, 1/(1 + age), has a fat tail. A five-year-old document still scores
    ///    about a tenth of a new one, which in a linear combination is enough for a
    ///    stale document with good text to beat a fresh one, forever. News ranking fails
    ///    in exactly this way.
    ///  - Linear decay with a cutoff is cheap but has a discontinuity, and every
    ///    document past the cutoff is equally stale, so ordering within the long tail is
    ///    lost entirely.
    ///
    /// Exponential decay is bounded in (0, 1] so it cannot dominate a weighted sum, is
    /// smooth so a document does not jump rank as it ages past a boundary, and has one
    /// parameter expressed in the units the domain already thinks in — "how many days
    /// until this matters half as much". That last property is why it is tunable by
    /// someone who is not the author.
    double freshness_half_life_days = 30.0;

    /// Documents published after `now` (clock skew, or scheduled publication) are
    /// clamped to maximum freshness rather than allowed to score above 1.0, which would
    /// let a wrong timestamp buy unbounded rank.
    bool clamp_future_to_now = true;
};

/// Computes the feature vector for a candidate set.
///
/// Separate from the reranker that consumes it, so a learned reranker can be dropped in
/// later against the same features, and so the evaluation harness can dump features
/// without running a ranker at all.
class FeatureExtractor {
public:
    explicit FeatureExtractor(FeatureConfig config = {}) : config_(config) {}

    /// Returns one feature map per candidate, in the candidates' input order, plus each
    /// candidate's external id (which the caller needs for the `Hit` and which this
    /// already paid to load).
    struct Row {
        DocId doc = 0;
        std::string external_id;
        std::unordered_map<std::string, double> features;
    };

    Result<std::vector<Row>> Extract(const Generation& gen, const Query& query,
                                     const std::vector<Candidate>& candidates) const;

    static std::vector<std::string> Names();

private:
    FeatureConfig config_;
};

/// Weights for `LinearReranker`, keyed by feature name.
struct LinearWeights {
    std::unordered_map<std::string, double> weights;

    /// The default weights.
    ///
    /// A caution that belongs next to the numbers: `bm25` is the only feature not bounded
    /// to roughly [0, 1]. Its weight is therefore in BM25 units and is only meaningful
    /// relative to the scale the score arrives on — raw for a lexical-only query,
    /// normalised to [0,1] or z-scored when `HybridMerger` ran first. Changing the merge
    /// strategy changes what this weight means, which is the one coupling between the
    /// two files and is why the evaluation harness sweeps them together.
    static LinearWeights Default();

    double Get(const std::string& name) const;
};

/// A linear model over the features.
///
/// The default reranker, and deliberately so. A linear model is the only kind whose
/// ranking decisions can be read off directly: the contribution of each feature is a
/// single multiplication, so "this document moved up because its freshness contribution
/// is 0.28 and the one it passed is 0.02" is a statement that can be made from the
/// `Hit::features` map alone. A gradient-boosted model would very likely rank better and
/// would make that sentence unwriteable, which is a real cost when the question being
/// asked is "did the ranking change for the reason we intended".
class LinearReranker : public Reranker {
public:
    explicit LinearReranker(LinearWeights weights = LinearWeights::Default(),
                            FeatureConfig config = {})
        : weights_(std::move(weights)), extractor_(config) {}

    Result<std::vector<Hit>> Rank(const Generation& gen, const Query& query,
                                  const std::vector<Candidate>& candidates) const override;

    std::vector<std::string> FeatureNames() const override;

    const LinearWeights& weights() const { return weights_; }

private:
    LinearWeights weights_;
    FeatureExtractor extractor_;
};

}  // namespace meridian::rank
