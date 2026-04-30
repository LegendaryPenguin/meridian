#pragma once

// Evaluation, and why it makes optimisation multi-objective.
//
// The point of this header is a constraint rather than a metric: **a ranking change
// that raises NDCG and doubles p99 latency is not automatically an improvement.**
// Quality and latency are separate axes, a change can move them in opposite
// directions, and a harness that reports only one of them will happily walk you into
// a slower search engine that scores better on a dataset nobody is waiting on.
//
// So every result this header produces carries both, and the comparison function
// refuses to declare a winner on quality alone.

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "meridian/retrieval.hpp"
#include "meridian/status.hpp"

namespace meridian {

/// Graded relevance for one document against one query.
///
/// Graded rather than binary because NDCG needs grades to mean anything, and because
/// "relevant / not relevant" cannot express the difference between the right answer
/// and a reasonable one.
struct Judgement {
    std::string external_id;
    /// 0 = irrelevant, 1 = marginal, 2 = relevant, 3 = highly relevant.
    int grade = 0;
};

/// One labelled query.
struct JudgedQuery {
    std::string id;
    std::string text;
    Filters filters;
    std::vector<Judgement> judgements;

    /// Grade for a document, 0 when unjudged.
    ///
    /// Unjudged-means-zero is the standard assumption and it is also a known source of
    /// bias: a genuinely good result that nobody labelled is scored as a mistake. The
    /// harness reports how many returned documents were unjudged, so that bias is
    /// visible rather than silently folded into the score.
    int GradeOf(const std::string& external_id) const;
};

/// Quality metrics at a cutoff.
struct QualityMetrics {
    uint32_t k = 10;

    /// Fraction of the top k that are relevant (grade >= 1).
    double precision_at_k = 0.0;

    /// Fraction of all known relevant documents found in the top k. Bounded above by
    /// k / |relevant|, so it is not comparable across queries with different numbers
    /// of relevant documents — reported per query as well as averaged for that reason.
    double recall_at_k = 0.0;

    /// Reciprocal of the rank of the first relevant result. Sensitive only to that
    /// one position, which makes it the right metric when a user reads one answer and
    /// the wrong one when they read a list.
    double mrr = 0.0;

    /// Discounted cumulative gain normalised by the ideal ordering. The only metric
    /// here that uses the grades rather than thresholding them.
    double ndcg_at_k = 0.0;

    /// How many of the returned top-k documents had no judgement at all. High values
    /// mean the other numbers are measuring the judgements' coverage as much as the
    /// ranking's quality.
    uint32_t unjudged_in_top_k = 0;
};

/// Latency distribution over a query set.
struct LatencyMetrics {
    uint64_t p50_ns = 0;
    uint64_t p95_ns = 0;
    uint64_t p99_ns = 0;
    uint64_t max_ns = 0;
    uint64_t mean_ns = 0;

    /// Per-stage medians, so a regression can be attributed rather than merely noticed.
    uint64_t lexical_p50_ns = 0;
    uint64_t ann_p50_ns = 0;
    uint64_t rerank_p50_ns = 0;
};

/// One evaluation run: a configuration measured on both axes.
struct EvalRun {
    std::string label;
    QualityMetrics quality;
    LatencyMetrics latency;

    uint32_t queries = 0;
    uint32_t queries_timed_out = 0;
    double mean_candidates = 0.0;
    double cache_hit_rate = 0.0;

    /// Per-query quality, so a mean that hides a bimodal distribution can be caught.
    std::vector<std::pair<std::string, QualityMetrics>> per_query;
};

/// The verdict when two configurations are compared.
///
/// `kTradeoff` exists because it is the honest answer most of the time, and a
/// comparison that can only say "better" or "worse" will label a tradeoff as a win.
enum class Verdict : uint8_t {
    kBetter,    // quality up or equal, and latency not materially worse
    kWorse,     // quality down, or latency materially worse with no quality gain
    kTradeoff,  // quality up AND latency materially worse — a decision, not a result
    kNoChange,  // neither axis moved beyond noise
};

const char* VerdictName(Verdict v);

struct Comparison {
    Verdict verdict = Verdict::kNoChange;
    double ndcg_delta = 0.0;
    double p99_ratio = 1.0;
    std::string explanation;
};

/// Compares two runs on both axes.
///
/// Refuses to call a quality gain an improvement when p99 has materially regressed;
/// that case returns kTradeoff with both numbers in the explanation, because the
/// decision belongs to a human who knows what the latency budget is for.
Comparison Compare(const EvalRun& baseline, const EvalRun& candidate,
                   double latency_tolerance = 0.10);

/// Computes quality metrics for one query's results.
QualityMetrics Evaluate(const JudgedQuery& query, const std::vector<Hit>& hits, uint32_t k);

/// Aggregates per-query metrics. Macro-averaged — every query counts equally,
/// regardless of how many relevant documents it has — because micro-averaging lets a
/// handful of densely-judged queries dominate the result.
QualityMetrics Aggregate(const std::vector<QualityMetrics>& per_query);

/// Loads a judgement set from a TREC-style qrels file plus a queries file. Two files
/// rather than one because that is the format real labelled collections come in, and
/// a bespoke format would make an external dataset unusable.
Result<std::vector<JudgedQuery>> LoadJudgements(const std::string& queries_path,
                                                const std::string& qrels_path);

}  // namespace meridian
