// Quality metrics, and the comparison that refuses to pick a winner on one axis.
//
// Everything here is deliberately arithmetic over plain data: no index, no clock, no
// files. That is what makes the hand-computed values in tests/metrics_test.cpp possible
// to write, and a metrics implementation that is only ever checked against itself is
// not checked at all.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <unordered_map>

#include "meridian/eval.hpp"

namespace meridian {
namespace {

/// Exponential gain, 2^g - 1.
///
/// Linear gain (gain == grade) makes grade 3 three times as valuable as grade 1, which
/// understates what the grades mean: a highly relevant answer is not three marginal
/// answers. The exponential form is also what every published NDCG number this would be
/// compared against uses, and silently choosing the other one would make the comparison
/// meaningless without making it look wrong.
double Gain(int grade) {
    if (grade <= 0) return 0.0;
    return std::pow(2.0, static_cast<double>(grade)) - 1.0;
}

/// Rank discount, 1 / log2(rank + 1) with rank counted from one.
double Discount(size_t zero_based_rank) {
    return 1.0 / std::log2(static_cast<double>(zero_based_rank) + 2.0);
}

/// Quality movement below this is treated as noise rather than signal.
///
/// A fixed floor rather than a confidence interval because the harness runs a fixed
/// judged set deterministically: the run-to-run variation in NDCG is zero, and what this
/// guards against is a difference in the fourth decimal place being reported as a win.
constexpr double kQualityNoise = 1e-3;

std::string Format(const char* fmt, double a) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), fmt, a);
    return buffer;
}

std::string Format(const char* fmt, unsigned long long a) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), fmt, a);
    return buffer;
}

}  // namespace

int JudgedQuery::GradeOf(const std::string& external_id) const {
    for (const auto& judgement : judgements) {
        if (judgement.external_id == external_id) return judgement.grade;
    }
    // Unjudged means zero. See the header: this is the standard assumption and also a
    // bias, which is why QualityMetrics carries unjudged_in_top_k beside the scores.
    return 0;
}

const char* VerdictName(Verdict v) {
    switch (v) {
        case Verdict::kBetter: return "better";
        case Verdict::kWorse: return "worse";
        case Verdict::kTradeoff: return "tradeoff";
        case Verdict::kNoChange: return "no-change";
    }
    return "unknown";
}

QualityMetrics Evaluate(const JudgedQuery& query, const std::vector<Hit>& hits, uint32_t k) {
    QualityMetrics m;
    m.k = k;
    if (k == 0) return m;

    // A set rather than repeated GradeOf scans: the distinction that matters below is
    // "judged at grade 0" versus "not judged at all", and only a membership test can
    // tell those apart.
    std::unordered_map<std::string, int> judged;
    judged.reserve(query.judgements.size() * 2);
    size_t relevant_total = 0;
    for (const auto& judgement : query.judgements) {
        judged[judgement.external_id] = judgement.grade;
        if (judgement.grade >= 1) ++relevant_total;
    }

    const size_t cutoff = std::min<size_t>(k, hits.size());
    size_t relevant_in_top = 0;
    double dcg = 0.0;
    size_t first_relevant_rank = 0;  // one-based; 0 means none found

    for (size_t i = 0; i < cutoff; ++i) {
        auto it = judged.find(hits[i].external_id);
        if (it == judged.end()) {
            ++m.unjudged_in_top_k;
            continue;
        }
        const int grade = it->second;
        if (grade >= 1) {
            ++relevant_in_top;
            if (first_relevant_rank == 0) first_relevant_rank = i + 1;
        }
        dcg += Gain(grade) * Discount(i);
    }

    // Divided by k, not by the number of results returned. A system that answers a
    // request for ten documents with three has failed to fill the page, and dividing by
    // three would score that identically to filling it — the short list would be
    // rewarded for the positions it never risked.
    m.precision_at_k = static_cast<double>(relevant_in_top) / static_cast<double>(k);

    m.recall_at_k = relevant_total == 0
                        ? 0.0
                        : static_cast<double>(relevant_in_top) / static_cast<double>(relevant_total);

    m.mrr = first_relevant_rank == 0 ? 0.0 : 1.0 / static_cast<double>(first_relevant_rank);

    // The ideal ordering is the best one achievable *from the judged set*, which is the
    // only ordering an NDCG of 1.0 can honestly mean "could not have done better" about.
    // Using the returned list's own grades sorted descending would instead normalise
    // each ranking by itself, and every system would score 1.0 on its own output.
    std::vector<int> ideal_grades;
    ideal_grades.reserve(query.judgements.size());
    for (const auto& judgement : query.judgements) ideal_grades.push_back(judgement.grade);
    std::sort(ideal_grades.begin(), ideal_grades.end(), std::greater<int>());

    double ideal_dcg = 0.0;
    const size_t ideal_cutoff = std::min<size_t>(k, ideal_grades.size());
    for (size_t i = 0; i < ideal_cutoff; ++i) ideal_dcg += Gain(ideal_grades[i]) * Discount(i);

    m.ndcg_at_k = ideal_dcg > 0.0 ? dcg / ideal_dcg : 0.0;
    return m;
}

QualityMetrics Aggregate(const std::vector<QualityMetrics>& per_query) {
    QualityMetrics mean;
    if (per_query.empty()) {
        // k = 0 rather than the default 10, so an aggregate over nothing cannot be
        // mistaken for an aggregate at k = 10 that happened to score zero.
        mean.k = 0;
        return mean;
    }

    mean.k = per_query.front().k;
    const double n = static_cast<double>(per_query.size());
    for (const auto& q : per_query) {
        // Macro-averaging: each query contributes 1/n regardless of how many documents
        // were judged for it. Micro-averaging would pool the numerators and
        // denominators, which hands the result to whichever handful of queries happens
        // to have the densest judgements — a property of the labelling effort, not of
        // the ranker being measured.
        mean.precision_at_k += q.precision_at_k / n;
        mean.recall_at_k += q.recall_at_k / n;
        mean.mrr += q.mrr / n;
        mean.ndcg_at_k += q.ndcg_at_k / n;
        // Summed, not averaged: it is a count of unjudged results across the run, and an
        // average of 0.4 unjudged documents per query says less than "19 of 50 returned
        // documents were never labelled".
        mean.unjudged_in_top_k += q.unjudged_in_top_k;
    }
    return mean;
}

Comparison Compare(const EvalRun& baseline, const EvalRun& candidate, double latency_tolerance) {
    Comparison result;
    result.ndcg_delta = candidate.quality.ndcg_at_k - baseline.quality.ndcg_at_k;

    if (baseline.latency.p99_ns == 0) {
        // No baseline latency to divide by. A candidate with a p99 is infinitely worse
        // than a baseline with none in ratio terms, which is not a useful number, so the
        // ratio stays at 1.0 and the explanation says the latency comparison was not
        // available. Reporting 1.0 silently would claim latency was unchanged.
        result.p99_ratio = 1.0;
    } else {
        result.p99_ratio = static_cast<double>(candidate.latency.p99_ns) /
                           static_cast<double>(baseline.latency.p99_ns);
    }

    const bool have_latency = baseline.latency.p99_ns != 0 && candidate.latency.p99_ns != 0;
    const bool quality_up = result.ndcg_delta > kQualityNoise;
    const bool quality_down = result.ndcg_delta < -kQualityNoise;
    const bool latency_worse = have_latency && result.p99_ratio > 1.0 + latency_tolerance;
    const bool latency_better = have_latency && result.p99_ratio < 1.0 - latency_tolerance;

    // Both numbers go into the explanation unconditionally. The verdict is a summary and
    // a summary is not enough to decide with: whether +0.004 NDCG is worth 1.4x p99
    // depends on a latency budget this code does not know.
    const std::string quality_text =
        "NDCG@" + std::to_string(baseline.quality.k) + " " +
        Format("%+.4f", result.ndcg_delta) + " (" +
        Format("%.4f", baseline.quality.ndcg_at_k) + " -> " +
        Format("%.4f", candidate.quality.ndcg_at_k) + ")";

    const std::string latency_text =
        have_latency ? ("p99 " + Format("%.2fx", result.p99_ratio) + " (" +
                        Format("%lluns", static_cast<unsigned long long>(baseline.latency.p99_ns)) +
                        " -> " +
                        Format("%lluns", static_cast<unsigned long long>(candidate.latency.p99_ns)) +
                        ", tolerance " + Format("%.0f%%", latency_tolerance * 100.0) + ")")
                     : std::string("p99 unavailable (no latency samples on one side)");

    if (quality_up && latency_worse) {
        // The case this whole header exists for. Not "better": the quality gain is real
        // and so is the latency cost, and collapsing that into a win is how a search
        // engine ends up slower in exchange for a number on a dataset nobody is waiting
        // on. The decision belongs to whoever owns the budget.
        result.verdict = Verdict::kTradeoff;
        result.explanation = "tradeoff: quality improved and p99 regressed — " + quality_text +
                             " but " + latency_text;
    } else if (quality_up) {
        result.verdict = Verdict::kBetter;
        result.explanation = "better: " + quality_text + ", " + latency_text;
    } else if (quality_down) {
        result.verdict = Verdict::kWorse;
        result.explanation = "worse: quality regressed — " + quality_text + ", " + latency_text;
    } else if (latency_worse) {
        result.verdict = Verdict::kWorse;
        result.explanation =
            "worse: p99 regressed with no quality gain — " + latency_text + ", " + quality_text;
    } else if (latency_better) {
        result.verdict = Verdict::kBetter;
        result.explanation =
            "better: same quality, lower p99 — " + latency_text + ", " + quality_text;
    } else {
        result.verdict = Verdict::kNoChange;
        result.explanation = "no change: " + quality_text + ", " + latency_text;
    }
    return result;
}

}  // namespace meridian
