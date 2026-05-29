// Metric tests against values computed by hand in the comments.
//
// Every expectation below is derived from the definition on paper, not from running the
// code and recording what it said. That distinction is the whole point: an
// implementation checked against its own output agrees with itself about a shared
// mistake, and the mistakes that matter here (linear instead of exponential gain, an
// ideal DCG taken from the returned list rather than the judged set, a discount off by
// one position) all produce numbers that look entirely reasonable.
//
// Conventions used throughout, stated once:
//   gain(g)      = 2^g - 1
//   discount(r)  = 1 / log2(r + 1), r counted from 1
//   log2(2) = 1, log2(3) = 1.5849625007, log2(4) = 2, log2(5) = 2.3219280949,
//   log2(6) = 2.5849625007
//   P@k divides by k, not by the number of results returned.

#include <cmath>
#include <string>
#include <vector>

#include "meridian/eval.hpp"
#include "testing.hpp"

using namespace meridian;

namespace {

/// Tolerance for a metric compared against a decimal worked out by hand. Loose enough
/// that the comment can carry nine digits rather than seventeen, tight enough that
/// linear-versus-exponential gain or an off-by-one discount cannot hide inside it.
constexpr double kEpsilon = 1e-9;

void CheckNear(double actual, double expected, const char* what, int line) {
    if (std::fabs(actual - expected) > kEpsilon) {
        testing::ReportFailure(__FILE__, line,
                               std::string(what) + ": expected " + std::to_string(expected) +
                                   ", got " + std::to_string(actual));
    }
}

#define CHECK_NEAR(actual, expected) CheckNear((actual), (expected), #actual, __LINE__)

std::vector<Hit> HitsFor(const std::vector<std::string>& ids) {
    std::vector<Hit> hits;
    hits.reserve(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        Hit hit;
        hit.doc = static_cast<DocId>(i);
        hit.external_id = ids[i];
        // Descending so the hit list is in the order a ranker would have produced it;
        // Evaluate reads position, not score, but a test fixture that contradicts itself
        // is a trap for whoever changes Evaluate later.
        hit.score = 1.0 - static_cast<double>(i) * 0.01;
        hits.push_back(std::move(hit));
    }
    return hits;
}

JudgedQuery Judged(const std::vector<std::pair<std::string, int>>& grades) {
    JudgedQuery query;
    query.id = "q1";
    query.text = "test query";
    for (const auto& [id, grade] : grades) query.judgements.push_back(Judgement{id, grade});
    return query;
}

}  // namespace

// ---------------------------------------------------------------------------
// GradeOf
// ---------------------------------------------------------------------------

MERIDIAN_TEST(GradeOfReturnsZeroForUnjudged) {
    const JudgedQuery query = Judged({{"d1", 3}, {"d2", 0}});
    CHECK_EQ(query.GradeOf("d1"), 3);
    // An explicit grade of 0 and no judgement at all both read as 0 here. They are
    // distinguished by unjudged_in_top_k, not by this function.
    CHECK_EQ(query.GradeOf("d2"), 0);
    CHECK_EQ(query.GradeOf("d-nonexistent"), 0);
}

// ---------------------------------------------------------------------------
// Hand-computed metric cases
// ---------------------------------------------------------------------------

MERIDIAN_TEST(AllRelevant) {
    // 4 hits, all judged grade 2, k = 4.
    //   relevant in top 4 = 4  ->  P@4 = 4/4 = 1.0
    //   |relevant| = 4         ->  R@4 = 4/4 = 1.0
    //   first relevant at rank 1 -> MRR = 1.0
    //   DCG  = 3/1 + 3/1.5849625007 + 3/2 + 3/2.3219280949
    //   IDCG is the same sum (same grades, best possible order) -> NDCG = 1.0
    //   unjudged = 0
    const JudgedQuery query = Judged({{"d1", 2}, {"d2", 2}, {"d3", 2}, {"d4", 2}});
    const QualityMetrics m = Evaluate(query, HitsFor({"d1", "d2", "d3", "d4"}), 4);

    CHECK_EQ(m.k, 4u);
    CHECK_NEAR(m.precision_at_k, 1.0);
    CHECK_NEAR(m.recall_at_k, 1.0);
    CHECK_NEAR(m.mrr, 1.0);
    CHECK_NEAR(m.ndcg_at_k, 1.0);
    CHECK_EQ(m.unjudged_in_top_k, 0u);
}

MERIDIAN_TEST(NoneRelevant) {
    // 3 hits, all explicitly judged grade 0, k = 3.
    //   relevant in top = 0 -> P@3 = 0
    //   |relevant| = 0      -> R@3 = 0 (nothing to find; not 1.0)
    //   no relevant result  -> MRR = 0
    //   DCG = 0, IDCG = 0   -> NDCG = 0 (0/0 defined as 0, not NaN)
    //   unjudged = 0: they were judged, and judged irrelevant.
    const JudgedQuery query = Judged({{"d1", 0}, {"d2", 0}, {"d3", 0}});
    const QualityMetrics m = Evaluate(query, HitsFor({"d1", "d2", "d3"}), 3);

    CHECK_NEAR(m.precision_at_k, 0.0);
    CHECK_NEAR(m.recall_at_k, 0.0);
    CHECK_NEAR(m.mrr, 0.0);
    CHECK_NEAR(m.ndcg_at_k, 0.0);
    CHECK_EQ(m.unjudged_in_top_k, 0u);
    CHECK(!std::isnan(m.ndcg_at_k));
}

MERIDIAN_TEST(OneRelevantAtRankOne) {
    // 5 hits, only d1 judged (grade 2), k = 5.
    //   P@5 = 1/5 = 0.2
    //   |relevant| = 1 -> R@5 = 1/1 = 1.0
    //   MRR = 1/1 = 1.0
    //   DCG  = 3/log2(2) = 3
    //   IDCG = 3/log2(2) = 3  ->  NDCG = 1.0
    //   unjudged = 4 (d2..d5 carry no judgement at all)
    const JudgedQuery query = Judged({{"d1", 2}});
    const QualityMetrics m = Evaluate(query, HitsFor({"d1", "d2", "d3", "d4", "d5"}), 5);

    CHECK_NEAR(m.precision_at_k, 0.2);
    CHECK_NEAR(m.recall_at_k, 1.0);
    CHECK_NEAR(m.mrr, 1.0);
    CHECK_NEAR(m.ndcg_at_k, 1.0);
    CHECK_EQ(m.unjudged_in_top_k, 4u);
}

MERIDIAN_TEST(OneRelevantAtRankFive) {
    // 5 hits, only d5 judged (grade 1), k = 5.
    //   P@5 = 1/5 = 0.2
    //   R@5 = 1/1 = 1.0
    //   MRR = 1/5 = 0.2
    //   DCG  = (2^1-1)/log2(6) = 1/2.5849625007 = 0.3868528072...
    //   IDCG = 1/log2(2) = 1
    //   NDCG = 0.3868528072
    //   unjudged = 4
    const JudgedQuery query = Judged({{"d5", 1}});
    const QualityMetrics m = Evaluate(query, HitsFor({"d1", "d2", "d3", "d4", "d5"}), 5);

    CHECK_NEAR(m.precision_at_k, 0.2);
    CHECK_NEAR(m.recall_at_k, 1.0);
    CHECK_NEAR(m.mrr, 0.2);
    CHECK_NEAR(m.ndcg_at_k, 1.0 / std::log2(6.0));
    CHECK_NEAR(m.ndcg_at_k, 0.38685280723454163);
    CHECK_EQ(m.unjudged_in_top_k, 4u);
}

MERIDIAN_TEST(PerfectGradedOrdering) {
    // Judgements d1=3, d2=2, d3=1. Hits in that exact order, k = 3.
    //   gains 7, 3, 1; discounts 1/1, 1/1.5849625007, 1/2
    //   DCG  = 7 + 1.8927892607 + 0.5 = 9.3927892607
    //   IDCG = the same, since this *is* the best ordering of the judged set
    //   NDCG = 1.0 exactly
    //   P@3 = 3/3 = 1.0, MRR = 1.0
    const JudgedQuery query = Judged({{"d1", 3}, {"d2", 2}, {"d3", 1}});
    const QualityMetrics m = Evaluate(query, HitsFor({"d1", "d2", "d3"}), 3);

    CHECK_NEAR(m.ndcg_at_k, 1.0);
    CHECK_NEAR(m.precision_at_k, 1.0);
    CHECK_NEAR(m.recall_at_k, 1.0);
    CHECK_NEAR(m.mrr, 1.0);
}

MERIDIAN_TEST(ReversedGradedOrderingScoresLowerAtIdenticalPrecision) {
    // Same judgements, hits reversed: d3 (grade 1), d2 (2), d1 (3). k = 3.
    //   DCG  = 1/1 + 3/1.5849625007 + 7/2
    //        = 1 + 1.8927892607 + 3.5 = 6.3927892607
    //   IDCG = 9.3927892607 (unchanged: the ideal does not depend on what was returned)
    //   NDCG = 6.3927892607 / 9.3927892607 = 0.6806060568
    //
    // Precision is 1.0 in both orderings, which is the point: this pair is the
    // demonstration that NDCG reads the grades and precision does not. A thresholded
    // "NDCG" built on relevant/not-relevant would score both orderings 1.0.
    const JudgedQuery query = Judged({{"d1", 3}, {"d2", 2}, {"d3", 1}});
    const QualityMetrics forward = Evaluate(query, HitsFor({"d1", "d2", "d3"}), 3);
    const QualityMetrics reversed = Evaluate(query, HitsFor({"d3", "d2", "d1"}), 3);

    const double ideal = 7.0 / 1.0 + 3.0 / std::log2(3.0) + 1.0 / 2.0;
    const double actual = 1.0 / 1.0 + 3.0 / std::log2(3.0) + 7.0 / 2.0;
    CHECK_NEAR(reversed.ndcg_at_k, actual / ideal);
    CHECK_NEAR(reversed.ndcg_at_k, 0.6806060567602009);

    CHECK_NEAR(forward.precision_at_k, reversed.precision_at_k);
    CHECK(reversed.ndcg_at_k < forward.ndcg_at_k);
}

MERIDIAN_TEST(NdcgUsesGradesNotThresholdedRelevance) {
    // Two rankings over the same five documents with identical precision, recall and MRR,
    // differing only in which grade sits at which position.
    //
    // Judgements: a=3, b=3, c=1, d=1, e=0. k = 4.
    //
    // Ranking A: a, b, c, d  (grades 3,3,1,1)
    //   DCG = 7/1 + 7/1.5849625007 + 1/2 + 1/2.3219280949
    //       = 7 + 4.4165082750 + 0.5 + 0.4306765581 = 12.3471848331
    // Ranking B: c, d, a, b  (grades 1,1,3,3)
    //   DCG = 1/1 + 1/1.5849625007 + 7/2 + 7/2.3219280949
    //       = 1 + 0.6309297536 + 3.5 + 3.0147359069 = 8.1456656601
    // IDCG for both = ideal ordering 3,3,1,1 = 12.3471848331
    //   NDCG(A) = 1.0
    //   NDCG(B) = 8.1456656601 / 12.3471848331 = 0.6597184516
    //
    // Both rankings return 4 documents of which 4 are relevant (grade >= 1): P@4 = 1.0,
    // R@4 = 4/4 = 1.0, MRR = 1.0. Any metric that thresholds the grades cannot tell them
    // apart; NDCG separates them by a third of its range.
    const JudgedQuery query = Judged({{"a", 3}, {"b", 3}, {"c", 1}, {"d", 1}, {"e", 0}});
    const QualityMetrics a = Evaluate(query, HitsFor({"a", "b", "c", "d"}), 4);
    const QualityMetrics b = Evaluate(query, HitsFor({"c", "d", "a", "b"}), 4);

    CHECK_NEAR(a.precision_at_k, b.precision_at_k);
    CHECK_NEAR(a.recall_at_k, b.recall_at_k);
    CHECK_NEAR(a.mrr, b.mrr);

    const double ideal = 7.0 + 7.0 / std::log2(3.0) + 1.0 / 2.0 + 1.0 / std::log2(5.0);
    const double dcg_b = 1.0 + 1.0 / std::log2(3.0) + 7.0 / 2.0 + 7.0 / std::log2(5.0);
    CHECK_NEAR(a.ndcg_at_k, 1.0);
    CHECK_NEAR(b.ndcg_at_k, dcg_b / ideal);
    CHECK_NEAR(b.ndcg_at_k, 0.6597184516316583);
}

MERIDIAN_TEST(IdealDcgComesFromTheJudgedSetNotTheReturnedList) {
    // Judgements: x=3, y=3, z=3. The ranker returns only z, and returns it first.
    // k = 3.
    //   DCG  = 7/1 = 7
    //   IDCG = 7/1 + 7/1.5849625007 + 7/2 = 14.9165082750
    //   NDCG = 7 / 14.9165082750 = 0.4692787260
    //
    // An implementation that normalised by the best ordering of the *returned* list would
    // compute IDCG = 7 and report NDCG = 1.0 — a perfect score for finding one of three
    // equally good answers.
    const JudgedQuery query = Judged({{"x", 3}, {"y", 3}, {"z", 3}});
    const QualityMetrics m = Evaluate(query, HitsFor({"z"}), 3);

    const double ideal = 7.0 + 7.0 / std::log2(3.0) + 7.0 / 2.0;
    CHECK_NEAR(m.ndcg_at_k, 7.0 / ideal);
    CHECK_NEAR(m.ndcg_at_k, 0.4692787260227565);
    CHECK(m.ndcg_at_k < 1.0);
}

// ---------------------------------------------------------------------------
// Boundaries
// ---------------------------------------------------------------------------

MERIDIAN_TEST(KLargerThanResultList) {
    // Judgements d1=3, d2=3, d3=3. Hits: d1, d2 only. k = 5.
    //   relevant in top = 2 -> P@5 = 2/5 = 0.4 (divided by k: the page was not filled)
    //   R@5 = 2/3 = 0.6666666667
    //   MRR = 1.0
    //   DCG  = 7/1 + 7/1.5849625007 = 11.4165082750
    //   IDCG = top min(5, 3) = 3 ideal grades = 14.9165082750
    //   NDCG = 11.4165082750 / 14.9165082750 = 0.7653606370
    //   unjudged = 0
    const JudgedQuery query = Judged({{"d1", 3}, {"d2", 3}, {"d3", 3}});
    const QualityMetrics m = Evaluate(query, HitsFor({"d1", "d2"}), 5);

    CHECK_NEAR(m.precision_at_k, 0.4);
    CHECK_NEAR(m.recall_at_k, 2.0 / 3.0);
    CHECK_NEAR(m.mrr, 1.0);

    const double ideal = 7.0 + 7.0 / std::log2(3.0) + 7.0 / 2.0;
    const double dcg = 7.0 + 7.0 / std::log2(3.0);
    CHECK_NEAR(m.ndcg_at_k, dcg / ideal);
    CHECK_NEAR(m.ndcg_at_k, 0.7653606369886218);
    CHECK_EQ(m.unjudged_in_top_k, 0u);
}

MERIDIAN_TEST(EmptyResults) {
    // Nothing returned, three relevant documents known. Every metric is zero, and recall
    // in particular is 0 rather than undefined: the documents exist and were not found.
    const JudgedQuery query = Judged({{"d1", 3}, {"d2", 2}, {"d3", 1}});
    const QualityMetrics m = Evaluate(query, {}, 10);

    CHECK_EQ(m.k, 10u);
    CHECK_NEAR(m.precision_at_k, 0.0);
    CHECK_NEAR(m.recall_at_k, 0.0);
    CHECK_NEAR(m.mrr, 0.0);
    CHECK_NEAR(m.ndcg_at_k, 0.0);
    CHECK_EQ(m.unjudged_in_top_k, 0u);
}

MERIDIAN_TEST(EmptyJudgements) {
    // Three results, no judgements at all. Scores are zero and all three results are
    // counted unjudged, which is the signal that the zeros describe the labelling rather
    // than the ranking.
    JudgedQuery query;
    query.id = "q-unlabelled";
    const QualityMetrics m = Evaluate(query, HitsFor({"d1", "d2", "d3"}), 3);

    CHECK_NEAR(m.precision_at_k, 0.0);
    CHECK_NEAR(m.recall_at_k, 0.0);
    CHECK_NEAR(m.ndcg_at_k, 0.0);
    CHECK_EQ(m.unjudged_in_top_k, 3u);
}

MERIDIAN_TEST(KZero) {
    // k = 0 asks for the quality of no results. Every metric is zero and k is reported
    // as 0, so the caller can see the cutoff it actually got rather than a default.
    const JudgedQuery query = Judged({{"d1", 3}});
    const QualityMetrics m = Evaluate(query, HitsFor({"d1", "d2"}), 0);

    CHECK_EQ(m.k, 0u);
    CHECK_NEAR(m.precision_at_k, 0.0);
    CHECK_NEAR(m.recall_at_k, 0.0);
    CHECK_NEAR(m.mrr, 0.0);
    CHECK_NEAR(m.ndcg_at_k, 0.0);
    CHECK_EQ(m.unjudged_in_top_k, 0u);
}

// ---------------------------------------------------------------------------
// Recall's bound, and why it is reported per query
// ---------------------------------------------------------------------------

MERIDIAN_TEST(RecallIsBoundedByKOverRelevantSetSize) {
    // 10 relevant documents, k = 3, and a ranker that puts three of them first.
    // R@3 = 3/10 = 0.3, which is k/|relevant| — the maximum this query can score.
    // A perfect ranker is capped at 0.3 here, so 0.3 is the ceiling and not a weak result.
    std::vector<std::pair<std::string, int>> grades;
    for (int i = 1; i <= 10; ++i) grades.emplace_back("r" + std::to_string(i), 2);
    const JudgedQuery many = Judged(grades);
    const QualityMetrics m = Evaluate(many, HitsFor({"r1", "r2", "r3"}), 3);

    CHECK_NEAR(m.recall_at_k, 0.3);
    CHECK_NEAR(m.precision_at_k, 1.0);
    CHECK_NEAR(m.recall_at_k, 3.0 / 10.0);
}

MERIDIAN_TEST(RecallIsNotComparableAcrossQueriesWithDifferentRelevantSetSizes) {
    // Two perfectly-ranked queries, k = 3. The first has 10 relevant documents and scores
    // R@3 = 0.3; the second has 2 and scores R@3 = 1.0. Both rankings are optimal. The
    // 3.3x difference is entirely a property of the judgements, which is why EvalRun
    // carries per_query and the aggregate is not the whole story.
    std::vector<std::pair<std::string, int>> ten;
    for (int i = 1; i <= 10; ++i) ten.emplace_back("r" + std::to_string(i), 2);

    const QualityMetrics dense = Evaluate(Judged(ten), HitsFor({"r1", "r2", "r3"}), 3);
    const QualityMetrics sparse = Evaluate(Judged({{"s1", 2}, {"s2", 2}}), HitsFor({"s1", "s2"}), 3);

    CHECK_NEAR(dense.recall_at_k, 0.3);
    CHECK_NEAR(sparse.recall_at_k, 1.0);
    // Both are optimal orderings of their judged sets.
    CHECK_NEAR(dense.ndcg_at_k, 1.0);
    CHECK_NEAR(sparse.ndcg_at_k, 1.0);
}

// ---------------------------------------------------------------------------
// unjudged_in_top_k
// ---------------------------------------------------------------------------

MERIDIAN_TEST(UnjudgedCountedOnlyWithinTheCutoff) {
    // Judgements: d1=2, d3=0. Hits: d1, u1, d3, u2, u3. k = 3.
    //   Within the top 3: d1 judged, u1 unjudged, d3 judged (as irrelevant).
    //   unjudged_in_top_k = 1. u2 and u3 sit outside the cutoff and are not counted:
    //   documents the user never saw cannot bias a score computed over the ones they did.
    const JudgedQuery query = Judged({{"d1", 2}, {"d3", 0}});
    const QualityMetrics m = Evaluate(query, HitsFor({"d1", "u1", "d3", "u2", "u3"}), 3);

    CHECK_EQ(m.unjudged_in_top_k, 1u);
    // P@3 = 1/3: only d1 is relevant.
    CHECK_NEAR(m.precision_at_k, 1.0 / 3.0);
}

MERIDIAN_TEST(UnjudgedResultsDoNotContributeGain) {
    // Judgements: d1=1 only. Hits: u1 (unjudged), d1. k = 2.
    //   The unjudged result at rank 1 contributes nothing to DCG, pushing d1 to rank 2.
    //   DCG  = 0 + 1/log2(3) = 0.6309297536
    //   IDCG = 1/log2(2) = 1
    //   NDCG = 0.6309297536
    //   unjudged = 1
    // If u1 were in fact a good answer that nobody labelled, this score is an
    // understatement — and unjudged_in_top_k = 1 is how that possibility stays visible.
    const JudgedQuery query = Judged({{"d1", 1}});
    const QualityMetrics m = Evaluate(query, HitsFor({"u1", "d1"}), 2);

    CHECK_NEAR(m.ndcg_at_k, 1.0 / std::log2(3.0));
    CHECK_NEAR(m.ndcg_at_k, 0.6309297535714574);
    CHECK_EQ(m.unjudged_in_top_k, 1u);
    CHECK_NEAR(m.mrr, 0.5);
}

// ---------------------------------------------------------------------------
// Aggregate: macro-averaging
// ---------------------------------------------------------------------------

MERIDIAN_TEST(AggregateIsMacroAveraged) {
    // Query A: 100 judgements, all 100 relevant, and a perfect top-10. k = 10.
    //   P@10 = 1.0, R@10 = 10/100 = 0.1, NDCG = 1.0, MRR = 1.0
    // Query B: 2 judgements, neither returned. k = 10.
    //   P@10 = 0, R@10 = 0, NDCG = 0, MRR = 0
    //
    // Macro average: P = (1.0 + 0.0)/2 = 0.5, NDCG = 0.5, MRR = 0.5,
    //                R = (0.1 + 0.0)/2 = 0.05
    //
    // Micro-averaging would pool numerators and denominators: 10 relevant retrieved out
    // of 20 positions = P 0.5 here by coincidence, but recall would become 10/102 =
    // 0.098 — essentially query A's own number, with query B contributing 2% of the
    // weight because it has 2% of the judgements. The 50-fold difference in labelling
    // effort would become a 50-fold difference in influence.
    std::vector<std::pair<std::string, int>> hundred;
    std::vector<std::string> top_ten;
    for (int i = 1; i <= 100; ++i) {
        hundred.emplace_back("a" + std::to_string(i), 2);
        if (i <= 10) top_ten.push_back("a" + std::to_string(i));
    }

    const QualityMetrics dense = Evaluate(Judged(hundred), HitsFor(top_ten), 10);
    const QualityMetrics sparse =
        Evaluate(Judged({{"b1", 2}, {"b2", 2}}), HitsFor({"z1", "z2", "z3"}), 10);

    CHECK_NEAR(dense.precision_at_k, 1.0);
    CHECK_NEAR(dense.recall_at_k, 0.1);
    CHECK_NEAR(sparse.precision_at_k, 0.0);

    const QualityMetrics mean = Aggregate({dense, sparse});
    CHECK_EQ(mean.k, 10u);
    CHECK_NEAR(mean.precision_at_k, 0.5);
    CHECK_NEAR(mean.ndcg_at_k, 0.5);
    CHECK_NEAR(mean.mrr, 0.5);
    CHECK_NEAR(mean.recall_at_k, 0.05);

    // Order must not matter, and a float accumulation that divides before summing is the
    // usual reason it would.
    const QualityMetrics swapped = Aggregate({sparse, dense});
    CHECK_NEAR(swapped.ndcg_at_k, mean.ndcg_at_k);
}

MERIDIAN_TEST(AggregateSumsUnjudgedCounts) {
    // Unjudged is a count, not a rate, so the aggregate adds rather than averages:
    // 3 + 5 = 8 returned documents across the run had no judgement.
    QualityMetrics a;
    a.k = 10;
    a.unjudged_in_top_k = 3;
    QualityMetrics b;
    b.k = 10;
    b.unjudged_in_top_k = 5;

    const QualityMetrics mean = Aggregate({a, b});
    CHECK_EQ(mean.unjudged_in_top_k, 8u);
}

MERIDIAN_TEST(AggregateOfNothingReportsKZero) {
    // An aggregate over an empty set reports k = 0 rather than the struct's default of
    // 10, so "no queries were evaluated" cannot be read as "every query scored zero".
    const QualityMetrics mean = Aggregate({});
    CHECK_EQ(mean.k, 0u);
    CHECK_NEAR(mean.ndcg_at_k, 0.0);
}

// ---------------------------------------------------------------------------
// Compare: the function this header exists for
// ---------------------------------------------------------------------------

namespace {

EvalRun MakeRun(const std::string& label, double ndcg, uint64_t p99_ns) {
    EvalRun run;
    run.label = label;
    run.quality.k = 10;
    run.quality.ndcg_at_k = ndcg;
    run.latency.p99_ns = p99_ns;
    run.latency.p50_ns = p99_ns / 2;
    run.queries = 50;
    return run;
}

bool Contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

}  // namespace

MERIDIAN_TEST(CompareQualityUpLatencyUpIsATradeoffNotAWin) {
    // Baseline NDCG 0.6000, p99 1,000,000ns. Candidate NDCG 0.7000, p99 1,500,000ns.
    //   ndcg_delta = +0.1000
    //   p99_ratio  = 1,500,000 / 1,000,000 = 1.50
    //   1.50 > 1 + 0.10, so the latency regression is material.
    // This is the header's example in miniature: a real quality gain bought with a 50%
    // tail regression. It is a decision, and the verdict says so.
    const Comparison c = Compare(MakeRun("base", 0.60, 1'000'000), MakeRun("cand", 0.70, 1'500'000));

    CHECK(c.verdict == Verdict::kTradeoff);
    CHECK_EQ(std::string(VerdictName(c.verdict)), std::string("tradeoff"));
    CHECK_NEAR(c.ndcg_delta, 0.1);
    CHECK_NEAR(c.p99_ratio, 1.5);

    // Both numbers must be in the explanation, because the verdict alone cannot be acted
    // on without knowing how much quality and how much latency.
    CHECK(Contains(c.explanation, "+0.1000"));
    CHECK(Contains(c.explanation, "1.50x"));
    CHECK(Contains(c.explanation, "0.6000"));
    CHECK(Contains(c.explanation, "0.7000"));
    CHECK(Contains(c.explanation, "1000000ns"));
    CHECK(Contains(c.explanation, "1500000ns"));
}

MERIDIAN_TEST(CompareQualityUpLatencyFlatIsBetter) {
    // NDCG 0.60 -> 0.70, p99 1,000,000 -> 1,020,000ns.
    //   p99_ratio = 1.02, which is inside the 10% tolerance, so latency did not move.
    const Comparison c = Compare(MakeRun("base", 0.60, 1'000'000), MakeRun("cand", 0.70, 1'020'000));

    CHECK(c.verdict == Verdict::kBetter);
    CHECK_NEAR(c.p99_ratio, 1.02);
    CHECK(Contains(c.explanation, "+0.1000"));
    CHECK(Contains(c.explanation, "1.02x"));
}

MERIDIAN_TEST(CompareLatencyUpQualityFlatIsWorse) {
    // NDCG unchanged at 0.60, p99 1,000,000 -> 1,500,000ns. Paying 50% of the tail for
    // nothing.
    const Comparison c = Compare(MakeRun("base", 0.60, 1'000'000), MakeRun("cand", 0.60, 1'500'000));

    CHECK(c.verdict == Verdict::kWorse);
    CHECK_NEAR(c.ndcg_delta, 0.0);
    CHECK_NEAR(c.p99_ratio, 1.5);
    CHECK(Contains(c.explanation, "+0.0000"));
    CHECK(Contains(c.explanation, "1.50x"));
}

MERIDIAN_TEST(CompareWithinToleranceIsNoChange) {
    // NDCG 0.6000 -> 0.6005 (delta +0.0005, below the 0.001 noise floor) and
    // p99 1,000,000 -> 1,050,000ns (ratio 1.05, inside the 10% tolerance).
    // Neither axis moved beyond noise.
    const Comparison c = Compare(MakeRun("base", 0.60, 1'000'000), MakeRun("cand", 0.6005, 1'050'000));

    CHECK(c.verdict == Verdict::kNoChange);
    CHECK_NEAR(c.ndcg_delta, 0.0005);
    CHECK_NEAR(c.p99_ratio, 1.05);
    CHECK(Contains(c.explanation, "+0.0005"));
    CHECK(Contains(c.explanation, "1.05x"));
}

MERIDIAN_TEST(CompareQualityDownIsWorseEvenWhenFaster) {
    // NDCG 0.70 -> 0.60, p99 1,000,000 -> 500,000ns. Twice as fast and measurably worse
    // at answering the question. Reported as worse, with the speedup in the explanation
    // so the reader can disagree with the verdict on the evidence.
    const Comparison c = Compare(MakeRun("base", 0.70, 1'000'000), MakeRun("cand", 0.60, 500'000));

    CHECK(c.verdict == Verdict::kWorse);
    CHECK_NEAR(c.ndcg_delta, -0.1);
    CHECK_NEAR(c.p99_ratio, 0.5);
    CHECK(Contains(c.explanation, "-0.1000"));
    CHECK(Contains(c.explanation, "0.50x"));
}

MERIDIAN_TEST(CompareSameQualityLowerLatencyIsBetter) {
    // NDCG unchanged, p99 1,000,000 -> 700,000ns. The unambiguous win, and the only
    // shape of change that is one.
    const Comparison c = Compare(MakeRun("base", 0.60, 1'000'000), MakeRun("cand", 0.60, 700'000));

    CHECK(c.verdict == Verdict::kBetter);
    CHECK_NEAR(c.p99_ratio, 0.7);
    CHECK(Contains(c.explanation, "0.70x"));
}

MERIDIAN_TEST(CompareToleranceIsHonoured) {
    // p99 ratio 1.20 with a 50% tolerance is not a material regression, so a quality gain
    // is a plain improvement rather than a tradeoff. The same pair at the default 10%
    // tolerance is a tradeoff. The tolerance is the latency budget's voice in the
    // decision, and it has to actually change the answer.
    const EvalRun base = MakeRun("base", 0.60, 1'000'000);
    const EvalRun candidate = MakeRun("cand", 0.70, 1'200'000);

    CHECK(Compare(base, candidate, 0.50).verdict == Verdict::kBetter);
    CHECK(Compare(base, candidate, 0.10).verdict == Verdict::kTradeoff);
    CHECK(Contains(Compare(base, candidate, 0.50).explanation, "tolerance 50%"));
}

MERIDIAN_TEST(CompareWithoutLatencySamplesSaysSo) {
    // A run with no latency samples cannot be compared on latency. The ratio stays at 1.0
    // and the explanation refuses to imply latency was measured and found unchanged.
    const Comparison c = Compare(MakeRun("base", 0.60, 0), MakeRun("cand", 0.70, 0));

    CHECK_NEAR(c.p99_ratio, 1.0);
    CHECK(Contains(c.explanation, "p99 unavailable"));
    CHECK(c.verdict == Verdict::kBetter);
}

MERIDIAN_TEST(VerdictNamesAreStable) {
    CHECK_EQ(std::string(VerdictName(Verdict::kBetter)), std::string("better"));
    CHECK_EQ(std::string(VerdictName(Verdict::kWorse)), std::string("worse"));
    CHECK_EQ(std::string(VerdictName(Verdict::kTradeoff)), std::string("tradeoff"));
    CHECK_EQ(std::string(VerdictName(Verdict::kNoChange)), std::string("no-change"));
}

MERIDIAN_TEST(CompareReportsTheBaselineCutoffInTheExplanation) {
    // The NDCG delta is meaningless without the cutoff it was measured at: +0.04 at k=1
    // and +0.04 at k=100 are not the same claim.
    EvalRun base = MakeRun("base", 0.60, 1'000'000);
    base.quality.k = 20;
    EvalRun candidate = MakeRun("cand", 0.65, 1'000'000);
    candidate.quality.k = 20;

    CHECK(Contains(Compare(base, candidate).explanation, "NDCG@20"));
}

MERIDIAN_TEST_MAIN()
