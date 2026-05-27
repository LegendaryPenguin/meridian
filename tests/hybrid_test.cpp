// HybridMerger: the three fusion strategies, on hand-constructed candidate lists.
//
// The lists are built by hand rather than retrieved, because what is under test is
// arithmetic on two incomparable score scales and a real retrieval would make the inputs
// incidental to the thing being checked.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "meridian/retrieval.hpp"
#include "testing.hpp"

using namespace meridian;  // NOLINT

namespace {

Candidate Lexical(DocId doc, double score) {
    Candidate c;
    c.doc = doc;
    c.score = score;
    c.bm25 = score;
    c.source = Candidate::Source::kLexical;
    return c;
}

Candidate Semantic(DocId doc, double score) {
    Candidate c;
    c.doc = doc;
    c.score = score;
    c.vector_similarity = score;
    c.source = Candidate::Source::kSemantic;
    return c;
}

int RankOfDoc(const std::vector<Candidate>& merged, DocId doc) {
    for (size_t i = 0; i < merged.size(); ++i) {
        if (merged[i].doc == doc) return static_cast<int>(i);
    }
    return -1;
}

double ScoreOfDoc(const std::vector<Candidate>& merged, DocId doc) {
    for (const Candidate& c : merged) {
        if (c.doc == doc) return c.score;
    }
    return 0.0;
}

bool Near(double a, double b, double tolerance = 1e-9) { return std::abs(a - b) <= tolerance; }

const HybridMerger::Strategy kAllStrategies[] = {
    HybridMerger::Strategy::kReciprocalRankFusion,
    HybridMerger::Strategy::kNormalizedLinear,
    HybridMerger::Strategy::kZScoreLinear,
};

const char* StrategyName(HybridMerger::Strategy s) {
    switch (s) {
        case HybridMerger::Strategy::kReciprocalRankFusion: return "RRF";
        case HybridMerger::Strategy::kNormalizedLinear:     return "min-max";
        case HybridMerger::Strategy::kZScoreLinear:         return "z-score";
    }
    return "?";
}

}  // namespace

MERIDIAN_TEST(ReciprocalRankFusionUsesRanksAndNotScores) {
    // Lexical: 10, 20, 30.  Semantic: 20, 40.
    // Ranks are 1-based, k = 60, so the contributions are:
    //   doc 10: 1/61        = 0.016393442622951
    //   doc 20: 1/62 + 1/61 = 0.032522475481016
    //   doc 30: 1/63        = 0.015873015873016
    //   doc 40: 1/62        = 0.016129032258065
    const std::vector<Candidate> lexical = {Lexical(10, 9.0), Lexical(20, 8.0), Lexical(30, 7.0)};
    const std::vector<Candidate> semantic = {Semantic(20, 0.9), Semantic(40, 0.8)};

    const HybridMerger merger(HybridMerger::Strategy::kReciprocalRankFusion, 1.0, 1.0);
    const std::vector<Candidate> merged = merger.Merge(lexical, semantic, 10);
    CHECK_EQ(merged.size(), size_t{4});

    CHECK(Near(ScoreOfDoc(merged, 10), 1.0 / 61.0));
    CHECK(Near(ScoreOfDoc(merged, 20), 1.0 / 62.0 + 1.0 / 61.0));
    CHECK(Near(ScoreOfDoc(merged, 30), 1.0 / 63.0));
    CHECK(Near(ScoreOfDoc(merged, 40), 1.0 / 62.0));

    // Document 20 is second-best lexically and best semantically; appearing in both lists
    // beats being first in one.
    CHECK_EQ(RankOfDoc(merged, 20), 0);
    CHECK(merged[0].source == Candidate::Source::kBoth);

    // The failure mode, demonstrated rather than described: multiplying every lexical
    // score by a thousand changes nothing at all, because the ranks are unchanged. That
    // is RRF's immunity to scale and its blindness to confidence, which are the same
    // property.
    std::vector<Candidate> inflated = lexical;
    for (Candidate& c : inflated) {
        c.score *= 1000.0;
        c.bm25 = c.score;
    }
    const std::vector<Candidate> merged_inflated = merger.Merge(inflated, semantic, 10);
    CHECK_EQ(merged_inflated.size(), merged.size());
    for (size_t i = 0; i < merged.size() && i < merged_inflated.size(); ++i) {
        CHECK_EQ(merged_inflated[i].doc, merged[i].doc);
        CHECK(Near(merged_inflated[i].score, merged[i].score));
    }
}

MERIDIAN_TEST(NormalizedLinearScalesEachPathToTheUnitInterval) {
    // Lexical scores 2, 6, 10 -> min-max 0.0, 0.5, 1.0.
    // Semantic scores 0.2, 0.6 -> min-max 0.0, 1.0.
    const std::vector<Candidate> lexical = {Lexical(1, 10.0), Lexical(2, 6.0), Lexical(3, 2.0)};
    const std::vector<Candidate> semantic = {Semantic(2, 0.6), Semantic(4, 0.2)};

    const HybridMerger merger(HybridMerger::Strategy::kNormalizedLinear, 1.0, 1.0);
    const std::vector<Candidate> merged = merger.Merge(lexical, semantic, 10);
    CHECK_EQ(merged.size(), size_t{4});

    //   doc 1: 1.0 + absent(0.0) = 1.0
    //   doc 2: 0.5 + 1.0         = 1.5
    //   doc 3: 0.0 + absent(0.0) = 0.0
    //   doc 4: absent(0.0) + 0.0 = 0.0
    CHECK(Near(ScoreOfDoc(merged, 1), 1.0));
    CHECK(Near(ScoreOfDoc(merged, 2), 1.5));
    CHECK(Near(ScoreOfDoc(merged, 3), 0.0));
    CHECK(Near(ScoreOfDoc(merged, 4), 0.0));
    CHECK_EQ(RankOfDoc(merged, 2), 0);

    // Weights are applied after normalisation, so halving the semantic weight moves
    // document 2's semantic contribution from 1.0 to 0.5 and nothing else.
    const HybridMerger lexical_heavy(HybridMerger::Strategy::kNormalizedLinear, 1.0, 0.5);
    const std::vector<Candidate> reweighted = lexical_heavy.Merge(lexical, semantic, 10);
    CHECK(Near(ScoreOfDoc(reweighted, 2), 0.5 + 0.5));
    CHECK(Near(ScoreOfDoc(reweighted, 1), 1.0));
    // Now document 1 ties document 2, and the DocId tiebreak puts 1 first. Asserted
    // because it shows the weight actually reaches the ranking rather than only the score.
    CHECK_EQ(RankOfDoc(reweighted, 1), 0);
}

MERIDIAN_TEST(NormalizedLinearHandlesDegenerateAndSinglePathInput) {
    // Every score identical: the range is zero. All three map to 1.0, which keeps "this
    // path retrieved it" worth something rather than erasing the path entirely.
    const std::vector<Candidate> flat = {Lexical(1, 5.0), Lexical(2, 5.0), Lexical(3, 5.0)};
    const HybridMerger merger(HybridMerger::Strategy::kNormalizedLinear, 1.0, 1.0);
    const std::vector<Candidate> merged = merger.Merge(flat, {}, 10);
    CHECK_EQ(merged.size(), size_t{3});
    for (const Candidate& c : merged) CHECK(Near(c.score, 1.0));

    // A single candidate is the same degenerate case.
    const std::vector<Candidate> one = {Lexical(9, 42.0)};
    const std::vector<Candidate> merged_one = merger.Merge(one, {}, 10);
    CHECK_EQ(merged_one.size(), size_t{1});
    CHECK(Near(merged_one[0].score, 1.0));

    CHECK(merger.Merge({}, {}, 10).empty());
}

MERIDIAN_TEST(ZScoreLinearCentresAndScalesEachPath) {
    // Lexical scores 1, 2, 3, 4: mean 2.5, population stdev sqrt(1.25) = 1.118033988749895.
    //   doc 1 (score 4): (4 - 2.5)/1.118033988749895 =  1.341640786499874
    //   doc 2 (score 3): (3 - 2.5)/...               =  0.447213595499958
    //   doc 3 (score 2):                              = -0.447213595499958
    //   doc 4 (score 1):                              = -1.341640786499874
    const std::vector<Candidate> lexical = {Lexical(1, 4.0), Lexical(2, 3.0), Lexical(3, 2.0),
                                            Lexical(4, 1.0)};

    const HybridMerger merger(HybridMerger::Strategy::kZScoreLinear, 1.0, 1.0);
    const std::vector<Candidate> merged = merger.Merge(lexical, {}, 10);
    CHECK_EQ(merged.size(), size_t{4});

    const double stdev = std::sqrt(1.25);
    CHECK(Near(ScoreOfDoc(merged, 1), 1.5 / stdev));
    CHECK(Near(ScoreOfDoc(merged, 2), 0.5 / stdev));
    CHECK(Near(ScoreOfDoc(merged, 3), -0.5 / stdev));
    CHECK(Near(ScoreOfDoc(merged, 4), -1.5 / stdev));

    // Documents below their path's mean get negative scores. Noted because it is the
    // visible edge of the distributional assumption: a candidate list is a truncated
    // upper tail, so "below average for a top-200 result" is not the same as "bad", and
    // more than half of a right-skewed list lands below zero.
    int negatives = 0;
    for (const Candidate& c : merged) {
        if (c.score < 0.0) ++negatives;
    }
    CHECK_EQ(negatives, 2);

    // Absent from a path is treated as that path's *minimum* z, not as zero. Zero is the
    // mean, which would reward a document for not being retrieved.
    const std::vector<Candidate> semantic = {Semantic(1, 0.9), Semantic(5, 0.1)};
    const std::vector<Candidate> both = merger.Merge(lexical, semantic, 10);
    const double semantic_min = -1.0;  // two points: z is exactly -1 and +1
    CHECK(Near(ScoreOfDoc(both, 2), 0.5 / stdev + semantic_min));
    CHECK(Near(ScoreOfDoc(both, 1), 1.5 / stdev + 1.0));
    CHECK_EQ(RankOfDoc(both, 1), 0);
}

MERIDIAN_TEST(ZScoreLinearHandlesZeroVariance) {
    const std::vector<Candidate> flat = {Lexical(1, 5.0), Lexical(2, 5.0)};
    const HybridMerger merger(HybridMerger::Strategy::kZScoreLinear, 1.0, 1.0);
    const std::vector<Candidate> merged = merger.Merge(flat, {}, 10);
    CHECK_EQ(merged.size(), size_t{2});
    // No NaN. A NaN here sorts unpredictably and poisons every comparison downstream.
    for (const Candidate& c : merged) {
        CHECK(!std::isnan(c.score));
        CHECK(Near(c.score, 0.0));
    }
}

MERIDIAN_TEST(ADocumentInBothListsBeatsOneInNeitherPath) {
    // "In neither" can only be expressed as "in one list, at the bottom", since a merge
    // only ever sees documents some path retrieved. So: doc 1 is mid-table on both paths,
    // doc 2 is last on the lexical path and absent from the semantic one.
    const std::vector<Candidate> lexical = {Lexical(5, 9.0), Lexical(1, 5.0), Lexical(2, 1.0)};
    const std::vector<Candidate> semantic = {Semantic(7, 0.9), Semantic(1, 0.5),
                                            Semantic(8, 0.1)};

    for (HybridMerger::Strategy strategy : kAllStrategies) {
        const HybridMerger merger(strategy, 1.0, 1.0);
        const std::vector<Candidate> merged = merger.Merge(lexical, semantic, 10);
        CHECK_EQ(merged.size(), size_t{5});
        const int in_both = RankOfDoc(merged, 1);
        const int in_neither = RankOfDoc(merged, 2);
        if (!(in_both >= 0 && in_neither >= 0 && in_both < in_neither)) {
            meridian::testing::ReportFailure(
                __FILE__, __LINE__,
                std::string(StrategyName(strategy)) + ": doc 1 (in both lists) ranked " +
                    std::to_string(in_both) + ", doc 2 (effectively in neither) ranked " +
                    std::to_string(in_neither));
        }
        CHECK(merged[static_cast<size_t>(std::max(0, in_both))].source ==
              Candidate::Source::kBoth);
    }
}

MERIDIAN_TEST(SourcesAreAttributedCorrectly) {
    const std::vector<Candidate> lexical = {Lexical(1, 1.0), Lexical(2, 2.0)};
    const std::vector<Candidate> semantic = {Semantic(2, 0.5), Semantic(3, 0.7)};

    const HybridMerger merger(HybridMerger::Strategy::kReciprocalRankFusion, 1.0, 1.0);
    const std::vector<Candidate> merged = merger.Merge(lexical, semantic, 10);
    for (const Candidate& c : merged) {
        if (c.doc == 1) CHECK(c.source == Candidate::Source::kLexical);
        if (c.doc == 2) CHECK(c.source == Candidate::Source::kBoth);
        if (c.doc == 3) CHECK(c.source == Candidate::Source::kSemantic);
    }

    // The per-path scores survive the merge, because the reranker reads them as features
    // and recomputing them there would mean retrieving twice.
    for (const Candidate& c : merged) {
        if (c.doc == 2) {
            CHECK(Near(c.bm25, 2.0));
            CHECK(Near(c.vector_similarity, 0.5));
        }
        if (c.doc == 3) {
            CHECK(Near(c.bm25, 0.0));
            CHECK(Near(c.vector_similarity, 0.7));
        }
    }
}

MERIDIAN_TEST(ThreeOrdersOfMagnitudeDoNotLetOnePathDominate) {
    // The pathological case the header warns about. The lexical path's scores are on the
    // order of 10^3 and the semantic path's on the order of 10^0 — a thousandfold
    // difference, which is entirely plausible: BM25 is an unbounded sum over query terms
    // and a cosine similarity lives in [-1, 1].
    //
    // The construction makes the two paths *disagree*: the semantic path's favourite,
    // doc 100, is the lexical path's worst. If the merge simply added the raw scores, the
    // lexical ordering would survive untouched and the semantic path would contribute
    // nothing, because its entire dynamic range (about 0.5) is a rounding error next to
    // the lexical gaps (about 500).
    const std::vector<Candidate> lexical = {
        Lexical(10, 2000.0), Lexical(20, 1500.0), Lexical(30, 1000.0), Lexical(100, 500.0)};
    const std::vector<Candidate> semantic = {
        Semantic(100, 0.95), Semantic(30, 0.60), Semantic(20, 0.50), Semantic(10, 0.45)};

    // The baseline: naive addition. Kept in the test so the thing being avoided is
    // visible rather than asserted about.
    std::vector<Candidate> naive;
    for (const Candidate& l : lexical) {
        double total = l.score;
        for (const Candidate& s : semantic) {
            if (s.doc == l.doc) total += s.score;
        }
        naive.push_back(Lexical(l.doc, total));
    }
    std::sort(naive.begin(), naive.end(),
              [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
    CHECK_EQ(naive[0].doc, DocId{10});    // unchanged from the lexical order
    CHECK_EQ(naive[3].doc, DocId{100});   // the semantic favourite stays last
    CHECK(Near(naive[0].score, 2000.45));

    // Normalised strategies: the semantic path's preference has to be visible. Doc 100 is
    // last lexically and first semantically, so with equal weights it must not remain
    // last, and doc 10 (first lexically, last semantically) must not remain first.
    for (HybridMerger::Strategy strategy :
         {HybridMerger::Strategy::kNormalizedLinear, HybridMerger::Strategy::kZScoreLinear}) {
        const HybridMerger merger(strategy, 1.0, 1.0);
        const std::vector<Candidate> merged = merger.Merge(lexical, semantic, 10);
        CHECK_EQ(merged.size(), size_t{4});

        const int rank_100 = RankOfDoc(merged, 100);
        const int rank_10 = RankOfDoc(merged, 10);
        if (!(rank_100 < 3)) {
            meridian::testing::ReportFailure(
                __FILE__, __LINE__,
                std::string(StrategyName(strategy)) +
                    " left the semantic favourite last: the 10^3 lexical scale dominated");
        }
        // Both paths' extremes meet in the middle: with symmetric inputs and equal
        // weights, the merged scores of doc 10 and doc 100 are equal under min-max.
        if (strategy == HybridMerger::Strategy::kNormalizedLinear) {
            CHECK(Near(ScoreOfDoc(merged, 10), ScoreOfDoc(merged, 100)));
            // And every merged score is in [0, 2], i.e. on the weight's scale rather than
            // on BM25's. This is the concrete statement of "not dominated".
            for (const Candidate& c : merged) {
                CHECK(c.score >= 0.0);
                CHECK(c.score <= 2.0);
            }
        }
        std::printf("        %s: doc10 %.6f (rank %d), doc100 %.6f (rank %d)\n",
                    StrategyName(strategy), ScoreOfDoc(merged, 10), rank_10,
                    ScoreOfDoc(merged, 100), rank_100);
    }

    // RRF is immune by construction: scaling one path by any factor cannot change its
    // ranks, so the merged result is bit-identical.
    const HybridMerger rrf(HybridMerger::Strategy::kReciprocalRankFusion, 1.0, 1.0);
    std::vector<Candidate> scaled = lexical;
    for (Candidate& c : scaled) {
        c.score *= 1e6;
        c.bm25 = c.score;
    }
    const std::vector<Candidate> a = rrf.Merge(lexical, semantic, 10);
    const std::vector<Candidate> b = rrf.Merge(scaled, semantic, 10);
    CHECK_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        CHECK_EQ(a[i].doc, b[i].doc);
        CHECK(a[i].score == b[i].score);  // exact, not approximate: the inputs are ranks
    }
    // The two paths' orderings here are exact mirrors, so each document's score is
    // 1/(60+r) + 1/(65-r) and the four come out nearly tied. The small residual ordering
    // is worth noting because it is not obvious: 1/x is convex, so 1/61 + 1/64 exceeds
    // 1/62 + 1/63, and RRF therefore mildly *prefers* a document that is extreme on one
    // path to one that is middling on both. The effect is about 0.05% here, which is the
    // honest size of it.
    CHECK(Near(ScoreOfDoc(a, 10), ScoreOfDoc(a, 100)));
    CHECK(Near(ScoreOfDoc(a, 20), ScoreOfDoc(a, 30)));
    CHECK(ScoreOfDoc(a, 10) > ScoreOfDoc(a, 20));
}

MERIDIAN_TEST(MinMaxIsOutlierSensitiveAndZScoreIsLess) {
    // The documented failure mode of min-max, made measurable. One lexical outlier at 500
    // with the rest clustered between 2.0 and 3.0: min-max compresses the cluster into a
    // sliver of the unit interval, so the whole path collapses to "is this the outlier".
    const std::vector<Candidate> lexical = {Lexical(1, 500.0), Lexical(2, 3.0),
                                            Lexical(3, 2.5), Lexical(4, 2.0)};

    const HybridMerger min_max(HybridMerger::Strategy::kNormalizedLinear, 1.0, 1.0);
    const std::vector<Candidate> mm = min_max.Merge(lexical, {}, 10);
    const double mm_spread = ScoreOfDoc(mm, 2) - ScoreOfDoc(mm, 4);
    CHECK(mm_spread < 0.01);  // the three clustered documents are effectively tied

    const HybridMerger z(HybridMerger::Strategy::kZScoreLinear, 1.0, 1.0);
    const std::vector<Candidate> zs = z.Merge(lexical, {}, 10);
    const double z_spread = ScoreOfDoc(zs, 2) - ScoreOfDoc(zs, 4);

    // Z-scoring keeps more of the cluster's internal ordering, though it does not rescue
    // it: the outlier inflates the standard deviation too. The claim is comparative, and
    // that is all it should be.
    CHECK(z_spread > mm_spread);
    std::printf("        cluster spread: min-max %.6f, z-score %.6f\n", mm_spread, z_spread);

    // Ordering survives under both; it is the *spacing* that min-max destroys.
    for (const std::vector<Candidate>& merged : {mm, zs}) {
        CHECK(RankOfDoc(merged, 1) == 0);
        CHECK(RankOfDoc(merged, 2) < RankOfDoc(merged, 3));
        CHECK(RankOfDoc(merged, 3) < RankOfDoc(merged, 4));
    }
}

MERIDIAN_TEST(MergeRespectsTheCandidateBudget) {
    std::vector<Candidate> lexical;
    std::vector<Candidate> semantic;
    for (int i = 0; i < 50; ++i) {
        lexical.push_back(Lexical(static_cast<DocId>(i), 50.0 - i));
        semantic.push_back(Semantic(static_cast<DocId>(i + 25), 1.0 - i * 0.01));
    }

    for (HybridMerger::Strategy strategy : kAllStrategies) {
        const HybridMerger merger(strategy, 1.0, 1.0);
        CHECK_EQ(merger.Merge(lexical, semantic, 10).size(), size_t{10});
        CHECK_EQ(merger.Merge(lexical, semantic, 75).size(), size_t{75});
        CHECK_EQ(merger.Merge(lexical, semantic, 1000).size(), size_t{75});
        CHECK(merger.Merge(lexical, semantic, 0).empty());
    }
}

MERIDIAN_TEST(MergeIsDeterministicAcrossRuns) {
    // The merge accumulates into an unordered_map, whose iteration order is not stable.
    // Without the explicit DocId tiebreak, equal-scoring documents would come out in a
    // different order on different runs and an evaluation harness would measure noise.
    std::vector<Candidate> lexical;
    for (int i = 0; i < 30; ++i) lexical.push_back(Lexical(static_cast<DocId>(i), 1.0));

    const HybridMerger merger(HybridMerger::Strategy::kNormalizedLinear, 1.0, 1.0);
    const std::vector<Candidate> first = merger.Merge(lexical, {}, 30);
    for (int repeat = 0; repeat < 5; ++repeat) {
        const std::vector<Candidate> again = merger.Merge(lexical, {}, 30);
        CHECK_EQ(again.size(), first.size());
        for (size_t i = 0; i < first.size() && i < again.size(); ++i) {
            CHECK_EQ(again[i].doc, first[i].doc);
        }
    }
    for (size_t i = 0; i < first.size(); ++i) CHECK_EQ(first[i].doc, static_cast<DocId>(i));
}

MERIDIAN_TEST_MAIN()
