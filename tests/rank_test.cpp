// Second-stage ranking: each feature in isolation, the feature map, weight sensitivity,
// and one end-to-end retrieval over a hand-built index.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "../src/query/parse.hpp"
#include "../src/rank/reranker.hpp"
#include "index_fixture.hpp"
#include "testing.hpp"

using namespace meridian;  // NOLINT
using meridian::rank::FeatureConfig;
using meridian::rank::FeatureExtractor;
using meridian::rank::LinearReranker;
using meridian::rank::LinearWeights;
using meridian::testfixture::CorpusBuilder;
using meridian::testfixture::Doc;
using meridian::testfixture::Near;
using meridian::testfixture::RankOf;
namespace features = meridian::rank::features;

namespace {

/// A fixed "now", so freshness is deterministic. 2024-01-01T00:00:00Z.
constexpr int64_t kNow = 1704067200;
constexpr int64_t kDay = 86400;

FeatureConfig FixedClock() {
    FeatureConfig config;
    config.now_epoch_seconds = kNow;
    return config;
}

Candidate Candidate_(DocId doc, double bm25 = 0.0, double similarity = 0.0) {
    Candidate c;
    c.doc = doc;
    c.score = bm25;
    c.bm25 = bm25;
    c.vector_similarity = similarity;
    return c;
}

double FeatureOf(const FeatureExtractor::Row& row, const char* name) {
    auto it = row.features.find(name);
    return it == row.features.end() ? std::nan("") : it->second;
}

double FeatureOf(const Hit& hit, const char* name) {
    auto it = hit.features.find(name);
    return it == hit.features.end() ? std::nan("") : it->second;
}

/// All weights zero except one, so a feature can be observed through the ranker without
/// the others interfering.
LinearWeights OnlyWeight(const char* name, double weight) {
    LinearWeights w;
    for (const std::string& feature : FeatureExtractor::Names()) w.weights[feature] = 0.0;
    w.weights[name] = weight;
    return w;
}

}  // namespace

MERIDIAN_TEST(TextFeaturesAreComputedPerField) {
    CorpusBuilder builder;
    // "quartz" in the title only, "basalt" in the body only, "granite" in the tags only,
    // so each field feature can be isolated.
    builder.Add({"title_hit", "quartz report", "unrelated prose here", "mineral", "en",
                 "geology", kNow, 0.0, false});
    builder.Add({"body_hit", "unrelated heading", "a study of basalt flows", "mineral", "en",
                 "geology", kNow, 0.0, false});
    builder.Add({"tags_hit", "unrelated heading", "unrelated prose", "granite", "en",
                 "geology", kNow, 0.0, false});
    std::shared_ptr<Generation> gen = builder.Build();

    const FeatureExtractor extractor(FixedClock());

    Result<Query> title_query = query::Parse("quartz");
    CHECK(title_query.ok());
    if (!title_query.ok()) return;
    Result<std::vector<FeatureExtractor::Row>> rows =
        extractor.Extract(*gen, title_query.value(), {Candidate_(0), Candidate_(1)});
    CHECK(rows.ok());
    if (!rows.ok()) return;
    CHECK_EQ(rows.value().size(), size_t{2});
    CHECK(Near(FeatureOf(rows.value()[0], features::kTitleMatch), 1.0));
    CHECK(Near(FeatureOf(rows.value()[0], features::kBodyMatch), 0.0));
    CHECK(Near(FeatureOf(rows.value()[0], features::kTagsMatch), 0.0));
    // The second candidate does not contain the term at all; every match feature is zero
    // rather than missing, so the map is comparable across hits.
    CHECK(Near(FeatureOf(rows.value()[1], features::kTitleMatch), 0.0));
    CHECK(Near(FeatureOf(rows.value()[1], features::kBodyMatch), 0.0));

    Result<Query> body_query = query::Parse("basalt");
    CHECK(body_query.ok());
    Result<std::vector<FeatureExtractor::Row>> body_rows =
        extractor.Extract(*gen, body_query.value(), {Candidate_(1)});
    CHECK(body_rows.ok());
    if (!body_rows.ok()) return;
    CHECK(Near(FeatureOf(body_rows.value()[0], features::kBodyMatch), 1.0));
    CHECK(Near(FeatureOf(body_rows.value()[0], features::kTitleMatch), 0.0));

    Result<Query> tags_query = query::Parse("granite");
    CHECK(tags_query.ok());
    Result<std::vector<FeatureExtractor::Row>> tags_rows =
        extractor.Extract(*gen, tags_query.value(), {Candidate_(2)});
    CHECK(tags_rows.ok());
    if (!tags_rows.ok()) return;
    CHECK(Near(FeatureOf(tags_rows.value()[0], features::kTagsMatch), 1.0));
}

MERIDIAN_TEST(MatchFeaturesAreFractionsOfQueryCoverage) {
    CorpusBuilder builder;
    builder.Add({"both", "copper and zinc", "", "", "en", "general", kNow, 0.0, false});
    builder.Add({"one", "copper only", "", "", "en", "general", kNow, 0.0, false});
    std::shared_ptr<Generation> gen = builder.Build();

    const FeatureExtractor extractor(FixedClock());
    Result<Query> parsed = query::Parse("copper zinc");
    CHECK(parsed.ok());
    if (!parsed.ok()) return;
    CHECK_EQ(parsed.value().terms.size(), size_t{2});

    Result<std::vector<FeatureExtractor::Row>> rows =
        extractor.Extract(*gen, parsed.value(), {Candidate_(0), Candidate_(1)});
    CHECK(rows.ok());
    if (!rows.ok()) return;
    // Two of two terms in the title, versus one of two. Expressed as a fraction so the
    // feature means the same thing for a one-term query and a five-term one.
    CHECK(Near(FeatureOf(rows.value()[0], features::kTitleMatch), 1.0));
    CHECK(Near(FeatureOf(rows.value()[1], features::kTitleMatch), 0.5));
}

MERIDIAN_TEST(PhraseFeatureRequiresAdjacency) {
    CorpusBuilder builder;
    builder.Add({"adjacent", "", "a study of machine learning methods", "", "en", "general",
                 kNow, 0.0, false});
    builder.Add({"separated", "", "machine tooling and statistical learning", "", "en",
                 "general", kNow, 0.0, false});
    std::shared_ptr<Generation> gen = builder.Build();

    const FeatureExtractor extractor(FixedClock());
    // Deliberately a disjunctive query: the phrase *feature* is computed for any query
    // whose terms happen to be adjacent, which is the point — it is evidence the
    // retriever did not demand.
    Result<Query> parsed = query::Parse("machine learning");
    CHECK(parsed.ok());
    if (!parsed.ok()) return;

    Result<std::vector<FeatureExtractor::Row>> rows =
        extractor.Extract(*gen, parsed.value(), {Candidate_(0), Candidate_(1)});
    CHECK(rows.ok());
    if (!rows.ok()) return;
    CHECK(Near(FeatureOf(rows.value()[0], features::kPhraseMatch), 1.0));
    CHECK(Near(FeatureOf(rows.value()[1], features::kPhraseMatch), 0.0));
    // Both documents contain both terms, so the phrase feature is the only thing that
    // separates them.
    CHECK(Near(FeatureOf(rows.value()[0], features::kBodyMatch),
               FeatureOf(rows.value()[1], features::kBodyMatch)));
}

MERIDIAN_TEST(FreshnessHalvesEveryHalfLife) {
    CorpusBuilder builder;
    builder.Add({"today", "", "topic", "", "en", "general", kNow, 0.0, false});
    builder.Add({"30d", "", "topic", "", "en", "general", kNow - 30 * kDay, 0.0, false});
    builder.Add({"60d", "", "topic", "", "en", "general", kNow - 60 * kDay, 0.0, false});
    builder.Add({"90d", "", "topic", "", "en", "general", kNow - 90 * kDay, 0.0, false});
    builder.Add({"future", "", "topic", "", "en", "general", kNow + 10 * kDay, 0.0, false});
    std::shared_ptr<Generation> gen = builder.Build();

    const FeatureExtractor extractor(FixedClock());  // half-life 30 days
    Result<Query> parsed = query::Parse("topic");
    CHECK(parsed.ok());
    if (!parsed.ok()) return;

    Result<std::vector<FeatureExtractor::Row>> rows = extractor.Extract(
        *gen, parsed.value(),
        {Candidate_(0), Candidate_(1), Candidate_(2), Candidate_(3), Candidate_(4)});
    CHECK(rows.ok());
    if (!rows.ok()) return;
    CHECK_EQ(rows.value().size(), size_t{5});

    CHECK(Near(FeatureOf(rows.value()[0], features::kFreshness), 1.0));
    CHECK(Near(FeatureOf(rows.value()[1], features::kFreshness), 0.5));
    CHECK(Near(FeatureOf(rows.value()[2], features::kFreshness), 0.25));
    CHECK(Near(FeatureOf(rows.value()[3], features::kFreshness), 0.125));
    // A future timestamp is clamped, not extrapolated: otherwise a wrong publication date
    // buys unbounded rank.
    CHECK(Near(FeatureOf(rows.value()[4], features::kFreshness), 1.0));

    // A longer half-life decays more slowly at the same age. Asserted because an
    // exponential with the sign or the reciprocal wrong is still monotone in age.
    FeatureConfig slow = FixedClock();
    slow.freshness_half_life_days = 365.0;
    Result<std::vector<FeatureExtractor::Row>> slow_rows =
        FeatureExtractor(slow).Extract(*gen, parsed.value(), {Candidate_(3)});
    CHECK(slow_rows.ok());
    if (!slow_rows.ok()) return;
    CHECK(FeatureOf(slow_rows.value()[0], features::kFreshness) > 0.8);
}

MERIDIAN_TEST(PopularityIsCompressedAndBounded) {
    CorpusBuilder builder;
    builder.Add({"unknown", "", "topic", "", "en", "general", kNow, 0.0, false});
    builder.Add({"modest", "", "topic", "", "en", "general", kNow, 100.0, false});
    builder.Add({"viral", "", "topic", "", "en", "general", kNow, 100000000.0, false});
    std::shared_ptr<Generation> gen = builder.Build();

    const FeatureExtractor extractor(FixedClock());
    Result<Query> parsed = query::Parse("topic");
    CHECK(parsed.ok());
    if (!parsed.ok()) return;
    Result<std::vector<FeatureExtractor::Row>> rows =
        extractor.Extract(*gen, parsed.value(), {Candidate_(0), Candidate_(1), Candidate_(2)});
    CHECK(rows.ok());
    if (!rows.ok()) return;

    const double unknown = FeatureOf(rows.value()[0], features::kPopularity);
    const double modest = FeatureOf(rows.value()[1], features::kPopularity);
    const double viral = FeatureOf(rows.value()[2], features::kPopularity);

    CHECK(Near(unknown, 0.0));
    CHECK(modest > unknown);
    CHECK(viral > modest);
    CHECK(viral < 1.0);  // bounded, so it cannot swamp a weighted sum

    // A millionfold difference in raw popularity becomes less than a 2x difference in the
    // feature. That compression is the whole point: without it the single most popular
    // document outranks everything regardless of text.
    CHECK((viral / modest) < 2.0);
    std::printf("        popularity: 0 -> %.4f, 100 -> %.4f, 1e8 -> %.4f\n", unknown, modest,
                viral);
}

MERIDIAN_TEST(RetrievalScoresArePassedThroughAsFeatures) {
    CorpusBuilder builder;
    builder.Add({"d0", "", "topic", "", "en", "general", kNow, 0.0, false});
    std::shared_ptr<Generation> gen = builder.Build();

    const FeatureExtractor extractor(FixedClock());
    Result<Query> parsed = query::Parse("topic");
    CHECK(parsed.ok());
    if (!parsed.ok()) return;
    Result<std::vector<FeatureExtractor::Row>> rows =
        extractor.Extract(*gen, parsed.value(), {Candidate_(0, 7.25, 0.875)});
    CHECK(rows.ok());
    if (!rows.ok()) return;
    CHECK(Near(FeatureOf(rows.value()[0], features::kBm25), 7.25));
    CHECK(Near(FeatureOf(rows.value()[0], features::kVectorSimilarity), 0.875));
    CHECK_EQ(rows.value()[0].external_id, std::string("d0"));
}

MERIDIAN_TEST(FeatureMapIsFullyPopulatedAndExplainsTheScore) {
    CorpusBuilder builder;
    builder.Add({"d0", "machine learning", "machine learning in practice", "ml", "en",
                 "general", kNow - 15 * kDay, 500.0, false});
    std::shared_ptr<Generation> gen = builder.Build();

    Result<Query> parsed = query::Parse("machine learning");
    CHECK(parsed.ok());
    if (!parsed.ok()) return;

    const LinearReranker reranker(LinearWeights::Default(), FixedClock());
    Result<std::vector<Hit>> hits = reranker.Rank(*gen, parsed.value(), {Candidate_(0, 2.0, 0.5)});
    CHECK(hits.ok());
    if (!hits.ok()) return;
    CHECK_EQ(hits.value().size(), size_t{1});
    const Hit& hit = hits.value()[0];
    CHECK_EQ(hit.external_id, std::string("d0"));

    // Every declared feature is present. A feature the ranker uses but does not report is
    // a ranking decision nothing downstream can account for.
    double reconstructed = 0.0;
    for (const std::string& name : reranker.FeatureNames()) {
        const bool present = hit.features.count(name) != 0;
        if (!present) {
            meridian::testing::ReportFailure(__FILE__, __LINE__,
                                            "feature \"" + name + "\" is missing from the map");
            continue;
        }
        const std::string contribution_key = std::string(meridian::rank::kContributionPrefix) + name;
        CHECK(hit.features.count(contribution_key) != 0);
        const double value = hit.features.at(name);
        const double contribution = hit.features.at(contribution_key);
        CHECK(Near(contribution, reranker.weights().Get(name) * value));
        reconstructed += contribution;
    }

    // The score is exactly the sum of the reported contributions, which is what makes the
    // map an explanation rather than a decoration.
    CHECK(Near(reconstructed, hit.score, 1e-12));
    CHECK(hit.score > 0.0);
}

MERIDIAN_TEST(ChangingAWeightMovesTheExpectedDocument) {
    // Two documents matching the query equally well in text, differing only in age and
    // popularity. Which one wins is therefore entirely a function of the weights, which is
    // the cleanest possible demonstration that the weights reach the ranking.
    CorpusBuilder builder;
    builder.Add({"old_but_popular", "sensor calibration", "sensor calibration guide", "", "en",
                 "general", kNow - 365 * kDay, 1000000.0, false});
    builder.Add({"fresh_but_obscure", "sensor calibration", "sensor calibration guide", "", "en",
                 "general", kNow, 0.0, false});
    std::shared_ptr<Generation> gen = builder.Build();

    Result<Query> parsed = query::Parse("sensor calibration");
    CHECK(parsed.ok());
    if (!parsed.ok()) return;
    const std::vector<Candidate> candidates = {Candidate_(0, 1.0, 0.0), Candidate_(1, 1.0, 0.0)};

    const LinearReranker freshness_matters(OnlyWeight(features::kFreshness, 1.0), FixedClock());
    Result<std::vector<Hit>> fresh_first = freshness_matters.Rank(*gen, parsed.value(), candidates);
    CHECK(fresh_first.ok());
    if (!fresh_first.ok()) return;
    CHECK_EQ(RankOf(fresh_first.value(), "fresh_but_obscure"), 0);
    CHECK_EQ(RankOf(fresh_first.value(), "old_but_popular"), 1);

    const LinearReranker popularity_matters(OnlyWeight(features::kPopularity, 1.0), FixedClock());
    Result<std::vector<Hit>> popular_first =
        popularity_matters.Rank(*gen, parsed.value(), candidates);
    CHECK(popular_first.ok());
    if (!popular_first.ok()) return;
    CHECK_EQ(RankOf(popular_first.value(), "old_but_popular"), 0);
    CHECK_EQ(RankOf(popular_first.value(), "fresh_but_obscure"), 1);

    // And the crossover is findable: with popularity fixed at weight 1, there is a
    // freshness weight at which the fresh document takes the lead. Searched rather than
    // asserted at a magic number, because the point is that the trade-off is continuous
    // and interpretable, not that it sits at any particular value.
    bool crossed = false;
    for (double freshness_weight = 0.0; freshness_weight <= 4.0; freshness_weight += 0.05) {
        LinearWeights weights = OnlyWeight(features::kPopularity, 1.0);
        weights.weights[features::kFreshness] = freshness_weight;
        Result<std::vector<Hit>> ranked =
            LinearReranker(weights, FixedClock()).Rank(*gen, parsed.value(), candidates);
        if (!ranked.ok()) continue;
        if (RankOf(ranked.value(), "fresh_but_obscure") == 0) {
            std::printf("        freshness outweighs popularity at weight %.2f\n",
                        freshness_weight);
            crossed = true;
            break;
        }
    }
    CHECK(crossed);
}

MERIDIAN_TEST(RerankerHandlesMultipleSegmentsAndGlobalDocIds) {
    CorpusBuilder builder;
    builder.Add({"s0d0", "", "alloy", "", "en", "general", kNow, 0.0, false});
    builder.Add({"s0d1", "", "filler", "", "en", "general", kNow, 0.0, false});
    builder.NewSegment();
    builder.Add({"s1d0", "alloy handbook", "alloy", "", "en", "general", kNow, 0.0, false});
    std::shared_ptr<Generation> gen = builder.Build();

    Result<Query> parsed = query::Parse("alloy");
    CHECK(parsed.ok());
    if (!parsed.ok()) return;

    const LinearReranker reranker(LinearWeights::Default(), FixedClock());
    // Candidates deliberately out of document order, so the monotone-advance assumption in
    // the extractor is exercised rather than accidentally satisfied.
    Result<std::vector<Hit>> hits =
        reranker.Rank(*gen, parsed.value(), {Candidate_(2, 1.0), Candidate_(0, 1.0)});
    CHECK(hits.ok());
    if (!hits.ok()) return;
    CHECK_EQ(hits.value().size(), size_t{2});

    // The segment-1 document is the one with a title match, and global DocId 2 must have
    // resolved to segment 1 local 0 for that to be visible.
    CHECK_EQ(RankOf(hits.value(), "s1d0"), 0);
    for (const Hit& hit : hits.value()) {
        if (hit.external_id == "s1d0") CHECK(Near(FeatureOf(hit, features::kTitleMatch), 1.0));
        if (hit.external_id == "s0d0") CHECK(Near(FeatureOf(hit, features::kTitleMatch), 0.0));
    }

    // A candidate from no segment at all (a DocId from another generation) is dropped
    // rather than ranked on zeros.
    Result<std::vector<Hit>> with_stranger =
        reranker.Rank(*gen, parsed.value(), {Candidate_(0, 1.0), Candidate_(9999, 1.0)});
    CHECK(with_stranger.ok());
    if (with_stranger.ok()) CHECK_EQ(with_stranger.value().size(), size_t{1});
}

MERIDIAN_TEST(RerankerOnAnEmptyCandidateSetIsEmpty) {
    CorpusBuilder builder;
    builder.Add({"d0", "", "anything", "", "en", "general", kNow, 0.0, false});
    std::shared_ptr<Generation> gen = builder.Build();
    Result<Query> parsed = query::Parse("anything");
    CHECK(parsed.ok());
    if (!parsed.ok()) return;

    const LinearReranker reranker(LinearWeights::Default(), FixedClock());
    Result<std::vector<Hit>> hits = reranker.Rank(*gen, parsed.value(), {});
    CHECK(hits.ok());
    if (hits.ok()) CHECK(hits.value().empty());
}

// ===========================================================================
// End to end
// ===========================================================================

MERIDIAN_TEST(EndToEndRetrieveMergeRerankOverAKnownCorpus) {
    // A corpus small enough to reason about completely, and a query whose correct answer
    // is decidable by reading it.
    //
    // Query: "machine learning", disjunctive, with a language filter.
    //
    //   d0 "ml_intro"     title and body both about machine learning, 10 days old
    //   d1 "ml_history"   same topic, 3 years old, unpopular
    //   d2 "machine_shop" "machine" only, and in a different sense
    //   d3 "fr_ml"        exactly on topic but in French, so the filter excludes it
    //   d4 "gardening"    matches nothing
    //
    // The correct answer: d0 first (on topic, phrase match in the title, fresh), d1 second
    // (on topic, phrase match, stale), d2 third (half the query, no phrase), d3 and d4 not
    // returned at all — d3 because of the filter and d4 because it matches no term.
    CorpusBuilder builder;
    builder.Add({"ml_intro", "machine learning basics",
                 "an introduction to machine learning for engineers", "tutorial", "en",
                 "tech", kNow - 10 * kDay, 5000.0, false});
    builder.Add({"ml_history", "machine learning history",
                 "the history of machine learning from perceptrons onward", "essay", "en",
                 "tech", kNow - 1095 * kDay, 10.0, false});
    builder.Add({"machine_shop", "machine shop safety",
                 "operating a lathe in the machine shop requires care", "safety", "en",
                 "tech", kNow - 20 * kDay, 50.0, false});
    builder.NewSegment();
    builder.Add({"fr_ml", "apprentissage machine",
                 "une introduction au machine learning", "tutoriel", "fr", "tech",
                 kNow - 5 * kDay, 7000.0, false});
    builder.Add({"gardening", "tomato growing", "how to grow tomatoes in a small garden",
                 "garden", "en", "lifestyle", kNow - 1 * kDay, 9000.0, false});
    std::shared_ptr<Generation> gen = builder.Build();
    CHECK_EQ(gen->segments().size(), size_t{2});
    CHECK_EQ(gen->stats().document_count, uint32_t{5});

    Result<Query> parsed = query::Parse("machine learning language:en");
    CHECK(parsed.ok());
    if (!parsed.ok()) return;
    CHECK_EQ(parsed.value().terms.size(), size_t{2});
    CHECK(parsed.value().filters.language.has_value());

    const LexicalRetriever retriever({}, FieldWeights{});
    Result<std::vector<Candidate>> lexical = retriever.Retrieve(*gen, parsed.value());
    CHECK(lexical.ok());
    if (!lexical.ok()) return;

    // Retrieval stage: three candidates, the French document excluded by the filter and
    // the gardening document excluded by having no query term.
    CHECK_EQ(lexical.value().size(), size_t{3});
    CHECK_EQ(retriever.stats().segments_searched, uint64_t{2});
    CHECK_EQ(retriever.stats().candidates_scored, uint64_t{3});

    // A semantic path, hand-built to disagree slightly: it ranks the history document top.
    // Included so the merge is actually doing something.
    std::vector<Candidate> semantic;
    for (const Candidate& c : lexical.value()) {
        Candidate s;
        s.doc = c.doc;
        s.source = Candidate::Source::kSemantic;
        s.score = c.doc == 1 ? 0.91 : (c.doc == 0 ? 0.88 : 0.40);
        s.vector_similarity = s.score;
        semantic.push_back(s);
    }

    const HybridMerger merger(HybridMerger::Strategy::kNormalizedLinear, 1.0, 1.0);
    std::vector<Candidate> merged =
        merger.Merge(lexical.value(), semantic, parsed.value().candidate_k);
    CHECK_EQ(merged.size(), size_t{3});
    for (const Candidate& c : merged) CHECK(c.source == Candidate::Source::kBoth);

    const LinearReranker reranker(LinearWeights::Default(), FixedClock());
    Result<std::vector<Hit>> hits = reranker.Rank(*gen, parsed.value(), merged);
    CHECK(hits.ok());
    if (!hits.ok()) return;
    CHECK_EQ(hits.value().size(), size_t{3});

    CHECK_EQ(RankOf(hits.value(), "ml_intro"), 0);
    CHECK_EQ(RankOf(hits.value(), "ml_history"), 1);
    CHECK_EQ(RankOf(hits.value(), "machine_shop"), 2);
    CHECK_EQ(RankOf(hits.value(), "fr_ml"), -1);
    CHECK_EQ(RankOf(hits.value(), "gardening"), -1);

    // And the ranking is explainable. The two on-topic documents both have a phrase match
    // and full query coverage; what separates them is freshness, which is precisely the
    // sentence the feature map should support.
    for (const Hit& hit : hits.value()) {
        if (hit.external_id == "ml_intro" || hit.external_id == "ml_history") {
            CHECK(Near(FeatureOf(hit, features::kPhraseMatch), 1.0));
            CHECK(Near(FeatureOf(hit, features::kTitleMatch), 1.0));
        }
        if (hit.external_id == "machine_shop") {
            CHECK(Near(FeatureOf(hit, features::kPhraseMatch), 0.0));
            CHECK(Near(FeatureOf(hit, features::kTitleMatch), 0.5));
        }
    }
    const double intro_freshness = FeatureOf(hits.value()[0], features::kFreshness);
    const double history_freshness = FeatureOf(hits.value()[1], features::kFreshness);
    CHECK(intro_freshness > history_freshness);
    std::printf("        end to end: ml_intro %.4f, ml_history %.4f, machine_shop %.4f\n",
                hits.value()[0].score, hits.value()[1].score, hits.value()[2].score);

    // The same query without the filter picks up the French document, which confirms the
    // filter was the reason it was absent rather than the analysis.
    Result<Query> unfiltered = query::Parse("machine learning");
    CHECK(unfiltered.ok());
    if (!unfiltered.ok()) return;
    const LexicalRetriever plain({}, FieldWeights{});
    Result<std::vector<Candidate>> all = plain.Retrieve(*gen, unfiltered.value());
    CHECK(all.ok());
    if (all.ok()) CHECK_EQ(all.value().size(), size_t{4});
}

MERIDIAN_TEST(EndToEndPhraseQueryHasOneCorrectAnswer) {
    CorpusBuilder builder;
    builder.Add({"exact", "", "the quick brown fox jumps", "", "en", "general", kNow, 0.0,
                 false});
    builder.Add({"scrambled", "", "the brown quick fox jumps", "", "en", "general", kNow,
                 0.0, false});
    builder.Add({"partial", "", "a quick animal", "", "en", "general", kNow, 0.0, false});
    std::shared_ptr<Generation> gen = builder.Build();

    Result<Query> parsed = query::Parse("\"quick brown fox\"");
    CHECK(parsed.ok());
    if (!parsed.ok()) return;

    const LexicalRetriever retriever({}, FieldWeights{});
    Result<std::vector<Candidate>> result = retriever.Retrieve(*gen, parsed.value());
    CHECK(result.ok());
    if (!result.ok()) return;
    CHECK_EQ(result.value().size(), size_t{1});
    if (result.value().empty()) return;

    const LinearReranker reranker(LinearWeights::Default(), FixedClock());
    Result<std::vector<Hit>> hits = reranker.Rank(*gen, parsed.value(), result.value());
    CHECK(hits.ok());
    if (!hits.ok()) return;
    CHECK_EQ(hits.value().size(), size_t{1});
    CHECK_EQ(hits.value()[0].external_id, std::string("exact"));
    CHECK(Near(FeatureOf(hits.value()[0], features::kPhraseMatch), 1.0));
}

MERIDIAN_TEST_MAIN()
