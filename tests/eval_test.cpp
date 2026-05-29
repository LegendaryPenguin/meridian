// Harness, percentile, qrels-parsing and sweep tests.
//
// The runner under test is a scripted stand-in rather than the real engine, and that is
// deliberate: a harness test driven by a real index would measure the index, and the
// thing being checked here is whether the harness reports what it was given. Scripted
// latencies make the percentiles exactly predictable, which is the only way to tell a
// percentile computed from samples apart from one computed from a mean.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "../src/eval/harness.hpp"
#include "../src/eval/sweep.hpp"
#include "meridian/eval.hpp"
#include "testing.hpp"

using namespace meridian;
using namespace meridian::eval;

namespace {

constexpr double kEpsilon = 1e-9;

void CheckNear(double actual, double expected, const char* what, int line) {
    if (std::fabs(actual - expected) > kEpsilon) {
        testing::ReportFailure(__FILE__, line,
                               std::string(what) + ": expected " + std::to_string(expected) +
                                   ", got " + std::to_string(actual));
    }
}

#define CHECK_NEAR(actual, expected) CheckNear((actual), (expected), #actual, __LINE__)

bool Contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

/// One scripted answer: what to return and how long to claim it took.
struct ScriptedAnswer {
    std::vector<std::string> hit_ids;
    uint64_t total_ns = 0;
    uint64_t lexical_ns = 0;
    uint64_t ann_ns = 0;
    uint64_t rerank_ns = 0;
    uint32_t merged_candidates = 0;
    bool cache_hit = false;
    bool deadline_exceeded = false;
};

/// Replays a script, one answer per call, cycling once exhausted.
class ScriptedRunner : public QueryRunner {
public:
    explicit ScriptedRunner(std::vector<ScriptedAnswer> script, std::string description = "scripted")
        : script_(std::move(script)), description_(std::move(description)) {}

    Result<SearchResponse> Run(const Query& query) override {
        if (script_.empty()) return Status::Internal("empty script");
        const ScriptedAnswer& answer = script_[calls_ % script_.size()];
        ++calls_;

        if (answer.deadline_exceeded) {
            return Status::DeadlineExceeded("scripted timeout");
        }

        SearchResponse response;
        for (size_t i = 0; i < answer.hit_ids.size() && i < query.top_k; ++i) {
            Hit hit;
            hit.doc = static_cast<DocId>(i);
            hit.external_id = answer.hit_ids[i];
            hit.score = 1.0 - static_cast<double>(i) * 0.01;
            response.hits.push_back(std::move(hit));
        }
        response.trace.total_ns = answer.total_ns;
        response.trace.lexical_ns = answer.lexical_ns;
        response.trace.ann_ns = answer.ann_ns;
        response.trace.rerank_ns = answer.rerank_ns;
        response.trace.merged_candidates = answer.merged_candidates;
        response.trace.cache_hit_results = answer.cache_hit;
        response.total_candidates = answer.merged_candidates;
        return response;
    }

    void ClearCaches() override { ++clears_; }
    std::string Describe() const override { return description_; }

    size_t calls() const { return calls_; }
    size_t clears() const { return clears_; }

private:
    std::vector<ScriptedAnswer> script_;
    std::string description_;
    size_t calls_ = 0;
    size_t clears_ = 0;
};

JudgedQuery Judged(const std::string& id, const std::vector<std::pair<std::string, int>>& grades) {
    JudgedQuery query;
    query.id = id;
    query.text = id + " text";
    for (const auto& [doc, grade] : grades) query.judgements.push_back(Judgement{doc, grade});
    return query;
}

/// A unique scratch directory per test, so a failing test's files can be inspected and
/// two tests cannot collide on a shared name.
std::filesystem::path ScratchDir(const std::string& name) {
    const auto dir = std::filesystem::temp_directory_path() / ("meridian-eval-" + name);
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}

void WriteFile(const std::filesystem::path& path, const std::string& contents) {
    std::ofstream out(path);
    out << contents;
}

}  // namespace

// ---------------------------------------------------------------------------
// Percentiles from real samples
// ---------------------------------------------------------------------------

MERIDIAN_TEST(PercentilesMatchHandCheckedValuesOnAKnownArray) {
    // Samples 1..100 inclusive, deliberately shuffled on input so the function has to
    // sort rather than index.
    //
    // Nearest-rank: index = ceil(fraction * 100) - 1, zero-based.
    //   p50 -> ceil(50.0) - 1 = 49  -> 50
    //   p95 -> ceil(95.0) - 1 = 94  -> 95
    //   p99 -> ceil(99.0) - 1 = 98  -> 99
    //   p100                        -> 100
    // Every one of these is a value some sample actually had, which is the property an
    // interpolated percentile gives up.
    std::vector<uint64_t> samples;
    for (uint64_t i = 1; i <= 100; ++i) samples.push_back(i);
    // A fixed non-trivial permutation; reversing would be sorted-by-accident under a
    // descending comparator bug.
    std::rotate(samples.begin(), samples.begin() + 37, samples.end());

    CHECK_EQ(Percentile(samples, 0.50), 50u);
    CHECK_EQ(Percentile(samples, 0.95), 95u);
    CHECK_EQ(Percentile(samples, 0.99), 99u);
    CHECK_EQ(Percentile(samples, 1.00), 100u);
    CHECK_EQ(Percentile(samples, 0.00), 1u);
    CHECK_EQ(Median(samples), 50u);
}

MERIDIAN_TEST(PercentilesOnSmallAndDegenerateInputs) {
    // Empty: 0 rather than undefined behaviour. A harness that ran no queries has no p99,
    // and Compare treats a zero p99 as "not measured" rather than "instant".
    CHECK_EQ(Percentile({}, 0.99), 0u);
    CHECK_EQ(Median({}), 0u);

    // One sample is every percentile of itself.
    CHECK_EQ(Percentile({42}, 0.50), 42u);
    CHECK_EQ(Percentile({42}, 0.99), 42u);

    // {1,2,3}: median index = ceil(1.5) - 1 = 1 -> 2.
    CHECK_EQ(Median({3, 1, 2}), 2u);

    // {10,20,30,40}: median index = ceil(2.0) - 1 = 1 -> 20. The lower of the two middle
    // samples, not their mean: nearest-rank never invents a value.
    CHECK_EQ(Median({40, 10, 30, 20}), 20u);

    // Three samples, p99: ceil(2.97) - 1 = 2 -> 3. With so few samples a p99 is the max,
    // which is worth knowing when reading a tail number off a short run.
    CHECK_EQ(Percentile({1, 2, 3}, 0.99), 3u);
}

// ---------------------------------------------------------------------------
// The harness
// ---------------------------------------------------------------------------

MERIDIAN_TEST(HarnessLatencyComesFromSamplesNotFromAMean) {
    // 100 queries, scripted total_ns = 1000, 2000, ... 100000.
    //   p50  = 50000   (sample 50)
    //   p95  = 95000
    //   p99  = 99000
    //   max  = 100000
    //   mean = 1000 * (1 + ... + 100) / 100 = 1000 * 5050 / 100 = 50500
    // The mean and the p99 differ by a factor of two here. A harness that reported the
    // mean under the name p99 would be off by that much and still look plausible.
    std::vector<ScriptedAnswer> script;
    std::vector<JudgedQuery> queries;
    for (uint64_t i = 1; i <= 100; ++i) {
        ScriptedAnswer answer;
        answer.hit_ids = {"d" + std::to_string(i)};
        answer.total_ns = i * 1000;
        answer.lexical_ns = 400;
        answer.ann_ns = 800;
        answer.rerank_ns = 1200;
        answer.merged_candidates = 200;
        answer.cache_hit = i <= 25;  // 25 of 100 -> hit rate 0.25
        script.push_back(answer);
        queries.push_back(Judged("q" + std::to_string(i), {{"d" + std::to_string(i), 3}}));
    }

    ScriptedRunner runner(std::move(script));
    HarnessOptions options;
    options.label = "scripted-latency";
    options.k = 10;

    Result<EvalRun> result = RunHarness(runner, queries, options);
    CHECK(result.ok());
    const EvalRun& run = result.value();

    CHECK_EQ(run.latency.p50_ns, 50'000u);
    CHECK_EQ(run.latency.p95_ns, 95'000u);
    CHECK_EQ(run.latency.p99_ns, 99'000u);
    CHECK_EQ(run.latency.max_ns, 100'000u);
    CHECK_EQ(run.latency.mean_ns, 50'500u);

    // Per-stage medians come straight from the traces.
    CHECK_EQ(run.latency.lexical_p50_ns, 400u);
    CHECK_EQ(run.latency.ann_p50_ns, 800u);
    CHECK_EQ(run.latency.rerank_p50_ns, 1200u);

    CHECK_EQ(run.queries, 100u);
    CHECK_EQ(run.queries_timed_out, 0u);
    CHECK_NEAR(run.mean_candidates, 200.0);
    CHECK_NEAR(run.cache_hit_rate, 0.25);
    CHECK_EQ(run.per_query.size(), size_t{100});
    CHECK_EQ(run.label, std::string("scripted-latency"));

    // Each query's single relevant document is returned at rank 1, so NDCG is 1.0 for all
    // of them and the macro average is 1.0.
    CHECK_NEAR(run.quality.ndcg_at_k, 1.0);
    CHECK_NEAR(run.quality.mrr, 1.0);
    // P@10 = 1/10 for every query: one relevant document is all there is to find.
    CHECK_NEAR(run.quality.precision_at_k, 0.1);
    CHECK_NEAR(run.quality.recall_at_k, 1.0);
}

MERIDIAN_TEST(HarnessCountsTimeoutsAndStillScoresTheQuery) {
    // Three queries; the middle one times out. The timeout is counted, the query is
    // scored zero rather than dropped, and the run still completes — a configuration that
    // times out on the queries it is bad at must not be able to improve its NDCG that way.
    std::vector<ScriptedAnswer> script(3);
    script[0].hit_ids = {"a1"};
    script[0].total_ns = 1000;
    script[1].deadline_exceeded = true;
    script[2].hit_ids = {"c1"};
    script[2].total_ns = 3000;

    const std::vector<JudgedQuery> queries = {
        Judged("qa", {{"a1", 3}}),
        Judged("qb", {{"b1", 3}}),
        Judged("qc", {{"c1", 3}}),
    };

    ScriptedRunner runner(std::move(script));
    HarnessOptions options;
    options.k = 10;

    Result<EvalRun> result = RunHarness(runner, queries, options);
    CHECK(result.ok());
    const EvalRun& run = result.value();

    CHECK_EQ(run.queries, 3u);
    CHECK_EQ(run.queries_timed_out, 1u);
    CHECK_EQ(run.per_query.size(), size_t{3});
    CHECK_EQ(run.per_query[1].first, std::string("qb"));
    CHECK_NEAR(run.per_query[1].second.ndcg_at_k, 0.0);
    // Macro average of 1.0, 0.0, 1.0.
    CHECK_NEAR(run.quality.ndcg_at_k, 2.0 / 3.0);
}

MERIDIAN_TEST(HarnessRepetitionsAddLatencySamplesWithoutReweightingQuality) {
    // Two queries, three repetitions: six latency samples, two quality records. Quality is
    // deterministic for a fixed generation, so repeating it would add nothing but would
    // suggest it varied.
    std::vector<ScriptedAnswer> script(2);
    script[0].hit_ids = {"a1"};
    script[0].total_ns = 1000;
    script[1].hit_ids = {"b1"};
    script[1].total_ns = 2000;

    const std::vector<JudgedQuery> queries = {Judged("qa", {{"a1", 3}}), Judged("qb", {{"b1", 3}})};

    ScriptedRunner runner(std::move(script));
    HarnessOptions options;
    options.k = 10;
    options.repetitions = 3;

    Result<EvalRun> result = RunHarness(runner, queries, options);
    CHECK(result.ok());
    CHECK_EQ(runner.calls(), size_t{6});
    CHECK_EQ(result.value().per_query.size(), size_t{2});
    CHECK_EQ(result.value().queries, 2u);
}

MERIDIAN_TEST(HarnessColdCacheClearsBeforeEverySample) {
    // Cold-cache mode clears before each of the 2 queries x 2 repetitions = 4 samples.
    // Without this a "cold" number is the warm number for every sample after the first.
    std::vector<ScriptedAnswer> script(1);
    script[0].hit_ids = {"a1"};
    script[0].total_ns = 1000;

    const std::vector<JudgedQuery> queries = {Judged("qa", {{"a1", 3}}), Judged("qb", {{"a1", 3}})};

    ScriptedRunner runner(std::move(script));
    HarnessOptions options;
    options.k = 10;
    options.repetitions = 2;
    options.cold_cache = true;

    CHECK(RunHarness(runner, queries, options).ok());
    CHECK_EQ(runner.clears(), size_t{4});
}

MERIDIAN_TEST(HarnessUsesRunnerDescriptionWhenUnlabelled) {
    std::vector<ScriptedAnswer> script(1);
    script[0].hit_ids = {"a1"};
    script[0].total_ns = 1000;

    ScriptedRunner runner(std::move(script), "ck=200 rrf");
    Result<EvalRun> result = RunHarness(runner, {Judged("qa", {{"a1", 3}})}, HarnessOptions{});
    CHECK(result.ok());
    CHECK_EQ(result.value().label, std::string("ck=200 rrf"));
}

MERIDIAN_TEST(HarnessRejectsEmptyInputRatherThanReportingZeroes) {
    // An evaluation over no queries has no result, and returning an all-zero EvalRun would
    // be a score of zero that looks like a measurement.
    std::vector<ScriptedAnswer> script(1);
    script[0].hit_ids = {"a1"};
    ScriptedRunner runner(std::move(script));

    CHECK(!RunHarness(runner, {}, HarnessOptions{}).ok());

    HarnessOptions zero_k;
    zero_k.k = 0;
    CHECK(!RunHarness(runner, {Judged("qa", {{"a1", 3}})}, zero_k).ok());
}

MERIDIAN_TEST(HarnessAttachesQueryEmbeddingsWhenProvided) {
    // The semantic path only runs when the query carries an embedding, so a judged set
    // with embeddings and a harness that dropped them would silently evaluate
    // lexical-only retrieval and report it as hybrid.
    const std::vector<float> vector = {0.5f, 0.5f, 0.70710678f};
    bool saw_embedding = false;

    FunctionRunner runner(
        [&](const Query& query) -> Result<SearchResponse> {
            saw_embedding = query.embedding.has_value() && query.embedding->size() == 3;
            SearchResponse response;
            Hit hit;
            hit.external_id = "a1";
            response.hits.push_back(hit);
            response.trace.total_ns = 1000;
            return response;
        },
        "embedding-aware");

    HarnessOptions options;
    options.embedding_for = [&](const std::string& id) -> const std::vector<float>* {
        return id == "qa" ? &vector : nullptr;
    };

    CHECK(RunHarness(runner, {Judged("qa", {{"a1", 3}})}, options).ok());
    CHECK(saw_embedding);
}

// ---------------------------------------------------------------------------
// qrels and queries parsing
// ---------------------------------------------------------------------------

MERIDIAN_TEST(LoadJudgementsReadsTrecStyleFiles) {
    const auto dir = ScratchDir("qrels-good");
    const auto queries = dir / "queries.tsv";
    const auto qrels = dir / "qrels.txt";

    WriteFile(queries,
              "# topic file: qid<TAB>text[<TAB>filters]\n"
              "\n"
              "q1\tvector index construction\n"
              "q2\tbm25 length normalisation\tlanguage=en;category=papers\n");
    WriteFile(qrels,
              "# qid iteration docid grade\n"
              "q1 0 doc-7 3\n"
              "q1 0 doc-9 1\n"
              "q2 0 doc-3 2\n"
              "q9 0 doc-1 3\n");  // a topic not in the queries file: ignored, not an error

    Result<std::vector<JudgedQuery>> loaded = LoadJudgements(queries.string(), qrels.string());
    CHECK(loaded.ok());
    if (!loaded.ok()) return;

    const auto& set = loaded.value();
    CHECK_EQ(set.size(), size_t{2});
    CHECK_EQ(set[0].id, std::string("q1"));
    CHECK_EQ(set[0].text, std::string("vector index construction"));
    CHECK_EQ(set[0].judgements.size(), size_t{2});
    CHECK_EQ(set[0].GradeOf("doc-7"), 3);
    CHECK_EQ(set[0].GradeOf("doc-9"), 1);
    CHECK_EQ(set[0].GradeOf("doc-absent"), 0);

    CHECK_EQ(set[1].judgements.size(), size_t{1});
    CHECK(set[1].filters.language.has_value());
    CHECK_EQ(*set[1].filters.language, std::string("en"));
    CHECK_EQ(*set[1].filters.category, std::string("papers"));
}

MERIDIAN_TEST(LoadJudgementsAcceptsThreeColumnQrels) {
    // A good share of published qrels drop the iteration column. Rejecting them would be
    // rejecting the data rather than finding a problem with it.
    const auto dir = ScratchDir("qrels-three-col");
    WriteFile(dir / "queries.tsv", "q1\tsome query\n");
    WriteFile(dir / "qrels.txt", "q1 doc-1 2\nq1 doc-2 0\n");

    Result<std::vector<JudgedQuery>> loaded =
        LoadJudgements((dir / "queries.tsv").string(), (dir / "qrels.txt").string());
    CHECK(loaded.ok());
    if (!loaded.ok()) return;
    CHECK_EQ(loaded.value()[0].judgements.size(), size_t{2});
    CHECK_EQ(loaded.value()[0].GradeOf("doc-1"), 2);
}

MERIDIAN_TEST(MalformedQrelsLineIsReportedWithItsLineNumber) {
    // Line 3 has two fields. The error names the file and the line, because "malformed
    // qrels" on a 50,000-line file is not an actionable message.
    const auto dir = ScratchDir("qrels-bad-fields");
    WriteFile(dir / "queries.tsv", "q1\tsome query\n");
    WriteFile(dir / "qrels.txt", "q1 0 doc-1 3\n# comment\nq1 doc-2\nq1 0 doc-3 1\n");

    Result<std::vector<JudgedQuery>> loaded =
        LoadJudgements((dir / "queries.tsv").string(), (dir / "qrels.txt").string());
    CHECK(!loaded.ok());
    if (loaded.ok()) return;
    CHECK(Contains(loaded.status().message(), "qrels.txt:3:"));
    CHECK(Contains(loaded.status().message(), "got 2"));
}

MERIDIAN_TEST(NonIntegerGradeIsReportedWithItsLineNumber) {
    // "2.5" parses as 2 under std::stoi and as 2 under atoi. Both would produce a
    // judgement set that is quietly not the one on disk.
    const auto dir = ScratchDir("qrels-bad-grade");
    WriteFile(dir / "queries.tsv", "q1\tsome query\n");
    WriteFile(dir / "qrels.txt", "q1 0 doc-1 3\nq1 0 doc-2 2.5\n");

    Result<std::vector<JudgedQuery>> loaded =
        LoadJudgements((dir / "queries.tsv").string(), (dir / "qrels.txt").string());
    CHECK(!loaded.ok());
    if (loaded.ok()) return;
    CHECK(Contains(loaded.status().message(), "qrels.txt:2:"));
    CHECK(Contains(loaded.status().message(), "2.5"));
}

MERIDIAN_TEST(NegativeGradeIsRejectedRatherThanTurnedIntoNegativeGain) {
    // Some collections use -1 for "pooled and judged non-relevant, do not reuse". A
    // negative gain would make DCG non-monotone in the ranking, which breaks the metric
    // rather than the data, so the caller is told to map it instead.
    const auto dir = ScratchDir("qrels-negative");
    WriteFile(dir / "queries.tsv", "q1\tsome query\n");
    WriteFile(dir / "qrels.txt", "q1 0 doc-1 -1\n");

    Result<std::vector<JudgedQuery>> loaded =
        LoadJudgements((dir / "queries.tsv").string(), (dir / "qrels.txt").string());
    CHECK(!loaded.ok());
    if (loaded.ok()) return;
    CHECK(Contains(loaded.status().message(), "qrels.txt:1:"));
    CHECK(Contains(loaded.status().message(), "negative grade"));
}

MERIDIAN_TEST(MalformedQueriesLineIsReportedWithItsLineNumber) {
    // Line 2 has no tab, so there is no way to tell the id from the text.
    const auto dir = ScratchDir("queries-bad");
    WriteFile(dir / "queries.tsv", "q1\tgood query\nq2 missing tab\n");
    WriteFile(dir / "qrels.txt", "q1 0 doc-1 3\n");

    Result<std::vector<JudgedQuery>> loaded =
        LoadJudgements((dir / "queries.tsv").string(), (dir / "qrels.txt").string());
    CHECK(!loaded.ok());
    if (loaded.ok()) return;
    CHECK(Contains(loaded.status().message(), "queries.tsv:2:"));
}

MERIDIAN_TEST(DuplicateQueryIdIsReportedWithItsLineNumber) {
    // Two topics with the same id would merge their judgements silently, producing a query
    // that was never labelled as such.
    const auto dir = ScratchDir("queries-dup");
    WriteFile(dir / "queries.tsv", "q1\tfirst\nq1\tsecond\n");
    WriteFile(dir / "qrels.txt", "q1 0 doc-1 3\n");

    Result<std::vector<JudgedQuery>> loaded =
        LoadJudgements((dir / "queries.tsv").string(), (dir / "qrels.txt").string());
    CHECK(!loaded.ok());
    if (loaded.ok()) return;
    CHECK(Contains(loaded.status().message(), "queries.tsv:2:"));
    CHECK(Contains(loaded.status().message(), "duplicate"));
}

MERIDIAN_TEST(UnknownFilterKeyIsReportedWithItsLineNumber) {
    const auto dir = ScratchDir("queries-bad-filter");
    WriteFile(dir / "queries.tsv", "q1\tgood\nq2\talso good\tsentiment=happy\n");
    WriteFile(dir / "qrels.txt", "q1 0 doc-1 3\n");

    Result<std::vector<JudgedQuery>> loaded =
        LoadJudgements((dir / "queries.tsv").string(), (dir / "qrels.txt").string());
    CHECK(!loaded.ok());
    if (loaded.ok()) return;
    CHECK(Contains(loaded.status().message(), "queries.tsv:2:"));
    CHECK(Contains(loaded.status().message(), "sentiment"));
}

MERIDIAN_TEST(MissingFilesAreIoErrorsNotEmptyResults) {
    const auto dir = ScratchDir("qrels-missing");
    WriteFile(dir / "queries.tsv", "q1\tgood\n");

    Result<std::vector<JudgedQuery>> no_qrels =
        LoadJudgements((dir / "queries.tsv").string(), (dir / "absent.txt").string());
    CHECK(!no_qrels.ok());
    if (!no_qrels.ok()) CHECK(no_qrels.status().code() == Code::kIoError);

    Result<std::vector<JudgedQuery>> no_queries =
        LoadJudgements((dir / "absent.tsv").string(), (dir / "queries.tsv").string());
    CHECK(!no_queries.ok());
    if (!no_queries.ok()) CHECK(no_queries.status().code() == Code::kIoError);
}

MERIDIAN_TEST(QueriesFileWithNoQueriesIsAnError) {
    const auto dir = ScratchDir("queries-empty");
    WriteFile(dir / "queries.tsv", "# nothing but a comment\n\n");
    WriteFile(dir / "qrels.txt", "q1 0 doc-1 3\n");

    CHECK(!LoadJudgements((dir / "queries.tsv").string(), (dir / "qrels.txt").string()).ok());
}

// ---------------------------------------------------------------------------
// Sweeps
// ---------------------------------------------------------------------------

namespace {

/// A candidate_k sweep with the shape every real one has: quality saturates, latency does
/// not. Hand-chosen so the verdicts are predictable.
///
///   candidate_k  NDCG   p99 (ns)   vs baseline (ck=50, tolerance 10%)
///   50           0.50   100000     baseline
///   100          0.60   140000     quality +0.10, p99 1.40x -> tradeoff
///   200          0.62   280000     quality +0.12, p99 2.80x -> tradeoff
///   400          0.62   560000     quality  0.00, p99 5.60x -> worse
/// Quality is expressed as *how many of three relevant documents the larger candidate set
/// reaches*, which is what candidate_k physically changes, rather than by writing an NDCG
/// in directly. With three grade-3 judgements and IDCG = 7 + 7/log2(3) + 7/2 =
/// 14.9165082750:
///
///   1 of 3 found, at rank 1:      DCG = 7                -> NDCG 0.4693
///   2 of 3 found, at ranks 1,2:   DCG = 11.4165082750    -> NDCG 0.7654
///   3 of 3 found, at ranks 1,2,3: DCG = 14.9165082750    -> NDCG 1.0000
Result<std::unique_ptr<QueryRunner>> SaturatingFactory(const SweepPoint& point) {
    uint64_t p99 = 0;
    std::vector<std::string> hits;
    switch (point.candidate_k) {
        case 50: p99 = 100'000; hits = {"r1", "u1", "u2"}; break;
        case 100: p99 = 140'000; hits = {"r1", "r2", "u1"}; break;
        case 200: p99 = 280'000; hits = {"r1", "r2", "r3"}; break;
        case 400: p99 = 560'000; hits = {"r1", "r2", "r3"}; break;
        default: return Status::InvalidArgument("no scripted answer for this candidate_k");
    }

    ScriptedAnswer answer;
    answer.hit_ids = hits;
    answer.total_ns = p99;
    answer.lexical_ns = p99 / 4;
    answer.ann_ns = p99 / 4;
    answer.rerank_ns = p99 / 4;
    answer.merged_candidates = point.candidate_k;

    return std::unique_ptr<QueryRunner>(
        new ScriptedRunner({answer}, "candidate_k=" + std::to_string(point.candidate_k)));
}

std::vector<JudgedQuery> ThreeRelevantQuery() {
    return {Judged("q1", {{"r1", 3}, {"r2", 3}, {"r3", 3}})};
}

}  // namespace

MERIDIAN_TEST(SweepLabelsAQualityGainWithALatencyCostAsATradeoff) {
    // The whole reason sweep.cpp compares every row against the baseline rather than
    // sorting by quality. The ck=200 and ck=400 rows have the best NDCG in the table and
    // neither is an improvement.
    //
    // Hand-computed per row (one query, three grade-3 judgements, k = 10):
    //   ck=50:  hits r1,u1,u2     -> DCG 7,            NDCG 7/14.9165082750 = 0.4693
    //           p99 100000ns (baseline)
    //   ck=100: hits r1,r2,u1     -> DCG 11.4165082750, NDCG = 0.7654
    //           p99 140000ns -> ratio 1.40 > 1.10, quality +0.2961 -> tradeoff
    //   ck=200: hits r1,r2,r3     -> DCG 14.9165082750, NDCG = 1.0000
    //           p99 280000ns -> ratio 2.80, quality +0.5307 -> tradeoff
    //   ck=400: hits r1,r2,r3     -> NDCG = 1.0000 (saturated)
    //           p99 560000ns -> ratio 5.60, quality +0.5307 -> tradeoff
    const auto points = SweepCandidateK(SweepPoint{}, {50, 100, 200, 400});
    SweepOptions options;
    options.harness.k = 10;
    options.latency_tolerance = 0.10;

    Result<SweepResult> result =
        RunSweep(points, SaturatingFactory, ThreeRelevantQuery(), options);
    CHECK(result.ok());
    if (!result.ok()) return;

    const auto& rows = result.value().rows;
    CHECK_EQ(rows.size(), size_t{4});

    const double ideal = 7.0 + 7.0 / std::log2(3.0) + 7.0 / 2.0;
    CHECK_NEAR(rows[0].run.quality.ndcg_at_k, 7.0 / ideal);
    CHECK_NEAR(rows[1].run.quality.ndcg_at_k, (7.0 + 7.0 / std::log2(3.0)) / ideal);
    CHECK_NEAR(rows[2].run.quality.ndcg_at_k, 1.0);
    CHECK_NEAR(rows[3].run.quality.ndcg_at_k, 1.0);

    CHECK_EQ(rows[0].run.latency.p99_ns, 100'000u);
    CHECK_EQ(rows[3].run.latency.p99_ns, 560'000u);

    // The baseline compares to itself and is unchanged by construction.
    CHECK(rows[0].vs_baseline.verdict == Verdict::kNoChange);

    // Every quality gain in this sweep cost tail latency, so none of them is a win.
    CHECK(rows[1].vs_baseline.verdict == Verdict::kTradeoff);
    CHECK(rows[2].vs_baseline.verdict == Verdict::kTradeoff);
    CHECK(rows[3].vs_baseline.verdict == Verdict::kTradeoff);

    CHECK_NEAR(rows[1].vs_baseline.p99_ratio, 1.4);
    CHECK_NEAR(rows[3].vs_baseline.p99_ratio, 5.6);

    // ck=400 bought nothing over ck=200: identical NDCG, twice the p99. That is the row a
    // table sorted by quality would have tied for first place.
    CHECK_NEAR(rows[3].run.quality.ndcg_at_k, rows[2].run.quality.ndcg_at_k);
    CHECK(rows[3].run.latency.p99_ns > rows[2].run.latency.p99_ns);
}

MERIDIAN_TEST(SweepSaturationIsVisibleWhenRebaselinedOnTheSaturatedRow) {
    // Re-running the same sweep with ck=200 as the baseline isolates the saturation: ck=400
    // doubles p99 for no quality at all, which Compare calls worse rather than a tie.
    const auto points = SweepCandidateK(SweepPoint{}, {50, 100, 200, 400});
    SweepOptions options;
    options.harness.k = 10;
    options.baseline_index = 2;  // ck=200

    Result<SweepResult> result =
        RunSweep(points, SaturatingFactory, ThreeRelevantQuery(), options);
    CHECK(result.ok());
    if (!result.ok()) return;

    const auto& rows = result.value().rows;
    CHECK(rows[3].vs_baseline.verdict == Verdict::kWorse);
    CHECK_NEAR(rows[3].vs_baseline.ndcg_delta, 0.0);
    CHECK_NEAR(rows[3].vs_baseline.p99_ratio, 2.0);
    CHECK(Contains(rows[3].vs_baseline.explanation, "no quality gain"));
    // And the cheap rows are now worse on quality.
    CHECK(rows[0].vs_baseline.verdict == Verdict::kWorse);
}

MERIDIAN_TEST(SweepTableCarriesBothAxesAndTheVerdict) {
    const auto points = SweepCandidateK(SweepPoint{}, {50, 200});
    SweepOptions options;
    options.harness.k = 10;

    Result<SweepResult> result =
        RunSweep(points, SaturatingFactory, ThreeRelevantQuery(), options);
    CHECK(result.ok());
    if (!result.ok()) return;

    const std::string table = FormatTable(result.value());
    CHECK(Contains(table, "NDCG@k"));
    CHECK(Contains(table, "p99 us"));
    CHECK(Contains(table, "verdict"));
    CHECK(Contains(table, "candidate_k=50 (baseline)"));
    CHECK(Contains(table, "candidate_k=200"));
    CHECK(Contains(table, "tradeoff"));
    // Latency is rendered in microseconds: 280000ns -> 280.00.
    CHECK(Contains(table, "280.00"));

    const std::string verdicts = FormatVerdicts(result.value());
    CHECK(Contains(verdicts, "candidate_k=200"));
    CHECK(Contains(verdicts, "2.80x"));
    CHECK(!Contains(verdicts, "candidate_k=50:"));
}

MERIDIAN_TEST(SweepAxisBuildersVaryOneParameterAtATime) {
    SweepPoint base;
    base.candidate_k = 200;
    base.ef_search = 64;
    base.bm25.k1 = 1.2;
    base.bm25.b = 0.75;

    const auto candidate_points = SweepCandidateK(base, {50, 100});
    CHECK_EQ(candidate_points.size(), size_t{2});
    CHECK_EQ(candidate_points[0].candidate_k, 50u);
    CHECK_EQ(candidate_points[0].ef_search, 64u);
    CHECK_NEAR(candidate_points[0].bm25.k1, 1.2);
    CHECK_EQ(candidate_points[1].label, std::string("candidate_k=100"));

    const auto ef_points = SweepEfSearch(base, {16, 64, 256});
    CHECK_EQ(ef_points.size(), size_t{3});
    CHECK_EQ(ef_points[2].ef_search, 256u);
    CHECK_EQ(ef_points[2].candidate_k, 200u);

    // A grid: 2 k1 values x 3 b values = 6 points, with candidate_k untouched.
    const auto bm25_points = SweepBm25(base, {0.9, 1.6}, {0.0, 0.4, 0.75});
    CHECK_EQ(bm25_points.size(), size_t{6});
    CHECK_NEAR(bm25_points[0].bm25.k1, 0.9);
    CHECK_NEAR(bm25_points[0].bm25.b, 0.0);
    CHECK_NEAR(bm25_points[5].bm25.k1, 1.6);
    CHECK_NEAR(bm25_points[5].bm25.b, 0.75);
    CHECK_EQ(bm25_points[5].candidate_k, 200u);
    CHECK_EQ(bm25_points[0].label, std::string("k1=0.90,b=0.00"));

    const auto hybrid_points = SweepHybrid(
        base,
        {HybridMerger::Strategy::kReciprocalRankFusion, HybridMerger::Strategy::kZScoreLinear},
        {{1.0, 0.0}, {0.5, 0.5}, {0.0, 1.0}});
    CHECK_EQ(hybrid_points.size(), size_t{6});
    CHECK(hybrid_points[0].strategy == HybridMerger::Strategy::kReciprocalRankFusion);
    CHECK_NEAR(hybrid_points[2].semantic_weight, 1.0);
    CHECK(hybrid_points[3].strategy == HybridMerger::Strategy::kZScoreLinear);
    CHECK_EQ(hybrid_points[3].label, std::string("zscore-linear 1.00/0.00"));
}

MERIDIAN_TEST(SweepPointAppliesCandidateKToTheQuery) {
    SweepPoint point;
    point.candidate_k = 512;
    Query query;
    point.ApplyTo(&query);
    CHECK_EQ(query.candidate_k, 512u);
}

MERIDIAN_TEST(SweepRejectsBadInput) {
    SweepOptions options;
    options.harness.k = 10;

    CHECK(!RunSweep({}, SaturatingFactory, ThreeRelevantQuery(), options).ok());

    options.baseline_index = 7;
    CHECK(!RunSweep(SweepCandidateK(SweepPoint{}, {50}), SaturatingFactory, ThreeRelevantQuery(),
                    options)
               .ok());

    // A factory failure propagates rather than leaving a hole in the table that would read
    // as a gap in the parameter range.
    options.baseline_index = 0;
    CHECK(!RunSweep(SweepCandidateK(SweepPoint{}, {50, 999}), SaturatingFactory,
                    ThreeRelevantQuery(), options)
               .ok());
}

MERIDIAN_TEST(DescribePointFallsBackToTheParameters) {
    SweepPoint point;
    point.candidate_k = 128;
    point.ef_search = 32;
    const std::string described = DescribePoint(point);
    CHECK(Contains(described, "ck=128"));
    CHECK(Contains(described, "ef=32"));
    CHECK(Contains(described, "rrf"));

    point.label = "explicit";
    CHECK_EQ(DescribePoint(point), std::string("explicit"));
}

MERIDIAN_TEST_MAIN()
