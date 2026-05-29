#pragma once

// Running a judged query set against one configuration.
//
// The harness deliberately knows nothing about how a query is served. It takes a
// `QueryRunner` and measures it. Two reasons, and the second is the one that matters:
//
//  - The engine's serving layer (src/serve/), its retrieval (src/query/, src/rank/) and
//    its index (src/index/) are owned elsewhere and are being built concurrently. A
//    harness that included those headers would not compile whenever one of them moved,
//    and the point of an evaluation harness is that it keeps working while the thing
//    under evaluation changes.
//
//  - A sweep needs to instantiate the same pipeline several times with different
//    parameters. An interface makes that a factory call; a concrete dependency would make
//    it a rebuild.
//
// The adapter from the real pipeline to `QueryRunner` lives in tools/meridian_eval.cpp,
// which is the single place that has to track the serving layer's churn.

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "meridian/eval.hpp"
#include "meridian/retrieval.hpp"
#include "meridian/status.hpp"

namespace meridian::eval {

/// Anything that can answer a query and report where the time went.
class QueryRunner {
public:
    virtual ~QueryRunner() = default;

    /// Answers one query. A failure is a measurement, not an abort: a deadline exceeded
    /// is counted and the run continues, because "this configuration times out on 4% of
    /// queries" is exactly the kind of result the harness exists to produce.
    virtual Result<SearchResponse> Run(const Query& query) = 0;

    /// Drops whatever caches sit in front of the engine. Used for cold-cache
    /// measurement, which is a different and usually more honest number than the warm
    /// one — a benchmark that re-asks the same query is measuring a hash lookup.
    virtual void ClearCaches() {}

    /// A human-readable description of the configuration being measured, for the run's
    /// label when the caller does not supply one.
    virtual std::string Describe() const { return "runner"; }
};

/// A `QueryRunner` built from a callable, so a test or a tool does not need a subclass.
class FunctionRunner : public QueryRunner {
public:
    using RunFn = std::function<Result<SearchResponse>(const Query&)>;

    explicit FunctionRunner(RunFn run, std::string description = "function",
                            std::function<void()> clear = {})
        : run_(std::move(run)), clear_(std::move(clear)), description_(std::move(description)) {}

    Result<SearchResponse> Run(const Query& query) override { return run_(query); }
    void ClearCaches() override {
        if (clear_) clear_();
    }
    std::string Describe() const override { return description_; }

private:
    RunFn run_;
    std::function<void()> clear_;
    std::string description_;
};

struct HarnessOptions {
    std::string label;

    /// Metric cutoff. Distinct from top_k: asking for 20 results and scoring the top 10
    /// is a legitimate configuration and conflating the two hides it.
    uint32_t k = 10;

    uint32_t top_k = 10;
    uint32_t candidate_k = 200;
    uint32_t deadline_ms = 250;

    /// Latency samples taken per query.
    ///
    /// More than one because a single sample per query makes the p99 of a 50-query set
    /// the single slowest query, which is a measurement of one query rather than of a
    /// tail. Quality is computed from the first repetition only — repeating a
    /// deterministic ranker cannot change its NDCG, and averaging it would only invite
    /// the reader to think it could.
    uint32_t repetitions = 1;

    /// Clear caches before every sample. Turns the run into a cold-cache measurement.
    bool cold_cache = false;

    /// Attach this query's embedding, when the judged set carries one. Looked up by
    /// query id; absent means the semantic path is skipped for that query.
    std::function<const std::vector<float>*(const std::string&)> embedding_for;
};

/// Nearest-rank percentile over recorded samples.
///
/// Exposed because the harness's latency numbers are only trustworthy if this function
/// is, and it is the one piece of the latency path that can be checked against a
/// hand-computed answer on a known array. Nearest-rank rather than interpolated: an
/// interpolated p99 reports a latency that no query actually had, which is awkward to
/// defend when someone asks which request it was.
uint64_t Percentile(std::vector<uint64_t> samples, double fraction);

/// Median, as a named thing because the per-stage numbers are all medians.
uint64_t Median(std::vector<uint64_t> samples);

/// Runs the judged set and measures both axes.
Result<EvalRun> RunHarness(QueryRunner& runner, const std::vector<JudgedQuery>& queries,
                           const HarnessOptions& options);

}  // namespace meridian::eval
