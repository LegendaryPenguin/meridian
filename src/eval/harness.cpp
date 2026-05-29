#include "harness.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace meridian::eval {
namespace {

using Clock = std::chrono::steady_clock;

uint64_t Mean(const std::vector<uint64_t>& samples) {
    if (samples.empty()) return 0;
    // Accumulated in a wider unsigned type; a 64-bit nanosecond sum overflows only after
    // several centuries of samples, but the cast is where a reader would look for the bug.
    unsigned long long total = 0;
    for (uint64_t s : samples) total += s;
    return static_cast<uint64_t>(total / samples.size());
}

}  // namespace

uint64_t Percentile(std::vector<uint64_t> samples, double fraction) {
    if (samples.empty()) return 0;
    std::sort(samples.begin(), samples.end());
    if (fraction <= 0.0) return samples.front();
    if (fraction >= 1.0) return samples.back();

    // Nearest-rank: the smallest observed value at or above which `fraction` of the
    // samples lie. index = ceil(fraction * n) - 1, so for n = 100 the p99 is sample 99
    // (the second-largest), which is a value a query actually had.
    const auto n = static_cast<double>(samples.size());
    auto index = static_cast<size_t>(std::ceil(fraction * n));
    if (index == 0) index = 1;
    if (index > samples.size()) index = samples.size();
    return samples[index - 1];
}

uint64_t Median(std::vector<uint64_t> samples) { return Percentile(std::move(samples), 0.5); }

Result<EvalRun> RunHarness(QueryRunner& runner, const std::vector<JudgedQuery>& queries,
                           const HarnessOptions& options) {
    if (queries.empty()) {
        return Status::InvalidArgument("evaluation needs at least one judged query");
    }
    if (options.k == 0) {
        return Status::InvalidArgument("evaluation cutoff k must be positive");
    }

    const uint32_t repetitions = options.repetitions == 0 ? 1 : options.repetitions;

    EvalRun run;
    run.label = options.label.empty() ? runner.Describe() : options.label;
    run.queries = static_cast<uint32_t>(queries.size());

    // Every sample is kept. An average would be cheaper to accumulate and would make the
    // percentiles impossible: a mean cannot be turned back into a tail, and the tail is
    // the half of the latency story that a p50 hides.
    std::vector<uint64_t> total_samples;
    std::vector<uint64_t> lexical_samples;
    std::vector<uint64_t> ann_samples;
    std::vector<uint64_t> rerank_samples;
    total_samples.reserve(queries.size() * repetitions);

    std::vector<QualityMetrics> per_query;
    per_query.reserve(queries.size());

    uint64_t candidate_total = 0;
    uint64_t candidate_observations = 0;
    uint64_t cache_hits = 0;
    uint64_t cache_lookups = 0;

    for (const auto& judged : queries) {
        Query query;
        query.raw = judged.text;
        query.filters = judged.filters;
        query.top_k = options.top_k;
        query.candidate_k = options.candidate_k;
        query.deadline_ms = options.deadline_ms;
        if (options.embedding_for) {
            if (const std::vector<float>* embedding = options.embedding_for(judged.id)) {
                query.embedding = *embedding;
            }
        }

        bool quality_recorded = false;
        for (uint32_t rep = 0; rep < repetitions; ++rep) {
            if (options.cold_cache) runner.ClearCaches();

            const auto started = Clock::now();
            Result<SearchResponse> response = runner.Run(query);
            const auto elapsed = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - started)
                    .count());

            if (!response.ok()) {
                if (response.status().code() == Code::kDeadlineExceeded) {
                    ++run.queries_timed_out;
                    // A timed-out query still has a latency, and it is the interesting
                    // one: excluding it would make a configuration that times out look
                    // faster than one that answers slowly.
                    total_samples.push_back(elapsed);
                    continue;
                }
                return response.status();
            }

            const SearchResponse& value = response.value();
            // The trace's own total is preferred when the engine recorded one: it
            // excludes the harness's own call overhead, which at single-digit
            // microseconds per query is not negligible against the thing being measured.
            total_samples.push_back(value.trace.total_ns != 0 ? value.trace.total_ns : elapsed);
            lexical_samples.push_back(value.trace.lexical_ns);
            ann_samples.push_back(value.trace.ann_ns);
            rerank_samples.push_back(value.trace.rerank_ns);

            const uint32_t candidates = value.trace.merged_candidates != 0
                                            ? value.trace.merged_candidates
                                            : value.total_candidates;
            candidate_total += candidates;
            ++candidate_observations;

            ++cache_lookups;
            if (value.trace.cache_hit_results) ++cache_hits;

            if (!quality_recorded) {
                // First repetition only. Later ones answer the identical query against
                // the identical generation, so their ranking is the same by construction
                // and averaging it would imply otherwise.
                QualityMetrics metrics = Evaluate(judged, value.hits, options.k);
                per_query.push_back(metrics);
                run.per_query.emplace_back(judged.id, metrics);
                quality_recorded = true;
            }
        }

        if (!quality_recorded) {
            // Every repetition failed. The query still counts, scored as zero — dropping
            // it would let a configuration improve its NDCG by timing out on the queries
            // it is bad at.
            QualityMetrics zero;
            zero.k = options.k;
            per_query.push_back(zero);
            run.per_query.emplace_back(judged.id, zero);
        }
    }

    run.quality = Aggregate(per_query);
    run.quality.k = options.k;

    run.latency.p50_ns = Percentile(total_samples, 0.50);
    run.latency.p95_ns = Percentile(total_samples, 0.95);
    run.latency.p99_ns = Percentile(total_samples, 0.99);
    run.latency.max_ns = Percentile(total_samples, 1.0);
    run.latency.mean_ns = Mean(total_samples);

    // Medians rather than p99s per stage. A stage's p99 and the request p99 need not come
    // from the same request, so adding stage p99s produces a number no request ever had;
    // the medians at least describe a typical request's shape.
    run.latency.lexical_p50_ns = Median(std::move(lexical_samples));
    run.latency.ann_p50_ns = Median(std::move(ann_samples));
    run.latency.rerank_p50_ns = Median(std::move(rerank_samples));

    run.mean_candidates = candidate_observations == 0
                              ? 0.0
                              : static_cast<double>(candidate_total) /
                                    static_cast<double>(candidate_observations);
    run.cache_hit_rate =
        cache_lookups == 0 ? 0.0
                           : static_cast<double>(cache_hits) / static_cast<double>(cache_lookups);

    return run;
}

}  // namespace meridian::eval
