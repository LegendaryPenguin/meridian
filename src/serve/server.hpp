#pragma once

// The query server: admission control, deadlines, and latency that is measured rather
// than averaged.
//
// Three decisions, each of which is the reason the class exists.
//
// **Bounded concurrency, with refusal.** A fixed worker pool and a fixed-size queue.
// When both are full the server answers kOverloaded immediately. The alternative — an
// unbounded queue — does not remove the overload, it converts it into latency: offered
// load above capacity makes the queue grow without limit, and every query in it still
// gets answered eventually, at a response time nobody has any use for. A search engine
// that replies in forty seconds has already failed; it has merely failed in a way that
// looks like success in the logs. Refusing is also the only honest signal available to
// whatever is upstream — a load balancer can shed or retry elsewhere, but it cannot
// react to a queue it cannot see.
//
// **Deadlines measured from arrival, not from dequeue.** The budget starts when the
// query is submitted, so time spent waiting for a worker is charged against it. The
// opposite choice makes queue delay invisible to the deadline, which is exactly the
// case the deadline was for.
//
// **A query that runs out of budget returns nothing but the reason.** The choice was
// between a kDeadlineExceeded carrying the partial hits produced so far and a clean
// refusal, and this is a clean refusal: `Result<SearchResponse>` holds either a response
// or a Status, so there is no short hit list for a caller to mistake for a complete
// answer. Partial results would be worse than useless here — a top-10 request answered
// with the four documents that happened to be scored before the clock ran out is
// indistinguishable, at the call site, from a corpus that only contains four matches, and
// the caller has no way to tell which it got. What the refusal does carry is diagnosis:
// the status message names the stage boundary where the budget expired, and the per-stage
// trace of the failed attempt is folded into the server's DeadlineBreakdown counters. The
// one exception, documented at the end of Pipeline::Run, is that a query which has already
// finished every stage is never discarded for being late.
//
// **Percentiles from samples.** A mean latency hides the tail by construction: the
// requests users complain about are the ones the mean averages away. This keeps real
// samples and sorts them.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <vector>

#include "meridian/retrieval.hpp"
#include "meridian/status.hpp"
#include "pipeline.hpp"
#include "task_pool.hpp"

namespace meridian::serve {

/// A bounded reservoir of latency samples with exact percentiles over what it holds.
///
/// Why samples rather than a histogram with fixed buckets: a bucketed histogram answers
/// a percentile only to within a bucket width, and the bucket widths have to be chosen
/// before anyone knows what the latency distribution looks like. Why bounded rather
/// than every sample: a server that keeps every sample forever has a memory leak with a
/// metrics label on it. Beyond capacity this switches to reservoir sampling (Vitter's
/// algorithm R), which keeps a uniform sample of the whole stream — so a late burst of
/// slow queries can still move p99, which a "keep the first N" scheme would miss
/// entirely.
class LatencySamples {
public:
    explicit LatencySamples(size_t capacity = 1u << 16);

    void Record(uint64_t nanos);

    /// Nearest-rank percentile: the smallest sample at or above which `p` of the
    /// samples lie, with `p` in [0,1]. Stated precisely because a hand-checked
    /// percentile depends on the definition — nearest-rank and linear interpolation
    /// disagree on small sample sets, and a test that assumes the wrong one fails for
    /// the wrong reason.
    uint64_t Percentile(double p) const;

    uint64_t count() const;
    uint64_t max() const;
    size_t stored() const;
    void Reset();

private:
    mutable std::mutex mutex_;
    std::vector<uint64_t> samples_;
    size_t capacity_;
    uint64_t count_ = 0;
    uint64_t max_ = 0;

    /// Fixed seed. Percentile output should not depend on which run it was: a
    /// reproducible sample is worth more here than an unpredictable one, and nothing
    /// about this is security-relevant.
    uint64_t rng_state_ = 0x2545f4914f6cdd1dULL;
    uint64_t NextRandom();
};

struct ServerOptions {
    /// Concurrency limit. Beyond this, queries wait in the queue; beyond the queue,
    /// they are refused. The number should be near the core count — more workers than
    /// cores does not increase throughput for CPU-bound retrieval, it just spreads each
    /// query's latency over more context switches.
    size_t worker_threads = 4;

    /// Queue depth. The knob that sets the worst-case queue wait: at a given service
    /// time, queue_capacity directly bounds how long an accepted query can sit. Set it
    /// from the latency budget, not from a desire to avoid refusals.
    size_t queue_capacity = 64;

    size_t latency_samples = 1u << 16;
};

/// Accepts queries, runs them under a budget, refuses when full.
class QueryServer {
public:
    QueryServer(ServerOptions options, const GenerationSource* generations, const Pipeline* pipeline);
    ~QueryServer();

    QueryServer(const QueryServer&) = delete;
    QueryServer& operator=(const QueryServer&) = delete;

    /// Submits a query. The returned future is already satisfied with kOverloaded when
    /// the server has no room — a refusal is immediate by definition, since the point of
    /// refusing is not to wait.
    std::future<Result<SearchResponse>> Submit(const Query& query);

    /// Submit and wait. The wait is bounded by the query's own deadline, which the
    /// worker enforces, so this cannot block indefinitely on a slow pipeline.
    Result<SearchResponse> Search(const Query& query);

    /// Stops accepting, drains what was accepted, joins the workers. Queued queries are
    /// completed rather than dropped because each has a caller blocked on a future.
    void Shutdown();

    /// Why deadline-exceeded queries died, derived from the partial traces rather than
    /// from parsing status messages. A server whose expiries are all "before merge" has
    /// a candidate-generation problem; one where they are all "before materialize" is
    /// mostly meeting its budget and losing on the last stage.
    struct DeadlineBreakdown {
        uint64_t before_parse = 0;
        uint64_t before_merge = 0;
        uint64_t before_rerank = 0;
        uint64_t before_materialize = 0;
    };

    struct Metrics {
        uint64_t accepted = 0;
        uint64_t rejected_overloaded = 0;
        uint64_t completed = 0;
        uint64_t failed = 0;
        uint64_t deadline_exceeded = 0;
        uint64_t queue_high_water = 0;
        uint64_t in_flight = 0;

        /// Percentiles over successful queries only. Mixing in refusals would make the
        /// tail look better the more the server sheds, since a refusal is fast.
        uint64_t p50_ns = 0;
        uint64_t p95_ns = 0;
        uint64_t p99_ns = 0;
        uint64_t max_ns = 0;

        DeadlineBreakdown deadlines;
    };
    Metrics metrics() const;

    /// Samples of successful query latency, exposed so a test or a benchmark can check
    /// a percentile by hand against the same numbers the server reported.
    const LatencySamples& latency() const { return latency_; }

private:
    void Execute(const Query& query, Deadline deadline,
                 std::shared_ptr<std::promise<Result<SearchResponse>>> promise);
    void RecordDeadline(const QueryTrace& partial);

    ServerOptions options_;
    const GenerationSource* generations_;
    const Pipeline* pipeline_;
    std::unique_ptr<TaskPool> pool_;
    LatencySamples latency_;

    std::atomic<uint64_t> accepted_{0};
    std::atomic<uint64_t> rejected_{0};
    std::atomic<uint64_t> completed_{0};
    std::atomic<uint64_t> failed_{0};
    std::atomic<uint64_t> deadline_exceeded_{0};
    std::atomic<uint64_t> queue_high_water_{0};
    std::atomic<uint64_t> in_flight_{0};
    std::atomic<uint64_t> expired_before_parse_{0};
    std::atomic<uint64_t> expired_before_merge_{0};
    std::atomic<uint64_t> expired_before_rerank_{0};
    std::atomic<uint64_t> expired_before_materialize_{0};
};

}  // namespace meridian::serve
