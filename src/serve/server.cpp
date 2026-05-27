#include "server.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <utility>

namespace meridian::serve {
namespace {

using Clock = Deadline::Clock;

uint64_t NanosSince(Clock::time_point start) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}

}  // namespace

LatencySamples::LatencySamples(size_t capacity) : capacity_(capacity == 0 ? 1 : capacity) {
    samples_.reserve(std::min<size_t>(capacity_, 4096));
}

uint64_t LatencySamples::NextRandom() {
    // xorshift64*. Small, fast, and good enough for choosing a reservoir slot; a
    // Mersenne twister per server would be a few kilobytes of state for no gain.
    rng_state_ ^= rng_state_ >> 12;
    rng_state_ ^= rng_state_ << 25;
    rng_state_ ^= rng_state_ >> 27;
    return rng_state_ * 0x2545f4914f6cdd1dULL;
}

void LatencySamples::Record(uint64_t nanos) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++count_;
    max_ = std::max(max_, nanos);
    if (samples_.size() < capacity_) {
        samples_.push_back(nanos);
        return;
    }
    const uint64_t slot = NextRandom() % count_;
    if (slot < capacity_) samples_[static_cast<size_t>(slot)] = nanos;
}

uint64_t LatencySamples::Percentile(double p) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (samples_.empty()) return 0;
    std::vector<uint64_t> sorted = samples_;
    std::sort(sorted.begin(), sorted.end());

    const double clamped = std::min(1.0, std::max(0.0, p));
    // ceil(p*n) - 1, clamped: the nearest-rank convention. For p=0 this is index 0.
    size_t rank = static_cast<size_t>(std::ceil(clamped * static_cast<double>(sorted.size())));
    if (rank == 0) rank = 1;
    return sorted[rank - 1];
}

uint64_t LatencySamples::count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return count_;
}

uint64_t LatencySamples::max() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return max_;
}

size_t LatencySamples::stored() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return samples_.size();
}

void LatencySamples::Reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    samples_.clear();
    count_ = 0;
    max_ = 0;
}

QueryServer::QueryServer(ServerOptions options, const GenerationSource* generations,
                         const Pipeline* pipeline)
    : options_(options),
      generations_(generations),
      pipeline_(pipeline),
      pool_(std::make_unique<TaskPool>(options.worker_threads == 0 ? 1 : options.worker_threads,
                                       options.queue_capacity)),
      latency_(options.latency_samples) {}

QueryServer::~QueryServer() { Shutdown(); }

void QueryServer::Shutdown() {
    if (pool_) pool_->Shutdown();
}

std::future<Result<SearchResponse>> QueryServer::Submit(const Query& query) {
    auto promise = std::make_shared<std::promise<Result<SearchResponse>>>();
    auto future = promise->get_future();

    // The budget starts here, at arrival, so queue wait is charged to it.
    const Deadline deadline = Deadline::AfterMillis(query.deadline_ms);

    if (!generations_ || !pipeline_) {
        promise->set_value(Status::Internal("server has no generation source or pipeline"));
        return future;
    }

    Query copy = query;
    const size_t pending = pool_->pending();
    uint64_t observed = queue_high_water_.load(std::memory_order_relaxed);
    while (pending > observed &&
           !queue_high_water_.compare_exchange_weak(observed, pending, std::memory_order_relaxed)) {
    }

    // in_flight is incremented before the post, not after. A worker can start and
    // finish the task before this thread resumes, and decrementing an unsigned counter
    // that has not been incremented yet reads as four billion queries in flight.
    ++in_flight_;

    // Refusal, not queueing. Note what is *not* here: no retry, no brief wait for a
    // slot, no "just this once" overflow. Any of those is an unbounded queue with more
    // steps, and the bound is the feature.
    const bool posted = pool_->TryPost([this, copy, deadline, promise]() mutable {
        Execute(copy, deadline, promise);
    });

    if (!posted && pool_->stopping()) {
        --in_flight_;
        promise->set_value(Status::Internal("server is shut down and is not accepting queries"));
        return future;
    }
    if (!posted) {
        --in_flight_;
        ++rejected_;
        promise->set_value(Status::Overloaded(
            "query refused: " + std::to_string(options_.worker_threads) + " workers and " +
            std::to_string(options_.queue_capacity) + " queue slots are full"));
        return future;
    }
    ++accepted_;
    return future;
}

Result<SearchResponse> QueryServer::Search(const Query& query) {
    auto future = Submit(query);
    return future.get();
}

void QueryServer::Execute(const Query& query, Deadline deadline,
                          std::shared_ptr<std::promise<Result<SearchResponse>>> promise) {
    const auto start = Clock::now();

    // One pin for the whole query, acquired here rather than inside the pipeline so the
    // lifetime is visibly the lifetime of this function: the generation cannot be
    // retired under any stage.
    const GenerationPin pin = generations_->Acquire();

    QueryTrace partial;
    auto result = pipeline_->Run(query, pin, deadline, &partial);

    if (result.ok()) {
        // Latency is recorded from the worker's start rather than from arrival so the
        // samples describe the pipeline's service time. Queue wait is a separate concern
        // with its own metric (queue_high_water); folding the two together makes a
        // service-time regression and a capacity shortfall look identical.
        latency_.Record(NanosSince(start));
        ++completed_;
    } else {
        ++failed_;
        if (result.status().code() == Code::kDeadlineExceeded) {
            ++deadline_exceeded_;
            RecordDeadline(partial);
        }
    }
    --in_flight_;
    promise->set_value(std::move(result));
}

void QueryServer::RecordDeadline(const QueryTrace& partial) {
    // Derived from which stages recorded time rather than from the status message: a
    // metric that depends on string formatting breaks the first time someone improves
    // the wording.
    if (partial.parse_ns == 0) {
        ++expired_before_parse_;
    } else if (partial.merge_ns == 0) {
        ++expired_before_merge_;
    } else if (partial.rerank_ns == 0) {
        ++expired_before_rerank_;
    } else {
        ++expired_before_materialize_;
    }
}

QueryServer::Metrics QueryServer::metrics() const {
    Metrics m;
    m.accepted = accepted_.load();
    m.rejected_overloaded = rejected_.load();
    m.completed = completed_.load();
    m.failed = failed_.load();
    m.deadline_exceeded = deadline_exceeded_.load();
    m.queue_high_water = queue_high_water_.load();
    m.in_flight = in_flight_.load();
    m.p50_ns = latency_.Percentile(0.50);
    m.p95_ns = latency_.Percentile(0.95);
    m.p99_ns = latency_.Percentile(0.99);
    m.max_ns = latency_.max();
    m.deadlines.before_parse = expired_before_parse_.load();
    m.deadlines.before_merge = expired_before_merge_.load();
    m.deadlines.before_rerank = expired_before_rerank_.load();
    m.deadlines.before_materialize = expired_before_materialize_.load();
    return m;
}

}  // namespace meridian::serve
