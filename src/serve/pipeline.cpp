#include "pipeline.hpp"

#include <future>
#include <string>
#include <utility>

namespace meridian::serve {
namespace {

using Clock = Deadline::Clock;

uint64_t NanosSince(Clock::time_point start) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}

/// Message for a budget that ran out, naming the boundary it ran out at.
///
/// The stage name is the whole value of this string. "deadline exceeded" tells an
/// operator nothing they could act on; "deadline exceeded before rerank" says the
/// candidate generation ate the budget, which is a different fix from the reranker
/// being slow.
Status Expired(const char* boundary, Clock::time_point start, uint32_t budget_ms) {
    const double elapsed_ms = static_cast<double>(NanosSince(start)) / 1e6;
    // No "deadline exceeded" in the text: Status::ToString already prefixes the code, and
    // repeating it produces "deadline exceeded: deadline exceeded before merge".
    return Status::DeadlineExceeded("expired before " + std::string(boundary) + ": " +
                                   std::to_string(elapsed_ms) + "ms elapsed of " +
                                   std::to_string(budget_ms) + "ms budget");
}

}  // namespace

Pipeline::Pipeline(PipelineStages stages, PipelineOptions options, QueryResultCache* result_cache,
                   PostingListCache* posting_cache)
    : stages_(stages),
      options_(options),
      result_cache_(result_cache),
      posting_cache_(posting_cache) {
    if (options_.parallel_paths && options_.path_threads > 0) {
        path_pool_ = std::make_unique<TaskPool>(options_.path_threads, options_.path_queue);
    }
}

Pipeline::Metrics Pipeline::metrics() const {
    Metrics m;
    m.queries = queries_.load();
    m.result_cache_hits = result_cache_hits_.load();
    m.parallel_paths_used = parallel_used_.load();
    m.parallel_paths_declined = parallel_declined_.load();
    m.deadline_exceeded = deadline_exceeded_.load();
    return m;
}

Result<SearchResponse> Pipeline::Run(const Query& request, const GenerationPin& pin,
                                    Deadline deadline, QueryTrace* partial_trace) const {
    ++queries_;
    const auto started = Clock::now();

    QueryTrace trace;
    trace.generation_version = pin.version;

    // Publishing the trace on every exit path, including the failures. A failed query
    // is the one whose timing is most worth having.
    struct TracePublisher {
        QueryTrace* out;
        const QueryTrace* trace;
        ~TracePublisher() {
            if (out) *out = *trace;
        }
    } publisher{partial_trace, &trace};

    if (!stages_.parse || !stages_.lexical || !stages_.merge || !stages_.rerank ||
        !stages_.materialize) {
        return Status::Internal("pipeline is missing a required stage");
    }
    if (!pin.valid()) {
        return Status::Internal("query ran without a pinned generation");
    }

    // Boundary zero. A query can arrive already over budget — it sat in the server's
    // queue, or the caller sent a budget it was never going to meet — and refusing here
    // costs nothing. Doing the parse first and then refusing would spend work on an
    // answer nobody can use.
    if (deadline.expired()) {
        ++deadline_exceeded_;
        return Expired("parse", started, request.deadline_ms);
    }

    const auto parse_start = Clock::now();
    Query query;
    {
        auto result = stages_.parse->Parse(request);
        if (!result.ok()) return result.status();
        query = result.take();
    }

    // The result cache key is derived from the parse output, so the lookup is attributed
    // to the parse stage. There is no cache field in the frozen QueryTrace, and charging
    // the lookup to lexical_ns would corrupt the one number the decomposition exists to
    // provide.
    std::optional<ResultCacheKey> cache_key;
    if (options_.use_result_cache && result_cache_) {
        cache_key = MakeResultCacheKey(query, pin.version);
        if (auto cached = result_cache_->Lookup(*cache_key); cached.has_value() && *cached) {
            ++result_cache_hits_;
            SearchResponse response = **cached;

            // The hits are returned exactly as cached; the trace is replaced, not
            // inherited. Reporting the original execution's lexical_ns on a cache hit
            // would mean the per-stage numbers describe work this query did not do,
            // which makes every latency aggregate built on them fiction.
            trace.parse_ns = NanosSince(parse_start);
            trace.lexical_candidates = response.trace.lexical_candidates;
            trace.semantic_candidates = response.trace.semantic_candidates;
            trace.merged_candidates = response.trace.merged_candidates;
            trace.cache_hit_results = true;
            trace.total_ns = NanosSince(started);
            response.trace = trace;
            return response;
        }
    }
    trace.parse_ns = NanosSince(parse_start);

    StageCounters lexical_counters;
    StageCounters semantic_counters;

    StageContext lexical_ctx;
    lexical_ctx.pin = &pin;
    lexical_ctx.posting_cache = posting_cache_;
    lexical_ctx.deadline = deadline;
    lexical_ctx.counters = &lexical_counters;

    StageContext semantic_ctx = lexical_ctx;
    semantic_ctx.counters = &semantic_counters;

    const bool run_semantic = stages_.semantic != nullptr && query.embedding.has_value();

    std::vector<Candidate> semantic_candidates;
    Status semantic_status = Status::Ok();
    uint64_t ann_ns = 0;

    auto run_semantic_path = [&] {
        const auto start = Clock::now();
        auto result = stages_.semantic->Search(semantic_ctx, query);
        ann_ns = NanosSince(start);
        if (result.ok()) {
            semantic_candidates = result.take();
        } else {
            semantic_status = result.status();
        }
    };

    // Offload the semantic path and keep the lexical path on this thread. Which path
    // gets offloaded is arbitrary for correctness and not for cost: keeping one inline
    // means a lexical-only query never touches the pool at all.
    std::future<void> semantic_future;
    bool offloaded = false;
    if (run_semantic && path_pool_) {
        auto task = std::make_shared<std::packaged_task<void()>>(run_semantic_path);
        semantic_future = task->get_future();
        if (path_pool_->TryPost([task] { (*task)(); })) {
            offloaded = true;
            ++parallel_used_;
        } else {
            ++parallel_declined_;
        }
    }

    const auto lexical_start = Clock::now();
    std::vector<Candidate> lexical_candidates;
    {
        auto result = stages_.lexical->Retrieve(lexical_ctx, query);
        trace.lexical_ns = NanosSince(lexical_start);
        if (!result.ok()) {
            // The offloaded path is still running and holds references to locals. It
            // must be joined before unwinding, whatever the lexical path said.
            if (offloaded) semantic_future.wait();
            return result.status();
        }
        lexical_candidates = result.take();
    }

    if (offloaded) {
        semantic_future.wait();
    } else if (run_semantic) {
        run_semantic_path();
    }
    if (!semantic_status.ok()) return semantic_status;

    trace.ann_ns = ann_ns;
    trace.lexical_candidates = static_cast<uint32_t>(lexical_candidates.size());
    trace.semantic_candidates = static_cast<uint32_t>(semantic_candidates.size());
    trace.cache_hit_postings = lexical_counters.posting_cache_hit;

    if (deadline.expired()) {
        ++deadline_exceeded_;
        return Expired("merge", started, request.deadline_ms);
    }

    const auto merge_start = Clock::now();
    auto merged = stages_.merge->Merge(lexical_candidates, semantic_candidates, query.candidate_k);
    trace.merge_ns = NanosSince(merge_start);
    trace.merged_candidates = static_cast<uint32_t>(merged.size());

    if (deadline.expired()) {
        ++deadline_exceeded_;
        return Expired("rerank", started, request.deadline_ms);
    }

    const auto rerank_start = Clock::now();
    std::vector<Hit> hits;
    {
        auto result = stages_.rerank->Rank(lexical_ctx, query, merged);
        trace.rerank_ns = NanosSince(rerank_start);
        if (!result.ok()) return result.status();
        hits = result.take();
    }

    // Truncation happens here rather than before reranking: the reranker is given
    // candidate_k documents precisely so it has something to reorder, and cutting to
    // top_k first would make the second stage a no-op.
    if (hits.size() > query.top_k) hits.resize(query.top_k);

    if (deadline.expired()) {
        ++deadline_exceeded_;
        return Expired("materialize", started, request.deadline_ms);
    }

    const auto materialize_start = Clock::now();
    if (Status status = stages_.materialize->Materialize(lexical_ctx, query, hits); !status.ok()) {
        trace.materialize_ns = NanosSince(materialize_start);
        return status;
    }
    trace.materialize_ns = NanosSince(materialize_start);

    // No deadline check after the last stage. A finished answer is never discarded for
    // being late: the caller already paid for it, and the trace reports the overrun
    // honestly. Throwing it away would turn a slow success into a failure and lose the
    // only evidence of where the time went.
    trace.total_ns = NanosSince(started);

    SearchResponse response;
    response.hits = std::move(hits);
    response.total_candidates = trace.merged_candidates;
    response.trace = trace;

    if (cache_key) {
        result_cache_->Insert(*cache_key, std::make_shared<const SearchResponse>(response));
    }
    return response;
}

}  // namespace meridian::serve
