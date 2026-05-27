#pragma once

// The query pipeline: parse, retrieve, merge, rerank, materialize.
//
// Two properties shape this file.
//
// **One pinned generation.** The pin is acquired once, before parsing, and every stage
// reads it. Re-acquiring per stage would let a generation be published between
// retrieval and reranking, so the reranker would score documents that the retriever
// never saw and resolve ids against a segment set that no longer matches — duplicates
// or missing documents, depending on which way the merge went. Pinning once also makes
// the trace's generation_version meaningful: it names the single index state the whole
// answer came from.
//
// **Deadlines are checked at stage boundaries.** Not inside stages, because a check in
// an inner loop costs more than it saves and gets wrong answers anyway (a stage
// abandoned halfway has produced a candidate set that is not a prefix of anything).
// Between stages the state is a complete, discardable intermediate, which is the only
// place a cheap check is also a correct one.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "cache.hpp"
#include "meridian/index.hpp"
#include "meridian/retrieval.hpp"
#include "meridian/status.hpp"
#include "task_pool.hpp"

namespace meridian::serve {

/// A query's hold on one index state.
///
/// `generation` is the engine's view, kept as a shared_ptr for exactly the reason
/// index.hpp gives: a merge may publish a replacement and then wait for old readers to
/// drain, and this handle is what it waits on.
///
/// `attachment` carries an equivalent snapshot from a source that is not the engine's
/// `Index` — the in-memory corpus the CLIs and these tests run on, which publishes
/// generations and runs merges on demand without a filesystem. The semantics are the same
/// ones index.hpp specifies (immutable, versioned, published atomically, held by
/// reference for the life of the query); only the type differs. A stage reads exactly one
/// of the two fields and must never assume the other is populated; `EngineGeneration` in
/// stages_engine.cpp and `SnapshotFrom` in reference_index.cpp are the two places that
/// decide which, and both fail loudly rather than returning an empty candidate list.
struct GenerationPin {
    std::shared_ptr<Generation> generation;
    std::shared_ptr<const void> attachment;
    uint64_t version = 0;

    bool valid() const { return generation != nullptr || attachment != nullptr; }
};

/// Where a query gets its generation. An interface rather than a direct `Index&` so the
/// serving layer can be driven by the real index, by a fixture that republishes on a
/// background thread, or by a corpus held in memory, without any of them being a special
/// case inside the pipeline.
class GenerationSource {
public:
    virtual ~GenerationSource() = default;
    virtual GenerationPin Acquire() const = 0;
};

/// A monotonic deadline.
///
/// steady_clock, not system_clock: a wall-clock adjustment must not extend or expire a
/// query's budget, and on a machine that runs NTP that is not a theoretical concern.
class Deadline {
public:
    using Clock = std::chrono::steady_clock;

    Deadline() = default;
    explicit Deadline(Clock::time_point expiry) : expiry_(expiry) {}

    static Deadline AfterMillis(uint32_t ms) {
        return Deadline(Clock::now() + std::chrono::milliseconds(ms));
    }
    static Deadline Never() { return Deadline(Clock::time_point::max()); }

    bool expired() const { return Clock::now() >= expiry_; }
    Clock::time_point expiry() const { return expiry_; }

    int64_t remaining_us() const {
        return std::chrono::duration_cast<std::chrono::microseconds>(expiry_ - Clock::now())
            .count();
    }

private:
    Clock::time_point expiry_ = Clock::time_point::max();
};

/// Counters a stage reports back. Separate from QueryTrace because the lexical and
/// semantic paths may run concurrently, and two threads writing into one QueryTrace is
/// a data race no matter how benign the fields look. Each path fills its own copy and
/// the pipeline folds them in after the join.
struct StageCounters {
    bool posting_cache_hit = false;
    uint64_t postings_decoded = 0;
};

/// What a stage is given. The pin is a pointer rather than a reference so the struct
/// stays assignable, which the parallel path needs when it captures a copy.
struct StageContext {
    const GenerationPin* pin = nullptr;
    PostingListCache* posting_cache = nullptr;
    Deadline deadline;
    StageCounters* counters = nullptr;
};

class ParseStage {
public:
    virtual ~ParseStage() = default;

    /// Normalizes `request.raw` into terms, preserving the request's non-text fields.
    /// Takes the whole Query so a parser can also interpret mode-changing syntax.
    virtual Result<Query> Parse(const Query& request) const = 0;
};

class LexicalStage {
public:
    virtual ~LexicalStage() = default;
    virtual Result<std::vector<Candidate>> Retrieve(const StageContext& ctx,
                                                    const Query& query) const = 0;
};

class SemanticStage {
public:
    virtual ~SemanticStage() = default;
    virtual Result<std::vector<Candidate>> Search(const StageContext& ctx,
                                                  const Query& query) const = 0;
};

class MergeStage {
public:
    virtual ~MergeStage() = default;
    virtual std::vector<Candidate> Merge(const std::vector<Candidate>& lexical,
                                        const std::vector<Candidate>& semantic,
                                        uint32_t candidate_k) const = 0;
};

class RerankStage {
public:
    virtual ~RerankStage() = default;
    virtual Result<std::vector<Hit>> Rank(const StageContext& ctx, const Query& query,
                                          const std::vector<Candidate>& candidates) const = 0;
};

class MaterializeStage {
public:
    virtual ~MaterializeStage() = default;

    /// Fills external ids and anything else that needs a segment read. Last so that it
    /// touches only the top_k documents actually being returned.
    virtual Status Materialize(const StageContext& ctx, const Query& query,
                               std::vector<Hit>& hits) const = 0;
};

struct PipelineStages {
    const ParseStage* parse = nullptr;
    const LexicalStage* lexical = nullptr;
    const SemanticStage* semantic = nullptr;  // optional: absent means lexical-only
    const MergeStage* merge = nullptr;
    const RerankStage* rerank = nullptr;
    const MaterializeStage* materialize = nullptr;
};

struct PipelineOptions {
    /// Run the lexical and semantic paths concurrently.
    ///
    /// Safe because of the pin: both paths only *read* an immutable generation, they
    /// share no mutable state (each gets its own StageCounters and writes its own
    /// candidate vector), and the posting cache they both may touch is internally
    /// sharded and thread-safe. Nothing either path does is visible to the other, so
    /// there is no ordering to get wrong.
    ///
    /// Whether it helps is a separate question from whether it is safe, and the answer
    /// depends on the ratio between the two paths' costs and the hand-off cost. See the
    /// measurement in tests/serve_test.cpp: with both paths in the single-digit
    /// microseconds the hand-off dominates and this is a small loss; once either path
    /// costs more than roughly 50us the overlap wins, and it approaches
    /// max(lexical, ann) instead of their sum.
    bool parallel_paths = true;

    /// Threads for the offloaded path. Bounded, and when they are all busy the pipeline
    /// runs the path inline rather than waiting — a saturated helper pool must degrade
    /// to serial execution, not add queueing delay to a query that already has a budget.
    size_t path_threads = 2;
    size_t path_queue = 128;

    /// Consult and populate the result cache. Turned off by the CLI's cold-latency mode
    /// and by benchmarks that are measuring the engine rather than the cache.
    bool use_result_cache = true;
};

/// Wires the stages together and fills the trace.
class Pipeline {
public:
    Pipeline(PipelineStages stages, PipelineOptions options, QueryResultCache* result_cache,
             PostingListCache* posting_cache);

    /// Runs one query against one pinned generation.
    ///
    /// On failure — including a deadline — `partial_trace`, when provided, holds the
    /// stages that did run. That is deliberately the *only* thing a failed query
    /// reports: see the deadline note on QueryServer for why no hits come back with it.
    Result<SearchResponse> Run(const Query& request, const GenerationPin& pin, Deadline deadline,
                               QueryTrace* partial_trace = nullptr) const;

    struct Metrics {
        uint64_t queries = 0;
        uint64_t result_cache_hits = 0;
        uint64_t parallel_paths_used = 0;

        /// Times the helper pool was full and the semantic path ran inline. A number
        /// that is not near zero means path_threads is undersized for the offered load.
        uint64_t parallel_paths_declined = 0;
        uint64_t deadline_exceeded = 0;
    };
    Metrics metrics() const;

    const PipelineOptions& options() const { return options_; }

private:
    PipelineStages stages_;
    PipelineOptions options_;
    QueryResultCache* result_cache_;
    PostingListCache* posting_cache_;
    std::unique_ptr<TaskPool> path_pool_;

    mutable std::atomic<uint64_t> queries_{0};
    mutable std::atomic<uint64_t> result_cache_hits_{0};
    mutable std::atomic<uint64_t> parallel_used_{0};
    mutable std::atomic<uint64_t> parallel_declined_{0};
    mutable std::atomic<uint64_t> deadline_exceeded_{0};
};

}  // namespace meridian::serve
