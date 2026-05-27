// Serving layer tests: deadlines, admission control, the trace, pinning, and
// percentiles.
//
// Two of these are the ones worth reading. The deadline test asserts that an impossible
// budget produces a *failure* and not a short result list, because a result list that is
// quietly three documents shorter than requested looks exactly like a correct answer.
// The overload test asserts that offered load above capacity turns into refusals rather
// than into latency, and that the queries that were accepted did not get slower because
// of the ones that were not.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "../src/serve/reference_index.hpp"
#include "../src/serve/server.hpp"
#include "testing.hpp"

using namespace meridian;
using namespace meridian::serve;
using Clock = std::chrono::steady_clock;

namespace {

double Millis(uint64_t nanos) { return static_cast<double>(nanos) / 1e6; }

double MillisSince(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

ReferenceIndex::Document MakeDocument(const std::string& id, int serial, size_t embedding_dim = 0) {
    ReferenceIndex::Document doc;
    doc.meta.external_id = id;
    doc.meta.language = "en";
    doc.meta.category = (serial % 2 == 0) ? "systems" : "ir";
    doc.meta.popularity = static_cast<double>(serial % 10) / 10.0;
    doc.tokens[Field::kTitle] = Tokenize("segment merge generation " + std::to_string(serial));
    doc.tokens[Field::kBody] =
        Tokenize("an inverted index maps each term to the documents containing it number " +
                 std::to_string(serial));
    doc.tokens[Field::kTags] = Tokenize("index postings serving");
    if (embedding_dim > 0) {
        doc.embedding.assign(embedding_dim, 0.0f);
        for (size_t i = 0; i < embedding_dim; ++i) {
            doc.embedding[i] = static_cast<float>(((serial * 31 + static_cast<int>(i) * 7) % 17)) /
                               17.0f;
        }
    }
    return doc;
}

void SeedCorpus(ReferenceIndex& index, int documents, int docs_per_segment = 100,
                size_t embedding_dim = 0) {
    for (int i = 0; i < documents; ++i) {
        index.Add(MakeDocument("doc-" + std::to_string(i), i, embedding_dim));
        if ((i + 1) % docs_per_segment == 0) index.Flush();
    }
    index.Flush();
}

Query MakeQuery(const std::string& text = "segment merge index", uint32_t top_k = 5,
                uint32_t deadline_ms = 60'000) {
    Query query;
    query.raw = text;
    query.top_k = top_k;
    query.candidate_k = 100;
    query.deadline_ms = deadline_ms;
    return query;
}

std::vector<float> MakeEmbedding(size_t dim) {
    std::vector<float> embedding(dim);
    for (size_t i = 0; i < dim; ++i) embedding[i] = static_cast<float>((i * 13) % 17) / 17.0f;
    return embedding;
}

/// Wraps a stage and makes it cost a known amount of time. Sleeping rather than spinning:
/// these tests run alongside each other and on CI machines, and a spin loop would make
/// the measurement depend on how many cores happened to be free.
class SlowLexicalStage : public LexicalStage {
public:
    SlowLexicalStage(const LexicalStage* inner, std::chrono::microseconds delay)
        : inner_(inner), delay_(delay) {}

    Result<std::vector<Candidate>> Retrieve(const StageContext& ctx,
                                            const Query& query) const override {
        std::this_thread::sleep_for(delay_);
        return inner_->Retrieve(ctx, query);
    }

private:
    const LexicalStage* inner_;
    std::chrono::microseconds delay_;
};

class SlowSemanticStage : public SemanticStage {
public:
    SlowSemanticStage(const SemanticStage* inner, std::chrono::microseconds delay)
        : inner_(inner), delay_(delay) {}

    Result<std::vector<Candidate>> Search(const StageContext& ctx,
                                          const Query& query) const override {
        std::this_thread::sleep_for(delay_);
        return inner_->Search(ctx, query);
    }

private:
    const SemanticStage* inner_;
    std::chrono::microseconds delay_;
};

/// Publishes a new generation from inside the lexical stage. The only honest way to test
/// that the pin holds: the generation has to change while a query is between stages.
class RepublishingLexicalStage : public LexicalStage {
public:
    RepublishingLexicalStage(const LexicalStage* inner, ReferenceIndex* index)
        : inner_(inner), index_(index) {}

    Result<std::vector<Candidate>> Retrieve(const StageContext& ctx,
                                            const Query& query) const override {
        if (!published_.exchange(true)) {
            index_->Add(MakeDocument("mid-query", 999));
            published_version_ = index_->Flush();
        }
        return inner_->Retrieve(ctx, query);
    }

    uint64_t published_version() const { return published_version_; }

private:
    const LexicalStage* inner_;
    ReferenceIndex* index_;
    mutable std::atomic<bool> published_{false};
    mutable uint64_t published_version_ = 0;
};

}  // namespace

MERIDIAN_TEST(TraceIsFilledForEveryStageAndSumsToTheTotal) {
    PipelineOptions options;
    options.parallel_paths = false;  // overlapping paths make the stages sum to more than the total
    options.use_result_cache = false;
    ReferenceEngine engine(options);
    SeedCorpus(engine.index, 300, 100, 16);

    Query query = MakeQuery();
    query.embedding = MakeEmbedding(16);

    const GenerationPin pin = engine.index.Acquire();
    auto result = engine.pipeline.Run(query, pin, Deadline::Never());
    CHECK(result.ok());

    const QueryTrace& trace = result.value().trace;
    CHECK(trace.parse_ns > 0);
    CHECK(trace.lexical_ns > 0);
    CHECK(trace.ann_ns > 0);
    CHECK(trace.merge_ns > 0);
    CHECK(trace.rerank_ns > 0);
    CHECK(trace.materialize_ns > 0);
    CHECK(trace.total_ns > 0);
    CHECK_EQ(trace.generation_version, pin.version);
    CHECK(trace.lexical_candidates > 0);
    CHECK(trace.semantic_candidates > 0);
    CHECK(trace.merged_candidates > 0);

    const uint64_t accounted = trace.parse_ns + trace.lexical_ns + trace.ann_ns + trace.merge_ns +
                               trace.rerank_ns + trace.materialize_ns;
    CHECK(accounted <= trace.total_ns);
    const double share = static_cast<double>(accounted) / static_cast<double>(trace.total_ns);
    // The unaccounted remainder is the pin, the cache key and the trace bookkeeping. If it
    // ever grows past a tenth of the query, the decomposition has stopped being useful and
    // something unmeasured has become expensive.
    CHECK(share > 0.90);
    std::printf("      stages account for %.1f%% of %.3f ms\n", share * 100.0,
                Millis(trace.total_ns));
}

MERIDIAN_TEST(ParallelPathsOverlapAndTheTraceSaysSo) {
    const auto delay = std::chrono::milliseconds(20);

    auto run = [&](bool parallel) {
        PipelineOptions options;
        options.parallel_paths = parallel;
        options.use_result_cache = false;
        ReferenceEngine engine(options);
        SeedCorpus(engine.index, 100, 100, 16);

        SlowLexicalStage slow_lexical(&engine.lexical,
                                      std::chrono::duration_cast<std::chrono::microseconds>(delay));
        SlowSemanticStage slow_semantic(
            &engine.semantic, std::chrono::duration_cast<std::chrono::microseconds>(delay));
        PipelineStages stages{&engine.parse,  &slow_lexical, &slow_semantic,
                              &engine.merge, &engine.rerank, &engine.materialize};
        Pipeline pipeline(stages, options, nullptr, &engine.posting_cache);

        Query query = MakeQuery();
        query.embedding = MakeEmbedding(16);
        auto result = pipeline.Run(query, engine.index.Acquire(), Deadline::Never());
        CHECK(result.ok());
        return result.value().trace;
    };

    const QueryTrace serial = run(false);
    const QueryTrace parallel = run(true);

    const uint64_t serial_stages = serial.lexical_ns + serial.ann_ns;
    const uint64_t parallel_stages = parallel.lexical_ns + parallel.ann_ns;

    std::printf("      two %lldms paths: serial total %.1f ms, parallel total %.1f ms\n",
                static_cast<long long>(delay.count()), Millis(serial.total_ns),
                Millis(parallel.total_ns));

    // Safe because both paths only read the pinned, immutable generation; useful because
    // the wall time approaches max(lexical, ann) instead of their sum.
    CHECK(serial.total_ns >= serial_stages);
    CHECK(parallel.total_ns < serial.total_ns * 3 / 4);
    CHECK(parallel_stages > parallel.total_ns);  // the stage times overlap, by construction
}

MERIDIAN_TEST(ParallelHandoffCostOnACheapQuery) {
    // The same mechanism measured where it does not pay: with both paths costing a few
    // microseconds, posting a task and waiting on a future is the dominant term. Reported
    // rather than asserted, because a timing assertion at this scale would be a flaky test
    // rather than a finding.
    auto measure = [](bool parallel) {
        PipelineOptions options;
        options.parallel_paths = parallel;
        options.use_result_cache = false;
        ReferenceEngine engine(options);
        SeedCorpus(engine.index, 200, 200, 16);

        Query query = MakeQuery();
        query.embedding = MakeEmbedding(16);
        const GenerationPin pin = engine.index.Acquire();

        std::vector<uint64_t> totals;
        for (int i = 0; i < 200; ++i) {
            auto result = engine.pipeline.Run(query, pin, Deadline::Never());
            CHECK(result.ok());
            totals.push_back(result.value().trace.total_ns);
        }
        std::sort(totals.begin(), totals.end());
        return totals[totals.size() / 2];
    };

    const uint64_t serial = measure(false);
    const uint64_t parallel = measure(true);
    std::printf("      cheap query median total: serial %.3f ms, parallel %.3f ms (hand-off "
                "costs %.3f ms)\n",
                Millis(serial), Millis(parallel),
                Millis(parallel > serial ? parallel - serial : 0));
    CHECK(serial > 0);
    CHECK(parallel > 0);
}

// Offered load beyond the limit must become refusals, not latency.
MERIDIAN_TEST(OverloadIsRefusedAndAcceptedQueriesStayFast) {
    const auto service_time = std::chrono::milliseconds(25);

    PipelineOptions options;
    options.parallel_paths = false;
    options.use_result_cache = false;
    ReferenceEngine engine(options);
    SeedCorpus(engine.index, 50, 50);

    SlowLexicalStage slow(&engine.lexical,
                          std::chrono::duration_cast<std::chrono::microseconds>(service_time));
    PipelineStages stages{&engine.parse,  &slow,           nullptr,
                          &engine.merge, &engine.rerank, &engine.materialize};
    Pipeline pipeline(stages, options, nullptr, &engine.posting_cache);

    ServerOptions server_options;
    server_options.worker_threads = 1;
    server_options.queue_capacity = 2;
    QueryServer server(server_options, &engine.index, &pipeline);

    const int offered = 60;
    const Query query = MakeQuery("segment merge index", 5, 60'000);

    std::vector<std::future<Result<SearchResponse>>> futures;
    std::vector<double> submit_latencies;
    futures.reserve(offered);
    const auto wall_start = Clock::now();
    for (int i = 0; i < offered; ++i) {
        const auto submit_start = Clock::now();
        futures.push_back(server.Submit(query));
        submit_latencies.push_back(MillisSince(submit_start));
    }

    int accepted = 0, refused = 0;
    double worst_accepted_ms = 0.0;
    for (auto& future : futures) {
        const auto before = Clock::now();
        auto result = future.get();
        const double waited = MillisSince(before);
        if (result.ok()) {
            ++accepted;
            worst_accepted_ms = std::max(worst_accepted_ms, waited);
            CHECK_EQ(result.value().hits.size(), 5u);
        } else {
            CHECK_EQ(static_cast<int>(result.status().code()), static_cast<int>(Code::kOverloaded));
            ++refused;
        }
    }
    const double wall_ms = MillisSince(wall_start);

    std::printf("      offered %d, accepted %d, refused %d in %.1f ms\n", offered, accepted,
                refused, wall_ms);

    // The refusals are the point: with one worker and two queue slots, most of this load
    // has nowhere to go and is told so.
    CHECK(refused > 40);
    CHECK(accepted >= 1);
    CHECK_EQ(accepted + refused, offered);
    CHECK_EQ(server.metrics().rejected_overloaded, static_cast<uint64_t>(refused));

    // Submitting never blocks, so a refusal costs the caller nothing.
    const double worst_submit = *std::max_element(submit_latencies.begin(), submit_latencies.end());
    CHECK(worst_submit < 20.0);

    // And the queries that were accepted were not slowed down by the ones that were not.
    // Worst case for an accepted query is its own service time plus the queue ahead of it:
    // (queue_capacity + 1) * service_time. That bound does not move when the offered load
    // triples, which is exactly what an unbounded queue would fail to give.
    const double bound = static_cast<double>((server_options.queue_capacity + 1) *
                                            static_cast<size_t>(service_time.count())) * 4.0;
    std::printf("      worst accepted wait %.1f ms, bound %.1f ms\n", worst_accepted_ms, bound);
    CHECK(worst_accepted_ms < bound);
}

// An impossible budget must fail, and must not come back with a short list of hits.
MERIDIAN_TEST(ImpossibleDeadlineIsRefusedAndReturnsNoHits) {
    PipelineOptions options;
    options.parallel_paths = false;
    ReferenceEngine engine(options);
    SeedCorpus(engine.index, 100, 100);

    QueryServer server(ServerOptions{}, &engine.index, &engine.pipeline);

    Query impossible = MakeQuery("segment merge index", 5, 0);
    auto result = server.Search(impossible);

    CHECK(!result.ok());
    CHECK_EQ(static_cast<int>(result.status().code()), static_cast<int>(Code::kDeadlineExceeded));
    // Result<T> holds either a response or a status, so there is no hit list to be short:
    // the refusal is structural rather than a convention a caller has to remember.
    CHECK(result.status().message().find("expired before parse") != std::string::npos);

    const auto metrics = server.metrics();
    CHECK_EQ(metrics.deadline_exceeded, 1u);
    CHECK_EQ(metrics.deadlines.before_parse, 1u);
    CHECK_EQ(metrics.completed, 0u);

    // The same query with a real budget returns exactly what was asked for, which is what
    // makes the failure above a refusal and not an empty index.
    const Query generous = MakeQuery("segment merge index", 5, 60'000);
    auto good = server.Search(generous);
    CHECK(good.ok());
    CHECK_EQ(good.value().hits.size(), 5u);
}

MERIDIAN_TEST(DeadlineExpiringMidQueryNamesTheStageAndReturnsNothing) {
    PipelineOptions options;
    options.parallel_paths = false;
    options.use_result_cache = false;
    ReferenceEngine engine(options);
    SeedCorpus(engine.index, 100, 100);

    SlowLexicalStage slow(&engine.lexical, std::chrono::milliseconds(40));
    PipelineStages stages{&engine.parse,  &slow,           nullptr,
                          &engine.merge, &engine.rerank, &engine.materialize};
    Pipeline pipeline(stages, options, nullptr, &engine.posting_cache);
    QueryServer server(ServerOptions{}, &engine.index, &pipeline);

    const Query query = MakeQuery("segment merge index", 5, 5);
    auto result = server.Search(query);

    CHECK(!result.ok());
    CHECK_EQ(static_cast<int>(result.status().code()), static_cast<int>(Code::kDeadlineExceeded));
    // The stage boundary is in the message, so an operator can tell a slow retriever from
    // a slow reranker without reproducing the query.
    CHECK(result.status().message().find("before merge") != std::string::npos);

    const auto metrics = server.metrics();
    CHECK_EQ(metrics.deadline_exceeded, 1u);
    CHECK_EQ(metrics.deadlines.before_merge, 1u);
    CHECK_EQ(metrics.deadlines.before_parse, 0u);
    // No latency sample: percentiles describe successful queries, and a refused query's
    // duration would flatter them.
    CHECK_EQ(server.latency().count(), 0u);
}

MERIDIAN_TEST(GenerationIsPinnedForTheWholeQuery) {
    PipelineOptions options;
    options.parallel_paths = false;
    options.use_result_cache = false;
    ReferenceEngine engine(options);
    SeedCorpus(engine.index, 50, 50);

    const uint64_t before = engine.index.Current()->version;

    RepublishingLexicalStage republisher(&engine.lexical, &engine.index);
    PipelineStages stages{&engine.parse,  &republisher,    nullptr,
                          &engine.merge, &engine.rerank, &engine.materialize};
    Pipeline pipeline(stages, options, nullptr, &engine.posting_cache);
    QueryServer server(ServerOptions{}, &engine.index, &pipeline);

    auto result = server.Search(MakeQuery("segment merge generation", 10, 60'000));
    CHECK(result.ok());

    CHECK(republisher.published_version() > before);
    CHECK(engine.index.Current()->version > before);
    // The query ran to completion against the generation it started with. Re-acquiring per
    // stage would have let the reranker score candidates from an index the retriever never
    // saw.
    CHECK_EQ(result.value().trace.generation_version, before);
    for (const auto& hit : result.value().hits) {
        CHECK_NE(hit.external_id, std::string("mid-query"));
    }

    // The next query sees the new state, so the pin delays visibility by exactly one query
    // and no longer.
    auto next = server.Search(MakeQuery("segment merge generation", 10, 60'000));
    CHECK(next.ok());
    CHECK_EQ(next.value().trace.generation_version, engine.index.Current()->version);
}

// A real merge, running in another thread, while queries run.
MERIDIAN_TEST(QueriesStayConsistentWhileTheIndexIsMerging) {
    PipelineOptions options;
    options.parallel_paths = true;
    options.use_result_cache = true;
    ReferenceEngine engine(options);
    SeedCorpus(engine.index, 200, 20);  // ten segments, so a merge has real work to do

    ServerOptions server_options;
    server_options.worker_threads = 4;
    server_options.queue_capacity = 256;
    QueryServer server(server_options, &engine.index, &engine.pipeline);

    std::atomic<bool> stop{false};
    std::atomic<uint64_t> merges{0};
    std::atomic<uint64_t> flushes{0};

    std::thread mutator([&] {
        int serial = 1000;
        while (!stop.load()) {
            engine.index.Add(MakeDocument("live-" + std::to_string(serial), serial));
            engine.index.Flush();
            ++flushes;
            if (serial % 3 == 0) {
                engine.index.Merge();
                ++merges;
            }
            ++serial;
        }
    });

    std::atomic<uint64_t> queries{0};
    std::atomic<uint64_t> failures{0};
    std::vector<std::thread> readers;
    for (int t = 0; t < 4; ++t) {
        readers.emplace_back([&] {
            for (int i = 0; i < 150; ++i) {
                auto result = server.Search(MakeQuery("segment merge generation", 10, 60'000));
                if (!result.ok()) {
                    ++failures;
                    continue;
                }
                const SearchResponse& response = result.value();

                // Consistency, stated as three properties a torn segment set would break:
                // every hit resolves to an external id, no document appears twice, and the
                // scores are ordered.
                std::set<std::string> seen;
                double previous = std::numeric_limits<double>::infinity();
                for (const auto& hit : response.hits) {
                    if (hit.external_id.empty()) ++failures;
                    if (!seen.insert(hit.external_id).second) ++failures;
                    if (hit.score > previous + 1e-9) ++failures;
                    previous = hit.score;
                }
                if (response.hits.empty()) ++failures;
                if (response.trace.generation_version == 0) ++failures;
                ++queries;
            }
        });
    }

    for (auto& reader : readers) reader.join();
    stop.store(true);
    mutator.join();

    std::printf("      %llu queries against %llu flushes and %llu merges\n",
                static_cast<unsigned long long>(queries.load()),
                static_cast<unsigned long long>(flushes.load()),
                static_cast<unsigned long long>(merges.load()));
    CHECK_EQ(failures.load(), 0u);
    CHECK_EQ(queries.load(), 600u);
    CHECK(merges.load() > 0u);
}

MERIDIAN_TEST(PercentilesComeFromSamplesAndMatchAHandCheck) {
    LatencySamples samples(1000);
    // 1..100 ns. Nearest-rank: p50 is the 50th value, p95 the 95th, p99 the 99th.
    for (uint64_t i = 1; i <= 100; ++i) samples.Record(i);

    CHECK_EQ(samples.count(), 100u);
    CHECK_EQ(samples.stored(), 100u);
    CHECK_EQ(samples.Percentile(0.50), 50u);
    CHECK_EQ(samples.Percentile(0.95), 95u);
    CHECK_EQ(samples.Percentile(0.99), 99u);
    CHECK_EQ(samples.Percentile(1.00), 100u);
    CHECK_EQ(samples.max(), 100u);
}

MERIDIAN_TEST(TheMeanHidesTheTailThatThePercentilesShow) {
    LatencySamples samples(1000);
    std::vector<uint64_t> raw;
    // 2% slow. Deliberately above 1%: with exactly 1% in the tail, nearest-rank p99 lands
    // on the last fast sample, which is correct and makes a poor illustration.
    for (int i = 0; i < 980; ++i) raw.push_back(1'000);     // 1us
    for (int i = 0; i < 20; ++i) raw.push_back(500'000);    // 0.5ms
    for (uint64_t value : raw) samples.Record(value);

    const double mean = static_cast<double>(std::accumulate(raw.begin(), raw.end(), uint64_t{0})) /
                        static_cast<double>(raw.size());

    CHECK_EQ(samples.Percentile(0.50), 1'000u);
    CHECK_EQ(samples.Percentile(0.99), 500'000u);
    // The mean sits between the body and the tail, describing neither: it is eleven times
    // the typical query and a fiftieth of the worst one.
    CHECK(mean > 1'000.0);
    CHECK(mean < 500'000.0);
    std::printf("      mean %.0f ns, p50 %llu ns, p99 %llu ns\n", mean,
                static_cast<unsigned long long>(samples.Percentile(0.50)),
                static_cast<unsigned long long>(samples.Percentile(0.99)));
}

MERIDIAN_TEST(ReservoirKeepsPercentilesRepresentativeBeyondCapacity) {
    LatencySamples samples(1024);
    // 200k samples uniform over 1..1000 into a 1024-slot reservoir. A "keep the first N"
    // scheme would be fine here too, so the distribution is deliberately shifted halfway
    // through: the second half is drawn from a higher range, and only a uniform sample of
    // the whole stream reflects it.
    for (uint64_t i = 0; i < 100'000; ++i) samples.Record(1 + (i % 1000));
    for (uint64_t i = 0; i < 100'000; ++i) samples.Record(2'000 + (i % 1000));

    CHECK_EQ(samples.count(), 200'000u);
    CHECK_EQ(samples.stored(), 1024u);
    const uint64_t p50 = samples.Percentile(0.50);
    const uint64_t p99 = samples.Percentile(0.99);
    std::printf("      p50 %llu (true 1000/2000 boundary), p99 %llu (true ~2990)\n",
                static_cast<unsigned long long>(p50), static_cast<unsigned long long>(p99));
    // The true median of the combined stream is at the boundary between the two ranges.
    CHECK(p50 > 500u);
    CHECK(p50 < 2'600u);
    CHECK(p99 > 2'800u);
    CHECK(p99 <= 3'000u);
}

MERIDIAN_TEST(ServerMetricsAccountForEveryQuery) {
    PipelineOptions options;
    options.parallel_paths = false;
    ReferenceEngine engine(options);
    SeedCorpus(engine.index, 60, 30);
    QueryServer server(ServerOptions{}, &engine.index, &engine.pipeline);

    for (int i = 0; i < 20; ++i) {
        auto result = server.Search(MakeQuery("segment merge index " + std::to_string(i), 5, 60'000));
        CHECK(result.ok());
    }
    auto refused_budget = server.Search(MakeQuery("segment merge index", 5, 0));
    CHECK(!refused_budget.ok());

    const auto metrics = server.metrics();
    CHECK_EQ(metrics.accepted, 21u);
    CHECK_EQ(metrics.completed, 20u);
    CHECK_EQ(metrics.failed, 1u);
    CHECK_EQ(metrics.rejected_overloaded, 0u);
    CHECK_EQ(metrics.in_flight, 0u);
    CHECK_EQ(server.latency().count(), 20u);
    CHECK(metrics.p50_ns > 0u);
    CHECK(metrics.p99_ns >= metrics.p50_ns);
    CHECK(metrics.max_ns >= metrics.p99_ns);
}

MERIDIAN_TEST(ConcurrentQueriesShareCachesSafely) {
    PipelineOptions options;
    options.parallel_paths = true;
    options.use_result_cache = true;
    ReferenceEngine engine(options);
    SeedCorpus(engine.index, 200, 50, 16);

    ServerOptions server_options;
    server_options.worker_threads = 8;
    server_options.queue_capacity = 512;
    QueryServer server(server_options, &engine.index, &engine.pipeline);

    std::atomic<uint64_t> failures{0};
    std::vector<std::thread> workers;
    for (int t = 0; t < 8; ++t) {
        workers.emplace_back([&, t] {
            for (int i = 0; i < 100; ++i) {
                // A quarter of the load is one shared hot query, so the result cache is
                // read concurrently by every thread rather than being private per thread.
                const std::string text =
                    (i % 4 == 0) ? "segment merge index"
                                 : "segment merge index " + std::to_string(t * 100 + i);
                Query query = MakeQuery(text, 5, 60'000);
                query.embedding = MakeEmbedding(16);
                auto result = server.Search(query);
                if (!result.ok() || result.value().hits.empty()) ++failures;
            }
        });
    }
    for (auto& worker : workers) worker.join();

    CHECK_EQ(failures.load(), 0u);
    const auto cache_stats = engine.result_cache.stats();
    CHECK(cache_stats.hits > 0u);
    CHECK_EQ(cache_stats.hits + cache_stats.misses, 800u);
    const auto metrics = server.metrics();
    CHECK_EQ(metrics.completed, 800u);
    std::printf("      800 concurrent queries, result cache hit rate %.1f%%, p99 %.3f ms\n",
                cache_stats.hit_rate() * 100.0, Millis(metrics.p99_ns));
}

MERIDIAN_TEST_MAIN()
