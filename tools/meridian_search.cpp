// meridian_search: index a document file, run a query, show the ranked results and
// where the time went.
//
// The trace is the point of this tool. A single end-to-end latency number tells a reader
// nothing they can act on, so the output breaks the query into its stages, shows each
// one's share, and prints the arithmetic that should hold — the stages plus the
// unaccounted remainder add up to the total, and when the lexical and semantic paths run
// concurrently the overlap is reported rather than quietly absorbed.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "../src/ann/hnsw.hpp"
#include "../src/serve/reference_index.hpp"
#include "../src/serve/semantic_ann.hpp"
#include "../src/serve/server.hpp"
#include "../src/serve/stages_engine.hpp"
#include "corpus_file.hpp"

namespace {

using meridian::Code;
using meridian::Query;
using meridian::QueryMode;
using meridian::QueryTrace;
using meridian::SearchResponse;
using meridian::tools::HumanBytes;

void Usage() {
    std::fprintf(stderr,
                 "usage: meridian_search --corpus FILE [options] QUERY...\n"
                 "\n"
                 "  --corpus FILE         corpus to index before querying\n"
                 "  --top-k N             results to return (default 10)\n"
                 "  --candidate-k N       candidates to generate before reranking (default 200)\n"
                 "  --mode or|and|phrase  how terms combine (default or)\n"
                 "  --deadline-ms N       latency budget; 0 refuses immediately (default 250)\n"
                 "  --filter K=V          language=, category=, after=, before=\n"
                 "  --hybrid              also run the vector path (stand-in embeddings)\n"
                 "  --semantic exact|hnsw vector path: brute force or the HNSW graph\n"
                 "                        (default hnsw; needs --hybrid to do anything)\n"
                 "  --ef-search N         HNSW beam width at layer 0 (default 64)\n"
                 "  --serial              do not overlap the lexical and semantic paths\n"
                 "  --no-cache            bypass the query result cache\n"
                 "  --docs-per-segment N  segment size while loading (default 10000)\n"
                 "  --repeat N            run the query N times (shows cache and percentiles)\n"
                 "  --workers N           server worker threads (default 4)\n");
}

double Millis(uint64_t nanos) { return static_cast<double>(nanos) / 1e6; }

void PrintStageRow(const char* name, uint64_t nanos, uint64_t total) {
    const double share = total == 0 ? 0.0 : 100.0 * static_cast<double>(nanos) /
                                                static_cast<double>(total);
    // Bar length in half-percent units, capped: a stage over the total (possible when
    // paths overlap) must not print a line that wraps the terminal.
    const int bar = std::min(40, static_cast<int>(share / 2.5));
    std::printf("  %-13s %9.3f ms  %5.1f%%  %s\n", name, Millis(nanos), share,
                std::string(static_cast<size_t>(bar), '#').c_str());
}

void PrintTrace(const QueryTrace& trace) {
    const uint64_t accounted = trace.parse_ns + trace.lexical_ns + trace.ann_ns + trace.merge_ns +
                               trace.rerank_ns + trace.materialize_ns;

    std::printf("\ntrace (generation %llu%s%s)\n",
                static_cast<unsigned long long>(trace.generation_version),
                trace.cache_hit_results ? ", result cache HIT" : "",
                trace.cache_hit_postings ? ", posting cache HIT" : "");
    std::printf("  %-13s %9s     %6s\n", "stage", "time", "share");
    PrintStageRow("parse", trace.parse_ns, trace.total_ns);
    PrintStageRow("lexical", trace.lexical_ns, trace.total_ns);
    PrintStageRow("ann", trace.ann_ns, trace.total_ns);
    PrintStageRow("merge", trace.merge_ns, trace.total_ns);
    PrintStageRow("rerank", trace.rerank_ns, trace.total_ns);
    PrintStageRow("materialize", trace.materialize_ns, trace.total_ns);
    std::printf("  %-13s %9.3f ms\n", "stages", Millis(accounted));
    std::printf("  %-13s %9.3f ms\n", "total", Millis(trace.total_ns));

    if (accounted > trace.total_ns) {
        std::printf("  the stages exceed the total by %.3f ms: the lexical and semantic paths\n"
                    "  ran concurrently, so their times overlap. Each stage figure is still its\n"
                    "  own wall time.\n",
                    Millis(accounted - trace.total_ns));
    } else {
        std::printf("  %-13s %9.3f ms  (pin, cache key, trace bookkeeping)\n", "unaccounted",
                    Millis(trace.total_ns - accounted));
    }

    std::printf("  candidates: lexical=%u semantic=%u merged=%u\n", trace.lexical_candidates,
                trace.semantic_candidates, trace.merged_candidates);
}

void PrintHits(const SearchResponse& response) {
    if (response.hits.empty()) {
        std::printf("\nno results\n");
        return;
    }
    std::printf("\n%-5s %-24s %10s   features\n", "rank", "id", "score");
    int rank = 1;
    for (const auto& hit : response.hits) {
        // Features sorted by name so two runs are diffable; an unordered_map's iteration
        // order is not stable and would make every diff noise.
        std::vector<std::pair<std::string, double>> features(hit.features.begin(),
                                                             hit.features.end());
        std::sort(features.begin(), features.end());
        std::string rendered;
        for (const auto& [name, value] : features) {
            char buffer[96];
            std::snprintf(buffer, sizeof(buffer), "%s=%.4f ", name.c_str(), value);
            rendered += buffer;
        }
        std::printf("%-5d %-24s %10.4f   %s\n", rank++, hit.external_id.c_str(), hit.score,
                    rendered.c_str());
    }
}

/// The reference stages, the two caches and a selectable semantic stage, assembled after
/// the corpus is loaded.
///
/// `ReferenceEngine` would be the convenient thing to use and cannot be: it builds its
/// pipeline in its constructor, and an HNSW stage cannot exist until the corpus it indexes
/// has been loaded into that same engine's index. Giving ReferenceEngine a setter for its
/// semantic stage would make a published pipeline mutable while a query could be running on
/// it, which is the one thing the pinning design is built to avoid. So the wiring is spelled
/// out here and assembled once, in order.
struct SearchEngine {
    meridian::serve::ReferenceIndex index;
    meridian::serve::QueryResultCache result_cache{8u << 20};
    meridian::serve::PostingListCache posting_cache{64u << 20};
    meridian::serve::ReferenceParseStage parse;
    meridian::serve::ReferenceLexicalStage lexical;
    meridian::serve::ReferenceSemanticStage exact_semantic;
    meridian::serve::ReferenceMergeStage merge;
    meridian::serve::ReferenceRerankStage rerank;
    meridian::serve::ReferenceMaterializeStage materialize;

    std::unique_ptr<meridian::ann::HnswIndex> hnsw;
    std::unique_ptr<meridian::serve::EngineSemanticStage> engine_semantic;
    std::unique_ptr<meridian::serve::AnnSemanticStage> ann_semantic;
    std::unique_ptr<meridian::serve::Pipeline> pipeline;

    /// `hnsw` non-null selects the approximate path; null leaves the exact one in place.
    void Assemble(meridian::serve::PipelineOptions options, uint32_t ef_search) {
        const meridian::serve::SemanticStage* semantic = &exact_semantic;
        if (hnsw != nullptr) {
            engine_semantic =
                std::make_unique<meridian::serve::EngineSemanticStage>(hnsw.get(), ef_search);
            ann_semantic =
                std::make_unique<meridian::serve::AnnSemanticStage>(engine_semantic.get());
            semantic = ann_semantic.get();
        }
        meridian::serve::PipelineStages stages{&parse,  &lexical, semantic,
                                               &merge,  &rerank,  &materialize};
        pipeline = std::make_unique<meridian::serve::Pipeline>(stages, options, &result_cache,
                                                              &posting_cache);
    }
};

bool ApplyFilter(const std::string& spec, meridian::Filters& filters) {
    const size_t equals = spec.find('=');
    if (equals == std::string::npos) return false;
    const std::string key = spec.substr(0, equals);
    const std::string value = spec.substr(equals + 1);
    if (key == "language" || key == "lang") {
        filters.language = value;
    } else if (key == "category" || key == "cat") {
        filters.category = value;
    } else if (key == "after") {
        filters.published_after = std::stoll(value);
    } else if (key == "before") {
        filters.published_before = std::stoll(value);
    } else {
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::string corpus;
    std::string query_text;
    meridian::tools::LoadOptions load_options;
    meridian::serve::PipelineOptions pipeline_options;
    meridian::serve::ServerOptions server_options;

    Query request;
    bool hybrid = false;
    bool approximate = true;
    uint32_t ef_search = 64;
    int repeat = 1;
    size_t embedding_dim = 32;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--corpus") {
            corpus = next("--corpus");
        } else if (arg == "--top-k") {
            request.top_k = static_cast<uint32_t>(std::stoul(next("--top-k")));
        } else if (arg == "--candidate-k") {
            request.candidate_k = static_cast<uint32_t>(std::stoul(next("--candidate-k")));
        } else if (arg == "--deadline-ms") {
            request.deadline_ms = static_cast<uint32_t>(std::stoul(next("--deadline-ms")));
        } else if (arg == "--mode") {
            const std::string mode = next("--mode");
            if (mode == "or") {
                request.mode = QueryMode::kDisjunctive;
            } else if (mode == "and") {
                request.mode = QueryMode::kConjunctive;
            } else if (mode == "phrase") {
                request.mode = QueryMode::kPhrase;
            } else {
                std::fprintf(stderr, "unknown mode: %s\n", mode.c_str());
                return 2;
            }
        } else if (arg == "--filter") {
            const std::string spec = next("--filter");
            if (!ApplyFilter(spec, request.filters)) {
                std::fprintf(stderr, "unrecognised filter: %s\n", spec.c_str());
                return 2;
            }
        } else if (arg == "--hybrid") {
            hybrid = true;
        } else if (arg == "--semantic") {
            const std::string choice = next("--semantic");
            if (choice == "hnsw") {
                approximate = true;
            } else if (choice == "exact") {
                approximate = false;
            } else {
                std::fprintf(stderr, "--semantic must be exact or hnsw, got: %s\n", choice.c_str());
                return 2;
            }
        } else if (arg == "--ef-search") {
            ef_search = static_cast<uint32_t>(std::stoul(next("--ef-search")));
        } else if (arg == "--serial") {
            pipeline_options.parallel_paths = false;
        } else if (arg == "--no-cache") {
            pipeline_options.use_result_cache = false;
        } else if (arg == "--docs-per-segment") {
            load_options.docs_per_segment =
                static_cast<uint32_t>(std::stoul(next("--docs-per-segment")));
        } else if (arg == "--repeat") {
            repeat = std::stoi(next("--repeat"));
        } else if (arg == "--workers") {
            server_options.worker_threads = static_cast<size_t>(std::stoul(next("--workers")));
        } else if (arg == "--help" || arg == "-h") {
            Usage();
            return 0;
        } else if (arg.rfind("--", 0) == 0) {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            Usage();
            return 2;
        } else {
            if (!query_text.empty()) query_text += ' ';
            query_text += arg;
        }
    }

    if (corpus.empty() || query_text.empty() || repeat < 1) {
        Usage();
        return 2;
    }
    load_options.embedding_dim = hybrid ? embedding_dim : 0;

    SearchEngine engine;

    const auto load_start = std::chrono::steady_clock::now();
    const auto load = meridian::tools::LoadCorpus(corpus, engine.index, load_options);
    if (!load.ok()) {
        std::fprintf(stderr, "error: %s\n", load.error.c_str());
        return 1;
    }
    const double load_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - load_start).count();

    const auto index_stats = engine.index.stats();
    std::printf("index: %llu documents, %llu segments, %llu distinct terms, %s, built in %.3f s\n",
                static_cast<unsigned long long>(index_stats.documents),
                static_cast<unsigned long long>(index_stats.segments),
                static_cast<unsigned long long>(index_stats.terms),
                HumanBytes(index_stats.bytes).c_str(), load_seconds);

    if (hybrid && approximate) {
        const auto build_start = std::chrono::steady_clock::now();
        meridian::serve::HnswBuildReport build_report;
        auto built = meridian::serve::BuildHnswOverSnapshot(
            *engine.index.Current(), meridian::ann::HnswParams{}, &build_report);
        if (!built.ok()) {
            std::fprintf(stderr, "error: %s\n", built.status().ToString().c_str());
            return 1;
        }
        engine.hnsw = built.take();
        const double build_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - build_start).count();
        std::printf("hnsw: %llu vectors, dimension %u, M %u, ef_construction %u, built in %.3f s\n",
                    static_cast<unsigned long long>(build_report.vectors), build_report.dim,
                    engine.hnsw->params().M, engine.hnsw->params().ef_construction, build_seconds);
    }
    engine.Assemble(pipeline_options, ef_search);

    request.raw = query_text;
    if (hybrid) {
        request.embedding = meridian::tools::HashedEmbedding(
            meridian::serve::Tokenize(query_text), embedding_dim);
    }

    meridian::serve::QueryServer server(server_options, &engine.index, engine.pipeline.get());
    std::printf("query: \"%s\" mode=%s top_k=%u candidate_k=%u budget=%ums%s%s\n",
                query_text.c_str(),
                request.mode == QueryMode::kDisjunctive ? "or"
                : request.mode == QueryMode::kConjunctive ? "and"
                                                          : "phrase",
                request.top_k, request.candidate_k, request.deadline_ms,
                hybrid ? (engine.hnsw != nullptr ? " hybrid semantic=hnsw" : " hybrid semantic=exact")
                       : " lexical-only",
                pipeline_options.parallel_paths ? " parallel-paths" : " serial-paths");
    if (hybrid && engine.hnsw != nullptr && ef_search < request.candidate_k) {
        // The same clamp meridian_eval reports: the semantic stage asks the graph for
        // candidate_k neighbours, and hnsw.hpp raises ef_search to k when it is lower, so a
        // beam narrower than candidate_k is not reachable from here. Printed because a flag
        // that is silently overridden is indistinguishable from a flag that does nothing.
        std::printf("  ef_search=%u is below candidate_k=%u, so the beam runs at %u: the "
                    "semantic stage searches for candidate_k neighbours and ef_search is "
                    "clamped up to that\n",
                    ef_search, request.candidate_k, request.candidate_k);
    }

    int exit_code = 0;
    for (int iteration = 0; iteration < repeat; ++iteration) {
        auto result = server.Search(request);
        if (!result.ok()) {
            const auto& status = result.status();
            std::printf("\nrun %d: %s\n", iteration + 1, status.ToString().c_str());
            if (status.code() == Code::kDeadlineExceeded) {
                std::printf("  no partial result list is returned: a short list that looks like\n"
                            "  a complete answer is the failure this engine refuses to have.\n");
            }
            exit_code = 1;
            continue;
        }
        if (repeat > 1) std::printf("\n--- run %d of %d ---", iteration + 1, repeat);
        const SearchResponse& response = result.value();
        PrintHits(response);
        PrintTrace(response.trace);
    }

    const auto metrics = server.metrics();
    const auto result_cache = engine.result_cache.stats();
    const auto posting_cache = engine.posting_cache.stats();
    const auto pipeline_metrics = engine.pipeline->metrics();

    std::printf("\nserver: accepted=%llu refused_overloaded=%llu completed=%llu failed=%llu "
                "deadline_exceeded=%llu\n",
                static_cast<unsigned long long>(metrics.accepted),
                static_cast<unsigned long long>(metrics.rejected_overloaded),
                static_cast<unsigned long long>(metrics.completed),
                static_cast<unsigned long long>(metrics.failed),
                static_cast<unsigned long long>(metrics.deadline_exceeded));
    if (metrics.completed > 0) {
        std::printf("latency over %llu samples: p50=%.3f ms p95=%.3f ms p99=%.3f ms max=%.3f ms\n",
                    static_cast<unsigned long long>(server.latency().count()),
                    Millis(metrics.p50_ns), Millis(metrics.p95_ns), Millis(metrics.p99_ns),
                    Millis(metrics.max_ns));
    }
    std::printf("result cache: hits=%llu misses=%llu evictions=%llu entries=%zu %s\n",
                static_cast<unsigned long long>(result_cache.hits),
                static_cast<unsigned long long>(result_cache.misses),
                static_cast<unsigned long long>(result_cache.evictions),
                engine.result_cache.entries(), HumanBytes(engine.result_cache.bytes()).c_str());
    std::printf("posting cache: hits=%llu misses=%llu evictions=%llu promotions=%llu entries=%zu "
                "%s\n",
                static_cast<unsigned long long>(posting_cache.hits),
                static_cast<unsigned long long>(posting_cache.misses),
                static_cast<unsigned long long>(posting_cache.evictions),
                static_cast<unsigned long long>(posting_cache.promotions),
                engine.posting_cache.entries(), HumanBytes(engine.posting_cache.bytes()).c_str());
    if (pipeline_options.parallel_paths) {
        std::printf("pipeline: parallel paths used=%llu declined=%llu (declined means the helper "
                    "pool was full and the path ran inline)\n",
                    static_cast<unsigned long long>(pipeline_metrics.parallel_paths_used),
                    static_cast<unsigned long long>(pipeline_metrics.parallel_paths_declined));
    }

    return exit_code;
}
