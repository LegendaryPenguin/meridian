// meridian_eval: runs a judged query set against a configuration and prints both axes.
//
// This file is the single point where the evaluation harness meets the serving layer, and
// that is deliberate. src/eval/ depends only on the frozen headers plus its own
// `QueryRunner` interface, so it keeps compiling while src/serve/, src/query/, src/rank/
// and src/index/ move underneath it. The adapter below is the part that has to track that
// churn, and it is one file rather than four.
//
// The semantic path is selectable, and both choices are here for a reason. `--semantic
// hnsw` runs the real HNSW graph, which is what makes `ef_search` a dial that reaches code
// instead of a number in a label. `--semantic exact` runs brute force, which is the ground
// truth the approximate path's recall is measured against — `--ann-recall` reports exactly
// that comparison, because an ANN latency figure with no recall number beside it is not a
// claim about anything.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../src/ann/hnsw.hpp"
#include "../src/eval/harness.hpp"
#include "../src/eval/sweep.hpp"
#include "../src/serve/reference_index.hpp"
#include "../src/serve/semantic_ann.hpp"
#include "../src/serve/stages_engine.hpp"
#include "corpus_file.hpp"
#include "meridian/eval.hpp"

namespace meridian::tools {
namespace {

using eval::HarnessOptions;
using eval::QueryRunner;
using eval::SweepPoint;

// ---------------------------------------------------------------------------
// Loading the generated collection
// ---------------------------------------------------------------------------

/// Embeddings keyed by external id, read from the file the Python generator writes.
///
/// Read from a file rather than recomputed here, because the whole value of generated
/// embeddings is that they come from the generative process that also produced the
/// judgements. A hashed bag of words computed on this side would be unrelated to the topics
/// the grades are based on, and every recall number would be measuring a hash function.
using VectorTable = std::unordered_map<std::string, std::vector<float>>;

Result<VectorTable> LoadVectors(const std::string& path) {
    std::ifstream input(path);
    if (!input) return Status::IoError("cannot open vector file " + path);

    VectorTable table;
    std::string line;
    size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty() || line[0] == '#') continue;
        const auto fields = detail::SplitTabs(line);
        if (fields.size() < 2) {
            return Status::InvalidArgument(path + ":" + std::to_string(line_number) +
                                          ": expected 'id<TAB>comma-separated vector'");
        }
        std::vector<float> vector;
        std::stringstream parts(fields[1]);
        std::string value;
        while (std::getline(parts, value, ',')) {
            if (value.empty()) continue;
            vector.push_back(std::strtof(value.c_str(), nullptr));
        }
        if (vector.empty()) {
            return Status::InvalidArgument(path + ":" + std::to_string(line_number) +
                                          ": vector is empty");
        }
        table.emplace(fields[0], std::move(vector));
    }
    return table;
}

struct CorpusLoad {
    uint64_t documents = 0;
    uint64_t with_embeddings = 0;
    size_t embedding_dim = 0;
};

Result<CorpusLoad> LoadCorpusWithVectors(const std::string& corpus_path, const VectorTable& vectors,
                                        uint32_t docs_per_segment, serve::ReferenceIndex* index) {
    std::ifstream input(corpus_path);
    if (!input) return Status::IoError("cannot open corpus file " + corpus_path);

    CorpusLoad report;
    std::string line;
    size_t line_number = 0;
    uint32_t buffered = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty() || line[0] == '#') continue;

        const auto fields = detail::SplitTabs(line);
        if (fields.size() < 4) {
            return Status::InvalidArgument(corpus_path + ":" + std::to_string(line_number) +
                                          ": expected at least id, title, body and tags");
        }

        serve::ReferenceIndex::Document doc;
        doc.meta.external_id = fields[0];
        doc.meta.language = fields.size() > 4 ? fields[4] : "";
        doc.meta.category = fields.size() > 5 ? fields[5] : "";
        doc.meta.published_at = fields.size() > 6 ? detail::ParseInt(fields[6], 0) : 0;
        doc.meta.popularity = fields.size() > 7 ? detail::ParseDouble(fields[7], 0.0) : 0.0;

        auto title = serve::Tokenize(fields[1]);
        auto body = serve::Tokenize(fields[2]);
        auto tags = serve::Tokenize(fields[3]);
        doc.meta.length = static_cast<uint32_t>(title.size() + body.size() + tags.size());

        auto it = vectors.find(doc.meta.external_id);
        if (it != vectors.end()) {
            doc.embedding = it->second;
            ++report.with_embeddings;
            report.embedding_dim = it->second.size();
        }

        doc.tokens[Field::kTitle] = std::move(title);
        doc.tokens[Field::kBody] = std::move(body);
        doc.tokens[Field::kTags] = std::move(tags);

        index->Add(std::move(doc));
        ++report.documents;
        if (++buffered >= docs_per_segment) {
            index->Flush();
            buffered = 0;
        }
    }
    if (buffered > 0) index->Flush();
    if (report.documents == 0) {
        return Status::InvalidArgument(corpus_path + ": no documents");
    }
    return report;
}

// ---------------------------------------------------------------------------
// A merge stage that implements all three fusion strategies
// ---------------------------------------------------------------------------

/// Merges the lexical and semantic candidate lists under a selectable strategy.
///
/// The reference merge stage implements reciprocal rank fusion only, and the sweep needs to
/// compare the three strategies retrieval.hpp describes — which one is right is an empirical
/// question, and a tool that can only run one of them cannot answer it.
///
/// All three normalise *within the query* before combining, for the reason retrieval.hpp
/// gives: a BM25 score and a cosine similarity are not on the same scale, are not bounded
/// alike, and adding them directly lets whichever has the larger numeric range win.
class StrategyMergeStage : public serve::MergeStage {
public:
    StrategyMergeStage(HybridMerger::Strategy strategy, double lexical_weight,
                       double semantic_weight)
        : strategy_(strategy), lexical_weight_(lexical_weight), semantic_weight_(semantic_weight) {}

    std::vector<Candidate> Merge(const std::vector<Candidate>& lexical,
                                const std::vector<Candidate>& semantic,
                                uint32_t candidate_k) const override {
        std::map<DocId, Candidate> merged;
        Contribute(lexical, lexical_weight_, /*is_lexical=*/true, &merged);
        Contribute(semantic, semantic_weight_, /*is_lexical=*/false, &merged);

        std::vector<Candidate> out;
        out.reserve(merged.size());
        for (auto& [doc, candidate] : merged) out.push_back(candidate);
        std::sort(out.begin(), out.end(), [](const Candidate& a, const Candidate& b) {
            // Doc id breaks score ties, so the output is a total order. A tie resolved by
            // whatever order the map happened to produce would make two identical runs
            // disagree on rank, and every ranking comparison downstream would see noise.
            if (a.score != b.score) return a.score > b.score;
            return a.doc < b.doc;
        });
        if (out.size() > candidate_k) out.resize(candidate_k);
        return out;
    }

private:
    void Contribute(const std::vector<Candidate>& candidates, double weight, bool is_lexical,
                    std::map<DocId, Candidate>* merged) const {
        if (candidates.empty() || weight == 0.0) return;

        std::vector<double> normalized(candidates.size(), 0.0);
        switch (strategy_) {
            case HybridMerger::Strategy::kReciprocalRankFusion: {
                // 1/(k + rank). Insensitive to score scale by construction, which is its
                // main virtue and the reason it is the default.
                constexpr double kRrfK = 60.0;
                for (size_t i = 0; i < candidates.size(); ++i) {
                    normalized[i] = 1.0 / (kRrfK + static_cast<double>(i) + 1.0);
                }
                break;
            }
            case HybridMerger::Strategy::kNormalizedLinear: {
                // Min-max into [0,1]. Sensitive to outliers: one very high score compresses
                // everything else toward zero.
                double lo = candidates.front().score;
                double hi = lo;
                for (const auto& c : candidates) {
                    lo = std::min(lo, c.score);
                    hi = std::max(hi, c.score);
                }
                const double span = hi - lo;
                for (size_t i = 0; i < candidates.size(); ++i) {
                    normalized[i] = span > 0.0 ? (candidates[i].score - lo) / span : 1.0;
                }
                break;
            }
            case HybridMerger::Strategy::kZScoreLinear: {
                // Handles outliers better than min-max and assumes a distribution shape a
                // short candidate list may not have.
                double sum = 0.0;
                for (const auto& c : candidates) sum += c.score;
                const double mean = sum / static_cast<double>(candidates.size());
                double variance = 0.0;
                for (const auto& c : candidates) variance += (c.score - mean) * (c.score - mean);
                variance /= static_cast<double>(candidates.size());
                const double deviation = std::sqrt(variance);
                for (size_t i = 0; i < candidates.size(); ++i) {
                    normalized[i] =
                        deviation > 0.0 ? (candidates[i].score - mean) / deviation : 0.0;
                }
                break;
            }
        }

        for (size_t i = 0; i < candidates.size(); ++i) {
            Candidate& target = (*merged)[candidates[i].doc];
            if (target.doc != candidates[i].doc) {
                target = candidates[i];
                target.score = 0.0;
                target.source = is_lexical ? Candidate::Source::kLexical
                                           : Candidate::Source::kSemantic;
            } else {
                target.source = Candidate::Source::kBoth;
            }
            target.score += weight * normalized[i];
            if (is_lexical) {
                target.bm25 = candidates[i].bm25 != 0.0 ? candidates[i].bm25 : candidates[i].score;
            } else {
                target.vector_similarity = candidates[i].vector_similarity != 0.0
                                               ? candidates[i].vector_similarity
                                               : candidates[i].score;
            }
        }
    }

    HybridMerger::Strategy strategy_;
    double lexical_weight_;
    double semantic_weight_;
};

/// A lexical stage that retrieves nothing, for semantic-only configurations.
///
/// An empty stage rather than a null pointer: the pipeline's contract is that each stage it
/// was given runs, and "lexical retrieval found nothing" is a legitimate answer that keeps
/// the trace's shape comparable across configurations.
class NoLexicalStage : public serve::LexicalStage {
public:
    Result<std::vector<Candidate>> Retrieve(const serve::StageContext&,
                                            const Query&) const override {
        return std::vector<Candidate>{};
    }
};

// ---------------------------------------------------------------------------
// The runner
// ---------------------------------------------------------------------------

enum class Mode { kHybrid, kLexical, kSemantic };

const char* ModeName(Mode mode) {
    switch (mode) {
        case Mode::kHybrid: return "hybrid";
        case Mode::kLexical: return "lexical";
        case Mode::kSemantic: return "semantic";
    }
    return "unknown";
}

/// Which semantic stage the pipeline runs.
///
/// Both are kept because they answer different questions. kHnsw is what a deployment would
/// run and the only setting under which `ef_search` means anything. kExact is the reference:
/// recall is defined as agreement with it, so deleting it would leave the approximate path's
/// quality unknowable rather than merely unmeasured.
enum class SemanticBackend { kExact, kHnsw };

const char* BackendName(SemanticBackend backend) {
    return backend == SemanticBackend::kHnsw ? "hnsw" : "exact";
}

struct EngineOptions {
    Mode mode = Mode::kHybrid;
    SemanticBackend semantic = SemanticBackend::kHnsw;

    /// The graph the HNSW backend searches. Owned by the caller and shared across every
    /// point in a sweep: the graph's structure depends on M and ef_construction, not on
    /// ef_search, so rebuilding it per row would spend minutes proving that an unchanged
    /// build produces an unchanged graph.
    const AnnIndex* ann = nullptr;

    bool use_result_cache = true;
    size_t result_cache_bytes = 8u << 20;
    size_t posting_cache_bytes = 64u << 20;
};

/// Wires the reference stages into a pipeline and presents it as a `QueryRunner`.
class EngineRunner : public QueryRunner {
public:
    EngineRunner(serve::ReferenceIndex* index, const SweepPoint& point, EngineOptions options,
                 std::string label)
        : index_(index),
          options_(options),
          label_(std::move(label)),
          result_cache_(options.result_cache_bytes),
          posting_cache_(options.posting_cache_bytes),
          lexical_(point.bm25),
          merge_(point.strategy, point.lexical_weight, point.semantic_weight),
          engine_semantic_(options.ann, point.ef_search),
          ann_semantic_(&engine_semantic_) {
        serve::PipelineStages stages;
        stages.parse = &parse_;
        stages.lexical =
            options.mode == Mode::kSemantic ? static_cast<serve::LexicalStage*>(&no_lexical_)
                                            : static_cast<serve::LexicalStage*>(&lexical_);
        const serve::SemanticStage* semantic =
            options.semantic == SemanticBackend::kHnsw
                ? static_cast<const serve::SemanticStage*>(&ann_semantic_)
                : static_cast<const serve::SemanticStage*>(&semantic_);
        stages.semantic = options.mode == Mode::kLexical ? nullptr : semantic;
        stages.merge = &merge_;
        stages.rerank = &rerank_;
        stages.materialize = &materialize_;

        serve::PipelineOptions pipeline_options;
        pipeline_options.use_result_cache = options.use_result_cache;
        pipeline_ = std::make_unique<serve::Pipeline>(stages, pipeline_options, &result_cache_,
                                                     &posting_cache_);
    }

    Result<SearchResponse> Run(const Query& query) override {
        Query request = query;
        if (options_.mode == Mode::kLexical) {
            // Dropping the embedding rather than leaving the semantic stage out of the wiring
            // would be subtly different: the stage list is what the trace reports, and a
            // lexical-only measurement must not include the cost of a vector search whose
            // results were thrown away.
            request.embedding.reset();
        }
        const serve::GenerationPin pin = index_->Acquire();
        QueryTrace partial;
        return pipeline_->Run(request, pin, serve::Deadline::AfterMillis(request.deadline_ms),
                              &partial);
    }

    void ClearCaches() override {
        result_cache_.Clear();
        posting_cache_.Clear();
    }

    std::string Describe() const override { return label_; }

    serve::Pipeline::Metrics metrics() const { return pipeline_->metrics(); }

private:
    serve::ReferenceIndex* index_;
    EngineOptions options_;
    std::string label_;

    serve::QueryResultCache result_cache_;
    serve::PostingListCache posting_cache_;
    serve::ReferenceParseStage parse_;
    serve::ReferenceLexicalStage lexical_;
    NoLexicalStage no_lexical_;
    serve::ReferenceSemanticStage semantic_;
    StrategyMergeStage merge_;
    serve::EngineSemanticStage engine_semantic_;
    serve::AnnSemanticStage ann_semantic_;
    serve::ReferenceRerankStage rerank_;
    serve::ReferenceMaterializeStage materialize_;
    std::unique_ptr<serve::Pipeline> pipeline_;
};

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

std::string Fixed(double value, int precision) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.*f", precision, value);
    return buffer;
}

double Micros(uint64_t nanos) { return static_cast<double>(nanos) / 1000.0; }

void PrintRun(const EvalRun& run) {
    std::cout << "configuration: " << run.label << "\n"
              << "  queries              " << run.queries << " (" << run.queries_timed_out
              << " timed out)\n"
              << "  NDCG@" << run.quality.k << "              " << Fixed(run.quality.ndcg_at_k, 4)
              << "\n"
              << "  P@" << run.quality.k << "                 "
              << Fixed(run.quality.precision_at_k, 4) << "\n"
              << "  recall@" << run.quality.k << "            "
              << Fixed(run.quality.recall_at_k, 4) << "\n"
              << "  MRR                  " << Fixed(run.quality.mrr, 4) << "\n"
              << "  unjudged in top k    " << run.quality.unjudged_in_top_k << "\n"
              << "  p50 / p95 / p99 us   " << Fixed(Micros(run.latency.p50_ns), 2) << " / "
              << Fixed(Micros(run.latency.p95_ns), 2) << " / "
              << Fixed(Micros(run.latency.p99_ns), 2) << "\n"
              << "  max / mean us        " << Fixed(Micros(run.latency.max_ns), 2) << " / "
              << Fixed(Micros(run.latency.mean_ns), 2) << "\n"
              << "  stage p50 us         lexical " << Fixed(Micros(run.latency.lexical_p50_ns), 2)
              << ", ann " << Fixed(Micros(run.latency.ann_p50_ns), 2) << ", rerank "
              << Fixed(Micros(run.latency.rerank_p50_ns), 2) << "\n"
              << "  mean candidates      " << Fixed(run.mean_candidates, 1) << "\n"
              << "  result cache hits    " << Fixed(run.cache_hit_rate * 100.0, 1) << "%\n";
}

/// Per-query quality, because a mean hides a bimodal distribution and a judged set that
/// mixes query kinds is bimodal by design.
void PrintPerQuery(const EvalRun& run, size_t limit) {
    std::cout << "\nper-query quality (first " << std::min(limit, run.per_query.size()) << " of "
              << run.per_query.size() << ")\n";
    std::cout << "  query            NDCG     P@k      R@k      MRR      unjudged\n";
    for (size_t i = 0; i < run.per_query.size() && i < limit; ++i) {
        const auto& [id, metrics] = run.per_query[i];
        std::printf("  %-15s  %-7s  %-7s  %-7s  %-7s  %u\n", id.c_str(),
                    Fixed(metrics.ndcg_at_k, 4).c_str(), Fixed(metrics.precision_at_k, 4).c_str(),
                    Fixed(metrics.recall_at_k, 4).c_str(), Fixed(metrics.mrr, 4).c_str(),
                    metrics.unjudged_in_top_k);
    }
}

std::string JsonEscape(const std::string& text) {
    std::string out;
    for (char c : text) {
        if (c == '"' || c == '\\') {
            out.push_back('\\');
            out.push_back(c);
        } else if (c == '\n') {
            out += "\\n";
        } else {
            out.push_back(c);
        }
    }
    return out;
}

/// Writes the results file python/meridian_eval/report.py reads.
Status WriteJson(const std::string& path, const eval::SweepResult& result, uint32_t k) {
    std::ofstream out(path);
    if (!out) return Status::IoError("cannot write " + path);

    const std::string baseline = eval::DescribePoint(result.rows[result.baseline_index].point);
    out << "{\n";
    out << "  \"k\": " << k << ",\n";
    out << "  \"baseline\": \"" << JsonEscape(baseline) << "\",\n";
    out << "  \"latency_tolerance\": " << Fixed(result.latency_tolerance, 4) << ",\n";
    out << "  \"queries\": " << result.rows.front().run.queries << ",\n";
    out << "  \"rows\": [\n";
    for (size_t i = 0; i < result.rows.size(); ++i) {
        const auto& row = result.rows[i];
        const bool is_baseline = i == result.baseline_index;
        out << "    {\n"
            << "      \"label\": \"" << JsonEscape(eval::DescribePoint(row.point)) << "\",\n"
            << "      \"ndcg\": " << Fixed(row.run.quality.ndcg_at_k, 6) << ",\n"
            << "      \"precision\": " << Fixed(row.run.quality.precision_at_k, 6) << ",\n"
            << "      \"recall\": " << Fixed(row.run.quality.recall_at_k, 6) << ",\n"
            << "      \"mrr\": " << Fixed(row.run.quality.mrr, 6) << ",\n"
            << "      \"unjudged\": " << row.run.quality.unjudged_in_top_k << ",\n"
            << "      \"p50_us\": " << Fixed(Micros(row.run.latency.p50_ns), 3) << ",\n"
            << "      \"p95_us\": " << Fixed(Micros(row.run.latency.p95_ns), 3) << ",\n"
            << "      \"p99_us\": " << Fixed(Micros(row.run.latency.p99_ns), 3) << ",\n"
            << "      \"mean_candidates\": " << Fixed(row.run.mean_candidates, 3) << ",\n"
            << "      \"timeouts\": " << row.run.queries_timed_out << ",\n"
            << "      \"cache_hit_rate\": " << Fixed(row.run.cache_hit_rate, 4) << ",\n"
            << "      \"verdict\": \"" << (is_baseline ? "" : VerdictName(row.vs_baseline.verdict))
            << "\",\n"
            << "      \"explanation\": \""
            << (is_baseline ? "" : JsonEscape(row.vs_baseline.explanation)) << "\"\n"
            << "    }" << (i + 1 == result.rows.size() ? "" : ",") << "\n";
    }
    out << "  ]\n}\n";
    return Status::Ok();
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

struct Args {
    std::string corpus;
    std::string embeddings;
    std::string queries;
    std::string qrels;
    std::string query_embeddings;
    std::string json;
    std::string single_query;
    std::string sweep_axis;
    std::vector<std::string> sweep_values;
    std::vector<uint32_t> ann_recall_efs;

    uint32_t k = 10;
    uint32_t ef_search = 64;
    uint32_t top_k = 10;
    uint32_t candidate_k = 200;
    uint32_t deadline_ms = 250;
    uint32_t repetitions = 3;
    uint32_t per_query_limit = 12;
    uint32_t docs_per_segment = 10'000;
    double tolerance = 0.10;
    double k1 = 1.2;
    double b = 0.75;
    double lexical_weight = 1.0;
    double semantic_weight = 1.0;
    Mode mode = Mode::kHybrid;
    SemanticBackend semantic = SemanticBackend::kHnsw;
    bool cold = false;
    bool no_cache = false;
    bool show_per_query = true;
};

void PrintUsage() {
    std::cout <<
        R"(meridian_eval — runs a judged query set and reports quality and latency together.

Required:
  --corpus PATH              corpus.tsv as written by `python3 -m meridian_eval generate`
  --queries PATH             queries.tsv (qid<TAB>text)
  --qrels PATH               qrels.txt (TREC qid iteration docid grade)

Optional:
  --embeddings PATH          document embeddings; without them the semantic path is off
  --query-embeddings PATH    query embeddings, keyed by qid
  --mode hybrid|lexical|semantic
  --semantic exact|hnsw      semantic stage: brute force or the HNSW graph (default hnsw)
  --ef-search N              HNSW beam width at layer 0 (default 64); exact ignores it
  --ann-recall EF,EF,EF      report the HNSW path's recall against exact search and exit
  --k N                      metric cutoff (default 10)
  --top-k N                  results requested per query (default 10)
  --candidate-k N            candidates before reranking (default 200)
  --deadline-ms N            per-query latency budget (default 250)
  --repeat N                 latency samples per query (default 3)
  --k1 F --b F               BM25 parameters
  --lexical-weight F --semantic-weight F
  --tolerance F              fractional p99 regression treated as material (default 0.10)
  --cold                     drop caches before every sample
  --no-cache                 disable the result cache entirely
  --no-per-query             omit the per-query table
  --docs-per-segment N       segments to build the index with
  --query TEXT               run this one query and print its hits and stage trace
  --sweep AXIS=V,V,V         candidate_k, ef_search, k1, b, weights or strategy
  --json PATH                write a results file for `python3 -m meridian_eval report`
)";
}

bool ParseUint(const std::string& text, uint32_t* out) {
    try {
        const long value = std::stol(text);
        if (value < 0) return false;
        *out = static_cast<uint32_t>(value);
        return true;
    } catch (...) {
        return false;
    }
}

std::vector<std::string> Split(const std::string& text, char separator) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string part;
    while (std::getline(stream, part, separator)) {
        if (!part.empty()) parts.push_back(part);
    }
    return parts;
}

Result<Args> ParseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        auto next = [&](std::string* target) -> bool {
            if (i + 1 >= argc) return false;
            *target = argv[++i];
            return true;
        };
        std::string value;

        if (flag == "--help" || flag == "-h") {
            PrintUsage();
            std::exit(0);
        } else if (flag == "--corpus") {
            if (!next(&args.corpus)) return Status::InvalidArgument("--corpus needs a path");
        } else if (flag == "--embeddings") {
            if (!next(&args.embeddings)) return Status::InvalidArgument("--embeddings needs a path");
        } else if (flag == "--queries") {
            if (!next(&args.queries)) return Status::InvalidArgument("--queries needs a path");
        } else if (flag == "--qrels") {
            if (!next(&args.qrels)) return Status::InvalidArgument("--qrels needs a path");
        } else if (flag == "--query-embeddings") {
            if (!next(&args.query_embeddings)) {
                return Status::InvalidArgument("--query-embeddings needs a path");
            }
        } else if (flag == "--json") {
            if (!next(&args.json)) return Status::InvalidArgument("--json needs a path");
        } else if (flag == "--query") {
            if (!next(&args.single_query)) return Status::InvalidArgument("--query needs text");
        } else if (flag == "--mode") {
            if (!next(&value)) return Status::InvalidArgument("--mode needs a value");
            if (value == "hybrid") {
                args.mode = Mode::kHybrid;
            } else if (value == "lexical") {
                args.mode = Mode::kLexical;
            } else if (value == "semantic") {
                args.mode = Mode::kSemantic;
            } else {
                return Status::InvalidArgument("--mode must be hybrid, lexical or semantic");
            }
        } else if (flag == "--semantic") {
            if (!next(&value)) return Status::InvalidArgument("--semantic needs a value");
            if (value == "exact") {
                args.semantic = SemanticBackend::kExact;
            } else if (value == "hnsw") {
                args.semantic = SemanticBackend::kHnsw;
            } else {
                return Status::InvalidArgument("--semantic must be exact or hnsw");
            }
        } else if (flag == "--ann-recall") {
            if (!next(&value)) return Status::InvalidArgument("--ann-recall needs EF,EF,EF");
            for (const auto& text : Split(value, ',')) {
                uint32_t parsed = 0;
                if (!ParseUint(text, &parsed)) {
                    return Status::InvalidArgument("--ann-recall value '" + text +
                                                  "' is not a number");
                }
                args.ann_recall_efs.push_back(parsed);
            }
            if (args.ann_recall_efs.empty()) {
                return Status::InvalidArgument("--ann-recall needs at least one ef_search value");
            }
        } else if (flag == "--sweep") {
            if (!next(&value)) return Status::InvalidArgument("--sweep needs AXIS=V,V,V");
            const size_t equals = value.find('=');
            if (equals == std::string::npos) {
                return Status::InvalidArgument("--sweep needs AXIS=V,V,V, got " + value);
            }
            args.sweep_axis = value.substr(0, equals);
            args.sweep_values = Split(value.substr(equals + 1), ',');
            if (args.sweep_values.empty()) {
                return Status::InvalidArgument("--sweep " + args.sweep_axis + " has no values");
            }
        } else if (flag == "--k" || flag == "--top-k" || flag == "--candidate-k" ||
                   flag == "--deadline-ms" || flag == "--repeat" || flag == "--docs-per-segment" ||
                   flag == "--ef-search") {
            if (!next(&value)) return Status::InvalidArgument(flag + " needs a number");
            uint32_t parsed = 0;
            if (!ParseUint(value, &parsed)) {
                return Status::InvalidArgument(flag + " needs a number, got " + value);
            }
            if (flag == "--k") args.k = parsed;
            if (flag == "--top-k") args.top_k = parsed;
            if (flag == "--candidate-k") args.candidate_k = parsed;
            if (flag == "--deadline-ms") args.deadline_ms = parsed;
            if (flag == "--repeat") args.repetitions = parsed;
            if (flag == "--docs-per-segment") args.docs_per_segment = parsed;
            if (flag == "--ef-search") args.ef_search = parsed;
        } else if (flag == "--k1" || flag == "--b" || flag == "--tolerance" ||
                   flag == "--lexical-weight" || flag == "--semantic-weight") {
            if (!next(&value)) return Status::InvalidArgument(flag + " needs a number");
            const double parsed = std::strtod(value.c_str(), nullptr);
            if (flag == "--k1") args.k1 = parsed;
            if (flag == "--b") args.b = parsed;
            if (flag == "--tolerance") args.tolerance = parsed;
            if (flag == "--lexical-weight") args.lexical_weight = parsed;
            if (flag == "--semantic-weight") args.semantic_weight = parsed;
        } else if (flag == "--cold") {
            args.cold = true;
        } else if (flag == "--no-cache") {
            args.no_cache = true;
        } else if (flag == "--no-per-query") {
            args.show_per_query = false;
        } else {
            return Status::InvalidArgument("unknown flag " + flag);
        }
    }

    if (args.corpus.empty()) return Status::InvalidArgument("--corpus is required");
    if (args.single_query.empty() && (args.queries.empty() || args.qrels.empty())) {
        return Status::InvalidArgument("--queries and --qrels are required unless --query is used");
    }
    return args;
}

Result<std::vector<SweepPoint>> BuildSweep(const Args& args, const SweepPoint& base) {
    if (args.sweep_axis.empty()) return std::vector<SweepPoint>{base};

    if (args.sweep_axis == "candidate_k" || args.sweep_axis == "ef_search") {
        std::vector<uint32_t> values;
        for (const auto& text : args.sweep_values) {
            uint32_t parsed = 0;
            if (!ParseUint(text, &parsed)) {
                return Status::InvalidArgument("sweep value '" + text + "' is not a number");
            }
            values.push_back(parsed);
        }
        return args.sweep_axis == "candidate_k" ? eval::SweepCandidateK(base, values)
                                                : eval::SweepEfSearch(base, values);
    }

    if (args.sweep_axis == "k1" || args.sweep_axis == "b") {
        std::vector<double> values;
        for (const auto& text : args.sweep_values) values.push_back(std::strtod(text.c_str(), nullptr));
        return args.sweep_axis == "k1" ? eval::SweepBm25(base, values, {base.bm25.b})
                                       : eval::SweepBm25(base, {base.bm25.k1}, values);
    }

    if (args.sweep_axis == "weights") {
        // `L/S` pairs. 1/0 is lexical-only fusion and 0/1 is semantic-only, which is how the
        // two paths are compared on equal footing: both still run, so the latency includes
        // the same work and only the fusion changes. Turning a path off in the wiring instead
        // would compare quality at one latency against quality at another.
        std::vector<std::pair<double, double>> weights;
        for (const auto& text : args.sweep_values) {
            const size_t slash = text.find('/');
            if (slash == std::string::npos) {
                return Status::InvalidArgument("weight '" + text + "' must be LEXICAL/SEMANTIC");
            }
            weights.emplace_back(std::strtod(text.substr(0, slash).c_str(), nullptr),
                                 std::strtod(text.substr(slash + 1).c_str(), nullptr));
        }
        return eval::SweepHybrid(base, {base.strategy}, weights);
    }

    if (args.sweep_axis == "strategy") {
        std::vector<HybridMerger::Strategy> strategies;
        for (const auto& text : args.sweep_values) {
            if (text == "rrf") {
                strategies.push_back(HybridMerger::Strategy::kReciprocalRankFusion);
            } else if (text == "norm-linear") {
                strategies.push_back(HybridMerger::Strategy::kNormalizedLinear);
            } else if (text == "zscore-linear") {
                strategies.push_back(HybridMerger::Strategy::kZScoreLinear);
            } else {
                return Status::InvalidArgument("unknown strategy '" + text +
                                              "'; use rrf, norm-linear or zscore-linear");
            }
        }
        return eval::SweepHybrid(base, strategies,
                                 {{base.lexical_weight, base.semantic_weight}});
    }

    return Status::InvalidArgument("unknown sweep axis '" + args.sweep_axis +
                                  "'; use candidate_k, ef_search, k1, b, weights or strategy");
}

// ---------------------------------------------------------------------------
// Recall of the approximate path against the exact one
// ---------------------------------------------------------------------------

/// Measures the HNSW index against its own brute-force search, at the index level.
///
/// This is the number that has to exist before any ef_search sweep can be read. NDCG after
/// fusion and reranking answers a different question — whether the pipeline's output got
/// worse — and a hybrid pipeline can hold NDCG flat while the semantic path loses half its
/// neighbours, because the lexical path independently found many of the same documents. So
/// recall is measured here in isolation, where nothing can compensate for it.
///
/// Both sides run through `HnswIndex`, which means both use the same float distance kernels
/// over the same contiguous vector array. Timing `Search` against `ReferenceSemanticStage`
/// instead would compare a float kernel against a double one and a graph walk against a
/// snapshot scan, and would report part of the difference between two implementations as a
/// property of approximate search.
int RunAnnRecall(const Args& args, const ann::HnswIndex& ann,
                 const std::vector<JudgedQuery>& queries, const VectorTable& query_vectors) {
    using Clock = std::chrono::steady_clock;

    std::vector<const std::vector<float>*> embeddings;
    for (const auto& query : queries) {
        auto it = query_vectors.find(query.id);
        if (it != query_vectors.end()) embeddings.push_back(&it->second);
    }
    if (embeddings.empty()) {
        std::cerr << "--ann-recall needs query embeddings; pass --query-embeddings\n";
        return 1;
    }

    const uint32_t k = std::max(1u, args.k);
    const uint32_t repetitions = std::max(1u, args.repetitions);

    // Ground truth once per query, not once per (query, ef_search) pair. It does not depend
    // on ef_search, and recomputing it would make the exact path's timing samples outnumber
    // each approximate row's by the number of rows.
    std::vector<std::vector<DocId>> truth;
    std::vector<uint64_t> exact_samples;
    truth.reserve(embeddings.size());
    for (const std::vector<float>* embedding : embeddings) {
        std::vector<DocId> ids;
        for (uint32_t rep = 0; rep < repetitions; ++rep) {
            const auto start = Clock::now();
            Result<std::vector<Candidate>> exact = ann.SearchExact(*embedding, k);
            const auto elapsed = Clock::now() - start;
            if (!exact.ok()) {
                std::cerr << "exact search failed: " << exact.status().ToString() << "\n";
                return 1;
            }
            exact_samples.push_back(
                static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()));
            if (rep == 0) {
                for (const Candidate& candidate : exact.value()) ids.push_back(candidate.doc);
            }
        }
        truth.push_back(std::move(ids));
    }

    std::cout << "\nANN recall against exact search over the same vectors\n"
              << "  " << ann.size() << " vectors, dimension " << ann.dim() << ", metric cosine, M "
              << ann.params().M << ", M0 " << ann.params().M0 << ", ef_construction "
              << ann.params().ef_construction << "\n"
              << "  " << embeddings.size() << " queries with embeddings, top-" << k << ", "
              << repetitions << " timing sample(s) each\n\n";
    std::printf("  %-10s  %-10s  %-10s  %-10s  %-12s\n", "ef_search", "recall@k", "p50 us",
                "p99 us", "dist/query");
    std::printf("  %-10s  %-10s  %-10s  %-10s  %-12s\n", "exact", "1.0000",
                Fixed(Micros(eval::Percentile(exact_samples, 0.50)), 2).c_str(),
                Fixed(Micros(eval::Percentile(exact_samples, 0.99)), 2).c_str(),
                std::to_string(ann.size()).c_str());

    for (uint32_t ef : args.ann_recall_efs) {
        double recall_sum = 0.0;
        uint64_t distances = 0;
        std::vector<uint64_t> samples;
        for (size_t q = 0; q < embeddings.size(); ++q) {
            std::vector<DocId> returned;
            for (uint32_t rep = 0; rep < repetitions; ++rep) {
                ann.ResetStats();
                const auto start = Clock::now();
                Result<std::vector<Candidate>> hits = ann.Search(*embeddings[q], k, ef);
                const auto elapsed = Clock::now() - start;
                if (!hits.ok()) {
                    std::cerr << "approximate search failed: " << hits.status().ToString() << "\n";
                    return 1;
                }
                samples.push_back(
                    static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()));
                if (rep == 0) {
                    distances += ann.stats().distance_computations;
                    for (const Candidate& candidate : hits.value()) returned.push_back(candidate.doc);
                }
            }
            // Recall is against the size of the exact list, not against k. A query on an
            // index smaller than k has fewer than k true neighbours, and dividing by k there
            // would report a recall below 1.0 for an index that returned every vector it has.
            const std::unordered_set<DocId> found(returned.begin(), returned.end());
            size_t hit_count = 0;
            for (DocId doc : truth[q]) {
                if (found.count(doc) != 0) ++hit_count;
            }
            recall_sum += truth[q].empty()
                              ? 1.0
                              : static_cast<double>(hit_count) /
                                    static_cast<double>(truth[q].size());
        }

        std::printf("  %-10u  %-10s  %-10s  %-10s  %-12s\n", ef,
                    Fixed(recall_sum / static_cast<double>(embeddings.size()), 4).c_str(),
                    Fixed(Micros(eval::Percentile(samples, 0.50)), 2).c_str(),
                    Fixed(Micros(eval::Percentile(samples, 0.99)), 2).c_str(),
                    Fixed(static_cast<double>(distances) /
                              static_cast<double>(embeddings.size()), 1).c_str());
    }
    return 0;
}

int RunSingleQuery(const Args& args, serve::ReferenceIndex* index, const SweepPoint& base,
                   const EngineOptions& engine_options, const VectorTable& query_vectors) {
    EngineRunner runner(index, base, engine_options, ModeName(args.mode));

    Query query;
    query.raw = args.single_query;
    query.top_k = args.top_k;
    query.candidate_k = args.candidate_k;
    query.deadline_ms = args.deadline_ms;

    // A free-text query has no precomputed embedding. One is available only when the text
    // happens to be a known query id's text, so the semantic path runs for the generated
    // queries and is skipped otherwise — which is honest, and visible in the trace's
    // semantic_candidates count.
    auto it = query_vectors.find(args.single_query);
    if (it != query_vectors.end()) query.embedding = it->second;

    for (uint32_t rep = 0; rep < std::max(1u, args.repetitions); ++rep) {
        if (args.cold) runner.ClearCaches();
        Result<SearchResponse> response = runner.Run(query);
        if (!response.ok()) {
            std::cerr << "query failed: " << response.status().ToString() << "\n";
            return 1;
        }
        const SearchResponse& value = response.value();
        std::cout << "\nrun " << (rep + 1) << " of " << std::max(1u, args.repetitions) << " — "
                  << ModeName(args.mode) << ", candidate_k " << query.candidate_k << "\n";
        std::cout << "  result cache: " << (value.trace.cache_hit_results ? "HIT" : "miss")
                  << ", generation " << value.trace.generation_version << "\n";
        std::cout << "  stage ns: parse " << value.trace.parse_ns << ", lexical "
                  << value.trace.lexical_ns << ", ann " << value.trace.ann_ns << ", merge "
                  << value.trace.merge_ns << ", rerank " << value.trace.rerank_ns
                  << ", materialize " << value.trace.materialize_ns << ", total "
                  << value.trace.total_ns << "\n";
        std::cout << "  candidates: lexical " << value.trace.lexical_candidates << ", semantic "
                  << value.trace.semantic_candidates << ", merged "
                  << value.trace.merged_candidates << "\n";
        std::cout << "  hits (" << value.hits.size() << "):\n";
        for (size_t i = 0; i < value.hits.size(); ++i) {
            std::cout << "    " << (i + 1) << ". " << value.hits[i].external_id << "  score "
                      << Fixed(value.hits[i].score, 6) << "\n";
        }
        if (value.hits.empty()) {
            std::cout << "    (nothing matched — which is itself the result for a query this "
                         "path cannot answer)\n";
        }
    }
    return 0;
}

int Main(int argc, char** argv) {
    Result<Args> parsed = ParseArgs(argc, argv);
    if (!parsed.ok()) {
        std::cerr << parsed.status().ToString() << "\n\n";
        PrintUsage();
        return 2;
    }
    const Args args = parsed.take();

    VectorTable document_vectors;
    if (!args.embeddings.empty()) {
        Result<VectorTable> loaded = LoadVectors(args.embeddings);
        if (!loaded.ok()) {
            std::cerr << loaded.status().ToString() << "\n";
            return 1;
        }
        document_vectors = loaded.take();
    }

    serve::ReferenceIndex index;
    Result<CorpusLoad> load =
        LoadCorpusWithVectors(args.corpus, document_vectors, args.docs_per_segment, &index);
    if (!load.ok()) {
        std::cerr << load.status().ToString() << "\n";
        return 1;
    }
    const auto stats = index.stats();
    std::cout << "index: " << load.value().documents << " documents, " << stats.segments
              << " segment(s), " << stats.terms << " term entries, " << HumanBytes(stats.bytes)
              << "\n";
    std::cout << "embeddings: " << load.value().with_embeddings << " document(s), dimension "
              << load.value().embedding_dim << "\n";
    if (load.value().with_embeddings == 0 && args.mode != Mode::kLexical) {
        std::cout << "note: no document embeddings loaded, so the semantic path cannot "
                     "contribute. Pass --embeddings.\n";
    }

    // Built once, over the generation the load just published, and shared by every point in
    // a sweep. The graph depends on M and ef_construction; ef_search is a query-time beam
    // width, so a per-row rebuild would cost minutes to produce an identical graph.
    std::unique_ptr<ann::HnswIndex> hnsw;
    SemanticBackend backend = args.semantic;
    if (load.value().with_embeddings == 0) {
        // Not a silent downgrade: with no vectors in the corpus there is nothing for either
        // semantic stage to search, and a zero-dimension HNSW index would reject every query
        // outright. Said out loud so that a reader who passed --semantic hnsw and sees exact
        // search in the configuration line knows why.
        if (backend == SemanticBackend::kHnsw) {
            std::cout << "note: --semantic hnsw was requested but the corpus carries no "
                         "embeddings, so there is no graph to build. The semantic stage is "
                         "wired to exact search, which also returns nothing.\n";
            backend = SemanticBackend::kExact;
        }
    } else if (backend == SemanticBackend::kHnsw || !args.ann_recall_efs.empty()) {
        const auto build_start = std::chrono::steady_clock::now();
        serve::HnswBuildReport build_report;
        Result<std::unique_ptr<ann::HnswIndex>> built = serve::BuildHnswOverSnapshot(
            *index.Current(), ann::HnswParams{}, &build_report);
        if (!built.ok()) {
            std::cerr << built.status().ToString() << "\n";
            return 1;
        }
        hnsw = built.take();
        const double build_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - build_start).count();
        std::cout << "hnsw: " << build_report.vectors << " vectors, dimension "
                  << build_report.dim << ", M " << hnsw->params().M << ", ef_construction "
                  << hnsw->params().ef_construction << ", built in " << Fixed(build_seconds, 3)
                  << " s";
        if (build_report.without_embedding > 0) {
            std::cout << " (" << build_report.without_embedding
                      << " document(s) have no embedding and are unreachable by the semantic "
                         "path)";
        }
        std::cout << "\n";
    }

    SweepPoint base;
    base.candidate_k = args.candidate_k;
    base.ef_search = args.ef_search;
    base.bm25.k1 = args.k1;
    base.bm25.b = args.b;
    base.lexical_weight = args.lexical_weight;
    base.semantic_weight = args.semantic_weight;

    EngineOptions engine_options;
    engine_options.mode = args.mode;
    engine_options.semantic = backend;
    engine_options.ann = hnsw.get();
    engine_options.use_result_cache = !args.no_cache;

    VectorTable query_vectors;
    if (!args.query_embeddings.empty()) {
        Result<VectorTable> loaded = LoadVectors(args.query_embeddings);
        if (!loaded.ok()) {
            std::cerr << loaded.status().ToString() << "\n";
            return 1;
        }
        query_vectors = loaded.take();
    }

    if (!args.single_query.empty()) {
        // Single-query mode keys the embedding table by query text as well as by id, so a
        // demo can ask for a generated query by its words.
        VectorTable by_text = query_vectors;
        if (!args.queries.empty() && !args.qrels.empty()) {
            Result<std::vector<JudgedQuery>> judged = LoadJudgements(args.queries, args.qrels);
            if (judged.ok()) {
                for (const auto& query : judged.value()) {
                    auto it = query_vectors.find(query.id);
                    if (it != query_vectors.end()) by_text.emplace(query.text, it->second);
                }
            }
        }
        return RunSingleQuery(args, &index, base, engine_options, by_text);
    }

    Result<std::vector<JudgedQuery>> judged = LoadJudgements(args.queries, args.qrels);
    if (!judged.ok()) {
        std::cerr << judged.status().ToString() << "\n";
        return 1;
    }
    const std::vector<JudgedQuery> queries = judged.take();
    uint64_t judgement_count = 0;
    for (const auto& query : queries) judgement_count += query.judgements.size();
    std::cout << "judged set: " << queries.size() << " queries, " << judgement_count
              << " judgements\n";

    if (!args.ann_recall_efs.empty()) {
        if (hnsw == nullptr) {
            std::cerr << "--ann-recall needs document embeddings; pass --embeddings\n";
            return 1;
        }
        return RunAnnRecall(args, *hnsw, queries, query_vectors);
    }

    Result<std::vector<SweepPoint>> points = BuildSweep(args, base);
    if (!points.ok()) {
        std::cerr << points.status().ToString() << "\n";
        return 2;
    }
    if (backend == SemanticBackend::kHnsw) {
        // hnsw.hpp clamps ef_search up to k, and the k the semantic stage asks for is
        // candidate_k, not top_k. So every configured ef_search below candidate_k collapses
        // to candidate_k, and an ef_search sweep run at the default candidate_k of 200
        // produces seven identical rows for the same reason the exact stage used to: the
        // parameter reaches code that ignores it. Naming the threshold is what keeps that
        // from being rediscovered as a mystery.
        uint32_t clamped = 0;
        uint32_t floor_k = 0;
        for (const SweepPoint& point : points.value()) {
            if (point.ef_search < point.candidate_k) {
                ++clamped;
                floor_k = std::max(floor_k, point.candidate_k);
            }
        }
        if (clamped > 0) {
            std::cout << "note: " << clamped << " of " << points.value().size()
                      << " configuration(s) set ef_search below candidate_k, so the beam they "
                         "actually run is candidate_k — up to "
                      << floor_k
                      << ". The semantic stage asks the graph for candidate_k neighbours and a "
                         "beam cannot hold fewer results than that, so ef_search only takes "
                         "effect above candidate_k.\n";
        }
    }

    if (args.sweep_axis == "ef_search" && backend == SemanticBackend::kExact) {
        // The axis is live under --semantic hnsw and inert under --semantic exact, and which
        // one is running is not visible from the table. Saying so is the difference between a
        // reader concluding "search effort does not matter" and "this row did not use it".
        std::cout << "note: --semantic exact is brute force, so ef_search reaches no code. "
                     "These rows will be identical by construction, which is a fact about "
                     "this stage and not about ANN search effort. Use --semantic hnsw.\n";
    }

    eval::SweepOptions sweep_options;
    sweep_options.latency_tolerance = args.tolerance;
    sweep_options.harness.k = args.k;
    sweep_options.harness.top_k = args.top_k;
    sweep_options.harness.candidate_k = args.candidate_k;
    sweep_options.harness.deadline_ms = args.deadline_ms;
    sweep_options.harness.repetitions = args.repetitions;
    sweep_options.harness.cold_cache = args.cold;
    sweep_options.harness.embedding_for = [&query_vectors](
                                              const std::string& id) -> const std::vector<float>* {
        auto it = query_vectors.find(id);
        return it == query_vectors.end() ? nullptr : &it->second;
    };

    eval::RunnerFactory factory =
        [&](const SweepPoint& point) -> Result<std::unique_ptr<QueryRunner>> {
        return std::unique_ptr<QueryRunner>(
            new EngineRunner(&index, point, engine_options, eval::DescribePoint(point)));
    };

    Result<eval::SweepResult> result = eval::RunSweep(points.value(), factory, queries, sweep_options);
    if (!result.ok()) {
        std::cerr << result.status().ToString() << "\n";
        return 1;
    }

    std::cout << "\nmode: " << ModeName(args.mode) << ", semantic " << BackendName(backend);
    if (backend == SemanticBackend::kHnsw && args.sweep_axis != "ef_search") {
        std::cout << " ef_search " << args.ef_search;
    }
    std::cout << ", k = " << args.k << ", latency tolerance "
              << Fixed(args.tolerance * 100.0, 0) << "% on p99"
              << (args.cold ? ", cold cache" : "") << "\n\n";

    if (result.value().rows.size() == 1) {
        PrintRun(result.value().rows.front().run);
        if (args.show_per_query) PrintPerQuery(result.value().rows.front().run, args.per_query_limit);
    } else {
        std::cout << eval::FormatTable(result.value());
        std::cout << "\nverdicts against the baseline:\n" << eval::FormatVerdicts(result.value());
        size_t tradeoffs = 0;
        for (size_t i = 0; i < result.value().rows.size(); ++i) {
            if (i == result.value().baseline_index) continue;
            if (result.value().rows[i].vs_baseline.verdict == Verdict::kTradeoff) ++tradeoffs;
        }
        if (tradeoffs > 0) {
            std::cout << "\n" << tradeoffs
                      << " row(s) improve quality and regress p99. Those are decisions about "
                         "the latency budget, not results.\n";
        }
    }

    if (result.value().rows.front().run.quality.unjudged_in_top_k > 0) {
        std::cout << "\nnote: " << result.value().rows.front().run.quality.unjudged_in_top_k
                  << " returned document(s) in the baseline run carried no judgement. They "
                     "score as irrelevant, so the quality figures are a lower bound.\n";
    }

    if (!args.json.empty()) {
        Status written = WriteJson(args.json, result.value(), args.k);
        if (!written.ok()) {
            std::cerr << written.ToString() << "\n";
            return 1;
        }
        std::cout << "\nwrote " << args.json << "\n";
    }
    return 0;
}

}  // namespace
}  // namespace meridian::tools

int main(int argc, char** argv) { return meridian::tools::Main(argc, argv); }
