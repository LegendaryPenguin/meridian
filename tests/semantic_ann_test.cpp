// The HNSW semantic stage inside the serving pipeline.
//
// hnsw_test.cpp establishes that the graph is a good approximate index. serve_test.cpp
// establishes that the pipeline serves queries. Neither says anything about the two being
// connected, and for most of this project's life they were not: the serving and eval paths
// ran exact brute force, so `ef_search` reached no code and the graph was exercised only by
// its own unit tests. These tests are about the join.
//
// The one worth reading is TheSemanticBackendSwitchChangesWhatRuns. A selector flag that
// silently resolves to the same behaviour either way is exactly the defect this wiring
// exists to fix, so it is pinned by a test rather than by the fact that it currently works:
// the exact stage must touch the graph zero times and the approximate stage must touch a
// small fraction of the corpus, and both are counted rather than assumed.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "../src/ann/hnsw.hpp"
#include "../src/serve/reference_index.hpp"
#include "../src/serve/semantic_ann.hpp"
#include "../src/serve/stages_engine.hpp"
#include "testing.hpp"

using namespace meridian;
using namespace meridian::serve;

namespace {

/// Clustered vectors, for the reason hnsw_test.cpp gives at length: uniformly random points
/// in high dimensions are all nearly equidistant, so "the top 10" is an arbitrary slice that
/// any graph stumbles into and recall comes out near 1.0 even for a broken index. Clusters
/// give the search something it can genuinely fail to find.
///
/// SplitMix64 rather than <random>, because std::normal_distribution is not specified to
/// produce the same sequence across standard libraries and a recall threshold measured
/// against one implementation would fail on another.
class Clustered {
public:
    Clustered(uint32_t dim, uint32_t clusters, uint64_t seed) : dim_(dim), rng_(seed) {
        centers_.resize(clusters);
        for (auto& center : centers_) {
            center.resize(dim_);
            for (float& x : center) x = static_cast<float>(Gaussian());
        }
    }

    std::vector<float> Draw(double sigma) {
        const std::vector<float>& center = centers_[rng_.Next() % centers_.size()];
        std::vector<float> v(dim_);
        for (uint32_t d = 0; d < dim_; ++d) {
            v[d] = static_cast<float>(center[d] + sigma * Gaussian());
        }
        return v;
    }

private:
    double Gaussian() {
        const double u1 = rng_.NextUnitHalfOpen();
        const double u2 = rng_.NextUnitHalfOpen();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    }

    uint32_t dim_;
    ann::SplitMix64 rng_;
    std::vector<std::vector<float>> centers_;
};

ReferenceIndex::Document MakeDocument(const std::string& id, int serial,
                                     std::vector<float> embedding) {
    ReferenceIndex::Document doc;
    doc.meta.external_id = id;
    doc.meta.language = (serial % 3 == 0) ? "fr" : "en";
    doc.meta.category = (serial % 2 == 0) ? "systems" : "ir";
    doc.meta.popularity = 0.0;  // zero so the reranker cannot reorder on anything but fusion
    doc.tokens[Field::kTitle] = Tokenize("document " + std::to_string(serial));
    doc.tokens[Field::kBody] = Tokenize("vector embedding neighbour " + std::to_string(serial));
    doc.embedding = std::move(embedding);
    return doc;
}

struct Corpus {
    ReferenceIndex index;
    std::vector<std::vector<float>> queries;
};

/// A corpus plus a set of queries drawn from the same distribution. Queries from the data's
/// own distribution are the harder case: a query far from everything has its neighbour
/// ordering dominated by one direction, which flatters the index.
std::unique_ptr<Corpus> MakeCorpus(uint32_t documents, uint32_t dim, uint32_t clusters,
                                   uint32_t queries, double sigma = 0.35,
                                   uint32_t docs_per_segment = 250, uint64_t seed = 0xC0FFEEULL) {
    auto corpus = std::make_unique<Corpus>();
    Clustered source(dim, clusters, seed);
    for (uint32_t i = 0; i < documents; ++i) {
        char id[32];
        std::snprintf(id, sizeof(id), "doc-%05u", i);
        corpus->index.Add(MakeDocument(id, static_cast<int>(i), source.Draw(sigma)));
        if ((i + 1) % docs_per_segment == 0) corpus->index.Flush();
    }
    corpus->index.Flush();
    for (uint32_t i = 0; i < queries; ++i) corpus->queries.push_back(source.Draw(sigma));
    return corpus;
}

/// One corpus and one graph, built once and shared by the tests that only read them.
///
/// Shared because the parameters are what make these measurements mean anything, and they
/// are expensive. Few clusters with a spread comparable to the distance between their
/// centres is the case that is genuinely hard: well-separated clusters at this corpus size
/// give the graph recall above 0.99 at its narrowest beam, so every threshold below would
/// read as satisfied no matter how the wiring regressed. Ten thousand vectors is the other
/// half — at two thousand, a beam of ten over a degree-32 layer already reaches a large
/// fraction of the corpus, and recall is near 1.0 for the uninteresting reason.
///
/// Rebuilding it per test would spend several seconds proving that an unchanged build
/// produces an unchanged graph. Tests that mutate an index build their own.
struct Shared {
    std::unique_ptr<Corpus> corpus;
    std::unique_ptr<ann::HnswIndex> hnsw;
};

const Shared& SharedFixture() {
    static const Shared shared = [] {
        Shared built;
        built.corpus = MakeCorpus(10'000, 32, 4, 32, /*sigma=*/1.0);
        auto index = BuildHnswOverSnapshot(*built.corpus->index.Current(), ann::HnswParams{});
        if (index.ok()) built.hnsw = index.take();
        return built;
    }();
    return shared;
}

/// The five stages with a selectable semantic path, which is the thing under test.
struct Wiring {
    /// `use_result_cache` false is what a test measuring the semantic stage needs: a cached
    /// response skips every stage, so a run that re-asks a question it has already asked
    /// measures a hash lookup and nothing else.
    Wiring(ReferenceIndex* index, const AnnIndex* ann, uint32_t ef_search, bool approximate,
           bool use_result_cache = true)
        : engine_semantic(ann, ef_search), ann_semantic(&engine_semantic) {
        const SemanticStage* semantic =
            approximate ? static_cast<const SemanticStage*>(&ann_semantic)
                        : static_cast<const SemanticStage*>(&exact_semantic);
        PipelineStages stages{&parse, &lexical, semantic, &merge, &rerank, &materialize};
        PipelineOptions options;
        options.use_result_cache = use_result_cache;
        pipeline = std::make_unique<Pipeline>(stages, options, &result_cache, &posting_cache);
        source = index;
    }

    Result<SearchResponse> Run(const Query& query) const {
        return pipeline->Run(query, source->Acquire(), Deadline::AfterMillis(query.deadline_ms));
    }

    QueryResultCache result_cache{1u << 20};
    PostingListCache posting_cache{4u << 20};
    ReferenceParseStage parse;
    ReferenceLexicalStage lexical;
    ReferenceSemanticStage exact_semantic;
    ReferenceMergeStage merge;
    ReferenceRerankStage rerank;
    ReferenceMaterializeStage materialize;
    EngineSemanticStage engine_semantic;
    AnnSemanticStage ann_semantic;
    std::unique_ptr<Pipeline> pipeline;
    ReferenceIndex* source = nullptr;
};

/// `serial` distinguishes the query text, which is load-bearing rather than cosmetic.
///
/// cache.hpp hashes the terms and a hybrid flag into the result cache key, not the embedding:
/// two requests with the same text and different embeddings are documented to be assumed
/// identical, because upstream derives the vector from the text. A test that reused one text
/// across different embeddings would therefore be served the first query's answer from cache
/// and would measure the cache rather than the semantic stage.
///
/// The shared terms are ones the corpus contains, so the lexical path is not empty: what the
/// semantic path contributes *inside a hybrid pipeline* is the question, and a pipeline with
/// one dead path does not answer it.
Query SemanticQuery(const std::vector<float>& embedding, uint32_t top_k, uint32_t candidate_k,
                    size_t serial = 0) {
    Query query;
    query.raw = "vector embedding neighbour q" + std::to_string(serial);
    query.top_k = top_k;
    query.candidate_k = candidate_k;
    query.deadline_ms = 60'000;
    query.embedding = embedding;
    return query;
}

std::vector<DocId> DocsOf(const std::vector<Hit>& hits) {
    std::vector<DocId> docs;
    docs.reserve(hits.size());
    for (const Hit& hit : hits) docs.push_back(hit.doc);
    return docs;
}

double Overlap(const std::vector<DocId>& a, const std::vector<DocId>& b) {
    if (a.empty()) return 1.0;
    const std::unordered_set<DocId> set(b.begin(), b.end());
    size_t shared = 0;
    for (DocId doc : a) {
        if (set.count(doc) != 0) ++shared;
    }
    return static_cast<double>(shared) / static_cast<double>(a.size());
}

}  // namespace

MERIDIAN_TEST(HnswBuildCoversExactlyTheEmbeddedDocuments) {
    auto corpus = MakeCorpus(2'000, 32, 12, 0);
    HnswBuildReport report;
    auto built = BuildHnswOverSnapshot(*corpus->index.Current(), ann::HnswParams{}, &report);
    CHECK_OK(built.ok() ? Status::Ok() : built.status());
    if (!built.ok()) return;

    CHECK_EQ(report.vectors, 2'000u);
    CHECK_EQ(report.dim, 32u);
    CHECK_EQ(report.without_embedding, 0u);
    CHECK_EQ(built.value()->size(), size_t{2'000});
}

MERIDIAN_TEST(HnswBackedStageReturnsSaneResultsThroughTheWholePipeline) {
    const Shared& shared = SharedFixture();
    CHECK(shared.hnsw != nullptr);
    if (shared.hnsw == nullptr) return;
    Corpus* corpus = shared.corpus.get();

    Wiring wiring(&corpus->index, shared.hnsw.get(), 128, /*approximate=*/true);
    for (size_t q = 0; q < corpus->queries.size(); ++q) {
        Result<SearchResponse> response =
            wiring.Run(SemanticQuery(corpus->queries[q], 10, 100, q));
        CHECK_OK(response.ok() ? Status::Ok() : response.status());
        if (!response.ok()) continue;
        const SearchResponse& value = response.value();

        CHECK_EQ(value.hits.size(), size_t{10});
        CHECK(value.trace.semantic_candidates > 0);
        CHECK(value.trace.ann_ns > 0);
        for (const Hit& hit : value.hits) {
            // A materialized external id is the end-to-end check: the candidate came out of
            // the graph as a global document id, survived fusion and reranking, and resolved
            // in the pinned generation. An id-space mistake anywhere in that chain shows up
            // here as an empty string or a resolve failure rather than as a plausible answer.
            CHECK(!hit.external_id.empty());
            CHECK(hit.features.count("vector_similarity") == 1);
        }
        // Scores are non-increasing: the reranker sorts, and a semantic path that handed back
        // a mis-ordered list would otherwise be invisible.
        for (size_t i = 1; i < value.hits.size(); ++i) {
            CHECK(value.hits[i - 1].score >= value.hits[i].score);
        }
    }
}

MERIDIAN_TEST(ExactAndHnswAgreeAtHighEfSearch) {
    const Shared& shared = SharedFixture();
    CHECK(shared.hnsw != nullptr);
    if (shared.hnsw == nullptr) return;
    Corpus* corpus = shared.corpus.get();

    // ef_search is set above candidate_k on purpose. The semantic stage asks the graph for
    // candidate_k neighbours and hnsw.hpp clamps ef_search up to that, so a beam set below
    // candidate_k is not a beam the stage can run — a test that asked for ef_search=400 with
    // candidate_k=500 would be measuring 500 and proving nothing about 400.
    Wiring approximate(&corpus->index, shared.hnsw.get(), 400, /*approximate=*/true);
    Wiring exact(&corpus->index, shared.hnsw.get(), 400, /*approximate=*/false);

    double total = 0.0;
    for (size_t q = 0; q < corpus->queries.size(); ++q) {
        Result<SearchResponse> a = approximate.Run(SemanticQuery(corpus->queries[q], 10, 50, q));
        Result<SearchResponse> e = exact.Run(SemanticQuery(corpus->queries[q], 10, 50, q));
        CHECK_OK(a.ok() ? Status::Ok() : a.status());
        CHECK_OK(e.ok() ? Status::Ok() : e.status());
        if (!a.ok() || !e.ok()) continue;
        total += Overlap(DocsOf(e.value().hits), DocsOf(a.value().hits));
    }
    const double agreement = total / static_cast<double>(corpus->queries.size());
    std::printf("  top-10 agreement with exact search at a beam of 400: %.4f\n", agreement);
    // Not asserted as exactly 1.0. Agreement is bounded below by the graph's recall, which is
    // high but not guaranteed at any finite beam width, and a threshold of 1.0 would make this
    // test a coin flip on the data. 0.98 is far above what the same graph gives at its
    // narrowest beam, so a regression in the approximate path still fails here.
    CHECK(agreement >= 0.98);
}

MERIDIAN_TEST(TheSemanticBackendSwitchChangesWhatRuns) {
    const Shared& shared = SharedFixture();
    CHECK(shared.hnsw != nullptr);
    if (shared.hnsw == nullptr) return;
    Corpus* corpus = shared.corpus.get();
    const ann::HnswIndex& hnsw = *shared.hnsw;

    Wiring approximate(&corpus->index, &hnsw, 64, /*approximate=*/true);
    Wiring exact(&corpus->index, &hnsw, 64, /*approximate=*/false);

    // The exact stage must not touch the graph at all. This is the assertion that would have
    // failed for the whole period when the semantic stage was brute force regardless of
    // configuration: the switch existed in spirit and selected nothing.
    hnsw.ResetStats();
    Result<SearchResponse> exact_response = exact.Run(SemanticQuery(corpus->queries[0], 10, 64));
    CHECK_OK(exact_response.ok() ? Status::Ok() : exact_response.status());
    CHECK_EQ(hnsw.stats().distance_computations, uint64_t{0});

    hnsw.ResetStats();
    Result<SearchResponse> approximate_response =
        approximate.Run(SemanticQuery(corpus->queries[0], 10, 64, 1));
    CHECK_OK(approximate_response.ok() ? Status::Ok() : approximate_response.status());
    const uint64_t distances = hnsw.stats().distance_computations;
    std::printf("  distance computations: exact stage 0, hnsw stage %llu, corpus %zu\n",
                static_cast<unsigned long long>(distances), hnsw.size());
    CHECK(distances > 0);
    // Strictly fewer than a scan of the corpus, which is the only reason to run a graph at
    // all. A wrapper that quietly fell back to brute force would satisfy "greater than zero"
    // and fail this.
    CHECK(distances < hnsw.size());

    // And the two backends must disagree somewhere, or the switch is cosmetic. The beam is
    // dropped to 10 to match candidate_k, because ef_search is clamped up to the number of
    // neighbours asked for: at the ef_search=64 used above the graph is accurate enough that
    // every query agrees, and a test that read that as "the switch does nothing" would be
    // reading the clamp. Measured over the whole query set rather than per query — most
    // queries agree even at the narrowest beam, and demanding a disagreement from any
    // particular one would be a test that fails on a better graph.
    Wiring narrow(&corpus->index, &hnsw, 10, /*approximate=*/true);
    double worst = 1.0;
    for (size_t q = 0; q < corpus->queries.size(); ++q) {
        Result<SearchResponse> a = narrow.Run(SemanticQuery(corpus->queries[q], 10, 10, q));
        Result<SearchResponse> e = exact.Run(SemanticQuery(corpus->queries[q], 10, 10, q));
        if (!a.ok() || !e.ok()) continue;
        worst = std::min(worst, Overlap(DocsOf(e.value().hits), DocsOf(a.value().hits)));
    }
    std::printf("  worst per-query top-10 agreement at the narrowest beam: %.4f\n", worst);
    CHECK(worst < 1.0);
}

MERIDIAN_TEST(RecallRisesWithEfSearchThroughTheServingPath) {
    const Shared& shared = SharedFixture();
    CHECK(shared.hnsw != nullptr);
    if (shared.hnsw == nullptr) return;
    Corpus* corpus = shared.corpus.get();

    Wiring exact(&corpus->index, shared.hnsw.get(), 10, /*approximate=*/false);

    // Ground truth from the exact stage, through the same pipeline, so the comparison is
    // between two semantic stages and not between two pipelines.
    std::vector<std::vector<DocId>> truth;
    for (size_t q = 0; q < corpus->queries.size(); ++q) {
        Result<SearchResponse> e = exact.Run(SemanticQuery(corpus->queries[q], 10, 10, q));
        CHECK_OK(e.ok() ? Status::Ok() : e.status());
        truth.push_back(e.ok() ? DocsOf(e.value().hits) : std::vector<DocId>{});
    }

    double previous = 0.0;
    for (uint32_t ef : {10u, 40u, 160u, 640u}) {
        Wiring approximate(&corpus->index, shared.hnsw.get(), ef, /*approximate=*/true);
        double total = 0.0;
        for (size_t q = 0; q < corpus->queries.size(); ++q) {
            Result<SearchResponse> a =
                approximate.Run(SemanticQuery(corpus->queries[q], 10, 10, q));
            if (!a.ok()) continue;
            total += Overlap(truth[q], DocsOf(a.value().hits));
        }
        const double recall = total / static_cast<double>(corpus->queries.size());
        std::printf("  ef_search %4u: top-10 agreement with exact %.4f\n", ef, recall);
        // Monotone within a tolerance rather than strictly. Recall in ef is monotone for the
        // graph's own top-k (hnsw_test.cpp asserts that directly); here the comparison is
        // after fusion and reranking, where a candidate the wider beam added can displace one
        // the narrower beam happened to rank tenth. The tolerance admits that without
        // admitting a real regression.
        CHECK(recall >= previous - 0.02);
        previous = recall;
    }
    CHECK(previous >= 0.95);
}

MERIDIAN_TEST(ApproximatePathHonoursTombstonesAndFilters) {
    auto corpus = MakeCorpus(1'000, 24, 8, 4);
    auto built = BuildHnswOverSnapshot(*corpus->index.Current(), ann::HnswParams{});
    CHECK_OK(built.ok() ? Status::Ok() : built.status());
    if (!built.ok()) return;
    std::unique_ptr<ann::HnswIndex> hnsw = built.take();

    Wiring approximate(&corpus->index, hnsw.get(), 200, /*approximate=*/true);

    // The graph was built before these deletes and knows nothing about them. An ANN index has
    // no tombstones, so without the serving layer applying them a deleted document is
    // returned as an ordinary neighbour — a wrong answer that looks entirely normal.
    std::unordered_set<std::string> deleted;
    for (int i = 0; i < 200; ++i) {
        char id[32];
        std::snprintf(id, sizeof(id), "doc-%05d", i * 5);
        CHECK(corpus->index.Delete(id));
        deleted.insert(id);
    }

    for (size_t q = 0; q < corpus->queries.size(); ++q) {
        Result<SearchResponse> response =
            approximate.Run(SemanticQuery(corpus->queries[q], 20, 200, q));
        CHECK_OK(response.ok() ? Status::Ok() : response.status());
        if (!response.ok()) continue;
        for (const Hit& hit : response.value().hits) {
            CHECK(deleted.count(hit.external_id) == 0);
        }
    }

    // Same argument for filters: the predicate is evaluated against DocumentMeta, which the
    // graph does not hold, so a filtered query would otherwise fuse ineligible documents into
    // the final list and rank them above eligible ones.
    for (size_t q = 0; q < corpus->queries.size(); ++q) {
        Query query = SemanticQuery(corpus->queries[q], 20, 200, q);
        query.filters.language = "fr";
        Result<SearchResponse> response = approximate.Run(query);
        CHECK_OK(response.ok() ? Status::Ok() : response.status());
        if (!response.ok()) continue;
        CHECK(!response.value().hits.empty());
        for (const Hit& hit : response.value().hits) {
            const auto* doc = corpus->index.Current()->Resolve(hit.doc);
            CHECK(doc != nullptr);
            if (doc != nullptr) CHECK_EQ(doc->meta.language, std::string("fr"));
        }
    }
}

MERIDIAN_TEST(ConcurrentQueriesThroughTheApproximatePathAgreeWithSerialOnes) {
    auto corpus = MakeCorpus(1'500, 32, 10, 8);
    auto built = BuildHnswOverSnapshot(*corpus->index.Current(), ann::HnswParams{});
    CHECK_OK(built.ok() ? Status::Ok() : built.status());
    if (!built.ok()) return;
    std::unique_ptr<ann::HnswIndex> hnsw = built.take();

    Wiring approximate(&corpus->index, hnsw.get(), 100, /*approximate=*/true,
                       /*use_result_cache=*/false);

    std::vector<std::vector<DocId>> serial;
    for (size_t q = 0; q < corpus->queries.size(); ++q) {
        Result<SearchResponse> response =
            approximate.Run(SemanticQuery(corpus->queries[q], 10, 100, q));
        CHECK_OK(response.ok() ? Status::Ok() : response.status());
        serial.push_back(response.ok() ? DocsOf(response.value().hits) : std::vector<DocId>{});
    }

    // hnsw.hpp states that Search is not thread-safe: it stamps a shared visited array, so
    // two concurrent searches corrupt each other's traversal and return the wrong neighbours
    // without failing. The serving layer serialises for that reason, and this is the test
    // that says so — without the lock it returns mismatched lists here and a data race under
    // the sanitizer.
    std::atomic<int> mismatches{0};
    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&, t] {
            for (int round = 0; round < 12; ++round) {
                const size_t q =
                    (static_cast<size_t>(t) + static_cast<size_t>(round)) % corpus->queries.size();
                Result<SearchResponse> response =
                    approximate.Run(SemanticQuery(corpus->queries[q], 10, 100, q));
                if (!response.ok()) {
                    ++failures;
                    continue;
                }
                if (DocsOf(response.value().hits) != serial[q]) ++mismatches;
            }
        });
    }
    for (auto& thread : threads) thread.join();
    CHECK_EQ(failures.load(), 0);
    CHECK_EQ(mismatches.load(), 0);
}

MERIDIAN_TEST(ACorpusWithNoEmbeddingsIsRefusedRatherThanBuiltAtZeroDimensions) {
    ReferenceIndex index;
    for (int i = 0; i < 20; ++i) {
        index.Add(MakeDocument("doc-" + std::to_string(i), i, {}));
    }
    index.Flush();

    HnswBuildReport report;
    auto built = BuildHnswOverSnapshot(*index.Current(), ann::HnswParams{}, &report);
    // Refused, because a zero-dimension index rejects every real query with an argument
    // error that blames the query. The failure has to name the corpus instead.
    CHECK(!built.ok());
    if (!built.ok()) {
        CHECK(built.status().code() == Code::kInvalidArgument);
        CHECK(built.status().message().find("no embedded documents") != std::string::npos);
    }
}

MERIDIAN_TEST(MixedEmbeddingDimensionsFailAndNameTheDocument) {
    ReferenceIndex index;
    index.Add(MakeDocument("doc-wide", 0, std::vector<float>(16, 0.5f)));
    index.Add(MakeDocument("doc-narrow", 1, std::vector<float>(8, 0.5f)));
    index.Flush();

    auto built = BuildHnswOverSnapshot(*index.Current(), ann::HnswParams{});
    // Two embedding dimensions in one corpus is a data-preparation bug. Dropping the minority
    // silently would show up only as recall that nobody can account for.
    CHECK(!built.ok());
    if (!built.ok()) {
        CHECK(built.status().message().find("doc-narrow") != std::string::npos);
    }
}

MERIDIAN_TEST(DeletedDocumentsAreLeftOutOfTheGraphWhenItIsRebuilt) {
    auto corpus = MakeCorpus(500, 16, 6, 2);
    CHECK(corpus->index.Delete("doc-00000"));
    CHECK(corpus->index.Delete("doc-00001"));

    HnswBuildReport report;
    auto built = BuildHnswOverSnapshot(*corpus->index.Current(), ann::HnswParams{}, &report);
    CHECK_OK(built.ok() ? Status::Ok() : built.status());
    if (!built.ok()) return;
    // The build reads one snapshot, so a document tombstoned in that snapshot never enters
    // the graph. Post-filtering in the stage covers deletes that happen after the build;
    // this covers the ones that happened before, and the two together are why a deleted
    // document cannot be returned by the semantic path at any point in a rebuild cycle.
    CHECK_EQ(report.vectors, 498u);
    CHECK_EQ(built.value()->size(), size_t{498});
}

MERIDIAN_TEST_MAIN()
