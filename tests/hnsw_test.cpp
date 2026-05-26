// HNSW tests. The deliverable here is a number, not a pass.
//
// An approximate index cannot be tested the way a posting list can. There is no
// "correct" output to compare against — the whole point of the structure is to be
// wrong sometimes, cheaply — so what the tests do instead is measure how wrong, at
// what cost, and check that the shape of that tradeoff is the one the algorithm is
// supposed to produce: recall rising with effort, approaching 1.0, with distance
// counts far below brute force. A suite that only asserted "returns k results" would
// pass against a flat scan wrapped in a graph, and would pass against a graph so
// badly connected that recall topped out at 0.6. Both are reported here instead.
//
// Run the binary directly, or `ctest -V`, to see the tables.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "../src/ann/hnsw.hpp"
#include "../src/ann/persist.hpp"
#include "testing.hpp"

using meridian::Candidate;
using meridian::Code;
using meridian::DocId;
using meridian::Status;
using meridian::ann::HnswIndex;
using meridian::ann::HnswParams;
using meridian::ann::Metric;
using meridian::ann::NeighborSelection;
using meridian::ann::SplitMix64;

namespace {

/// Standard normal deviates by Box-Muller, on the project's own RNG.
///
/// Rolled by hand for the same reason the index's RNG is: std::normal_distribution
/// is not specified to produce the same numbers across standard library
/// implementations, so a recall figure measured against it would not be comparable
/// between a developer's machine and CI.
class Gaussian {
public:
    explicit Gaussian(uint64_t seed) : rng_(seed) {}

    double Next() {
        if (have_spare_) {
            have_spare_ = false;
            return spare_;
        }
        const double u1 = rng_.NextUnitHalfOpen();
        const double u2 = rng_.NextUnitHalfOpen();
        const double r = std::sqrt(-2.0 * std::log(u1));
        const double theta = 6.283185307179586 * u2;
        spare_ = r * std::sin(theta);
        have_spare_ = true;
        return r * std::cos(theta);
    }

    uint64_t Bits() { return rng_.Next(); }

private:
    SplitMix64 rng_;
    bool have_spare_ = false;
    double spare_ = 0.0;
};

struct Dataset {
    uint32_t dim = 0;
    std::vector<std::vector<float>> vectors;
    std::vector<std::vector<float>> queries;
};

/// Clustered data, not uniform random.
///
/// This choice decides whether the recall number means anything. In high dimensions
/// uniformly random points are all nearly the same distance from each other — the
/// ratio between the nearest and the hundredth-nearest neighbour approaches 1 — so
/// "the top 10" is an almost arbitrary slice of a flat distribution, any ten of which
/// a graph search will stumble into. Recall measured that way comes out near 1.0 for
/// almost any graph, including a deliberately broken one, which makes the benchmark a
/// advertisement rather than a measurement. Real embeddings are strongly clustered:
/// there are a few genuinely nearest neighbours and a sharp fall-off, finding them
/// requires getting into the right cluster, and that is the thing a proximity graph
/// can fail at.
Dataset MakeClustered(uint32_t n, uint32_t dim, uint32_t clusters, double sigma,
                      uint32_t query_count, uint64_t seed) {
    Gaussian g(seed);
    Dataset ds;
    ds.dim = dim;

    std::vector<std::vector<float>> centers(clusters, std::vector<float>(dim));
    for (auto& c : centers) {
        for (float& x : c) x = static_cast<float>(g.Next());
    }

    ds.vectors.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        const std::vector<float>& c = centers[g.Bits() % clusters];
        std::vector<float> v(dim);
        for (uint32_t d = 0; d < dim; ++d) {
            v[d] = static_cast<float>(c[d] + sigma * g.Next());
        }
        ds.vectors.push_back(std::move(v));
    }

    // Queries drawn from the same distribution as the data. An out-of-distribution
    // query is an easier problem — it is far from everything, so the ordering of its
    // neighbours is dominated by one direction — and would flatter the index.
    ds.queries.reserve(query_count);
    for (uint32_t i = 0; i < query_count; ++i) {
        const std::vector<float>& c = centers[g.Bits() % clusters];
        std::vector<float> v(dim);
        for (uint32_t d = 0; d < dim; ++d) {
            v[d] = static_cast<float>(c[d] + sigma * g.Next());
        }
        ds.queries.push_back(std::move(v));
    }
    return ds;
}

struct RecallPoint {
    uint32_t ef = 0;
    double recall = 0.0;
    double distance_computations = 0.0;
    double nodes_visited = 0.0;
    double hops = 0.0;
};

using GroundTruth = std::vector<std::unordered_set<DocId>>;

/// The true top-k for each query, from brute force.
///
/// Computed once per dataset and reused across every ef_search value, which is not
/// only cheaper but more honest: every point on the curve is scored against exactly
/// the same answer key, and so is every variant of the index built over the same
/// vectors.
///
/// Ground truth comes from SearchExact rather than from a separately written scan,
/// deliberately: the comparison is then apples to apples on the same vectors and the
/// same metric, and a bug in the distance function would move both sides rather than
/// silently inflating recall.
GroundTruth ComputeGroundTruth(const HnswIndex& index, const Dataset& ds, uint32_t k) {
    GroundTruth truth;
    truth.reserve(ds.queries.size());
    for (const std::vector<float>& q : ds.queries) {
        auto exact = index.SearchExact(q, k);
        std::unordered_set<DocId> expected;
        if (!exact.ok()) {
            std::printf("  SearchExact failed: %s\n", exact.status().ToString().c_str());
        } else {
            for (const Candidate& c : exact.value()) expected.insert(c.doc);
        }
        truth.push_back(std::move(expected));
    }
    return truth;
}

/// Recall@k against brute force, averaged over the query set.
///
/// Recall is computed over document ids, intersected as a set. Comparing ordered
/// lists would conflate two different failures — missing a neighbour, and finding it
/// but ranking it third instead of second — and only the first is the index's job.
/// Ties are the reason this matters in practice: with duplicate vectors several
/// different sets are all exactly correct.
RecallPoint MeasureRecall(const HnswIndex& index, const Dataset& ds, const GroundTruth& truth,
                          uint32_t k, uint32_t ef) {
    RecallPoint point;
    point.ef = ef;

    double hit_fraction_sum = 0.0;
    uint64_t queries = 0;
    meridian::AnnIndex::Stats total{};

    for (size_t qi = 0; qi < ds.queries.size(); ++qi) {
        const std::vector<float>& q = ds.queries[qi];
        const std::unordered_set<DocId>& expected = truth[qi];

        index.ResetStats();
        auto got = index.Search(q, k, ef);
        if (!got.ok()) {
            std::printf("  Search failed: %s\n", got.status().ToString().c_str());
            return point;
        }
        const meridian::AnnIndex::Stats s = index.stats();
        total.distance_computations += s.distance_computations;
        total.nodes_visited += s.nodes_visited;
        total.hops += s.hops;

        uint32_t hits = 0;
        for (const Candidate& c : got.value()) {
            if (expected.count(c.doc) != 0) ++hits;
        }
        if (!expected.empty()) {
            hit_fraction_sum += double(hits) / double(expected.size());
            ++queries;
        }
    }

    if (queries == 0) return point;
    point.recall = hit_fraction_sum / double(queries);
    point.distance_computations = double(total.distance_computations) / double(queries);
    point.nodes_visited = double(total.nodes_visited) / double(queries);
    point.hops = double(total.hops) / double(queries);
    return point;
}

Status BuildIndex(HnswIndex& index, const Dataset& ds) {
    for (size_t i = 0; i < ds.vectors.size(); ++i) {
        MERIDIAN_TRY(index.Add(static_cast<DocId>(i), ds.vectors[i]));
    }
    return index.Build();
}

std::vector<DocId> DocsOf(const std::vector<Candidate>& candidates) {
    std::vector<DocId> docs;
    docs.reserve(candidates.size());
    for (const Candidate& c : candidates) docs.push_back(c.doc);
    return docs;
}

// The sweep configuration, and these numbers were chosen by measurement rather than
// by taste.
//
// The first attempt used n=6000, dim=64, 40 tight clusters and M=16, and reported
// recall@10 of 1.0000 at every ef_search from 16 upwards. That is a useless
// benchmark: the curve has no shape, so it cannot distinguish a good graph from a bad
// one, and it would have reported 1.0 for an index with serious connectivity
// problems. The size and dimension were raised and M lowered until the dial had real
// range — recall@10 climbs from roughly 0.7 to 1.0 across the ef sweep — because a
// benchmark that cannot fail is not measuring anything.
//
// The same sizes are used under a sanitiser. ASAN costs about 3x here, not the 10x
// that would have justified a smaller dataset, and shrinking it would have flattened
// the curve back out and left the sanitiser build asserting something weaker than the
// normal build.
constexpr uint32_t kSweepN = 20000;
constexpr uint32_t kSweepDim = 128;
constexpr uint32_t kSweepClusters = 50;
constexpr uint32_t kSweepQueries = 80;
constexpr double kSweepSigma = 0.5;

std::string TempPath(const char* name) {
    const char* dir = std::getenv("TMPDIR");
    std::string base = dir != nullptr ? dir : "/tmp";
    if (!base.empty() && base.back() != '/') base.push_back('/');
    return base + "meridian_hnsw_test_" + name;
}

}  // namespace

MERIDIAN_TEST(RecallRisesWithEfSearchAndApproachesOne) {
    const Dataset ds = MakeClustered(kSweepN, kSweepDim, kSweepClusters, kSweepSigma,
                                     kSweepQueries, 0xA11CE);
    const uint32_t k = 10;
    const std::vector<uint32_t> efs{10, 16, 24, 32, 48, 64, 96, 128, 192, 256};

    // Two degree budgets, because the curve's shape depends on M and reporting only
    // one hides that. M=16 is the conventional default and on this data it saturates
    // almost immediately — the dial has no range left, which is a fact about the data
    // being easier than a real embedding corpus, not evidence that the index is
    // excellent. M=8 is where the tradeoff is visible.
    struct Config {
        uint32_t M;
        uint32_t M0;
    };
    for (const Config& cfg : {Config{8, 16}, Config{16, 32}}) {
        HnswParams params;
        params.M = cfg.M;
        params.M0 = cfg.M0;
        params.ef_construction = 200;
        params.metric = Metric::kCosine;
        params.selection = NeighborSelection::kHeuristic;

        HnswIndex index(kSweepDim, params);
        CHECK_OK(BuildIndex(index, ds));
        CHECK_EQ(index.size(), size_t(kSweepN));

        const GroundTruth truth = ComputeGroundTruth(index, ds, k);

        std::printf("\n  recall@%u vs ef_search  (n=%u, dim=%u, %u clusters, sigma=%.2f, "
                    "M=%u/M0=%u, ef_construction=%u, cosine, heuristic selection)\n",
                    k, kSweepN, kSweepDim, kSweepClusters, kSweepSigma, params.M, params.M0,
                    params.ef_construction);
        std::printf("  %9s %10s %14s %14s %9s %12s\n", "ef_search", "recall@10", "dist/query",
                    "visited/query", "hops", "vs brute");

        std::vector<RecallPoint> curve;
        for (uint32_t ef : efs) {
            const RecallPoint p = MeasureRecall(index, ds, truth, k, ef);
            curve.push_back(p);
            std::printf("  %9u %10.4f %14.1f %14.1f %9.1f %11.1fx\n", p.ef, p.recall,
                        p.distance_computations, p.nodes_visited, p.hops,
                        double(kSweepN) / std::max(1.0, p.distance_computations));
        }
        std::printf("  brute force costs %u distance computations per query by definition; "
                    "the graph has %u layers.\n",
                    kSweepN, index.max_layer() + 1);

        // The curve must rise. A flat or falling curve means the beam is not reaching
        // anything new as it widens, which is what a badly connected graph looks like
        // from the outside, and it is a bug rather than a tuning outcome.
        CHECK(curve.back().recall > curve.front().recall);

        // And it must arrive. If spending 25x the beam width still cannot find 99% of
        // the true top ten, the graph is wrong — the honest response is to fix the
        // graph, not to report the number.
        CHECK(curve.back().recall >= 0.99);

        // Effort has to buy something at the bottom end too: ef == k is the cheapest
        // legal search and should already be useful.
        CHECK(curve.front().recall >= 0.5);

        // And it must be cheap. An index that reaches high recall by visiting most of
        // the corpus is a brute-force scan with extra steps.
        CHECK(curve.back().distance_computations < double(kSweepN) / 4.0);

        // The sparse configuration is the one that has to show a real tradeoff. If
        // this ever starts passing trivially, the dataset got easier and the whole
        // measurement needs revisiting rather than celebrating.
        if (cfg.M == 8) {
            if (curve.front().recall > 0.9) {
                std::printf("  NOTE: recall at ef=k is already %.4f, so this dataset no longer "
                            "exercises the recall/effort tradeoff.\n",
                            curve.front().recall);
            }
            CHECK(curve.front().recall < 0.9);
        }
    }
}

MERIDIAN_TEST(RecallIsMonotoneInEfSearch) {
    // A different data seed and a smaller corpus than the headline sweep: the point is
    // to check that monotonicity is not a property of one lucky dataset.
    const Dataset ds = MakeClustered(8000, 64, 30, 0.5, 60, 0xB0B);
    HnswParams params;
    params.M = 8;
    params.M0 = 16;
    params.ef_construction = 200;
    HnswIndex index(64, params);
    CHECK_OK(BuildIndex(index, ds));
    const GroundTruth truth = ComputeGroundTruth(index, ds, 10);

    double previous = -1.0;
    uint32_t previous_ef = 0;
    for (uint32_t ef : {10u, 14u, 20u, 28u, 40u, 56u, 80u, 160u, 320u}) {
        const RecallPoint p = MeasureRecall(index, ds, truth, 10, ef);
        std::printf("    ef=%-4u recall=%.4f\n", ef, p.recall);
        if (previous >= 0.0) {
            // A wider beam is a superset of the search a narrower one would have done
            // only in expectation, not per query: the termination condition fires at a
            // different point, so one query can lose a tied neighbour. A small
            // tolerance covers that; a real regression is much larger than this.
            if (p.recall < previous - 0.01) {
                std::printf("  recall fell from %.4f at ef=%u to %.4f at ef=%u\n", previous,
                            previous_ef, p.recall, ef);
            }
            CHECK(p.recall >= previous - 0.01);
        }
        previous = p.recall;
        previous_ef = ef;
    }
}

MERIDIAN_TEST(HeuristicSelectionBeatsTopM) {
    // The contrast this project cares about. Same data, same seed, same everything
    // except which candidates become edges.
    const Dataset ds = MakeClustered(kSweepN, kSweepDim, kSweepClusters, kSweepSigma,
                                     kSweepQueries, 0xC0FFEE);

    HnswParams heuristic;
    heuristic.M = 8;
    heuristic.M0 = 16;
    heuristic.ef_construction = 200;
    heuristic.selection = NeighborSelection::kHeuristic;

    HnswParams top_m = heuristic;
    top_m.selection = NeighborSelection::kTopM;

    HnswIndex a(kSweepDim, heuristic);
    HnswIndex b(kSweepDim, top_m);
    CHECK_OK(BuildIndex(a, ds));
    CHECK_OK(BuildIndex(b, ds));

    // One answer key for both, taken from either index — SearchExact does not consult
    // the graph, so the two agree by construction.
    const GroundTruth truth = ComputeGroundTruth(a, ds, 10);

    std::printf("\n  neighbour selection: heuristic vs nearest-M (n=%u, dim=%u, M=%u/M0=%u, "
                "recall@10)\n",
                kSweepN, kSweepDim, heuristic.M, heuristic.M0);
    std::printf("  %9s %12s %12s %10s %14s %14s\n", "ef_search", "heuristic", "top-M", "delta",
                "dist heur", "dist top-M");

    double worst_delta = 1.0;
    for (uint32_t ef : {10u, 32u, 64u, 128u, 256u, 512u}) {
        const RecallPoint ph = MeasureRecall(a, ds, truth, 10, ef);
        const RecallPoint pt = MeasureRecall(b, ds, truth, 10, ef);
        std::printf("  %9u %12.4f %12.4f %+10.4f %14.1f %14.1f\n", ef, ph.recall, pt.recall,
                    ph.recall - pt.recall, ph.distance_computations, pt.distance_computations);
        worst_delta = std::min(worst_delta, ph.recall - pt.recall);
    }

    // Asserted loosely rather than tightly. The claim worth defending is that the
    // heuristic is not worse; how much better it is depends on the data's cluster
    // structure, and pinning a margin here would make the test a measurement of the
    // generator rather than of the algorithm.
    CHECK(worst_delta >= -0.01);
}

MERIDIAN_TEST(EfSearchBelowKIsClampedUpToK) {
    const Dataset ds = MakeClustered(500, 16, 8, 0.3, 1, 0x5EED);
    HnswIndex index(16, HnswParams{});
    CHECK_OK(BuildIndex(index, ds));

    // Documented behaviour: a beam narrower than k is widened to k, because a beam of
    // width ef cannot hold more than ef results and returning four hits for k=10
    // would look like a small index rather than a misconfiguration.
    for (uint32_t ef : {0u, 1u, 3u, 9u}) {
        auto got = index.Search(ds.queries[0], 10, ef);
        CHECK(got.ok());
        CHECK_EQ(got.value().size(), size_t(10));
    }

    // And the clamped search is a real search: ef=0 must give the same answer as
    // ef=k, not a degenerate one.
    auto clamped = index.Search(ds.queries[0], 10, 0);
    auto explicit_k = index.Search(ds.queries[0], 10, 10);
    CHECK(clamped.ok() && explicit_k.ok());
    CHECK(DocsOf(clamped.value()) == DocsOf(explicit_k.value()));
}

MERIDIAN_TEST(EmptyIndexReturnsNothingRatherThanFailing) {
    HnswIndex index(8, HnswParams{});
    CHECK_EQ(index.size(), size_t(0));
    CHECK_EQ(index.entry_point(), HnswIndex::kNoEntryPoint);

    const std::vector<float> q(8, 0.25f);
    auto got = index.Search(q, 10, 64);
    CHECK(got.ok());
    CHECK(got.value().empty());

    auto exact = index.SearchExact(q, 10);
    CHECK(exact.ok());
    CHECK(exact.value().empty());

    // Build on an empty index is legal: an index generation with no vectors in it is
    // an ordinary state during a cold start, not an error.
    CHECK_OK(index.Build());
}

MERIDIAN_TEST(SingleElementIndexReturnsThatElement) {
    HnswIndex index(4, HnswParams{});
    const std::vector<float> v{1.0f, 0.0f, 0.0f, 0.0f};
    CHECK_OK(index.Add(42, v));
    CHECK_OK(index.Build());

    CHECK_EQ(index.size(), size_t(1));
    CHECK_EQ(index.entry_point(), 0u);
    CHECK(index.Neighbors(0, 0).empty());

    auto got = index.Search(v, 10, 64);
    CHECK(got.ok());
    CHECK_EQ(got.value().size(), size_t(1));
    CHECK_EQ(got.value()[0].doc, DocId(42));
    // Cosine of a vector with itself: the similarity reported to the caller must be
    // 1, not 1 minus a distance someone forgot to convert.
    CHECK(std::fabs(got.value()[0].vector_similarity - 1.0) < 1e-5);
    CHECK(std::fabs(got.value()[0].score - 1.0) < 1e-5);
    CHECK(got.value()[0].source == Candidate::Source::kSemantic);
}

MERIDIAN_TEST(KLargerThanTheIndexReturnsEverythingOnce) {
    const Dataset ds = MakeClustered(37, 8, 4, 0.3, 3, 0x1234);
    HnswIndex index(8, HnswParams{});
    CHECK_OK(BuildIndex(index, ds));

    auto got = index.Search(ds.queries[0], 1000, 2000);
    CHECK(got.ok());
    CHECK_EQ(got.value().size(), size_t(37));

    // Every document exactly once. A beam search that failed to mark a node visited
    // would return duplicates and still look like a full result list.
    std::unordered_set<DocId> seen;
    for (const Candidate& c : got.value()) seen.insert(c.doc);
    CHECK_EQ(seen.size(), size_t(37));

    // With k >= n the approximate search has no excuse: it must match brute force
    // exactly, which makes this the one place recall is asserted rather than reported.
    auto exact = index.SearchExact(ds.queries[0], 1000);
    CHECK(exact.ok());
    CHECK(DocsOf(got.value()) == DocsOf(exact.value()));

    // k == 0 is a request for nothing, which is satisfiable.
    auto none = index.Search(ds.queries[0], 0, 64);
    CHECK(none.ok());
    CHECK(none.value().empty());
}

MERIDIAN_TEST(DuplicateVectorsDoNotBreakTheGraph) {
    // Duplicates are the adversarial case for a proximity graph: every candidate is
    // at distance zero from every other, so the heuristic's "closer to a selected
    // neighbour than to the query" test is true for all of them at once, and a naive
    // implementation either builds a star, loops forever, or returns fewer results
    // than it has nodes.
    HnswParams params;
    params.ef_construction = 64;
    HnswIndex index(8, params);

    const std::vector<float> same{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    for (DocId i = 0; i < 300; ++i) CHECK_OK(index.Add(i, same));
    // A handful of distinct vectors so the graph is not degenerate everywhere.
    const Dataset other = MakeClustered(50, 8, 3, 0.3, 2, 0xD00D);
    for (size_t i = 0; i < other.vectors.size(); ++i) {
        CHECK_OK(index.Add(static_cast<DocId>(1000 + i), other.vectors[i]));
    }
    CHECK_OK(index.Build());
    CHECK_EQ(index.size(), size_t(350));

    auto got = index.Search(same, 20, 128);
    CHECK(got.ok());
    CHECK_EQ(got.value().size(), size_t(20));
    // All twenty must be exact duplicates, since 300 of them exist at distance zero.
    for (const Candidate& c : got.value()) {
        CHECK(c.doc < 300);
        CHECK(std::fabs(c.vector_similarity - 1.0) < 1e-5);
    }

    // A zero vector alongside them: cosine has no angle to report, so it must not
    // poison the comparisons it takes part in.
    const std::vector<float> zero(8, 0.0f);
    CHECK_OK(index.Add(9999, zero));
    auto with_zero = index.Search(same, 20, 128);
    CHECK(with_zero.ok());
    CHECK_EQ(with_zero.value().size(), size_t(20));
}

MERIDIAN_TEST(RejectsMalformedVectorsAndQueries) {
    HnswIndex index(4, HnswParams{});
    const std::vector<float> wrong_size{1.0f, 2.0f};
    CHECK_NOT_OK(index.Add(1, wrong_size));

    const std::vector<float> with_nan{1.0f, std::nanf(""), 0.0f, 0.0f};
    CHECK_NOT_OK(index.Add(2, with_nan));
    const std::vector<float> with_inf{1.0f, HUGE_VALF, 0.0f, 0.0f};
    CHECK_NOT_OK(index.Add(3, with_inf));

    // A rejected Add must leave nothing behind, or the vector block and the node
    // array drift apart and every later lookup reads the wrong row.
    CHECK_EQ(index.size(), size_t(0));

    const std::vector<float> ok{1.0f, 0.0f, 0.0f, 0.0f};
    CHECK_OK(index.Add(4, ok));
    CHECK_EQ(index.size(), size_t(1));

    auto bad_query = index.Search(wrong_size, 5, 32);
    CHECK(!bad_query.ok());
    CHECK(bad_query.status().code() == Code::kInvalidArgument);
}

MERIDIAN_TEST(GraphInvariantsHoldAfterManyInsertions) {
    // Run over several seeds. Degree bounds and symmetry are structural and would
    // fail on any seed; reachability is the one that could in principle depend on an
    // unlucky sequence of prunes, so it is worth asking more than once.
    for (uint64_t seed : {0x11u, 0x22u, 0x33u}) {
        const uint32_t n = 3000;
        const uint32_t dim = 24;
        const Dataset ds = MakeClustered(n, dim, 25, 0.3, 1, seed);

        HnswParams params;
        params.M = 8;
        params.M0 = 16;
        params.ef_construction = 100;
        HnswIndex index(dim, params);
        CHECK_OK(BuildIndex(index, ds));
        CHECK_EQ(index.size(), size_t(n));

        const uint32_t max_layer = index.max_layer();
        CHECK(index.entry_point() != HnswIndex::kNoEntryPoint);
        CHECK_EQ(index.top_layer_of(index.entry_point()), max_layer);

        uint64_t edges = 0;
        for (uint32_t layer = 0; layer <= max_layer; ++layer) {
            for (uint32_t u = 0; u < n; ++u) {
                const auto neighbors = index.Neighbors(u, layer);

                // Nothing above its own top layer.
                if (layer > index.top_layer_of(u)) {
                    CHECK(neighbors.empty());
                    continue;
                }

                // Degree bound. Exceeding it is not merely untidy: the adjacency is
                // stored at a fixed stride per layer, so an over-full list at layer 1
                // would be read as layer 2's edges.
                CHECK(neighbors.size() <= index.MaxDegree(layer));
                edges += neighbors.size();

                for (uint32_t v : neighbors) {
                    CHECK(v < n);
                    CHECK(v != u);  // no self-loops: a wasted slot and an endless hop
                    CHECK(layer <= index.top_layer_of(v));

                    // Symmetry.
                    bool back = false;
                    for (uint32_t w : index.Neighbors(v, layer)) {
                        if (w == u) back = true;
                    }
                    if (!back) {
                        std::printf("  asymmetric edge %u->%u at layer %u (seed %llu)\n", u, v,
                                    layer, static_cast<unsigned long long>(seed));
                    }
                    CHECK(back);
                }

                // No duplicate edges.
                std::vector<uint32_t> sorted(neighbors.begin(), neighbors.end());
                std::sort(sorted.begin(), sorted.end());
                CHECK(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end());
            }

            // Reachability from the entry point, within this layer.
            //
            // A node that cannot be reached is a document no query can ever return,
            // whatever ef_search is set to. It costs nothing at build time and shows
            // up only as a recall ceiling, which is the most expensive kind of bug to
            // find later.
            std::vector<char> seen(n, 0);
            std::vector<uint32_t> stack{index.entry_point()};
            seen[index.entry_point()] = 1;
            uint32_t reached = 0;
            while (!stack.empty()) {
                const uint32_t u = stack.back();
                stack.pop_back();
                ++reached;
                for (uint32_t v : index.Neighbors(u, layer)) {
                    if (seen[v] != 0) continue;
                    seen[v] = 1;
                    stack.push_back(v);
                }
            }

            uint32_t expected = 0;
            for (uint32_t u = 0; u < n; ++u) {
                if (index.top_layer_of(u) >= layer) ++expected;
            }
            if (reached != expected) {
                std::printf("  layer %u: reached %u of %u nodes (seed %llu)\n", layer, reached,
                            expected, static_cast<unsigned long long>(seed));
            }
            CHECK_EQ(reached, expected);
        }

        std::printf("  seed %#llx: %u nodes, %u layers, %llu directed edges, %.1f avg degree "
                    "at layer 0\n",
                    static_cast<unsigned long long>(seed), n, max_layer + 1,
                    static_cast<unsigned long long>(edges),
                    double(edges) / double(n));
    }
}

MERIDIAN_TEST(LevelDistributionDecaysExponentially) {
    // The 1/ln(M) normalisation is what makes each layer about M times smaller than
    // the one below. Getting it wrong — using ln(M) instead of its reciprocal, say —
    // produces an index that still works and is quietly much slower, because the
    // upper layers are either empty or nearly as large as layer 0.
    const uint32_t n = 20000;
    const uint32_t dim = 4;
    HnswParams params;
    params.M = 16;
    params.ef_construction = 16;  // structure only; this test does not measure recall
    HnswIndex index(dim, params);

    Gaussian g(0xFEED);
    for (uint32_t i = 0; i < n; ++i) {
        std::vector<float> v(dim);
        for (float& x : v) x = static_cast<float>(g.Next());
        CHECK_OK(index.Add(i, v));
    }

    std::vector<uint32_t> at_or_above(index.max_layer() + 1, 0);
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t l = 0; l <= index.top_layer_of(i); ++l) ++at_or_above[l];
    }

    std::printf("\n  layer occupancy (M=%u, so each layer should be ~%ux smaller):\n", params.M,
                params.M);
    for (uint32_t l = 0; l < at_or_above.size(); ++l) {
        std::printf("    layer %u: %6u nodes", l, at_or_above[l]);
        if (l > 0 && at_or_above[l] > 0) {
            std::printf("   ratio to layer below: %.1f",
                        double(at_or_above[l - 1]) / double(at_or_above[l]));
        }
        std::printf("\n");
    }

    CHECK_EQ(at_or_above[0], n);
    CHECK(index.max_layer() >= 2);
    // Within a factor of two of M either way. The draw is random, so a tight bound
    // would be flaky; a factor of two still catches a wrong normalisation, which is
    // off by orders of magnitude.
    const double ratio = double(at_or_above[0]) / double(at_or_above[1]);
    CHECK(ratio > params.M / 2.0);
    CHECK(ratio < params.M * 2.0);
}

MERIDIAN_TEST(ReopenReproducesIdenticalResults) {
    const Dataset ds = MakeClustered(1500, 32, 20, 0.3, 12, 0xBEEF);
    HnswParams params;
    params.M = 12;
    params.M0 = 24;
    params.ef_construction = 120;
    params.seed = 0x5151515151515151ull;

    HnswIndex index(32, params);
    CHECK_OK(BuildIndex(index, ds));

    const std::string path = TempPath("reopen.hnsw");
    CHECK_OK(meridian::ann::SaveHnsw(index, path));

    auto reopened = meridian::ann::LoadHnsw(path);
    if (!reopened.ok()) {
        std::printf("  load failed: %s\n", reopened.status().ToString().c_str());
    }
    CHECK(reopened.ok());
    if (!reopened.ok()) return;
    const HnswIndex& loaded = *reopened.value();

    CHECK_EQ(loaded.size(), index.size());
    CHECK_EQ(loaded.dim(), index.dim());
    CHECK_EQ(loaded.max_layer(), index.max_layer());
    CHECK_EQ(loaded.entry_point(), index.entry_point());
    CHECK_EQ(loaded.params().M, params.M);
    CHECK_EQ(loaded.params().ef_construction, params.ef_construction);

    // The graph, edge for edge. Identical search results could also come from two
    // different graphs that happen to agree on twelve queries, so the structure is
    // compared directly as well.
    for (uint32_t u = 0; u < index.size(); ++u) {
        CHECK_EQ(loaded.doc_of(u), index.doc_of(u));
        CHECK_EQ(loaded.top_layer_of(u), index.top_layer_of(u));
        for (uint32_t l = 0; l <= index.top_layer_of(u); ++l) {
            const auto a = index.Neighbors(u, l);
            const auto b = loaded.Neighbors(u, l);
            CHECK_EQ(a.size(), b.size());
            if (a.size() != b.size()) continue;
            for (size_t i = 0; i < a.size(); ++i) CHECK_EQ(a[i], b[i]);
        }
    }

    for (const std::vector<float>& q : ds.queries) {
        for (uint32_t ef : {10u, 64u, 200u}) {
            auto before = index.Search(q, 10, ef);
            auto after = loaded.Search(q, 10, ef);
            CHECK(before.ok() && after.ok());
            if (!before.ok() || !after.ok()) continue;
            CHECK(DocsOf(before.value()) == DocsOf(after.value()));
            CHECK_EQ(before.value().size(), after.value().size());
            for (size_t i = 0; i < before.value().size(); ++i) {
                // Scores must match bit for bit, not approximately: the norms are
                // recomputed on load, and a different summation order there would
                // mean the reopened index ranks ties differently.
                CHECK_EQ(before.value()[i].score, after.value()[i].score);
            }
        }
    }

    // Insertions after a reopen must take the same level draws, which is why the RNG
    // state is serialised rather than just the seed.
    HnswIndex& mutable_loaded = *reopened.value();
    HnswIndex fresh_copy(32, params);
    CHECK_OK(BuildIndex(fresh_copy, ds));
    const std::vector<float> extra = ds.queries[0];
    CHECK_OK(mutable_loaded.Add(777, extra));
    CHECK_OK(fresh_copy.Add(777, extra));
    CHECK_EQ(mutable_loaded.top_layer_of(1500), fresh_copy.top_layer_of(1500));

    std::remove(path.c_str());
}

MERIDIAN_TEST(EmptyIndexRoundTrips) {
    HnswIndex index(16, HnswParams{});
    std::vector<uint8_t> bytes;
    CHECK_OK(meridian::ann::EncodeHnsw(index, bytes));
    auto loaded = meridian::ann::DecodeHnsw(bytes);
    CHECK(loaded.ok());
    if (!loaded.ok()) return;
    CHECK_EQ(loaded.value()->size(), size_t(0));
    CHECK_EQ(loaded.value()->entry_point(), HnswIndex::kNoEntryPoint);
}

MERIDIAN_TEST(CorruptGraphFileIsDetected) {
    const Dataset ds = MakeClustered(400, 16, 8, 0.3, 1, 0xDEAD);
    HnswIndex index(16, HnswParams{});
    CHECK_OK(BuildIndex(index, ds));

    std::vector<uint8_t> original;
    CHECK_OK(meridian::ann::EncodeHnsw(index, original));
    CHECK(original.size() > 64);

    // Every single-byte flip in the payload must be caught. This is the property
    // worth having: a corrupt adjacency array still decodes into a valid graph that
    // returns k plausible neighbours, so nothing downstream would ever notice.
    const size_t payload = meridian::ann::HnswChecksumOffset(original);
    const size_t step = payload / 64 + 1;
    uint32_t checked = 0;
    for (size_t i = 0; i < payload; i += step) {
        std::vector<uint8_t> bytes = original;
        bytes[i] ^= 0x40;
        auto loaded = meridian::ann::DecodeHnsw(bytes);
        if (loaded.ok()) {
            std::printf("  byte %zu flipped and the file still loaded\n", i);
        }
        CHECK(!loaded.ok());
        if (!loaded.ok()) CHECK(loaded.status().code() == Code::kCorruption);
        ++checked;
    }
    std::printf("  %u single-byte corruptions, all rejected\n", checked);

    // A flipped checksum is a mismatch too — the detector must not trust itself.
    {
        std::vector<uint8_t> bytes = original;
        bytes[payload + 3] ^= 0x01;
        auto loaded = meridian::ann::DecodeHnsw(bytes);
        CHECK(!loaded.ok());
    }

    // Truncation. A half-written file is the realistic failure, not a cosmic ray, and
    // it must not read past the end of the buffer — which is what the sanitiser build
    // of this test is checking.
    for (size_t keep : {size_t(0), size_t(1), size_t(8), size_t(16), size_t(40),
                        original.size() / 2, original.size() - 1}) {
        std::vector<uint8_t> bytes(original.begin(), original.begin() + keep);
        auto loaded = meridian::ann::DecodeHnsw(bytes);
        CHECK(!loaded.ok());
    }

    // Wrong magic, right length: a file from another system entirely.
    {
        std::vector<uint8_t> bytes = original;
        bytes[0] = 'X';
        auto loaded = meridian::ann::DecodeHnsw(bytes);
        CHECK(!loaded.ok());
        CHECK(loaded.status().code() == Code::kCorruption);
    }

    // Trailing garbage, with the checksum recomputed over it. The checksum cannot
    // help here, so the structural checks have to.
    {
        std::vector<uint8_t> bytes(original.begin(), original.begin() + payload);
        bytes.push_back(0x7F);
        uint64_t hash = 0xCBF29CE484222325ull;
        for (uint8_t b : bytes) {
            hash ^= b;
            hash *= 0x100000001B3ull;
        }
        for (int i = 0; i < 8; ++i) bytes.push_back(static_cast<uint8_t>(hash >> (8 * i)));
        auto loaded = meridian::ann::DecodeHnsw(bytes);
        CHECK(!loaded.ok());
    }
}

MERIDIAN_TEST(MissingFileIsAnIoErrorNotACorruption) {
    auto loaded = meridian::ann::LoadHnsw(TempPath("definitely_not_here.hnsw"));
    CHECK(!loaded.ok());
    CHECK(loaded.status().code() == Code::kIoError);
}

MERIDIAN_TEST(RecallCurveUnderInnerProductAndL2) {
    // Cosine is the default and gets the headline sweep. The other two metrics get a
    // smaller one, because inner product is not a metric at all — it violates the
    // triangle inequality, so the graph's navigability is an empirical claim rather
    // than a geometric one, and the only way to know it holds is to measure it.
    const uint32_t n = 10000;
    const uint32_t dim = 64;
    const Dataset ds = MakeClustered(n, dim, 30, 0.5, 40, 0x9999);

    struct Case {
        const char* name;
        Metric metric;
    };
    for (const Case& c : {Case{"inner product", Metric::kInnerProduct},
                          Case{"squared L2", Metric::kL2Squared}}) {
        HnswParams params;
        params.metric = c.metric;
        params.M = 8;
        params.M0 = 16;
        params.ef_construction = 200;
        HnswIndex index(dim, params);
        CHECK_OK(BuildIndex(index, ds));
        const GroundTruth truth = ComputeGroundTruth(index, ds, 10);

        std::printf("\n  recall@10 under %s (n=%u, dim=%u)\n", c.name, n, dim);
        double best = 0.0;
        for (uint32_t ef : {10u, 32u, 128u}) {
            const RecallPoint p = MeasureRecall(index, ds, truth, 10, ef);
            std::printf("    ef=%-4u recall=%.4f  dist/query=%.1f\n", ef, p.recall,
                        p.distance_computations);
            best = std::max(best, p.recall);
        }
        // Lower bar than cosine for inner product, deliberately: this is the metric
        // where the graph has no geometric guarantee, and the number is here to be
        // read rather than to be a pass.
        CHECK(best >= 0.85);
    }
}

MERIDIAN_TEST(StatsAreCountedHonestly) {
    const Dataset ds = MakeClustered(2000, 32, 15, 0.3, 1, 0x4242);
    HnswIndex index(32, HnswParams{});
    CHECK_OK(BuildIndex(index, ds));

    // Brute force over n vectors costs exactly n distance computations. If
    // SearchExact reported fewer, the recall denominator would be wrong and every
    // "Nx cheaper than brute force" claim in this file would be inflated.
    index.ResetStats();
    auto exact = index.SearchExact(ds.queries[0], 10);
    CHECK(exact.ok());
    CHECK_EQ(index.stats().distance_computations, uint64_t(2000));

    index.ResetStats();
    auto approx = index.Search(ds.queries[0], 10, 64);
    CHECK(approx.ok());
    const auto s = index.stats();

    // A node is only visited after its distance is computed, so one bounds the other.
    CHECK(s.nodes_visited <= s.distance_computations);
    // Every hop reads an adjacency list, and the graph has edges, so a search that
    // found anything took at least one hop.
    CHECK(s.hops >= 1);
    CHECK(s.distance_computations > 0);
    // The whole point: far less than brute force.
    CHECK(s.distance_computations < 2000);

    // Counters accumulate rather than reset per call, matching LexicalRetriever.
    const uint64_t after_one = index.stats().distance_computations;
    auto again = index.Search(ds.queries[0], 10, 64);
    CHECK(again.ok());
    CHECK(index.stats().distance_computations > after_one);

    std::printf("  one ef=64 query over 2000 vectors: %llu distances, %llu visits, %llu hops\n",
                static_cast<unsigned long long>(s.distance_computations),
                static_cast<unsigned long long>(s.nodes_visited),
                static_cast<unsigned long long>(s.hops));
}

MERIDIAN_TEST_MAIN()
