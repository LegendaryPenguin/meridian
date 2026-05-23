#pragma once

// HNSW: Hierarchical Navigable Small World graphs.
//
// The idea in one paragraph. Build a proximity graph over the vectors and search it
// greedily: step to whichever neighbour is closer to the query, repeat. A flat
// proximity graph gets stuck in local minima, so stack several, each a random
// sample of the one below with roughly 1/M the nodes. The top layer is tiny and its
// edges span the whole space, so a greedy walk there covers huge distances in a few
// hops; each descent hands a progressively better starting point to a
// progressively denser layer. The bottom layer holds every vector and is where the
// actual beam search happens. Total cost is logarithmic in the corpus rather than
// linear, which is the entire reason not to brute-force.
//
// What makes it work is not the layering, though — it is neighbour selection. See
// SelectNeighbors below. Picking a node's M closest neighbours is the obvious thing
// and it builds a graph where every edge is short, every cluster is a clique, and
// there is no way to get from one cluster to another. The heuristic deliberately
// keeps some longer edges, and that difference is measured in hnsw_test.cpp rather
// than asserted here.
//
// Not thread-safe. Search mutates a visited-stamp array and the statistics
// counters, so concurrent Search calls on one index race. Stated rather than
// papered over with a mutex, because the serving layer wants one index per shard
// and a lock here would be a false promise of scalability.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "meridian/postings.hpp"
#include "meridian/retrieval.hpp"
#include "meridian/status.hpp"

#include "distance.hpp"

namespace meridian::ann {

/// How a node's neighbours are chosen from the candidates found during insertion.
enum class NeighborSelection : uint8_t {
    /// The paper's Algorithm 4. Prunes a candidate that sits closer to an
    /// already-chosen neighbour than to the query point, which keeps the retained
    /// edges pointing in diverse directions and preserves long-range connectivity.
    kHeuristic,
    /// Plain nearest-M. Present only so the recall cost of getting this wrong can
    /// be measured instead of described; not a supported production setting.
    kTopM,
};

struct HnswParams {
    /// Edges per node per layer above 0.
    uint32_t M = 16;

    /// Edges per node at layer 0. Twice M by convention: layer 0 holds every vector
    /// and is where the beam search spends its time, so it is the layer that can
    /// actually use the extra degree.
    uint32_t M0 = 32;

    /// Beam width during insertion. Larger means a better-informed neighbour
    /// choice, a better graph, and a slower build — it costs nothing at query time,
    /// which is why it is usually set high and forgotten.
    uint32_t ef_construction = 200;

    /// Seeds the level-assignment RNG. Fixed by default: an index whose structure
    /// depends on unseeded randomness cannot be compared against itself across
    /// runs, which makes every recall measurement anecdotal.
    uint64_t seed = 0x9E3779B97F4A7C15ull;

    Metric metric = Metric::kCosine;

    NeighborSelection selection = NeighborSelection::kHeuristic;

    /// When the heuristic rejects enough candidates that a node would end up with
    /// fewer than its full quota of edges, backfill from the rejects. Costs some
    /// diversity, buys degree, and matters most for small indexes where a thin
    /// graph is a disconnected graph.
    bool keep_pruned_connections = true;
};

/// A deterministic RNG, hand-rolled rather than taken from <random>.
///
/// std::mt19937 would do, but its distribution objects are not specified to
/// produce the same sequence across standard libraries, so "same seed, same index"
/// would hold only on one platform. SplitMix64 is a few lines, is specified
/// entirely by those lines, and has a state small enough to serialise.
class SplitMix64 {
public:
    explicit SplitMix64(uint64_t seed) : state_(seed) {}

    uint64_t Next() {
        state_ += 0x9E3779B97F4A7C15ull;
        uint64_t z = state_;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

    /// Uniform in (0, 1]. The open lower bound is load-bearing: the level draw
    /// takes a logarithm of this value, and log(0) is -inf.
    double NextUnitHalfOpen() {
        return static_cast<double>((Next() >> 11) + 1) * (1.0 / 9007199254740992.0);
    }

    uint64_t state() const { return state_; }
    void set_state(uint64_t s) { state_ = s; }

private:
    uint64_t state_;
};

class HnswIndex final : public AnnIndex {
public:
    HnswIndex(uint32_t dim, HnswParams params);

    /// Inserts immediately — this is an incremental structure, and deferring the
    /// work to Build() would only mean holding every vector twice.
    ///
    /// Rejects a vector of the wrong length, and rejects a non-finite component: a
    /// NaN propagates into every distance it touches, compares false against all
    /// comparisons, and silently turns the graph into a structure that returns
    /// arbitrary neighbours.
    Status Add(DocId doc, std::span<const float> vector) override;

    /// Nothing left to do by the time this is called; it exists to satisfy the
    /// interface and to be the place where a future quantisation pass would go.
    Status Build() override;

    /// `ef_search` is the beam width at layer 0 and the recall/latency dial.
    ///
    /// ef_search < k is clamped up to k rather than refused. A beam of width ef
    /// cannot hold more than ef results, so an unclamped ef=5, k=10 would return
    /// five hits and look like a short index rather than a misconfiguration.
    /// Clamping makes k the floor on effort, which is the only behaviour that
    /// keeps the contract "returns min(k, size()) results". ef_search == 0 means
    /// the same thing: use k.
    Result<std::vector<Candidate>> Search(std::span<const float> query, uint32_t k,
                                          uint32_t ef_search) const override;

    Result<std::vector<Candidate>> SearchExact(std::span<const float> query,
                                               uint32_t k) const override;

    size_t size() const override { return nodes_.size(); }

    /// What the three counters mean, exactly, because the recall-versus-effort curve
    /// is plotted against them and a counter whose definition is vague is worse than
    /// no counter:
    ///
    ///   distance_computations  every call into the distance kernel, including the
    ///                         candidate-to-candidate comparisons the selection
    ///                         heuristic makes during insertion.
    ///   nodes_visited         every node whose distance was evaluated. The beam
    ///                         search evaluates each node at most once per layer
    ///                         thanks to the visit stamps; the greedy descent through
    ///                         the upper layers can re-evaluate a node it has already
    ///                         stepped past, and those are counted again rather than
    ///                         deduplicated, because the cost was really paid.
    ///   hops                  every node whose adjacency list was read, which is the
    ///                         graph-traversal step count.
    ///
    /// During Search, nodes_visited and distance_computations are equal by
    /// construction: nothing is visited without being measured and nothing is measured
    /// without being visited. They diverge only in the cumulative total, where
    /// insertion's heuristic adds distances that visit no node. Both are kept because
    /// the interface asks for both and because a future quantised prefilter would break
    /// the equality.
    Stats stats() const override { return stats_; }

    /// Counters accumulate across calls, matching LexicalRetriever. Per-query
    /// numbers come from resetting first, which is what the recall harness does.
    void ResetStats() const { stats_ = Stats{}; }

    uint32_t dim() const { return dim_; }
    const HnswParams& params() const { return params_; }

    // --- graph inspection, for the invariant tests ------------------------------
    //
    // A graph that silently violates its degree bound or orphans a node still
    // returns plausible-looking results, so the invariants have to be checkable
    // from outside.

    /// kNoEntryPoint when the index is empty.
    static constexpr uint32_t kNoEntryPoint = UINT32_MAX;
    uint32_t entry_point() const { return entry_point_; }
    uint32_t max_layer() const { return max_layer_; }
    uint32_t top_layer_of(uint32_t node) const { return nodes_[node].top_layer; }
    DocId doc_of(uint32_t node) const { return nodes_[node].doc; }
    std::span<const uint32_t> Neighbors(uint32_t node, uint32_t layer) const;

    /// Edges per node permitted at `layer`.
    uint32_t MaxDegree(uint32_t layer) const { return layer == 0 ? params_.M0 : params_.M; }

    // --- persistence hand-off ---------------------------------------------------
    //
    // Serialisation gets a flat value type rather than friendship, so persist.cpp
    // sees exactly the state that defines the index and nothing else. If a field is
    // added here and not to Snapshot, reopen stops reproducing results and the
    // reopen test says so.
    struct Snapshot {
        uint32_t dim = 0;
        HnswParams params;
        uint64_t rng_state = 0;
        uint32_t entry_point = kNoEntryPoint;
        uint32_t max_layer = 0;
        std::vector<DocId> docs;
        std::vector<uint32_t> top_layers;
        std::vector<float> vectors;  // docs.size() * dim, row-major
        /// neighbours[node][layer]; the inner vectors are sized by top_layer + 1.
        std::vector<std::vector<std::vector<uint32_t>>> neighbors;
    };

    Snapshot TakeSnapshot() const;

    /// Rebuilds an index from a snapshot, validating the structure. Fails rather
    /// than constructing a graph with out-of-range edges, because a bad edge index
    /// is a wrong answer in release and a crash in a sanitiser build.
    static Result<std::unique_ptr<HnswIndex>> FromSnapshot(Snapshot snapshot);

private:
    struct Node {
        DocId doc = 0;
        uint32_t top_layer = 0;
        float norm = 0.0f;  // precomputed so cosine is one dot product per compare
        /// Flattened adjacency: layer L occupies [layer_offset(L), +degree). Sized
        /// at insertion to (top_layer + 1) * MaxDegree(layer), so an edge list
        /// never reallocates and neighbour pointers stay valid for the node's life.
        std::vector<uint32_t> links;
        std::vector<uint32_t> degrees;
    };

    /// (distance, node), the unit of currency in every heap below.
    struct DistNode {
        float dist;
        uint32_t node;
    };

    /// Index at which `layer`'s edge list starts within a node's flat links array.
    size_t LinkOffset(uint32_t layer) const;

    const float* VectorOf(uint32_t node) const { return vectors_.data() + size_t(node) * dim_; }

    float DistanceTo(const float* query, float query_norm, uint32_t node) const;

    uint32_t DrawLevel();

    /// Walks downhill from `entry` until no neighbour improves. ef=1 search, used
    /// for the layers above the one being inserted into or searched at.
    DistNode GreedyDescend(const float* query, float query_norm, DistNode entry,
                           uint32_t layer) const;

    /// The beam search. Returns up to `ef` nodes nearest the query, unordered.
    std::vector<DistNode> SearchLayer(const float* query, float query_norm,
                                      std::span<const DistNode> entries, uint32_t ef,
                                      uint32_t layer) const;

    bool HasLink(uint32_t from, uint32_t to, uint32_t layer) const;
    /// Appends without checking for room; every caller has just established it.
    void AppendLink(uint32_t from, uint32_t to, uint32_t layer);
    void RemoveLink(uint32_t from, uint32_t to, uint32_t layer);

    /// Appends to `out` until it holds `m` entries. Does not clear it, so a caller
    /// can seed a neighbour that must survive selection and still have the heuristic
    /// prune the rest relative to it.
    ///
    /// The heuristic: walk the candidates nearest-first and keep a candidate only
    /// if it is closer to the query than to every neighbour already kept. A
    /// candidate that fails this test lies "behind" one that was already accepted —
    /// the edge to it is redundant, because the accepted neighbour is a better
    /// stepping stone towards it. Dropping it frees a slot for an edge in a
    /// direction the node cannot otherwise reach, and those long edges are the
    /// whole navigability story. Without them, nearest-M builds a graph of tight
    /// mutually-connected clumps whose recall stops improving however wide the beam
    /// gets, because the beam cannot leave the clump it started in.
    /// `candidates` carries each candidate's distance to the query point already, so
    /// the query vector itself is not needed here.
    void SelectNeighbors(std::vector<DistNode>& candidates, uint32_t m,
                         std::vector<uint32_t>& out) const;

    void SelectTopM(std::vector<DistNode>& candidates, uint32_t m,
                    std::vector<uint32_t>& out) const;

    /// Links node and neighbour in both directions, pruning the neighbour's list
    /// back to its degree bound with the same selection rule when it overflows.
    ///
    /// The graph is kept strictly symmetric: when a prune drops the edge b->e, the
    /// edge e->b goes too. The paper leaves the reverse edge in place, which is
    /// cheaper and gives a directed graph where "u is a neighbour of v" is not the
    /// same question as "v is a neighbour of u". Symmetry is kept here because it
    /// makes the structure's two real invariants — degree bounded, every node
    /// reachable — checkable from outside, and an unreachable node in an ANN index
    /// is a document that can never be returned for any query, which is a silent
    /// recall bug of exactly the kind this project exists not to have.
    void ConnectBidirectional(uint32_t node, uint32_t neighbor, uint32_t layer);

    uint32_t dim_;
    HnswParams params_;
    double level_mult_;  // 1 / ln(M)
    SplitMix64 rng_;

    std::vector<Node> nodes_;
    std::vector<float> vectors_;

    uint32_t entry_point_ = kNoEntryPoint;
    uint32_t max_layer_ = 0;

    /// Visited marking by generation stamp rather than a cleared bitset or a hash
    /// set: a search touches a few hundred nodes out of millions, so clearing an
    /// n-bit array per query would dominate, and a hash set costs a hash and a
    /// probe per edge examined. Mutable because Search is const by interface and
    /// this is scratch space, not state.
    mutable std::vector<uint32_t> visit_stamp_;
    mutable uint32_t visit_epoch_ = 0;

    mutable Stats stats_;
};

}  // namespace meridian::ann
