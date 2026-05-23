#include "hnsw.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace meridian::ann {
namespace {

// A binary heap, written out rather than taken from <algorithm>.
//
// std::priority_queue would serve, but the beam search needs to read the worst
// element of the result set on every edge it examines and to pop it immediately
// after — and it needs the *container*, because the result set is the return value.
// std::priority_queue hides its container, so the alternative is copying it out.
// Twenty lines of sift is cheaper than that, and makes the one subtle part visible:
// the tie-break.
//
// `worse(a, b)` means "a has lower priority than b", so the root is the highest
// priority element: nearest for the candidate heap, farthest for the result heap.

template <typename T, typename Worse>
void HeapPush(std::vector<T>& heap, const T& value, Worse worse) {
    heap.push_back(value);
    size_t i = heap.size() - 1;
    while (i > 0) {
        const size_t parent = (i - 1) / 2;
        if (!worse(heap[parent], heap[i])) break;
        std::swap(heap[parent], heap[i]);
        i = parent;
    }
}

template <typename T, typename Worse>
T HeapPop(std::vector<T>& heap, Worse worse) {
    T top = heap.front();
    heap.front() = heap.back();
    heap.pop_back();
    const size_t n = heap.size();
    size_t i = 0;
    while (true) {
        const size_t left = 2 * i + 1;
        const size_t right = left + 1;
        size_t best = i;
        if (left < n && worse(heap[best], heap[left])) best = left;
        if (right < n && worse(heap[best], heap[right])) best = right;
        if (best == i) break;
        std::swap(heap[i], heap[best]);
        i = best;
    }
    return top;
}

// Ties on distance are broken by node index, which is what makes a result list
// reproducible. Duplicate vectors are ordinary in a real corpus — the same
// paragraph appears in two documents — and with a partial order the heap's output
// would depend on the order elements happened to be pushed, so a reopened index
// could return the same set in a different order and the reopen test would be
// measuring nothing.
constexpr auto kNearestFirst = [](const auto& a, const auto& b) {
    return a.dist != b.dist ? a.dist > b.dist : a.node > b.node;
};
constexpr auto kFarthestFirst = [](const auto& a, const auto& b) {
    return a.dist != b.dist ? a.dist < b.dist : a.node < b.node;
};
constexpr auto kAscending = [](const auto& a, const auto& b) {
    return a.dist != b.dist ? a.dist < b.dist : a.node < b.node;
};

}  // namespace

HnswIndex::HnswIndex(uint32_t dim, HnswParams params)
    : dim_(dim), params_(params), rng_(params.seed) {
    // M == 1 would make the level normalisation 1/ln(1), which is a division by
    // zero. M == 0 is a graph with no edges. Clamping rather than failing keeps the
    // constructor non-fallible, and the clamp is visible in params().
    if (params_.M < 2) params_.M = 2;
    if (params_.M0 < params_.M) params_.M0 = params_.M;
    if (params_.ef_construction < 1) params_.ef_construction = 1;

    // 1/ln(M) is the normalisation that makes the expected number of nodes at
    // layer L+1 a factor of M smaller than at layer L. That is the whole point of
    // the layering: it makes the top layer's edge length comparable to the diameter
    // of the data, so a greedy walk there crosses the space in a few hops.
    level_mult_ = 1.0 / std::log(static_cast<double>(params_.M));
}

size_t HnswIndex::LinkOffset(uint32_t layer) const {
    return layer == 0 ? 0 : size_t(params_.M0) + size_t(layer - 1) * params_.M;
}

std::span<const uint32_t> HnswIndex::Neighbors(uint32_t node, uint32_t layer) const {
    const Node& n = nodes_[node];
    if (layer > n.top_layer) return {};
    return std::span<const uint32_t>(n.links.data() + LinkOffset(layer), n.degrees[layer]);
}

float HnswIndex::DistanceTo(const float* query, float query_norm, uint32_t node) const {
    ++stats_.distance_computations;
    return Distance(params_.metric, query, query_norm, VectorOf(node), nodes_[node].norm, dim_);
}

uint32_t HnswIndex::DrawLevel() {
    // floor(-ln(U) / ln(M)): the exponentially decaying draw from the paper. Most
    // nodes land on layer 0; one in M reaches layer 1, one in M^2 layer 2.
    const double u = rng_.NextUnitHalfOpen();
    return static_cast<uint32_t>(-std::log(u) * level_mult_);
}

Status HnswIndex::Add(DocId doc, std::span<const float> vector) {
    if (vector.size() != dim_) {
        return Status::InvalidArgument("vector has " + std::to_string(vector.size()) +
                                      " dimensions, index has " + std::to_string(dim_));
    }
    for (float v : vector) {
        // A NaN compares false against every bound in the beam search, so it is
        // never rejected and never accepted — it propagates into the graph and
        // turns neighbour lists into noise, with no symptom at the call site.
        if (!std::isfinite(v)) {
            return Status::InvalidArgument("vector contains a non-finite component");
        }
    }

    const uint32_t id = static_cast<uint32_t>(nodes_.size());
    vectors_.insert(vectors_.end(), vector.begin(), vector.end());

    Node node;
    node.doc = doc;
    node.top_layer = DrawLevel();
    node.norm = Norm(vectors_.data() + size_t(id) * dim_, dim_);
    node.degrees.assign(node.top_layer + 1, 0);
    size_t link_capacity = 0;
    for (uint32_t l = 0; l <= node.top_layer; ++l) link_capacity += MaxDegree(l);
    node.links.assign(link_capacity, 0);
    const uint32_t top = node.top_layer;
    nodes_.push_back(std::move(node));
    visit_stamp_.push_back(0);

    if (entry_point_ == kNoEntryPoint) {
        entry_point_ = id;
        max_layer_ = top;
        return Status::Ok();
    }

    const float* q = VectorOf(id);
    const float q_norm = nodes_[id].norm;

    DistNode ep{DistanceTo(q, q_norm, entry_point_), entry_point_};

    // Phase one: fall through the layers above this node's top layer with a width-1
    // greedy walk. No candidate list is kept, because nothing is being linked here
    // — the only product is a good entry point for the layer below.
    for (uint32_t l = max_layer_; l > top; --l) ep = GreedyDescend(q, q_norm, ep, l);

    // Phase two: from this node's top layer down to 0, a real beam search at each
    // layer, then neighbour selection and linking.
    const uint32_t start = std::min(max_layer_, top);
    for (uint32_t l = start + 1; l-- > 0;) {
        std::vector<DistNode> found = SearchLayer(q, q_norm, std::span(&ep, 1),
                                                 params_.ef_construction, l);

        std::vector<DistNode> candidates = found;
        std::vector<uint32_t> chosen;
        SelectNeighbors(candidates, MaxDegree(l), chosen);
        for (uint32_t nb : chosen) ConnectBidirectional(id, nb, l);

        // The nearest node found here is the entry point for the layer below: it is
        // strictly better than the one this layer started from, which is how the
        // descent stays cheap.
        for (const DistNode& c : found) {
            if (kAscending(c, ep)) ep = c;
        }
    }

    if (top > max_layer_) {
        max_layer_ = top;
        entry_point_ = id;
    }
    return Status::Ok();
}

Status HnswIndex::Build() { return Status::Ok(); }

HnswIndex::DistNode HnswIndex::GreedyDescend(const float* query, float query_norm, DistNode entry,
                                             uint32_t layer) const {
    DistNode best = entry;
    bool improved = true;
    while (improved) {
        improved = false;
        ++stats_.hops;
        // The whole of the current node's adjacency is scanned before moving, not
        // just until the first improvement: taking the best of the list is what
        // makes the walk descend steeply instead of wandering.
        const std::span<const uint32_t> neighbors = Neighbors(best.node, layer);
        DistNode next = best;
        for (uint32_t e : neighbors) {
            const float d = DistanceTo(query, query_norm, e);
            ++stats_.nodes_visited;
            if (d < next.dist) {
                next = DistNode{d, e};
                improved = true;
            }
        }
        best = next;
    }
    return best;
}

std::vector<HnswIndex::DistNode> HnswIndex::SearchLayer(const float* query, float query_norm,
                                                        std::span<const DistNode> entries,
                                                        uint32_t ef, uint32_t layer) const {
    // Stamp-based visited marking wraps at 2^32 searches. Clearing on wrap costs
    // one O(n) pass every four billion queries; not clearing would let a stale
    // stamp from the first search mark a node visited, silently shrinking recall in
    // a way no test would ever catch.
    if (visit_epoch_ == UINT32_MAX) {
        std::fill(visit_stamp_.begin(), visit_stamp_.end(), 0u);
        visit_epoch_ = 0;
    }
    ++visit_epoch_;

    std::vector<DistNode> candidates;  // min-heap: nearest at the root
    std::vector<DistNode> result;      // max-heap: farthest at the root
    candidates.reserve(ef * 2);
    result.reserve(ef + 1);

    for (const DistNode& e : entries) {
        if (visit_stamp_[e.node] == visit_epoch_) continue;
        visit_stamp_[e.node] = visit_epoch_;
        ++stats_.nodes_visited;
        HeapPush(candidates, e, kNearestFirst);
        HeapPush(result, e, kFarthestFirst);
        if (result.size() > ef) HeapPop(result, kFarthestFirst);
    }

    while (!candidates.empty()) {
        const DistNode c = HeapPop(candidates, kNearestFirst);
        // The termination condition: the nearest unexplored candidate is farther
        // than the worst result already held, and the result set is full. Anything
        // reachable through c is, in a navigable graph, no closer than c.
        if (result.size() >= ef && c.dist > result.front().dist) break;

        ++stats_.hops;
        for (uint32_t e : Neighbors(c.node, layer)) {
            if (visit_stamp_[e] == visit_epoch_) continue;
            visit_stamp_[e] = visit_epoch_;
            ++stats_.nodes_visited;
            const float d = DistanceTo(query, query_norm, e);
            if (result.size() < ef || d < result.front().dist) {
                HeapPush(candidates, DistNode{d, e}, kNearestFirst);
                HeapPush(result, DistNode{d, e}, kFarthestFirst);
                if (result.size() > ef) HeapPop(result, kFarthestFirst);
            }
        }
    }
    return result;
}

bool HnswIndex::HasLink(uint32_t from, uint32_t to, uint32_t layer) const {
    for (uint32_t e : Neighbors(from, layer)) {
        if (e == to) return true;
    }
    return false;
}

void HnswIndex::AppendLink(uint32_t from, uint32_t to, uint32_t layer) {
    Node& n = nodes_[from];
    n.links[LinkOffset(layer) + n.degrees[layer]] = to;
    ++n.degrees[layer];
}

void HnswIndex::RemoveLink(uint32_t from, uint32_t to, uint32_t layer) {
    Node& n = nodes_[from];
    if (layer > n.top_layer) return;
    const size_t offset = LinkOffset(layer);
    uint32_t& degree = n.degrees[layer];
    for (uint32_t i = 0; i < degree; ++i) {
        if (n.links[offset + i] != to) continue;
        // Swap the last entry into the hole. Adjacency order carries no meaning —
        // the search reads the whole list — so compacting in order would be work
        // for nothing.
        n.links[offset + i] = n.links[offset + degree - 1];
        --degree;
        return;
    }
}

void HnswIndex::SelectTopM(std::vector<DistNode>& candidates, uint32_t m,
                           std::vector<uint32_t>& out) const {
    std::sort(candidates.begin(), candidates.end(), kAscending);
    for (const DistNode& c : candidates) {
        if (out.size() >= m) break;
        if (std::find(out.begin(), out.end(), c.node) == out.end()) out.push_back(c.node);
    }
}

void HnswIndex::SelectNeighbors(std::vector<DistNode>& candidates, uint32_t m,
                                std::vector<uint32_t>& out) const {
    if (params_.selection == NeighborSelection::kTopM) {
        SelectTopM(candidates, m, out);
        return;
    }

    std::sort(candidates.begin(), candidates.end(), kAscending);
    out.reserve(m);

    std::vector<DistNode> discarded;
    for (const DistNode& c : candidates) {
        if (out.size() >= m) break;
        if (std::find(out.begin(), out.end(), c.node) != out.end()) continue;
        const float* cv = VectorOf(c.node);
        const float cn = nodes_[c.node].norm;

        // Keep c only if the query point is its closest already-selected thing. If
        // some already-chosen neighbour r is nearer to c than the query is, then r
        // is on the way to c and the direct edge is redundant.
        bool keep = true;
        for (uint32_t r : out) {
            ++stats_.distance_computations;
            const float d_cr =
                Distance(params_.metric, cv, cn, VectorOf(r), nodes_[r].norm, dim_);
            if (d_cr < c.dist) {
                keep = false;
                break;
            }
        }
        if (keep) {
            out.push_back(c.node);
        } else if (params_.keep_pruned_connections) {
            discarded.push_back(c);
        }
    }

    // Backfill, nearest-first, from what the heuristic rejected. A node left with
    // three of its thirty-two slots filled is a node the beam search walks into and
    // cannot walk out of, and that is worse than a slightly redundant edge.
    for (const DistNode& d : discarded) {
        if (out.size() >= m) break;
        out.push_back(d.node);
    }
}

void HnswIndex::ConnectBidirectional(uint32_t node, uint32_t neighbor, uint32_t layer) {
    if (node == neighbor) return;
    // A node can legitimately be offered the same neighbour twice across a re-link,
    // and a duplicate edge wastes a slot and double-counts in the degree. One test
    // is enough because the graph is symmetric: if the forward edge exists, so does
    // the reverse.
    if (HasLink(node, neighbor, layer)) return;

    const uint32_t bound = MaxDegree(layer);
    // `node` is the vector being inserted and was given at most `bound` neighbours,
    // so it has room by construction.
    AppendLink(node, neighbor, layer);

    if (nodes_[neighbor].degrees[layer] < bound) {
        AppendLink(neighbor, node, layer);
        return;
    }

    // The neighbour is full. Re-run selection over its existing edges plus the new
    // one, from its own point of view — this is the only place the degree bound is
    // enforced, and running the heuristic rather than simply dropping the farthest
    // edge is what stops a saturated hub from shedding its long-range edges first,
    // which is where navigability lives.
    const float* bv = VectorOf(neighbor);
    const float b_norm = nodes_[neighbor].norm;

    std::vector<uint32_t> existing;
    {
        const std::span<const uint32_t> nb = Neighbors(neighbor, layer);
        existing.assign(nb.begin(), nb.end());
    }

    std::vector<DistNode> candidates;
    candidates.reserve(existing.size());
    for (uint32_t e : existing) candidates.push_back(DistNode{DistanceTo(bv, b_norm, e), e});

    // `node` is seeded into the keep set rather than offered as a candidate, so it
    // cannot be pruned out. A freshly inserted vector whose every chosen neighbour
    // rejected it would have no edges at all and be unreachable for the rest of the
    // index's life — permanently invisible to search, with no symptom but a recall
    // number slightly lower than it should be. One possibly-redundant edge is a much
    // smaller price.
    std::vector<uint32_t> keep;
    keep.reserve(bound);
    keep.push_back(node);
    SelectNeighbors(candidates, bound, keep);

    for (uint32_t e : existing) {
        if (std::find(keep.begin(), keep.end(), e) == keep.end()) RemoveLink(e, neighbor, layer);
    }

    Node& b = nodes_[neighbor];
    const size_t b_offset = LinkOffset(layer);
    for (size_t i = 0; i < keep.size(); ++i) b.links[b_offset + i] = keep[i];
    b.degrees[layer] = static_cast<uint32_t>(keep.size());
}

Result<std::vector<Candidate>> HnswIndex::Search(std::span<const float> query, uint32_t k,
                                                 uint32_t ef_search) const {
    if (query.size() != dim_) {
        return Status::InvalidArgument("query has " + std::to_string(query.size()) +
                                      " dimensions, index has " + std::to_string(dim_));
    }
    std::vector<Candidate> hits;
    // An empty index has no neighbours. That is an answer, not an error: the
    // serving path merges this with the lexical side, and a Status here would turn
    // "nothing indexed yet" into a failed query.
    if (nodes_.empty() || k == 0) return hits;

    const uint32_t ef = std::max(ef_search == 0 ? k : ef_search, k);

    const float* q = query.data();
    const float q_norm = params_.metric == Metric::kCosine ? Norm(q, dim_) : 0.0f;

    DistNode ep{DistanceTo(q, q_norm, entry_point_), entry_point_};
    for (uint32_t l = max_layer_; l >= 1; --l) ep = GreedyDescend(q, q_norm, ep, l);

    std::vector<DistNode> found = SearchLayer(q, q_norm, std::span(&ep, 1), ef, 0);
    std::sort(found.begin(), found.end(), kAscending);

    const size_t take = std::min<size_t>(k, found.size());
    hits.reserve(take);
    for (size_t i = 0; i < take; ++i) {
        Candidate c;
        c.doc = nodes_[found[i].node].doc;
        c.vector_similarity = ScoreFromDistance(params_.metric, found[i].dist);
        c.score = c.vector_similarity;
        c.source = Candidate::Source::kSemantic;
        hits.push_back(c);
    }
    return hits;
}

Result<std::vector<Candidate>> HnswIndex::SearchExact(std::span<const float> query,
                                                      uint32_t k) const {
    if (query.size() != dim_) {
        return Status::InvalidArgument("query has " + std::to_string(query.size()) +
                                      " dimensions, index has " + std::to_string(dim_));
    }
    std::vector<Candidate> hits;
    if (nodes_.empty() || k == 0) return hits;

    const float* q = query.data();
    const float q_norm = params_.metric == Metric::kCosine ? Norm(q, dim_) : 0.0f;

    // A bounded max-heap rather than sorting all n: ground truth is recomputed for
    // every query in the recall sweep, and at n=20000 the sort would cost more than
    // the distances.
    std::vector<DistNode> best;
    best.reserve(k + 1);
    for (uint32_t i = 0; i < nodes_.size(); ++i) {
        const DistNode d{DistanceTo(q, q_norm, i), i};
        if (best.size() < k || kAscending(d, best.front())) {
            HeapPush(best, d, kFarthestFirst);
            if (best.size() > k) HeapPop(best, kFarthestFirst);
        }
    }
    std::sort(best.begin(), best.end(), kAscending);

    hits.reserve(best.size());
    for (const DistNode& d : best) {
        Candidate c;
        c.doc = nodes_[d.node].doc;
        c.vector_similarity = ScoreFromDistance(params_.metric, d.dist);
        c.score = c.vector_similarity;
        c.source = Candidate::Source::kSemantic;
        hits.push_back(c);
    }
    return hits;
}

HnswIndex::Snapshot HnswIndex::TakeSnapshot() const {
    Snapshot s;
    s.dim = dim_;
    s.params = params_;
    s.rng_state = rng_.state();
    s.entry_point = entry_point_;
    s.max_layer = max_layer_;
    s.vectors = vectors_;
    s.docs.reserve(nodes_.size());
    s.top_layers.reserve(nodes_.size());
    s.neighbors.reserve(nodes_.size());
    for (uint32_t i = 0; i < nodes_.size(); ++i) {
        const Node& n = nodes_[i];
        s.docs.push_back(n.doc);
        s.top_layers.push_back(n.top_layer);
        std::vector<std::vector<uint32_t>> per_layer;
        per_layer.reserve(n.top_layer + 1);
        for (uint32_t l = 0; l <= n.top_layer; ++l) {
            const std::span<const uint32_t> nb = Neighbors(i, l);
            per_layer.emplace_back(nb.begin(), nb.end());
        }
        s.neighbors.push_back(std::move(per_layer));
    }
    return s;
}

Result<std::unique_ptr<HnswIndex>> HnswIndex::FromSnapshot(Snapshot snapshot) {
    if (snapshot.dim == 0) return Status::Corruption("snapshot has zero dimensions");

    const size_t n = snapshot.docs.size();
    if (snapshot.top_layers.size() != n || snapshot.neighbors.size() != n) {
        return Status::Corruption("snapshot node arrays disagree on length");
    }
    if (snapshot.vectors.size() != n * snapshot.dim) {
        return Status::Corruption("snapshot vector block is the wrong size");
    }
    if (n == 0) {
        if (snapshot.entry_point != kNoEntryPoint) {
            return Status::Corruption("empty snapshot names an entry point");
        }
    } else {
        if (snapshot.entry_point >= n) return Status::Corruption("entry point out of range");
        if (snapshot.top_layers[snapshot.entry_point] != snapshot.max_layer) {
            // The entry point is by construction the highest-layer node. A file
            // claiming otherwise would make the descent start below the top layer,
            // losing recall in a way that looks like a tuning problem.
            return Status::Corruption("entry point is not on the top layer");
        }
    }

    auto index = std::unique_ptr<HnswIndex>(new HnswIndex(snapshot.dim, snapshot.params));
    // The clamps in the constructor must not silently change the degree bounds the
    // file was written with, or LinkOffset would read the adjacency at the wrong
    // stride.
    if (index->params_.M != snapshot.params.M || index->params_.M0 != snapshot.params.M0) {
        return Status::Corruption("snapshot degree bounds are out of range");
    }
    index->rng_.set_state(snapshot.rng_state);
    index->entry_point_ = snapshot.entry_point;
    index->max_layer_ = snapshot.max_layer;
    index->vectors_ = std::move(snapshot.vectors);
    index->visit_stamp_.assign(n, 0u);
    index->nodes_.resize(n);

    for (size_t i = 0; i < n; ++i) {
        Node& node = index->nodes_[i];
        node.doc = snapshot.docs[i];
        node.top_layer = snapshot.top_layers[i];
        if (node.top_layer > snapshot.max_layer) {
            return Status::Corruption("node layer above the index maximum");
        }
        if (snapshot.neighbors[i].size() != node.top_layer + 1u) {
            return Status::Corruption("node adjacency has the wrong layer count");
        }
        node.norm = Norm(index->vectors_.data() + i * snapshot.dim, snapshot.dim);
        node.degrees.assign(node.top_layer + 1, 0);
        size_t capacity = 0;
        for (uint32_t l = 0; l <= node.top_layer; ++l) capacity += index->MaxDegree(l);
        node.links.assign(capacity, 0);

        for (uint32_t l = 0; l <= node.top_layer; ++l) {
            const std::vector<uint32_t>& nb = snapshot.neighbors[i][l];
            if (nb.size() > index->MaxDegree(l)) {
                return Status::Corruption("node exceeds its degree bound at layer " +
                                          std::to_string(l));
            }
            const size_t offset = index->LinkOffset(l);
            for (size_t j = 0; j < nb.size(); ++j) {
                if (nb[j] >= n) return Status::Corruption("edge points past the last node");
                node.links[offset + j] = nb[j];
            }
            node.degrees[l] = static_cast<uint32_t>(nb.size());
        }
    }

    return index;
}

}  // namespace meridian::ann
