#pragma once

// Building an HNSW index over the stand-in corpus, and the stage that makes its output
// safe to fuse.
//
// Why this file exists at all: `EngineSemanticStage` in stages_engine.hpp already adapts
// an `AnnIndex` to the pipeline's semantic stage, but nothing built an `AnnIndex` over the
// corpus the CLIs and the evaluation harness actually load. The result was that every
// serving and eval path ran `ReferenceSemanticStage`, which is exact brute force, so
// `ef_search` was a parameter that reached no code — the sweep axis produced identical
// rows and src/ann/ was exercised only by its own unit tests. This file is the missing
// link, and it is deliberately small: the graph is src/ann/'s, the adaptation is
// stages_engine.hpp's, and what is left is the corpus-to-graph load plus the two things an
// ANN index cannot know about.
//
// It is kept out of stages_engine.cpp for the reason that file states about itself: that
// translation unit references `Index::Acquire` and `SegmentReader::Document`, so linking
// it drags in src/index/. This file needs neither, and a tool that wants an HNSW semantic
// path over the in-memory corpus should not have to link the on-disk index to get one.

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "../ann/hnsw.hpp"
#include "meridian/retrieval.hpp"
#include "pipeline.hpp"
#include "reference_index.hpp"

namespace meridian::serve {

struct HnswBuildReport {
    uint64_t vectors = 0;
    uint32_t dim = 0;

    /// Live documents that carried no embedding, so the ANN index cannot return them.
    ///
    /// Reported rather than ignored because it is the difference between "the semantic
    /// path missed this document" and "the semantic path was never able to see it", and
    /// those two call for opposite responses from whoever reads a recall number.
    uint64_t without_embedding = 0;
};

/// Builds an HNSW index over the live, embedded documents of one published snapshot.
///
/// Snapshot-shaped rather than incremental, because the graph is not generation-versioned:
/// `AnnIndex` has no notion of a pin, so the only honest thing to build it from is one
/// consistent view, named in the caller's output next to the version it came from. The
/// cost is that a generation published afterwards is invisible to the semantic path until
/// the index is rebuilt. That is acceptable here — the CLIs load a corpus once and then
/// query it — and would not be in a system that indexes continuously, where the graph
/// would need its own incremental publication scheme.
///
/// Document ids are the snapshot's global ids, which is what lets the downstream rerank
/// and materialize stages resolve a candidate the ANN path produced. Using local or
/// insertion-order ids would make every semantic candidate resolve to the wrong document
/// in a multi-segment index, and nothing downstream would notice: the ids are valid, so
/// the query succeeds and returns the wrong documents.
///
/// Fails rather than skipping a vector whose length disagrees with the first one seen. A
/// corpus with two embedding dimensions in it is a data-preparation bug, and dropping the
/// minority silently would show up only as unexplained missing recall.
Result<std::unique_ptr<ann::HnswIndex>> BuildHnswOverSnapshot(
    const ReferenceIndex::Snapshot& snapshot, ann::HnswParams params,
    HnswBuildReport* report = nullptr);

/// Wraps a semantic stage that reads an ANN index, and supplies what the index cannot.
///
/// An `AnnIndex` holds vectors and nothing else. It does not know which documents the
/// pinned generation has tombstoned, and it does not know the query's structured filters,
/// so its raw output can contain documents that must not be returned. `ReferenceSemanticStage`
/// checks both inline while it scans, because it is scanning the snapshot anyway; a graph
/// search has no such opportunity. Without this wrapper the two semantic stages disagree on
/// which documents are eligible, and the disagreement surfaces as a deleted or filtered-out
/// document appearing in the final results — a wrong answer that no test of either stage
/// alone would catch, since each is behaving exactly as written.
///
/// Post-filtering is a mitigation and not a fix, and the shape of its failure is worth
/// stating: a filter that selects one percent of the corpus removes almost everything the
/// graph returned, so the semantic path contributes a handful of candidates instead of
/// `candidate_k` of them. Over-fetching when a filter is present buys back some of that at
/// proportional cost. The real fix is a filter-aware traversal — evaluating the predicate
/// during the beam search so the beam keeps only eligible nodes — which `HnswIndex` does
/// not offer and which is a change to the graph search, not to the serving layer.
///
/// Calls to the inner stage are serialised. hnsw.hpp states that `Search` is not
/// thread-safe: it stamps a shared visited array and bumps shared counters, so two
/// concurrent searches corrupt each other's traversal and return wrong neighbours rather
/// than crashing. The pipeline runs the semantic path on a helper thread and a server runs
/// several queries at once, so concurrent entry here is ordinary, not exotic. A lock is the
/// cost of honouring that: it serialises the ANN path across concurrent queries, which is
/// the wrong answer for a real deployment — there the fix is one index per shard with
/// per-thread scratch space, which is why hnsw.hpp refuses to put a mutex inside the index
/// and promise scalability it does not have.
class AnnSemanticStage : public SemanticStage {
public:
    /// `over_fetch` multiplies `candidate_k` when the query carries a filter. 1 disables
    /// over-fetching, which is the right setting when measuring the ANN path itself: a
    /// stage that quietly searched deeper than it was asked to would make an `ef_search`
    /// row report a cost that no configured value explains.
    AnnSemanticStage(const SemanticStage* inner, uint32_t over_fetch = 4)
        : inner_(inner), over_fetch_(over_fetch == 0 ? 1 : over_fetch) {}

    Result<std::vector<Candidate>> Search(const StageContext& ctx,
                                          const Query& query) const override;

private:
    const SemanticStage* inner_;
    uint32_t over_fetch_;
    mutable std::mutex mutex_;
};

}  // namespace meridian::serve
