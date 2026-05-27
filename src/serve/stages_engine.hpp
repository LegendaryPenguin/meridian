#pragma once

// Adapters from the serving layer's stage interfaces to the engine's own types.
//
// These are thin by design: every one of them translates a `GenerationPin` into the
// `const Generation&` the engine takes, forwards the call, and returns. No logic lives
// here, because logic that lived here would be logic the engine's own tests do not
// cover.
//
// Why these are in their own translation unit, separate from the stand-in stages in
// reference_index.cpp: they reference `Index::Acquire`, `Generation::Resolve` and
// `SegmentReader::Document`, which are owned by src/index/. In a static library the
// linker pulls in an object file only when something it defines is referenced, so a
// binary that uses the stand-in stages does not depend on src/index/ at all, and a binary
// that uses these gets an honest unresolved-symbol error naming exactly what is missing
// when it does not. That mattered concretely: these were written while src/index/ was
// still empty, and merging the two sets of stages into one file would have made every
// test and tool in the project unlinkable until it landed.

#include <memory>
#include <vector>

#include "meridian/index.hpp"
#include "meridian/retrieval.hpp"
#include "pipeline.hpp"

namespace meridian::serve {

/// Pins the index's current generation. The shared_ptr is the reference count
/// index.hpp describes: a merge waits on it before unlinking the files it replaced.
class IndexGenerationSource : public GenerationSource {
public:
    explicit IndexGenerationSource(const Index* index) : index_(index) {}
    GenerationPin Acquire() const override;

private:
    const Index* index_;
};

class EngineLexicalStage : public LexicalStage {
public:
    explicit EngineLexicalStage(const LexicalRetriever* retriever) : retriever_(retriever) {}
    Result<std::vector<Candidate>> Retrieve(const StageContext& ctx,
                                            const Query& query) const override;

private:
    const LexicalRetriever* retriever_;
};

/// Wraps an ANN index. `ef_search` lives here rather than in Query because it is a
/// property of how this deployment is tuned, not of what the caller asked for.
class EngineSemanticStage : public SemanticStage {
public:
    EngineSemanticStage(const AnnIndex* ann, uint32_t ef_search)
        : ann_(ann), ef_search_(ef_search) {}
    Result<std::vector<Candidate>> Search(const StageContext& ctx,
                                          const Query& query) const override;

private:
    const AnnIndex* ann_;
    uint32_t ef_search_;
};

class EngineMergeStage : public MergeStage {
public:
    explicit EngineMergeStage(const HybridMerger* merger) : merger_(merger) {}
    std::vector<Candidate> Merge(const std::vector<Candidate>& lexical,
                                const std::vector<Candidate>& semantic,
                                uint32_t candidate_k) const override;

private:
    const HybridMerger* merger_;
};

class EngineRerankStage : public RerankStage {
public:
    explicit EngineRerankStage(const Reranker* reranker) : reranker_(reranker) {}
    Result<std::vector<Hit>> Rank(const StageContext& ctx, const Query& query,
                                  const std::vector<Candidate>& candidates) const override;

private:
    const Reranker* reranker_;
};

/// Fills external ids by resolving each hit through the pinned generation.
class EngineMaterializeStage : public MaterializeStage {
public:
    Status Materialize(const StageContext& ctx, const Query& query,
                       std::vector<Hit>& hits) const override;
};

}  // namespace meridian::serve
