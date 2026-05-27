#include "stages_engine.hpp"

namespace meridian::serve {
namespace {

/// Every adapter needs the same thing and must fail the same way when it does not get
/// it: a pin carrying an engine generation. Returning an empty candidate list instead
/// would turn a wiring mistake into a query that succeeds with no results.
const Generation* EngineGeneration(const StageContext& ctx) {
    if (ctx.pin == nullptr) return nullptr;
    return ctx.pin->generation.get();
}

}  // namespace

GenerationPin IndexGenerationSource::Acquire() const {
    GenerationPin pin;
    if (index_ == nullptr) return pin;
    pin.generation = index_->Acquire();
    if (pin.generation) pin.version = pin.generation->version();
    return pin;
}

Result<std::vector<Candidate>> EngineLexicalStage::Retrieve(const StageContext& ctx,
                                                           const Query& query) const {
    const Generation* generation = EngineGeneration(ctx);
    if (retriever_ == nullptr || generation == nullptr) {
        return Status::Internal("lexical stage has no retriever or no engine generation");
    }
    return retriever_->Retrieve(*generation, query);
}

Result<std::vector<Candidate>> EngineSemanticStage::Search(const StageContext& ctx,
                                                          const Query& query) const {
    if (ann_ == nullptr) return Status::Internal("semantic stage has no ANN index");
    if (!query.embedding) return std::vector<Candidate>{};
    (void)ctx;  // the ANN index is generation-independent; see the note below
    // The ANN index is not pinned the way the inverted index is. That is a real
    // difference and not an oversight: the graph is built over a document set and
    // searched by vector, so a candidate it returns still has to be resolved and
    // filtered against the pinned generation downstream. A candidate for a document the
    // pinned generation does not contain is dropped by the reranker's resolve, which is
    // why that resolve failure is treated as an error rather than skipped.
    return ann_->Search(*query.embedding, query.candidate_k, ef_search_);
}

std::vector<Candidate> EngineMergeStage::Merge(const std::vector<Candidate>& lexical,
                                              const std::vector<Candidate>& semantic,
                                              uint32_t candidate_k) const {
    if (merger_ == nullptr) return {};
    return merger_->Merge(lexical, semantic, candidate_k);
}

Result<std::vector<Hit>> EngineRerankStage::Rank(const StageContext& ctx, const Query& query,
                                                const std::vector<Candidate>& candidates) const {
    const Generation* generation = EngineGeneration(ctx);
    if (reranker_ == nullptr || generation == nullptr) {
        return Status::Internal("rerank stage has no reranker or no engine generation");
    }
    return reranker_->Rank(*generation, query, candidates);
}

Status EngineMaterializeStage::Materialize(const StageContext& ctx, const Query& query,
                                           std::vector<Hit>& hits) const {
    const Generation* generation = EngineGeneration(ctx);
    if (generation == nullptr) return Status::Internal("materialize stage has no engine generation");
    (void)query;

    for (Hit& hit : hits) {
        auto resolved = generation->Resolve(hit.doc);
        if (!resolved.ok()) return resolved.status();
        const auto [segment, local] = resolved.value();
        auto document = segment->Document(local);
        if (!document.ok()) return document.status();
        hit.external_id = document.value().external_id;
    }
    return Status::Ok();
}

}  // namespace meridian::serve
