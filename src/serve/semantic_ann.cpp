#include "semantic_ann.hpp"

#include <algorithm>
#include <string>

namespace meridian::serve {

Result<std::unique_ptr<ann::HnswIndex>> BuildHnswOverSnapshot(
    const ReferenceIndex::Snapshot& snapshot, ann::HnswParams params, HnswBuildReport* report) {
    HnswBuildReport local;

    // The dimension comes from the first embedded document rather than from a caller
    // argument. A caller that had to supply it would be restating something the corpus
    // already knows, and a mismatch between the two would be reported as "query has 10
    // dimensions, index has 12" at search time, long after the information needed to fix it
    // was discarded.
    for (const auto& segment : snapshot.segments) {
        for (size_t local_id = 0; local_id < segment->docs.size(); ++local_id) {
            const DocId global = segment->doc_base + static_cast<DocId>(local_id);
            if (snapshot.IsDeleted(global)) continue;
            const std::vector<float>& embedding = segment->docs[local_id].embedding;
            if (embedding.empty()) {
                ++local.without_embedding;
                continue;
            }
            if (local.dim == 0) local.dim = static_cast<uint32_t>(embedding.size());
        }
    }

    // A snapshot with nothing to index is refused rather than turned into a zero-dimension
    // index. `HnswIndex::Search` checks the query's dimension before it checks whether the
    // graph is empty, so a dim-0 index rejects every real query with an argument error — the
    // whole query fails, and the message blames the query for a corpus that simply has no
    // embeddings in it.
    if (local.dim == 0) {
        return Status::InvalidArgument(
            "snapshot has no embedded documents, so there is no ANN index to build (" +
            std::to_string(local.without_embedding) + " live document(s) carried no embedding)");
    }

    auto index = std::make_unique<ann::HnswIndex>(local.dim, params);
    for (const auto& segment : snapshot.segments) {
        for (size_t local_id = 0; local_id < segment->docs.size(); ++local_id) {
            const DocId global = segment->doc_base + static_cast<DocId>(local_id);
            if (snapshot.IsDeleted(global)) continue;
            const std::vector<float>& embedding = segment->docs[local_id].embedding;
            if (embedding.empty()) continue;
            Status added = index->Add(global, embedding);
            if (!added.ok()) {
                return added.WithContext("document " + segment->docs[local_id].meta.external_id);
            }
            ++local.vectors;
        }
    }
    MERIDIAN_TRY(index->Build());

    if (report != nullptr) *report = local;
    return index;
}

Result<std::vector<Candidate>> AnnSemanticStage::Search(const StageContext& ctx,
                                                       const Query& query) const {
    if (inner_ == nullptr) return Status::Internal("ANN semantic stage has no inner stage");
    auto snapshot = SnapshotFrom(*ctx.pin);
    if (!snapshot) return Status::Internal("semantic stage received a pin it cannot read");
    if (!query.embedding) return std::vector<Candidate>{};

    const bool filtered = !query.filters.empty();
    Query deeper = query;
    if (filtered) {
        // Saturating rather than wrapping: candidate_k is a uint32_t that a caller controls,
        // and a multiply that overflowed would ask for a tiny beam and look like the
        // over-fetch having no effect.
        const uint64_t wanted =
            static_cast<uint64_t>(query.candidate_k) * static_cast<uint64_t>(over_fetch_);
        deeper.candidate_k = static_cast<uint32_t>(std::min<uint64_t>(wanted, UINT32_MAX));
    }

    Result<std::vector<Candidate>> found = [&] {
        std::lock_guard<std::mutex> lock(mutex_);
        return inner_->Search(ctx, filtered ? deeper : query);
    }();
    if (!found.ok()) return found.status();

    std::vector<Candidate> candidates = found.take();
    // The inner list is already ordered by similarity, so erase-remove keeps that order and
    // the truncation below takes the best survivors. Re-sorting would be wasted work and
    // would also discard the ANN index's own tie-break, which is what makes two runs of the
    // same query return the same list.
    candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                    [&](const Candidate& candidate) {
                                        if (snapshot->IsDeleted(candidate.doc)) return true;
                                        if (!filtered) return false;
                                        const ReferenceIndex::Entry* doc =
                                            snapshot->Resolve(candidate.doc);
                                        // A candidate the snapshot cannot resolve is kept
                                        // rather than dropped, so the rerank stage reports it
                                        // as the pin-mismatch it is. Dropping it here would
                                        // turn a graph built over a different generation into
                                        // a quietly shorter result list.
                                        if (doc == nullptr) return false;
                                        return !query.filters.Matches(doc->meta);
                                    }),
                     candidates.end());
    if (candidates.size() > query.candidate_k) candidates.resize(query.candidate_k);
    return candidates;
}

}  // namespace meridian::serve
