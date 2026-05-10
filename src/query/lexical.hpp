#pragma once

// Diagnostic entry points into lexical retrieval.
//
// `LexicalRetriever::Retrieve` always orders a conjunctive merge rarest-first, because
// that is the right thing to do and a production knob for doing the wrong thing is a
// production knob for being slow. But the claim "rarest-first decodes fewer postings"
// is only a claim until something measures both, so the evaluation harness and the
// tests need a way to run the deliberately bad order. That is what this header is for,
// and it is why it lives in src/ rather than in the public include tree.

#include <span>
#include <vector>

#include "meridian/index.hpp"
#include "meridian/postings.hpp"
#include "meridian/retrieval.hpp"

namespace meridian::query {

/// True when the per-term position lists, given **in query order**, contain a run of
/// consecutive positions — i.e. the phrase occurs.
///
/// Shared with the reranker's `phrase_match` feature rather than reimplemented there.
/// Two copies of an adjacency check is how a retriever and a reranker come to disagree
/// about whether a document contains a phrase, and that disagreement surfaces as a
/// document that matches the query but ranks as though it does not.
///
/// Each span must be ascending; posting lists guarantee that and this relies on it to
/// binary-search rather than scan.
bool PositionsAreAdjacent(const std::vector<std::span<const Position>>& by_term);

enum class ConjunctiveOrder {
    /// Iterators sorted ascending by `size_hint`. The rarest term drives iteration, so
    /// cost is proportional to its posting count.
    kRarestFirst,
    /// Sorted descending. The commonest term drives iteration and every one of its
    /// postings is decoded. Exists to be measured against, not to be used.
    kCommonestFirst,
};

/// Runs a conjunctive retrieval with an explicit iterator order.
///
/// `stats_out`, when non-null, receives the counters for this call alone rather than a
/// running total, so a test can compute a ratio between two orders.
Result<std::vector<Candidate>> RetrieveConjunctiveOrdered(const Generation& gen,
                                                          const Query& query,
                                                          Bm25Params params,
                                                          FieldWeights weights,
                                                          ConjunctiveOrder order,
                                                          LexicalRetriever::Stats* stats_out);

}  // namespace meridian::query
