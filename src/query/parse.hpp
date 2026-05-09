#pragma once

// Query parsing: text in, a `Query` out, or a `Status::QueryError` naming the byte
// offset that failed.
//
// The syntax is small on purpose. Every operator is a thing the retrieval path has to
// implement well, and a parser that accepts more than the evaluator does well is a
// generator of slow queries.
//
// Grammar:
//
//   query      := element*
//   element    := filter | phrase | operator | term
//   filter     := field ':' value
//   field      := "language" | "category" | "after" | "before"
//   value      := bare-word | '"' ... '"'
//   phrase     := '"' term (WS term)* '"'
//   operator   := "AND" | "OR"          (uppercase, standalone)
//   term       := any run of token characters
//
// Defined behaviour for the three awkward inputs, each decided rather than left to
// fall out of the implementation:
//
//  1. **Unbalanced quote** -> `QueryError`, naming the offset of the *opening* quote.
//     Rejected rather than auto-closed at end of input. Auto-closing turns a typo into
//     a silently different query — `"red shoes` would become the phrase "red shoes"
//     instead of the OR query the user was in the middle of typing — and a phrase
//     query and an OR query return very different result sets.
//
//  2. **Empty query** (empty, whitespace-only, or punctuation-only) -> `QueryError` at
//     offset 0. The alternative, matching every document, is a full-corpus scan
//     triggered by an accidental submit, and it is the single easiest way to take a
//     search tier down.
//
//  3. **Only stopwords** -> accepted, with the stopwords retained as terms. This falls
//     out of the guarantee on `Tokenizer::Tokenize`: dropping them would leave zero
//     terms and return zero results, which is indistinguishable from a bug in the
//     index. "the who" and "to be or not to be" are real queries. They are expensive
//     because the posting lists are long, and that is the retriever's problem to bound
//     with `candidate_k`, not the parser's problem to pretend away.
//
// Two further decisions worth naming:
//
//  4. A query with filters and no terms (`language:en`) is accepted. It means "browse
//     the filtered set"; see `LexicalRetriever::Retrieve` for what it scores.
//
//  5. Mixing `AND` and `OR` in one query is a `QueryError` naming the offset of the
//     second, different operator. `Query` carries one `mode` for the whole query, so
//     there is nowhere to put a mixed boolean tree, and silently letting the last
//     operator win would make `a AND b OR c` quietly mean something nobody asked for.

#include <string_view>

#include "meridian/retrieval.hpp"
#include "meridian/status.hpp"
#include "tokenize.hpp"

namespace meridian::query {

struct ParseOptions {
    /// Mode when the query contains no explicit operator. Disjunctive, because OR is
    /// the recall-preserving default and the reranker is what restores precision.
    QueryMode default_mode = QueryMode::kDisjunctive;

    uint32_t top_k = 10;
    uint32_t candidate_k = 200;
    uint32_t deadline_ms = 250;

    TokenizerOptions tokenizer;
};

/// Parses `text`. The returned `Query` has `raw` set to the input unchanged, so a log
/// line can show what the user typed next to what it was understood as.
Result<Query> Parse(std::string_view text, const ParseOptions& options = {});

}  // namespace meridian::query
