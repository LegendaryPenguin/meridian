#pragma once

// Tokenization and normalization, shared by the indexing path and the query path.
//
// THIS IS THE SINGLE SHARED ENTRY POINT. `Tokenizer::Tokenize` is called by both
// `TokenizeForIndex` and `TokenizeForQuery`, which exist only as named aliases so the
// call sites read clearly and so a test can assert the two paths agree. There is
// deliberately no second implementation to drift out of sync.
//
// The reason for the insistence: if the index lowercases and the query does not, the
// posting lookup misses and the system returns *zero results rather than an error*.
// Nothing in the stack can distinguish that from "no matching documents", so the bug
// survives every integration test that only checks for absence of failure. The same
// applies to stemming and to stopword handling. Any analysis asymmetry here is
// invisible in production and expensive to find.
//
// Unicode policy, stated because the alternatives are all worse:
//
//  - Case folding is applied to ASCII only. Folding the rest of Unicode needs a case
//    table this project does not ship, and a byte-wise `tolower` over UTF-8 corrupts
//    multi-byte sequences. So non-ASCII code points pass through byte-for-byte: "Café"
//    becomes "café" (the C folds, the é is preserved), and "ÜBER" stays "ÜBER". That
//    is a recall limitation, not a correctness one, and it is symmetric across both
//    paths, which is the property that actually matters.
//
//  - Multi-byte UTF-8 sequences are word constituents. Splitting on "non-alphanumeric"
//    by byte value would split every accented word in half.
//
//  - Structurally invalid UTF-8 is treated as a token boundary and dropped, rather
//    than passed through. A corrupt byte that reaches the term dictionary makes a term
//    nothing can ever query for.

#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace meridian::query {

/// Analysis configuration. One object configures both paths; see the file comment.
struct TokenizerOptions {
    /// Fold ASCII uppercase. Off is supported only so the tests can demonstrate what
    /// an index/query mismatch looks like.
    bool lowercase = true;

    /// Apply the Porter subset in `Stem`.
    bool stem = true;

    /// Drop stopwords. Subject to the non-empty guarantee documented on `Tokenize`.
    bool remove_stopwords = true;

    /// Tokens longer than this are truncated. A single 4 MB "word" from a broken
    /// document should not become a 4 MB dictionary key.
    size_t max_token_bytes = 64;
};

class Tokenizer {
public:
    Tokenizer() = default;
    explicit Tokenizer(TokenizerOptions options) : options_(options) {}

    /// Splits `text` into normalized terms.
    ///
    /// Guarantee relied on by the parser: if stopword removal would empty a non-empty
    /// token list, the stopwords are kept instead. A query of "the who" that analyses
    /// to zero terms cannot match anything, and silently matching nothing is the worst
    /// available answer. The indexing path never hits this case in practice (documents
    /// are long) so the asymmetry is theoretical, and it is still the same function.
    std::vector<std::string> Tokenize(std::string_view text) const;

    /// Normalizes one already-split token. Exposed because phrase parsing has to
    /// normalize a quoted span term by term, and it must use exactly this.
    std::string Normalize(std::string_view token) const;

    bool IsStopword(std::string_view token) const;

    /// A small English stopword list. Deliberately short: aggressive stopword removal
    /// breaks phrase queries ("to be or not to be" analyses to nothing) and BM25
    /// already discounts common terms through IDF, so the list only needs to cover
    /// the terms whose posting lists are large enough to be worth not decoding.
    static const std::unordered_set<std::string>& DefaultStopwords();

    /// Porter stemmer, steps 1a, 1b, 1c, 2, 3, 4, 5a and 5b.
    ///
    /// Documented divergences from the published algorithm:
    ///  - Applied only to tokens that are entirely ASCII letters. A token containing
    ///    digits or non-ASCII bytes is returned unchanged, because every rule below
    ///    assumes the English consonant/vowel alphabet and would mangle "x86_64" or
    ///    "café".
    ///  - Tokens of two letters or fewer are returned unchanged. Porter's measure is
    ///    zero for them and every rule is a no-op anyway, so this is a short-circuit
    ///    rather than a behaviour change.
    ///  - Step 2 and step 4 use the standard suffix tables; step 2's "logi" -> "log"
    ///    extension from later revisions is included because it is what makes
    ///    "analogi"/"analog" conflate.
    static std::string Stem(std::string_view word);

    const TokenizerOptions& options() const { return options_; }

private:
    TokenizerOptions options_;
};

/// The indexing path's entry point.
///
/// Forwards to `Tokenizer::Tokenize`. It is a named alias and nothing more; see the
/// file comment for why there is no separate implementation.
std::vector<std::string> TokenizeForIndex(std::string_view text, const Tokenizer& tokenizer);

/// The query path's entry point. Same function, same options object.
std::vector<std::string> TokenizeForQuery(std::string_view text, const Tokenizer& tokenizer);

}  // namespace meridian::query
