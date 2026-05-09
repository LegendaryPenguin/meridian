// Tokenization and query parsing.
//
// The first group of tests exists for one reason: an analysis mismatch between the index
// and query paths returns zero results instead of an error, so no amount of
// "did it fail" testing finds it. The only thing that does is running the same input
// through both entry points and comparing.

#include <string>
#include <vector>

#include "../src/query/parse.hpp"
#include "../src/query/tokenize.hpp"
#include "testing.hpp"

using namespace meridian;            // NOLINT: test-local convenience
using namespace meridian::query;     // NOLINT

namespace {

std::string Join(const std::vector<std::string>& tokens) {
    std::string out;
    for (const std::string& t : tokens) {
        if (!out.empty()) out += '|';
        out += t;
    }
    return out;
}

/// Runs one input through both entry points and returns the shared result, failing if
/// they differ.
std::vector<std::string> BothPaths(const std::string& text, const Tokenizer& tokenizer,
                                   const char* file, int line) {
    std::vector<std::string> indexed = TokenizeForIndex(text, tokenizer);
    std::vector<std::string> queried = TokenizeForQuery(text, tokenizer);
    if (indexed != queried) {
        meridian::testing::ReportFailure(
            file, line,
            "index and query analysis disagree on \"" + text + "\": index produced [" +
                Join(indexed) + "], query produced [" + Join(queried) + "]");
    }
    return indexed;
}

#define BOTH_PATHS(text, tokenizer) BothPaths((text), (tokenizer), __FILE__, __LINE__)

}  // namespace

MERIDIAN_TEST(TokenizationIsIdenticalAcrossBothPaths) {
    // Stemming off here so the comparison is about splitting and folding; the stemmer
    // gets its own test below and is exercised through both paths in the next case.
    TokenizerOptions options;
    options.stem = false;
    options.remove_stopwords = false;
    const Tokenizer tokenizer(options);

    CHECK_EQ(Join(BOTH_PATHS("Hello World", tokenizer)), "hello|world");
    CHECK_EQ(Join(BOTH_PATHS("MiXeD CaSe AND lower", tokenizer)), "mixed|case|and|lower");
    CHECK_EQ(Join(BOTH_PATHS("punctuation, here; really---yes!", tokenizer)),
             "punctuation|here|really|yes");
    CHECK_EQ(Join(BOTH_PATHS("numbers 42 and 3.14 and x86", tokenizer)),
             "numbers|42|and|3|14|and|x86");
    CHECK_EQ(Join(BOTH_PATHS("don't split's", tokenizer)), "don|t|split|s");

    // UTF-8: the multi-byte sequences survive byte-for-byte and the ASCII around them
    // folds. A byte-wise tolower would have corrupted these.
    CHECK_EQ(Join(BOTH_PATHS("Café RÉSUMÉ naïve", tokenizer)), "café|rÉsumÉ|naïve");
    CHECK_EQ(Join(BOTH_PATHS("日本語 検索", tokenizer)), "日本語|検索");
    CHECK_EQ(Join(BOTH_PATHS("emoji \xF0\x9F\x8E\xAF target", tokenizer)),
             "emoji|\xF0\x9F\x8E\xAF|target");

    // Structurally invalid UTF-8 is a boundary, not a term constituent. "ab\xFFcd"
    // becomes two terms rather than one term containing a byte nothing can query for.
    CHECK_EQ(Join(BOTH_PATHS(std::string("ab\xFF") + "cd", tokenizer)), "ab|cd");
    // A truncated sequence at end of input must not read past the end; ASAN is the real
    // assertion here.
    CHECK_EQ(Join(BOTH_PATHS(std::string("trail \xE6\x97"), tokenizer)), "trail");

    CHECK_EQ(Join(BOTH_PATHS("", tokenizer)), "");
    CHECK_EQ(Join(BOTH_PATHS("   \t\n  ", tokenizer)), "");
}

MERIDIAN_TEST(StemmingIsIdenticalAcrossBothPaths) {
    TokenizerOptions options;
    options.remove_stopwords = false;
    const Tokenizer tokenizer(options);

    // The point of the case: an index that stems and a query that does not produces
    // zero results for "running" against a document containing "running".
    CHECK_EQ(Join(BOTH_PATHS("Running Runner RUNS", tokenizer)), "run|runner|run");
    CHECK_EQ(Join(BOTH_PATHS("retrieval RETRIEVING retrieved", tokenizer)),
             "retriev|retriev|retriev");
    CHECK_EQ(Join(BOTH_PATHS("Caresses ponies caress cats", tokenizer)),
             "caress|poni|caress|cat");
    CHECK_EQ(Join(BOTH_PATHS("relational conditional rational", tokenizer)),
             "relat|condit|ration");

    // Not stemmed, by the documented restriction: digits and non-ASCII are left alone
    // because every Porter rule assumes English letters.
    CHECK_EQ(Join(BOTH_PATHS("x86_64 42s Café", tokenizer)), "x86|64|42s|café");
}

MERIDIAN_TEST(StopwordsAreRemovedButNeverAllOfThem) {
    const Tokenizer tokenizer;

    CHECK_EQ(Join(tokenizer.Tokenize("the quick brown fox")), "quick|brown|fox");

    // The documented non-empty guarantee. Returning nothing here would make the query
    // match nothing, which is indistinguishable from an index bug.
    CHECK_EQ(Join(tokenizer.Tokenize("the and of")), "the|and|of");
    CHECK_EQ(Join(tokenizer.Tokenize("to be or not to be")), "to|be|or|not|to|be");
}

MERIDIAN_TEST(MismatchedOptionsAreWhatThisGuardsAgainst) {
    // Demonstrates the failure the shared entry point prevents: one side folding case
    // and the other not produces terms that can never meet in the dictionary.
    TokenizerOptions folding;
    TokenizerOptions not_folding;
    not_folding.lowercase = false;

    const std::vector<std::string> a = TokenizeForIndex("Postings", Tokenizer(folding));
    const std::vector<std::string> b = TokenizeForQuery("Postings", Tokenizer(not_folding));
    CHECK_NE(Join(a), Join(b));
    CHECK_EQ(Join(a), "post");
    // Note the second-order damage: with folding off the token is not all-lowercase, so
    // the stemmer declines it too and the divergence is larger than the one option that
    // was changed. One misconfigured flag produces two mismatches, and the only symptom
    // either way is an empty result list.
    CHECK_EQ(Join(b), "Postings");
}

MERIDIAN_TEST(ParsesTermsAndDefaultsToDisjunctive) {
    Result<Query> parsed = Parse("Posting Lists");
    CHECK_OK(parsed.ok() ? Status::Ok() : parsed.status());
    CHECK_EQ(parsed.value().raw, std::string("Posting Lists"));
    CHECK_EQ(Join(parsed.value().terms), "post|list");
    CHECK(parsed.value().mode == QueryMode::kDisjunctive);
    CHECK(parsed.value().filters.empty());
}

MERIDIAN_TEST(ParsesExplicitOperators) {
    Result<Query> conj = Parse("alpha AND beta");
    CHECK(conj.ok());
    CHECK(conj.value().mode == QueryMode::kConjunctive);
    CHECK_EQ(Join(conj.value().terms), "alpha|beta");

    Result<Query> disj = Parse("alpha OR beta");
    CHECK(disj.ok());
    CHECK(disj.value().mode == QueryMode::kDisjunctive);

    // Repeating the same operator is fine; it is only a conflict that is refused.
    CHECK(Parse("a AND b AND c").ok());

    Result<Query> mixed = Parse("alpha AND beta OR gamma");
    CHECK(!mixed.ok());
    CHECK(mixed.status().code() == Code::kQueryError);
    CHECK(mixed.status().message().find("offset 15") != std::string::npos);
}

MERIDIAN_TEST(ParsesPhrasesAndRejectsUnbalancedQuotes) {
    Result<Query> phrase = Parse("\"machine learning\"");
    CHECK(phrase.ok());
    CHECK(phrase.value().mode == QueryMode::kPhrase);
    CHECK_EQ(Join(phrase.value().terms), "machin|learn");

    // Stopwords survive inside a phrase: the adjacency constraint is over what was
    // typed, so dropping "to" would make this match "not be" instead.
    Result<Query> with_stopwords = Parse("\"not to be\"");
    CHECK(with_stopwords.ok());
    CHECK_EQ(Join(with_stopwords.value().terms), "not|to|be");

    Result<Query> unbalanced = Parse("red \"shoes");
    CHECK(!unbalanced.ok());
    CHECK(unbalanced.status().code() == Code::kQueryError);
    // Named by the position of the *opening* quote, which is the character the user has
    // to go and look at.
    CHECK(unbalanced.status().message().find("offset 4") != std::string::npos);

    // A phrase cannot be combined with loose terms, because Query carries one mode.
    Result<Query> mixed = Parse("\"machine learning\" tutorial");
    CHECK(!mixed.ok());
    CHECK(mixed.status().message().find("offset 19") != std::string::npos);

    Result<Query> two = Parse("\"a b\" \"c d\"");
    CHECK(!two.ok());

    Result<Query> empty_phrase = Parse("\"   \"");
    CHECK(!empty_phrase.ok());
}

MERIDIAN_TEST(RejectsEmptyQueries) {
    for (const char* input : {"", "    ", "\t\n", "!!!", "---", "AND"}) {
        Result<Query> parsed = Parse(input);
        CHECK(!parsed.ok());
        if (parsed.ok()) continue;
        CHECK(parsed.status().code() == Code::kQueryError);
        CHECK(parsed.status().message().find("offset 0") != std::string::npos);
    }
}

MERIDIAN_TEST(AcceptsAQueryOfOnlyStopwords) {
    // Decided behaviour: accepted, with the stopwords retained. "the who" is a real
    // query and a parser that turns it into zero terms turns it into zero results.
    Result<Query> parsed = Parse("the who");
    CHECK(parsed.ok());
    CHECK_EQ(Join(parsed.value().terms), "the|who");

    Result<Query> all_stop = Parse("the and of");
    CHECK(all_stop.ok());
    CHECK_EQ(parsed.value().terms.size(), size_t{2});
    CHECK_EQ(all_stop.value().terms.size(), size_t{3});
}

MERIDIAN_TEST(ParsesFilters) {
    Result<Query> parsed = Parse("search language:en category:news after:100 before:200");
    CHECK(parsed.ok());
    const Query& q = parsed.value();
    CHECK_EQ(Join(q.terms), "search");
    CHECK(q.filters.language.has_value());
    CHECK_EQ(*q.filters.language, std::string("en"));
    CHECK_EQ(*q.filters.category, std::string("news"));
    CHECK_EQ(*q.filters.published_after, int64_t{100});
    CHECK_EQ(*q.filters.published_before, int64_t{200});

    // Quoted values, for categories with spaces.
    Result<Query> quoted = Parse("cats category:\"big animals\"");
    CHECK(quoted.ok());
    CHECK_EQ(*quoted.value().filters.category, std::string("big animals"));

    // Filter values are not analysed: they are structured data, so the capital survives
    // and the stemmer does not touch it.
    Result<Query> exact = Parse("x category:Reviews");
    CHECK(exact.ok());
    CHECK_EQ(*exact.value().filters.category, std::string("Reviews"));

    // A filter-only query is legal: "browse the filtered set".
    Result<Query> browse = Parse("language:fr");
    CHECK(browse.ok());
    CHECK(browse.value().terms.empty());
    CHECK(!browse.value().filters.empty());
}

MERIDIAN_TEST(RejectsMalformedFiltersWithAPosition) {
    struct Case {
        const char* input;
        const char* expected_offset;
    };
    const Case cases[] = {
        {"x colour:red", "offset 2"},          // unknown field
        {"x language:", "offset 11"},          // empty value
        {"x after:soon", "offset 8"},          // non-integer timestamp
        {"x after:12x", "offset 8"},           // trailing garbage
        {"x after:\"100\"", "offset 9"},       // quoted integer
        {"x language:en language:fr", "offset 14"},  // contradictory duplicate
        {"x category:\"unclosed", "offset 11"},
    };
    for (const Case& c : cases) {
        Result<Query> parsed = Parse(c.input);
        CHECK(!parsed.ok());
        if (parsed.ok()) continue;
        CHECK(parsed.status().code() == Code::kQueryError);
        const bool found = parsed.status().message().find(c.expected_offset) != std::string::npos;
        if (!found) {
            meridian::testing::ReportFailure(__FILE__, __LINE__,
                                            std::string("parsing \"") + c.input +
                                                "\" reported \"" +
                                                parsed.status().message() +
                                                "\", expected it to name " +
                                                c.expected_offset);
        }
    }
}

MERIDIAN_TEST(ParseCarriesBudgetsThrough) {
    ParseOptions options;
    options.top_k = 5;
    options.candidate_k = 50;
    options.deadline_ms = 17;
    options.default_mode = QueryMode::kConjunctive;

    Result<Query> parsed = Parse("alpha beta", options);
    CHECK(parsed.ok());
    CHECK_EQ(parsed.value().top_k, uint32_t{5});
    CHECK_EQ(parsed.value().candidate_k, uint32_t{50});
    CHECK_EQ(parsed.value().deadline_ms, uint32_t{17});
    CHECK(parsed.value().mode == QueryMode::kConjunctive);
}

MERIDIAN_TEST(FiltersMatchDocumentMeta) {
    DocumentMeta doc;
    doc.language = "en";
    doc.category = "news";
    doc.published_at = 150;

    Filters none;
    CHECK(none.empty());
    CHECK(none.Matches(doc));

    Filters language;
    language.language = "en";
    CHECK(language.Matches(doc));
    language.language = "fr";
    CHECK(!language.Matches(doc));

    Filters range;
    range.published_after = 150;
    range.published_before = 150;
    CHECK(range.Matches(doc));  // bounds are inclusive on both ends
    range.published_after = 151;
    CHECK(!range.Matches(doc));
}

MERIDIAN_TEST_MAIN()
