#include "parse.hpp"

#include <charconv>
#include <string>

namespace meridian {

// Filters are evaluated against DocumentMeta, not against the term dictionary. The
// header on `Filters` explains why; the consequence here is that this is an ordinary
// struct comparison and costs nothing next to a posting decode.
//
// The date bounds are *inclusive* on both ends, so [after, before] is a closed
// interval and `after:X before:X` selects exactly the documents at X rather than
// nothing. Half-open would be defensible; silently differing between the two ends
// would not.
bool Filters::Matches(const DocumentMeta& doc) const {
    if (language && *language != doc.language) return false;
    if (category && *category != doc.category) return false;
    if (published_after && doc.published_at < *published_after) return false;
    if (published_before && doc.published_at > *published_before) return false;
    return true;
}

}  // namespace meridian

namespace meridian::query {
namespace {

bool IsSpace(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

std::string At(size_t offset) { return " at offset " + std::to_string(offset); }

enum class FilterField { kLanguage, kCategory, kAfter, kBefore, kUnknown };

FilterField ClassifyField(std::string_view name) {
    if (name == "language" || name == "lang") return FilterField::kLanguage;
    if (name == "category" || name == "cat") return FilterField::kCategory;
    if (name == "after" || name == "published_after") return FilterField::kAfter;
    if (name == "before" || name == "published_before") return FilterField::kBefore;
    return FilterField::kUnknown;
}

/// Applies one `field:value` pair.
///
/// Filter values are structured data, not text, so they are *not* analysed: no
/// lowercasing and no stemming. Running the stemmer over them would make
/// `category:dining` fail against the stored category "dining", and lowercasing would
/// make `category:C++` unmatchable against a corpus that stores it with capitals.
/// Exact bytes, compared exactly.
///
/// `quoted_value` rejects `after:"123"`, because a quoted integer is more likely a
/// user confusing the two syntaxes than a deliberate request.
Status ApplyFilter(Filters& filters, std::string_view field, std::string_view value,
                   size_t field_offset, size_t value_offset, bool quoted_value) {
    FilterField which = ClassifyField(field);
    if (which == FilterField::kUnknown) {
        return Status::QueryError("unknown filter field '" + std::string(field) + "'" +
                                  At(field_offset) +
                                  " (known: language, category, after, before)");
    }
    if (value.empty()) {
        return Status::QueryError("empty filter value" + At(value_offset));
    }

    switch (which) {
        case FilterField::kLanguage:
            if (filters.language) {
                return Status::QueryError(
                    "duplicate language filter" + At(field_offset) +
                    ": two values for one field can never both match");
            }
            filters.language = std::string(value);
            return Status::Ok();

        case FilterField::kCategory:
            if (filters.category) {
                return Status::QueryError(
                    "duplicate category filter" + At(field_offset) +
                    ": two values for one field can never both match");
            }
            filters.category = std::string(value);
            return Status::Ok();

        case FilterField::kAfter:
        case FilterField::kBefore: {
            if (quoted_value) {
                return Status::QueryError("filter '" + std::string(field) +
                                          "' takes a bare integer timestamp, not a "
                                          "quoted string" + At(value_offset));
            }
            int64_t parsed = 0;
            const char* begin = value.data();
            const char* end = value.data() + value.size();
            auto [ptr, ec] = std::from_chars(begin, end, parsed);
            if (ec != std::errc() || ptr != end) {
                return Status::QueryError("filter '" + std::string(field) +
                                          "' needs an integer timestamp, got '" +
                                          std::string(value) + "'" + At(value_offset));
            }
            if (which == FilterField::kAfter) {
                if (filters.published_after) {
                    return Status::QueryError("duplicate after filter" + At(field_offset));
                }
                filters.published_after = parsed;
            } else {
                if (filters.published_before) {
                    return Status::QueryError("duplicate before filter" + At(field_offset));
                }
                filters.published_before = parsed;
            }
            return Status::Ok();
        }

        case FilterField::kUnknown:
            break;
    }
    return Status::Internal("unhandled filter field");
}

}  // namespace

Result<Query> Parse(std::string_view text, const ParseOptions& options) {
    Query query;
    query.raw = std::string(text);
    query.top_k = options.top_k;
    query.candidate_k = options.candidate_k;
    query.deadline_ms = options.deadline_ms;
    query.mode = options.default_mode;

    // Phrase terms keep stopwords: "to be or not to be" is a phrase whose positions
    // *are* its content, and analysing it with stopword removal leaves an adjacency
    // constraint over the wrong tokens — "not" adjacent to "be" when the document
    // actually reads "not to be". Loose terms use the configured options.
    TokenizerOptions phrase_options = options.tokenizer;
    phrase_options.remove_stopwords = false;
    const Tokenizer loose_tokenizer(options.tokenizer);
    const Tokenizer phrase_tokenizer(phrase_options);

    bool saw_phrase = false;
    size_t phrase_offset = 0;
    bool saw_operator = false;
    QueryMode operator_mode = options.default_mode;
    size_t operator_offset = 0;

    size_t i = 0;
    while (i < text.size()) {
        if (IsSpace(text[i])) {
            ++i;
            continue;
        }

        // --- quoted phrase ---------------------------------------------------
        if (text[i] == '"') {
            const size_t open = i;
            const size_t close = text.find('"', open + 1);
            if (close == std::string_view::npos) {
                return Status::QueryError("unterminated quote opened" + At(open));
            }
            if (saw_phrase) {
                return Status::QueryError(
                    "a second phrase" + At(open) + " conflicts with the phrase" +
                    At(phrase_offset) +
                    ": a Query carries one mode, so two independent adjacency "
                    "constraints are not expressible");
            }
            if (!query.terms.empty()) {
                return Status::QueryError(
                    "phrase" + At(open) +
                    " cannot be combined with loose terms: a Query carries one mode, so "
                    "phrase-AND-term is not expressible");
            }
            query.terms = phrase_tokenizer.Tokenize(text.substr(open + 1, close - open - 1));
            if (query.terms.empty()) {
                return Status::QueryError("phrase" + At(open) + " contains no terms");
            }
            saw_phrase = true;
            phrase_offset = open;
            i = close + 1;
            continue;
        }

        // --- one whitespace-delimited element, stopping at a quote so that
        //     `category:"big cats"` is seen as a filter with a quoted value -----
        const size_t start = i;
        size_t colon = std::string_view::npos;
        while (i < text.size() && !IsSpace(text[i]) && text[i] != '"') {
            if (text[i] == ':' && colon == std::string_view::npos) colon = i;
            ++i;
        }

        if (colon != std::string_view::npos) {
            const std::string_view field = text.substr(start, colon - start);
            if (i < text.size() && text[i] == '"' && i == colon + 1) {
                const size_t open = i;
                const size_t close = text.find('"', open + 1);
                if (close == std::string_view::npos) {
                    return Status::QueryError("unterminated quote opened" + At(open));
                }
                MERIDIAN_TRY(ApplyFilter(query.filters, field,
                                         text.substr(open + 1, close - open - 1), start,
                                         open + 1, /*quoted_value=*/true));
                i = close + 1;
            } else {
                MERIDIAN_TRY(ApplyFilter(query.filters, field,
                                         text.substr(colon + 1, i - colon - 1), start,
                                         colon + 1, /*quoted_value=*/false));
            }
            continue;
        }

        const std::string_view element = text.substr(start, i - start);
        if (element.empty()) {
            ++i;  // a bare quote where a word was expected; skip it as a separator
            continue;
        }

        // --- boolean operator -------------------------------------------------
        if (element == "AND" || element == "OR") {
            const QueryMode mode =
                element == "AND" ? QueryMode::kConjunctive : QueryMode::kDisjunctive;
            if (saw_operator && mode != operator_mode) {
                return Status::QueryError(
                    "mixed boolean operators: '" + std::string(element) + "'" + At(start) +
                    " conflicts with the operator" + At(operator_offset) +
                    "; a Query carries one mode, so this must be split into separate "
                    "queries");
            }
            saw_operator = true;
            operator_mode = mode;
            operator_offset = start;
            continue;
        }

        // --- plain term --------------------------------------------------------
        if (saw_phrase) {
            return Status::QueryError("term '" + std::string(element) + "'" + At(start) +
                                      " cannot be combined with the phrase" +
                                      At(phrase_offset));
        }
        for (std::string& term : loose_tokenizer.Tokenize(element)) {
            query.terms.push_back(std::move(term));
        }
    }

    if (saw_phrase) {
        if (saw_operator) {
            return Status::QueryError("a phrase" + At(phrase_offset) +
                                      " already implies adjacency, so the operator" +
                                      At(operator_offset) + " has no meaning with it");
        }
        query.mode = QueryMode::kPhrase;
    } else if (saw_operator) {
        query.mode = operator_mode;
    }

    if (query.terms.empty() && query.filters.empty()) {
        return Status::QueryError(
            "empty query" + At(0) +
            ": refused rather than treated as match-all, because an accidental submit "
            "would otherwise become a full-corpus scan");
    }
    return query;
}

}  // namespace meridian::query
