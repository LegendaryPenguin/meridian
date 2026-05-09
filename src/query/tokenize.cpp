// Relative include: the build exports only `include/` on the target, so sources inside
// src/ refer to their siblings by path rather than adding a second include root.
#include "tokenize.hpp"

#include <cstdint>

namespace meridian::query {
namespace {

bool IsAsciiAlnum(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
}

bool IsAsciiUpper(unsigned char c) { return c >= 'A' && c <= 'Z'; }

/// Length of the UTF-8 sequence starting at `first`, or 0 if `first` is not a valid
/// lead byte. Overlong forms and the surrogate range are not checked: this is a
/// tokenizer boundary decision, not a validator, and the only thing it has to get
/// right is where one token ends and the next begins.
size_t Utf8SequenceLength(unsigned char first) {
    if ((first & 0xE0) == 0xC0) return 2;
    if ((first & 0xF0) == 0xE0) return 3;
    if ((first & 0xF8) == 0xF0) return 4;
    return 0;
}

bool IsContinuation(unsigned char c) { return (c & 0xC0) == 0x80; }

// ---------------------------------------------------------------------------
// Porter stemmer. See Tokenizer::Stem for the documented scope.
// ---------------------------------------------------------------------------

class Porter {
public:
    explicit Porter(std::string_view word) : w_(word) {}

    std::string Run() {
        Step1a();
        Step1b();
        Step1c();
        Step2();
        Step3();
        Step4();
        Step5a();
        Step5b();
        return w_;
    }

private:
    std::string w_;

    bool Consonant(size_t i) const {
        switch (w_[i]) {
            case 'a': case 'e': case 'i': case 'o': case 'u':
                return false;
            case 'y':
                // 'y' is a consonant only when it follows a vowel, which is what makes
                // "toy" and "happy" stem differently.
                return i == 0 ? true : !Consonant(i - 1);
            default:
                return true;
        }
    }

    /// Porter's measure m over the prefix w_[0, n): the number of VC pairs in
    /// [C](VC)^m[V]. It is the proxy for "is this stem long enough that removing a
    /// suffix still leaves a word".
    int Measure(size_t n) const {
        size_t i = 0;
        int count = 0;
        while (i < n && Consonant(i)) ++i;
        while (i < n) {
            while (i < n && !Consonant(i)) ++i;
            if (i >= n) break;  // trailing V with no following C does not count
            while (i < n && Consonant(i)) ++i;
            ++count;
        }
        return count;
    }

    bool HasVowel(size_t n) const {
        for (size_t i = 0; i < n; ++i) {
            if (!Consonant(i)) return true;
        }
        return false;
    }

    bool DoubleConsonant(size_t n) const {
        return n >= 2 && w_[n - 1] == w_[n - 2] && Consonant(n - 1);
    }

    /// Porter's *o: consonant-vowel-consonant where the final consonant is not w, x or
    /// y. Marks a stem that needs an 'e' restored ("hop" -> "hope").
    bool Cvc(size_t n) const {
        if (n < 3) return false;
        if (!Consonant(n - 1) || Consonant(n - 2) || !Consonant(n - 3)) return false;
        char last = w_[n - 1];
        return last != 'w' && last != 'x' && last != 'y';
    }

    bool Ends(std::string_view suffix, size_t& stem_length) const {
        if (w_.size() < suffix.size()) return false;
        size_t start = w_.size() - suffix.size();
        if (w_.compare(start, suffix.size(), suffix) != 0) return false;
        stem_length = start;
        return true;
    }

    void SetTo(size_t stem_length, std::string_view replacement) {
        w_.resize(stem_length);
        w_.append(replacement);
    }

    struct Rule {
        std::string_view suffix;
        std::string_view replacement;
    };

    /// Applies the first matching rule whose stem has measure > `min_measure`. The
    /// tables are ordered longest-suffix-first so that "ization" wins over "ation" and
    /// "ement" over "ent"; that ordering is the correctness condition for this helper.
    bool ApplyTable(const Rule* rules, size_t count, int min_measure) {
        for (size_t i = 0; i < count; ++i) {
            size_t stem = 0;
            if (!Ends(rules[i].suffix, stem)) continue;
            if (Measure(stem) > min_measure) SetTo(stem, rules[i].replacement);
            return true;  // first match wins whether or not the measure allowed it
        }
        return false;
    }

    void Step1a() {
        size_t stem = 0;
        if (Ends("sses", stem)) {
            w_.resize(stem + 2);
        } else if (Ends("ies", stem)) {
            w_.resize(stem + 1);
        } else if (Ends("ss", stem)) {
            // "ss" is already the normal form; left alone so "class" survives.
        } else if (Ends("s", stem)) {
            w_.resize(stem);
        }
    }

    void Step1b() {
        size_t stem = 0;
        bool removed_vowel_suffix = false;
        if (Ends("eed", stem)) {
            if (Measure(stem) > 0) w_.resize(stem + 2);
        } else if (Ends("ed", stem)) {
            if (HasVowel(stem)) {
                w_.resize(stem);
                removed_vowel_suffix = true;
            }
        } else if (Ends("ing", stem)) {
            if (HasVowel(stem)) {
                w_.resize(stem);
                removed_vowel_suffix = true;
            }
        }
        if (!removed_vowel_suffix) return;

        size_t n = w_.size();
        size_t ignored = 0;
        if (Ends("at", ignored) || Ends("bl", ignored) || Ends("iz", ignored)) {
            w_ += 'e';
        } else if (DoubleConsonant(n)) {
            char last = w_[n - 1];
            if (last != 'l' && last != 's' && last != 'z') w_.pop_back();
        } else if (Measure(n) == 1 && Cvc(n)) {
            w_ += 'e';
        }
    }

    void Step1c() {
        size_t stem = 0;
        if (Ends("y", stem) && HasVowel(stem)) w_[stem] = 'i';
    }

    void Step2() {
        static constexpr Rule kRules[] = {
            {"ational", "ate"},  {"ization", "ize"},  {"iveness", "ive"},
            {"fulness", "ful"},  {"ousness", "ous"},  {"tional", "tion"},
            {"biliti", "ble"},   {"lessli", "less"},  {"entli", "ent"},
            {"ousli", "ous"},    {"ation", "ate"},    {"alism", "al"},
            {"aliti", "al"},     {"iviti", "ive"},    {"enci", "ence"},
            {"anci", "ance"},    {"izer", "ize"},     {"ator", "ate"},
            {"alli", "al"},      {"bli", "ble"},      {"eli", "e"},
            {"logi", "log"},
        };
        ApplyTable(kRules, sizeof(kRules) / sizeof(kRules[0]), 0);
    }

    void Step3() {
        static constexpr Rule kRules[] = {
            {"icate", "ic"}, {"ative", ""}, {"alize", "al"},
            {"iciti", "ic"}, {"ical", "ic"}, {"ness", ""}, {"ful", ""},
        };
        ApplyTable(kRules, sizeof(kRules) / sizeof(kRules[0]), 0);
    }

    void Step4() {
        // "ion" is conditional on the stem ending in s or t, so it is handled outside
        // the table rather than by bending the table's contract.
        size_t stem = 0;
        if (Ends("ion", stem)) {
            if (stem > 0 && (w_[stem - 1] == 's' || w_[stem - 1] == 't') &&
                Measure(stem) > 1) {
                w_.resize(stem);
            }
            return;
        }
        static constexpr Rule kRules[] = {
            {"ement", ""}, {"ance", ""}, {"ence", ""}, {"able", ""}, {"ible", ""},
            {"ment", ""},  {"ant", ""},  {"ent", ""},  {"ate", ""},  {"iti", ""},
            {"ous", ""},   {"ive", ""},  {"ize", ""},  {"ism", ""},  {"al", ""},
            {"er", ""},    {"ic", ""},   {"ou", ""},
        };
        ApplyTable(kRules, sizeof(kRules) / sizeof(kRules[0]), 1);
    }

    void Step5a() {
        size_t stem = 0;
        if (!Ends("e", stem)) return;
        int m = Measure(stem);
        if (m > 1 || (m == 1 && !Cvc(stem))) w_.resize(stem);
    }

    void Step5b() {
        size_t n = w_.size();
        if (n >= 2 && w_[n - 1] == 'l' && DoubleConsonant(n) && Measure(n) > 1) {
            w_.pop_back();
        }
    }
};

bool IsAllAsciiLetters(std::string_view s) {
    for (char c : s) {
        unsigned char u = static_cast<unsigned char>(c);
        if (u < 'a' || u > 'z') return false;
    }
    return !s.empty();
}

}  // namespace

const std::unordered_set<std::string>& Tokenizer::DefaultStopwords() {
    // "s" and "t" are here because apostrophes are token separators, so "don't"
    // analyses to {"don", "t"} and "user's" to {"user", "s"}. The fragments carry no
    // information and would otherwise be real dictionary terms with enormous posting
    // lists.
    static const std::unordered_set<std::string> kStopwords = {
        "a",   "an",   "and",  "are", "as",   "at",  "be",  "but", "by",   "for",
        "if",  "in",   "into", "is",  "it",   "no",  "not", "of",  "on",   "or",
        "s",   "such", "t",    "that", "the", "their", "then", "there", "these",
        "they", "this", "to",  "was", "will", "with",
    };
    return kStopwords;
}

bool Tokenizer::IsStopword(std::string_view token) const {
    const auto& stopwords = DefaultStopwords();
    return stopwords.find(std::string(token)) != stopwords.end();
}

std::string Tokenizer::Stem(std::string_view word) {
    if (word.size() <= 2 || !IsAllAsciiLetters(word)) return std::string(word);
    return Porter(word).Run();
}

std::string Tokenizer::Normalize(std::string_view token) const {
    std::string out;
    out.reserve(token.size());

    size_t i = 0;
    while (i < token.size() && out.size() < options_.max_token_bytes) {
        unsigned char c = static_cast<unsigned char>(token[i]);
        if (c < 0x80) {
            if (!IsAsciiAlnum(c)) {
                ++i;
                continue;  // Normalize is lenient; splitting is Tokenize's job
            }
            out.push_back(options_.lowercase && IsAsciiUpper(c)
                              ? static_cast<char>(c - 'A' + 'a')
                              : static_cast<char>(c));
            ++i;
            continue;
        }
        size_t len = Utf8SequenceLength(c);
        if (len == 0 || i + len > token.size()) {
            ++i;
            continue;
        }
        bool valid = true;
        for (size_t k = 1; k < len; ++k) {
            if (!IsContinuation(static_cast<unsigned char>(token[i + k]))) valid = false;
        }
        if (!valid) {
            ++i;
            continue;
        }
        out.append(token.substr(i, len));
        i += len;
    }

    if (options_.stem) out = Stem(out);
    return out;
}

std::vector<std::string> Tokenizer::Tokenize(std::string_view text) const {
    std::vector<std::string> raw;
    std::string current;

    auto flush = [&] {
        if (current.empty()) return;
        if (current.size() > options_.max_token_bytes) {
            // Truncating mid-sequence would emit invalid UTF-8, so back off to the
            // last lead byte.
            size_t cut = options_.max_token_bytes;
            while (cut > 0 && IsContinuation(static_cast<unsigned char>(current[cut]))) --cut;
            current.resize(cut);
        }
        if (!current.empty()) raw.push_back(current);
        current.clear();
    };

    size_t i = 0;
    while (i < text.size()) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        if (c < 0x80) {
            if (IsAsciiAlnum(c)) {
                current.push_back(options_.lowercase && IsAsciiUpper(c)
                                      ? static_cast<char>(c - 'A' + 'a')
                                      : static_cast<char>(c));
            } else {
                flush();
            }
            ++i;
            continue;
        }

        size_t len = Utf8SequenceLength(c);
        bool valid = len != 0 && i + len <= text.size();
        for (size_t k = 1; valid && k < len; ++k) {
            if (!IsContinuation(static_cast<unsigned char>(text[i + k]))) valid = false;
        }
        if (!valid) {
            flush();
            ++i;  // drop the bad byte rather than letting it into a term
            continue;
        }
        current.append(text.substr(i, len));
        i += len;
    }
    flush();

    std::vector<std::string> out;
    out.reserve(raw.size());
    for (const std::string& token : raw) {
        if (options_.remove_stopwords && IsStopword(token)) continue;
        out.push_back(options_.stem ? Stem(token) : token);
    }

    // The non-empty guarantee documented in the header: a query that is entirely
    // stopwords keeps them, because matching nothing is indistinguishable from a bug.
    if (out.empty() && !raw.empty()) {
        for (const std::string& token : raw) {
            out.push_back(options_.stem ? Stem(token) : token);
        }
    }
    return out;
}

std::vector<std::string> TokenizeForIndex(std::string_view text, const Tokenizer& tokenizer) {
    return tokenizer.Tokenize(text);
}

std::vector<std::string> TokenizeForQuery(std::string_view text, const Tokenizer& tokenizer) {
    return tokenizer.Tokenize(text);
}

}  // namespace meridian::query
