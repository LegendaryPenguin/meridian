// Loading a judged query set from the formats real labelled collections ship in.
//
// Two files, because that is what TREC, MS MARCO and BEIR all do, and because a bespoke
// single-file format would mean an external collection could not be used without being
// converted first — which in practice means it is not used.
//
// Malformed lines are errors carrying their line number, not silent skips. A qrels file
// whose grade column is occasionally a float is a real thing that happens, and skipping
// those lines produces an evaluation that is quietly measuring a subset of the
// judgements. The resulting NDCG looks plausible and is wrong, which is the failure mode
// this project treats as the worst one.

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "meridian/eval.hpp"

namespace meridian {
namespace {

std::string Trim(const std::string& text) {
    size_t begin = 0;
    size_t end = text.size();
    while (begin < end && (text[begin] == ' ' || text[begin] == '\t' || text[begin] == '\r')) {
        ++begin;
    }
    while (end > begin && (text[end - 1] == ' ' || text[end - 1] == '\t' || text[end - 1] == '\r')) {
        --end;
    }
    return text.substr(begin, end - begin);
}

bool IsSkippable(const std::string& line) {
    const std::string trimmed = Trim(line);
    return trimmed.empty() || trimmed[0] == '#';
}

std::vector<std::string> SplitTabs(const std::string& line) {
    std::vector<std::string> fields;
    std::string current;
    for (char c : line) {
        if (c == '\t') {
            fields.push_back(Trim(current));
            current.clear();
        } else if (c != '\r') {
            current.push_back(c);
        }
    }
    fields.push_back(Trim(current));
    return fields;
}

std::vector<std::string> SplitWhitespace(const std::string& line) {
    std::vector<std::string> fields;
    std::istringstream stream(line);
    std::string field;
    while (stream >> field) fields.push_back(field);
    return fields;
}

std::string Where(const std::string& path, size_t line_number) {
    return path + ":" + std::to_string(line_number) + ": ";
}

/// Strict integer parse. `std::stoi` accepts "3abc" and `atoi` accepts "abc" as zero;
/// both would turn a corrupt grade column into a judgement that looks fine.
bool ParseInt(const std::string& text, int* out) {
    if (text.empty()) return false;
    size_t i = 0;
    bool negative = false;
    if (text[0] == '+' || text[0] == '-') {
        negative = text[0] == '-';
        i = 1;
    }
    if (i >= text.size()) return false;
    long value = 0;
    for (; i < text.size(); ++i) {
        if (std::isdigit(static_cast<unsigned char>(text[i])) == 0) return false;
        value = value * 10 + (text[i] - '0');
        if (value > 1'000'000) return false;
    }
    *out = static_cast<int>(negative ? -value : value);
    return true;
}

/// Optional third column of the queries file: `key=value;key=value`.
///
/// Filters are part of a query's identity — the same text with a language filter is a
/// different retrieval problem with different correct answers — so a judged set that
/// cannot express them cannot be used to evaluate filtered search at all.
Status ParseFilters(const std::string& text, const std::string& path, size_t line_number,
                    Filters* out) {
    std::string remaining = text;
    while (!remaining.empty()) {
        const size_t separator = remaining.find(';');
        const std::string clause = Trim(remaining.substr(0, separator));
        remaining = separator == std::string::npos ? std::string() : remaining.substr(separator + 1);
        if (clause.empty()) continue;

        const size_t equals = clause.find('=');
        if (equals == std::string::npos) {
            return Status::InvalidArgument(Where(path, line_number) + "filter clause '" + clause +
                                          "' is not key=value");
        }
        const std::string key = Trim(clause.substr(0, equals));
        const std::string value = Trim(clause.substr(equals + 1));

        if (key == "language") {
            out->language = value;
        } else if (key == "category") {
            out->category = value;
        } else if (key == "published_after" || key == "published_before") {
            int parsed = 0;
            if (!ParseInt(value, &parsed)) {
                return Status::InvalidArgument(Where(path, line_number) + "filter '" + key +
                                               "' needs an integer, got '" + value + "'");
            }
            if (key == "published_after") {
                out->published_after = parsed;
            } else {
                out->published_before = parsed;
            }
        } else {
            return Status::InvalidArgument(Where(path, line_number) + "unknown filter key '" + key +
                                          "'");
        }
    }
    return Status::Ok();
}

}  // namespace

Result<std::vector<JudgedQuery>> LoadJudgements(const std::string& queries_path,
                                               const std::string& qrels_path) {
    std::ifstream queries_file(queries_path);
    if (!queries_file) {
        return Status::IoError("cannot open queries file " + queries_path);
    }

    std::vector<JudgedQuery> queries;
    std::unordered_map<std::string, size_t> by_id;

    std::string line;
    size_t line_number = 0;
    while (std::getline(queries_file, line)) {
        ++line_number;
        if (IsSkippable(line)) continue;

        // Tab-separated `qid <TAB> text [<TAB> filters]`. Tabs rather than spaces because
        // query text contains spaces, and a format that cannot hold a two-word query is
        // not a query format.
        const auto fields = SplitTabs(line);
        if (fields.size() < 2 || fields[0].empty() || fields[1].empty()) {
            return Status::InvalidArgument(Where(queries_path, line_number) +
                                           "expected 'qid<TAB>text[<TAB>filters]', got " +
                                           std::to_string(fields.size()) + " field(s)");
        }
        if (fields.size() > 3) {
            return Status::InvalidArgument(Where(queries_path, line_number) +
                                           "expected at most 3 tab-separated fields, got " +
                                           std::to_string(fields.size()));
        }

        JudgedQuery query;
        query.id = fields[0];
        query.text = fields[1];
        if (fields.size() == 3 && !fields[2].empty()) {
            MERIDIAN_TRY(ParseFilters(fields[2], queries_path, line_number, &query.filters));
        }

        if (by_id.count(query.id) != 0) {
            return Status::InvalidArgument(Where(queries_path, line_number) + "duplicate query id '" +
                                           query.id + "'");
        }
        by_id.emplace(query.id, queries.size());
        queries.push_back(std::move(query));
    }

    if (queries.empty()) {
        return Status::InvalidArgument(queries_path + ": no queries found");
    }

    std::ifstream qrels_file(qrels_path);
    if (!qrels_file) {
        return Status::IoError("cannot open qrels file " + qrels_path);
    }

    line_number = 0;
    while (std::getline(qrels_file, line)) {
        ++line_number;
        if (IsSkippable(line)) continue;

        // TREC qrels: `qid iteration docid grade`, whitespace separated. The three-field
        // form `qid docid grade` is accepted because a good share of published qrels drop
        // the iteration column, and rejecting them would be rejecting the data rather
        // than finding a problem with it.
        const auto fields = SplitWhitespace(line);
        std::string qid;
        std::string docid;
        std::string grade_text;
        if (fields.size() == 4) {
            qid = fields[0];
            docid = fields[2];
            grade_text = fields[3];
        } else if (fields.size() == 3) {
            qid = fields[0];
            docid = fields[1];
            grade_text = fields[2];
        } else {
            return Status::InvalidArgument(Where(qrels_path, line_number) +
                                           "expected 3 or 4 whitespace-separated fields, got " +
                                           std::to_string(fields.size()));
        }

        int grade = 0;
        if (!ParseInt(grade_text, &grade)) {
            return Status::InvalidArgument(Where(qrels_path, line_number) +
                                           "grade '" + grade_text + "' is not an integer");
        }
        if (grade < 0) {
            // Some collections use -1 for "explicitly not relevant, do not pool". Turning
            // that into a negative gain would make DCG non-monotone in the ranking, which
            // breaks the metric rather than the data.
            return Status::InvalidArgument(Where(qrels_path, line_number) +
                                           "negative grade " + grade_text +
                                           " is not supported; map it to 0 before loading");
        }

        auto it = by_id.find(qid);
        if (it == by_id.end()) {
            // Not malformed. Published qrels routinely cover more topics than the query
            // subset being run, and failing here would make every standard collection
            // unusable with any subset of its topics.
            continue;
        }
        queries[it->second].judgements.push_back(Judgement{docid, grade});
    }

    return queries;
}

}  // namespace meridian
