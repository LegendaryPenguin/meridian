#pragma once

// Corpus file loading, shared by the two CLIs.
//
// The serving headers live under src/serve/ rather than include/, so they are included
// by relative path. include/meridian/ is the engine's published surface and the serving
// layer is not part of it: a tool and a test reach into the implementation deliberately,
// and nothing outside this repository should be able to.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "../src/serve/reference_index.hpp"

namespace meridian::tools {

/// A stand-in for a sentence embedding: a hashed bag of words, L2-normalised.
///
/// It is not semantic and does not pretend to be — two documents about the same topic in
/// different words get unrelated vectors. It exists so the vector path can be exercised
/// and traced end to end without this repository shipping an embedding model. Anything
/// drawing a conclusion about retrieval *quality* from it would be measuring a hash
/// function.
inline std::vector<float> HashedEmbedding(const std::vector<std::string>& tokens, size_t dim) {
    std::vector<float> vector(dim, 0.0f);
    if (dim == 0) return vector;
    for (const auto& token : tokens) {
        const size_t h = std::hash<std::string>{}(token);
        vector[h % dim] += 1.0f;
        // A second, differently-derived slot per token so two tokens that collide in one
        // dimension are still distinguishable in the other.
        vector[(h / dim) % dim] += 0.5f;
    }
    double norm = 0.0;
    for (float v : vector) norm += static_cast<double>(v) * v;
    if (norm > 0.0) {
        const float inverse = static_cast<float>(1.0 / std::sqrt(norm));
        for (float& v : vector) v *= inverse;
    }
    return vector;
}

struct LoadOptions {
    /// Documents per segment. A flush publishes a generation, so this also controls how
    /// many generations a load produces — visible in the tool output because the segment
    /// count is what later query cost depends on.
    uint32_t docs_per_segment = 10'000;

    /// 0 disables embeddings, and therefore the semantic path.
    size_t embedding_dim = 0;
};

struct LoadReport {
    uint64_t documents = 0;
    uint64_t lines_read = 0;
    uint64_t lines_skipped = 0;
    uint64_t bytes_read = 0;
    uint64_t flushes = 0;
    std::string error;

    bool ok() const { return error.empty(); }
};

namespace detail {

inline std::vector<std::string> SplitTabs(const std::string& line) {
    std::vector<std::string> fields;
    std::string current;
    for (char c : line) {
        if (c == '\t') {
            fields.push_back(current);
            current.clear();
        } else {
            current.push_back(c);
        }
    }
    fields.push_back(current);
    return fields;
}

inline int64_t ParseInt(const std::string& text, int64_t fallback) {
    try {
        return text.empty() ? fallback : std::stoll(text);
    } catch (...) {
        return fallback;
    }
}

inline double ParseDouble(const std::string& text, double fallback) {
    try {
        return text.empty() ? fallback : std::stod(text);
    } catch (...) {
        return fallback;
    }
}

}  // namespace detail

/// Reads a corpus file into an index.
///
/// Format, one document per line:
///   id <TAB> title <TAB> body <TAB> tags <TAB> language <TAB> category <TAB> published_at
///   <TAB> popularity
/// Trailing columns may be omitted. A line with no tab at all is taken as a body with a
/// generated id, so a plain text file works without being reformatted first. Blank lines
/// and lines starting with '#' are skipped and counted.
inline LoadReport LoadCorpus(const std::string& path, serve::ReferenceIndex& index,
                             const LoadOptions& options) {
    LoadReport report;
    std::ifstream input(path);
    if (!input) {
        report.error = "cannot open " + path;
        return report;
    }

    std::string line;
    uint32_t buffered = 0;
    while (std::getline(input, line)) {
        ++report.lines_read;
        report.bytes_read += line.size() + 1;
        if (line.empty() || line[0] == '#') {
            ++report.lines_skipped;
            continue;
        }

        const auto fields = detail::SplitTabs(line);
        serve::ReferenceIndex::Document doc;
        std::string title, body, tags;
        if (fields.size() == 1) {
            doc.meta.external_id = "doc-" + std::to_string(report.documents);
            body = fields[0];
        } else {
            doc.meta.external_id = fields[0].empty()
                                       ? "doc-" + std::to_string(report.documents)
                                       : fields[0];
            title = fields.size() > 1 ? fields[1] : "";
            body = fields.size() > 2 ? fields[2] : "";
            tags = fields.size() > 3 ? fields[3] : "";
            doc.meta.language = fields.size() > 4 ? fields[4] : "";
            doc.meta.category = fields.size() > 5 ? fields[5] : "";
            doc.meta.published_at = fields.size() > 6 ? detail::ParseInt(fields[6], 0) : 0;
            doc.meta.popularity = fields.size() > 7 ? detail::ParseDouble(fields[7], 0.0) : 0.0;
        }

        auto title_tokens = serve::Tokenize(title);
        auto body_tokens = serve::Tokenize(body);
        auto tag_tokens = serve::Tokenize(tags);

        if (options.embedding_dim > 0) {
            std::vector<std::string> all = title_tokens;
            all.insert(all.end(), body_tokens.begin(), body_tokens.end());
            all.insert(all.end(), tag_tokens.begin(), tag_tokens.end());
            doc.embedding = HashedEmbedding(all, options.embedding_dim);
        }

        doc.tokens[Field::kTitle] = std::move(title_tokens);
        doc.tokens[Field::kBody] = std::move(body_tokens);
        doc.tokens[Field::kTags] = std::move(tag_tokens);

        index.Add(std::move(doc));
        ++report.documents;
        if (++buffered >= options.docs_per_segment) {
            index.Flush();
            ++report.flushes;
            buffered = 0;
        }
    }

    if (buffered > 0) {
        index.Flush();
        ++report.flushes;
    }
    return report;
}

/// Formats a byte count for a human. Exact numbers are kept alongside, because a size
/// rounded to "1.2 MB" is useless for comparing two runs.
inline std::string HumanBytes(uint64_t bytes) {
    const char* units[] = {"B", "KiB", "MiB", "GiB"};
    double value = static_cast<double>(bytes);
    size_t unit = 0;
    while (value >= 1024.0 && unit + 1 < 4) {
        value /= 1024.0;
        ++unit;
    }
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.2f %s", value, units[unit]);
    return buffer;
}

}  // namespace meridian::tools
