#include "reranker.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <span>
#include <string_view>

#include "../query/lexical.hpp"

namespace meridian::rank {
namespace {

constexpr Field kAllFields[] = {Field::kTitle, Field::kBody, Field::kTags};
constexpr double kSecondsPerDay = 86400.0;

const char* MatchFeatureFor(Field f) {
    switch (f) {
        case Field::kTitle: return features::kTitleMatch;
        case Field::kBody:  return features::kBodyMatch;
        case Field::kTags:  return features::kTagsMatch;
        default:            return features::kBodyMatch;
    }
}

/// Resolves a global DocId to its segment without going through
/// `Generation::Resolve`.
///
/// Done locally because the ranking path needs candidates *grouped* by segment, not
/// resolved one at a time: grouping is what lets the posting cursors be opened once per
/// segment and advanced monotonically. Calling Resolve per candidate would give the same
/// answer and then throw away the structure that makes the pass cheap.
struct SegmentGroup {
    const SegmentReader* segment = nullptr;
    /// Indices into the caller's candidate vector, sorted by ascending local DocId.
    std::vector<size_t> candidates;
};

std::vector<SegmentGroup> GroupBySegment(const Generation& gen,
                                         const std::vector<Candidate>& candidates) {
    std::vector<SegmentGroup> groups;
    groups.reserve(gen.segments().size());
    for (const std::shared_ptr<SegmentReader>& s : gen.segments()) {
        if (s) groups.push_back({s.get(), {}});
    }

    for (size_t i = 0; i < candidates.size(); ++i) {
        const DocId doc = candidates[i].doc;
        for (SegmentGroup& g : groups) {
            const SegmentMeta& meta = g.segment->meta();
            if (doc >= meta.doc_base && doc < meta.doc_base + meta.document_count) {
                g.candidates.push_back(i);
                break;
            }
        }
        // A candidate outside every segment's range belongs to a different generation.
        // Dropped rather than guessed at: scoring it against this generation's segments
        // would silently attribute one document's features to another.
    }

    for (SegmentGroup& g : groups) {
        const DocId base = g.segment->meta().doc_base;
        std::sort(g.candidates.begin(), g.candidates.end(),
                  [&candidates, base](size_t a, size_t b) {
                      return candidates[a].doc - base < candidates[b].doc - base;
                  });
    }
    return groups;
}

/// The cursors for one segment: one per (unique term, field).
///
/// Unique terms, because a phrase query repeats terms ("to be or not to be") and the
/// position list for an occurrence does not depend on which occurrence asked for it.
/// `query_order` maps each query position back to its unique-term slot so adjacency can
/// still be checked in the order the user typed.
struct SegmentCursors {
    std::vector<std::string> unique_terms;
    std::vector<size_t> query_order;
    /// [term][field] -> cursor, null when the term is absent from that field.
    std::vector<std::vector<std::unique_ptr<PostingIterator>>> cursors;

    PostingIterator* At(size_t term, Field f) const {
        return cursors[term][static_cast<size_t>(f)].get();
    }
};

SegmentCursors OpenCursors(const SegmentReader& segment, const std::vector<std::string>& terms) {
    SegmentCursors out;
    for (const std::string& term : terms) {
        auto it = std::find(out.unique_terms.begin(), out.unique_terms.end(), term);
        if (it == out.unique_terms.end()) {
            out.query_order.push_back(out.unique_terms.size());
            out.unique_terms.push_back(term);
        } else {
            out.query_order.push_back(
                static_cast<size_t>(std::distance(out.unique_terms.begin(), it)));
        }
    }

    out.cursors.resize(out.unique_terms.size());
    for (size_t t = 0; t < out.unique_terms.size(); ++t) {
        out.cursors[t].resize(static_cast<size_t>(Field::kFieldCount));
        for (Field f : kAllFields) {
            std::unique_ptr<PostingIterator> it = segment.Postings(out.unique_terms[t], f);
            // advance(0) rather than next(): PostingIterator does not specify whether a
            // fresh iterator sits on its first posting or before it, and "first document
            // >= 0" is the first posting under either convention.
            if (it) it->advance(0);
            out.cursors[t][static_cast<size_t>(f)] = std::move(it);
        }
    }
    return out;
}

double Freshness(const FeatureConfig& config, int64_t published_at, int64_t now) {
    double age_days = static_cast<double>(now - published_at) / kSecondsPerDay;
    if (age_days < 0.0) {
        if (config.clamp_future_to_now) return 1.0;
        age_days = 0.0;
    }
    const double half_life =
        config.freshness_half_life_days > 0.0 ? config.freshness_half_life_days : 1.0;
    return std::exp2(-age_days / half_life);
}

/// Popularity, compressed and bounded.
///
/// Raw popularity is power-law distributed across several decades — a front-page document
/// can have a million times the traffic of a median one. A linear weight on that raw value
/// makes the single most popular document outrank everything regardless of text, which is
/// not a ranking function, it is a redirect. log1p compresses the decades; the x/(1+x)
/// squash bounds the result into [0, 1) so it sits on the same scale as every other
/// feature here and its weight means the same kind of thing theirs do.
double Popularity(double raw) {
    const double x = std::log1p(std::max(0.0, raw));
    return x / (1.0 + x);
}

}  // namespace

std::vector<std::string> FeatureExtractor::Names() {
    return {features::kBm25,        features::kVectorSimilarity, features::kTitleMatch,
            features::kBodyMatch,   features::kTagsMatch,        features::kPhraseMatch,
            features::kFreshness,   features::kPopularity};
}

Result<std::vector<FeatureExtractor::Row>> FeatureExtractor::Extract(
    const Generation& gen, const Query& query, const std::vector<Candidate>& candidates) const {
    const int64_t now =
        config_.now_epoch_seconds != 0
            ? config_.now_epoch_seconds
            : std::chrono::duration_cast<std::chrono::seconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count();

    std::vector<Row> rows(candidates.size());
    std::vector<bool> filled(candidates.size(), false);

    for (SegmentGroup& group : GroupBySegment(gen, candidates)) {
        if (group.candidates.empty()) continue;
        const SegmentReader& segment = *group.segment;
        const DocId base = segment.meta().doc_base;

        SegmentCursors cursors = OpenCursors(segment, query.terms);
        const double term_count = static_cast<double>(std::max<size_t>(1, cursors.unique_terms.size()));

        std::vector<std::span<const Position>> phrase_spans(query.terms.size());

        for (size_t index : group.candidates) {
            const Candidate& candidate = candidates[index];
            const DocId local = candidate.doc - base;

            Result<DocumentMeta> meta = segment.Document(local);
            if (!meta.ok()) continue;  // leaves `filled` false; see the gap check below
            const DocumentMeta& doc = meta.value();

            // Monotone advance. The candidates are in ascending local order, so each
            // cursor moves forward once per candidate at most and never rewinds. This is
            // what makes the whole candidate set one pass per posting list.
            std::vector<double> per_field_hits(static_cast<size_t>(Field::kFieldCount), 0.0);
            for (size_t t = 0; t < cursors.unique_terms.size(); ++t) {
                for (Field f : kAllFields) {
                    PostingIterator* it = cursors.At(t, f);
                    if (it == nullptr) continue;
                    if (it->doc() < local) it->advance(local);
                    if (it->doc() == local) per_field_hits[static_cast<size_t>(f)] += 1.0;
                }
            }

            double phrase_match = 0.0;
            if (!query.terms.empty()) {
                for (Field f : kAllFields) {
                    for (size_t q = 0; q < query.terms.size(); ++q) {
                        PostingIterator* it = cursors.At(cursors.query_order[q], f);
                        phrase_spans[q] = (it != nullptr && it->doc() == local)
                                              ? it->positions()
                                              : std::span<const Position>{};
                    }
                    if (query::PositionsAreAdjacent(phrase_spans)) {
                        phrase_match = 1.0;
                        break;
                    }
                }
            }

            Row& row = rows[index];
            row.doc = candidate.doc;
            row.external_id = doc.external_id;
            row.features[features::kBm25] = candidate.bm25;
            row.features[features::kVectorSimilarity] = candidate.vector_similarity;
            for (Field f : kAllFields) {
                row.features[MatchFeatureFor(f)] =
                    per_field_hits[static_cast<size_t>(f)] / term_count;
            }
            row.features[features::kPhraseMatch] = phrase_match;
            row.features[features::kFreshness] = Freshness(config_, doc.published_at, now);
            row.features[features::kPopularity] = Popularity(doc.popularity);
            filled[index] = true;
        }
    }

    // A candidate whose features could not be computed is dropped rather than ranked on
    // a zero vector. Ranking it on zeros would quietly bury a document that retrieval
    // had decided was good, and "result 7 vanished" is a far easier bug to notice than
    // "result 7 is inexplicably last".
    std::vector<Row> out;
    out.reserve(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
        if (filled[i]) out.push_back(std::move(rows[i]));
    }
    return out;
}

LinearWeights LinearWeights::Default() {
    LinearWeights w;
    w.weights = {
        {features::kBm25, 1.0},
        {features::kVectorSimilarity, 1.0},
        // Title matching is already boosted once inside BM25 by FieldWeights::title, so
        // this weight is the *additional* credit for covering the query in the title at
        // all, independent of how often. Set below the text weights because double-
        // counting the title is how a ranker learns to prefer short titles stuffed with
        // query words.
        {features::kTitleMatch, 0.5},
        {features::kBodyMatch, 0.1},
        {features::kTagsMatch, 0.2},
        // Phrase evidence is strong and cheap: an exact adjacent occurrence is usually a
        // better signal than any amount of term frequency, which is why this is the
        // largest of the bounded features.
        {features::kPhraseMatch, 0.8},
        {features::kFreshness, 0.3},
        {features::kPopularity, 0.2},
    };
    return w;
}

double LinearWeights::Get(const std::string& name) const {
    auto it = weights.find(name);
    return it == weights.end() ? 0.0 : it->second;
}

Result<std::vector<Hit>> LinearReranker::Rank(const Generation& gen, const Query& query,
                                              const std::vector<Candidate>& candidates) const {
    MERIDIAN_ASSIGN(std::vector<FeatureExtractor::Row> rows,
                    extractor_.Extract(gen, query, candidates));

    std::vector<Hit> hits;
    hits.reserve(rows.size());
    for (FeatureExtractor::Row& row : rows) {
        Hit hit;
        hit.doc = row.doc;
        hit.external_id = std::move(row.external_id);
        hit.score = 0.0;
        for (const auto& [name, value] : row.features) {
            const double contribution = weights_.Get(name) * value;
            hit.score += contribution;
            hit.features[kContributionPrefix + name] = contribution;
        }
        for (auto& [name, value] : row.features) {
            hit.features.emplace(name, value);
        }
        hits.push_back(std::move(hit));
    }

    std::sort(hits.begin(), hits.end(), [](const Hit& a, const Hit& b) {
        return a.score != b.score ? a.score > b.score : a.doc < b.doc;
    });

    // Not truncated to `top_k`. The serving layer truncates after materialisation, and
    // the evaluation harness needs the full ordering to compute NDCG over a deeper cut
    // than the user sees.
    return hits;
}

std::vector<std::string> LinearReranker::FeatureNames() const { return FeatureExtractor::Names(); }

}  // namespace meridian::rank
