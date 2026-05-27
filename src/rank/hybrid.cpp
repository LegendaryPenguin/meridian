#include <algorithm>
#include <cmath>
#include <cstddef>
#include <unordered_map>
#include <vector>

#include "meridian/retrieval.hpp"

namespace meridian {
namespace {

/// RRF's smoothing constant. 60 is the value from the original paper and it is not
/// arbitrary: it sets how fast the contribution decays with rank. At k=0 the top result
/// contributes infinitely more than the second; at k=60 ranks 1 and 2 differ by about
/// 1.6%, and rank 1 against rank 100 by about 2.6x. That flatness is the point — RRF is
/// for the case where the two paths' scores are not comparable but their *orderings*
/// are both roughly trustworthy.
constexpr double kRrfK = 60.0;

/// What each path contributes for one document.
struct Contribution {
    double lexical = 0.0;
    double semantic = 0.0;
    bool in_lexical = false;
    bool in_semantic = false;
    double bm25 = 0.0;
    double vector_similarity = 0.0;
};

/// Normalised values for one path, plus the value assigned to documents the path never
/// returned.
///
/// `absent` is the decision that most changes the outcome and is easiest to get wrong.
/// It is set to the *minimum* observed normalised value, not to zero-after-centring and
/// not to the mean. The reasoning: a candidate list is a top-k, so a document missing
/// from it did not score "averagely" on that path — it scored below everything in the
/// list, and the list's minimum is the tightest bound available. Using the mean (which
/// is what a z-score of 0.0 means) would reward a document for *not* being retrieved by
/// a path, which is how a hybrid system ends up ranking a semantically irrelevant
/// document above one the vector index actually liked.
struct Normalized {
    std::unordered_map<DocId, double> values;
    double absent = 0.0;
};

Normalized MinMax(const std::vector<Candidate>& list) {
    Normalized out;
    if (list.empty()) return out;

    double lo = list.front().score;
    double hi = list.front().score;
    for (const Candidate& c : list) {
        lo = std::min(lo, c.score);
        hi = std::max(hi, c.score);
    }

    // Failure mode, stated plainly: min-max is outlier-sensitive. One document scoring
    // 500 when the rest score between 2 and 3 compresses that whole tail into
    // [0.000, 0.002], so the path effectively contributes a single bit — "is this the
    // outlier" — and all its remaining discrimination is lost. That is not fixable by
    // tuning; it is what the transform does. kZScoreLinear exists for it.
    const double range = hi - lo;
    for (const Candidate& c : list) {
        // A degenerate range means every score is identical. Mapping them all to 1.0
        // rather than 0.0 keeps "this path retrieved the document" worth something,
        // which is the only information the path has left in that case.
        out.values[c.doc] = range > 0.0 ? (c.score - lo) / range : 1.0;
    }
    out.absent = 0.0;
    return out;
}

Normalized ZScore(const std::vector<Candidate>& list) {
    Normalized out;
    if (list.empty()) return out;

    const double n = static_cast<double>(list.size());
    double sum = 0.0;
    for (const Candidate& c : list) sum += c.score;
    const double mean = sum / n;

    double variance = 0.0;
    for (const Candidate& c : list) {
        const double d = c.score - mean;
        variance += d * d;
    }
    // Population rather than sample standard deviation: the candidate list is the whole
    // thing being normalised, not a sample drawn from something larger, so there is no
    // n-1 correction to make.
    const double stdev = std::sqrt(variance / n);

    // Failure mode: z-scoring assumes the scores have a mean and a spread that mean
    // something, and a 200-item candidate list is a *truncated upper tail* of the score
    // distribution, not a sample of it. BM25 over a top-200 is roughly exponential and
    // strongly right-skewed, so the "mean" sits well above the median and roughly 60%
    // of the list lands below zero. It is still monotone within a path, so the ordering
    // survives — what degrades is the claim that a z of 1.0 means the same thing on both
    // paths, which is exactly the claim the weighted sum relies on.
    double lo = 0.0;
    bool first = true;
    for (const Candidate& c : list) {
        const double z = stdev > 0.0 ? (c.score - mean) / stdev : 0.0;
        out.values[c.doc] = z;
        if (first || z < lo) {
            lo = z;
            first = false;
        }
    }
    out.absent = lo;
    return out;
}

/// Rank positions, 1-based, in the order the path returned them.
///
/// The path's own order is trusted rather than re-sorted by score: a path is entitled to
/// return candidates in an order its score field does not fully explain, and RRF's whole
/// premise is that the ordering is the signal.
std::unordered_map<DocId, size_t> Ranks(const std::vector<Candidate>& list) {
    std::unordered_map<DocId, size_t> out;
    out.reserve(list.size());
    for (size_t i = 0; i < list.size(); ++i) {
        out.emplace(list[i].doc, i + 1);  // first occurrence wins on duplicates
    }
    return out;
}

}  // namespace

HybridMerger::HybridMerger(Strategy strategy, double lexical_weight, double semantic_weight)
    : strategy_(strategy), lexical_weight_(lexical_weight), semantic_weight_(semantic_weight) {}

std::vector<Candidate> HybridMerger::Merge(const std::vector<Candidate>& lexical,
                                           const std::vector<Candidate>& semantic,
                                           uint32_t candidate_k) const {
    std::unordered_map<DocId, Contribution> merged;
    merged.reserve(lexical.size() + semantic.size());

    for (const Candidate& c : lexical) {
        Contribution& m = merged[c.doc];
        m.in_lexical = true;
        // bm25 is carried from whichever path measured it. The lexical path fills
        // Candidate::bm25; if it did not, the score itself is the best available value,
        // because the reranker needs this feature and a silent 0.0 would make a
        // retrieved document look like one that scored nothing.
        m.bm25 = c.bm25 != 0.0 ? c.bm25 : c.score;
    }
    for (const Candidate& c : semantic) {
        Contribution& m = merged[c.doc];
        m.in_semantic = true;
        m.vector_similarity = c.vector_similarity != 0.0 ? c.vector_similarity : c.score;
    }

    switch (strategy_) {
        case Strategy::kReciprocalRankFusion: {
            // Failure mode: RRF discards score magnitude entirely. A document that is
            // the runaway best lexical match, at a score ten times the second place,
            // contributes exactly what a barely-better-than-second match would. So a
            // path that is confident cannot express it, and a path that is uniformly
            // unsure cannot be discounted — two paths agreeing on a bad document beats
            // one path being certain about a good one. That insensitivity is the same
            // property that makes it immune to scale, which is why it is both the safe
            // default and the one that leaves signal on the floor.
            const std::unordered_map<DocId, size_t> lex_ranks = Ranks(lexical);
            const std::unordered_map<DocId, size_t> sem_ranks = Ranks(semantic);
            for (auto& [doc, m] : merged) {
                auto lex = lex_ranks.find(doc);
                auto sem = sem_ranks.find(doc);
                if (lex != lex_ranks.end()) {
                    m.lexical = 1.0 / (kRrfK + static_cast<double>(lex->second));
                }
                if (sem != sem_ranks.end()) {
                    m.semantic = 1.0 / (kRrfK + static_cast<double>(sem->second));
                }
            }
            break;
        }
        case Strategy::kNormalizedLinear:
        case Strategy::kZScoreLinear: {
            const Normalized lex = strategy_ == Strategy::kNormalizedLinear ? MinMax(lexical)
                                                                           : ZScore(lexical);
            const Normalized sem = strategy_ == Strategy::kNormalizedLinear
                                       ? MinMax(semantic)
                                       : ZScore(semantic);
            for (auto& [doc, m] : merged) {
                auto l = lex.values.find(doc);
                auto s = sem.values.find(doc);
                m.lexical = l != lex.values.end() ? l->second : lex.absent;
                m.semantic = s != sem.values.end() ? s->second : sem.absent;
                // An empty path contributes nothing rather than its `absent` value, so
                // a lexical-only query is not penalised on every document equally.
                if (lexical.empty()) m.lexical = 0.0;
                if (semantic.empty()) m.semantic = 0.0;
            }
            break;
        }
    }

    std::vector<Candidate> out;
    out.reserve(merged.size());
    for (const auto& [doc, m] : merged) {
        Candidate c;
        c.doc = doc;
        c.score = lexical_weight_ * m.lexical + semantic_weight_ * m.semantic;
        c.bm25 = m.bm25;
        c.vector_similarity = m.vector_similarity;
        c.source = m.in_lexical && m.in_semantic ? Candidate::Source::kBoth
                   : m.in_semantic               ? Candidate::Source::kSemantic
                                                 : Candidate::Source::kLexical;
        out.push_back(c);
    }

    // Deterministic tiebreak, as in the retriever: iteration order of an unordered_map
    // is not stable across runs, so without this the merged order would vary between
    // processes for equal scores.
    std::sort(out.begin(), out.end(), [](const Candidate& a, const Candidate& b) {
        return a.score != b.score ? a.score > b.score : a.doc < b.doc;
    });
    if (out.size() > candidate_k) out.resize(candidate_k);
    return out;
}

}  // namespace meridian
