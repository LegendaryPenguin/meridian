#include "lexical.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace meridian {

namespace query {

bool PositionsAreAdjacent(const std::vector<std::span<const Position>>& by_term) {
    if (by_term.empty()) return false;
    for (const std::span<const Position>& span : by_term) {
        if (span.empty()) return false;  // a term absent from this field cannot be adjacent
    }
    if (by_term.size() == 1) return true;

    for (Position start : by_term[0]) {
        bool ok = true;
        for (size_t t = 1; t < by_term.size() && ok; ++t) {
            const Position want = start + static_cast<Position>(t);
            // Position lists are ascending, so this is a binary search rather than a
            // scan. For a 10-term phrase against a term occurring 10,000 times in one
            // document that is the difference between 10^5 comparisons and about 10^2.
            ok = std::binary_search(by_term[t].begin(), by_term[t].end(), want);
        }
        if (ok) return true;
    }
    return false;
}

}  // namespace query

namespace {

constexpr Field kAllFields[] = {Field::kTitle, Field::kBody, Field::kTags};

double WeightFor(const FieldWeights& w, Field f) {
    switch (f) {
        case Field::kTitle: return w.title;
        case Field::kBody:  return w.body;
        case Field::kTags:  return w.tags;
        default:            return 0.0;
    }
}

/// One term's postings across all fields, presented as a single ascending cursor.
///
/// A conjunctive query means "every term appears somewhere in the document", not "every
/// term appears in the same field", so the unit the leapfrog iterates over has to be
/// the term, not the (term, field) pair. The per-field breakdown is still needed for
/// field-weighted scoring, which is why this keeps the field cursors rather than
/// collapsing them into a plain PostingIterator.
class TermCursor {
public:
    struct FieldCursor {
        Field field;
        std::unique_ptr<PostingIterator> it;
    };

    TermCursor(std::string term, std::vector<FieldCursor> fields)
        : term_(std::move(term)), fields_(std::move(fields)) {
        // Positioning via advance(0) rather than next().
        //
        // PostingIterator does not say whether a freshly returned iterator sits on its
        // first posting or before it, and the two conventions differ by one document —
        // which silently drops the first match. `advance(0)` is correct under both:
        // "first document >= 0" is the first posting from either starting state. The
        // cost is one extra virtual call per term per segment.
        for (FieldCursor& fc : fields_) fc.it->advance(0);
        Sync();
    }

    const std::string& term() const { return term_; }
    DocId doc() const { return doc_; }

    /// The first failure latched by any of this term's field iterators.
    ///
    /// Must be consulted before a result set is believed complete. `next()` and
    /// `advance()` report a corrupt block by going exhausted, which is indistinguishable
    /// from reaching the end of the list — so a retriever that only watches for
    /// `kNoMoreDocs` turns a checksum failure into a query that quietly returns four of
    /// its nine matches. There is no error anywhere in that outcome, which is what makes
    /// it the worst failure in the system.
    Status status() const {
        for (const FieldCursor& fc : fields_) {
            const Status s = fc.it->status();
            if (!s.ok()) return s;
        }
        return Status::Ok();
    }

    /// Upper bound on the union's length. Used only for ordering, where an upper bound
    /// is the right thing: it over-estimates by the overlap between fields, and the
    /// overlap is largest for exactly the common terms that should sort last anyway.
    uint32_t size_hint() const {
        uint64_t total = 0;
        for (const FieldCursor& fc : fields_) total += fc.it->size_hint();
        return static_cast<uint32_t>(std::min<uint64_t>(total, kNoMoreDocs));
    }

    DocId next() {
        if (doc_ == kNoMoreDocs) return kNoMoreDocs;
        const DocId from = doc_;
        for (FieldCursor& fc : fields_) {
            if (fc.it->doc() == from) fc.it->next();
        }
        Sync();
        return doc_;
    }

    DocId advance(DocId target) {
        if (doc_ >= target) return doc_;
        for (FieldCursor& fc : fields_) {
            if (fc.it->doc() < target) fc.it->advance(target);
        }
        Sync();
        return doc_;
    }

    /// Term frequency in `field` at the current document, or 0 if absent there.
    uint32_t term_frequency(Field field) const {
        for (const FieldCursor& fc : fields_) {
            if (fc.field == field && fc.it->doc() == doc_) return fc.it->term_frequency();
        }
        return 0;
    }

    /// Positions in `field` at the current document. Empty when the term is not in
    /// that field here, or when the field was indexed without positions.
    std::span<const Position> positions(Field field) const {
        for (const FieldCursor& fc : fields_) {
            if (fc.field == field && fc.it->doc() == doc_) return fc.it->positions();
        }
        return {};
    }

    PostingIterator::Stats AccumulatedStats() const {
        PostingIterator::Stats total;
        for (const FieldCursor& fc : fields_) {
            PostingIterator::Stats s = fc.it->stats();
            total.postings_decoded += s.postings_decoded;
            total.skips_taken += s.skips_taken;
            total.blocks_read += s.blocks_read;
        }
        return total;
    }

private:
    void Sync() {
        DocId min_doc = kNoMoreDocs;
        for (const FieldCursor& fc : fields_) min_doc = std::min(min_doc, fc.it->doc());
        doc_ = min_doc;
    }

    std::string term_;
    std::vector<FieldCursor> fields_;
    DocId doc_ = kNoMoreDocs;
};

/// A term's corpus-wide statistics, summed across segments.
///
/// Document frequency is taken as the *maximum* over fields within a segment, not the
/// sum. Summing double-counts every document that has the term in more than one field,
/// which is the normal case — a title word almost always recurs in the body — and the
/// over-count inflates df, which depresses IDF for exactly the terms field weighting
/// exists to promote. Max is a lower bound on the true df and is exact whenever one
/// field dominates, which the body field does. Getting this right matters because IDF
/// is corpus-wide: a per-segment df would make a document's score depend on which
/// segment it happened to land in.
std::unordered_map<std::string, TermStats> GlobalTermStats(
    const Generation& gen, const std::vector<std::string>& terms) {
    std::unordered_map<std::string, TermStats> out;
    for (const std::string& term : terms) {
        TermStats total;
        for (const auto& segment : gen.segments()) {
            uint32_t max_df = 0;
            uint64_t cf = 0;
            for (Field f : kAllFields) {
                std::optional<TermStats> ts = segment->Term(term, f);
                if (!ts) continue;
                max_df = std::max(max_df, ts->document_frequency);
                cf += ts->collection_frequency;
            }
            total.document_frequency += max_df;
            total.collection_frequency += cf;
        }
        out.emplace(term, total);
    }
    return out;
}

std::vector<TermCursor> OpenCursors(const SegmentReader& segment,
                                    const std::vector<std::string>& terms,
                                    bool& all_terms_present) {
    std::vector<TermCursor> cursors;
    cursors.reserve(terms.size());
    all_terms_present = true;
    for (const std::string& term : terms) {
        std::vector<TermCursor::FieldCursor> fields;
        for (Field f : kAllFields) {
            std::unique_ptr<PostingIterator> it = segment.Postings(term, f);
            if (it) fields.push_back({f, std::move(it)});
        }
        if (fields.empty()) {
            all_terms_present = false;
            continue;
        }
        cursors.emplace_back(term, std::move(fields));
    }
    return cursors;
}

/// True when every term occupies consecutive positions within a single field, in query
/// order.
///
/// Checked per field rather than over a merged position list, because a title ending in
/// "machine" followed by a body starting with "learning" is not a phrase match and a
/// merged list cannot tell the difference.
bool PhraseMatches(const std::vector<TermCursor*>& cursors) {
    if (cursors.empty()) return false;
    std::vector<std::span<const Position>> by_term(cursors.size());
    for (Field f : kAllFields) {
        for (size_t t = 0; t < cursors.size(); ++t) by_term[t] = cursors[t]->positions(f);
        if (query::PositionsAreAdjacent(by_term)) return true;
    }
    return false;
}

}  // namespace

LexicalRetriever::LexicalRetriever(Bm25Params params, FieldWeights weights)
    : params_(params), weights_(weights) {}

// Okapi BM25, with the **Lucene/Robertson non-negative IDF variant**:
//
//     idf = ln(1 + (N - df + 0.5) / (df + 0.5))
//
// rather than the textbook probabilistic form ln((N - df + 0.5) / (df + 0.5)).
//
// IDF floor decision: there is no floor, because this variant cannot go negative in
// the first place — the +1 inside the logarithm keeps the argument above 1 for every
// df in [0, N]. That is deliberately chosen over "use the textbook form and clamp at
// zero" for two reasons. First, clamping maps every term appearing in more than half
// the corpus to exactly 0.0, which destroys the ordering *among* those terms: "the"
// and a term in 51% of documents become indistinguishable, when the second is in fact
// mildly informative. Second, an unclamped negative IDF means adding a matching term
// can *lower* a document's score, so a document containing all of the query's words
// can rank below one containing fewer. That breaks the one monotonicity property users
// reliably expect, and it also breaks any future WAND-style upper-bound pruning, which
// needs per-term contributions to be non-negative to be sound.
//
// The cost of the variant is that a term in every document scores ln(1 + 0.5/(N+0.5)),
// a small positive number rather than exactly zero. The tests assert that it is
// positive and below 0.01 for a corpus of 5 documents rather than asserting it is 0.
double LexicalRetriever::Bm25(const CorpusStats& corpus, const TermStats& term, uint32_t tf,
                              uint32_t doc_length) const {
    if (tf == 0) return 0.0;

    const double n = static_cast<double>(corpus.document_count);
    const double df = std::min<double>(term.document_frequency, n);
    const double idf = std::log(1.0 + (n - df + 0.5) / (df + 0.5));

    // A zero average length means an empty or unpopulated corpus; normalising by it
    // would produce a NaN that then poisons every comparison in the sort.
    const double avgdl =
        corpus.average_document_length > 0.0 ? corpus.average_document_length : 1.0;
    const double length_norm =
        1.0 - params_.b + params_.b * (static_cast<double>(doc_length) / avgdl);

    const double f = static_cast<double>(tf);
    return idf * (f * (params_.k1 + 1.0)) / (f + params_.k1 * length_norm);
}

namespace {

/// Everything one segment-level evaluation needs, so the three evaluation strategies
/// can share scoring, filtering and statistics.
struct SegmentContext {
    const SegmentReader* segment = nullptr;
    const CorpusStats* corpus = nullptr;
    const std::unordered_map<std::string, TermStats>* term_stats = nullptr;
    const LexicalRetriever* retriever = nullptr;
    const FieldWeights* weights = nullptr;
    const Filters* filters = nullptr;
    LexicalRetriever::Stats* stats = nullptr;
};

/// Scores one matched document, or reports that it should not be returned at all.
///
/// **Filtering happens here, before scoring, and not after.** The reasoning:
///
///  - Scoring is the expensive half. A filter test is four optional comparisons on a
///    struct; a score is a logarithm and a division per (term, field) pair. Filtering
///    after scoring pays for scores that are then thrown away.
///
///  - More importantly, `candidate_k` is a budget. If filtering happened after
///    retrieval, the budget would be spent on documents that cannot be returned, and a
///    filter rejecting 90% of matches would leave the reranker with 20 candidates
///    instead of 200 — a silent recall collapse that looks like "the filter made the
///    results worse" rather than like a bug.
///
///  - The DocumentMeta fetch is needed regardless, because BM25 needs the document
///    length. So the filter test is genuinely free: it reuses a load that scoring has
///    already paid for. This is the argument that actually settles it.
///
/// Selectivity assumption: filters are assumed *not* to be far more selective than the
/// term match. The postings are still fully driven by the terms, so a filter matching
/// one document in a million still decodes every posting for the query's terms. For
/// that regime the right structure is a filter-driven iterator — a sorted doc list per
/// filter value, leapfrogged against the term cursors — and this implementation
/// deliberately does not have one. The assumption holds for the filters `Filters`
/// actually offers: language and category are low-cardinality, so they select whole
/// corpus fractions rather than handfuls, which is the reasoning `Filters`' own header
/// gives for keeping them out of the term dictionary in the first place.
bool ScoreDocument(const SegmentContext& ctx, DocId local,
                   const std::vector<TermCursor*>& matched, Candidate& out) {
    if (ctx.segment->IsDeleted(local)) return false;

    Result<DocumentMeta> meta = ctx.segment->Document(local);
    if (!meta.ok()) return false;  // a missing meta record cannot be scored or filtered
    const DocumentMeta& doc = meta.value();

    if (!ctx.filters->empty() && !ctx.filters->Matches(doc)) return false;

    double score = 0.0;
    for (const TermCursor* cursor : matched) {
        auto stats_it = ctx.term_stats->find(cursor->term());
        if (stats_it == ctx.term_stats->end()) continue;
        for (Field f : kAllFields) {
            const uint32_t tf = cursor->term_frequency(f);
            if (tf == 0) continue;
            score += WeightFor(*ctx.weights, f) *
                     ctx.retriever->Bm25(*ctx.corpus, stats_it->second, tf, doc.length);
        }
    }

    ++ctx.stats->candidates_scored;
    out.doc = ctx.segment->meta().doc_base + local;
    out.score = score;
    out.bm25 = score;
    out.source = Candidate::Source::kLexical;
    return true;
}

void AccumulateCursorStats(const std::vector<TermCursor>& cursors,
                           LexicalRetriever::Stats& stats) {
    for (const TermCursor& c : cursors) {
        PostingIterator::Stats s = c.AccumulatedStats();
        stats.postings_decoded += s.postings_decoded;
        stats.skips_taken += s.skips_taken;
    }
}

/// OR evaluation: a heap over every (term, field) cursor, keyed by document.
///
/// The heap is over term cursors rather than raw field cursors because scoring needs
/// all of a document's matching terms together; popping one field at a time would
/// require re-finding the others.
void EvaluateDisjunctive(const SegmentContext& ctx, std::vector<TermCursor>& cursors,
                         std::vector<Candidate>& out) {
    // Index into `cursors`, ordered by current document. Indices rather than pointers
    // so the comparator stays trivially copyable and the heap stays in one allocation.
    auto by_doc = [&cursors](size_t a, size_t b) {
        return cursors[a].doc() > cursors[b].doc();  // greater: min-heap
    };
    std::vector<size_t> heap;
    heap.reserve(cursors.size());
    for (size_t i = 0; i < cursors.size(); ++i) {
        if (cursors[i].doc() != kNoMoreDocs) heap.push_back(i);
    }
    std::make_heap(heap.begin(), heap.end(), by_doc);

    std::vector<TermCursor*> matched;
    while (!heap.empty()) {
        const DocId doc = cursors[heap.front()].doc();
        matched.clear();

        // Drain every cursor sitting on this document.
        while (!heap.empty() && cursors[heap.front()].doc() == doc) {
            std::pop_heap(heap.begin(), heap.end(), by_doc);
            matched.push_back(&cursors[heap.back()]);
            heap.pop_back();
        }

        Candidate candidate;
        if (ScoreDocument(ctx, doc, matched, candidate)) out.push_back(candidate);

        for (TermCursor* cursor : matched) {
            if (cursor->next() == kNoMoreDocs) continue;
            heap.push_back(static_cast<size_t>(cursor - cursors.data()));
            std::push_heap(heap.begin(), heap.end(), by_doc);
        }
    }
}

/// AND evaluation by leapfrog, with the iterators pre-ordered by the caller.
///
/// The ordering is the whole point. Element 0 drives the loop: every one of its
/// postings is visited exactly once, and the others are only ever probed with
/// `advance` at those documents, so a skip-list-backed iterator jumps over the ranges
/// in between instead of decoding them. Ordering by `size_hint` ascending therefore
/// makes total cost proportional to the *rarest* term's posting count. Reversing the
/// order makes it proportional to the commonest, which for a real corpus is several
/// orders of magnitude worse — "the AND abyssal" either decodes ten postings or ten
/// million depending only on this sort.
///
/// `in_query_order` is a second view of the same cursors. Adjacency is defined by the
/// order the user typed the terms in, and the cost-based order destroys that, so the
/// phrase check cannot reuse `ordered`. Both vectors point at the same cursors, so the
/// positions they report are the ones the leapfrog has just aligned.
void EvaluateConjunctive(const SegmentContext& ctx, std::vector<TermCursor*>& ordered,
                         const std::vector<TermCursor*>& in_query_order, bool require_phrase,
                         std::vector<Candidate>& out) {
    if (ordered.empty()) return;

    TermCursor* leader = ordered.front();
    DocId candidate_doc = leader->doc();
    while (candidate_doc != kNoMoreDocs) {
        bool aligned = true;
        for (size_t j = 1; j < ordered.size(); ++j) {
            const DocId got = ordered[j]->advance(candidate_doc);
            if (got != candidate_doc) {
                candidate_doc = got;
                aligned = false;
                break;
            }
        }
        if (!aligned) {
            if (candidate_doc == kNoMoreDocs) break;
            candidate_doc = leader->advance(candidate_doc);
            continue;
        }

        if (!require_phrase || PhraseMatches(in_query_order)) {
            Candidate candidate;
            if (ScoreDocument(ctx, candidate_doc, ordered, candidate)) {
                out.push_back(candidate);
            }
        }
        candidate_doc = leader->next();
    }
}

/// A query with filters and no terms: "browse the filtered set".
///
/// Accepted rather than rejected because `language:en category:news` is a legitimate
/// request, and every document scores 0.0 because there is no text evidence to score
/// on — pretending otherwise by inventing a score would make the ordering arbitrary
/// but look deliberate. The caller gets documents in ascending DocId order, which for
/// an append-mostly corpus means oldest first; the reranker's freshness feature is what
/// gives this a useful order.
void EvaluateFilterOnly(const SegmentContext& ctx, uint32_t limit,
                        std::vector<Candidate>& out) {
    const SegmentMeta& meta = ctx.segment->meta();
    const std::vector<TermCursor*> no_terms;
    for (DocId local = 0; local < meta.document_count && out.size() < limit; ++local) {
        Candidate candidate;
        if (ScoreDocument(ctx, local, no_terms, candidate)) out.push_back(candidate);
    }
}

/// Ties broken by ascending DocId so that two runs over the same generation produce the
/// same order. Without a tiebreak, `nth_element`'s unspecified ordering of equal
/// elements leaks into the results and an evaluation harness measures noise.
bool ByScoreThenDoc(const Candidate& a, const Candidate& b) {
    return a.score != b.score ? a.score > b.score : a.doc < b.doc;
}

void TruncateToTopK(std::vector<Candidate>& candidates, uint32_t k) {
    if (candidates.size() > k) {
        // nth_element then sort only the prefix: selecting the top 200 of 100,000
        // matches is linear, where a full sort is n log n for a result nobody reads.
        std::nth_element(candidates.begin(), candidates.begin() + k, candidates.end(),
                         ByScoreThenDoc);
        candidates.resize(k);
    }
    std::sort(candidates.begin(), candidates.end(), ByScoreThenDoc);
}

Result<std::vector<Candidate>> RetrieveImpl(const Generation& gen, const Query& query,
                                            const LexicalRetriever& retriever,
                                            const FieldWeights& weights,
                                            query::ConjunctiveOrder order,
                                            LexicalRetriever::Stats& stats) {
    const uint32_t limit = query.candidate_k == 0 ? 1 : query.candidate_k;
    const std::unordered_map<std::string, TermStats> term_stats =
        GlobalTermStats(gen, query.terms);

    std::vector<Candidate> all;
    for (const std::shared_ptr<SegmentReader>& segment : gen.segments()) {
        if (!segment) continue;
        ++stats.segments_searched;

        SegmentContext ctx;
        ctx.segment = segment.get();
        ctx.corpus = &gen.stats();
        ctx.term_stats = &term_stats;
        ctx.retriever = &retriever;
        ctx.weights = &weights;
        ctx.filters = &query.filters;
        ctx.stats = &stats;

        if (query.terms.empty()) {
            EvaluateFilterOnly(ctx, limit, all);
            continue;
        }

        bool all_terms_present = false;
        std::vector<TermCursor> cursors = OpenCursors(*segment, query.terms, all_terms_present);
        if (cursors.empty()) continue;

        const bool conjunctive = query.mode != QueryMode::kDisjunctive;
        if (conjunctive && !all_terms_present) {
            // A term absent from this segment's dictionary can never be satisfied here.
            // Bailing out before positioning the other cursors is what makes a
            // conjunctive query over a segment that lacks one term nearly free.
            continue;
        }

        if (conjunctive) {
            std::vector<TermCursor*> in_query_order;
            in_query_order.reserve(cursors.size());
            for (TermCursor& c : cursors) in_query_order.push_back(&c);
            std::vector<TermCursor*> ordered = in_query_order;
            std::stable_sort(ordered.begin(), ordered.end(),
                             [order](const TermCursor* a, const TermCursor* b) {
                                 return order == query::ConjunctiveOrder::kRarestFirst
                                            ? a->size_hint() < b->size_hint()
                                            : a->size_hint() > b->size_hint();
                             });
            EvaluateConjunctive(ctx, ordered, in_query_order,
                                query.mode == QueryMode::kPhrase, all);
        } else {
            EvaluateDisjunctive(ctx, cursors, all);
        }

        // Checked after the walk, before these candidates are believed complete. A
        // corrupt block stops an iterator early and looks exactly like the end of a
        // posting list, so without this the query returns a short result set and reports
        // success — the one failure mode with no observable symptom. Failing the whole
        // retrieval is right: a silently incomplete answer to "find me every document
        // matching X" is worse than no answer, because the caller cannot tell.
        for (const TermCursor& cursor : cursors) {
            const Status walk = cursor.status();
            if (!walk.ok()) {
                return walk.WithContext("term '" + cursor.term() + "' in segment " +
                                        std::to_string(segment->meta().id));
            }
        }

        AccumulateCursorStats(cursors, stats);

        // Trimming between segments rather than only at the end bounds peak memory to
        // a few multiples of the candidate budget, which matters for an OR query over a
        // common term where the match set is a large fraction of the corpus.
        if (all.size() > static_cast<size_t>(limit) * 8) TruncateToTopK(all, limit);
    }

    TruncateToTopK(all, limit);
    return all;
}

}  // namespace

// Deadlines are not enforced here. `Query::deadline_ms` is checked by the serving layer,
// which owns the clock and the decision to refuse; a retriever that polled a clock in
// its inner loop would make the posting-decode measurements it exists to produce depend
// on the timer's cost.
Result<std::vector<Candidate>> LexicalRetriever::Retrieve(const Generation& gen,
                                                          const Query& query) const {
    return RetrieveImpl(gen, query, *this, weights_, query::ConjunctiveOrder::kRarestFirst,
                        stats_);
}

namespace query {

Result<std::vector<Candidate>> RetrieveConjunctiveOrdered(const Generation& gen,
                                                          const Query& query,
                                                          Bm25Params params,
                                                          FieldWeights weights,
                                                          ConjunctiveOrder order,
                                                          LexicalRetriever::Stats* stats_out) {
    LexicalRetriever retriever(params, weights);
    LexicalRetriever::Stats local;
    Result<std::vector<Candidate>> result =
        RetrieveImpl(gen, query, retriever, weights, order, local);
    if (stats_out) *stats_out = local;
    return result;
}

}  // namespace query
}  // namespace meridian
