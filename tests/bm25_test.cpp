// BM25, checked against values worked out by hand.
//
// The reason this file is long and arithmetic-heavy: a BM25 implementation tested only
// against its own output agrees with itself about everything, including a transposed k1
// and b, a missing +1 in the IDF, or length normalisation applied in the wrong direction.
// Every number below was computed from the formula independently of the code.
//
// ===========================================================================
// THE HAND-WORKED CORPUS
// ===========================================================================
//
// Five documents, text in the body field only, field weights all 1.0 so the only thing
// under test is BM25 itself.
//
//   d1  "cat cat dog"        length 3
//   d2  "cat dog bird"       length 3
//   d3  "cat fish dog"       length 3
//   d4  "dog dog dog dog"    length 4
//   d5  "cat dog"            length 2
//
//   N = 5,  total tokens = 15,  avgdl = 15 / 5 = 3.0
//
//   df(cat)  = 4   (d1, d2, d3, d5)
//   df(dog)  = 5   (every document)
//   df(fish) = 1   (d3)
//
// --- IDF, the non-negative variant: idf = ln(1 + (N - df + 0.5) / (df + 0.5)) ---
//
//   idf(cat)  = ln(1 + (5 - 4 + 0.5) / 4.5) = ln(1 + 1.5/4.5) = ln(4/3)
//             = 0.287682072451781
//   idf(dog)  = ln(1 + (5 - 5 + 0.5) / 5.5) = ln(1 + 1/11)   = ln(12/11)
//             = 0.087011376989630      <- a term in EVERY document: small, positive
//   idf(fish) = ln(1 + (5 - 1 + 0.5) / 1.5) = ln(1 + 3)      = ln(4)
//             = 1.386294361119891
//
// --- The tf factor: tf * (k1 + 1) / (tf + k1 * (1 - b + b * len/avgdl)) ---
//     with k1 = 1.2, b = 0.75, avgdl = 3
//
//   len = 3:  K = 1.2 * (0.25 + 0.75 * 3/3) = 1.2 * 1.00 = 1.20
//   len = 4:  K = 1.2 * (0.25 + 0.75 * 4/3) = 1.2 * 1.25 = 1.50
//   len = 2:  K = 1.2 * (0.25 + 0.75 * 2/3) = 1.2 * 0.75 = 0.90
//
//   tf=1, len=3:  2.2 / (1 + 1.20) = 2.2 / 2.20 = 1.000000000000000
//   tf=2, len=3:  4.4 / (2 + 1.20) = 4.4 / 3.20 = 1.375000000000000
//   tf=4, len=4:  8.8 / (4 + 1.50) = 8.8 / 5.50 = 1.600000000000000
//   tf=1, len=2:  2.2 / (1 + 0.90) = 2.2 / 1.90 = 1.157894736842105
//
// --- Per (document, term) scores = idf * tf factor ---
//
//   d1.cat  = 0.287682072451781 * 1.375000 = 0.395562849621199
//   d2.cat  = 0.287682072451781 * 1.000000 = 0.287682072451781
//   d3.cat  = 0.287682072451781 * 1.000000 = 0.287682072451781
//   d5.cat  = 0.287682072451781 * 1.157895 = 0.333105557575746
//   d1.dog  = 0.087011376989630 * 1.000000 = 0.087011376989630
//   d2.dog  = 0.087011376989630 * 1.000000 = 0.087011376989630
//   d3.dog  = 0.087011376989630 * 1.000000 = 0.087011376989630
//   d4.dog  = 0.087011376989630 * 1.600000 = 0.139218203183408
//   d5.dog  = 0.087011376989630 * 1.157895 = 0.100750015461677
//   d3.fish = 1.386294361119891 * 1.000000 = 1.386294361119891
//
// --- Query "cat dog", disjunctive, sum over matching terms ---
//
//   d1 = 0.395562849621199 + 0.087011376989630 = 0.482574226610828
//   d5 = 0.333105557575746 + 0.100750015461677 = 0.433855573037423
//   d2 = 0.287682072451781 + 0.087011376989630 = 0.374693449441411
//   d3 = 0.287682072451781 + 0.087011376989630 = 0.374693449441411
//   d4 = 0.0             + 0.139218203183408 = 0.139218203183408
//
//   Expected order: d1, d5, then d2 and d3 tied (broken by DocId), then d4.
//
// The ordering is worth reading as a sanity check on BM25 itself rather than on the
// code: d1 wins on term frequency, d5 beats d2 and d3 on brevity despite a lower tf,
// and d4 loses badly because "dog" is in every document and therefore says nothing.

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "../src/query/lexical.hpp"
#include "../src/query/parse.hpp"
#include "index_fixture.hpp"
#include "meridian/retrieval.hpp"
#include "testing.hpp"

using namespace meridian;                 // NOLINT
using meridian::testfixture::CorpusBuilder;
using meridian::testfixture::Doc;
using meridian::testfixture::Near;

namespace {

constexpr double kIdfCat = 0.287682072451781;
constexpr double kIdfDog = 0.087011376989630;
constexpr double kIdfFish = 1.386294361119891;

FieldWeights UnweightedFields() {
    FieldWeights w;
    w.title = 1.0;
    w.body = 1.0;
    w.tags = 1.0;
    return w;
}

CorpusStats HandCorpus() {
    CorpusStats stats;
    stats.document_count = 5;
    stats.total_tokens = 15;
    stats.average_document_length = 3.0;
    return stats;
}

TermStats WithDf(uint32_t df) {
    TermStats stats;
    stats.document_frequency = df;
    return stats;
}

std::shared_ptr<Generation> HandBuiltGeneration() {
    CorpusBuilder builder;
    builder.Add({"d1", "", "cat cat dog", "", "en", "general", 0, 0.0, false});
    builder.Add({"d2", "", "cat dog bird", "", "en", "general", 0, 0.0, false});
    builder.Add({"d3", "", "cat fish dog", "", "en", "general", 0, 0.0, false});
    builder.Add({"d4", "", "dog dog dog dog", "", "en", "general", 0, 0.0, false});
    builder.Add({"d5", "", "cat dog", "", "en", "general", 0, 0.0, false});
    return builder.Build();
}

void CheckNear(double got, double want, double tolerance, const char* what, const char* file,
               int line) {
    if (!Near(got, want, tolerance)) {
        meridian::testing::ReportFailure(file, line,
                                        std::string(what) + ": got " + std::to_string(got) +
                                            ", hand-computed " + std::to_string(want));
    }
}

#define CHECK_NEAR(got, want, what) CheckNear((got), (want), 1e-12, (what), __FILE__, __LINE__)

}  // namespace

MERIDIAN_TEST(Bm25MatchesHandComputedScores) {
    const LexicalRetriever retriever({}, UnweightedFields());
    const CorpusStats corpus = HandCorpus();

    CHECK_NEAR(retriever.Bm25(corpus, WithDf(4), 2, 3), 0.395562849621199, "d1.cat");
    CHECK_NEAR(retriever.Bm25(corpus, WithDf(4), 1, 3), 0.287682072451781, "d2.cat");
    CHECK_NEAR(retriever.Bm25(corpus, WithDf(4), 1, 2), 0.333105557575746, "d5.cat");
    CHECK_NEAR(retriever.Bm25(corpus, WithDf(5), 1, 3), 0.087011376989630, "d1.dog");
    CHECK_NEAR(retriever.Bm25(corpus, WithDf(5), 4, 4), 0.139218203183408, "d4.dog");
    CHECK_NEAR(retriever.Bm25(corpus, WithDf(5), 1, 2), 0.100750015461677, "d5.dog");
    CHECK_NEAR(retriever.Bm25(corpus, WithDf(1), 1, 3), 1.386294361119891, "d3.fish");

    // A term the document does not contain contributes nothing, not a negative.
    CHECK_NEAR(retriever.Bm25(corpus, WithDf(1), 0, 3), 0.0, "tf=0");
}

MERIDIAN_TEST(Bm25ScoreRisesWithTermFrequencyButSaturates) {
    const LexicalRetriever retriever({}, UnweightedFields());
    const CorpusStats corpus = HandCorpus();

    const double tf1 = retriever.Bm25(corpus, WithDf(4), 1, 3);
    const double tf2 = retriever.Bm25(corpus, WithDf(4), 2, 3);
    const double tf3 = retriever.Bm25(corpus, WithDf(4), 3, 3);
    const double tf10 = retriever.Bm25(corpus, WithDf(4), 10, 3);
    const double tf1000 = retriever.Bm25(corpus, WithDf(4), 1000, 3);

    CHECK(tf1 < tf2);
    CHECK(tf2 < tf3);
    CHECK(tf3 < tf10);

    // Saturation: doubling tf gains far less than doubling the score, and the increments
    // shrink. This is k1's entire job — the fifth "cat" says much less than the second.
    CHECK(tf2 < 2.0 * tf1);
    CHECK((tf3 - tf2) < (tf2 - tf1));
    CHECK((tf10 - tf3) < (tf3 - tf1));

    // The asymptote is idf * (k1 + 1) = idf * 2.2, which tf=1000 is already close to and
    // can never exceed.
    const double ceiling = kIdfCat * 2.2;
    CHECK(tf1000 < ceiling);
    CHECK(tf1000 > 0.99 * ceiling);
    CHECK(tf10 < ceiling);
}

MERIDIAN_TEST(Bm25PenalizesLongerDocumentsForTheSameTermFrequency) {
    const LexicalRetriever retriever({}, UnweightedFields());
    const CorpusStats corpus = HandCorpus();

    const double short_doc = retriever.Bm25(corpus, WithDf(4), 1, 2);
    const double average = retriever.Bm25(corpus, WithDf(4), 1, 3);
    const double long_doc = retriever.Bm25(corpus, WithDf(4), 1, 6);
    const double huge = retriever.Bm25(corpus, WithDf(4), 1, 60);

    CHECK(short_doc > average);
    CHECK(average > long_doc);
    CHECK(long_doc > huge);

    // Hand-computed: tf=1, len=6 gives 2.2 / (1 + 1.2 * (0.25 + 0.75 * 2)) = 2.2/3.1,
    // so the factor is 0.709677419354839 and the score is idf * that.
    CHECK_NEAR(long_doc, kIdfCat * (2.2 / 3.1), "tf=1, len=6");

    // With b = 0 length normalisation is switched off entirely and the two lengths score
    // identically. Asserted because getting b's direction backwards still produces a
    // monotone function of length, just the wrong one.
    Bm25Params no_length_norm;
    no_length_norm.b = 0.0;
    const LexicalRetriever unnormalized(no_length_norm, UnweightedFields());
    CHECK_NEAR(unnormalized.Bm25(corpus, WithDf(4), 1, 2),
               unnormalized.Bm25(corpus, WithDf(4), 1, 60), "b=0 ignores length");
}

MERIDIAN_TEST(IdfOfAUbiquitousTermIsTinyAndPositive) {
    const LexicalRetriever retriever({}, UnweightedFields());
    const CorpusStats corpus = HandCorpus();

    // The decision this asserts: the non-negative Lucene variant is used and nothing is
    // floored, because the variant cannot go negative. A term in all 5 of 5 documents
    // contributes ln(12/11) = 0.087, not 0.0 and not a negative number.
    //
    // Measured at tf=1, len=avgdl so the tf factor is exactly 1.0 and the score *is* the
    // IDF.
    const double ubiquitous = retriever.Bm25(corpus, WithDf(5), 1, 3);
    CHECK(ubiquitous > 0.0);
    CHECK_NEAR(ubiquitous, kIdfDog, "idf of a term in every document");
    CHECK(ubiquitous < 0.09);

    // And it is nearly worthless next to a rare term: 16x less.
    const double rare = retriever.Bm25(corpus, WithDf(1), 1, 3);
    CHECK_NEAR(rare, kIdfFish, "idf of a term in one document");
    CHECK((rare / ubiquitous) > 10.0);

    // Monotone decreasing in df over the whole range, with no flat region. A floored
    // textbook variant would make every df > N/2 identical, which is what this rules
    // out: df=3, 4 and 5 all differ here.
    double previous = retriever.Bm25(corpus, WithDf(0), 1, 3);
    for (uint32_t df = 1; df <= 5; ++df) {
        const double current = retriever.Bm25(corpus, WithDf(df), 1, 3);
        CHECK(current < previous);
        CHECK(current > 0.0);
        previous = current;
    }
}

MERIDIAN_TEST(DisjunctiveRetrievalReproducesTheHandComputedRanking) {
    std::shared_ptr<Generation> gen = HandBuiltGeneration();
    CHECK_EQ(gen->stats().document_count, uint32_t{5});
    CHECK_EQ(gen->stats().total_tokens, uint64_t{15});
    CHECK_NEAR(gen->stats().average_document_length, 3.0, "avgdl");

    Result<Query> parsed = query::Parse("cat dog");
    CHECK(parsed.ok());

    const LexicalRetriever retriever({}, UnweightedFields());
    Result<std::vector<Candidate>> result = retriever.Retrieve(*gen, parsed.value());
    CHECK(result.ok());
    const std::vector<Candidate>& hits = result.value();
    CHECK_EQ(hits.size(), size_t{5});
    if (hits.size() != 5) return;

    // Global DocIds; the single segment has doc_base 0, so these are 0..4 for d1..d5.
    CHECK_EQ(hits[0].doc, DocId{0});
    CHECK_NEAR(hits[0].score, 0.482574226610828, "d1 total");
    CHECK_EQ(hits[1].doc, DocId{4});
    CHECK_NEAR(hits[1].score, 0.433855573037423, "d5 total");
    CHECK_EQ(hits[2].doc, DocId{1});
    CHECK_NEAR(hits[2].score, 0.374693449441411, "d2 total");
    CHECK_EQ(hits[3].doc, DocId{2});
    CHECK_NEAR(hits[3].score, 0.374693449441411, "d3 total");
    CHECK_EQ(hits[4].doc, DocId{3});
    CHECK_NEAR(hits[4].score, 0.139218203183408, "d4 total");

    // Candidate::bm25 carries the same value, because the reranker reads that field and
    // a zero there would make every retrieved document look textually irrelevant.
    CHECK_NEAR(hits[0].bm25, hits[0].score, "bm25 field mirrors score");
    CHECK(hits[0].source == Candidate::Source::kLexical);

    const LexicalRetriever::Stats stats = retriever.stats();
    CHECK_EQ(stats.segments_searched, uint64_t{1});
    CHECK_EQ(stats.candidates_scored, uint64_t{5});
    CHECK(stats.postings_decoded >= 9);  // 4 "cat" postings + 5 "dog" postings
}

MERIDIAN_TEST(FieldWeightsMultiplyTheContribution) {
    // Two documents of identical length containing the same term, one in the title and
    // one in the body. With title weight 3 and body weight 1, the title document must
    // score exactly 3x.
    CorpusBuilder builder;
    builder.Add({"title_doc", "zebra alpha beta", "", "", "en", "general", 0, 0.0, false});
    builder.Add({"body_doc", "", "zebra alpha beta", "", "en", "general", 0, 0.0, false});
    std::shared_ptr<Generation> gen = builder.Build();

    FieldWeights weights;  // defaults: title 3.0, body 1.0, tags 1.5
    const LexicalRetriever retriever({}, weights);
    Result<Query> parsed = query::Parse("zebra");
    CHECK(parsed.ok());

    Result<std::vector<Candidate>> result = retriever.Retrieve(*gen, parsed.value());
    CHECK(result.ok());
    CHECK_EQ(result.value().size(), size_t{2});
    if (result.value().size() != 2) return;

    CHECK_EQ(result.value()[0].doc, DocId{0});  // the title document
    CHECK_NEAR(result.value()[0].score, 3.0 * result.value()[1].score, "title weight is 3x");
}

MERIDIAN_TEST(SegmentsAreSearchedAndDocIdsAreGlobal) {
    // Three segments. DocIds inside a segment start at zero, so a retriever that forgot
    // doc_base would return document 0 three times and look like it had found three
    // matches in one segment.
    CorpusBuilder builder;
    builder.Add({"s0d0", "", "quartz", "", "en", "general", 0, 0.0, false});
    builder.Add({"s0d1", "", "filler text", "", "en", "general", 0, 0.0, false});
    builder.NewSegment();
    builder.Add({"s1d0", "", "quartz quartz", "", "en", "general", 0, 0.0, false});
    builder.NewSegment();
    builder.Add({"s2d0", "", "unrelated", "", "en", "general", 0, 0.0, false});
    builder.Add({"s2d1", "", "quartz", "", "en", "general", 0, 0.0, false});
    std::shared_ptr<Generation> gen = builder.Build();

    const LexicalRetriever retriever({}, UnweightedFields());
    Result<Query> parsed = query::Parse("quartz");
    CHECK(parsed.ok());
    Result<std::vector<Candidate>> result = retriever.Retrieve(*gen, parsed.value());
    CHECK(result.ok());
    CHECK_EQ(result.value().size(), size_t{3});
    CHECK_EQ(retriever.stats().segments_searched, uint64_t{3});

    std::vector<DocId> docs;
    for (const Candidate& c : result.value()) docs.push_back(c.doc);
    std::sort(docs.begin(), docs.end());
    CHECK_EQ(docs[0], DocId{0});  // segment 0, local 0
    CHECK_EQ(docs[1], DocId{2});  // segment 1, local 0, doc_base 2
    CHECK_EQ(docs[2], DocId{4});  // segment 2, local 1, doc_base 3

    // df is corpus-wide: "quartz" is in 3 of 5 documents spread across three segments, so
    // a per-segment df of 1 would give it a much higher IDF and the same document would
    // score differently depending only on which segment it landed in.
    const CorpusStats& corpus = gen->stats();
    CHECK_EQ(corpus.document_count, uint32_t{5});
    CHECK_NEAR(corpus.average_document_length, 7.0 / 5.0, "avgdl");

    const double with_global_df = retriever.Bm25(corpus, WithDf(3), 2, 2);
    const double with_segment_df = retriever.Bm25(corpus, WithDf(1), 2, 2);
    CHECK(with_global_df < with_segment_df);
    bool checked = false;
    for (const Candidate& c : result.value()) {
        if (c.doc != DocId{2}) continue;  // segment 1's only document, tf 2, length 2
        CHECK_NEAR(c.score, with_global_df, "scoring uses the corpus-wide df");
        checked = true;
    }
    CHECK(checked);
}

MERIDIAN_TEST(DeletedDocumentsAreNotReturned) {
    CorpusBuilder builder;
    builder.Add({"live", "", "obsidian", "", "en", "general", 0, 0.0, false});
    builder.Add({"tombstoned", "", "obsidian", "", "en", "general", 0, 0.0, true});
    std::shared_ptr<Generation> gen = builder.Build();

    const LexicalRetriever retriever({}, UnweightedFields());
    Result<Query> parsed = query::Parse("obsidian");
    CHECK(parsed.ok());
    Result<std::vector<Candidate>> result = retriever.Retrieve(*gen, parsed.value());
    CHECK(result.ok());
    CHECK_EQ(result.value().size(), size_t{1});
    if (!result.value().empty()) CHECK_EQ(result.value()[0].doc, DocId{0});
}

MERIDIAN_TEST(CandidateBudgetIsRespected) {
    CorpusBuilder builder;
    for (int i = 0; i < 50; ++i) {
        builder.Add({"d" + std::to_string(i), "", "granite", "", "en", "general", 0, 0.0, false});
    }
    std::shared_ptr<Generation> gen = builder.Build();

    query::ParseOptions options;
    options.candidate_k = 7;
    Result<Query> parsed = query::Parse("granite", options);
    CHECK(parsed.ok());

    const LexicalRetriever retriever({}, UnweightedFields());
    Result<std::vector<Candidate>> result = retriever.Retrieve(*gen, parsed.value());
    CHECK(result.ok());
    CHECK_EQ(result.value().size(), size_t{7});
}

// ===========================================================================
// Conjunctive evaluation: the iterator ordering claim
// ===========================================================================

MERIDIAN_TEST(ConjunctiveOrderingDecodesFarFewerPostings) {
    // 200 documents. "common" is in every one; "rare" is in three. A conjunctive query
    // over both can cost either three postings' worth of work or two hundred, and the
    // only difference is which iterator drives the loop.
    CorpusBuilder builder;
    for (int i = 0; i < 200; ++i) {
        const bool rare = (i == 7 || i == 100 || i == 193);
        builder.Add({"d" + std::to_string(i), "",
                     rare ? "common rare" : "common", "", "en", "general", 0, 0.0, false});
    }
    std::shared_ptr<Generation> gen = builder.Build();

    Result<Query> parsed = query::Parse("rare AND common");
    CHECK(parsed.ok());
    if (!parsed.ok()) return;
    CHECK(parsed.value().mode == QueryMode::kConjunctive);
    // Proof the shared analysis produced the terms the index holds; if it had not, both
    // orders would return zero documents and the ratio below would be meaningless.
    CHECK_EQ(parsed.value().terms.size(), size_t{2});

    LexicalRetriever::Stats rarest_stats;
    LexicalRetriever::Stats commonest_stats;
    Result<std::vector<Candidate>> rarest = query::RetrieveConjunctiveOrdered(
        *gen, parsed.value(), {}, UnweightedFields(), query::ConjunctiveOrder::kRarestFirst,
        &rarest_stats);
    Result<std::vector<Candidate>> commonest = query::RetrieveConjunctiveOrdered(
        *gen, parsed.value(), {}, UnweightedFields(),
        query::ConjunctiveOrder::kCommonestFirst, &commonest_stats);
    CHECK(rarest.ok());
    CHECK(commonest.ok());
    if (!rarest.ok() || !commonest.ok()) return;

    // Identical answers. The ordering is an optimisation and must not be observable in
    // the results, which is the property that makes it safe to apply unconditionally.
    CHECK_EQ(rarest.value().size(), size_t{3});
    CHECK_EQ(rarest.value().size(), commonest.value().size());
    for (size_t i = 0; i < rarest.value().size() && i < commonest.value().size(); ++i) {
        CHECK_EQ(rarest.value()[i].doc, commonest.value()[i].doc);
        CHECK(Near(rarest.value()[i].score, commonest.value()[i].score));
    }
    std::vector<DocId> docs;
    for (const Candidate& c : rarest.value()) docs.push_back(c.doc);
    std::sort(docs.begin(), docs.end());
    CHECK_EQ(docs[0], DocId{7});
    CHECK_EQ(docs[1], DocId{100});
    CHECK_EQ(docs[2], DocId{193});

    // Strictly fewer postings decoded, rarest-first.
    CHECK(rarest_stats.postings_decoded < commonest_stats.postings_decoded);
    CHECK_EQ(rarest_stats.candidates_scored, commonest_stats.candidates_scored);
    std::printf("        leapfrog: rarest-first %llu postings, commonest-first %llu "
                "(ratio %.2fx), skips %llu vs %llu\n",
                static_cast<unsigned long long>(rarest_stats.postings_decoded),
                static_cast<unsigned long long>(commonest_stats.postings_decoded),
                static_cast<double>(commonest_stats.postings_decoded) /
                    static_cast<double>(rarest_stats.postings_decoded),
                static_cast<unsigned long long>(rarest_stats.skips_taken),
                static_cast<unsigned long long>(commonest_stats.skips_taken));

    // The margin above is a couple of percent, and that number deserves an explanation
    // rather than a victory lap, because the usual claim for this optimisation is orders
    // of magnitude.
    //
    // The reason it is small *here* is that the implementation does not merely order its
    // iterators — after a misalignment it re-advances the leader to the follower's
    // document, so every cursor only ever moves forward towards a target that some
    // cursor has already reached. Total movement is then bounded by the document ids
    // actually visited, which is a property of the posting lists and not of the order
    // they are listed in. The ordering still wins, by the one `next()` per match that
    // lands on the leader's list and by failing fast on the rarest term, but it wins by
    // percent and not by multiples.
    //
    // Orders of magnitude are what separates both of these from the formulation the rule
    // is normally stated about: scan the leader's postings and probe the others at each
    // one, without re-advancing the leader. That version's cost *is* the leader's posting
    // count, so the order is everything. Measured below against the same corpus and the
    // same counters, because a claim about the size of a win should be measured and not
    // asserted.
    const SegmentReader& segment = *gen->segments().front();
    auto naive_cost = [&segment](const std::string& leader_term,
                                 const std::string& follower_term) {
        std::unique_ptr<PostingIterator> leader = segment.Postings(leader_term, Field::kBody);
        std::unique_ptr<PostingIterator> follower =
            segment.Postings(follower_term, Field::kBody);
        if (!leader || !follower) return uint64_t{0};
        uint64_t matches = 0;
        for (DocId doc = leader->doc(); doc != kNoMoreDocs; doc = leader->next()) {
            if (follower->doc() < doc) follower->advance(doc);
            if (follower->doc() == doc) ++matches;
        }
        CHECK_EQ(matches, uint64_t{3});
        return leader->stats().postings_decoded + follower->stats().postings_decoded;
    };
    // Analysed forms, taken from the parsed query rather than retyped, so the baseline
    // cannot accidentally probe a term the index does not hold.
    const std::string& rare_term = parsed.value().terms[0];
    const std::string& common_term = parsed.value().terms[1];
    const uint64_t naive_rarest = naive_cost(rare_term, common_term);
    const uint64_t naive_commonest = naive_cost(common_term, rare_term);
    CHECK(naive_rarest < naive_commonest);
    std::printf("        leader-scan: rarest-first %llu postings, commonest-first %llu "
                "(ratio %.2fx)\n",
                static_cast<unsigned long long>(naive_rarest),
                static_cast<unsigned long long>(naive_commonest),
                static_cast<double>(naive_commonest) / static_cast<double>(naive_rarest));
    // The ratio there tracks the ratio of the two posting lists' lengths, so on a real
    // corpus where "the" has 10^7 postings and the rare term has 10, it is not 5x.
    CHECK((static_cast<double>(naive_commonest) / static_cast<double>(naive_rarest)) > 3.0);
}

MERIDIAN_TEST(ConjunctiveRequiresEveryTerm) {
    CorpusBuilder builder;
    builder.Add({"both", "", "copper wire", "", "en", "general", 0, 0.0, false});
    builder.Add({"copper_only", "", "copper pipe", "", "en", "general", 0, 0.0, false});
    builder.Add({"wire_only", "", "steel wire", "", "en", "general", 0, 0.0, false});
    std::shared_ptr<Generation> gen = builder.Build();

    const LexicalRetriever retriever({}, UnweightedFields());
    Result<Query> conj = query::Parse("copper AND wire");
    CHECK(conj.ok());
    Result<std::vector<Candidate>> result = retriever.Retrieve(*gen, conj.value());
    CHECK(result.ok());
    CHECK_EQ(result.value().size(), size_t{1});
    if (!result.value().empty()) CHECK_EQ(result.value()[0].doc, DocId{0});

    Result<Query> disj = query::Parse("copper OR wire");
    CHECK(disj.ok());
    Result<std::vector<Candidate>> all = retriever.Retrieve(*gen, disj.value());
    CHECK(all.ok());
    CHECK_EQ(all.value().size(), size_t{3});
}

MERIDIAN_TEST(ConjunctiveSkipsSegmentsMissingATerm) {
    CorpusBuilder builder;
    builder.Add({"s0", "", "tungsten filament", "", "en", "general", 0, 0.0, false});
    builder.NewSegment();
    builder.Add({"s1", "", "tungsten only", "", "en", "general", 0, 0.0, false});
    std::shared_ptr<Generation> gen = builder.Build();

    const LexicalRetriever retriever({}, UnweightedFields());
    Result<Query> parsed = query::Parse("tungsten AND filament");
    CHECK(parsed.ok());
    Result<std::vector<Candidate>> result = retriever.Retrieve(*gen, parsed.value());
    CHECK(result.ok());
    CHECK_EQ(result.value().size(), size_t{1});
    // Both segments were visited, but only one could possibly match, and the one that
    // could not was abandoned before any posting was walked.
    CHECK_EQ(retriever.stats().segments_searched, uint64_t{2});
    CHECK_EQ(retriever.stats().candidates_scored, uint64_t{1});
}

// ===========================================================================
// Phrase matching
// ===========================================================================

MERIDIAN_TEST(PhraseQueriesRequireAdjacency) {
    CorpusBuilder builder;
    builder.Add({"adjacent", "", "the field of machine learning is broad", "", "en",
                 "general", 0, 0.0, false});
    builder.Add({"separated", "", "machine tooling and statistical learning", "", "en",
                 "general", 0, 0.0, false});
    builder.Add({"reversed", "", "learning machine", "", "en", "general", 0, 0.0, false});
    builder.Add({"only_one", "", "machine shop", "", "en", "general", 0, 0.0, false});
    std::shared_ptr<Generation> gen = builder.Build();

    const LexicalRetriever retriever({}, UnweightedFields());
    Result<Query> parsed = query::Parse("\"machine learning\"");
    CHECK(parsed.ok());
    if (!parsed.ok()) return;
    CHECK(parsed.value().mode == QueryMode::kPhrase);

    Result<std::vector<Candidate>> result = retriever.Retrieve(*gen, parsed.value());
    CHECK(result.ok());
    if (!result.ok()) return;
    // Only the adjacent occurrence matches. "separated" contains both terms but not next
    // to each other, and "reversed" contains them adjacent in the wrong order — both are
    // conjunctive matches and neither is a phrase match, which is exactly the distinction
    // positions exist to make.
    CHECK_EQ(result.value().size(), size_t{1});
    if (!result.value().empty()) CHECK_EQ(result.value()[0].doc, DocId{0});

    // The same terms as a conjunctive query match three of the four documents, which
    // confirms the phrase result is not just the intersection being empty.
    Result<Query> conjunctive = query::Parse("machine AND learning");
    CHECK(conjunctive.ok());
    Result<std::vector<Candidate>> intersection = retriever.Retrieve(*gen, conjunctive.value());
    CHECK(intersection.ok());
    CHECK_EQ(intersection.value().size(), size_t{3});
}

MERIDIAN_TEST(PhrasesDoNotSpanFieldBoundaries) {
    // A title ending in "machine" and a body starting with "learning" is not a phrase
    // match, however tempting a merged position list would make it look.
    CorpusBuilder builder;
    builder.Add({"split", "an industrial machine", "learning takes time", "", "en",
                 "general", 0, 0.0, false});
    builder.Add({"in_title", "machine learning weekly", "unrelated body", "", "en",
                 "general", 0, 0.0, false});
    std::shared_ptr<Generation> gen = builder.Build();

    const LexicalRetriever retriever({}, UnweightedFields());
    Result<Query> parsed = query::Parse("\"machine learning\"");
    CHECK(parsed.ok());
    Result<std::vector<Candidate>> result = retriever.Retrieve(*gen, parsed.value());
    CHECK(result.ok());
    CHECK_EQ(result.value().size(), size_t{1});
    if (!result.value().empty()) CHECK_EQ(result.value()[0].doc, DocId{1});
}

MERIDIAN_TEST(RepeatedTermsInAPhraseAreHandled) {
    // Indexed with stopwords retained, which is a real precondition rather than a test
    // convenience: a phrase query keeps its stopwords (otherwise the adjacency constraint
    // is over the wrong tokens), so an index that dropped them could not answer this
    // query at all. The two decisions have to agree, and this is the shape of that
    // agreement — the stopword list is an index-time commitment about which phrases are
    // answerable.
    meridian::query::TokenizerOptions keep_stopwords;
    keep_stopwords.remove_stopwords = false;
    CorpusBuilder builder(keep_stopwords);
    builder.Add({"match", "", "that is to be or not to be the question", "", "en",
                 "general", 0, 0.0, false});
    builder.Add({"no_match", "", "to be is not the same as be to", "", "en", "general", 0,
                 0.0, false});
    std::shared_ptr<Generation> gen = builder.Build();

    const LexicalRetriever retriever({}, UnweightedFields());
    Result<Query> parsed = query::Parse("\"to be or not to be\"");
    CHECK(parsed.ok());
    if (!parsed.ok()) return;
    CHECK_EQ(parsed.value().terms.size(), size_t{6});  // stopwords retained in a phrase

    Result<std::vector<Candidate>> result = retriever.Retrieve(*gen, parsed.value());
    CHECK(result.ok());
    CHECK_EQ(result.value().size(), size_t{1});
    if (!result.value().empty()) CHECK_EQ(result.value()[0].doc, DocId{0});
}

// ===========================================================================
// Filters
// ===========================================================================

namespace {

std::shared_ptr<Generation> FilterCorpus() {
    CorpusBuilder builder;
    builder.Add({"en_news_100", "", "basalt", "", "en", "news", 100, 0.0, false});
    builder.Add({"en_blog_200", "", "basalt", "", "en", "blog", 200, 0.0, false});
    builder.Add({"fr_news_300", "", "basalt", "", "fr", "news", 300, 0.0, false});
    builder.Add({"fr_blog_400", "", "basalt", "", "fr", "blog", 400, 0.0, false});
    return builder.Build();
}

size_t CountMatches(const std::shared_ptr<Generation>& gen, const std::string& query_text) {
    const LexicalRetriever retriever({}, UnweightedFields());
    Result<Query> parsed = query::Parse(query_text);
    if (!parsed.ok()) {
        meridian::testing::ReportFailure(__FILE__, __LINE__,
                                        "parse failed for \"" + query_text + "\": " +
                                            parsed.status().ToString());
        return 0;
    }
    Result<std::vector<Candidate>> result = retriever.Retrieve(*gen, parsed.value());
    if (!result.ok()) return 0;
    return result.value().size();
}

}  // namespace

MERIDIAN_TEST(FiltersApplyPerFieldAndInCombination) {
    std::shared_ptr<Generation> gen = FilterCorpus();

    CHECK_EQ(CountMatches(gen, "basalt"), size_t{4});
    CHECK_EQ(CountMatches(gen, "basalt language:en"), size_t{2});
    CHECK_EQ(CountMatches(gen, "basalt language:fr"), size_t{2});
    CHECK_EQ(CountMatches(gen, "basalt category:news"), size_t{2});
    CHECK_EQ(CountMatches(gen, "basalt category:blog"), size_t{2});
    CHECK_EQ(CountMatches(gen, "basalt after:200"), size_t{3});   // inclusive
    CHECK_EQ(CountMatches(gen, "basalt before:200"), size_t{2});  // inclusive
    CHECK_EQ(CountMatches(gen, "basalt after:150 before:350"), size_t{2});

    // Combinations narrow rather than widen.
    CHECK_EQ(CountMatches(gen, "basalt language:en category:news"), size_t{1});
    CHECK_EQ(CountMatches(gen, "basalt language:fr category:blog"), size_t{1});
    CHECK_EQ(CountMatches(gen, "basalt language:en after:150"), size_t{1});

    // A filter matching nothing returns nothing, rather than falling back to unfiltered
    // results — which is the failure mode that makes a broken filter invisible.
    CHECK_EQ(CountMatches(gen, "basalt language:de"), size_t{0});
    CHECK_EQ(CountMatches(gen, "basalt language:en category:sport"), size_t{0});
    CHECK_EQ(CountMatches(gen, "basalt after:1000"), size_t{0});
    CHECK_EQ(CountMatches(gen, "basalt after:300 before:200"), size_t{0});
}

MERIDIAN_TEST(FilteringHappensBeforeTheCandidateBudgetIsSpent) {
    // 100 documents, 10 of them in the wanted category, candidate_k of 10. Filtering
    // after scoring would spend the whole budget on documents that are then discarded and
    // return far fewer than 10; filtering during retrieval returns all 10.
    CorpusBuilder builder;
    for (int i = 0; i < 100; ++i) {
        builder.Add({"d" + std::to_string(i), "", "marble", "", "en",
                     i % 10 == 0 ? "wanted" : "other", 0, 0.0, false});
    }
    std::shared_ptr<Generation> gen = builder.Build();

    query::ParseOptions options;
    options.candidate_k = 10;
    Result<Query> parsed = query::Parse("marble category:wanted", options);
    CHECK(parsed.ok());
    if (!parsed.ok()) return;

    const LexicalRetriever retriever({}, UnweightedFields());
    Result<std::vector<Candidate>> result = retriever.Retrieve(*gen, parsed.value());
    CHECK(result.ok());
    CHECK_EQ(result.value().size(), size_t{10});
    // Only the surviving documents were scored, so the scoring work tracks the filtered
    // set rather than the match set.
    CHECK_EQ(retriever.stats().candidates_scored, uint64_t{10});
}

MERIDIAN_TEST(FilterOnlyQueriesBrowseTheFilteredSet) {
    std::shared_ptr<Generation> gen = FilterCorpus();
    Result<Query> parsed = query::Parse("language:fr");
    CHECK(parsed.ok());
    if (!parsed.ok()) return;
    CHECK(parsed.value().terms.empty());

    const LexicalRetriever retriever({}, UnweightedFields());
    Result<std::vector<Candidate>> result = retriever.Retrieve(*gen, parsed.value());
    CHECK(result.ok());
    if (!result.ok()) return;
    CHECK_EQ(result.value().size(), size_t{2});
    // No text evidence, so no score. Documented behaviour rather than an invented order.
    for (const Candidate& c : result.value()) CHECK_NEAR(c.score, 0.0, "filter-only score");
}

MERIDIAN_TEST(ACorruptPostingBlockFailsTheQueryInsteadOfShorteningIt) {
    // The failure this whole mechanism exists for, tested at the level where it does
    // damage. A corrupt block stops a walk and reports `kNoMoreDocs`, which is exactly what
    // the end of a list looks like — so retrieval could return a short result set and say
    // `ok()`. A query for every document matching a term, answered with half of them and no
    // error, is the worst outcome in the system, because nothing downstream can detect it.
    //
    // 12 documents all containing "zebra", with the lists breaking after 4 decodes.
    CorpusBuilder builder;
    for (int i = 0; i < 12; ++i) {
        builder.Add({"doc-" + std::to_string(i), "", "zebra alpha beta", "", "en", "general", 0,
                     0.0, false});
    }
    builder.CorruptTerm("zebra", 4);
    std::shared_ptr<Generation> gen = builder.Build();

    const LexicalRetriever retriever({}, FieldWeights{});
    Result<Query> parsed = query::Parse("zebra");
    CHECK(parsed.ok());

    Result<std::vector<Candidate>> result = retriever.Retrieve(*gen, parsed.value());

    // Must fail, and must say why: `kCorruption` tells an operator to re-index a damaged
    // segment, which is a different instruction from every other code here.
    CHECK(!result.ok());
    CHECK(result.status().code() == Code::kCorruption);
    // The context names the term and segment, so the message identifies what to re-index
    // rather than only that something is wrong.
    CHECK(result.status().message().find("zebra") != std::string::npos);
    std::printf("  corrupt block surfaced as: %s\n", result.status().ToString().c_str());
}

MERIDIAN_TEST(AnUncorruptedQueryOverTheSameCorpusStillSucceeds) {
    // The control. Without it the test above passes just as well against a retriever that
    // fails every query, which would prove nothing.
    CorpusBuilder builder;
    for (int i = 0; i < 12; ++i) {
        builder.Add({"doc-" + std::to_string(i), "", "zebra alpha beta", "", "en", "general", 0,
                     0.0, false});
    }
    builder.CorruptTerm("zebra", 4);
    std::shared_ptr<Generation> gen = builder.Build();

    const LexicalRetriever retriever({}, FieldWeights{});
    Result<Query> parsed = query::Parse("alpha");  // a term whose lists are intact
    CHECK(parsed.ok());

    Result<std::vector<Candidate>> result = retriever.Retrieve(*gen, parsed.value());
    CHECK(result.ok());
    CHECK_EQ(result.value().size(), 12u);
}

MERIDIAN_TEST_MAIN()
