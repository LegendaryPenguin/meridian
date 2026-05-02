// The Index: buffering, flushing, tombstones, generation publication, and a randomised
// comparison of what it answers against a std::map that cannot be wrong.

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <filesystem>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "../src/index/segment.hpp"
#include "meridian/index.hpp"
#include "testing.hpp"

using namespace meridian;

namespace {

class TempDirectory {
public:
    explicit TempDirectory(const std::string& tag) {
        static std::atomic<int> counter{0};
        path_ = (std::filesystem::temp_directory_path() /
                 ("meridian-" + tag + "-" + std::to_string(counter.fetch_add(1))))
                    .string();
        std::filesystem::remove_all(path_);
        std::filesystem::create_directories(path_);
    }
    ~TempDirectory() { std::filesystem::remove_all(path_); }

    TempDirectory(const TempDirectory&) = delete;
    TempDirectory& operator=(const TempDirectory&) = delete;

    const std::string& path() const { return path_; }

private:
    std::string path_;
};

using Tokens = std::unordered_map<Field, std::vector<std::string>>;

DocumentMeta MakeDoc(const std::string& id, const Tokens& tokens) {
    DocumentMeta meta;
    meta.external_id = id;
    meta.language = "en";
    meta.category = "doc";
    meta.published_at = 1'700'000'000;
    uint32_t length = 0;
    for (const auto& [field, list] : tokens) {
        (void)field;
        length += static_cast<uint32_t>(list.size());
    }
    meta.length = length;
    meta.popularity = 1.0;
    return meta;
}

/// A reference model of the corpus, kept in plain std::map so the comparison has a side
/// that is obviously right.
struct Reference {
    /// (field, term) -> set of external ids containing it
    std::map<std::pair<uint8_t, std::string>, std::set<std::string>> documents_for_term;
    /// (field, term) -> total occurrences
    std::map<std::pair<uint8_t, std::string>, uint64_t> occurrences_for_term;
    std::set<std::string> live;
    std::map<std::string, uint32_t> length;

    void Add(const std::string& id, const Tokens& tokens) {
        live.insert(id);
        uint32_t total = 0;
        for (const auto& [field, list] : tokens) {
            total += static_cast<uint32_t>(list.size());
            for (const std::string& token : list) {
                const std::pair<uint8_t, std::string> key{static_cast<uint8_t>(field), token};
                documents_for_term[key].insert(id);
                ++occurrences_for_term[key];
            }
        }
        length[id] = total;
    }
};

const std::vector<std::string>& Vocabulary() {
    static const std::vector<std::string> vocabulary = {
        "alpha",   "alphabet", "alphanumeric", "beta",    "betamax", "gamma",
        "gammas",  "delta",    "deltas",       "epsilon", "zeta",    "eta",
        "theta",   "iota",     "kappa",        "lambda",  "mu",      "nu",
        "xi",      "omicron",  "pi",           "rho",     "sigma",   "tau",
    };
    return vocabulary;
}

Tokens RandomTokens(std::mt19937& rng) {
    Tokens tokens;
    for (Field field : {Field::kTitle, Field::kBody, Field::kTags}) {
        const size_t count = 1 + rng() % 10;
        std::vector<std::string> list;
        for (size_t i = 0; i < count; ++i) list.push_back(Vocabulary()[rng() % Vocabulary().size()]);
        tokens[field] = std::move(list);
    }
    return tokens;
}

/// Walks a generation and returns the external ids a term matches, skipping tombstoned
/// documents the way a query must.
std::set<std::string> MatchingDocuments(const Generation& generation, const std::string& term,
                                       Field field) {
    std::set<std::string> ids;
    for (const auto& segment : generation.segments()) {
        std::unique_ptr<PostingIterator> it = segment->Postings(term, field);
        if (it == nullptr) continue;
        for (DocId doc = it->doc(); doc != kNoMoreDocs; doc = it->next()) {
            if (segment->IsDeleted(doc)) continue;
            auto meta = segment->Document(doc);
            if (!meta.ok()) continue;
            ids.insert(meta.value().external_id);
        }
    }
    return ids;
}

}  // namespace

MERIDIAN_TEST(DocumentsAreNotSearchableUntilFlushed) {
    const TempDirectory dir("index-flush");
    Index::Options options;
    options.directory = dir.path();
    options.docs_per_segment = 1000;  // larger than what the test adds, so no auto-flush

    Index index(options);
    CHECK_OK(index.Open());

    Tokens tokens;
    tokens[Field::kBody] = {"alpha", "beta"};
    CHECK_OK(index.Add(MakeDoc("a", tokens), tokens));

    // The buffer is deliberately invisible. A query that could see half a buffer would
    // see a document whose postings are written and whose metadata is not.
    std::shared_ptr<Generation> before = index.Acquire();
    CHECK(before != nullptr);
    CHECK_EQ(before->segments().size(), 0u);
    CHECK_EQ(before->stats().document_count, 0u);

    CHECK_OK(index.Flush());

    const std::shared_ptr<Generation> after = index.Acquire();
    CHECK(after->version() > before->version());
    CHECK_EQ(after->segments().size(), 1u);
    CHECK_EQ(after->stats().document_count, 1u);
    CHECK_EQ(after->stats().total_tokens, 2u);
    CHECK_EQ(after->stats().average_document_length, 2.0);

    // The old generation is still a valid, consistent view. It is simply older.
    CHECK_EQ(before->segments().size(), 0u);

    CHECK_OK(index.Close());
}

MERIDIAN_TEST(AddAutoFlushesAtTheSegmentThreshold) {
    const TempDirectory dir("index-autoflush");
    Index::Options options;
    options.directory = dir.path();
    options.docs_per_segment = 10;
    options.merge_factor = 100;  // keep merges out of this test

    Index index(options);
    CHECK_OK(index.Open());

    std::mt19937 rng(1);
    for (int i = 0; i < 35; ++i) {
        const Tokens tokens = RandomTokens(rng);
        CHECK_OK(index.Add(MakeDoc("doc-" + std::to_string(i), tokens), tokens));
    }
    const std::shared_ptr<Generation> generation = index.Acquire();
    CHECK_EQ(generation->segments().size(), 3u);
    CHECK_EQ(generation->stats().document_count, 30u);

    // doc_base must tile the generation exactly, or Resolve hands a hit to the wrong
    // segment and the ranked result names a different document than the one that matched.
    DocId expected_base = 0;
    for (const auto& segment : generation->segments()) {
        CHECK_EQ(segment->meta().doc_base, expected_base);
        expected_base += segment->meta().document_count;
    }
    for (DocId global = 0; global < expected_base; ++global) {
        auto resolved = generation->Resolve(global);
        CHECK(resolved.ok());
        if (!resolved.ok()) continue;
        CHECK(resolved.value().second < resolved.value().first->meta().document_count);
    }
    CHECK(!generation->Resolve(expected_base).ok());

    CHECK_OK(index.Close());
}

MERIDIAN_TEST(DeleteTombstonesAndPublishesANewGeneration) {
    const TempDirectory dir("index-delete");
    Index::Options options;
    options.directory = dir.path();
    options.docs_per_segment = 1000;

    Index index(options);
    CHECK_OK(index.Open());

    Tokens tokens;
    tokens[Field::kBody] = {"alpha"};
    for (int i = 0; i < 5; ++i) {
        CHECK_OK(index.Add(MakeDoc("doc-" + std::to_string(i), tokens), tokens));
    }
    CHECK_OK(index.Flush());

    const std::shared_ptr<Generation> before = index.Acquire();
    CHECK_EQ(MatchingDocuments(*before, "alpha", Field::kBody).size(), 5u);
    CHECK_EQ(before->stats().document_count, 5u);

    CHECK_OK(index.Delete("doc-2"));

    const std::shared_ptr<Generation> after = index.Acquire();
    CHECK(after->version() > before->version());
    const std::set<std::string> live = MatchingDocuments(*after, "alpha", Field::kBody);
    CHECK_EQ(live.size(), 4u);
    CHECK(live.count("doc-2") == 0);
    CHECK_EQ(after->stats().document_count, 4u);
    // Corpus statistics must drop with the deletion, or BM25's average document length
    // drifts upward as tombstones accumulate and every score shifts with it.
    CHECK_EQ(after->stats().total_tokens, 4u);

    // The generation acquired before the delete still sees the document. That is the
    // point: a query mid-flight gets one consistent view, not a view that changes under it.
    CHECK_EQ(MatchingDocuments(*before, "alpha", Field::kBody).size(), 5u);

    CHECK_EQ(static_cast<int>(index.Delete("not-a-document").code()),
             static_cast<int>(Code::kNotFound));

    // A delete naming a document that is still in the buffer has to wait for the flush
    // that gives it a local id, and must not make it searchable in between.
    Tokens more;
    more[Field::kBody] = {"beta"};
    CHECK_OK(index.Add(MakeDoc("buffered", more), more));
    CHECK_OK(index.Delete("buffered"));
    CHECK_OK(index.Flush());
    const std::shared_ptr<Generation> final_generation = index.Acquire();
    CHECK(MatchingDocuments(*final_generation, "beta", Field::kBody).empty());
    CHECK_EQ(final_generation->stats().document_count, 4u);

    CHECK_OK(index.Close());
}

MERIDIAN_TEST(IndexReopensFromItsManifest) {
    const TempDirectory dir("index-reopen");
    Index::Options options;
    options.directory = dir.path();
    options.docs_per_segment = 20;
    options.merge_factor = 100;

    Reference reference;
    std::mt19937 rng(77);
    {
        Index index(options);
        CHECK_OK(index.Open());
        for (int i = 0; i < 60; ++i) {
            const std::string id = "doc-" + std::to_string(i);
            const Tokens tokens = RandomTokens(rng);
            CHECK_OK(index.Add(MakeDoc(id, tokens), tokens));
            reference.Add(id, tokens);
        }
        CHECK_OK(index.Flush());
        CHECK_OK(index.Delete("doc-7"));
        reference.live.erase("doc-7");
        CHECK_OK(index.Close());
    }

    Index reopened(options);
    CHECK_OK(reopened.Open());
    const std::shared_ptr<Generation> generation = reopened.Acquire();
    CHECK_EQ(generation->stats().document_count, reference.live.size());

    for (const auto& [key, expected_ids] : reference.documents_for_term) {
        std::set<std::string> expected;
        for (const std::string& id : expected_ids) {
            if (reference.live.count(id) != 0) expected.insert(id);
        }
        const std::set<std::string> actual =
            MatchingDocuments(*generation, key.second, static_cast<Field>(key.first));
        CHECK(actual == expected);
    }

    // The tombstone survived the restart, which it only can because the manifest carries
    // it: the segment file itself is immutable and knows nothing about it.
    CHECK_EQ(static_cast<int>(reopened.Delete("doc-7").code()),
             static_cast<int>(Code::kNotFound));
    CHECK_OK(reopened.Close());
}

MERIDIAN_TEST(TermLookupsMatchABruteForceReference) {
    // The differential test. Several segments, a vocabulary with shared prefixes, and
    // every (field, term) pair compared against a std::map built from the same input.
    const TempDirectory dir("index-differential");
    Index::Options options;
    options.directory = dir.path();
    options.docs_per_segment = 37;  // not a round number, so segments do not align with
    options.merge_factor = 100;     // the document count

    Index index(options);
    CHECK_OK(index.Open());

    Reference reference;
    std::mt19937 rng(20240602);
    const int document_count = 400;
    for (int i = 0; i < document_count; ++i) {
        const std::string id = "doc-" + std::to_string(i);
        const Tokens tokens = RandomTokens(rng);
        CHECK_OK(index.Add(MakeDoc(id, tokens), tokens));
        reference.Add(id, tokens);
    }
    CHECK_OK(index.Flush());

    const std::shared_ptr<Generation> generation = index.Acquire();
    CHECK_EQ(generation->stats().document_count, static_cast<uint32_t>(document_count));

    size_t terms_checked = 0;
    for (const auto& [key, expected_ids] : reference.documents_for_term) {
        const Field field = static_cast<Field>(key.first);
        const std::string& term = key.second;
        ++terms_checked;

        CHECK(MatchingDocuments(*generation, term, field) == expected_ids);

        // Document frequency is the sum over segments, because that is how IDF is
        // computed and an off-by-one in one segment is invisible in any single list.
        uint32_t document_frequency = 0;
        uint64_t collection_frequency = 0;
        for (const auto& segment : generation->segments()) {
            const auto stats = segment->Term(term, field);
            if (!stats.has_value()) continue;
            document_frequency += stats->document_frequency;
            collection_frequency += stats->collection_frequency;
        }
        CHECK_EQ(document_frequency, static_cast<uint32_t>(expected_ids.size()));
        CHECK_EQ(collection_frequency, reference.occurrences_for_term.at(key));
    }

    // Terms that are not in the corpus must be absent from every segment, including the
    // ones that are prefixes and extensions of terms that are.
    for (const char* missing : {"alph", "alphabets", "zzz", "", "betama"}) {
        for (Field field : {Field::kTitle, Field::kBody, Field::kTags}) {
            for (const auto& segment : generation->segments()) {
                CHECK(!segment->Term(missing, field).has_value());
            }
        }
    }

    uint64_t total_tokens = 0;
    for (const auto& [id, length] : reference.length) {
        (void)id;
        total_tokens += length;
    }
    CHECK_EQ(generation->stats().total_tokens, total_tokens);

    std::printf("  compared %zu (field, term) pairs across %zu segments against the reference\n",
                terms_checked, generation->segments().size());
    CHECK_OK(index.Close());
}

MERIDIAN_TEST(DuplicateExternalIdsAreRefused) {
    const TempDirectory dir("index-duplicate");
    Index::Options options;
    options.directory = dir.path();
    options.docs_per_segment = 1000;

    Index index(options);
    CHECK_OK(index.Open());
    Tokens tokens;
    tokens[Field::kBody] = {"alpha"};
    CHECK_OK(index.Add(MakeDoc("same", tokens), tokens));
    // While still buffered.
    CHECK_NOT_OK(index.Add(MakeDoc("same", tokens), tokens));
    CHECK_OK(index.Flush());
    // And after flushing, where the locator is the thing that knows.
    CHECK_NOT_OK(index.Add(MakeDoc("same", tokens), tokens));
    CHECK_OK(index.Close());
}

MERIDIAN_TEST(WriteAmplificationIsReportedAndRisesWithMergeWork) {
    const TempDirectory dir("index-amplification");
    Index::Options options;
    options.directory = dir.path();
    options.docs_per_segment = 25;
    options.merge_factor = 100;  // no automatic merging; ForceMerge does it explicitly

    Index index(options);
    CHECK_OK(index.Open());

    std::mt19937 rng(5);
    for (int i = 0; i < 200; ++i) {
        const Tokens tokens = RandomTokens(rng);
        CHECK_OK(index.Add(MakeDoc("doc-" + std::to_string(i), tokens), tokens));
    }
    CHECK_OK(index.Flush());

    const Index::Stats flushed = index.stats();
    CHECK_EQ(flushed.documents_indexed, 200u);
    CHECK(flushed.bytes_written > 0);
    CHECK_EQ(flushed.merges_run, 0u);
    const double before = flushed.WriteAmplification();
    CHECK(before > 0.0);

    CHECK_OK(index.ForceMerge());
    const Index::Stats merged = index.stats();
    CHECK_EQ(merged.merges_run, 1u);
    CHECK_EQ(merged.segments_merged, 8u);
    CHECK(merged.bytes_written > flushed.bytes_written);

    const double after = merged.WriteAmplification();
    std::printf("  write amplification: %.1f bytes/doc after flushes, %.1f after one full "
                "merge (%llu bytes written for %llu documents)\n",
                before, after, static_cast<unsigned long long>(merged.bytes_written),
                static_cast<unsigned long long>(merged.documents_indexed));
    // A merge rewrites data that was already written once, so the figure has to go up,
    // and by roughly the size of the data it rewrote rather than by an arbitrary amount.
    CHECK(after > before);
    CHECK(after < before * 3.0);

    CHECK_OK(index.Close());
}

MERIDIAN_TEST(MaybeMergeIsSizeTieredAndIdempotentWhenNothingQualifies) {
    const TempDirectory dir("index-tiered");
    Index::Options options;
    options.directory = dir.path();
    options.docs_per_segment = 50;
    options.merge_factor = 4;

    Index index(options);
    CHECK_OK(index.Open());

    std::mt19937 rng(13);
    for (int i = 0; i < 200; ++i) {
        const Tokens tokens = RandomTokens(rng);
        CHECK_OK(index.Add(MakeDoc("doc-" + std::to_string(i), tokens), tokens));
    }
    CHECK_OK(index.Flush());
    CHECK_EQ(index.Acquire()->segments().size(), 4u);

    CHECK_OK(index.MaybeMerge());
    const std::shared_ptr<Generation> merged = index.Acquire();
    CHECK_EQ(merged->segments().size(), 1u);
    CHECK_EQ(merged->stats().document_count, 200u);
    CHECK_EQ(index.stats().merges_run, 1u);

    // Nothing qualifies now, and calling again must be a cheap no-op rather than a merge
    // of one segment into a copy of itself.
    CHECK_OK(index.MaybeMerge());
    CHECK_EQ(index.stats().merges_run, 1u);
    CHECK_EQ(index.Acquire()->segments().size(), 1u);

    CHECK_OK(index.Close());
}

MERIDIAN_TEST_MAIN()
