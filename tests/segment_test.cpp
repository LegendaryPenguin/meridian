// The term dictionary and the segment file: build it, reopen it, find everything in it,
// and check that reading less of it reads fewer bytes.

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

#include "../src/index/builder.hpp"
#include "../src/index/dictionary.hpp"
#include "../src/index/segment.hpp"
#include "../src/storage/file.hpp"
#include "meridian/index.hpp"
#include "testing.hpp"

using namespace meridian;
using meridian::index::Dictionary;
using meridian::index::DictionaryWriter;
using meridian::index::MakeTermKey;
using meridian::index::SegmentBuilder;
using meridian::index::SegmentReaderImpl;
using meridian::index::TermLocation;

namespace {

/// A directory that removes itself, so a failing test does not leave segments behind and
/// a passing one does not accumulate them.
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
    std::string file(const std::string& name) const { return path_ + "/" + name; }

private:
    std::string path_;
};

TermLocation MakeLocation(uint64_t offset, uint64_t length, uint32_t df, uint64_t cf) {
    TermLocation location;
    location.offset = offset;
    location.length = length;
    location.stats.document_frequency = df;
    location.stats.collection_frequency = cf;
    return location;
}

DocumentMeta MakeDoc(const std::string& id, uint32_t length) {
    DocumentMeta meta;
    meta.external_id = id;
    meta.language = "en";
    meta.category = "reference";
    meta.published_at = 1700000000;
    meta.length = length;
    meta.popularity = 0.25;
    return meta;
}

/// A small synthetic corpus with a controlled vocabulary, so a reference std::map can say
/// what the segment should contain.
struct Corpus {
    std::vector<DocumentMeta> documents;
    std::vector<std::unordered_map<Field, std::vector<std::string>>> tokens;
    std::map<std::pair<uint8_t, std::string>, TermStats> expected;
};

Corpus MakeCorpus(size_t document_count, uint32_t seed) {
    std::mt19937 rng(seed);
    const std::vector<std::string> vocabulary = {
        "retrieval", "retrieve",  "retrieved", "retriever", "retrieving", "index",
        "indexed",   "indexes",   "indexing",  "posting",   "postings",   "skip",
        "skiplist",  "segment",   "segments",  "merge",     "merged",     "query",
    };

    Corpus corpus;
    for (size_t d = 0; d < document_count; ++d) {
        std::unordered_map<Field, std::vector<std::string>> by_field;
        for (Field field : {Field::kTitle, Field::kBody, Field::kTags}) {
            const size_t count = 1 + rng() % 12;
            std::vector<std::string> tokens;
            for (size_t i = 0; i < count; ++i) {
                tokens.push_back(vocabulary[rng() % vocabulary.size()]);
            }
            by_field[field] = std::move(tokens);
        }

        uint32_t length = 0;
        for (const auto& [field, tokens] : by_field) {
            (void)field;
            length += static_cast<uint32_t>(tokens.size());
        }
        corpus.documents.push_back(MakeDoc("doc-" + std::to_string(d), length));

        for (const auto& [field, tokens] : by_field) {
            std::map<std::string, uint32_t> counts;
            for (const std::string& token : tokens) ++counts[token];
            for (const auto& [term, count] : counts) {
                TermStats& stats =
                    corpus.expected[{static_cast<uint8_t>(field), term}];
                ++stats.document_frequency;
                stats.collection_frequency += count;
            }
        }
        corpus.tokens.push_back(std::move(by_field));
    }
    return corpus;
}

Status BuildSegment(const Corpus& corpus, const std::string& path, SegmentMeta& meta_out) {
    SegmentBuilder::Options options;
    options.positions[static_cast<size_t>(Field::kTitle)] = true;
    options.positions[static_cast<size_t>(Field::kBody)] = true;
    options.positions[static_cast<size_t>(Field::kTags)] = false;

    SegmentBuilder builder(options);
    for (size_t d = 0; d < corpus.documents.size(); ++d) {
        MERIDIAN_TRY(builder.AddTokenizedDocument(corpus.documents[d], corpus.tokens[d], nullptr));
    }
    return builder.Build(path, meta_out);
}

}  // namespace

MERIDIAN_TEST(EmptyDictionaryProbesCleanly) {
    DictionaryWriter writer;
    std::vector<uint8_t> blocks, index;
    CHECK_OK(writer.Finish(blocks, index));
    CHECK_EQ(writer.term_count(), 0u);
    CHECK(blocks.empty());

    auto loaded = Dictionary::Load(index);
    CHECK(loaded.ok());
    if (!loaded.ok()) return;
    const Dictionary dictionary = loaded.take();
    CHECK_EQ(dictionary.block_count(), 0u);
    CHECK_EQ(dictionary.term_count(), 0u);

    const storage::MemoryByteSource source(blocks);
    Status status;
    // A dictionary with nothing in it must answer "absent", not fail and not reach for a
    // block that is not there.
    CHECK(!dictionary.Lookup(source, 0, MakeTermKey("anything", Field::kBody), &status)
               .has_value());
    CHECK_OK(status);
}

MERIDIAN_TEST(DictionaryFindsPresentTermsAndRejectsAbsentOnes) {
    // Deliberately prefix-heavy, because front coding is only correct if the shared
    // prefix is reconstructed from the *previous* key in the same block, and a block
    // boundary resets it.
    std::vector<std::string> terms;
    for (const std::string& stem : {"re", "ret", "retr", "retrie", "retrieval", "retrieve",
                                    "retrieved", "retriever", "retrieves", "retrieving"}) {
        terms.push_back(stem);
    }
    for (int i = 0; i < 100; ++i) terms.push_back("segment" + std::to_string(1000 + i));

    std::map<std::string, TermLocation> reference;
    uint64_t offset = 0;
    for (Field field : {Field::kTitle, Field::kBody, Field::kTags}) {
        for (const std::string& term : terms) {
            reference[MakeTermKey(term, field)] =
                MakeLocation(offset, 17, static_cast<uint32_t>(term.size()), term.size() * 3);
            offset += 23;
        }
    }

    DictionaryWriter writer;
    for (const auto& [key, location] : reference) CHECK_OK(writer.Add(key, location));
    std::vector<uint8_t> blocks, index;
    CHECK_OK(writer.Finish(blocks, index));
    CHECK_EQ(writer.term_count(), reference.size());

    auto loaded = Dictionary::Load(index);
    CHECK(loaded.ok());
    if (!loaded.ok()) return;
    const Dictionary dictionary = loaded.take();
    CHECK(dictionary.block_count() > 1);

    const storage::MemoryByteSource source(blocks);
    for (const auto& [key, expected] : reference) {
        Status status;
        const auto found = dictionary.Lookup(source, 0, key, &status);
        CHECK_OK(status);
        CHECK(found.has_value());
        if (!found.has_value()) continue;
        CHECK_EQ(found->offset, expected.offset);
        CHECK_EQ(found->length, expected.length);
        CHECK_EQ(found->stats.document_frequency, expected.stats.document_frequency);
        CHECK_EQ(found->stats.collection_frequency, expected.stats.collection_frequency);
    }

    // Absent terms, chosen to sit before, inside and after the key range, and to be
    // prefixes and extensions of present ones. "retriev" is the dangerous one: it is a
    // prefix of several present keys and front coding must not resurrect it.
    for (const char* missing : {"", "a", "retriev", "retrievals", "retrieverX", "zzz",
                                "segment0999", "segment1100"}) {
        for (Field field : {Field::kTitle, Field::kBody, Field::kTags}) {
            Status status;
            const auto found =
                dictionary.Lookup(source, 0, MakeTermKey(missing, field), &status);
            CHECK_OK(status);
            CHECK(!found.has_value());
        }
    }

    // A walk must produce every term exactly once, in key order. The merge depends on it.
    std::vector<std::string> walked;
    CHECK_OK(dictionary.ForEachTerm(
        source, 0, [&walked](std::string_view key, const TermLocation&) -> Status {
            walked.emplace_back(key);
            return Status::Ok();
        }));
    CHECK_EQ(walked.size(), reference.size());
    CHECK(std::is_sorted(walked.begin(), walked.end()));
}

MERIDIAN_TEST(DictionaryRejectsOutOfOrderKeys) {
    DictionaryWriter writer;
    CHECK_OK(writer.Add(MakeTermKey("beta", Field::kBody), MakeLocation(0, 1, 1, 1)));
    // Descending, and the same key twice: both would make the binary search miss terms
    // that are in the file.
    CHECK_NOT_OK(writer.Add(MakeTermKey("alpha", Field::kBody), MakeLocation(1, 1, 1, 1)));
    CHECK_NOT_OK(writer.Add(MakeTermKey("beta", Field::kBody), MakeLocation(1, 1, 1, 1)));
}

MERIDIAN_TEST(SegmentRoundTripsEveryTermAndItsStatistics) {
    const TempDirectory dir("segment-roundtrip");
    const Corpus corpus = MakeCorpus(300, 11);

    SegmentMeta built;
    CHECK_OK(BuildSegment(corpus, dir.file("segment-1.mrd"), built));
    CHECK_EQ(built.document_count, corpus.documents.size());
    CHECK_EQ(built.term_count, corpus.expected.size());

    auto opened = SegmentReaderImpl::Open(dir.file("segment-1.mrd"), 1, 0);
    CHECK(opened.ok());
    if (!opened.ok()) return;
    const std::shared_ptr<SegmentReaderImpl> reader = opened.take();

    CHECK_EQ(reader->meta().document_count, corpus.documents.size());
    CHECK_EQ(reader->meta().term_count, corpus.expected.size());

    for (const auto& [key, expected] : corpus.expected) {
        const Field field = static_cast<Field>(key.first);
        const std::string& term = key.second;

        const auto stats = reader->Term(term, field);
        CHECK(stats.has_value());
        if (!stats.has_value()) continue;
        CHECK_EQ(stats->document_frequency, expected.document_frequency);
        CHECK_EQ(stats->collection_frequency, expected.collection_frequency);

        // The posting list has to agree with the statistics, or IDF and the iterator
        // disagree about the same term.
        std::unique_ptr<PostingIterator> it = reader->Postings(term, field);
        CHECK(it != nullptr);
        if (it == nullptr) continue;
        uint32_t seen = 0;
        uint64_t total_frequency = 0;
        DocId previous = kNoMoreDocs;
        for (DocId doc = it->doc(); doc != kNoMoreDocs; doc = it->next()) {
            if (seen > 0) CHECK(doc > previous);
            previous = doc;
            ++seen;
            total_frequency += it->term_frequency();
        }
        CHECK_EQ(seen, expected.document_frequency);
        CHECK_EQ(total_frequency, expected.collection_frequency);
    }

    // Absent terms are nullopt and nullptr, not an error: most terms are absent from most
    // segments and the merge loops would be unreadable if that were a Status.
    CHECK(!reader->Term("nonexistent", Field::kBody).has_value());
    CHECK(reader->Postings("nonexistent", Field::kBody) == nullptr);

    for (size_t d = 0; d < corpus.documents.size(); ++d) {
        auto document = reader->Document(static_cast<DocId>(d));
        CHECK(document.ok());
        if (!document.ok()) continue;
        CHECK_EQ(document.value().external_id, corpus.documents[d].external_id);
        CHECK_EQ(document.value().language, corpus.documents[d].language);
        CHECK_EQ(document.value().category, corpus.documents[d].category);
        CHECK_EQ(document.value().published_at, corpus.documents[d].published_at);
        CHECK_EQ(document.value().length, corpus.documents[d].length);
        CHECK_EQ(document.value().popularity, corpus.documents[d].popularity);
        CHECK(!reader->IsDeleted(static_cast<DocId>(d)));
    }

    auto past_the_end = reader->Document(static_cast<DocId>(corpus.documents.size()));
    CHECK(!past_the_end.ok());
    CHECK_EQ(static_cast<int>(past_the_end.status().code()), static_cast<int>(Code::kNotFound));
}

MERIDIAN_TEST(SegmentHonoursThePerFieldPositionDecision) {
    const TempDirectory dir("segment-positions");
    const Corpus corpus = MakeCorpus(40, 5);
    SegmentMeta built;
    CHECK_OK(BuildSegment(corpus, dir.file("segment-1.mrd"), built));

    auto opened = SegmentReaderImpl::Open(dir.file("segment-1.mrd"), 1, 0);
    CHECK(opened.ok());
    if (!opened.ok()) return;
    const std::shared_ptr<SegmentReaderImpl> reader = opened.take();

    bool checked_body = false, checked_tags = false;
    for (const auto& [key, stats] : corpus.expected) {
        (void)stats;
        const Field field = static_cast<Field>(key.first);
        std::unique_ptr<PostingIterator> it = reader->Postings(key.second, field);
        if (it == nullptr) continue;
        if (field == Field::kBody) {
            CHECK(!it->positions().empty());
            checked_body = true;
        } else if (field == Field::kTags) {
            // kTags is configured without positions, so the whole position payload is
            // absent from the file rather than present and empty.
            CHECK(it->positions().empty());
            checked_tags = true;
        }
    }
    CHECK(checked_body);
    CHECK(checked_tags);
}

MERIDIAN_TEST(BytesReadTracksWhatWasActuallyAskedFor) {
    const TempDirectory dir("segment-bytes");
    const Corpus corpus = MakeCorpus(500, 23);
    SegmentMeta built;
    CHECK_OK(BuildSegment(corpus, dir.file("segment-1.mrd"), built));

    const auto open = [&dir]() {
        auto opened = SegmentReaderImpl::Open(dir.file("segment-1.mrd"), 1, 0);
        CHECK(opened.ok());
        return opened.ok() ? opened.take() : nullptr;
    };

    const std::shared_ptr<SegmentReaderImpl> idle = open();
    const std::shared_ptr<SegmentReaderImpl> one_term = open();
    const std::shared_ptr<SegmentReaderImpl> every_term = open();
    if (idle == nullptr || one_term == nullptr || every_term == nullptr) return;

    const uint64_t baseline = idle->bytes_read();
    CHECK(baseline > 0);  // the footer, the dictionary index and the length array
    CHECK(baseline < built.bytes);

    const auto first = corpus.expected.begin();
    std::unique_ptr<PostingIterator> single =
        one_term->Postings(first->first.second, static_cast<Field>(first->first.first));
    CHECK(single != nullptr);

    for (const auto& [key, stats] : corpus.expected) {
        (void)stats;
        std::unique_ptr<PostingIterator> it =
            every_term->Postings(key.second, static_cast<Field>(key.first));
        if (it == nullptr) continue;
        while (it->next() != kNoMoreDocs) {
        }
    }

    std::printf("  bytes_read: %llu idle, %llu for one term, %llu for all %zu terms (file is "
                "%llu bytes)\n",
                static_cast<unsigned long long>(idle->bytes_read()),
                static_cast<unsigned long long>(one_term->bytes_read()),
                static_cast<unsigned long long>(every_term->bytes_read()),
                corpus.expected.size(), static_cast<unsigned long long>(built.bytes));

    CHECK_EQ(idle->bytes_read(), baseline);
    CHECK(one_term->bytes_read() > baseline);
    CHECK(every_term->bytes_read() > one_term->bytes_read());
    // Three readers over one file, three independent counters. If the counter lived on the
    // file handle this would fail, and every projection measurement taken through a
    // reader that shares a file would be wrong.
    CHECK(idle->bytes_read() < one_term->bytes_read());
}

MERIDIAN_TEST(EmptySegmentIsBuildableAndReadable) {
    const TempDirectory dir("segment-empty");
    SegmentBuilder builder(SegmentBuilder::Options{});
    SegmentMeta built;
    CHECK_OK(builder.Build(dir.file("segment-1.mrd"), built));
    CHECK_EQ(built.document_count, 0u);
    CHECK_EQ(built.term_count, 0u);

    auto opened = SegmentReaderImpl::Open(dir.file("segment-1.mrd"), 1, 0);
    CHECK(opened.ok());
    if (!opened.ok()) return;
    const std::shared_ptr<SegmentReaderImpl> reader = opened.take();
    CHECK(!reader->Term("anything", Field::kBody).has_value());
    CHECK(reader->Postings("anything", Field::kBody) == nullptr);
    CHECK(!reader->Document(0).ok());
    CHECK_EQ(reader->live_tokens(), 0u);
}

MERIDIAN_TEST(DamagedSegmentIsRejectedRatherThanMisread) {
    const TempDirectory dir("segment-damage");
    const Corpus corpus = MakeCorpus(50, 31);
    const std::string path = dir.file("segment-1.mrd");
    SegmentMeta built;
    CHECK_OK(BuildSegment(corpus, path, built));

    // Truncating the tail removes the footer, which is the only way into the file.
    const std::string truncated = dir.file("truncated.mrd");
    std::filesystem::copy_file(path, truncated);
    std::filesystem::resize_file(truncated, std::filesystem::file_size(truncated) - 1);
    auto opened = SegmentReaderImpl::Open(truncated, 1, 0);
    CHECK(!opened.ok());
    if (!opened.ok()) {
        CHECK_EQ(static_cast<int>(opened.status().code()), static_cast<int>(Code::kCorruption));
    }

    // A file that is not a segment at all must be refused on the magic, not crash on an
    // offset read out of random bytes.
    const std::string junk = dir.file("junk.mrd");
    {
        std::vector<uint8_t> noise(4096);
        std::mt19937 rng(3);
        for (uint8_t& b : noise) b = static_cast<uint8_t>(rng());
        CHECK_OK(storage::WriteFileAtomic(junk, noise));
    }
    auto junk_opened = SegmentReaderImpl::Open(junk, 2, 0);
    CHECK(!junk_opened.ok());
}

MERIDIAN_TEST_MAIN()
