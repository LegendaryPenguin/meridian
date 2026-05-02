// The skip list, and the one test that makes the whole feature trustworthy.
//
// A skip-list bug does not crash and does not corrupt anything. It lands on the document
// before the one it should, and the query returns results that are plausible, mostly
// right, and wrong. The only defence is to keep a deliberately naive implementation
// around and insist the two agree on every seek, over randomised lists and randomised
// targets. That is what AdvanceAgreesWithSequentialScan does, and it is the reason the
// sequential iterator still exists.

#include <algorithm>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include "../src/postings/encoder.hpp"
#include "../src/postings/iterator.hpp"
#include "../src/postings/skiplist.hpp"
#include "meridian/postings.hpp"
#include "testing.hpp"

using namespace meridian;
using meridian::postings::IteratorMode;
using meridian::postings::kBlockPostings;
using meridian::postings::kSkipEntryBytes;
using meridian::postings::kSkipInterval;

namespace {

std::shared_ptr<const storage::ByteSource> SourceOver(std::vector<uint8_t> bytes) {
    return std::make_shared<storage::MemoryByteSource>(std::move(bytes));
}

std::vector<Posting> MakeList(const std::vector<DocId>& docs, bool with_positions) {
    std::vector<Posting> list;
    list.reserve(docs.size());
    for (DocId doc : docs) {
        Posting p;
        p.doc = doc;
        p.term_frequency = 1 + (doc % 5);
        if (with_positions) {
            for (uint32_t i = 0; i < (doc % 3); ++i) p.positions.push_back(doc + i);
        }
        list.push_back(std::move(p));
    }
    return list;
}

struct Encoded {
    std::vector<uint8_t> bytes;
    std::vector<DocId> docs;
};

Encoded Encode(const std::vector<DocId>& docs, bool with_positions) {
    Encoded out;
    out.docs = docs;
    const Status s = postings::EncodeList(MakeList(docs, with_positions), with_positions,
                                         out.bytes);
    CHECK_OK(s);
    return out;
}

std::unique_ptr<postings::ListIterator> Open(const Encoded& encoded, IteratorMode mode) {
    auto opened =
        postings::OpenListIterator(SourceOver(encoded.bytes), 0, encoded.bytes.size(), mode);
    CHECK(opened.ok());
    if (!opened.ok()) return nullptr;
    return opened.take();
}

/// The answer advance(target) must produce, computed the slowest, most obvious way.
DocId Expected(const std::vector<DocId>& docs, DocId from, DocId target) {
    // advance never moves backwards, so the search starts where the iterator is.
    const DocId effective = std::max(from, target);
    const auto it = std::lower_bound(docs.begin(), docs.end(), effective);
    return it == docs.end() ? kNoMoreDocs : *it;
}

}  // namespace

MERIDIAN_TEST(SkipEntryRoundTrips) {
    std::vector<uint8_t> table;
    const std::vector<SkipEntry> entries = {
        SkipEntry{0, 0, 0},
        SkipEntry{1, 12, 128},
        SkipEntry{UINT32_MAX - 1, UINT32_MAX, 1u << 20},
    };
    for (const SkipEntry& entry : entries) CHECK_OK(postings::AppendSkipEntry(entry, table));
    CHECK_EQ(table.size(), entries.size() * kSkipEntryBytes);

    for (uint32_t i = 0; i < entries.size(); ++i) {
        const SkipEntry read = postings::ReadSkipEntry(table, i);
        CHECK_EQ(read.doc, entries[i].doc);
        CHECK_EQ(read.byte_offset, entries[i].byte_offset);
        CHECK_EQ(read.postings_before, entries[i].postings_before);
    }
}

MERIDIAN_TEST(SkipEntryRejectsAnOffsetItCannotStore) {
    std::vector<uint8_t> table;
    SkipEntry entry;
    entry.byte_offset = uint64_t{UINT32_MAX} + 1;
    // Truncating silently would produce a skip that lands inside a varint, and the
    // document it then decodes exists — it is just not the right one.
    CHECK_NOT_OK(postings::AppendSkipEntry(entry, table));
}

MERIDIAN_TEST(FindSkipEntryPicksTheLastEntryAtOrBeforeTheTarget) {
    std::vector<uint8_t> table;
    const std::vector<DocId> block_first_docs = {10, 20, 20000, 20001, 90000};
    for (uint32_t i = 0; i < block_first_docs.size(); ++i) {
        CHECK_OK(postings::AppendSkipEntry(
            SkipEntry{block_first_docs[i], i * 100, i * kSkipInterval}, table));
    }
    const uint32_t count = static_cast<uint32_t>(block_first_docs.size());

    // Below the first entry there is nothing to skip to.
    CHECK_EQ(postings::FindSkipEntry(table, count, 0), -1);
    CHECK_EQ(postings::FindSkipEntry(table, count, 9), -1);

    // Exact hits land on their own entry, not the one before it.
    CHECK_EQ(postings::FindSkipEntry(table, count, 10), 0);
    CHECK_EQ(postings::FindSkipEntry(table, count, 20), 1);
    CHECK_EQ(postings::FindSkipEntry(table, count, 20001), 3);
    CHECK_EQ(postings::FindSkipEntry(table, count, 90000), 4);

    // A target between two entries belongs to the earlier block, because that is the
    // block that may contain it.
    CHECK_EQ(postings::FindSkipEntry(table, count, 11), 0);
    CHECK_EQ(postings::FindSkipEntry(table, count, 19999), 1);
    CHECK_EQ(postings::FindSkipEntry(table, count, 89999), 3);
    CHECK_EQ(postings::FindSkipEntry(table, count, UINT32_MAX), 4);

    // An empty table has no answer and must say so rather than index entry zero.
    CHECK_EQ(postings::FindSkipEntry(table, 0, 100), -1);
}

MERIDIAN_TEST(AdvanceAgreesWithSequentialScan) {
    // The headline correctness test. Randomised over list lengths, densities and target
    // patterns, comparing the skipping iterator against both the sequential iterator and
    // a std::lower_bound over the raw document vector.
    std::mt19937 rng(20240601);

    size_t lists_checked = 0, seeks_checked = 0;

    const std::vector<size_t> lengths = {1,   2,   kBlockPostings - 1,
                                         kBlockPostings,        kBlockPostings + 1,
                                         2 * kBlockPostings,    5 * kBlockPostings + 3,
                                         20 * kBlockPostings + 17};
    const std::vector<uint32_t> strides = {1, 2, 3, 97, 5000};

    for (size_t length : lengths) {
        for (uint32_t stride : strides) {
            for (bool with_positions : {false, true}) {
                std::vector<DocId> docs;
                docs.reserve(length);
                DocId doc = rng() % 7;
                for (size_t i = 0; i < length; ++i) {
                    docs.push_back(doc);
                    // Jitter so block boundaries do not line up with the stride, which is
                    // where a plain "first entry >= target" search happens to be right.
                    doc += stride + (rng() % std::max<uint32_t>(1, stride));
                }

                const Encoded encoded = Encode(docs, with_positions);
                ++lists_checked;

                // Fresh pair per target pattern: advance is monotone, so one pair cannot
                // test descending or repeated targets.
                const std::vector<int> patterns = {0, 1, 2, 3};
                for (int pattern : patterns) {
                    std::unique_ptr<postings::ListIterator> plain =
                        Open(encoded, IteratorMode::kSequential);
                    std::unique_ptr<postings::ListIterator> skipping =
                        Open(encoded, IteratorMode::kSkipping);
                    if (plain == nullptr || skipping == nullptr) return;

                    const DocId last = docs.back();
                    DocId target = 0;
                    for (int step = 0; step < 40; ++step) {
                        switch (pattern) {
                            case 0:  // uniformly random, so often behind the iterator
                                target = static_cast<DocId>(rng() % (last + 2));
                                break;
                            case 1:  // monotone jumps, the conjunctive-query pattern
                                target = static_cast<DocId>(
                                    std::min<uint64_t>(last + 1, uint64_t{target} + 1 +
                                                                     rng() % (stride * 40 + 1)));
                                break;
                            case 2:  // exactly on existing documents
                                target = docs[rng() % docs.size()];
                                break;
                            case 3:  // one past an existing document
                                target = static_cast<DocId>(
                                    std::min<uint64_t>(UINT32_MAX,
                                                       uint64_t{docs[rng() % docs.size()]} + 1));
                                break;
                            default:
                                break;
                        }

                        const DocId expected_plain = Expected(docs, plain->doc(), target);
                        const DocId expected_skip = Expected(docs, skipping->doc(), target);

                        const DocId got_plain = plain->advance(target);
                        const DocId got_skip = skipping->advance(target);
                        ++seeks_checked;

                        CHECK_EQ(got_plain, expected_plain);
                        CHECK_EQ(got_skip, expected_skip);
                        // The two iterators were advanced through the same target
                        // sequence from the same start, so they must be in the same place.
                        CHECK_EQ(got_skip, got_plain);
                        CHECK_OK(plain->error());
                        CHECK_OK(skipping->error());

                        if (got_skip != kNoMoreDocs) {
                            CHECK_EQ(skipping->term_frequency(), plain->term_frequency());
                            const auto a = skipping->positions();
                            const auto b = plain->positions();
                            CHECK_EQ(a.size(), b.size());
                            for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
                                CHECK_EQ(a[i], b[i]);
                            }
                            CHECK_EQ(skipping->size_hint(), plain->size_hint());
                        }

                        // Interleaving next() with advance() is what a real merge does,
                        // and it is where a stale within-block position shows up.
                        if (step % 3 == 0) {
                            CHECK_EQ(skipping->next(), plain->next());
                        }
                    }
                }
            }
        }
    }

    std::printf("  checked %zu seeks across %zu encoded lists\n", seeks_checked, lists_checked);
}

MERIDIAN_TEST(SkippingDecodesFewerPostingsOnASparseConjunction) {
    // The pattern the skip list exists for: `A AND B` where A is a common term with a
    // long list and B is rare. The intersection is driven by B, so A is only ever asked
    // about a few hundred widely separated documents.
    const size_t long_list_length = 200000;
    std::vector<DocId> long_docs;
    long_docs.reserve(long_list_length);
    for (size_t i = 0; i < long_list_length; ++i) long_docs.push_back(static_cast<DocId>(i * 2));

    std::vector<DocId> rare_docs;
    std::mt19937 rng(99);
    for (int i = 0; i < 200; ++i) {
        rare_docs.push_back(static_cast<DocId>(rng() % (long_list_length * 2)));
    }
    std::sort(rare_docs.begin(), rare_docs.end());
    rare_docs.erase(std::unique(rare_docs.begin(), rare_docs.end()), rare_docs.end());

    const Encoded encoded = Encode(long_docs, false);

    PostingIterator::Stats stats_by_mode[2];
    std::vector<DocId> landed[2];
    int index = 0;
    for (IteratorMode mode : {IteratorMode::kSequential, IteratorMode::kSkipping}) {
        std::unique_ptr<postings::ListIterator> it = Open(encoded, mode);
        if (it == nullptr) return;
        for (DocId target : rare_docs) {
            landed[index].push_back(it->advance(target));
        }
        CHECK_OK(it->error());
        stats_by_mode[index] = it->stats();
        ++index;
    }

    // Same answers, less work: both halves of the claim, in that order.
    CHECK(landed[0] == landed[1]);

    const PostingIterator::Stats& sequential = stats_by_mode[0];
    const PostingIterator::Stats& skipping = stats_by_mode[1];

    CHECK(skipping.postings_decoded < sequential.postings_decoded);
    CHECK(skipping.blocks_read < sequential.blocks_read);
    CHECK(skipping.skips_taken > 0);
    CHECK_EQ(sequential.skips_taken, 0u);

    const double ratio = static_cast<double>(sequential.postings_decoded) /
                         static_cast<double>(std::max<uint64_t>(1, skipping.postings_decoded));
    std::printf(
        "  %zu-posting list, %zu seeks: sequential decoded %llu postings in %llu blocks, "
        "skipping decoded %llu in %llu blocks taking %llu skips -> %.1fx fewer postings\n",
        long_docs.size(), rare_docs.size(),
        static_cast<unsigned long long>(sequential.postings_decoded),
        static_cast<unsigned long long>(sequential.blocks_read),
        static_cast<unsigned long long>(skipping.postings_decoded),
        static_cast<unsigned long long>(skipping.blocks_read),
        static_cast<unsigned long long>(skipping.skips_taken), ratio);

    // A skip can only ever save whole blocks, so the floor on what skipping decodes is
    // one block per distinct block touched. Asserting the order of magnitude rather than
    // an exact figure keeps the test from breaking when the block size is retuned.
    CHECK(ratio > 2.0);
}

MERIDIAN_TEST(CountersAreZeroedPerIteratorAndCountWholeBlocks) {
    std::vector<DocId> docs;
    for (DocId doc = 0; doc < 3 * kBlockPostings; ++doc) docs.push_back(doc);
    const Encoded encoded = Encode(docs, false);

    std::unique_ptr<postings::ListIterator> it = Open(encoded, IteratorMode::kSkipping);
    if (it == nullptr) return;

    // Opening positions on the first posting, which means the first block is decoded.
    // Stating that here rather than leaving it implicit: the counter is a measure of
    // decode work, and the open pays for one block of it.
    CHECK_EQ(it->stats().blocks_read, 1u);
    CHECK_EQ(it->stats().postings_decoded, kBlockPostings);

    // A seek into the last block adds exactly one more block, not two.
    it->advance(docs.back());
    CHECK_EQ(it->stats().blocks_read, 2u);
    CHECK_EQ(it->stats().postings_decoded, 2 * kBlockPostings);
    CHECK_EQ(it->stats().skips_taken, 1u);

    std::unique_ptr<postings::ListIterator> fresh = Open(encoded, IteratorMode::kSkipping);
    if (fresh == nullptr) return;
    CHECK_EQ(fresh->stats().skips_taken, 0u);
}

MERIDIAN_TEST(AdvanceNeverMovesBackwards) {
    std::vector<DocId> docs;
    for (DocId doc = 100; doc < 100 + 5 * kBlockPostings; doc += 2) docs.push_back(doc);
    const Encoded encoded = Encode(docs, false);

    for (IteratorMode mode : {IteratorMode::kSequential, IteratorMode::kSkipping}) {
        std::unique_ptr<postings::ListIterator> it = Open(encoded, mode);
        if (it == nullptr) return;

        const DocId far = docs[docs.size() / 2];
        CHECK_EQ(it->advance(far), far);
        // A leapfrog intersection re-advances to a target it has already passed on every
        // round. If that moved the iterator back, the loop would not terminate.
        CHECK_EQ(it->advance(0), far);
        CHECK_EQ(it->advance(far), far);
        CHECK_EQ(it->advance(docs.front()), far);
        CHECK_EQ(it->advance(kNoMoreDocs), kNoMoreDocs);
        CHECK_EQ(it->advance(docs.front()), kNoMoreDocs);
    }
}

MERIDIAN_TEST_MAIN()
