// Properties of the posting-list format: that a round trip is lossless, that the
// encoder refuses input it cannot represent faithfully, and that damaged bytes are
// detected rather than decoded.

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "../src/postings/encoder.hpp"
#include "../src/postings/iterator.hpp"
#include "../src/storage/checksum.hpp"
#include "../src/storage/file.hpp"
#include "meridian/postings.hpp"
#include "testing.hpp"

using namespace meridian;
using meridian::postings::DecodedBlock;
using meridian::postings::IteratorMode;
using meridian::postings::kBlockPostings;

namespace {

std::shared_ptr<const storage::ByteSource> SourceOver(std::vector<uint8_t> bytes) {
    return std::make_shared<storage::MemoryByteSource>(std::move(bytes));
}

struct Walked {
    std::vector<DocId> docs;
    std::vector<uint32_t> term_frequencies;
    std::vector<std::vector<Position>> positions;
};

Walked WalkAll(postings::ListIterator& it) {
    Walked out;
    for (DocId doc = it.doc(); doc != kNoMoreDocs; doc = it.next()) {
        out.docs.push_back(doc);
        out.term_frequencies.push_back(it.term_frequency());
        const auto span = it.positions();
        out.positions.emplace_back(span.begin(), span.end());
    }
    return out;
}

/// Round-trips a list through both iterator implementations and checks every field.
void CheckRoundTrip(const std::vector<Posting>& postings, bool with_positions) {
    std::vector<uint8_t> encoded;
    CHECK_OK(postings::EncodeList(postings, with_positions, encoded));

    for (IteratorMode mode : {IteratorMode::kSequential, IteratorMode::kSkipping}) {
        auto opened = postings::OpenListIterator(SourceOver(encoded), 0, encoded.size(), mode);
        CHECK(opened.ok());
        if (!opened.ok()) return;
        std::unique_ptr<postings::ListIterator> it = opened.take();

        const Walked walked = WalkAll(*it);
        CHECK_OK(it->error());
        CHECK_EQ(walked.docs.size(), postings.size());
        if (walked.docs.size() != postings.size()) return;
        for (size_t i = 0; i < postings.size(); ++i) {
            CHECK_EQ(walked.docs[i], postings[i].doc);
            CHECK_EQ(walked.term_frequencies[i], postings[i].term_frequency);
            if (with_positions) {
                CHECK(walked.positions[i] == postings[i].positions);
            } else {
                CHECK(walked.positions[i].empty());
            }
        }
    }
}

Posting MakePosting(DocId doc, uint32_t tf, std::vector<Position> positions) {
    Posting p;
    p.doc = doc;
    p.term_frequency = tf;
    p.positions = std::move(positions);
    return p;
}

}  // namespace

MERIDIAN_TEST(VarintRoundTripsTheBoundaryValues) {
    const std::vector<uint64_t> values = {0,
                                          1,
                                          127,
                                          128,
                                          129,
                                          16383,
                                          16384,
                                          UINT32_MAX - 1,
                                          UINT32_MAX,
                                          uint64_t{UINT32_MAX} + 1,
                                          UINT64_MAX - 1,
                                          UINT64_MAX};
    for (uint64_t value : values) {
        std::vector<uint8_t> encoded;
        const size_t written = varint::Encode(value, encoded);
        CHECK_EQ(written, encoded.size());

        size_t offset = 0;
        uint64_t decoded = 0;
        CHECK(varint::Decode(encoded, offset, decoded));
        CHECK_EQ(decoded, value);
        CHECK_EQ(offset, encoded.size());
    }
}

MERIDIAN_TEST(VarintPacksManyValuesBackToBack) {
    // The decoder has to advance the offset by exactly the right amount, which a single
    // value cannot show: a one-byte over-read is invisible until something follows.
    std::vector<uint64_t> values;
    std::mt19937_64 rng(7);
    for (int i = 0; i < 500; ++i) values.push_back(rng() >> (i % 64));

    std::vector<uint8_t> encoded;
    for (uint64_t v : values) varint::Encode(v, encoded);

    size_t offset = 0;
    for (uint64_t expected : values) {
        uint64_t decoded = 0;
        CHECK(varint::Decode(encoded, offset, decoded));
        CHECK_EQ(decoded, expected);
    }
    CHECK_EQ(offset, encoded.size());
}

MERIDIAN_TEST(VarintRejectsATruncatedSequence) {
    // A lone continuation byte: the value is not there, and returning 0 with success
    // would hand a caller a document ID of zero.
    const std::vector<uint8_t> truncated = {0x80};
    size_t offset = 0;
    uint64_t value = 1234;
    CHECK(!varint::Decode(truncated, offset, value));

    // Truncation in the middle of a long value.
    std::vector<uint8_t> encoded;
    varint::Encode(UINT64_MAX, encoded);
    encoded.pop_back();
    offset = 0;
    CHECK(!varint::Decode(encoded, offset, value));
}

MERIDIAN_TEST(VarintRejectsAnOverLongContinuationRun) {
    // Ten groups is the most a 64-bit value can need. An eleventh means the bytes are
    // not a varint, and without the bound the shift runs off the top of the word and
    // produces a plausible number.
    std::vector<uint8_t> over_long(10, 0xFF);
    over_long.push_back(0x01);
    size_t offset = 0;
    uint64_t value = 0;
    CHECK(!varint::Decode(over_long, offset, value));

    // A long run of pure continuation bytes, which is what a corrupt region of 0xFF
    // looks like, must fail rather than loop.
    const std::vector<uint8_t> all_continuations(64, 0xFF);
    offset = 0;
    CHECK(!varint::Decode(all_continuations, offset, value));
}

MERIDIAN_TEST(EncodeSortedRejectsUnsortedAndDuplicateInput) {
    std::vector<uint8_t> out;
    const std::vector<DocId> ascending = {0, 1, 5, 1000, 1001};
    CHECK_OK(varint::EncodeSorted(ascending, out));

    out.clear();
    const std::vector<DocId> unsorted = {0, 5, 4};
    CHECK_NOT_OK(varint::EncodeSorted(unsorted, out));

    out.clear();
    const std::vector<DocId> duplicated = {3, 7, 7};
    CHECK_NOT_OK(varint::EncodeSorted(duplicated, out));

    out.clear();
    const std::vector<DocId> empty;
    CHECK_OK(varint::EncodeSorted(empty, out));
    CHECK(out.empty());
}

MERIDIAN_TEST(EncodeListRejectsUnsortedAndDuplicatePostings) {
    std::vector<uint8_t> out;
    const std::vector<Posting> unsorted = {MakePosting(5, 1, {}), MakePosting(2, 1, {})};
    CHECK_NOT_OK(postings::EncodeList(unsorted, false, out));

    out.clear();
    const std::vector<Posting> duplicated = {MakePosting(5, 1, {}), MakePosting(5, 1, {})};
    CHECK_NOT_OK(postings::EncodeList(duplicated, false, out));
}

MERIDIAN_TEST(PostingListRoundTripsWithoutPositions) {
    // A term in exactly one document: the degenerate list, and the one that exposes an
    // encoder that assumes a gap exists to delta against.
    CheckRoundTrip({MakePosting(0, 1, {})}, false);
    CheckRoundTrip({MakePosting(999999, 3, {})}, false);

    // A term in every document, which is the dense case the gap encoding is tuned for
    // and which crosses several block boundaries.
    std::vector<Posting> dense;
    for (DocId doc = 0; doc < 5 * kBlockPostings + 7; ++doc) {
        dense.push_back(MakePosting(doc, 1 + (doc % 4), {}));
    }
    CheckRoundTrip(dense, false);

    // A sparse list, where the gaps need more than one byte each.
    std::vector<Posting> sparse;
    for (DocId doc = 0; doc < 400; ++doc) {
        sparse.push_back(MakePosting(doc * 50021, doc + 1, {}));
    }
    CheckRoundTrip(sparse, false);
}

MERIDIAN_TEST(PostingListRoundTripsWithPositions) {
    std::vector<Posting> list;
    list.push_back(MakePosting(0, 3, {0, 5, 17}));
    // Empty positions with a non-zero frequency is legal: the frequency came from a
    // field that is indexed without positions, or from a counter that ran ahead of the
    // position list. Either way it must survive a round trip rather than be normalised.
    list.push_back(MakePosting(1, 2, {}));
    list.push_back(MakePosting(7, 1, {1000000}));
    list.push_back(MakePosting(8, 4, {0, 1, 2, 3}));
    CheckRoundTrip(list, true);

    // Across block boundaries, with varying position counts so the per-posting count
    // varint is exercised at both ends of its range.
    std::vector<Posting> many;
    for (DocId doc = 0; doc < 3 * kBlockPostings + 1; ++doc) {
        std::vector<Position> positions;
        for (uint32_t i = 0; i < (doc % 5); ++i) positions.push_back(doc * 3 + i);
        many.push_back(MakePosting(doc * 2, static_cast<uint32_t>(positions.size()), positions));
    }
    CheckRoundTrip(many, true);
}

MERIDIAN_TEST(PostingListDropsPositionsWhenTheFieldDoesNotWantThem) {
    std::vector<Posting> list;
    list.push_back(MakePosting(4, 2, {1, 9}));
    list.push_back(MakePosting(6, 1, {3}));

    std::vector<uint8_t> with_positions, without_positions;
    CHECK_OK(postings::EncodeList(list, true, with_positions));
    CHECK_OK(postings::EncodeList(list, false, without_positions));
    // The point of the per-field decision is that it costs bytes. If it did not, there
    // would be nothing to decide.
    CHECK(without_positions.size() < with_positions.size());

    CheckRoundTrip(list, false);
}

MERIDIAN_TEST(EmptyPostingListIsReadableAndImmediatelyExhausted) {
    std::vector<uint8_t> encoded;
    CHECK_OK(postings::EncodeList({}, false, encoded));

    auto opened = postings::OpenListIterator(SourceOver(encoded), 0, encoded.size(),
                                             IteratorMode::kSkipping);
    CHECK(opened.ok());
    if (!opened.ok()) return;
    std::unique_ptr<postings::ListIterator> it = opened.take();
    CHECK_EQ(it->doc(), kNoMoreDocs);
    CHECK_EQ(it->next(), kNoMoreDocs);
    CHECK_EQ(it->advance(0), kNoMoreDocs);
    CHECK_EQ(it->size_hint(), 0u);
}

MERIDIAN_TEST(ShortListCarriesNoSkipTable) {
    // The long tail of a term dictionary is lists of a handful of postings, and twelve
    // bytes of skip table on a three-byte list would be the dominant cost.
    std::vector<Posting> few;
    for (DocId doc = 0; doc < 10; ++doc) few.push_back(MakePosting(doc, 1, {}));

    std::vector<uint8_t> encoded;
    CHECK_OK(postings::EncodeList(few, false, encoded));

    postings::ListHeader header;
    CHECK_OK(postings::ParseListHeader(encoded, header));
    CHECK_EQ(header.num_blocks, 1u);
    CHECK_EQ(header.skip_entries, 0u);
    CHECK_EQ(header.skip_offset, header.blocks_offset);
}

MERIDIAN_TEST(LongListCarriesOneSkipEntryPerBlock) {
    std::vector<Posting> list;
    for (DocId doc = 0; doc < 4 * kBlockPostings + 1; ++doc) list.push_back(MakePosting(doc, 1, {}));

    std::vector<uint8_t> encoded;
    CHECK_OK(postings::EncodeList(list, false, encoded));

    postings::ListHeader header;
    CHECK_OK(postings::ParseListHeader(encoded, header));
    CHECK_EQ(header.num_blocks, 5u);
    CHECK_EQ(header.skip_entries, 5u);
    CHECK_EQ(header.blocks_offset - header.skip_offset,
             5 * meridian::postings::kSkipEntryBytes);
}

MERIDIAN_TEST(CorruptBlockChecksumIsDetected) {
    std::vector<Posting> list;
    for (DocId doc = 0; doc < 3 * kBlockPostings; ++doc) {
        list.push_back(MakePosting(doc * 3, 1 + (doc % 7), {}));
    }
    std::vector<uint8_t> encoded;
    CHECK_OK(postings::EncodeList(list, false, encoded));

    postings::ListHeader header;
    CHECK_OK(postings::ParseListHeader(encoded, header));

    // A single flipped bit in the second block's payload. Decoded without verification
    // this does not fail: every document ID is a running sum, so the list stays
    // ascending and merely becomes wrong.
    const size_t victim = static_cast<size_t>(header.blocks_offset) + 40;
    CHECK(victim < encoded.size());

    std::vector<uint8_t> damaged = encoded;
    damaged[victim] = static_cast<uint8_t>(damaged[victim] ^ 0x08);
    CHECK(damaged != encoded);

    // Found at the block level.
    bool any_block_failed = false;
    {
        size_t offset = static_cast<size_t>(header.blocks_offset);
        DecodedBlock block;
        for (uint32_t i = 0; i < header.num_blocks && offset < damaged.size(); ++i) {
            const std::span<const uint8_t> rest(damaged.data() + offset, damaged.size() - offset);
            postings::BlockFrame frame;
            if (!postings::ParseBlockFrame(rest, frame).ok()) {
                any_block_failed = true;
                break;
            }
            if (!postings::DecodeBlock(rest, header.has_positions, block).ok()) {
                any_block_failed = true;
                break;
            }
            offset += frame.total_bytes();
        }
    }
    CHECK(any_block_failed);

    // And surfaced through the iterator, which is where it actually matters: a walk must
    // not simply end early and look like a short list.
    auto opened = postings::OpenListIterator(SourceOver(damaged), 0, damaged.size(),
                                             IteratorMode::kSequential);
    bool reported = false;
    if (!opened.ok()) {
        reported = opened.status().code() == Code::kCorruption;
    } else {
        std::unique_ptr<postings::ListIterator> it = opened.take();
        WalkAll(*it);
        reported = !it->error().ok() && it->error().code() == Code::kCorruption;

        // And through the base interface, which is the form every consumer of a posting
        // list actually holds. The reason was always latched here; until `status()` was on
        // `PostingIterator` a caller could not reach it, so the query layer walked to
        // exhaustion and reported success on a truncated list.
        const PostingIterator& base = *it;
        CHECK(!base.status().ok());
        CHECK(base.status().code() == Code::kCorruption);
    }
    CHECK(reported);
}

MERIDIAN_TEST(TruncatedPostingListIsRejected) {
    std::vector<Posting> list;
    for (DocId doc = 0; doc < 2 * kBlockPostings; ++doc) list.push_back(MakePosting(doc, 1, {}));
    std::vector<uint8_t> encoded;
    CHECK_OK(postings::EncodeList(list, false, encoded));

    std::vector<uint8_t> truncated(encoded.begin(), encoded.end() - 20);
    auto opened = postings::OpenListIterator(SourceOver(truncated), 0, truncated.size(),
                                             IteratorMode::kSkipping);
    bool failed = !opened.ok();
    if (!failed) {
        std::unique_ptr<postings::ListIterator> it = opened.take();
        WalkAll(*it);
        failed = !it->error().ok();
    }
    CHECK(failed);
}

MERIDIAN_TEST(DeltaVarintBeatsFixedWidthOnADenseList) {
    // The claim the format rests on, measured rather than asserted.
    std::vector<Posting> dense;
    for (DocId doc = 0; doc < 100000; ++doc) dense.push_back(MakePosting(doc * 3, 1, {}));

    std::vector<uint8_t> encoded;
    CHECK_OK(postings::EncodeList(dense, false, encoded));

    const size_t fixed_width = dense.size() * (sizeof(DocId) + sizeof(uint32_t));
    std::printf("  dense list: %zu postings, %zu bytes encoded vs %zu fixed-width (%.2fx)\n",
                dense.size(), encoded.size(), fixed_width,
                static_cast<double>(fixed_width) / static_cast<double>(encoded.size()));
    CHECK(encoded.size() < fixed_width);
}

MERIDIAN_TEST_MAIN()
