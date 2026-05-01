#pragma once

// The on-disk posting-list format.
//
// One list, for one (term, field), looks like this:
//
//   varint  num_postings
//   byte    flags            bit 0: positions present
//   varint  num_blocks
//   varint  skip_entries     num_blocks, or 0 for a single-block list
//   bytes   skip table       skip_entries * kSkipEntryBytes, fixed width
//   bytes   blocks           num_blocks framed blocks
//
// and one block is
//
//   varint  count
//   varint  payload_bytes
//   u32     crc32(payload)
//   bytes   payload          doc gaps, then frequencies, then positions
//
// Three decisions carry the format.
//
// *Blocks at all.* A skip has to land somewhere the decoder can start from, and in a
// single long delta-varint run the only such place is the beginning. Chunking the list
// and restarting the delta chain at each chunk boundary creates N legal entry points at
// the cost of storing one absolute document ID per block instead of a gap — a few bytes
// per 128 postings.
//
// *Checksum per block, not per list.* The unit of verification should be the unit of
// reading, for two reasons. A per-list checksum would force a seek to read and hash the
// entire list before touching one posting, which is exactly the cost skipping exists to
// avoid. And it would make detection all-or-nothing: with per-block checksums a corrupt
// block is identified, which is a fact a repair or a reindex-this-segment decision can
// use.
//
// *Checksum rather than nothing.* A flipped bit in a delta-varint run does not produce
// an obviously broken list. It produces a list whose document IDs are all plausible and
// all wrong from that point on, because every subsequent ID is a running sum. The
// result is a query that silently returns the wrong documents — the failure mode
// status.hpp was written to avoid. Verifying before decoding converts it into
// kCorruption.
//
// The payload is grouped by field (all gaps, then all frequencies, then all positions)
// rather than interleaved per posting. A conjunctive merge reads document IDs and only
// looks at a frequency once it has a match, and a phrase query only reads positions for
// documents that already matched on both terms; grouping means the common case touches
// a contiguous prefix of the payload instead of striding over it.

#include <cstdint>
#include <span>
#include <vector>

#include "meridian/postings.hpp"
#include "skiplist.hpp"

namespace meridian::postings {

/// Postings per block. Identical to the skip interval by construction, since the skip
/// table has one entry per block.
constexpr uint32_t kBlockPostings = kSkipInterval;

/// Upper bound on the list header: four varints of at most five bytes plus a flag byte.
/// Used to size the single read that opens an iterator.
constexpr size_t kMaxListHeaderBytes = 24;

struct ListHeader {
    uint32_t num_postings = 0;
    uint32_t num_blocks = 0;
    uint32_t skip_entries = 0;
    bool has_positions = false;

    /// Both relative to the first byte of the list, because a list is relocatable: the
    /// merge writer concatenates lists at whatever offsets fall out, and a self-relative
    /// list needs no rewriting.
    uint64_t skip_offset = 0;
    uint64_t blocks_offset = 0;
};

/// Encodes a posting list. `postings` must be strictly ascending by doc; an unsorted or
/// duplicated list is rejected here rather than corrupting every downstream merge.
///
/// When `with_positions` is false the positions are dropped, which is the per-field
/// size decision that Field exists to express.
Status EncodeList(std::span<const Posting> postings, bool with_positions,
                  std::vector<uint8_t>& out);

/// Parses the header out of a prefix of the list. The prefix may be shorter than
/// kMaxListHeaderBytes when the whole list is shorter.
Status ParseListHeader(std::span<const uint8_t> prefix, ListHeader& out);

/// One decoded block. Positions are kept as a flat array with a start index per posting
/// rather than a vector-of-vectors: a block holds at most 128 postings and allocating
/// 128 vectors to decode them would dominate the decode itself.
struct DecodedBlock {
    std::vector<DocId> docs;
    std::vector<uint32_t> term_frequencies;
    std::vector<uint32_t> position_starts;  // size() == docs.size() + 1
    std::vector<Position> positions;

    size_t count() const { return docs.size(); }

    std::span<const Position> positions_for(size_t i) const {
        if (position_starts.size() <= i + 1) return {};
        const uint32_t from = position_starts[i];
        const uint32_t to = position_starts[i + 1];
        return std::span<const Position>(positions.data() + from, to - from);
    }

    void Clear();
};

/// A block's framing, readable without touching the payload. `total_bytes` is what a
/// sequential walk advances by.
struct BlockFrame {
    uint32_t count = 0;
    uint32_t payload_bytes = 0;
    uint32_t crc = 0;
    size_t header_bytes = 0;
    size_t total_bytes() const { return header_bytes + payload_bytes; }
};

/// Largest a block frame header can be: two varints plus the checksum.
constexpr size_t kMaxBlockFrameBytes = 14;

Status ParseBlockFrame(std::span<const uint8_t> bytes, BlockFrame& out);

/// Verifies the checksum, then decodes. `bytes` must start at the block frame and
/// contain the whole block.
Status DecodeBlock(std::span<const uint8_t> bytes, bool has_positions, DecodedBlock& out);

}  // namespace meridian::postings
