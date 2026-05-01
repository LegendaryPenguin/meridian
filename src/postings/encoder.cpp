#include "encoder.hpp"

#include <algorithm>
#include <sstream>

#include "../storage/checksum.hpp"
#include "../storage/coding.hpp"

namespace meridian::postings {
namespace {

constexpr uint8_t kFlagPositions = 0x01;

/// Encodes one block's payload. Grouped by field; see the header comment for why.
Status EncodeBlockPayload(std::span<const Posting> block, bool with_positions,
                          std::vector<uint8_t>& payload) {
    DocId previous = 0;
    for (size_t i = 0; i < block.size(); ++i) {
        // The first document of every block is absolute. That is what makes the block a
        // legal entry point for a skip: the delta chain restarts here, so nothing
        // earlier in the list has to be decoded first.
        const uint64_t gap = (i == 0) ? block[0].doc : (block[i].doc - previous);
        varint::Encode(gap, payload);
        previous = block[i].doc;
    }

    for (const Posting& p : block) varint::Encode(p.term_frequency, payload);

    if (!with_positions) return Status::Ok();

    for (const Posting& p : block) {
        varint::Encode(p.positions.size(), payload);
        Position previous_position = 0;
        for (size_t i = 0; i < p.positions.size(); ++i) {
            const Position position = p.positions[i];
            if (i > 0 && position < previous_position) {
                std::ostringstream os;
                os << "positions for doc " << p.doc << " are not ascending: " << position
                   << " follows " << previous_position;
                return Status::InvalidArgument(os.str());
            }
            // Positions are delta-encoded within the document for the same reason
            // document IDs are across it: a term's occurrences cluster, so the gaps are
            // small even when the absolute offsets are not.
            varint::Encode(i == 0 ? position : (position - previous_position), payload);
            previous_position = position;
        }
    }
    return Status::Ok();
}

}  // namespace

void DecodedBlock::Clear() {
    docs.clear();
    term_frequencies.clear();
    position_starts.clear();
    positions.clear();
}

Status EncodeList(std::span<const Posting> postings, bool with_positions,
                  std::vector<uint8_t>& out) {
    for (size_t i = 1; i < postings.size(); ++i) {
        if (postings[i].doc <= postings[i - 1].doc) {
            std::ostringstream os;
            os << "posting list is not strictly ascending at index " << i << ": "
               << postings[i].doc << " follows " << postings[i - 1].doc;
            return Status::InvalidArgument(os.str());
        }
    }
    if (postings.size() > UINT32_MAX) {
        return Status::InvalidArgument("posting list longer than a DocId can address");
    }

    const uint32_t num_postings = static_cast<uint32_t>(postings.size());
    const uint32_t num_blocks =
        (num_postings + kBlockPostings - 1) / kBlockPostings;  // 0 postings -> 0 blocks

    std::vector<uint8_t> blocks;
    std::vector<uint8_t> skip_table;
    std::vector<uint8_t> payload;

    for (uint32_t b = 0; b < num_blocks; ++b) {
        const size_t start = static_cast<size_t>(b) * kBlockPostings;
        const size_t count = std::min<size_t>(kBlockPostings, num_postings - start);
        const std::span<const Posting> slice = postings.subspan(start, count);

        SkipEntry entry;
        entry.doc = slice.front().doc;
        entry.byte_offset = blocks.size();
        entry.postings_before = static_cast<uint32_t>(start);
        MERIDIAN_TRY(AppendSkipEntry(entry, skip_table));

        payload.clear();
        MERIDIAN_TRY(EncodeBlockPayload(slice, with_positions, payload));

        varint::Encode(count, blocks);
        varint::Encode(payload.size(), blocks);
        storage::PutU32(storage::Crc32(payload), blocks);
        blocks.insert(blocks.end(), payload.begin(), payload.end());
    }

    // A single-block list carries no skip table: there is nowhere to skip to, and 12
    // bytes of table on a list of three postings is larger than the postings. The long
    // tail of a term dictionary is overwhelmingly these.
    const bool emit_skips = num_blocks >= 2;

    varint::Encode(num_postings, out);
    out.push_back(with_positions ? kFlagPositions : 0);
    varint::Encode(num_blocks, out);
    varint::Encode(emit_skips ? num_blocks : 0, out);
    if (emit_skips) out.insert(out.end(), skip_table.begin(), skip_table.end());
    out.insert(out.end(), blocks.begin(), blocks.end());
    return Status::Ok();
}

Status ParseListHeader(std::span<const uint8_t> prefix, ListHeader& out) {
    size_t offset = 0;
    uint32_t num_postings = 0, num_blocks = 0, skip_entries = 0;
    if (!storage::GetVarint32(prefix, offset, num_postings)) {
        return Status::Corruption("posting list header is truncated at num_postings");
    }
    if (offset >= prefix.size()) {
        return Status::Corruption("posting list header is truncated at flags");
    }
    const uint8_t flags = prefix[offset++];
    if (!storage::GetVarint32(prefix, offset, num_blocks)) {
        return Status::Corruption("posting list header is truncated at num_blocks");
    }
    if (!storage::GetVarint32(prefix, offset, skip_entries)) {
        return Status::Corruption("posting list header is truncated at skip_entries");
    }

    const uint32_t expected_blocks =
        (num_postings + kBlockPostings - 1) / kBlockPostings;
    if (num_blocks != expected_blocks) {
        std::ostringstream os;
        os << "posting list claims " << num_blocks << " blocks for " << num_postings
           << " postings; " << expected_blocks << " expected";
        return Status::Corruption(os.str());
    }
    if (skip_entries != 0 && skip_entries != num_blocks) {
        std::ostringstream os;
        os << "posting list has " << skip_entries << " skip entries for " << num_blocks
           << " blocks";
        return Status::Corruption(os.str());
    }

    out.num_postings = num_postings;
    out.num_blocks = num_blocks;
    out.skip_entries = skip_entries;
    out.has_positions = (flags & kFlagPositions) != 0;
    out.skip_offset = offset;
    out.blocks_offset = offset + static_cast<uint64_t>(skip_entries) * kSkipEntryBytes;
    return Status::Ok();
}

Status ParseBlockFrame(std::span<const uint8_t> bytes, BlockFrame& out) {
    size_t offset = 0;
    uint32_t count = 0, payload_bytes = 0;
    if (!storage::GetVarint32(bytes, offset, count)) {
        return Status::Corruption("posting block frame is truncated at count");
    }
    if (!storage::GetVarint32(bytes, offset, payload_bytes)) {
        return Status::Corruption("posting block frame is truncated at length");
    }
    if (offset + 4 > bytes.size()) {
        return Status::Corruption("posting block frame is truncated at checksum");
    }
    if (count == 0 || count > kBlockPostings) {
        std::ostringstream os;
        os << "posting block claims " << count << " postings, limit is " << kBlockPostings;
        return Status::Corruption(os.str());
    }
    out.count = count;
    out.payload_bytes = payload_bytes;
    out.crc = storage::GetU32(bytes, offset);
    out.header_bytes = offset + 4;
    return Status::Ok();
}

Status DecodeBlock(std::span<const uint8_t> bytes, bool has_positions, DecodedBlock& out) {
    BlockFrame frame;
    MERIDIAN_TRY(ParseBlockFrame(bytes, frame));
    if (frame.total_bytes() > bytes.size()) {
        std::ostringstream os;
        os << "posting block needs " << frame.total_bytes() << " bytes, only "
           << bytes.size() << " available";
        return Status::Corruption(os.str());
    }

    const std::span<const uint8_t> payload =
        bytes.subspan(frame.header_bytes, frame.payload_bytes);

    // Verified before any of it is interpreted. Decoding first and checking afterwards
    // would be cheaper on the happy path and useless: by then the caller already has
    // document IDs it believes.
    const uint32_t actual = storage::Crc32(payload);
    if (actual != frame.crc) {
        std::ostringstream os;
        os << "posting block checksum mismatch: stored " << frame.crc << ", computed "
           << actual << " over " << frame.payload_bytes << " bytes";
        return Status::Corruption(os.str());
    }

    out.Clear();
    out.docs.reserve(frame.count);
    out.term_frequencies.reserve(frame.count);

    size_t offset = 0;
    DocId previous = 0;
    for (uint32_t i = 0; i < frame.count; ++i) {
        uint64_t gap = 0;
        if (!varint::Decode(payload, offset, gap)) {
            return Status::Corruption("posting block truncated in document gaps");
        }
        const uint64_t doc = (i == 0) ? gap : static_cast<uint64_t>(previous) + gap;
        if (doc > UINT32_MAX) {
            return Status::Corruption("posting block decodes a document id past DocId range");
        }
        if (i > 0 && doc <= previous) {
            // Only reachable on a gap of zero, which the encoder never writes. Checked
            // anyway because the merge loops above assume strict ascent and would
            // silently mis-intersect rather than fail.
            return Status::Corruption("posting block document ids are not ascending");
        }
        previous = static_cast<DocId>(doc);
        out.docs.push_back(previous);
    }

    for (uint32_t i = 0; i < frame.count; ++i) {
        uint32_t tf = 0;
        if (!storage::GetVarint32(payload, offset, tf)) {
            return Status::Corruption("posting block truncated in term frequencies");
        }
        out.term_frequencies.push_back(tf);
    }

    out.position_starts.assign(1, 0);
    if (has_positions) {
        out.position_starts.reserve(frame.count + 1);
        for (uint32_t i = 0; i < frame.count; ++i) {
            uint32_t n = 0;
            if (!storage::GetVarint32(payload, offset, n)) {
                return Status::Corruption("posting block truncated in position counts");
            }
            Position previous_position = 0;
            for (uint32_t j = 0; j < n; ++j) {
                uint32_t delta = 0;
                if (!storage::GetVarint32(payload, offset, delta)) {
                    return Status::Corruption("posting block truncated in positions");
                }
                previous_position = (j == 0) ? delta : (previous_position + delta);
                out.positions.push_back(previous_position);
            }
            out.position_starts.push_back(static_cast<uint32_t>(out.positions.size()));
        }
    } else {
        // Without positions every posting has an empty range, so one shared end marker
        // per posting keeps positions_for() branch-free for the caller.
        out.position_starts.assign(frame.count + 1, 0);
    }

    if (offset != payload.size()) {
        std::ostringstream os;
        os << "posting block has " << (payload.size() - offset)
           << " trailing bytes after decoding " << frame.count << " postings";
        return Status::Corruption(os.str());
    }
    return Status::Ok();
}

}  // namespace meridian::postings
