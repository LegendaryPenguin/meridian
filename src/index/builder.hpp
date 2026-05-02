#pragma once

// Builds one immutable segment file.
//
// The whole segment is assembled in memory and written once, atomically. That is a
// deliberate bound on segment size rather than an oversight: Options::docs_per_segment
// exists to keep a segment small enough that this is cheap, and in exchange the file
// either exists complete or does not exist at all, which is what lets the manifest
// rename be the only publication step. A streaming writer would have to deal with a
// half-written segment being discovered on restart.
//
// Layout, in write order:
//
//   postings       concatenated posting lists, in dictionary key order
//   dict blocks    front-coded term blocks
//   dict index     first key + offset + length per block; loaded whole at open
//   doc lengths    u32 per document, loaded whole at open (BM25 needs all of them)
//   doc offsets    u64 per document into the doc-meta region
//   doc meta       variable-length document records, read one at a time
//   footer         fixed size, at the very end
//
// Posting lists come first because they are the largest region and the only one whose
// size is not known until it is written; putting it first means every other region's
// offset is discovered in one pass instead of requiring a reservation. The footer is
// last and fixed-size so a reader can find everything from the end of the file without
// a sidecar: read kSegmentFooterBytes from the tail, check the magic at both ends of
// it, and every region is addressable.

#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "dictionary.hpp"
#include "meridian/index.hpp"
#include "meridian/postings.hpp"

namespace meridian::index {

constexpr uint64_t kSegmentMagic = 0x4D45524944534547ull;  // "MERIDSEG"
constexpr uint32_t kSegmentFormatVersion = 1;
constexpr size_t kSegmentFooterBytes = 124;

struct SegmentFooter {
    uint64_t postings_offset = 0;
    uint64_t postings_length = 0;
    uint64_t dict_blocks_offset = 0;
    uint64_t dict_blocks_length = 0;
    uint64_t dict_index_offset = 0;
    uint64_t dict_index_length = 0;
    uint64_t doc_lengths_offset = 0;  // length is document_count * 4
    uint64_t doc_offsets_offset = 0;  // length is document_count * 8
    uint64_t doc_meta_offset = 0;
    uint64_t doc_meta_length = 0;
    uint64_t term_count = 0;
    uint32_t document_count = 0;
    uint64_t total_tokens = 0;
};

void EncodeSegmentFooter(const SegmentFooter& footer, std::vector<uint8_t>& out);

/// Verifies both magics and the checksum before trusting a single offset. A footer that
/// is read without checking is a set of offsets that will be used to index into the
/// file, and a wrong one reads a different region's bytes as a posting list.
Status DecodeSegmentFooter(std::span<const uint8_t> bytes, SegmentFooter& out);

class SegmentBuilder {
public:
    struct Options {
        /// Indexed by Field. Positions roughly triple a field's posting size, so this is
        /// per field rather than global.
        bool positions[static_cast<size_t>(Field::kFieldCount)] = {true, true, false};
    };

    explicit SegmentBuilder(Options options) : options_(options) {}

    /// Appends a document and returns its segment-local id. Local ids are assigned
    /// densely from zero in insertion order, which is what makes the posting lists
    /// already-sorted by construction: every term of document n is added after every
    /// term of document n-1.
    DocId AddDocument(const DocumentMeta& meta);

    /// Records one term occurrence set for a document already added. `local` must not go
    /// backwards for a given term.
    Status AddTerm(Field field, std::string_view term, DocId local, uint32_t term_frequency,
                   std::span<const Position> positions);

    /// Convenience path for the indexing side: counts term frequencies and positions out
    /// of a token stream per field.
    Status AddTokenizedDocument(
        const DocumentMeta& meta,
        const std::unordered_map<Field, std::vector<std::string>>& tokens_by_field,
        DocId* local_out);

    /// Writes the segment to `path` and fills in the parts of SegmentMeta that are
    /// properties of the file. `id` and `doc_base` belong to the index, not the segment,
    /// and are left to the caller.
    Status Build(const std::string& path, SegmentMeta& meta_out);

    uint32_t document_count() const { return static_cast<uint32_t>(documents_.size()); }
    bool empty() const { return documents_.empty(); }
    uint64_t total_tokens() const { return total_tokens_; }
    const std::vector<std::string>& external_ids() const { return external_ids_; }

private:
    Options options_;

    /// Keyed by the dictionary key, so the sort that Build needs is a sort of this map's
    /// keys and nothing has to know about fields again.
    std::map<std::string, std::vector<Posting>> postings_;
    std::vector<DocumentMeta> documents_;
    std::vector<std::string> external_ids_;
    uint64_t total_tokens_ = 0;
};

}  // namespace meridian::index
