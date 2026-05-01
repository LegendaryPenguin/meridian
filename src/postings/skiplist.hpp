#pragma once

// The skip list: the table that lets advance(target) ignore most of a posting list.

#include <cstdint>
#include <span>
#include <vector>

#include "meridian/postings.hpp"

namespace meridian::postings {

/// Postings between skip entries, which is also the block size — one skip entry per
/// block, because a skip is only useful if it lands somewhere the decoder can start,
/// and the only such places are block boundaries.
///
/// N = 128. The tradeoff, with the numbers that decide it:
///
///   Index size. A skip entry is 12 bytes (below). At N=128 that is 0.094 bytes per
///   posting. A delta-varint posting in a dense list costs 1-2 bytes, so the skip
///   table is 5-9% of the list. At N=32 it would be 0.375 bytes per posting, 20-35% of
///   the list — paying a fifth of the index to save at most 96 decodes.
///
///   Seek granularity. A skip lands at the start of the block containing the target,
///   so the worst case is N-1 postings decoded that the query did not want: 127 at
///   N=128. Those 127 decodes are a few hundred nanoseconds against the ~100
///   microseconds a page fault on the next block would cost, so below about N=64 the
///   decode saving stops being measurable while the size cost keeps growing.
///
///   The other direction. At N=1024 a single skip wastes up to 1023 decodes, which is
///   more work than scanning several blocks sequentially would have been — the skip
///   stops paying for itself.
///
/// 128 also makes a block's doc-ID array 128 uint32s = 512 bytes when materialised,
/// which is eight cache lines and fits comfortably in L1 alongside the frequencies.
constexpr uint32_t kSkipInterval = 128;

/// A skip entry on disk: doc (u32), byte offset (u32), postings before (u32).
///
/// Fixed width, not varint, and this is the reason there is no second skip level.
///
/// The textbook multi-level skip list exists because its skip data is itself
/// variable-width and therefore has to be *scanned* to be read: finding the right entry
/// in a 78,000-entry table for a ten-million-posting list means decoding 78,000
/// varints, which costs more than the postings the skip was meant to avoid. A second
/// level is a skip list over the skip list, and it exists to make that scan shorter.
///
/// A fixed-width table does not have that problem. Entry i is at offset 12*i, so the
/// table is a sorted array and the search is a binary search *in place* with no decode
/// at all: 17 comparisons for those same 78,000 entries, touching 17 cache lines. A
/// second level would reduce 17 comparisons to maybe 10 while adding a second format,
/// a second set of offsets, and a second chance to get the boundary conditions wrong.
/// The price is 12 bytes per entry instead of ~6; at N=128 that is 0.05 bytes per
/// posting, which buys a great deal of simplicity.
///
/// The u32 offset caps one posting list's block region at 4 GiB. The encoder checks it
/// rather than truncating, because a wrapped offset is a skip that lands in the middle
/// of a varint and decodes to a document that exists but is wrong.
constexpr size_t kSkipEntryBytes = 12;

/// Appends one entry. Fails if the offset or posting count does not fit, which is the
/// only way this format can be outgrown.
Status AppendSkipEntry(const SkipEntry& entry, std::vector<uint8_t>& out);

/// Entry `index` of a table. The caller has already bounds-checked `index` against the
/// entry count from the list header; the table span is checked here.
SkipEntry ReadSkipEntry(std::span<const uint8_t> table, uint32_t index);

/// Index of the last entry whose doc is <= target, or -1 when target precedes the
/// first entry.
///
/// "Last entry <= target" rather than "first entry >= target" is the whole correctness
/// argument for skipping: the target may be *inside* the block that entry describes, so
/// landing on the first block whose first doc is already >= target would step over it.
/// That is the off-by-one the differential test against the sequential iterator exists
/// to catch, because it produces results that are merely slightly wrong.
int64_t FindSkipEntry(std::span<const uint8_t> table, uint32_t count, DocId target);

}  // namespace meridian::postings
