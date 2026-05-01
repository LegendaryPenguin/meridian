#include "skiplist.hpp"

#include <sstream>

#include "../storage/coding.hpp"

namespace meridian::postings {

Status AppendSkipEntry(const SkipEntry& entry, std::vector<uint8_t>& out) {
    if (entry.byte_offset > UINT32_MAX) {
        std::ostringstream os;
        os << "posting list block region exceeds 4 GiB: skip offset " << entry.byte_offset;
        return Status::InvalidArgument(os.str());
    }
    storage::PutU32(entry.doc, out);
    storage::PutU32(static_cast<uint32_t>(entry.byte_offset), out);
    storage::PutU32(entry.postings_before, out);
    return Status::Ok();
}

SkipEntry ReadSkipEntry(std::span<const uint8_t> table, uint32_t index) {
    const size_t base = static_cast<size_t>(index) * kSkipEntryBytes;
    SkipEntry entry;
    if (base + kSkipEntryBytes > table.size()) return entry;
    entry.doc = storage::GetU32(table, base);
    entry.byte_offset = storage::GetU32(table, base + 4);
    entry.postings_before = storage::GetU32(table, base + 8);
    return entry;
}

int64_t FindSkipEntry(std::span<const uint8_t> table, uint32_t count, DocId target) {
    if (count == 0) return -1;
    if (static_cast<size_t>(count) * kSkipEntryBytes > table.size()) return -1;

    // Invariant: everything at or below `lo` has doc <= target is *not* assumed; the
    // loop narrows [lo, hi) and keeps the best candidate found so far. Written this way
    // rather than as an upper_bound-minus-one because the subtraction is where the
    // off-by-one lives, and an explicit candidate is checkable by reading it.
    int64_t best = -1;
    uint32_t lo = 0, hi = count;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        const DocId doc = storage::GetU32(table, static_cast<size_t>(mid) * kSkipEntryBytes);
        if (doc <= target) {
            best = static_cast<int64_t>(mid);
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return best;
}

}  // namespace meridian::postings
