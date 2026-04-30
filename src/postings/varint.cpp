#include "meridian/postings.hpp"

#include <sstream>

namespace meridian {

std::string Status::ToString() const {
    if (ok()) return "OK";
    const char* label = "error";
    switch (code_) {
        case Code::kQueryError:       label = "query error"; break;
        case Code::kNotFound:         label = "not found"; break;
        case Code::kNotImplemented:   label = "not implemented"; break;
        case Code::kIoError:          label = "io error"; break;
        case Code::kCorruption:       label = "corruption"; break;
        case Code::kInvalidArgument:  label = "invalid argument"; break;
        case Code::kDeadlineExceeded: label = "deadline exceeded"; break;
        case Code::kOverloaded:       label = "overloaded"; break;
        case Code::kInternal:         label = "internal error"; break;
        default: break;
    }
    return std::string(label) + ": " + message_;
}

namespace varint {

// LEB128: seven payload bits per byte, high bit set to mean "another byte follows".
//
// Chosen over a fixed width because posting gaps are overwhelmingly small. In a dense
// sorted list most consecutive document IDs differ by a handful, so the common gap
// fits in one byte where a fixed uint32 would spend four. The decode cost is a branch
// per byte, which is cheap next to the memory traffic a fixed-width list would add.
size_t Encode(uint64_t value, std::vector<uint8_t>& out) {
    size_t written = 0;
    while (value >= 0x80) {
        out.push_back(static_cast<uint8_t>(value) | 0x80);
        value >>= 7;
        ++written;
    }
    out.push_back(static_cast<uint8_t>(value));
    return written + 1;
}

bool Decode(std::span<const uint8_t> in, size_t& offset, uint64_t& value) {
    uint64_t result = 0;
    int shift = 0;

    while (offset < in.size()) {
        const uint8_t byte = in[offset++];
        // A 64-bit value needs at most ten 7-bit groups. Refusing past that is not
        // pedantry: without the bound, a corrupt run of continuation bytes shifts
        // indefinitely and decodes to an arbitrary document ID rather than failing.
        if (shift > 63) return false;

        result |= static_cast<uint64_t>(byte & 0x7F) << shift;
        if ((byte & 0x80) == 0) {
            value = result;
            return true;
        }
        shift += 7;
    }
    // Ran off the end mid-value: the list is truncated.
    return false;
}

Status EncodeSorted(std::span<const DocId> docs, std::vector<uint8_t>& out) {
    DocId previous = 0;
    for (size_t i = 0; i < docs.size(); ++i) {
        const DocId doc = docs[i];
        // Rejecting unsorted input here rather than tolerating it is deliberate. An
        // out-of-order posting makes every downstream merge silently wrong — a
        // conjunctive intersection would skip past documents that do match — and the
        // failure would surface as missing search results rather than as an error.
        if (i > 0 && doc <= previous) {
            std::ostringstream os;
            os << "posting list is not strictly ascending at index " << i << ": " << doc
               << " follows " << previous;
            return Status::InvalidArgument(os.str());
        }
        Encode(static_cast<uint64_t>(doc - previous), out);
        previous = doc;
    }
    return Status::Ok();
}

}  // namespace varint
}  // namespace meridian
