#pragma once

// Fixed-width and length-prefixed primitives shared by the posting encoder, the term
// dictionary and the segment footer.
//
// Everything is little-endian explicitly rather than by memcpy of a native integer.
// A segment file is an interchange format even when both ends are the same process:
// the merge path reads files written by an earlier run, and a format that silently
// depends on the host's byte order is a format that produces garbage document IDs the
// first time it is read somewhere else.

#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "meridian/postings.hpp"

namespace meridian::storage {

inline void PutU32(uint32_t v, std::vector<uint8_t>& out) {
    out.push_back(static_cast<uint8_t>(v));
    out.push_back(static_cast<uint8_t>(v >> 8));
    out.push_back(static_cast<uint8_t>(v >> 16));
    out.push_back(static_cast<uint8_t>(v >> 24));
}

inline void PutU64(uint64_t v, std::vector<uint8_t>& out) {
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<uint8_t>(v >> (8 * i)));
}

inline uint32_t GetU32(std::span<const uint8_t> in, size_t offset) {
    return static_cast<uint32_t>(in[offset]) | (static_cast<uint32_t>(in[offset + 1]) << 8) |
           (static_cast<uint32_t>(in[offset + 2]) << 16) |
           (static_cast<uint32_t>(in[offset + 3]) << 24);
}

inline uint64_t GetU64(std::span<const uint8_t> in, size_t offset) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(in[offset + i]) << (8 * i);
    return v;
}

inline void PutVarint(uint64_t v, std::vector<uint8_t>& out) { varint::Encode(v, out); }

/// Reads a varint, returning false on truncation so the caller can report corruption
/// instead of continuing with a zero.
inline bool GetVarint(std::span<const uint8_t> in, size_t& offset, uint64_t& out) {
    return varint::Decode(in, offset, out);
}

inline bool GetVarint32(std::span<const uint8_t> in, size_t& offset, uint32_t& out) {
    uint64_t wide = 0;
    if (!varint::Decode(in, offset, wide)) return false;
    if (wide > UINT32_MAX) return false;
    out = static_cast<uint32_t>(wide);
    return true;
}

inline void PutString(std::string_view s, std::vector<uint8_t>& out) {
    PutVarint(s.size(), out);
    out.insert(out.end(), s.begin(), s.end());
}

inline bool GetString(std::span<const uint8_t> in, size_t& offset, std::string& out) {
    uint64_t length = 0;
    if (!varint::Decode(in, offset, length)) return false;
    if (length > in.size() - offset) return false;
    out.assign(reinterpret_cast<const char*>(in.data()) + offset, length);
    offset += length;
    return true;
}

/// Doubles go over the wire as their IEEE-754 bit pattern. Formatting them as text
/// would lose the last digit, and a popularity score that changes when it round-trips
/// makes a ranking regression impossible to reproduce.
inline void PutDouble(double v, std::vector<uint8_t>& out) {
    uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    PutU64(bits, out);
}

inline double GetDouble(std::span<const uint8_t> in, size_t offset) {
    const uint64_t bits = GetU64(in, offset);
    double v = 0;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

/// Signed values are zigzagged before varint encoding: a timestamp of -1 would
/// otherwise set the top bit and cost ten bytes.
inline uint64_t ZigZag(int64_t v) {
    return (static_cast<uint64_t>(v) << 1) ^ static_cast<uint64_t>(v >> 63);
}

inline int64_t UnZigZag(uint64_t v) {
    return static_cast<int64_t>(v >> 1) ^ -static_cast<int64_t>(v & 1);
}

}  // namespace meridian::storage
