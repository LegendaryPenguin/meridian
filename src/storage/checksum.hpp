#pragma once

// A checksum, kept separate from the file layer because both the posting-block
// encoder and the segment footer need it and neither should have to know about
// file descriptors to get it.

#include <cstdint>
#include <span>

namespace meridian::storage {

/// CRC-32 (the IEEE 802.3 reflected polynomial, as used by zlib and gzip).
///
/// Chosen over a cheaper sum because the failure it has to catch is a *plausible*
/// corruption: a flipped bit inside a delta-varint run still decodes, and what comes
/// out is a document ID that looks entirely ordinary. A sum of bytes misses
/// transpositions, which is exactly what a reordered or torn block looks like. CRC-32
/// catches every single- and double-bit error and all bursts up to 32 bits, which is
/// the class of damage a torn write or a bad sector produces.
uint32_t Crc32(std::span<const uint8_t> bytes);

}  // namespace meridian::storage
