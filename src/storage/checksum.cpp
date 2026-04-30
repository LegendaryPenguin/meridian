#include "checksum.hpp"

namespace meridian::storage {
namespace {

// Built once at first use rather than written out as a literal table: the generator
// polynomial is the specification, a 256-entry literal is a transcription of it, and
// a transcription error in a checksum table is invisible until it has already
// accepted a corrupt block.
struct Crc32Table {
    uint32_t entries[256];

    constexpr Crc32Table() : entries{} {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            }
            entries[i] = c;
        }
    }
};

constexpr Crc32Table kTable{};

}  // namespace

uint32_t Crc32(std::span<const uint8_t> bytes) {
    uint32_t crc = 0xFFFFFFFFu;
    for (uint8_t b : bytes) {
        crc = kTable.entries[(crc ^ b) & 0xFFu] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFFu;
}

}  // namespace meridian::storage
