#pragma once

// On-disk form of an HNSW graph.
//
// Rebuilding an index is the expensive part of the whole semantic path — an
// ef_construction=200 build does orders of magnitude more distance work than the
// queries it will ever serve — so the graph is written out, and the thing that must
// be true of the written form is narrow and testable: reopening it answers the same
// query with the same results.
//
// The file is checksummed, and the reason is specific to this data structure. A
// corrupt posting list usually decodes into something that fails a sortedness check
// and is caught. A corrupt adjacency array decodes into a perfectly valid graph
// that is simply the wrong graph: searches still return k neighbours, still return
// them quickly, and the neighbours are wrong. There is no downstream check that
// would notice, because "approximately nearest" has no self-evident ground truth at
// serving time. So the bytes are verified before they are trusted, and a mismatch
// refuses to load rather than loading something plausible.
//
// The checksum is FNV-1a, a corruption detector and not a tamper-proof one: it
// catches bit rot, truncation, and a half-written file, and it is not a MAC.

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "meridian/status.hpp"

#include "hnsw.hpp"

namespace meridian::ann {

/// Serialises everything that defines the index, including the RNG state, so an
/// index reopened and then extended takes the same level draws it would have taken.
Status EncodeHnsw(const HnswIndex& index, std::vector<uint8_t>& out);

/// Verifies the magic, version, checksum, and every structural bound before
/// building anything. Every failure mode is kCorruption except a short read.
Result<std::unique_ptr<HnswIndex>> DecodeHnsw(std::span<const uint8_t> bytes);

Status SaveHnsw(const HnswIndex& index, const std::string& path);
Result<std::unique_ptr<HnswIndex>> LoadHnsw(const std::string& path);

/// Exposed for the corruption test, which needs to know where the payload ends so
/// it can flip a byte inside it rather than inside the checksum.
constexpr uint32_t kHnswFormatVersion = 1;
size_t HnswChecksumOffset(std::span<const uint8_t> bytes);

}  // namespace meridian::ann
