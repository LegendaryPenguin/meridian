#include "persist.hpp"

#include <cstring>
#include <fstream>
#include <limits>

namespace meridian::ann {
namespace {

constexpr char kMagic[8] = {'M', 'R', 'D', 'N', 'H', 'N', 'S', 'W'};
constexpr size_t kHeaderBytes = 8 + 4 + 4;  // magic, version, reserved flags
constexpr size_t kChecksumBytes = 8;

/// A sanity ceiling on the node count read from the file, so a corrupt length field
/// cannot make the loader try to reserve an absurd amount before the checksum has
/// even been checked. Chosen well above any single-shard index this engine would
/// hold and well below anything that would exhaust memory.
constexpr uint64_t kMaxNodes = 1ull << 32;

uint64_t Fnv1a(std::span<const uint8_t> bytes) {
    uint64_t hash = 0xCBF29CE484222325ull;
    for (uint8_t b : bytes) {
        hash ^= b;
        hash *= 0x100000001B3ull;
    }
    return hash;
}

// Integers are written little-endian byte by byte rather than memcpy'd, so the file
// does not depend on the host's byte order. An index written on one architecture and
// read on another would otherwise load as a valid wrong graph, which is exactly the
// failure the checksum exists to catch and which it would not catch, because the
// bytes are intact and only their interpretation is wrong.

void PutU32(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back(static_cast<uint8_t>(v));
    out.push_back(static_cast<uint8_t>(v >> 8));
    out.push_back(static_cast<uint8_t>(v >> 16));
    out.push_back(static_cast<uint8_t>(v >> 24));
}

void PutU64(std::vector<uint8_t>& out, uint64_t v) {
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<uint8_t>(v >> (8 * i)));
}

void PutF32(std::vector<uint8_t>& out, float f) {
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    PutU32(out, bits);
}

/// A bounds-checked cursor. Every read can fail, and the cursor remembers that it
/// failed rather than requiring a check at each call site — a missed check is how a
/// truncated file turns into a read past the end of the buffer.
class Reader {
public:
    explicit Reader(std::span<const uint8_t> bytes) : bytes_(bytes) {}

    bool ok() const { return ok_; }
    size_t offset() const { return offset_; }
    size_t remaining() const { return ok_ ? bytes_.size() - offset_ : 0; }

    uint32_t U32() {
        if (!Have(4)) return 0;
        const uint32_t v = uint32_t(bytes_[offset_]) | uint32_t(bytes_[offset_ + 1]) << 8 |
                           uint32_t(bytes_[offset_ + 2]) << 16 |
                           uint32_t(bytes_[offset_ + 3]) << 24;
        offset_ += 4;
        return v;
    }

    uint64_t U64() {
        if (!Have(8)) return 0;
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i) v |= uint64_t(bytes_[offset_ + size_t(i)]) << (8 * i);
        offset_ += 8;
        return v;
    }

    float F32() {
        const uint32_t bits = U32();
        float f = 0.0f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    }

private:
    bool Have(size_t n) {
        if (!ok_ || bytes_.size() - offset_ < n) {
            ok_ = false;
            return false;
        }
        return true;
    }

    std::span<const uint8_t> bytes_;
    size_t offset_ = 0;
    bool ok_ = true;
};

}  // namespace

size_t HnswChecksumOffset(std::span<const uint8_t> bytes) {
    return bytes.size() >= kChecksumBytes ? bytes.size() - kChecksumBytes : 0;
}

Status EncodeHnsw(const HnswIndex& index, std::vector<uint8_t>& out) {
    const HnswIndex::Snapshot s = index.TakeSnapshot();

    out.clear();
    out.insert(out.end(), kMagic, kMagic + sizeof(kMagic));
    PutU32(out, kHnswFormatVersion);
    PutU32(out, 0);  // reserved; a future quantised-vector variant goes here

    PutU32(out, s.dim);
    PutU32(out, static_cast<uint32_t>(s.params.metric));
    PutU32(out, s.params.M);
    PutU32(out, s.params.M0);
    PutU32(out, s.params.ef_construction);
    PutU32(out, static_cast<uint32_t>(s.params.selection));
    PutU32(out, s.params.keep_pruned_connections ? 1u : 0u);
    PutU64(out, s.params.seed);
    PutU64(out, s.rng_state);
    PutU32(out, s.entry_point);
    PutU32(out, s.max_layer);
    PutU64(out, static_cast<uint64_t>(s.docs.size()));

    for (float v : s.vectors) PutF32(out, v);

    for (size_t i = 0; i < s.docs.size(); ++i) {
        PutU32(out, s.docs[i]);
        PutU32(out, s.top_layers[i]);
        for (const std::vector<uint32_t>& layer : s.neighbors[i]) {
            PutU32(out, static_cast<uint32_t>(layer.size()));
            for (uint32_t e : layer) PutU32(out, e);
        }
    }

    PutU64(out, Fnv1a(std::span<const uint8_t>(out.data(), out.size())));
    return Status::Ok();
}

Result<std::unique_ptr<HnswIndex>> DecodeHnsw(std::span<const uint8_t> bytes) {
    if (bytes.size() < kHeaderBytes + kChecksumBytes) {
        return Status::Corruption("hnsw file is shorter than its own header");
    }
    if (std::memcmp(bytes.data(), kMagic, sizeof(kMagic)) != 0) {
        return Status::Corruption("hnsw file has the wrong magic");
    }

    const size_t payload = bytes.size() - kChecksumBytes;
    uint64_t stored = 0;
    for (int i = 0; i < 8; ++i) stored |= uint64_t(bytes[payload + size_t(i)]) << (8 * i);
    const uint64_t actual = Fnv1a(bytes.subspan(0, payload));
    if (stored != actual) {
        return Status::Corruption("hnsw checksum mismatch: the graph on disk is not the graph "
                                 "that was written");
    }

    // Only the checksummed prefix is parsed. Handing the checksum bytes to the
    // reader as if they were data would let a corrupt length field consume them.
    Reader r(bytes.subspan(0, payload));
    r.U64();  // magic, already compared
    const uint32_t version = r.U32();
    if (version != kHnswFormatVersion) {
        return Status::Corruption("hnsw format version " + std::to_string(version) +
                                 " is not readable by this build");
    }
    r.U32();  // reserved flags

    HnswIndex::Snapshot s;
    s.dim = r.U32();
    const uint32_t metric = r.U32();
    if (metric > static_cast<uint32_t>(Metric::kL2Squared)) {
        return Status::Corruption("hnsw file names an unknown metric");
    }
    s.params.metric = static_cast<Metric>(metric);
    s.params.M = r.U32();
    s.params.M0 = r.U32();
    s.params.ef_construction = r.U32();
    const uint32_t selection = r.U32();
    if (selection > static_cast<uint32_t>(NeighborSelection::kTopM)) {
        return Status::Corruption("hnsw file names an unknown neighbour-selection rule");
    }
    s.params.selection = static_cast<NeighborSelection>(selection);
    s.params.keep_pruned_connections = r.U32() != 0;
    s.params.seed = r.U64();
    s.rng_state = r.U64();
    s.entry_point = r.U32();
    s.max_layer = r.U32();
    const uint64_t count = r.U64();
    if (!r.ok()) return Status::Corruption("hnsw header is truncated");

    if (s.dim == 0 || s.dim > (1u << 20)) {
        return Status::Corruption("hnsw file declares an implausible dimension");
    }
    if (count > kMaxNodes) return Status::Corruption("hnsw file declares too many nodes");

    // Floor check before reserving: every node costs at least a vector, a doc id and
    // a top-layer field, so a count that could not possibly fit in the remaining
    // bytes is rejected before any allocation happens.
    const uint64_t min_bytes_per_node = uint64_t(s.dim) * 4 + 4 + 4 + 4;
    if (count != 0 && count > r.remaining() / min_bytes_per_node) {
        return Status::Corruption("hnsw node count does not fit the file");
    }

    const size_t n = static_cast<size_t>(count);
    s.vectors.resize(n * s.dim);
    for (float& v : s.vectors) v = r.F32();
    if (!r.ok()) return Status::Corruption("hnsw vector block is truncated");

    s.docs.resize(n);
    s.top_layers.resize(n);
    s.neighbors.resize(n);
    for (size_t i = 0; i < n; ++i) {
        s.docs[i] = r.U32();
        const uint32_t top = r.U32();
        if (!r.ok()) return Status::Corruption("hnsw node record is truncated");
        if (top > s.max_layer) return Status::Corruption("hnsw node layer above the maximum");
        s.top_layers[i] = top;
        s.neighbors[i].resize(top + 1);
        for (uint32_t l = 0; l <= top; ++l) {
            const uint32_t degree = r.U32();
            if (!r.ok()) return Status::Corruption("hnsw adjacency is truncated");
            // Checked against the remaining bytes before resizing, so a corrupt
            // degree cannot turn into a four-billion-element allocation.
            if (degree > r.remaining() / 4) {
                return Status::Corruption("hnsw adjacency length exceeds the file");
            }
            s.neighbors[i][l].resize(degree);
            for (uint32_t& e : s.neighbors[i][l]) e = r.U32();
        }
    }
    if (!r.ok()) return Status::Corruption("hnsw graph block is truncated");
    if (r.remaining() != 0) {
        // Trailing bytes mean the writer and reader disagree about the format, and
        // loading anyway would be loading a graph this build does not understand.
        return Status::Corruption("hnsw file has " + std::to_string(r.remaining()) +
                                 " unread trailing bytes");
    }

    // The remaining structural checks — edges in range, degree bounds, entry point
    // on the top layer — live in FromSnapshot, so they also apply to a snapshot that
    // never went through a file.
    return HnswIndex::FromSnapshot(std::move(s));
}

Status SaveHnsw(const HnswIndex& index, const std::string& path) {
    std::vector<uint8_t> bytes;
    MERIDIAN_TRY(EncodeHnsw(index, bytes));

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return Status::IoError("cannot open " + path + " for writing");
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    out.flush();
    // Checked after the flush rather than after the write: a buffered write that
    // fails only on flush would otherwise report success for a truncated file, and a
    // truncated graph file is the exact case the checksum is there to catch on the
    // way back in.
    if (!out) return Status::IoError("write to " + path + " failed");
    return Status::Ok();
}

Result<std::unique_ptr<HnswIndex>> LoadHnsw(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) return Status::IoError("cannot open " + path + " for reading");

    const std::streamoff size = in.tellg();
    if (size < 0) return Status::IoError("cannot size " + path);
    if (static_cast<uint64_t>(size) > std::numeric_limits<size_t>::max()) {
        return Status::IoError(path + " is too large to map into memory");
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(bytes.data()), size);
    if (in.gcount() != size) return Status::IoError("short read on " + path);

    return DecodeHnsw(bytes);
}

}  // namespace meridian::ann
