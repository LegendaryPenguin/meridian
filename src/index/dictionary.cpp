#include "dictionary.hpp"

#include <algorithm>
#include <sstream>

#include "../storage/checksum.hpp"
#include "../storage/coding.hpp"

namespace meridian::index {
namespace {

/// Length of the common prefix of two keys, which is the only thing front coding needs.
size_t SharedPrefix(std::string_view a, std::string_view b) {
    const size_t limit = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < limit && a[i] == b[i]) ++i;
    return i;
}

}  // namespace

std::string MakeTermKey(std::string_view term, Field field) {
    std::string key;
    key.reserve(term.size() + 1);
    key.push_back(static_cast<char>(static_cast<uint8_t>(field)));
    key.append(term);
    return key;
}

bool SplitTermKey(std::string_view key, std::string_view& term, Field& field) {
    if (key.empty()) return false;
    const uint8_t raw = static_cast<uint8_t>(key[0]);
    if (raw >= static_cast<uint8_t>(Field::kFieldCount)) return false;
    field = static_cast<Field>(raw);
    term = key.substr(1);
    return true;
}

Status DictionaryWriter::Add(std::string_view key, const TermLocation& location) {
    if (finished_) return Status::Internal("DictionaryWriter::Add after Finish");
    if (term_count_ > 0 && !(last_key_ < key)) {
        std::ostringstream os;
        os << "dictionary keys are not strictly ascending: " << key << " follows " << last_key_;
        return Status::InvalidArgument(os.str());
    }
    last_key_.assign(key);
    ++term_count_;
    pending_.push_back(Entry{std::string(key), location});
    if (pending_.size() == kDictionaryBlockTerms) return FlushBlock();
    return Status::Ok();
}

Status DictionaryWriter::FlushBlock() {
    if (pending_.empty()) return Status::Ok();

    std::vector<uint8_t> body;
    storage::PutVarint(pending_.size(), body);

    uint64_t previous_offset = 0;
    for (size_t i = 0; i < pending_.size(); ++i) {
        const Entry& entry = pending_[i];
        if (i == 0) {
            storage::PutString(entry.key, body);
        } else {
            const size_t shared = SharedPrefix(pending_[i - 1].key, entry.key);
            storage::PutVarint(shared, body);
            storage::PutString(std::string_view(entry.key).substr(shared), body);
        }
        storage::PutVarint(entry.location.stats.document_frequency, body);
        storage::PutVarint(entry.location.stats.collection_frequency, body);
        if (i == 0) {
            storage::PutVarint(entry.location.offset, body);
        } else {
            if (entry.location.offset < previous_offset) {
                return Status::InvalidArgument(
                    "dictionary posting-list offsets must be non-decreasing in key order");
            }
            storage::PutVarint(entry.location.offset - previous_offset, body);
        }
        previous_offset = entry.location.offset;
        storage::PutVarint(entry.location.length, body);
    }
    storage::PutU32(storage::Crc32(body), body);

    block_first_keys_.push_back(pending_.front().key);
    block_offsets_.push_back(blocks_.size());
    block_lengths_.push_back(static_cast<uint32_t>(body.size()));
    blocks_.insert(blocks_.end(), body.begin(), body.end());
    pending_.clear();
    return Status::Ok();
}

Status DictionaryWriter::Finish(std::vector<uint8_t>& blocks, std::vector<uint8_t>& index) {
    if (finished_) return Status::Internal("DictionaryWriter::Finish called twice");
    MERIDIAN_TRY(FlushBlock());
    finished_ = true;

    std::vector<uint8_t> body;
    storage::PutVarint(block_first_keys_.size(), body);
    storage::PutVarint(term_count_, body);
    uint64_t previous_offset = 0;
    for (size_t i = 0; i < block_first_keys_.size(); ++i) {
        // The index keeps whole keys rather than front-coding them. It is one key per
        // 16 terms, so the sharing left to capture is small, and keeping them whole is
        // what lets the binary search compare without reconstructing anything.
        storage::PutString(block_first_keys_[i], body);
        storage::PutVarint(block_offsets_[i] - previous_offset, body);
        previous_offset = block_offsets_[i];
        storage::PutVarint(block_lengths_[i], body);
    }
    storage::PutU32(storage::Crc32(body), body);

    blocks = std::move(blocks_);
    index = std::move(body);
    return Status::Ok();
}

Result<Dictionary> Dictionary::Load(std::span<const uint8_t> index_bytes) {
    Dictionary dictionary;
    if (index_bytes.size() < 4) {
        return Status::Corruption("dictionary index is too short to hold a checksum");
    }
    const std::span<const uint8_t> body = index_bytes.first(index_bytes.size() - 4);
    const uint32_t stored = storage::GetU32(index_bytes, index_bytes.size() - 4);
    const uint32_t actual = storage::Crc32(body);
    if (stored != actual) {
        std::ostringstream os;
        os << "dictionary index checksum mismatch: stored " << stored << ", computed " << actual;
        return Status::Corruption(os.str());
    }

    size_t offset = 0;
    uint64_t block_count = 0;
    if (!storage::GetVarint(body, offset, block_count) ||
        !storage::GetVarint(body, offset, dictionary.term_count_)) {
        return Status::Corruption("dictionary index header is truncated");
    }

    dictionary.blocks_.reserve(block_count);
    uint64_t previous_offset = 0;
    for (uint64_t i = 0; i < block_count; ++i) {
        BlockRef ref;
        uint64_t delta = 0, length = 0;
        if (!storage::GetString(body, offset, ref.first_key) ||
            !storage::GetVarint(body, offset, delta) ||
            !storage::GetVarint(body, offset, length)) {
            return Status::Corruption("dictionary index is truncated mid-entry");
        }
        if (length == 0 || length > UINT32_MAX) {
            return Status::Corruption("dictionary index records an impossible block length");
        }
        ref.offset = previous_offset + delta;
        previous_offset = ref.offset;
        ref.length = static_cast<uint32_t>(length);
        dictionary.blocks_.push_back(std::move(ref));
    }
    if (offset != body.size()) {
        return Status::Corruption("dictionary index has trailing bytes");
    }
    return dictionary;
}

Status Dictionary::ReadBlock(const storage::ByteSource& source, uint64_t blocks_base,
                             size_t index,
                             std::vector<std::pair<std::string, TermLocation>>& out) const {
    const BlockRef& ref = blocks_[index];
    std::vector<uint8_t> bytes(ref.length);
    MERIDIAN_TRY(source.ReadAt(blocks_base + ref.offset, bytes.size(), bytes.data()));

    if (bytes.size() < 4) return Status::Corruption("dictionary block is too short");
    const std::span<const uint8_t> body(bytes.data(), bytes.size() - 4);
    const uint32_t stored = storage::GetU32(bytes, bytes.size() - 4);
    const uint32_t actual = storage::Crc32(body);
    if (stored != actual) {
        std::ostringstream os;
        os << "dictionary block " << index << " checksum mismatch: stored " << stored
           << ", computed " << actual;
        return Status::Corruption(os.str());
    }

    size_t offset = 0;
    uint64_t count = 0;
    if (!storage::GetVarint(body, offset, count) || count == 0 ||
        count > kDictionaryBlockTerms) {
        return Status::Corruption("dictionary block has an impossible term count");
    }

    out.clear();
    out.reserve(count);
    std::string key;
    uint64_t previous_offset = 0;
    for (uint64_t i = 0; i < count; ++i) {
        if (i == 0) {
            if (!storage::GetString(body, offset, key)) {
                return Status::Corruption("dictionary block is truncated at its first key");
            }
        } else {
            uint64_t shared = 0;
            std::string suffix;
            if (!storage::GetVarint(body, offset, shared) || shared > key.size() ||
                !storage::GetString(body, offset, suffix)) {
                return Status::Corruption("dictionary block is truncated at a front-coded key");
            }
            key.resize(shared);
            key.append(suffix);
        }

        TermLocation location;
        uint64_t df = 0, cf = 0, offset_delta = 0, length = 0;
        if (!storage::GetVarint(body, offset, df) || !storage::GetVarint(body, offset, cf) ||
            !storage::GetVarint(body, offset, offset_delta) ||
            !storage::GetVarint(body, offset, length)) {
            return Status::Corruption("dictionary block is truncated at a term's statistics");
        }
        if (df > UINT32_MAX) return Status::Corruption("dictionary document frequency overflows");
        location.stats.document_frequency = static_cast<uint32_t>(df);
        location.stats.collection_frequency = cf;
        location.offset = previous_offset + offset_delta;
        previous_offset = location.offset;
        location.length = length;
        out.emplace_back(key, location);
    }
    if (offset != body.size()) {
        return Status::Corruption("dictionary block has trailing bytes");
    }
    return Status::Ok();
}

std::optional<TermLocation> Dictionary::Lookup(const storage::ByteSource& source,
                                              uint64_t blocks_base, std::string_view key,
                                              Status* status) const {
    if (status) *status = Status::Ok();
    if (blocks_.empty()) return std::nullopt;

    // Last block whose first key is <= the probe. Any earlier block ends before the
    // probe and any later one starts after it, so this is the only block that can hold
    // it. A probe below the first key cannot be present at all.
    size_t lo = 0, hi = blocks_.size();
    bool found = false;
    size_t candidate = 0;
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        if (blocks_[mid].first_key <= key) {
            candidate = mid;
            found = true;
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (!found) return std::nullopt;

    std::vector<std::pair<std::string, TermLocation>> entries;
    const Status s = ReadBlock(source, blocks_base, candidate, entries);
    if (!s.ok()) {
        if (status) *status = s;
        return std::nullopt;
    }
    for (const auto& entry : entries) {
        if (entry.first == key) return entry.second;
        if (entry.first > key) break;  // sorted: nothing later can match
    }
    return std::nullopt;
}

Status Dictionary::ForEachTerm(
    const storage::ByteSource& source, uint64_t blocks_base,
    const std::function<Status(std::string_view, const TermLocation&)>& visit) const {
    std::vector<std::pair<std::string, TermLocation>> entries;
    for (size_t i = 0; i < blocks_.size(); ++i) {
        MERIDIAN_TRY(ReadBlock(source, blocks_base, i, entries));
        for (const auto& entry : entries) {
            MERIDIAN_TRY(visit(entry.first, entry.second));
        }
    }
    return Status::Ok();
}

}  // namespace meridian::index
