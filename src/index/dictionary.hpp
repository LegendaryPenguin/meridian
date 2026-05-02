#pragma once

// The term dictionary: (field, term) -> where the posting list is, plus its TermStats.
//
// Structure: a sorted sequence of fixed-count blocks with front-coded keys, plus an
// in-memory index holding the first key of each block.
//
// Why that and not the alternatives:
//
// *A hash table* probes in O(1), which sounds decisive until you count what it costs.
// It must store every key in full, and a term dictionary is overwhelmingly shared
// prefixes — in any real corpus the sorted term list reads "retriev", "retrieval",
// "retrieve", "retrieved", "retriever", and a hash stores those five prefixes five
// times. It also destroys order, and order is not decoration here: a prefix or range
// query needs it, and so does merging two segments' dictionaries, which is a linear
// merge of two sorted streams and would otherwise require sorting the union of both
// term sets in memory. Finally the table itself is a cost: load factor below 1 means
// empty slots, and every slot is a pointer or an offset.
//
// *An FST* is the compact answer — it shares suffixes as well as prefixes and gets a
// term dictionary down near the entropy of the term set. It is also a nontrivial
// construction algorithm with a state-minimisation step, and its probe is a walk over
// arcs that is not obviously faster than a binary search plus one small block scan. The
// deciding argument is that the thing an FST saves is *memory for the index*, and the
// block index here already costs only one key per 16 terms: a million-term dictionary
// has 62,500 index entries, a couple of megabytes. The FST would win below that, and
// below that nothing is in trouble.
//
// *Front-coded sorted blocks* take the prefix sharing, which is where most of the size
// is, and keep the order. The cost is that a block must be scanned rather than indexed,
// because entry i's key is only reconstructible after entry i-1's. That is precisely why
// the block is small: at 16 entries a scan is 15 varint reads and a few memcpys on data
// that arrived in the same read, against a binary search over the whole dictionary which
// would touch log2(n) scattered pages.
//
// Block size 16: the index holds 1/16 of the keys, and the in-block scan is bounded at
// 15 reconstructions. At 128 the index would be 8x smaller and the scan 8x longer, and
// the scan is on the query path while the index is paid once at open.

#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "../storage/file.hpp"
#include "meridian/index.hpp"
#include "meridian/postings.hpp"

namespace meridian::index {

/// Terms per dictionary block.
constexpr uint32_t kDictionaryBlockTerms = 16;

/// Where a term's posting list lives, and what BM25 needs to know about the term.
struct TermLocation {
    uint64_t offset = 0;  // into the segment's postings region
    uint64_t length = 0;
    TermStats stats;
};

/// The dictionary key.
///
/// Field byte first, so the sort groups all of one field's terms together. That matters
/// for two reasons: prefix sharing only works between adjacent keys and "body:retrieval"
/// shares nothing useful with "title:retrieve", and a field-restricted scan becomes a
/// contiguous range rather than a filter over the whole dictionary.
std::string MakeTermKey(std::string_view term, Field field);

/// Splits a key back into its parts. The term is a view into `key`.
bool SplitTermKey(std::string_view key, std::string_view& term, Field& field);

/// Builds the two byte regions a dictionary is made of. Terms must arrive in ascending
/// key order; the writer rejects anything else rather than producing a dictionary whose
/// binary search silently misses terms.
class DictionaryWriter {
public:
    Status Add(std::string_view key, const TermLocation& location);

    /// `blocks` is the on-disk term data, `index` the block index. They are separate
    /// regions because the index is loaded whole at open and the blocks are not.
    Status Finish(std::vector<uint8_t>& blocks, std::vector<uint8_t>& index);

    uint64_t term_count() const { return term_count_; }

private:
    Status FlushBlock();

    struct Entry {
        std::string key;
        TermLocation location;
    };

    std::vector<Entry> pending_;
    std::vector<uint8_t> blocks_;
    std::vector<std::string> block_first_keys_;
    std::vector<uint64_t> block_offsets_;
    std::vector<uint32_t> block_lengths_;
    std::string last_key_;
    uint64_t term_count_ = 0;
    bool finished_ = false;
};

/// The read side. Holds the block index in memory and reads a block per probe.
class Dictionary {
public:
    /// `index_bytes` is the region DictionaryWriter::Finish produced.
    static Result<Dictionary> Load(std::span<const uint8_t> index_bytes);

    /// `blocks_base` is where the blocks region starts in whatever the source is.
    /// Returns nullopt when the term is absent, which is the common case and not an
    /// error; a failed *read* is distinguishable through `status`.
    std::optional<TermLocation> Lookup(const storage::ByteSource& source, uint64_t blocks_base,
                                      std::string_view key, Status* status) const;

    /// Visits every term in key order. Used by the merge, which needs a sorted walk of
    /// each input's terms and would otherwise have to probe for terms it cannot name.
    Status ForEachTerm(const storage::ByteSource& source, uint64_t blocks_base,
                       const std::function<Status(std::string_view key,
                                                  const TermLocation&)>& visit) const;

    size_t block_count() const { return blocks_.size(); }
    uint64_t term_count() const { return term_count_; }

private:
    struct BlockRef {
        std::string first_key;
        uint64_t offset = 0;
        uint32_t length = 0;
    };

    Status ReadBlock(const storage::ByteSource& source, uint64_t blocks_base, size_t index,
                     std::vector<std::pair<std::string, TermLocation>>& out) const;

    std::vector<BlockRef> blocks_;
    uint64_t term_count_ = 0;
};

}  // namespace meridian::index
