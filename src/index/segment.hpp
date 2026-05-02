#pragma once

// Reading a built segment.
//
// Two things shape this class. The first is that it reads only what it is asked for:
// the footer, the dictionary block index and the document lengths are loaded at open
// because every query needs them, and nothing else is touched until a term is probed.
// `bytes_read()` is the measurement of that, and it is a real counter over real
// positioned reads rather than an estimate.
//
// The second is that deletions are not in the file. A segment is immutable, so a
// tombstone lives beside it, and that creates a question about who owns it. The answer
// here is that a reader is a *view*: an immutable deletion bitmap plus a shared handle on
// the immutable file. Deleting a document does not mutate a reader, it clones one — the
// clone shares the file handle, the dictionary index and the length array, and owns a new
// bitmap. That is what makes a published generation genuinely frozen: a query holding an
// old generation sees the deletions that existed when it was published and nothing else,
// with no locking on the read path at all.
//
// The cost is a bitmap copy per delete, O(documents/8). For a 10,000-document segment
// that is 1.25 kB, which is cheaper than the segment write a batched alternative would
// still eventually need. A delete-heavy workload would want the bitmap built once per
// batch; the structure here does not prevent that.

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "../postings/iterator.hpp"
#include "../storage/file.hpp"
#include "builder.hpp"
#include "dictionary.hpp"
#include "meridian/index.hpp"

namespace meridian::index {

/// The parts of an open segment that are immutable and therefore shareable between
/// readers: the file handle and everything loaded from it at open.
struct SegmentData {
    std::shared_ptr<storage::ReadableFile> file;
    SegmentFooter footer;
    Dictionary dictionary;
    std::vector<uint32_t> doc_lengths;
    std::vector<uint64_t> doc_meta_offsets;
};

class SegmentReaderImpl final : public SegmentReader {
public:
    static Result<std::shared_ptr<SegmentReaderImpl>> Open(const std::string& path, uint64_t id,
                                                          DocId doc_base);

    /// A new view over the same bytes. Used for deletion and for the doc_base shifting
    /// that happens when the segment set changes, both of which must produce a new
    /// immutable reader rather than mutate a published one.
    std::shared_ptr<SegmentReaderImpl> Clone(std::vector<uint8_t> deleted, uint32_t deleted_count,
                                             DocId doc_base) const;

    std::unique_ptr<PostingIterator> Postings(std::string_view term, Field field) const override;
    std::optional<TermStats> Term(std::string_view term, Field field) const override;
    Result<DocumentMeta> Document(DocId local) const override;
    bool IsDeleted(DocId local) const override;
    const SegmentMeta& meta() const override { return meta_; }
    uint64_t bytes_read() const override { return bytes_read_->load(std::memory_order_relaxed); }

    /// Exposed so a benchmark and the differential test can compare the two iterator
    /// implementations over the same list. Queries always get the skipping one.
    std::unique_ptr<postings::ListIterator> PostingsWithMode(std::string_view term, Field field,
                                                             postings::IteratorMode mode) const;

    /// Opens an iterator from a location already in hand, skipping a second dictionary
    /// probe. The merge path walks terms and immediately reads them.
    Result<std::unique_ptr<postings::ListIterator>> PostingsAt(const TermLocation& location,
                                                               postings::IteratorMode mode) const;

    /// Sorted walk over every term. The merge needs this: it cannot probe for terms it
    /// has no way to enumerate.
    Status ForEachTerm(
        const std::function<Status(Field, std::string_view, const TermLocation&)>& visit) const;

    const std::vector<uint8_t>& deleted_bitmap() const { return deleted_; }
    uint32_t doc_length(DocId local) const {
        return local < data_->doc_lengths.size() ? data_->doc_lengths[local] : 0;
    }

    /// Tokens in documents that are still live. The generation's CorpusStats is the sum
    /// of these, and it has to exclude deleted documents or BM25's length normalisation
    /// drifts as deletions accumulate.
    uint64_t live_tokens() const { return live_tokens_; }

private:
    SegmentReaderImpl(std::shared_ptr<const SegmentData> data, SegmentMeta meta,
                      std::vector<uint8_t> deleted);

    void RecomputeLiveTokens();

    std::shared_ptr<const SegmentData> data_;
    SegmentMeta meta_;
    std::vector<uint8_t> deleted_;
    uint64_t live_tokens_ = 0;

    /// Per-reader, not per-file: several readers share one file handle and each must
    /// report its own bytes.
    std::shared_ptr<std::atomic<uint64_t>> bytes_read_;
    std::shared_ptr<const storage::ByteSource> source_;
};

/// Sets or clears a bit in a deletion bitmap, growing it as needed.
void SetDeletedBit(std::vector<uint8_t>& bitmap, DocId local);
bool TestDeletedBit(const std::vector<uint8_t>& bitmap, DocId local);

}  // namespace meridian::index
