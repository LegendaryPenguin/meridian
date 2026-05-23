#pragma once

// Two PostingIterator implementations over the same bytes.
//
// They exist as a pair on purpose. The sequential one is obviously correct: it decodes
// every block in order and steps one posting at a time, so there is nowhere for a
// boundary condition to hide. The skipping one is the one that makes conjunctive
// queries affordable, and it is also the one where a wrong comparison lands on the
// document *before* the right one and returns results that look fine. Keeping the
// simple implementation alive gives the differential test something to compare against,
// which is the only way that class of bug gets caught.
//
// Positioning convention: an iterator is positioned on its first posting as soon as it
// is opened, so `doc()` is immediately meaningful and a walk reads
//
//     for (DocId d = it->doc(); d != kNoMoreDocs; d = it->next())
//
// rather than requiring a priming call. `advance(target)` never moves backwards; when
// the iterator is already at or past `target` it returns the current document
// unchanged, which is what makes a leapfrog intersection terminate.

#include <cstdint>
#include <memory>
#include <vector>

#include "../storage/file.hpp"
#include "encoder.hpp"
#include "meridian/postings.hpp"

namespace meridian::postings {

/// The shared half of both iterators: block framing, block loading, and the counters.
///
/// `next()` and `advance()` return a DocId and cannot report a failure, so a decode
/// failure part-way through a walk goes exhausted and latches the reason, surfaced through
/// `PostingIterator::status()`. Silently stopping would turn a corrupt block into a short
/// result list, which is the failure this project treats as the worst one — and until
/// `status()` existed on the interface, a caller holding the base class could not tell the
/// two apart even though the reason was recorded right here.
class ListIterator : public PostingIterator {
public:
    ListIterator(std::shared_ptr<const storage::ByteSource> source, uint64_t list_offset,
                 uint64_t list_length);

    /// Reads the header and skip table and positions on the first posting. Separate
    /// from the constructor because it can fail and a constructor cannot return a
    /// Status; OpenListIterator is the only intended caller.
    Status Initialise();

    DocId doc() const override { return doc_; }
    DocId next() override;
    uint32_t term_frequency() const override;
    std::span<const Position> positions() const override;
    uint32_t size_hint() const override;
    Stats stats() const override { return stats_; }

    Status status() const override { return error_; }

    /// Same value by reference, for callers that already hold the concrete type.
    const Status& error() const { return error_; }
    bool has_positions() const { return header_.has_positions; }
    uint32_t num_postings() const { return header_.num_postings; }
    uint32_t num_blocks() const { return header_.num_blocks; }

protected:
    struct BlockRef {
        uint64_t offset = 0;           // relative to the list start
        uint64_t length = 0;
        uint32_t postings_before = 0;  // global index of this block's first posting
        DocId first_doc = 0;
    };

    /// Reads and decodes block `index`, positioning on its first posting.
    Status LoadBlock(size_t index);

    /// One posting forward, crossing into the next block when needed. Returns false at
    /// the end of the list or on a latched error.
    bool Step();

    /// Walks forward one posting at a time until doc() >= target. Shared by both
    /// implementations: the skipping one uses it for the within-block remainder after
    /// the jump, which is where most advance() calls end up.
    DocId ScanTo(DocId target);

    std::shared_ptr<const storage::ByteSource> source_;
    uint64_t list_offset_;
    uint64_t list_length_;
    ListHeader header_;
    std::vector<uint8_t> skip_table_;
    std::vector<BlockRef> blocks_;

    DecodedBlock block_;
    int64_t block_index_ = -1;
    size_t position_in_block_ = 0;
    DocId doc_ = kNoMoreDocs;
    Status error_;
    Stats stats_;
};

/// Decodes every block in order. No skip table is consulted for seeking.
class SequentialListIterator final : public ListIterator {
public:
    using ListIterator::ListIterator;
    DocId advance(DocId target) override;
};

/// Binary-searches the skip table and jumps straight to the block containing `target`.
class SkippingListIterator final : public ListIterator {
public:
    using ListIterator::ListIterator;
    DocId advance(DocId target) override;
};

enum class IteratorMode {
    kSequential,
    kSkipping,
};

/// Opens an iterator over the list at [list_offset, list_offset + list_length).
///
/// The header and the whole skip table are read here, in one go, because both are
/// needed before the first posting can be found and a second syscall to fetch twelve
/// more bytes would cost more than reading them eagerly ever does.
Result<std::unique_ptr<ListIterator>> OpenListIterator(
    std::shared_ptr<const storage::ByteSource> source, uint64_t list_offset,
    uint64_t list_length, IteratorMode mode);

}  // namespace meridian::postings
