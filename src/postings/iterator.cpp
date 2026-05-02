#include "iterator.hpp"

#include <algorithm>
#include <sstream>

namespace meridian::postings {

ListIterator::ListIterator(std::shared_ptr<const storage::ByteSource> source,
                           uint64_t list_offset, uint64_t list_length)
    : source_(std::move(source)), list_offset_(list_offset), list_length_(list_length) {}

Status ListIterator::Initialise() {
    if (list_length_ == 0) {
        return Status::Corruption("posting list has zero length");
    }

    const size_t prefix_bytes =
        static_cast<size_t>(std::min<uint64_t>(kMaxListHeaderBytes, list_length_));
    std::vector<uint8_t> prefix(prefix_bytes);
    MERIDIAN_TRY(source_->ReadAt(list_offset_, prefix_bytes, prefix.data()));
    MERIDIAN_TRY(ParseListHeader(prefix, header_));

    if (header_.skip_entries > 0) {
        const size_t table_bytes =
            static_cast<size_t>(header_.skip_entries) * kSkipEntryBytes;
        if (header_.skip_offset + table_bytes > list_length_) {
            return Status::Corruption("posting list skip table runs past the end of the list");
        }
        skip_table_.resize(table_bytes);
        MERIDIAN_TRY(source_->ReadAt(list_offset_ + header_.skip_offset, table_bytes,
                                     skip_table_.data()));
    }

    if (header_.num_blocks == 0) {
        doc_ = kNoMoreDocs;
        return Status::Ok();
    }

    blocks_.resize(header_.num_blocks);
    if (header_.skip_entries > 0) {
        for (uint32_t i = 0; i < header_.num_blocks; ++i) {
            const SkipEntry entry = ReadSkipEntry(skip_table_, i);
            blocks_[i].offset = header_.blocks_offset + entry.byte_offset;
            blocks_[i].postings_before = entry.postings_before;
            blocks_[i].first_doc = entry.doc;
        }
        // Block lengths come from the gaps between successive skip offsets, so the
        // format does not store a length twice. The last block runs to the end of the
        // list, which is why an iterator is opened with the list's length rather than
        // left to discover it.
        for (uint32_t i = 0; i + 1 < header_.num_blocks; ++i) {
            if (blocks_[i + 1].offset <= blocks_[i].offset) {
                return Status::Corruption("posting list skip offsets are not ascending");
            }
            blocks_[i].length = blocks_[i + 1].offset - blocks_[i].offset;
        }
        blocks_.back().length = list_length_ - blocks_.back().offset;
    } else {
        blocks_[0].offset = header_.blocks_offset;
        blocks_[0].postings_before = 0;
        blocks_[0].length = list_length_ - header_.blocks_offset;
    }

    for (const BlockRef& ref : blocks_) {
        if (ref.offset >= list_length_ || ref.length == 0 ||
            ref.offset + ref.length > list_length_) {
            return Status::Corruption("posting list block extends past the end of the list");
        }
    }

    return LoadBlock(0);
}

Status ListIterator::LoadBlock(size_t index) {
    const BlockRef& ref = blocks_[index];
    std::vector<uint8_t> bytes(static_cast<size_t>(ref.length));
    MERIDIAN_TRY(source_->ReadAt(list_offset_ + ref.offset, bytes.size(), bytes.data()));
    MERIDIAN_TRY(DecodeBlock(bytes, header_.has_positions, block_));

    // The block frame carries its own posting count, and the header carries the total.
    // They have to agree, or a seek using postings_before lands at the wrong global
    // index and size_hint starts lying.
    const uint32_t expected = std::min<uint32_t>(
        kBlockPostings, header_.num_postings - ref.postings_before);
    if (block_.count() != expected) {
        std::ostringstream os;
        os << "posting block " << index << " holds " << block_.count() << " postings, "
           << expected << " expected";
        return Status::Corruption(os.str());
    }

    ++stats_.blocks_read;
    // postings_decoded counts postings the decoder actually produced, which is a whole
    // block at a time because that is the unit the format can start from. This is the
    // number the skip-versus-sequential claim is made on, so it deliberately charges
    // skipping for the postings it decodes but does not want: the measure is work done,
    // not postings returned.
    stats_.postings_decoded += block_.count();

    block_index_ = static_cast<int64_t>(index);
    position_in_block_ = 0;
    doc_ = block_.docs[0];
    return Status::Ok();
}

bool ListIterator::Step() {
    if (!error_.ok() || doc_ == kNoMoreDocs) return false;

    if (position_in_block_ + 1 < block_.count()) {
        ++position_in_block_;
        doc_ = block_.docs[position_in_block_];
        return true;
    }

    const size_t next_block = static_cast<size_t>(block_index_) + 1;
    if (next_block < blocks_.size()) {
        const Status s = LoadBlock(next_block);
        if (!s.ok()) {
            error_ = s;
            doc_ = kNoMoreDocs;
            return false;
        }
        return true;
    }

    doc_ = kNoMoreDocs;
    return false;
}

DocId ListIterator::ScanTo(DocId target) {
    while (doc_ < target) {
        if (!Step()) break;
    }
    return doc_;
}

DocId ListIterator::next() {
    if (doc_ == kNoMoreDocs) return kNoMoreDocs;
    Step();
    return doc_;
}

uint32_t ListIterator::term_frequency() const {
    if (doc_ == kNoMoreDocs || position_in_block_ >= block_.term_frequencies.size()) return 0;
    return block_.term_frequencies[position_in_block_];
}

std::span<const Position> ListIterator::positions() const {
    if (doc_ == kNoMoreDocs) return {};
    return block_.positions_for(position_in_block_);
}

uint32_t ListIterator::size_hint() const {
    if (doc_ == kNoMoreDocs || block_index_ < 0) return 0;
    const uint32_t consumed =
        blocks_[static_cast<size_t>(block_index_)].postings_before +
        static_cast<uint32_t>(position_in_block_);
    return header_.num_postings - consumed;
}

DocId SequentialListIterator::advance(DocId target) {
    // The honest baseline: every posting between here and the target is decoded.
    return ScanTo(target);
}

DocId SkippingListIterator::advance(DocId target) {
    if (doc_ >= target) return doc_;

    if (header_.skip_entries > 0) {
        const int64_t index =
            FindSkipEntry(skip_table_, header_.skip_entries, target);
        // index == -1 is unreachable here: doc_ is never below the first block's first
        // document, and doc_ < target was just established. Guarded anyway rather than
        // asserted, because the alternative is indexing blocks_ with -1.
        if (index > block_index_) {
            const Status s = LoadBlock(static_cast<size_t>(index));
            if (!s.ok()) {
                error_ = s;
                doc_ = kNoMoreDocs;
                return doc_;
            }
            ++stats_.skips_taken;
        }
    }

    // Whatever remains is inside one block, because the next block's first document is
    // past the target by construction.
    return ScanTo(target);
}

Result<std::unique_ptr<ListIterator>> OpenListIterator(
    std::shared_ptr<const storage::ByteSource> source, uint64_t list_offset,
    uint64_t list_length, IteratorMode mode) {
    std::unique_ptr<ListIterator> it;
    if (mode == IteratorMode::kSkipping) {
        it = std::make_unique<SkippingListIterator>(std::move(source), list_offset, list_length);
    } else {
        it = std::make_unique<SequentialListIterator>(std::move(source), list_offset,
                                                      list_length);
    }
    MERIDIAN_TRY(it->Initialise());
    return it;
}

}  // namespace meridian::postings
