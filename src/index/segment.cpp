#include "segment.hpp"

#include <sstream>

#include "../storage/coding.hpp"

namespace meridian::index {

void SetDeletedBit(std::vector<uint8_t>& bitmap, DocId local) {
    const size_t byte = local / 8;
    if (bitmap.size() <= byte) bitmap.resize(byte + 1, 0);
    bitmap[byte] = static_cast<uint8_t>(bitmap[byte] | (1u << (local % 8)));
}

bool TestDeletedBit(const std::vector<uint8_t>& bitmap, DocId local) {
    const size_t byte = local / 8;
    if (byte >= bitmap.size()) return false;
    return (bitmap[byte] & (1u << (local % 8))) != 0;
}

SegmentReaderImpl::SegmentReaderImpl(std::shared_ptr<const SegmentData> data, SegmentMeta meta,
                                     std::vector<uint8_t> deleted)
    : data_(std::move(data)),
      meta_(std::move(meta)),
      deleted_(std::move(deleted)),
      bytes_read_(std::make_shared<std::atomic<uint64_t>>(0)) {
    source_ = std::make_shared<storage::CountingByteSource>(data_->file, bytes_read_);
    RecomputeLiveTokens();
}

void SegmentReaderImpl::RecomputeLiveTokens() {
    live_tokens_ = 0;
    for (size_t i = 0; i < data_->doc_lengths.size(); ++i) {
        if (!TestDeletedBit(deleted_, static_cast<DocId>(i))) live_tokens_ += data_->doc_lengths[i];
    }
}

Result<std::shared_ptr<SegmentReaderImpl>> SegmentReaderImpl::Open(const std::string& path,
                                                                   uint64_t id, DocId doc_base) {
    MERIDIAN_ASSIGN(std::shared_ptr<storage::ReadableFile> file, storage::ReadableFile::Open(path));

    // The open-time reads go through a counter that is then handed to the reader, so the
    // bytes spent opening are part of what bytes_read() reports. Hiding them would make
    // the counter flattering rather than honest.
    auto counter = std::make_shared<std::atomic<uint64_t>>(0);
    storage::CountingByteSource source(file, counter);

    if (file->size() < kSegmentFooterBytes) {
        std::ostringstream os;
        os << path << " is " << file->size() << " bytes, too small to be a segment";
        return Status::Corruption(os.str());
    }

    auto data = std::make_shared<SegmentData>();
    data->file = file;

    std::vector<uint8_t> footer_bytes(kSegmentFooterBytes);
    MERIDIAN_TRY(source.ReadAt(file->size() - kSegmentFooterBytes, kSegmentFooterBytes,
                               footer_bytes.data()));
    MERIDIAN_TRY(DecodeSegmentFooter(footer_bytes, data->footer));
    const SegmentFooter& footer = data->footer;

    // Every offset the footer hands out is used to index into the file, so each is bounds
    // checked here rather than at the point of use. A corrupt footer that passed its own
    // checksum is unlikely; a footer from a truncated file is not.
    const uint64_t size = file->size();
    const auto within = [size](uint64_t offset, uint64_t length) {
        return offset <= size && length <= size - offset;
    };
    if (!within(footer.postings_offset, footer.postings_length) ||
        !within(footer.dict_blocks_offset, footer.dict_blocks_length) ||
        !within(footer.dict_index_offset, footer.dict_index_length) ||
        !within(footer.doc_lengths_offset, uint64_t{footer.document_count} * 4) ||
        !within(footer.doc_offsets_offset, uint64_t{footer.document_count} * 8) ||
        !within(footer.doc_meta_offset, footer.doc_meta_length)) {
        return Status::Corruption("segment footer describes a region outside the file");
    }

    std::vector<uint8_t> index_bytes(footer.dict_index_length);
    MERIDIAN_TRY(source.ReadAt(footer.dict_index_offset, index_bytes.size(), index_bytes.data()));
    MERIDIAN_ASSIGN(data->dictionary, Dictionary::Load(index_bytes));
    if (data->dictionary.term_count() != footer.term_count) {
        return Status::Corruption("segment dictionary term count disagrees with the footer");
    }

    const uint32_t documents = footer.document_count;
    if (documents > 0) {
        std::vector<uint8_t> raw(static_cast<size_t>(documents) * 4);
        MERIDIAN_TRY(source.ReadAt(footer.doc_lengths_offset, raw.size(), raw.data()));
        data->doc_lengths.resize(documents);
        for (uint32_t i = 0; i < documents; ++i) {
            data->doc_lengths[i] = storage::GetU32(raw, static_cast<size_t>(i) * 4);
        }

        std::vector<uint8_t> raw_offsets(static_cast<size_t>(documents) * 8);
        MERIDIAN_TRY(
            source.ReadAt(footer.doc_offsets_offset, raw_offsets.size(), raw_offsets.data()));
        data->doc_meta_offsets.resize(documents);
        for (uint32_t i = 0; i < documents; ++i) {
            const uint64_t offset = storage::GetU64(raw_offsets, static_cast<size_t>(i) * 8);
            if (offset > footer.doc_meta_length) {
                return Status::Corruption("segment document offset is outside the metadata region");
            }
            data->doc_meta_offsets[i] = offset;
        }
    }

    SegmentMeta meta;
    meta.id = id;
    meta.path = path;
    meta.doc_base = doc_base;
    meta.document_count = documents;
    meta.deleted_count = 0;
    meta.term_count = footer.term_count;
    meta.bytes = size;
    meta.created_at = 0;

    auto reader = std::shared_ptr<SegmentReaderImpl>(
        new SegmentReaderImpl(std::move(data), std::move(meta), std::vector<uint8_t>()));
    // Transferred rather than discarded: the open-time reads belong to this reader.
    reader->bytes_read_->store(counter->load(std::memory_order_relaxed),
                               std::memory_order_relaxed);
    return reader;
}

std::shared_ptr<SegmentReaderImpl> SegmentReaderImpl::Clone(std::vector<uint8_t> deleted,
                                                            uint32_t deleted_count,
                                                            DocId doc_base) const {
    SegmentMeta meta = meta_;
    meta.deleted_count = deleted_count;
    meta.doc_base = doc_base;
    return std::shared_ptr<SegmentReaderImpl>(
        new SegmentReaderImpl(data_, std::move(meta), std::move(deleted)));
}

Result<std::unique_ptr<postings::ListIterator>> SegmentReaderImpl::PostingsAt(
    const TermLocation& location, postings::IteratorMode mode) const {
    const SegmentFooter& footer = data_->footer;
    if (location.offset > footer.postings_length ||
        location.length > footer.postings_length - location.offset) {
        return Status::Corruption("term's posting list is outside the postings region");
    }
    return postings::OpenListIterator(source_, footer.postings_offset + location.offset,
                                      location.length, mode);
}

std::unique_ptr<postings::ListIterator> SegmentReaderImpl::PostingsWithMode(
    std::string_view term, Field field, postings::IteratorMode mode) const {
    Status status;
    const std::string key = MakeTermKey(term, field);
    const std::optional<TermLocation> location =
        data_->dictionary.Lookup(*source_, data_->footer.dict_blocks_offset, key, &status);
    if (!location.has_value()) return nullptr;

    auto iterator = PostingsAt(*location, mode);
    if (!iterator.ok()) return nullptr;
    return iterator.take();
}

std::unique_ptr<PostingIterator> SegmentReaderImpl::Postings(std::string_view term,
                                                            Field field) const {
    // Skipping, always, on the query path. The sequential implementation exists to be
    // compared against, not to serve queries.
    return PostingsWithMode(term, field, postings::IteratorMode::kSkipping);
}

std::optional<TermStats> SegmentReaderImpl::Term(std::string_view term, Field field) const {
    Status status;
    const std::string key = MakeTermKey(term, field);
    const std::optional<TermLocation> location =
        data_->dictionary.Lookup(*source_, data_->footer.dict_blocks_offset, key, &status);
    if (!location.has_value()) return std::nullopt;
    return location->stats;
}

Result<DocumentMeta> SegmentReaderImpl::Document(DocId local) const {
    if (local >= data_->doc_meta_offsets.size()) {
        std::ostringstream os;
        os << "document " << local << " is outside segment " << meta_.id << " ("
           << meta_.document_count << " documents)";
        return Status::NotFound(os.str());
    }

    const uint64_t start = data_->doc_meta_offsets[local];
    const uint64_t end = (local + 1 < data_->doc_meta_offsets.size())
                             ? data_->doc_meta_offsets[local + 1]
                             : data_->footer.doc_meta_length;
    if (end < start) return Status::Corruption("segment document offsets are not ascending");

    std::vector<uint8_t> bytes(static_cast<size_t>(end - start));
    MERIDIAN_TRY(source_->ReadAt(data_->footer.doc_meta_offset + start, bytes.size(), bytes.data()));

    size_t offset = 0;
    DocumentMeta meta;
    uint64_t published = 0, length = 0;
    if (!storage::GetString(bytes, offset, meta.external_id) ||
        !storage::GetString(bytes, offset, meta.language) ||
        !storage::GetString(bytes, offset, meta.category) ||
        !storage::GetVarint(bytes, offset, published) ||
        !storage::GetVarint(bytes, offset, length) || offset + 8 > bytes.size()) {
        return Status::Corruption("segment document record is truncated");
    }
    meta.published_at = storage::UnZigZag(published);
    meta.length = static_cast<uint32_t>(length);
    meta.popularity = storage::GetDouble(bytes, offset);
    return meta;
}

bool SegmentReaderImpl::IsDeleted(DocId local) const { return TestDeletedBit(deleted_, local); }

Status SegmentReaderImpl::ForEachTerm(
    const std::function<Status(Field, std::string_view, const TermLocation&)>& visit) const {
    return data_->dictionary.ForEachTerm(
        *source_, data_->footer.dict_blocks_offset,
        [&visit](std::string_view key, const TermLocation& location) -> Status {
            std::string_view term;
            Field field = Field::kTitle;
            if (!SplitTermKey(key, term, field)) {
                return Status::Corruption("dictionary key does not decode to a field and term");
            }
            return visit(field, term, location);
        });
}

}  // namespace meridian::index
