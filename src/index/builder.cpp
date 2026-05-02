#include "builder.hpp"

#include <chrono>
#include <sstream>

#include "../postings/encoder.hpp"
#include "../storage/checksum.hpp"
#include "../storage/coding.hpp"
#include "../storage/file.hpp"

namespace meridian::index {
namespace {

void EncodeDocumentMeta(const DocumentMeta& meta, std::vector<uint8_t>& out) {
    storage::PutString(meta.external_id, out);
    storage::PutString(meta.language, out);
    storage::PutString(meta.category, out);
    storage::PutVarint(storage::ZigZag(meta.published_at), out);
    storage::PutVarint(meta.length, out);
    storage::PutDouble(meta.popularity, out);
}

int64_t NowSeconds() {
    using namespace std::chrono;
    return duration_cast<seconds>(system_clock::now().time_since_epoch()).count();
}

}  // namespace

void EncodeSegmentFooter(const SegmentFooter& footer, std::vector<uint8_t>& out) {
    const size_t start = out.size();
    storage::PutU64(kSegmentMagic, out);
    storage::PutU32(kSegmentFormatVersion, out);
    storage::PutU64(footer.postings_offset, out);
    storage::PutU64(footer.postings_length, out);
    storage::PutU64(footer.dict_blocks_offset, out);
    storage::PutU64(footer.dict_blocks_length, out);
    storage::PutU64(footer.dict_index_offset, out);
    storage::PutU64(footer.dict_index_length, out);
    storage::PutU64(footer.doc_lengths_offset, out);
    storage::PutU64(footer.doc_offsets_offset, out);
    storage::PutU64(footer.doc_meta_offset, out);
    storage::PutU64(footer.doc_meta_length, out);
    storage::PutU64(footer.term_count, out);
    storage::PutU32(footer.document_count, out);
    storage::PutU64(footer.total_tokens, out);
    const uint32_t crc = storage::Crc32(
        std::span<const uint8_t>(out.data() + start, out.size() - start));
    storage::PutU32(crc, out);
    // The magic appears twice: once at the front so a footer read from the tail can be
    // recognised, and once at the very end so the *tail* can be recognised at all. With
    // only a leading magic, a file truncated by a few bytes puts the magic at an unknown
    // offset and the reader has to search for it.
    storage::PutU64(kSegmentMagic, out);
}

Status DecodeSegmentFooter(std::span<const uint8_t> bytes, SegmentFooter& out) {
    if (bytes.size() != kSegmentFooterBytes) {
        std::ostringstream os;
        os << "segment footer is " << bytes.size() << " bytes, expected "
           << kSegmentFooterBytes;
        return Status::Corruption(os.str());
    }
    if (storage::GetU64(bytes, 0) != kSegmentMagic ||
        storage::GetU64(bytes, kSegmentFooterBytes - 8) != kSegmentMagic) {
        return Status::Corruption("segment footer magic is wrong; this is not a segment file");
    }
    const size_t body_bytes = kSegmentFooterBytes - 12;  // minus crc and trailing magic
    const uint32_t stored = storage::GetU32(bytes, body_bytes);
    const uint32_t actual = storage::Crc32(bytes.first(body_bytes));
    if (stored != actual) {
        std::ostringstream os;
        os << "segment footer checksum mismatch: stored " << stored << ", computed " << actual;
        return Status::Corruption(os.str());
    }
    const uint32_t version = storage::GetU32(bytes, 8);
    if (version != kSegmentFormatVersion) {
        std::ostringstream os;
        os << "segment format version " << version << " is not " << kSegmentFormatVersion;
        return Status::Corruption(os.str());
    }

    size_t at = 12;
    const auto take = [&bytes, &at]() {
        const uint64_t v = storage::GetU64(bytes, at);
        at += 8;
        return v;
    };
    out.postings_offset = take();
    out.postings_length = take();
    out.dict_blocks_offset = take();
    out.dict_blocks_length = take();
    out.dict_index_offset = take();
    out.dict_index_length = take();
    out.doc_lengths_offset = take();
    out.doc_offsets_offset = take();
    out.doc_meta_offset = take();
    out.doc_meta_length = take();
    out.term_count = take();
    out.document_count = storage::GetU32(bytes, at);
    at += 4;
    out.total_tokens = storage::GetU64(bytes, at);
    return Status::Ok();
}

DocId SegmentBuilder::AddDocument(const DocumentMeta& meta) {
    const DocId local = static_cast<DocId>(documents_.size());
    documents_.push_back(meta);
    external_ids_.push_back(meta.external_id);
    total_tokens_ += meta.length;
    return local;
}

Status SegmentBuilder::AddTerm(Field field, std::string_view term, DocId local,
                              uint32_t term_frequency, std::span<const Position> positions) {
    if (static_cast<size_t>(field) >= static_cast<size_t>(Field::kFieldCount)) {
        return Status::InvalidArgument("term added for a field outside Field::kFieldCount");
    }
    if (local >= documents_.size()) {
        return Status::InvalidArgument("term added for a document that has not been added");
    }

    std::vector<Posting>& list = postings_[MakeTermKey(term, field)];
    if (!list.empty() && list.back().doc >= local) {
        std::ostringstream os;
        os << "term '" << term << "' added for document " << local << " after document "
           << list.back().doc;
        return Status::InvalidArgument(os.str());
    }

    Posting posting;
    posting.doc = local;
    posting.term_frequency = term_frequency;
    if (options_.positions[static_cast<size_t>(field)]) {
        posting.positions.assign(positions.begin(), positions.end());
    }
    list.push_back(std::move(posting));
    return Status::Ok();
}

Status SegmentBuilder::AddTokenizedDocument(
    const DocumentMeta& meta,
    const std::unordered_map<Field, std::vector<std::string>>& tokens_by_field,
    DocId* local_out) {
    DocumentMeta copy = meta;
    if (copy.length == 0) {
        // The caller is allowed to leave the length out and have it counted here, because
        // a length that disagrees with the postings makes BM25's normalisation wrong in a
        // way that is invisible in the results.
        uint64_t tokens = 0;
        for (const auto& [field, list] : tokens_by_field) {
            (void)field;
            tokens += list.size();
        }
        copy.length = static_cast<uint32_t>(tokens);
    }

    const DocId local = AddDocument(copy);
    if (local_out) *local_out = local;

    for (const auto& [field, tokens] : tokens_by_field) {
        if (static_cast<size_t>(field) >= static_cast<size_t>(Field::kFieldCount)) {
            return Status::InvalidArgument("tokens supplied for a field outside Field");
        }
        // Grouped per document before being handed over, so each term reaches AddTerm
        // once with its full position list. Accumulating per occurrence instead would
        // make a term's postings for one document non-contiguous.
        std::map<std::string, std::vector<Position>> occurrences;
        for (size_t i = 0; i < tokens.size(); ++i) {
            occurrences[tokens[i]].push_back(static_cast<Position>(i));
        }
        for (const auto& [term, positions] : occurrences) {
            MERIDIAN_TRY(AddTerm(field, term, local,
                                 static_cast<uint32_t>(positions.size()), positions));
        }
    }
    return Status::Ok();
}

Status SegmentBuilder::Build(const std::string& path, SegmentMeta& meta_out) {
    std::vector<uint8_t> file;

    // Region 1: posting lists, in dictionary key order. std::map is ordered, so the
    // "sort by term then document" step is already done: by term because the map is
    // ordered, by document because local ids are assigned in insertion order.
    SegmentFooter footer;
    footer.postings_offset = 0;
    std::vector<std::pair<std::string, TermLocation>> locations;
    locations.reserve(postings_.size());

    for (const auto& [key, list] : postings_) {
        TermLocation location;
        location.offset = file.size();
        location.stats.document_frequency = static_cast<uint32_t>(list.size());
        uint64_t collection_frequency = 0;
        for (const Posting& p : list) collection_frequency += p.term_frequency;
        location.stats.collection_frequency = collection_frequency;

        std::string_view term;
        Field field = Field::kTitle;
        if (!SplitTermKey(key, term, field)) {
            return Status::Internal("dictionary key does not decode to a field and term");
        }
        MERIDIAN_TRY(postings::EncodeList(list, options_.positions[static_cast<size_t>(field)],
                                         file));
        location.length = file.size() - location.offset;
        locations.emplace_back(key, location);
    }
    footer.postings_length = file.size() - footer.postings_offset;

    // Region 2 and 3: the dictionary.
    DictionaryWriter writer;
    for (const auto& [key, location] : locations) {
        MERIDIAN_TRY(writer.Add(key, location));
    }
    std::vector<uint8_t> dict_blocks, dict_index;
    MERIDIAN_TRY(writer.Finish(dict_blocks, dict_index));

    footer.dict_blocks_offset = file.size();
    footer.dict_blocks_length = dict_blocks.size();
    file.insert(file.end(), dict_blocks.begin(), dict_blocks.end());

    footer.dict_index_offset = file.size();
    footer.dict_index_length = dict_index.size();
    file.insert(file.end(), dict_index.begin(), dict_index.end());

    // Region 4: document lengths, fixed width. Fixed rather than varint because BM25
    // needs every scored document's length and a fixed stride makes that an array
    // lookup; 4 bytes per document is nothing next to its postings.
    footer.doc_lengths_offset = file.size();
    for (const DocumentMeta& meta : documents_) storage::PutU32(meta.length, file);

    // Regions 5 and 6: the document records and their offsets. Offsets are fixed width
    // for the same reason: Document(local) must be one positioned read, not a scan.
    std::vector<uint8_t> doc_meta;
    std::vector<uint64_t> doc_offsets;
    doc_offsets.reserve(documents_.size());
    for (const DocumentMeta& meta : documents_) {
        doc_offsets.push_back(doc_meta.size());
        EncodeDocumentMeta(meta, doc_meta);
    }

    footer.doc_offsets_offset = file.size();
    for (uint64_t offset : doc_offsets) storage::PutU64(offset, file);

    footer.doc_meta_offset = file.size();
    footer.doc_meta_length = doc_meta.size();
    file.insert(file.end(), doc_meta.begin(), doc_meta.end());

    footer.term_count = writer.term_count();
    footer.document_count = document_count();
    footer.total_tokens = total_tokens_;
    EncodeSegmentFooter(footer, file);

    MERIDIAN_TRY(storage::WriteFileAtomic(path, file));

    meta_out.path = path;
    meta_out.document_count = footer.document_count;
    meta_out.deleted_count = 0;
    meta_out.term_count = footer.term_count;
    meta_out.bytes = file.size();
    meta_out.created_at = NowSeconds();
    return Status::Ok();
}

}  // namespace meridian::index
