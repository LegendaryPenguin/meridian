#pragma once

// A hand-built in-memory index, for testing retrieval and ranking.
//
// Deliberately independent of src/postings/ and src/storage/: those build posting lists
// out of files and byte sources, and a retrieval test that has to write a segment to
// disk to ask "does AND order its iterators by size" is testing two things and will fail
// for the wrong reason. What the retrieval path actually depends on is the
// `SegmentReader` and `PostingIterator` contracts, so that is what this implements.
//
// The iterator here has a real skip table, because the single most important claim in
// this area — that a conjunctive merge ordered rarest-first decodes fewer postings — is
// only measurable if `advance` can genuinely jump. An iterator whose `advance` looped on
// `next` would make rarest-first and commonest-first cost the same and the test would
// pass while proving nothing.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <span>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../src/query/tokenize.hpp"
#include "meridian/index.hpp"
#include "meridian/postings.hpp"
#include "meridian/retrieval.hpp"

namespace meridian {

// ---------------------------------------------------------------------------
// Generation, provided here only until src/index/ lands it.
//
namespace testfixture {

/// Postings for one term in one field, ascending by document.
using PostingList = std::vector<Posting>;

/// Skip interval. Eight is small for a real index and right for a test: it means a
/// 200-posting list has 25 skip entries, so `advance` has something to jump over at the
/// corpus sizes a unit test can build by hand.
constexpr size_t kSkipInterval = 8;

class MemoryPostingIterator final : public PostingIterator {
public:
    explicit MemoryPostingIterator(const PostingList& list) : list_(&list) {
        // Positioned on the first posting at open, matching src/postings/'s documented
        // convention.
        if (!list_->empty()) ++stats_.postings_decoded;
    }

    DocId doc() const override {
        return index_ < list_->size() ? (*list_)[index_].doc : kNoMoreDocs;
    }

    DocId next() override {
        if (index_ >= list_->size()) return kNoMoreDocs;
        ++index_;
        if (index_ < list_->size()) ++stats_.postings_decoded;
        return doc();
    }

    DocId advance(DocId target) override {
        if (index_ >= list_->size()) return kNoMoreDocs;
        if ((*list_)[index_].doc >= target) return doc();  // never moves backwards

        // Jump whole blocks whose last document is still below the target. The postings
        // inside a skipped block are never touched, which is the entire point: it is
        // what makes the decode counters differ between iterator orders.
        while (index_ + kSkipInterval < list_->size() &&
               (*list_)[index_ + kSkipInterval].doc <= target) {
            index_ += kSkipInterval;
            ++stats_.skips_taken;
            ++stats_.postings_decoded;  // the posting landed on is read
        }
        while (index_ < list_->size() && (*list_)[index_].doc < target) {
            ++index_;
            if (index_ < list_->size()) ++stats_.postings_decoded;
        }
        return doc();
    }

    uint32_t term_frequency() const override {
        return index_ < list_->size() ? (*list_)[index_].term_frequency : 0;
    }

    std::span<const Position> positions() const override {
        if (index_ >= list_->size()) return {};
        return (*list_)[index_].positions;
    }

    uint32_t size_hint() const override {
        return static_cast<uint32_t>(list_->size() - std::min(index_, list_->size()));
    }

    Stats stats() const override { return stats_; }

private:
    const PostingList* list_;
    size_t index_ = 0;
    Stats stats_;
};

/// A posting list that fails partway through, the way a corrupt block does.
///
/// Exists to test the one failure with no natural symptom. A real decode failure stops the
/// walk and leaves `doc()` at `kNoMoreDocs`, which is byte-for-byte what the end of a short
/// list looks like — so the only thing distinguishing "finished" from "broke" is
/// `status()`, and the only way to know a caller consults it is to hand one a list that
/// breaks and check the caller complains.
class FailingPostingIterator final : public PostingIterator {
public:
    FailingPostingIterator(const PostingList& list, size_t fail_after)
        : list_(&list), fail_after_(fail_after) {}

    DocId doc() const override {
        if (!error_.ok() || index_ >= list_->size()) return kNoMoreDocs;
        return (*list_)[index_].doc;
    }

    DocId next() override {
        if (!error_.ok()) return kNoMoreDocs;
        if (index_ >= list_->size()) return kNoMoreDocs;
        ++index_;
        ++decoded_;
        if (decoded_ >= fail_after_) {
            // Exactly how the real iterator behaves: latch the reason, then report
            // exhaustion. It must not throw — the interface returns a DocId.
            error_ = Status::Corruption("block 2 failed its checksum");
            return kNoMoreDocs;
        }
        return doc();
    }

    DocId advance(DocId target) override {
        while (doc() != kNoMoreDocs && doc() < target) next();
        return doc();
    }

    uint32_t term_frequency() const override {
        return index_ < list_->size() ? (*list_)[index_].term_frequency : 0;
    }

    std::span<const Position> positions() const override {
        if (!error_.ok() || index_ >= list_->size()) return {};
        return (*list_)[index_].positions;
    }

    uint32_t size_hint() const override {
        return static_cast<uint32_t>(list_->size() - std::min(index_, list_->size()));
    }

    Status status() const override { return error_; }
    Stats stats() const override { return stats_; }

private:
    const PostingList* list_;
    size_t fail_after_;
    size_t index_ = 0;
    size_t decoded_ = 0;
    Status error_;
    Stats stats_;
};

class MemorySegmentReader final : public SegmentReader {
public:
    using FieldDictionary = std::unordered_map<std::string, PostingList>;

    MemorySegmentReader(SegmentMeta meta, std::vector<DocumentMeta> docs,
                        std::vector<FieldDictionary> fields, std::vector<bool> deleted)
        : meta_(std::move(meta)),
          docs_(std::move(docs)),
          fields_(std::move(fields)),
          deleted_(std::move(deleted)) {}

    std::unique_ptr<PostingIterator> Postings(std::string_view term, Field field) const override {
        const FieldDictionary& dict = fields_[static_cast<size_t>(field)];
        auto it = dict.find(std::string(term));
        if (it == dict.end() || it->second.empty()) return nullptr;
        if (!corrupt_term_.empty() && it->first == corrupt_term_) {
            return std::make_unique<FailingPostingIterator>(it->second, fail_after_);
        }
        return std::make_unique<MemoryPostingIterator>(it->second);
    }

    /// Makes one term's lists fail after `fail_after` decodes, in every field.
    void CorruptTerm(std::string term, size_t fail_after) {
        corrupt_term_ = std::move(term);
        fail_after_ = fail_after;
    }

    std::string corrupt_term_;
    size_t fail_after_ = 0;

public:
    std::optional<TermStats> Term(std::string_view term, Field field) const override {
        const FieldDictionary& dict = fields_[static_cast<size_t>(field)];
        auto it = dict.find(std::string(term));
        if (it == dict.end()) return std::nullopt;
        TermStats stats;
        stats.document_frequency = static_cast<uint32_t>(it->second.size());
        for (const Posting& p : it->second) stats.collection_frequency += p.term_frequency;
        return stats;
    }

    Result<DocumentMeta> Document(DocId local) const override {
        if (local >= docs_.size()) return Status::NotFound("no such local doc");
        return docs_[local];
    }

    bool IsDeleted(DocId local) const override {
        return local < deleted_.size() && deleted_[local];
    }

    const SegmentMeta& meta() const override { return meta_; }
    uint64_t bytes_read() const override { return 0; }

private:
    SegmentMeta meta_;
    std::vector<DocumentMeta> docs_;
    std::vector<FieldDictionary> fields_;
    std::vector<bool> deleted_;
};

/// One document, as raw text. The text is analysed with the *shared* tokenizer, which is
/// what makes an end-to-end test meaningful: if the index and query paths disagreed, this
/// fixture would reproduce the disagreement rather than hide it.
struct Doc {
    std::string external_id;
    std::string title;
    std::string body;
    std::string tags;
    std::string language = "en";
    std::string category = "general";
    int64_t published_at = 0;
    double popularity = 0.0;
    bool deleted = false;
};

class CorpusBuilder {
public:
    explicit CorpusBuilder(query::TokenizerOptions options = {}) : tokenizer_(options) {
        segments_.emplace_back();
    }

    CorpusBuilder& Add(Doc doc) {
        segments_.back().push_back(std::move(doc));
        return *this;
    }

    /// Starts a new segment. Multi-segment corpora are the interesting case: DocIds are
    /// segment-local and `doc_base` is what makes them global, and a merge bug across
    /// that boundary is invisible in a single-segment test.
    CorpusBuilder& NewSegment() {
        segments_.emplace_back();
        return *this;
    }

    std::shared_ptr<Generation> Build(uint64_t version = 1) {
        std::vector<std::shared_ptr<SegmentReader>> readers;
        DocId base = 0;
        uint64_t total_tokens = 0;
        uint32_t total_docs = 0;
        uint64_t segment_id = 1;

        for (const std::vector<Doc>& docs : segments_) {
            if (docs.empty()) continue;

            std::vector<DocumentMeta> metas;
            std::vector<bool> deleted;
            std::vector<MemorySegmentReader::FieldDictionary> fields(
                static_cast<size_t>(Field::kFieldCount));
            uint32_t deleted_count = 0;

            for (size_t i = 0; i < docs.size(); ++i) {
                const Doc& doc = docs[i];
                uint32_t length = 0;
                const std::pair<Field, const std::string*> field_text[] = {
                    {Field::kTitle, &doc.title},
                    {Field::kBody, &doc.body},
                    {Field::kTags, &doc.tags},
                };
                for (const auto& [field, text] : field_text) {
                    // The shared entry point. Nothing in this fixture has its own
                    // analysis.
                    std::vector<std::string> tokens =
                        query::TokenizeForIndex(*text, tokenizer_);
                    length += static_cast<uint32_t>(tokens.size());

                    std::unordered_map<std::string, std::vector<Position>> by_term;
                    for (size_t p = 0; p < tokens.size(); ++p) {
                        by_term[tokens[p]].push_back(static_cast<Position>(p));
                    }
                    for (auto& [term, positions] : by_term) {
                        Posting posting;
                        posting.doc = static_cast<DocId>(i);
                        posting.term_frequency = static_cast<uint32_t>(positions.size());
                        posting.positions = std::move(positions);
                        fields[static_cast<size_t>(field)][term].push_back(std::move(posting));
                    }
                }

                DocumentMeta meta;
                meta.external_id = doc.external_id;
                meta.language = doc.language;
                meta.category = doc.category;
                meta.published_at = doc.published_at;
                meta.length = length;
                meta.popularity = doc.popularity;
                metas.push_back(std::move(meta));
                deleted.push_back(doc.deleted);
                if (doc.deleted) ++deleted_count;

                total_tokens += length;
                ++total_docs;
            }

            // Posting lists must be ascending by document; they are built in document
            // order above, so this only guards against a future edit.
            for (auto& dict : fields) {
                for (auto& [term, list] : dict) {
                    std::sort(list.begin(), list.end(),
                              [](const Posting& a, const Posting& b) { return a.doc < b.doc; });
                }
            }

            SegmentMeta meta;
            meta.id = segment_id++;
            meta.doc_base = base;
            meta.document_count = static_cast<uint32_t>(docs.size());
            meta.deleted_count = deleted_count;
            base += meta.document_count;

            auto reader = std::make_shared<MemorySegmentReader>(
                std::move(meta), std::move(metas), std::move(fields), std::move(deleted));
            if (!corrupt_term_.empty()) reader->CorruptTerm(corrupt_term_, fail_after_);
            readers.push_back(std::move(reader));
        }

        CorpusStats stats;
        stats.document_count = total_docs;
        stats.total_tokens = total_tokens;
        stats.average_document_length =
            total_docs > 0 ? static_cast<double>(total_tokens) / total_docs : 0.0;

        return std::make_shared<Generation>(version, std::move(readers), stats);
    }

    /// Makes every segment's lists for `term` fail after `fail_after` decodes, simulating a
    /// corrupt block. See `FailingPostingIterator`.
    CorpusBuilder& CorruptTerm(std::string term, size_t fail_after) {
        corrupt_term_ = std::move(term);
        fail_after_ = fail_after;
        return *this;
    }

private:
    query::Tokenizer tokenizer_;
    std::vector<std::vector<Doc>> segments_;
    std::string corrupt_term_;
    size_t fail_after_ = 0;
};

/// Looks up a hit's position in a result list by external id, or -1.
template <typename HitLike>
int RankOf(const std::vector<HitLike>& hits, const std::string& external_id) {
    for (size_t i = 0; i < hits.size(); ++i) {
        if (hits[i].external_id == external_id) return static_cast<int>(i);
    }
    return -1;
}

inline bool Near(double a, double b, double tolerance = 1e-6) {
    return std::abs(a - b) <= tolerance;
}

}  // namespace testfixture
}  // namespace meridian
