#pragma once

// Posting lists: the data structure the whole lexical side is built on.
//
// A posting list maps one term to the documents containing it. The design decisions
// that matter are all about *access pattern*, not storage size:
//
//  - Lists are kept in ascending document-ID order, because query evaluation is a
//    merge over several lists and a merge needs sorted inputs. Sorted order is also
//    what makes delta encoding effective, so the compression is a consequence of the
//    access pattern rather than a separate goal.
//
//  - Document IDs are delta-encoded and variable-byte compressed. A list of 10
//    million IDs stored as fixed 32-bit integers is 40 MB; stored as gaps it is a
//    few bytes per posting, and the gaps are small precisely because the list is
//    sorted and dense.
//
//  - Long lists carry skip information, so a conjunctive query can jump over ranges
//    of document IDs rather than decoding every posting. Without skips, `A AND B`
//    where A has ten million postings and B has ten costs ten million decodes.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "meridian/status.hpp"

namespace meridian {

/// Internal document identifier. Dense and assigned at index time, not the
/// document's external key — dense IDs are what make delta encoding small and let a
/// posting list be a flat array rather than a hash table.
using DocId = uint32_t;

/// Sentinel meaning "past the end of this list". Chosen as the maximum DocId so a
/// comparison against it needs no special case in the merge loops.
constexpr DocId kNoMoreDocs = UINT32_MAX;

/// Position of a term occurrence within a document, used for phrase matching.
using Position = uint32_t;

/// One posting: a document, how often the term appears, and where.
///
/// Positions are optional per field because they roughly triple index size and only
/// phrase and proximity scoring reads them. Making that a per-field decision rather
/// than a global one is the difference between an index that fits and one that does
/// not.
struct Posting {
    DocId doc = 0;
    uint32_t term_frequency = 0;
    std::vector<Position> positions;
};

/// Statistics a term carries, which BM25 needs before it can score anything.
struct TermStats {
    /// Number of documents containing the term. This is the IDF input, and it is why
    /// an index has to be built before it can rank: document frequency is a property
    /// of the corpus, not of the query.
    uint32_t document_frequency = 0;

    /// Total occurrences across the corpus. Not used by BM25 but useful for
    /// diagnosing a term that is frequent in few documents.
    uint64_t collection_frequency = 0;
};

/// Corpus-wide statistics. BM25 cannot be computed without these, which is the
/// concrete reason "search" is an index problem rather than a matching problem.
struct CorpusStats {
    uint32_t document_count = 0;
    double average_document_length = 0.0;
    uint64_t total_tokens = 0;
};

/// A skip entry: at this document, the encoded list continues at this byte offset.
///
/// Stored every N postings. N trades index size against seek granularity, and the
/// implementation states its choice and the reasoning.
struct SkipEntry {
    DocId doc = 0;
    uint64_t byte_offset = 0;
    /// Running term-frequency-decode state at this point, so a seek can resume
    /// without replaying the list from the start.
    uint32_t postings_before = 0;
};

/// Iterator over one term's postings.
///
/// The interface is deliberately the classic one — `next`, `advance`, `doc` — because
/// every query evaluation strategy here is expressible on it, and because `advance`
/// is where skipping earns its keep: it must be able to jump without decoding the
/// postings it passes over.
class PostingIterator {
public:
    virtual ~PostingIterator() = default;

    /// Current document, or kNoMoreDocs when exhausted.
    ///
    /// A freshly opened iterator is **already positioned on its first posting**, so a walk
    /// needs no priming call:
    ///
    ///     for (DocId d = it->doc(); d != kNoMoreDocs; d = it->next()) { ... }
    ///
    /// Stated because the other convention — positioned *before* the first posting, so the
    /// walk opens with `next()` — is at least as common, and mixing them does not fail
    /// loudly: it silently drops or double-counts the first document of every list.
    virtual DocId doc() const = 0;

    /// Advances to the next posting.
    virtual DocId next() = 0;

    /// Advances to the first document >= target. Never moves backwards: a target at or
    /// below the current document leaves the iterator where it is.
    ///
    /// This is the operation skip lists exist for. A correct-but-slow implementation
    /// can loop on next(); an implementation that uses skips must land on exactly the
    /// same document, which is why the tests compare the two against each other.
    virtual DocId advance(DocId target) = 0;

    /// Term frequency in the current document.
    virtual uint32_t term_frequency() const = 0;

    /// Positions in the current document. Empty when the field was indexed without
    /// positions.
    virtual std::span<const Position> positions() const = 0;

    /// Documents remaining, as an upper bound. Used to order a conjunctive merge
    /// cheapest-list-first, which is the single biggest win in query evaluation.
    virtual uint32_t size_hint() const = 0;

    /// Why the walk stopped.
    ///
    /// `doc() == kNoMoreDocs` means the iterator has stopped, and on its own it cannot say
    /// whether the list ended or a block failed its checksum. Those must be
    /// distinguishable. A corrupt block that reads as "no more documents" turns into a
    /// short result list — a query that silently returns four of its nine matches, with
    /// nothing anywhere reporting a fault. That is the exact failure `status.hpp` exists to
    /// prevent, and it is worse here than an error would be, because the caller has no way
    /// to notice it happened.
    ///
    /// Contract: `kOk` until something fails, latched thereafter. A caller that walks to
    /// exhaustion must check this before trusting the list to be complete.
    virtual Status status() const { return Status::Ok(); }

    /// Counters that make the skip-list claim measurable rather than asserted.
    struct Stats {
        uint64_t postings_decoded = 0;
        uint64_t skips_taken = 0;
        uint64_t blocks_read = 0;
    };
    virtual Stats stats() const = 0;
};

/// Variable-byte encoding for a sorted sequence of document IDs.
///
/// Delta-then-varint rather than a fancier scheme: it is simple enough to be obviously
/// correct, decodes fast, and the project's measurements are about access patterns
/// rather than about squeezing the last byte. The benchmarks report what it achieves
/// against a fixed-width baseline so the tradeoff is visible.
namespace varint {

/// Appends one value. Returns bytes written.
size_t Encode(uint64_t value, std::vector<uint8_t>& out);

/// Decodes one value starting at `offset`, advancing it. Returns false on a
/// truncated or over-long sequence — a corrupt posting list must be detected rather
/// than decoded into a plausible wrong document ID.
bool Decode(std::span<const uint8_t> in, size_t& offset, uint64_t& value);

/// Encodes a sorted sequence as deltas. Rejects unsorted input, because an unsorted
/// list silently breaks every merge downstream and the failure surfaces far away.
Status EncodeSorted(std::span<const DocId> docs, std::vector<uint8_t>& out);

}  // namespace varint

}  // namespace meridian
