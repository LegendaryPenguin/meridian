#pragma once

// Two caches that look alike and are not.
//
// The query result cache and the posting list cache share an eviction policy and a
// locking scheme, and nothing else. Their *invalidation* stories are opposites, and
// that difference is the interesting content of this file:
//
//  - A result cache entry is a statement about the whole index ("for query Q, these
//    are the ten best documents"). That statement stops being true the moment a new
//    generation is published, so the generation version is part of the key.
//
//  - A posting cache entry is a statement about one immutable segment ("term T in
//    field F has these postings in segment S"). Segments are never modified after they
//    are written, so that statement never stops being true. No invalidation, no
//    versioning, no epoch checks — the index design pays for itself here.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "meridian/index.hpp"
#include "meridian/postings.hpp"
#include "meridian/retrieval.hpp"

namespace meridian::serve {

/// Counters reported as observed, with no smoothing or derived guesses.
///
/// `demotions` and `promotions` are here because they are the only way to tell whether
/// the segmented policy is doing anything: a cache with promotions near zero is
/// behaving as a plain LRU over its probationary segment and the extra machinery is
/// not earning its keep.
struct CacheStats {
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t insertions = 0;
    uint64_t updates = 0;
    uint64_t evictions = 0;
    uint64_t promotions = 0;
    uint64_t demotions = 0;

    /// Entries refused because one of them exceeded a whole shard's budget. Counted
    /// separately from evictions: admitting such an entry would evict everything else
    /// to store one item, which is worse than not caching it at all.
    uint64_t rejected_too_large = 0;

    double hit_rate() const {
        const uint64_t total = hits + misses;
        return total == 0 ? 0.0 : static_cast<double>(hits) / static_cast<double>(total);
    }
};

/// Sharded segmented-LRU cache, bounded by a cost budget rather than entry count.
///
/// **Eviction policy: segmented LRU (SLRU).** Entries enter a small probationary
/// segment and are only promoted into the large protected segment on a *second*
/// access. The reason is a measured property of search traffic: the query stream is a
/// long tail, and a run of queries that will never be repeated is not an anomaly but
/// the normal background. Under plain LRU each of those misses inserts at the MRU end
/// and pushes one genuinely hot entry out, so a few thousand one-shot queries evict
/// the working set that was serving most of the hits. Under SLRU the one-shot entries
/// churn within the probationary segment — they are never touched twice, so they never
/// reach the protected segment — and the hot set survives. The cost is one extra list
/// and the 20/80 split constant.
///
/// **Bounded by cost, not by entry count.** Posting lists differ in size by four
/// orders of magnitude between a rare term and a stopword. An entry-count bound sized
/// for the common case has no idea how much memory it is actually holding, and sized
/// for the worst case holds almost nothing.
///
/// **Lock granularity: one mutex per shard.** A single lock for the whole cache
/// serialises every query in the server behind a hash lookup, which is the failure this
/// class exists to avoid — at four workers it is a measurable throughput ceiling and at
/// thirty-two it is the dominant cost. Each shard owns its own mutex, its own lists and
/// its own slice of the budget, so independent keys never contend. Two further notes:
///
///  - `std::mutex`, not `std::shared_mutex`. A cache *hit* mutates recency order and
///    may promote across segments, so there is no read-only lookup path for a reader
///    lock to parallelise. A shared lock here would be pure overhead plus a
///    correctness trap for whoever assumed lookups were const.
///
///  - Per-shard budgets mean a skewed key distribution can evict from a full shard
///    while another shard has room. That is the price of not having a global lock; with
///    a decent hash and enough shards the imbalance is small, and it is strictly better
///    than the alternative of serialising everything to get perfect accounting.
template <typename Key, typename Value, typename Hash = std::hash<Key>>
class ShardedSlruCache {
public:
    using CostFn = std::function<size_t(const Value&)>;

    /// `capacity_cost` is the total budget in whatever unit `cost` returns (bytes, for
    /// both caches here). `shards` must be positive; it is rounded up to a power of two
    /// so shard selection is a mask rather than a division.
    ShardedSlruCache(size_t capacity_cost, size_t shards, CostFn cost)
        : cost_(std::move(cost)) {
        size_t n = 1;
        while (n < shards) n <<= 1;
        shard_mask_ = n - 1;
        shards_ = std::vector<std::unique_ptr<Shard>>(n);
        // At least one unit per shard, so a tiny test-sized cache still functions
        // rather than rejecting every entry.
        const size_t per_shard = capacity_cost / n == 0 ? 1 : capacity_cost / n;
        for (auto& shard : shards_) {
            shard = std::make_unique<Shard>();
            shard->capacity = per_shard;
            // 80% protected. Large enough that the hot set lives there rather than
            // being demoted by ordinary churn, small enough that the probationary
            // segment can still hold a plausible burst of first-time entries — a
            // probationary segment of one or two entries promotes nothing because the
            // second access never arrives before eviction.
            shard->protected_capacity = per_shard * 4 / 5;
        }
    }

    /// Returns the value and records the access, promoting on a second hit.
    std::optional<Value> Lookup(const Key& key) {
        Shard& shard = ShardFor(key);
        std::lock_guard<std::mutex> lock(shard.mutex);

        auto it = shard.index.find(key);
        if (it == shard.index.end()) {
            ++shard.stats.misses;
            return std::nullopt;
        }
        ++shard.stats.hits;

        Location& loc = it->second;
        if (loc.in_protected) {
            shard.protected_list.splice(shard.protected_list.begin(), shard.protected_list,
                                        loc.node);
        } else {
            // Second access: this key has proven it is not part of a scan.
            const size_t node_cost = loc.node->cost;
            shard.protected_list.splice(shard.protected_list.begin(), shard.probation_list,
                                        loc.node);
            shard.probation_cost -= node_cost;
            shard.protected_cost += node_cost;
            loc.in_protected = true;
            ++shard.stats.promotions;
            Rebalance(shard);
        }
        return loc.node->value;
    }

    /// Inserts or replaces. A replacement keeps the entry's segment: an update is not
    /// evidence about how hot the key is, so it must not reset the promotion decision.
    void Insert(const Key& key, Value value) {
        Shard& shard = ShardFor(key);
        const size_t entry_cost = cost_(value);

        std::lock_guard<std::mutex> lock(shard.mutex);
        if (entry_cost > shard.capacity) {
            ++shard.stats.rejected_too_large;
            return;
        }

        auto it = shard.index.find(key);
        if (it != shard.index.end()) {
            Location& loc = it->second;
            size_t& segment_cost = loc.in_protected ? shard.protected_cost : shard.probation_cost;
            segment_cost = segment_cost - loc.node->cost + entry_cost;
            loc.node->value = std::move(value);
            loc.node->cost = entry_cost;
            auto& list = loc.in_protected ? shard.protected_list : shard.probation_list;
            list.splice(list.begin(), list, loc.node);
            ++shard.stats.updates;
            Rebalance(shard);
            return;
        }

        shard.probation_list.push_front(Node{key, std::move(value), entry_cost});
        shard.probation_cost += entry_cost;
        shard.index.emplace(key, Location{shard.probation_list.begin(), false});
        ++shard.stats.insertions;
        Rebalance(shard);
    }

    /// Drops everything but keeps the counters. Used by tests and by an operator who
    /// wants a cold-cache latency number; the counters survive so the cache's history
    /// is not erased along with its contents.
    void Clear() {
        for (auto& shard : shards_) {
            std::lock_guard<std::mutex> lock(shard->mutex);
            shard->index.clear();
            shard->protected_list.clear();
            shard->probation_list.clear();
            shard->protected_cost = 0;
            shard->probation_cost = 0;
        }
    }

    bool Contains(const Key& key) const {
        Shard& shard = ShardFor(key);
        std::lock_guard<std::mutex> lock(shard.mutex);
        return shard.index.count(key) != 0;
    }

    CacheStats stats() const {
        CacheStats total;
        for (auto& shard : shards_) {
            std::lock_guard<std::mutex> lock(shard->mutex);
            total.hits += shard->stats.hits;
            total.misses += shard->stats.misses;
            total.insertions += shard->stats.insertions;
            total.updates += shard->stats.updates;
            total.evictions += shard->stats.evictions;
            total.promotions += shard->stats.promotions;
            total.demotions += shard->stats.demotions;
            total.rejected_too_large += shard->stats.rejected_too_large;
        }
        return total;
    }

    size_t entries() const {
        size_t n = 0;
        for (auto& shard : shards_) {
            std::lock_guard<std::mutex> lock(shard->mutex);
            n += shard->index.size();
        }
        return n;
    }

    size_t cost() const {
        size_t c = 0;
        for (auto& shard : shards_) {
            std::lock_guard<std::mutex> lock(shard->mutex);
            c += shard->protected_cost + shard->probation_cost;
        }
        return c;
    }

    size_t shard_count() const { return shards_.size(); }

private:
    struct Node {
        Key key;
        Value value;
        size_t cost;
    };
    using NodeList = std::list<Node>;

    struct Location {
        typename NodeList::iterator node;
        bool in_protected;
    };

    struct Shard {
        mutable std::mutex mutex;
        NodeList protected_list;  // front is most recently used
        NodeList probation_list;
        std::unordered_map<Key, Location, Hash> index;
        size_t capacity = 0;
        size_t protected_capacity = 0;
        size_t protected_cost = 0;
        size_t probation_cost = 0;
        CacheStats stats;
    };

    Shard& ShardFor(const Key& key) const {
        return *shards_[Hash{}(key) & shard_mask_];
    }

    /// Restores both invariants: protected within its own budget, and the shard within
    /// its total budget. Overflow from the protected segment is *demoted* into
    /// probation rather than dropped, so an entry that was hot and has gone quiet gets
    /// one more chance before it leaves — the behaviour that distinguishes SLRU from
    /// two independent LRUs.
    void Rebalance(Shard& shard) {
        while (shard.protected_cost > shard.protected_capacity && !shard.protected_list.empty()) {
            auto victim = std::prev(shard.protected_list.end());
            const size_t victim_cost = victim->cost;
            shard.probation_list.splice(shard.probation_list.begin(), shard.protected_list, victim);
            shard.protected_cost -= victim_cost;
            shard.probation_cost += victim_cost;
            shard.index.find(victim->key)->second.in_protected = false;
            ++shard.stats.demotions;
        }

        while (shard.protected_cost + shard.probation_cost > shard.capacity) {
            if (!shard.probation_list.empty()) {
                auto victim = std::prev(shard.probation_list.end());
                shard.probation_cost -= victim->cost;
                shard.index.erase(victim->key);
                shard.probation_list.erase(victim);
            } else if (!shard.protected_list.empty()) {
                // Only reachable when protected_capacity == capacity, i.e. a shard
                // budget too small to split. Evicting the protected LRU is still
                // better than exceeding the budget.
                auto victim = std::prev(shard.protected_list.end());
                shard.protected_cost -= victim->cost;
                shard.index.erase(victim->key);
                shard.protected_list.erase(victim);
            } else {
                break;
            }
            ++shard.stats.evictions;
        }
    }

    std::vector<std::unique_ptr<Shard>> shards_;
    size_t shard_mask_ = 0;
    CostFn cost_;
};

// ---------------------------------------------------------------------------
// Query result cache
// ---------------------------------------------------------------------------

/// Identity of a cached answer.
///
/// `generation_version` is the field that makes this cache correct, and it is worth
/// being explicit about what happens without it. A result cache entry is an assertion
/// about the state of the entire index. Publish a new generation — a flush of new
/// documents, a merge that reclaimed tombstones — and every cached answer becomes a
/// claim about an index that no longer exists. Omit the version from the key and the
/// engine keeps serving those answers: a document indexed an hour ago never appears, a
/// deleted document keeps appearing, and *nothing anywhere reports an error*. The
/// query succeeds, the latency looks excellent, and the results are wrong. There is no
/// alarm that catches it, because from the cache's point of view nothing unusual
/// happened. With the version in the key, a new generation simply misses, and the stale
/// entries age out of the cache on their own.
struct ResultCacheKey {
    /// Canonical form of everything about the query that changes the answer: the
    /// normalized terms, the query mode, and candidate_k. candidate_k belongs here
    /// because it changes which documents reach the reranker, so two queries that
    /// differ only in candidate_k genuinely have different correct answers.
    std::string normalized_query;

    /// Canonical encoding of the structured filters.
    std::string filters;

    uint32_t top_k = 0;
    uint64_t generation_version = 0;

    bool operator==(const ResultCacheKey& other) const {
        return top_k == other.top_k && generation_version == other.generation_version &&
               normalized_query == other.normalized_query && filters == other.filters;
    }
};

struct ResultCacheKeyHash {
    size_t operator()(const ResultCacheKey& key) const;
};

/// Canonical key text for a parsed query. Uses the already-normalized terms rather
/// than the raw string, so "Deep  Learning" and "deep learning" share an entry — the
/// normalization that retrieval applies is exactly the normalization the cache must
/// apply, and deriving it from `terms` means it cannot drift from the parser's.
std::string NormalizeQueryKey(const Query& query);
std::string EncodeFilters(const Filters& filters);

/// Build the full key for a query against a pinned generation.
ResultCacheKey MakeResultCacheKey(const Query& query, uint64_t generation_version);

/// The cached response is shared and immutable. Immutable because handing a mutable
/// response to two concurrent callers invites one of them to edit the other's results,
/// and shared because the alternative is copying every hit list on every hit, which
/// would eat a large part of what the cache saves.
using CachedResponse = std::shared_ptr<const SearchResponse>;

class QueryResultCache {
public:
    /// `capacity_bytes` bounds the estimated footprint of the stored responses.
    explicit QueryResultCache(size_t capacity_bytes = 8u << 20, size_t shards = 16);

    std::optional<CachedResponse> Lookup(const ResultCacheKey& key);
    void Insert(const ResultCacheKey& key, CachedResponse response);

    /// Presence without recording an access. Tests use it to assert what a cache still
    /// holds without the assertion itself promoting the entry and changing the answer.
    bool Contains(const ResultCacheKey& key) const { return cache_.Contains(key); }

    CacheStats stats() const { return cache_.stats(); }
    size_t entries() const { return cache_.entries(); }
    size_t bytes() const { return cache_.cost(); }
    size_t shard_count() const { return cache_.shard_count(); }
    void Clear() { cache_.Clear(); }

    /// Estimated heap footprint of a response. Approximate on purpose: the point of
    /// the budget is to stop the cache growing without bound, and an estimate that is
    /// within a factor of two does that as well as an exact figure would.
    static size_t EstimateBytes(const SearchResponse& response);

private:
    ShardedSlruCache<ResultCacheKey, CachedResponse, ResultCacheKeyHash> cache_;
};

// ---------------------------------------------------------------------------
// Posting list cache
// ---------------------------------------------------------------------------

/// Identity of a cached posting list.
///
/// No version field, and that absence is a design consequence rather than an omission.
/// A segment is immutable once written: a new document goes into a new segment, a
/// delete writes a tombstone elsewhere, and a merge produces a *new* segment with a new
/// id and leaves its inputs untouched. So (term, field, segment) names bytes that can
/// never change. Entries stay valid across every generation that contains the segment,
/// survive flushes and merges untouched, and need no epoch check on the read path. When
/// a merge eventually retires a segment, its entries simply stop being looked up and
/// fall out of the cache by LRU. This is the clearest payoff of immutable segments:
/// the cache that would otherwise need the most careful invalidation logic in the
/// system needs none.
struct PostingCacheKey {
    std::string term;
    Field field = Field::kBody;
    uint64_t segment_id = 0;

    bool operator==(const PostingCacheKey& other) const {
        return segment_id == other.segment_id && field == other.field && term == other.term;
    }
};

struct PostingCacheKeyHash {
    size_t operator()(const PostingCacheKey& key) const;
};

/// A decoded posting list plus the term statistics that came with it.
///
/// Decoded rather than raw bytes: the cache exists to skip the varint decode, not the
/// read. Caching the compressed bytes would save the I/O and leave the per-query decode
/// cost in place, which is the larger of the two for a hot term already in page cache.
struct CachedPostings {
    std::vector<Posting> postings;
    TermStats stats;

    size_t EstimateBytes() const;
};

using CachedPostingList = std::shared_ptr<const CachedPostings>;

class PostingListCache {
public:
    explicit PostingListCache(size_t capacity_bytes = 64u << 20, size_t shards = 16);

    std::optional<CachedPostingList> Lookup(const PostingCacheKey& key);
    void Insert(const PostingCacheKey& key, CachedPostingList postings);

    bool Contains(const PostingCacheKey& key) const { return cache_.Contains(key); }

    CacheStats stats() const { return cache_.stats(); }
    size_t entries() const { return cache_.entries(); }
    size_t bytes() const { return cache_.cost(); }
    size_t shard_count() const { return cache_.shard_count(); }
    void Clear() { cache_.Clear(); }

private:
    ShardedSlruCache<PostingCacheKey, CachedPostingList, PostingCacheKeyHash> cache_;
};

}  // namespace meridian::serve
