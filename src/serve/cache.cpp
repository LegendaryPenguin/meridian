#include "cache.hpp"

#include <functional>
#include <string>

namespace meridian::serve {
namespace {

/// Combines a hash into an accumulator. The constant and shifts are boost's
/// hash_combine; what matters here is only that the result spreads across shards,
/// because a hash that clusters turns per-shard budgets into one overloaded shard.
void HashCombine(size_t& seed, size_t value) {
    seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

const char* ModeName(QueryMode mode) {
    switch (mode) {
        case QueryMode::kDisjunctive: return "or";
        case QueryMode::kConjunctive: return "and";
        case QueryMode::kPhrase: return "phrase";
    }
    return "?";
}

}  // namespace

size_t ResultCacheKeyHash::operator()(const ResultCacheKey& key) const {
    size_t seed = std::hash<std::string>{}(key.normalized_query);
    HashCombine(seed, std::hash<std::string>{}(key.filters));
    HashCombine(seed, std::hash<uint32_t>{}(key.top_k));
    HashCombine(seed, std::hash<uint64_t>{}(key.generation_version));
    return seed;
}

size_t PostingCacheKeyHash::operator()(const PostingCacheKey& key) const {
    size_t seed = std::hash<std::string>{}(key.term);
    HashCombine(seed, static_cast<size_t>(key.field));
    HashCombine(seed, std::hash<uint64_t>{}(key.segment_id));
    return seed;
}

std::string NormalizeQueryKey(const Query& query) {
    // '\x1f' (unit separator) as the delimiter rather than a space: a term containing a
    // space would otherwise let two different queries produce the same key, which is a
    // silent wrong-answer bug of exactly the kind this cache is supposed to avoid.
    std::string key;
    key.reserve(query.raw.size() + 16);
    key += ModeName(query.mode);
    for (const auto& term : query.terms) {
        key += '\x1f';
        key += term;
    }
    // Split literal: "\x1fck" would be read as one over-long hex escape.
    key += "\x1f" "ck=";
    key += std::to_string(query.candidate_k);
    if (query.embedding) {
        // The embedding itself is not hashed into the key. Two requests carrying the
        // same text and an embedding are assumed to carry the *same* embedding, because
        // it is produced deterministically from that text upstream. Hashing a few
        // hundred floats per lookup would cost more than the lookup saves; the flag
        // below still separates a hybrid query from a lexical-only one, which is the
        // difference that changes the answer.
        key += "\x1fhybrid";
    }
    return key;
}

std::string EncodeFilters(const Filters& filters) {
    if (filters.empty()) return {};
    std::string key;
    if (filters.language) key += "lang=" + *filters.language + '\x1f';
    if (filters.category) key += "cat=" + *filters.category + '\x1f';
    if (filters.published_after) key += "after=" + std::to_string(*filters.published_after) + '\x1f';
    if (filters.published_before) {
        key += "before=" + std::to_string(*filters.published_before) + '\x1f';
    }
    return key;
}

ResultCacheKey MakeResultCacheKey(const Query& query, uint64_t generation_version) {
    ResultCacheKey key;
    key.normalized_query = NormalizeQueryKey(query);
    key.filters = EncodeFilters(query.filters);
    key.top_k = query.top_k;
    key.generation_version = generation_version;
    return key;
}

size_t QueryResultCache::EstimateBytes(const SearchResponse& response) {
    size_t bytes = sizeof(SearchResponse);
    for (const auto& hit : response.hits) {
        bytes += sizeof(Hit) + hit.external_id.capacity();
        for (const auto& [name, value] : hit.features) {
            // The map node overhead dominates for small feature maps, so it is counted
            // rather than ignored; a response whose footprint is mostly hash nodes
            // would otherwise look free.
            bytes += name.capacity() + sizeof(value) + 48;
        }
    }
    return bytes;
}

QueryResultCache::QueryResultCache(size_t capacity_bytes, size_t shards)
    : cache_(capacity_bytes, shards, [](const CachedResponse& response) {
          return response ? EstimateBytes(*response) : sizeof(SearchResponse);
      }) {}

std::optional<CachedResponse> QueryResultCache::Lookup(const ResultCacheKey& key) {
    return cache_.Lookup(key);
}

void QueryResultCache::Insert(const ResultCacheKey& key, CachedResponse response) {
    if (!response) return;
    cache_.Insert(key, std::move(response));
}

size_t CachedPostings::EstimateBytes() const {
    size_t bytes = sizeof(CachedPostings) + postings.capacity() * sizeof(Posting);
    for (const auto& posting : postings) {
        bytes += posting.positions.capacity() * sizeof(Position);
    }
    return bytes;
}

PostingListCache::PostingListCache(size_t capacity_bytes, size_t shards)
    : cache_(capacity_bytes, shards, [](const CachedPostingList& list) {
          return list ? list->EstimateBytes() : sizeof(CachedPostings);
      }) {}

std::optional<CachedPostingList> PostingListCache::Lookup(const PostingCacheKey& key) {
    return cache_.Lookup(key);
}

void PostingListCache::Insert(const PostingCacheKey& key, CachedPostingList postings) {
    if (!postings) return;
    cache_.Insert(key, std::move(postings));
}

}  // namespace meridian::serve
