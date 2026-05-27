// Cache tests.
//
// The one that matters most is ResultCacheDoesNotSurviveNewGeneration. Everything else
// here is about bounds and counters; that one is about whether the engine can serve a
// stale answer, which is the only cache bug that produces no error, no slow query and no
// log line — just wrong results.

#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include "../src/serve/cache.hpp"
#include "../src/serve/reference_index.hpp"
#include "testing.hpp"

using namespace meridian;
using namespace meridian::serve;

namespace {

ReferenceIndex::Document MakeDocument(const std::string& id, const std::string& title,
                                      const std::string& body) {
    ReferenceIndex::Document doc;
    doc.meta.external_id = id;
    doc.meta.language = "en";
    doc.meta.category = "systems";
    doc.tokens[Field::kTitle] = Tokenize(title);
    doc.tokens[Field::kBody] = Tokenize(body);
    return doc;
}

/// A corpus with a stable vocabulary, so a query's term set is known and the posting
/// cache's lookup count is predictable.
void SeedCorpus(ReferenceIndex& index, int documents, const std::string& prefix = "d") {
    for (int i = 0; i < documents; ++i) {
        index.Add(MakeDocument(prefix + std::to_string(i), "segment merge generation " +
                                                               std::to_string(i),
                               "an inverted index maps terms to documents in sorted order"));
    }
    index.Flush();
}

Query MakeQuery(const std::string& text, uint32_t top_k = 5) {
    Query query;
    query.raw = text;
    query.top_k = top_k;
    query.candidate_k = 50;
    query.deadline_ms = 60'000;  // generous: these tests are about caching, not deadlines
    return query;
}

CachedResponse MakeResponse(const std::string& id, double score) {
    auto response = std::make_shared<SearchResponse>();
    Hit hit;
    hit.doc = 7;
    hit.external_id = id;
    hit.score = score;
    hit.features["bm25"] = score;
    response->hits.push_back(hit);
    response->total_candidates = 1;
    return response;
}

CachedPostingList MakePostings(size_t count) {
    auto postings = std::make_shared<CachedPostings>();
    postings->postings.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        postings->postings.push_back(Posting{static_cast<DocId>(i), 1, {}});
    }
    postings->stats.document_frequency = static_cast<uint32_t>(count);
    return postings;
}

}  // namespace

MERIDIAN_TEST(ResultCacheHitReturnsTheSameResponse) {
    QueryResultCache cache(1 << 20, 4);
    ResultCacheKey key{"\x1for" "\x1fterm", "", 10, 3};

    auto stored = MakeResponse("doc-1", 2.5);
    cache.Insert(key, stored);

    auto first = cache.Lookup(key);
    CHECK(first.has_value());
    // Pointer identity, not just equal contents: the cache hands out a shared handle to
    // one immutable response rather than copying the hit list on every hit.
    CHECK(first->get() == stored.get());
    CHECK_EQ((*first)->hits.size(), 1u);
    CHECK_EQ((*first)->hits[0].external_id, std::string("doc-1"));
    CHECK_EQ((*first)->hits[0].score, 2.5);

    auto second = cache.Lookup(key);
    CHECK(second.has_value());
    CHECK(second->get() == first->get());

    const auto stats = cache.stats();
    CHECK_EQ(stats.hits, 2u);
    CHECK_EQ(stats.misses, 0u);
    CHECK_EQ(stats.insertions, 1u);
}

MERIDIAN_TEST(ResultCacheKeyDistinguishesEverythingThatChangesTheAnswer) {
    QueryResultCache cache(1 << 20, 4);
    const ResultCacheKey base{"\x1for" "\x1fterm", "", 10, 1};
    cache.Insert(base, MakeResponse("doc-1", 1.0));

    ResultCacheKey other_top_k = base;
    other_top_k.top_k = 20;
    CHECK(!cache.Lookup(other_top_k).has_value());

    ResultCacheKey other_filters = base;
    other_filters.filters = "lang=en\x1f";
    CHECK(!cache.Lookup(other_filters).has_value());

    ResultCacheKey other_generation = base;
    other_generation.generation_version = 2;
    CHECK(!cache.Lookup(other_generation).has_value());

    CHECK(cache.Lookup(base).has_value());
}

MERIDIAN_TEST(NormalizedKeyIgnoresSpacingAndCase) {
    ReferenceParseStage parser;

    auto first = parser.Parse(MakeQuery("Deep   LEARNING!"));
    auto second = parser.Parse(MakeQuery("deep learning"));
    CHECK(first.ok());
    CHECK(second.ok());
    CHECK_EQ(NormalizeQueryKey(first.value()), NormalizeQueryKey(second.value()));

    // Mode changes the answer, so it must change the key: "a AND b" and "a OR b" have
    // the same terms and different results.
    Query conjunctive = first.take();
    conjunctive.mode = QueryMode::kConjunctive;
    CHECK_NE(NormalizeQueryKey(conjunctive), NormalizeQueryKey(second.value()));

    // candidate_k changes which documents reach the reranker, so it is in the key too.
    Query wider = second.take();
    wider.candidate_k = 1000;
    CHECK_NE(NormalizeQueryKey(wider), NormalizeQueryKey(conjunctive));
}

// The correctness test. A cached answer must not outlive the index state it describes.
MERIDIAN_TEST(ResultCacheDoesNotSurviveNewGeneration) {
    PipelineOptions options;
    options.parallel_paths = false;
    ReferenceEngine engine(options);
    SeedCorpus(engine.index, 8);

    // top_k above the corpus size, so "did the new document appear" is a question about
    // generations rather than about whether it scored into a short list.
    const Query query = MakeQuery("segment merge", 20);

    const GenerationPin first_pin = engine.index.Acquire();
    auto cold = engine.pipeline.Run(query, first_pin, Deadline::Never());
    CHECK(cold.ok());
    CHECK(!cold.value().trace.cache_hit_results);
    const size_t cold_hits = cold.value().hits.size();

    auto warm = engine.pipeline.Run(query, first_pin, Deadline::Never());
    CHECK(warm.ok());
    CHECK(warm.value().trace.cache_hit_results);
    CHECK_EQ(warm.value().hits.size(), cold_hits);

    // Publish a new generation: one new document, which this query matches.
    engine.index.Add(MakeDocument("new-doc", "segment merge generation newest",
                                  "a freshly indexed document that the old answer cannot contain"));
    const uint64_t new_version = engine.index.Flush();
    CHECK(new_version > first_pin.version);

    const GenerationPin second_pin = engine.index.Acquire();
    CHECK_EQ(second_pin.version, new_version);

    auto after = engine.pipeline.Run(query, second_pin, Deadline::Never());
    CHECK(after.ok());
    // The miss is the assertion. Had the generation version been left out of the key,
    // this would be a hit and the new document would be invisible for as long as the
    // entry survived — with nothing anywhere reporting a problem.
    CHECK(!after.value().trace.cache_hit_results);
    CHECK_EQ(after.value().trace.generation_version, new_version);

    bool found_new_document = false;
    for (const auto& hit : after.value().hits) {
        if (hit.external_id == "new-doc") found_new_document = true;
    }
    CHECK(found_new_document);

    // The old entry is still in the cache and still unreachable: it is keyed to a
    // generation nobody will pin again, and it ages out by eviction rather than by an
    // invalidation pass. That is the mechanism, stated so it is not mistaken for a leak.
    const auto old_key = MakeResultCacheKey(ReferenceParseStage().Parse(query).take(),
                                            first_pin.version);
    CHECK(engine.result_cache.Contains(old_key));
}

MERIDIAN_TEST(PostingCacheHitsOnHotTerm) {
    PipelineOptions options;
    options.parallel_paths = false;
    // Result caching off: with it on, the second identical query never reaches the
    // posting cache at all, and the test would pass while measuring nothing.
    options.use_result_cache = false;
    ReferenceEngine engine(options);
    SeedCorpus(engine.index, 16);

    const Query query = MakeQuery("segment merge generation");
    const GenerationPin pin = engine.index.Acquire();

    auto first = engine.pipeline.Run(query, pin, Deadline::Never());
    CHECK(first.ok());
    const auto cold = engine.posting_cache.stats();
    CHECK_EQ(cold.hits, 0u);
    CHECK(cold.misses > 0u);
    CHECK(cold.insertions > 0u);
    CHECK(!first.value().trace.cache_hit_postings);

    auto second = engine.pipeline.Run(query, pin, Deadline::Never());
    CHECK(second.ok());
    CHECK(second.value().trace.cache_hit_postings);

    const auto warm = engine.posting_cache.stats();
    CHECK_EQ(warm.hits, cold.insertions);
    CHECK_EQ(warm.insertions, cold.insertions);  // nothing re-read, nothing re-inserted
    CHECK_EQ(warm.evictions, 0u);
    CHECK_EQ(second.value().hits.size(), first.value().hits.size());
}

// Segment immutability asserted rather than assumed: entries keyed on a segment that a
// new generation did not touch stay valid, and are reused across the generation boundary.
MERIDIAN_TEST(PostingCacheEntriesSurviveNewGenerations) {
    PipelineOptions options;
    options.parallel_paths = false;
    options.use_result_cache = false;
    ReferenceEngine engine(options);
    SeedCorpus(engine.index, 12);

    const Query query = MakeQuery("segment merge generation");
    const GenerationPin first_pin = engine.index.Acquire();
    CHECK(engine.pipeline.Run(query, first_pin, Deadline::Never()).ok());

    auto snapshot = engine.index.Current();
    CHECK_EQ(snapshot->segments.size(), 1u);
    const uint64_t original_segment = snapshot->segments[0]->id;
    const PostingCacheKey hot{"segment", Field::kTitle, original_segment};
    CHECK(engine.posting_cache.Contains(hot));

    const auto before = engine.posting_cache.stats();

    // A flush publishes a new generation containing a *new* segment. The original segment
    // is byte-for-byte what it was, so its cached posting lists are still correct and no
    // invalidation is needed or performed.
    engine.index.Add(MakeDocument("extra", "segment merge generation extra", "another document"));
    const uint64_t new_version = engine.index.Flush();
    const GenerationPin second_pin = engine.index.Acquire();
    CHECK_EQ(second_pin.version, new_version);
    CHECK(second_pin.version > first_pin.version);

    auto after = engine.pipeline.Run(query, second_pin, Deadline::Never());
    CHECK(after.ok());

    const auto later = engine.posting_cache.stats();
    // The old segment's lists were served from cache under the new generation.
    CHECK(later.hits > before.hits);
    CHECK(engine.posting_cache.Contains(hot));
    CHECK_EQ(later.evictions, 0u);

    // And the new segment's lists were read and cached, which is what the new misses are.
    CHECK(later.misses > before.misses);
    CHECK(later.insertions > before.insertions);

    auto new_snapshot = engine.index.Current();
    CHECK_EQ(new_snapshot->segments.size(), 2u);
    const uint64_t added_segment = new_snapshot->segments[1]->id;
    CHECK_NE(added_segment, original_segment);
    CHECK(engine.posting_cache.Contains(PostingCacheKey{"segment", Field::kTitle, added_segment}));
}

MERIDIAN_TEST(EvictionIsBoundedAndCountersAddUp) {
    // One shard, so the accounting is exact and not spread over per-shard budgets.
    const size_t entry_bytes = MakePostings(4)->EstimateBytes();
    const size_t capacity = entry_bytes * 10;
    PostingListCache cache(capacity, 1);

    const int inserted = 200;
    for (int i = 0; i < inserted; ++i) {
        cache.Insert(PostingCacheKey{"term" + std::to_string(i), Field::kBody, 1}, MakePostings(4));
    }

    const auto stats = cache.stats();
    CHECK_EQ(stats.insertions, static_cast<uint64_t>(inserted));
    CHECK_EQ(stats.updates, 0u);
    CHECK(cache.bytes() <= capacity);
    CHECK(cache.entries() <= 10u);
    // Everything inserted is either resident or evicted, with nothing unaccounted for.
    CHECK_EQ(stats.insertions - stats.evictions, cache.entries());

    // An update replaces in place: it must not count as an insertion and must not change
    // the resident set size.
    const size_t resident = cache.entries();
    auto any = PostingCacheKey{"term" + std::to_string(inserted - 1), Field::kBody, 1};
    cache.Insert(any, MakePostings(4));
    const auto updated = cache.stats();
    CHECK_EQ(updated.updates, 1u);
    CHECK_EQ(updated.insertions, static_cast<uint64_t>(inserted));
    CHECK_EQ(cache.entries(), resident);
}

MERIDIAN_TEST(OversizedEntryIsRefusedRatherThanFlushingTheCache) {
    const size_t entry_bytes = MakePostings(4)->EstimateBytes();
    PostingListCache cache(entry_bytes * 4, 1);
    for (int i = 0; i < 3; ++i) {
        cache.Insert(PostingCacheKey{"small" + std::to_string(i), Field::kBody, 1}, MakePostings(4));
    }
    const size_t before = cache.entries();

    cache.Insert(PostingCacheKey{"enormous", Field::kBody, 1}, MakePostings(10'000));

    const auto stats = cache.stats();
    CHECK_EQ(stats.rejected_too_large, 1u);
    CHECK_EQ(stats.evictions, 0u);
    CHECK_EQ(cache.entries(), before);
    CHECK(!cache.Contains(PostingCacheKey{"enormous", Field::kBody, 1}));
}

// The claim the segmented policy is here for: a scan of never-repeated keys must not
// evict the working set. Run against a plain LRU of the same size, every hot entry would
// be gone.
MERIDIAN_TEST(ScanDoesNotEvictTheHotSet) {
    const size_t entry_bytes = MakePostings(4)->EstimateBytes();
    const size_t capacity = entry_bytes * 20;  // 20 entries, of which 16 may be protected
    PostingListCache cache(capacity, 1);

    const int hot_count = 8;
    for (int i = 0; i < hot_count; ++i) {
        const PostingCacheKey key{"hot" + std::to_string(i), Field::kBody, 1};
        cache.Insert(key, MakePostings(4));
        // The second access is what promotes it out of probation. One access is exactly
        // what a scan produces, and is deliberately not enough.
        CHECK(cache.Lookup(key).has_value());
    }

    for (int i = 0; i < 500; ++i) {
        cache.Insert(PostingCacheKey{"scan" + std::to_string(i), Field::kBody, 2}, MakePostings(4));
    }

    int survivors = 0;
    for (int i = 0; i < hot_count; ++i) {
        if (cache.Contains(PostingCacheKey{"hot" + std::to_string(i), Field::kBody, 1})) {
            ++survivors;
        }
    }
    CHECK_EQ(survivors, hot_count);

    // And the scan did churn: its entries occupied only the probationary segment.
    int scan_survivors = 0;
    for (int i = 0; i < 500; ++i) {
        if (cache.Contains(PostingCacheKey{"scan" + std::to_string(i), Field::kBody, 2})) {
            ++scan_survivors;
        }
    }
    // The scan's entries live only in the space the protected segment is not using, so at
    // most (capacity - hot_count) of the 500 are still resident and the rest were evicted.
    CHECK(scan_survivors <= 20 - hot_count);
    CHECK_EQ(cache.stats().evictions, static_cast<uint64_t>(500 - scan_survivors));
    CHECK_EQ(static_cast<size_t>(survivors + scan_survivors), cache.entries());
    CHECK_EQ(cache.stats().promotions, static_cast<uint64_t>(hot_count));
}

MERIDIAN_TEST(ProtectedOverflowDemotesRatherThanEvicting) {
    const size_t entry_bytes = MakePostings(4)->EstimateBytes();
    PostingListCache cache(entry_bytes * 10, 1);  // 8 protected, 2 probationary

    // Promote more entries than the protected segment can hold.
    for (int i = 0; i < 10; ++i) {
        const PostingCacheKey key{"k" + std::to_string(i), Field::kBody, 1};
        cache.Insert(key, MakePostings(4));
        CHECK(cache.Lookup(key).has_value());
    }

    const auto stats = cache.stats();
    CHECK(stats.promotions >= 9u);
    CHECK(stats.demotions > 0u);
    // A demoted entry is still resident, so demotion is not eviction by another name.
    CHECK(cache.entries() >= 8u);
}

MERIDIAN_TEST(ShardingKeepsIndependentKeysApart) {
    PostingListCache cache(1 << 20, 16);
    CHECK_EQ(cache.shard_count(), 16u);
    QueryResultCache responses(1 << 20, 3);
    // Rounded up to a power of two so shard selection is a mask.
    CHECK_EQ(responses.shard_count(), 4u);
}

// Thread-safety under contention. Run as written (stress) and again under
// -fsanitize=thread; see the report for the TSan result.
MERIDIAN_TEST(ConcurrentAccessIsSafeAndCountersAreConsistent) {
    PostingListCache postings(1 << 18, 16);
    QueryResultCache responses(1 << 18, 16);

    const int threads = 8;
    const int iterations = 4000;
    std::atomic<uint64_t> lookups{0};
    std::vector<std::thread> workers;

    for (int t = 0; t < threads; ++t) {
        workers.emplace_back([&, t] {
            for (int i = 0; i < iterations; ++i) {
                // A mix of a shared hot key and thread-private keys, so the test exercises
                // both contention on one shard and spread across all of them.
                const int id = (i % 4 == 0) ? 0 : (t * iterations + i);
                const PostingCacheKey posting_key{"term" + std::to_string(id), Field::kBody, 1};
                if (!postings.Lookup(posting_key).has_value()) {
                    postings.Insert(posting_key, MakePostings(8));
                }
                ++lookups;

                const ResultCacheKey response_key{"q" + std::to_string(id), "", 10, 1};
                if (!responses.Lookup(response_key).has_value()) {
                    responses.Insert(response_key, MakeResponse("doc" + std::to_string(id), 1.0));
                }
                ++lookups;
            }
        });
    }
    for (auto& worker : workers) worker.join();

    const auto posting_stats = postings.stats();
    const auto response_stats = responses.stats();
    const uint64_t expected_per_cache = static_cast<uint64_t>(threads) * iterations;
    CHECK_EQ(posting_stats.hits + posting_stats.misses, expected_per_cache);
    CHECK_EQ(response_stats.hits + response_stats.misses, expected_per_cache);
    CHECK_EQ(lookups.load(), expected_per_cache * 2);
    CHECK(postings.bytes() <= (1u << 18));
    CHECK(responses.bytes() <= (1u << 18));
    // Only insertions add entries. An update replaces one in place, and two threads racing
    // to populate the same missing key produce exactly that — one insertion and one
    // update — so counting updates here would overstate the resident set.
    CHECK_EQ(posting_stats.insertions - posting_stats.evictions, postings.entries());
}

MERIDIAN_TEST_MAIN()
