// Merging, and the property the whole generation design exists to provide: a query that
// started before a merge keeps working, sees a consistent view, and keeps its input files
// alive until it is done with them.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "meridian/index.hpp"
#include "testing.hpp"

using namespace meridian;

namespace {

class TempDirectory {
public:
    explicit TempDirectory(const std::string& tag) {
        static std::atomic<int> counter{0};
        path_ = (std::filesystem::temp_directory_path() /
                 ("meridian-" + tag + "-" + std::to_string(counter.fetch_add(1))))
                    .string();
        std::filesystem::remove_all(path_);
        std::filesystem::create_directories(path_);
    }
    ~TempDirectory() { std::filesystem::remove_all(path_); }

    TempDirectory(const TempDirectory&) = delete;
    TempDirectory& operator=(const TempDirectory&) = delete;

    const std::string& path() const { return path_; }

private:
    std::string path_;
};

using Tokens = std::unordered_map<Field, std::vector<std::string>>;

DocumentMeta MakeDoc(const std::string& id, const Tokens& tokens) {
    DocumentMeta meta;
    meta.external_id = id;
    meta.language = "en";
    meta.category = "doc";
    meta.published_at = 1'700'000'000;
    uint32_t length = 0;
    for (const auto& [field, list] : tokens) {
        (void)field;
        length += static_cast<uint32_t>(list.size());
    }
    meta.length = length;
    meta.popularity = 0.5;
    return meta;
}

const std::vector<std::string>& Vocabulary() {
    static const std::vector<std::string> vocabulary = {
        "index", "indexed", "indexes", "indexing", "merge", "merged", "merges",
        "skip",  "skipped", "posting", "postings", "term",  "terms",  "segment",
    };
    return vocabulary;
}

Tokens RandomTokens(std::mt19937& rng) {
    Tokens tokens;
    for (Field field : {Field::kTitle, Field::kBody, Field::kTags}) {
        const size_t count = 1 + rng() % 8;
        std::vector<std::string> list;
        for (size_t i = 0; i < count; ++i) list.push_back(Vocabulary()[rng() % Vocabulary().size()]);
        tokens[field] = std::move(list);
    }
    return tokens;
}

std::set<std::string> MatchingDocuments(const Generation& generation, const std::string& term,
                                       Field field) {
    std::set<std::string> ids;
    for (const auto& segment : generation.segments()) {
        std::unique_ptr<PostingIterator> it = segment->Postings(term, field);
        if (it == nullptr) continue;
        for (DocId doc = it->doc(); doc != kNoMoreDocs; doc = it->next()) {
            if (segment->IsDeleted(doc)) continue;
            auto meta = segment->Document(doc);
            if (!meta.ok()) continue;
            ids.insert(meta.value().external_id);
        }
    }
    return ids;
}

std::set<std::string> LiveDocuments(const Generation& generation) {
    std::set<std::string> ids;
    for (const auto& segment : generation.segments()) {
        for (DocId local = 0; local < segment->meta().document_count; ++local) {
            if (segment->IsDeleted(local)) continue;
            auto meta = segment->Document(local);
            if (!meta.ok()) continue;
            ids.insert(meta.value().external_id);
        }
    }
    return ids;
}

std::vector<std::string> SegmentFiles(const Generation& generation) {
    std::vector<std::string> paths;
    for (const auto& segment : generation.segments()) paths.push_back(segment->meta().path);
    return paths;
}

struct Model {
    std::set<std::string> live;
    std::map<std::pair<uint8_t, std::string>, std::set<std::string>> documents_for_term;

    void Add(const std::string& id, const Tokens& tokens) {
        live.insert(id);
        for (const auto& [field, list] : tokens) {
            for (const std::string& token : list) {
                documents_for_term[{static_cast<uint8_t>(field), token}].insert(id);
            }
        }
    }

    void Delete(const std::string& id) {
        live.erase(id);
        for (auto& [key, ids] : documents_for_term) {
            (void)key;
            ids.erase(id);
        }
    }
};

}  // namespace

MERIDIAN_TEST(MergeKeepsEveryLiveDocumentAndDropsEveryDeletedOne) {
    const TempDirectory dir("merge-liveness");
    Index::Options options;
    options.directory = dir.path();
    options.docs_per_segment = 30;
    options.merge_factor = 1000;  // merges happen only when this test asks for them

    Index index(options);
    CHECK_OK(index.Open());

    Model model;
    std::mt19937 rng(4242);
    const int document_count = 300;
    for (int i = 0; i < document_count; ++i) {
        const std::string id = "doc-" + std::to_string(i);
        const Tokens tokens = RandomTokens(rng);
        CHECK_OK(index.Add(MakeDoc(id, tokens), tokens));
        model.Add(id, tokens);
    }
    CHECK_OK(index.Flush());

    // Deletions spread across every segment, including the first and last document of
    // each, because a merge that drops the wrong document is most likely to do it at a
    // boundary.
    for (int i = 0; i < document_count; i += 7) {
        const std::string id = "doc-" + std::to_string(i);
        CHECK_OK(index.Delete(id));
        model.Delete(id);
    }
    CHECK_OK(index.Delete("doc-" + std::to_string(document_count - 1)));
    model.Delete("doc-" + std::to_string(document_count - 1));

    const std::shared_ptr<Generation> before = index.Acquire();
    CHECK(before->segments().size() > 1);
    CHECK(LiveDocuments(*before) == model.live);

    CHECK_OK(index.ForceMerge());

    const std::shared_ptr<Generation> after = index.Acquire();
    CHECK_EQ(after->segments().size(), 1u);
    // The merged segment holds exactly the live documents and nothing else: the tombstones
    // are gone because the postings they pointed at are gone, not merely hidden.
    CHECK_EQ(after->segments()[0]->meta().deleted_count, 0u);
    CHECK_EQ(after->segments()[0]->meta().document_count, model.live.size());
    CHECK(LiveDocuments(*after) == model.live);
    CHECK_EQ(after->stats().document_count, model.live.size());

    for (const auto& [key, expected] : model.documents_for_term) {
        const std::set<std::string> actual =
            MatchingDocuments(*after, key.second, static_cast<Field>(key.first));
        CHECK(actual == expected);
    }

    std::printf("  merged %zu segments into 1: %zu live documents kept, %d tombstones dropped\n",
                before->segments().size(), model.live.size(),
                document_count - static_cast<int>(model.live.size()));
    CHECK_OK(index.Close());
}

MERIDIAN_TEST(MergeRemapsDocumentIdsConsistently) {
    const TempDirectory dir("merge-remap");
    Index::Options options;
    options.directory = dir.path();
    options.docs_per_segment = 10;
    options.merge_factor = 1000;

    Index index(options);
    CHECK_OK(index.Open());

    std::mt19937 rng(8);
    std::map<std::string, Tokens> by_id;
    for (int i = 0; i < 50; ++i) {
        const std::string id = "doc-" + std::to_string(i);
        const Tokens tokens = RandomTokens(rng);
        by_id[id] = tokens;
        CHECK_OK(index.Add(MakeDoc(id, tokens), tokens));
    }
    CHECK_OK(index.Flush());
    CHECK_OK(index.Delete("doc-13"));
    CHECK_OK(index.ForceMerge());

    const std::shared_ptr<Generation> generation = index.Acquire();
    // Every global id in the merged generation must resolve, and the document it resolves
    // to must be the one whose postings name that id. A remap that shifted by one would
    // still produce a complete, plausible, entirely misattributed index.
    for (DocId global = 0; global < generation->stats().document_count; ++global) {
        auto resolved = generation->Resolve(global);
        CHECK(resolved.ok());
        if (!resolved.ok()) continue;
        const SegmentReader* segment = resolved.value().first;
        const DocId local = resolved.value().second;
        auto meta = segment->Document(local);
        CHECK(meta.ok());
        if (!meta.ok()) continue;

        const auto& tokens = by_id.at(meta.value().external_id);
        for (const auto& [field, list] : tokens) {
            for (const std::string& token : list) {
                std::unique_ptr<PostingIterator> it = segment->Postings(token, field);
                CHECK(it != nullptr);
                if (it == nullptr) continue;
                CHECK_EQ(it->advance(local), local);
            }
        }
    }
    CHECK_OK(index.Close());
}

MERIDIAN_TEST(AQueryHoldingAnOldGenerationSurvivesAMergeAndKeepsItsFiles) {
    // The test that justifies the generation design. A real thread queries an acquired
    // generation while the main thread merges every segment out from under it. The old
    // segment files must stay on disk for as long as that generation is held, and the
    // queries must keep returning the same answers.
    const TempDirectory dir("merge-concurrent");
    Index::Options options;
    options.directory = dir.path();
    options.docs_per_segment = 40;
    options.merge_factor = 1000;

    Index index(options);
    CHECK_OK(index.Open());

    Model model;
    std::mt19937 rng(31337);
    for (int i = 0; i < 400; ++i) {
        const std::string id = "doc-" + std::to_string(i);
        const Tokens tokens = RandomTokens(rng);
        CHECK_OK(index.Add(MakeDoc(id, tokens), tokens));
        model.Add(id, tokens);
    }
    CHECK_OK(index.Flush());

    std::shared_ptr<Generation> held = index.Acquire();
    const std::vector<std::string> old_files = SegmentFiles(*held);
    CHECK(old_files.size() > 1);
    for (const std::string& path : old_files) CHECK(std::filesystem::exists(path));

    const std::set<std::string> expected_live = LiveDocuments(*held);
    CHECK(expected_live == model.live);

    std::atomic<bool> stop{false};
    std::atomic<uint64_t> queries{0};
    std::atomic<bool> mismatch{false};

    std::thread reader([&]() {
        const std::vector<std::string> terms = Vocabulary();
        size_t next = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            const std::string& term = terms[next++ % terms.size()];
            const std::set<std::string> got = MatchingDocuments(*held, term, Field::kBody);
            const auto expected =
                model.documents_for_term.find({static_cast<uint8_t>(Field::kBody), term});
            // Every vocabulary term appears in this corpus, so a term the model does not
            // know about would itself be a failure.
            if (expected == model.documents_for_term.end() || got != expected->second) {
                mismatch.store(true);
            }
            queries.fetch_add(1, std::memory_order_relaxed);
        }
    });

    // Merge while the reader is running. Nothing in the reader's path takes the index
    // lock after Acquire, so this is genuinely concurrent rather than serialised by it.
    CHECK_OK(index.ForceMerge());
    const std::shared_ptr<Generation> fresh = index.Acquire();
    CHECK_EQ(fresh->segments().size(), 1u);
    CHECK(fresh->version() > held->version());
    CHECK(LiveDocuments(*fresh) == model.live);

    // Let the reader run against the superseded generation for a while after the merge
    // has published, which is the window where a merge that unlinked eagerly would break.
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
    while (std::chrono::steady_clock::now() < until) {
        std::this_thread::yield();
    }

    // Still held, so still on disk. A merge is not allowed to delete what a reader has.
    for (const std::string& path : old_files) {
        CHECK(std::filesystem::exists(path));
    }
    // And calling the collector explicitly must not change that.
    CHECK_OK(index.MaybeMerge());
    for (const std::string& path : old_files) {
        CHECK(std::filesystem::exists(path));
    }

    stop.store(true);
    reader.join();
    CHECK(!mismatch.load());
    CHECK(queries.load() > 0);

    // The old generation is still coherent after the merge published over it.
    CHECK(LiveDocuments(*held) == expected_live);

    std::printf("  %llu queries ran against the superseded generation while %zu segments merged\n",
                static_cast<unsigned long long>(queries.load()), old_files.size());

    // Release it. The reference count is what the collector watches, so the files go only
    // now — and this is the assertion that proves the shared_ptr is load-bearing rather
    // than decorative.
    held.reset();
    CHECK_OK(index.MaybeMerge());
    for (const std::string& path : old_files) {
        CHECK(!std::filesystem::exists(path));
    }
    CHECK(std::filesystem::exists(fresh->segments()[0]->meta().path));

    CHECK_OK(index.Close());
}

MERIDIAN_TEST(MergeIsCorrectWhenEveryDocumentIsDeleted) {
    const TempDirectory dir("merge-all-deleted");
    Index::Options options;
    options.directory = dir.path();
    options.docs_per_segment = 10;
    options.merge_factor = 1000;

    Index index(options);
    CHECK_OK(index.Open());

    std::mt19937 rng(2);
    for (int i = 0; i < 20; ++i) {
        const Tokens tokens = RandomTokens(rng);
        CHECK_OK(index.Add(MakeDoc("doc-" + std::to_string(i), tokens), tokens));
    }
    CHECK_OK(index.Flush());
    for (int i = 0; i < 20; ++i) CHECK_OK(index.Delete("doc-" + std::to_string(i)));

    CHECK_OK(index.ForceMerge());
    const std::shared_ptr<Generation> generation = index.Acquire();
    CHECK_EQ(generation->stats().document_count, 0u);
    CHECK_EQ(generation->stats().average_document_length, 0.0);
    CHECK(LiveDocuments(*generation).empty());
    // An empty merged segment is still a valid segment rather than a special case, which
    // is what keeps the publication path from needing one.
    CHECK_EQ(generation->segments().size(), 1u);
    CHECK_EQ(generation->segments()[0]->meta().document_count, 0u);

    CHECK_OK(index.Close());
}

MERIDIAN_TEST(RepeatedMergesConvergeAndKeepTheDocumentSet) {
    const TempDirectory dir("merge-repeated");
    Index::Options options;
    options.directory = dir.path();
    options.docs_per_segment = 15;
    options.merge_factor = 3;

    Index index(options);
    CHECK_OK(index.Open());

    Model model;
    std::mt19937 rng(606);
    for (int round = 0; round < 10; ++round) {
        for (int i = 0; i < 20; ++i) {
            const std::string id = "doc-" + std::to_string(round) + "-" + std::to_string(i);
            const Tokens tokens = RandomTokens(rng);
            CHECK_OK(index.Add(MakeDoc(id, tokens), tokens));
            model.Add(id, tokens);
        }
        CHECK_OK(index.Flush());
        // Interleaved with merging, which is the shape of a real ingest: the merge policy
        // never sees a quiet index.
        CHECK_OK(index.MaybeMerge());
        if (round % 3 == 0) {
            const std::string victim = "doc-" + std::to_string(round) + "-5";
            CHECK_OK(index.Delete(victim));
            model.Delete(victim);
        }
    }

    const std::shared_ptr<Generation> generation = index.Acquire();
    CHECK(LiveDocuments(*generation) == model.live);
    CHECK_EQ(generation->stats().document_count, model.live.size());
    for (const auto& [key, expected] : model.documents_for_term) {
        CHECK(MatchingDocuments(*generation, key.second, static_cast<Field>(key.first)) ==
              expected);
    }

    const Index::Stats stats = index.stats();
    std::printf("  %llu merges consumed %llu segments; %zu segments remain holding %zu "
                "documents\n",
                static_cast<unsigned long long>(stats.merges_run),
                static_cast<unsigned long long>(stats.segments_merged),
                generation->segments().size(), model.live.size());
    CHECK(stats.merges_run > 0);

    CHECK_OK(index.Close());
}

MERIDIAN_TEST_MAIN()
