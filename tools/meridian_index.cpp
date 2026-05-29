// meridian_index: build an index from a corpus file and report what it cost.
//
// The numbers it prints are the ones that decide whether an index is healthy — document
// count, segment count, distinct terms, size, and build throughput. Segment count is
// here rather than hidden because it is the figure that predicts query cost: every
// segment is a separate dictionary lookup and a separate posting list to merge, so an
// index of fifty small segments and the same index compacted into one behave quite
// differently at query time. `--merge` makes that comparison directly.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "../src/serve/reference_index.hpp"
#include "corpus_file.hpp"

namespace {

using meridian::tools::HumanBytes;

void Usage() {
    std::fprintf(stderr,
                 "usage: meridian_index --corpus FILE [options]\n"
                 "\n"
                 "  --corpus FILE            corpus to index (TSV or plain text; see below)\n"
                 "  --docs-per-segment N     flush a segment every N documents (default 10000)\n"
                 "  --embedding-dim N        also build N-dimensional stand-in vectors (default 0)\n"
                 "  --merge                  compact into a single segment and report again\n"
                 "  --delete ID              tombstone a document before reporting\n"
                 "\n"
                 "corpus format, one document per line, trailing columns optional:\n"
                 "  id<TAB>title<TAB>body<TAB>tags<TAB>language<TAB>category<TAB>published_at"
                 "<TAB>popularity\n"
                 "a line with no tab is treated as a body with a generated id\n");
}

void PrintStats(const char* label, const meridian::serve::ReferenceIndex& index) {
    const auto stats = index.stats();
    std::printf("%-18s documents=%llu segments=%llu distinct_terms=%llu size=%s (%llu bytes) "
                "generation=%llu\n",
                label, static_cast<unsigned long long>(stats.documents),
                static_cast<unsigned long long>(stats.segments),
                static_cast<unsigned long long>(stats.terms), HumanBytes(stats.bytes).c_str(),
                static_cast<unsigned long long>(stats.bytes),
                static_cast<unsigned long long>(index.Current()->version));
}

}  // namespace

int main(int argc, char** argv) {
    std::string corpus;
    std::string to_delete;
    meridian::tools::LoadOptions options;
    bool merge = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--corpus") {
            corpus = next("--corpus");
        } else if (arg == "--docs-per-segment") {
            options.docs_per_segment = static_cast<uint32_t>(std::stoul(next("--docs-per-segment")));
        } else if (arg == "--embedding-dim") {
            options.embedding_dim = static_cast<size_t>(std::stoul(next("--embedding-dim")));
        } else if (arg == "--delete") {
            to_delete = next("--delete");
        } else if (arg == "--merge") {
            merge = true;
        } else if (arg == "--help" || arg == "-h") {
            Usage();
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            Usage();
            return 2;
        }
    }

    if (corpus.empty()) {
        Usage();
        return 2;
    }
    if (options.docs_per_segment == 0) {
        std::fprintf(stderr, "--docs-per-segment must be positive\n");
        return 2;
    }

    meridian::serve::ReferenceIndex index;

    const auto started = std::chrono::steady_clock::now();
    const auto report = meridian::tools::LoadCorpus(corpus, index, options);
    const auto elapsed = std::chrono::steady_clock::now() - started;

    if (!report.ok()) {
        std::fprintf(stderr, "error: %s\n", report.error.c_str());
        return 1;
    }

    const double seconds = std::chrono::duration<double>(elapsed).count();
    std::printf("indexed %llu documents from %s\n",
                static_cast<unsigned long long>(report.documents), corpus.c_str());
    std::printf("  input            %llu lines, %llu skipped, %s read\n",
                static_cast<unsigned long long>(report.lines_read),
                static_cast<unsigned long long>(report.lines_skipped),
                HumanBytes(report.bytes_read).c_str());
    std::printf("  build time       %.3f s (%.0f docs/s)\n", seconds,
                seconds > 0.0 ? static_cast<double>(report.documents) / seconds : 0.0);
    std::printf("  flushes          %llu\n", static_cast<unsigned long long>(report.flushes));
    PrintStats("  after load", index);

    if (!to_delete.empty()) {
        if (index.Delete(to_delete)) {
            std::printf("  tombstoned %s (space returns at the next merge, not now: the\n"
                        "             segment holding it is immutable)\n",
                        to_delete.c_str());
            PrintStats("  after delete", index);
        } else {
            std::fprintf(stderr, "  no document with id %s\n", to_delete.c_str());
        }
    }

    if (merge) {
        const auto merge_start = std::chrono::steady_clock::now();
        const uint64_t version = index.Merge();
        const double merge_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - merge_start).count();
        std::printf("  merged into one segment in %.3f s, generation %llu\n", merge_seconds,
                    static_cast<unsigned long long>(version));
        PrintStats("  after merge", index);
    }

    return 0;
}
