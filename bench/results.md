# Measurements

Every number the README quotes, re-derived by running the suite and the CLIs on one
machine, with what was varied, what was held fixed, and what the number does and does
not support.

Nothing here was copied forward from a previous run. Where a figure in the README did
not reproduce, this file says so and gives what the code produces today.

## Contents

- [Test bed](#test-bed)
- [How to reproduce](#how-to-reproduce)
- [The collections](#the-collections)
- [1. Storage and posting lists](#1-storage-and-posting-lists)
- [2. Term dictionary and segments](#2-term-dictionary-and-segments)
- [3. Merging](#3-merging)
- [4. Lexical retrieval and BM25](#4-lexical-retrieval-and-bm25)
- [5. Vector search and HNSW](#5-vector-search-and-hnsw)
- [6. Fusion](#6-fusion)
- [7. Serving and caches](#7-serving-and-caches)
- [8. Evaluation harness](#8-evaluation-harness)
- [9. The five claims, re-checked](#9-the-five-claims-re-checked)
- [10. What is not measured](#10-what-is-not-measured)

## Test bed

| | |
|---|---|
| machine | Apple M4 Pro, 14 cores, 48 GiB |
| OS | macOS 26.6.2 (Darwin 25.6.0), arm64 |
| compiler | Apple clang 21.0.0 |
| build | `RelWithDebInfo` (`-O2 -g -DNDEBUG`), directory `build/` |
| sanitizer builds | `build-asan` (`-DMERIDIAN_ASAN=ON`), `build-tsan` (`-DMERIDIAN_TSAN=ON`) |
| Python | 3, standard library only |

Everything is single-process on one machine. No competing load was removed beyond
closing nothing in particular, so treat every latency figure as having a few percent of
run-to-run spread; where that spread matters the spread is reported rather than a single
sample.

Source size, measured: **22,920 lines** of C++ across `src/`, `include/`, `tools/` and
`tests/` (13,916 of it engine and CLIs, 9,004 tests) and **2,974 lines** of Python
(1,617 library, 1,357 tests).

Suites, measured:

```
100% tests passed out of 16
Total Test time (real) =  12.62 sec
```

219 C++ test cases across those 16 suites, counted from each binary's own tally, plus
`python3 -m unittest discover -s python/tests` → `Ran 92 tests ... OK`. Both sanitizer
builds are clean: `build-asan` 16/16 in 64.88 s, `build-tsan` 16/16 in 327.33 s.

## How to reproduce

```bash
cmake -S . -B build && cmake --build build -j8
ctest --test-dir build

# the self-reporting benchmarks live inside the test binaries
./build/postings_test ./build/skiplist_test ./build/index_test ./build/merge_test
./build/hnsw_test ./build/vector_test ./build/bm25_test ./build/serve_test
./build/segment_test ./build/semantic_ann_test

# the collections
PYTHONPATH=python python3 -m meridian_eval generate --out /tmp/demo
PYTHONPATH=python python3 -m meridian_eval generate --out /tmp/coll \
    --documents 50000 --topics 48 --vocabulary-size 28800
```

The CLI invocations are given inline with each table below.

## The collections

Two generated collections carry almost every end-to-end number, and the difference
between them turns out to matter more than any parameter swept inside either one.

| | demo (`--out /tmp/demo`, all defaults) | large (`--documents 50000 --topics 48 --vocabulary-size 28800`) |
|---|---|---|
| documents | 1,200 | 50,000 |
| topics | 6 | 48 |
| vocabulary | 3,600 | 28,800 |
| embedding dimension | 10 | 52 |
| judged queries | 48 (36 identifier, 6 lexical, 6 semantic) | 132 (36 identifier, 48 lexical, 48 semantic) |
| judgements | 2,718 | 135,470 |
| queries with embeddings | 12 | 96 |
| index as built | 1 segment, 4,700 term entries, 2.98 MiB | 5 segments, 182,255 term entries, 133.07 MiB |
| HNSW build time | 0.066 s | 5.9–6.9 s |

Both use seed 20260101 and Zipf exponent 1.05, so the corpora themselves are
deterministic; only latency moves between runs.

Generator properties, measured rather than asserted (the Python tests check these but do
not print them, so these were computed with the same method the tests use —
`_log_log_slope` over ranks 2…400, and mean pairwise cosine over the first 150
documents):

| collection | distinct terms | log-log rank-frequency slope | top / median / min term frequency | intra-topic cosine | inter-topic cosine |
|---|---|---|---|---|---|
| test `SMALL`, 900 docs, s=1.05 | 1,845 | −1.090 | 9,599 / 9 / 1 | 0.9659 | 0.2136 |
| demo, 1,200 docs, s=1.05 | 3,292 | −1.096 | 11,655 / 6 / 1 | 0.9645 | 0.1811 |
| large, 50,000 docs, s=1.05 | 26,436 | −0.995 | 237,836 / 27 / 1 | 0.9157 | 0.0270 |

And the exponent is actually tracked, not merely sampled from:

```
exponent 0.85: measured slope -0.889
exponent 1.05: measured slope -1.090
exponent 1.30: measured slope -1.360
```

What this supports: the corpora are skewed in the way that makes posting-list length vary
and skip tables worth having, and the embeddings are topically separable with internal
spread, so an ANN recall number on them is meaningful. What it does not support: any
claim about real text. The slope is a property of the generator and so is every absolute
quality figure downstream of it.

README cross-check: the README quotes slope **−1.086** and intra/inter cosine
**0.963 / 0.189**. The nearest configuration measured here gives **−1.090** and
**0.9645 / 0.1811**. Same conclusion, different last digits — the README's exact triple
does not correspond to any configuration in the repository today.

---

## 1. Storage and posting lists

### 1.1 Delta + varint against fixed width

`./build/postings_test` → `DeltaVarintBeatsFixedWidthOnADenseList`

```
dense list: 100000 postings, 217166 bytes encoded vs 800000 fixed-width (3.68x)
```

Varied: nothing — a single dense 100,000-posting list. Held fixed: block size, the
field's position policy (off), the gap distribution (dense, so gaps are small and nearly
all fit in one varint byte).

**3.68x**, deterministic, reproduces byte-for-byte.

What it supports: on a dense list, delta coding plus a byte-oriented varint is worth
roughly 3.7x against four fixed bytes per document id. What it does not support: anything
about a sparse list. A list with gaps in the thousands spends two or three varint bytes
per gap and the ratio falls toward 1.3–2x; this benchmark deliberately picks the dense
case and so reports close to the best the codec can do, not the average.

### 1.2 Skipping against sequential decode

`./build/skiplist_test` → `SkippingDecodesFewerPostingsOnASparseConjunction`

```
200000-posting list, 200 seeks: sequential decoded 200000 postings in 1563 blocks,
skipping decoded 24000 in 188 blocks taking 187 skips -> 8.3x fewer postings
```

| | postings decoded | blocks read |
|---|---|---|
| sequential | 200,000 | 1,563 |
| skipping | 24,000 | 188 |

Varied: the iterator. Held fixed: the list (200,000 postings), the 200 seek targets,
block size 128. Both iterators land on an identical document for every one of the 200
seeks — that is an assertion, not an observation.

**8.3x fewer postings, 8.3x fewer blocks.** Deterministic.

What it supports: the skip table does the thing it exists for when targets are widely
separated, and 188 blocks for 200 seeks is near the floor of 200 (two seeks occasionally
land in one block), so most of the available win is being taken rather than a fraction of
it. What it does not support: a speedup figure. This counts decodes and block reads, not
nanoseconds; a wall-clock number would also fold in the binary search over the table and
the per-block checksum, which this does not isolate.

### 1.3 The differential test behind it

`AdvanceAgreesWithSequentialScan` → `checked 12800 seeks across 80 encoded lists`

Three implementations — the skipping iterator, the sequential iterator, and
`std::lower_bound` over the decoded reference — are required to agree on document, term
frequency, positions and size hint at every one of the 12,800 seeks. This is the thing
that makes 1.2 believable; a faster iterator that is wrong is not faster.

### 1.4 The fixed-width skip table

Not a measurement, and this file should not present it as one. `src/postings/skiplist.hpp`
argues that a 78,000-entry table (ten million postings at 128 per block) costs 78,000
varint decodes to scan and 17 comparisons to binary-search. That is arithmetic —
`log2(78125) ≈ 16.3` — not a benchmark. There is no second-level-skip-list implementation
in the tree to measure against, so the comparison is a design argument, and the only
measured consequence of the choice is 1.2 above plus the entry width: 12 bytes per entry,
one entry per 128-posting block, so **0.094 bytes per posting** of skip overhead.

### 1.5 Corruption is distinguishable from absence

`./build/bm25_test` → `ACorruptPostingBlockFailsTheQueryInsteadOfShorteningIt`

```
corrupt block surfaced as: corruption: term 'zebra' in segment 1: block 2 failed its checksum
```

Not a performance number; it is the regression test for the worst defect in the project's
history, where a CRC failure mid-walk was indistinguishable from the end of the list and a
query returned a truncated result set with status `ok`. The companion test
`AnUncorruptedQueryOverTheSameCorpusStillSucceeds` is what stops the fix from being "fail
always". `./build/postings_test` additionally checks detection at the codec layer
(`CorruptBlockChecksumIsDetected`, `TruncatedPostingListIsRejected`).

---

## 2. Term dictionary and segments

The honest statement here is that the front-coded dictionary has **no size benchmark**.
Nothing in the tree reports dictionary bytes separately from segment bytes, so there is no
front-coding compression ratio to quote, and this file is not going to invent one. What is
measured is I/O attribution and lookup correctness.

### 2.1 `bytes_read` accounts for what was asked for

`./build/segment_test` → `BytesReadTracksWhatWasActuallyAskedFor`

```
bytes_read: 6177 idle, 6984 for one term, 49318 for all 54 terms (file is 55332 bytes)
```

Varied: how much of the segment is touched. Held fixed: the segment (54 terms, 55,332
bytes on disk).

Reading one term costs 6,984 − 6,177 = **807 bytes** beyond the fixed open cost; reading
all 54 costs 49,318, which is 89% of the file.

What it supports: the counter is real. Per-term cost is a small fraction of the file, and
a full sweep approaches but does not exceed the file size, which is the shape you want
from an instrument. This is the measurement that only exists because the storage layer
uses `pread` rather than `mmap` — a page fault does not increment a counter, and the
README's claim that the instrumentation is bought with a syscall per block is structurally
true of the design.

What it does not support: a bytes-per-query figure for a real workload. 54 terms is a toy
segment.

### 2.2 Dictionary lookups against brute force

`./build/index_test` → `TermLookupsMatchABruteForceReference`

```
compared 72 (field, term) pairs across 11 segments against the reference
```

Block stride is 16 terms (`kDictionaryBlockTerms`), so a lookup front-decodes at most 15
keys after a binary search over block first-keys. Present and absent terms are both
checked; absence is not an error at this layer.

---

## 3. Merging

### 3.1 Write amplification

`./build/index_test` → `WriteAmplificationIsReportedAndRisesWithMergeWork`

```
write amplification: 1.00 after flushes, 2.57 after one full merge;
bytes/doc 156.4 -> 256.2 (51241 bytes written, 19952 live, 200 documents)
```

| | write amplification | bytes/doc |
|---|---|---|
| after flushes only | 1.00 | 156.4 |
| after one full merge | 2.57 | 256.2 |

Varied: whether a full merge has run. Held fixed: 200 documents, the flush threshold, one
tombstone applied before the merge. Deterministic — reproduces exactly.

Amplification is cumulative bytes written over *currently live* bytes. Both halves of that
sentence are load-bearing:

- 1.00 before any merge is the check that the denominator measures what it claims. Every
  byte written is still live, so the ratio must be exactly 1, and if the live figure were
  derived rather than recorded this is where it would drift.
- 2.57 after one merge is above the naive 2.0 because the merge **wrote** the data a
  second time (numerator roughly doubles) and then **shrank** it (denominator falls below
  what the flushes wrote). A merged segment holds one dictionary entry per term instead of
  one per term per input segment, and it drops the tombstoned document's postings
  entirely.

`bytes/doc` is the companion that does not move when space is reclaimed — it is the
numerator over documents accepted, so it rises from 156.4 to 256.2 purely because of the
extra write. Reading the pair together is what makes 2.57 legible instead of alarming.

**Independent confirmation of the mechanism**, via the CLI on 4,999 real documents:

```
$ ./build/meridian_index --corpus /tmp/mb5k.tsv --docs-per-segment 500 --merge
  flushes          10
  after load       documents=4999 segments=10 distinct_terms=103939 size=17.39 MiB
  after merge      documents=4999 segments=1  distinct_terms=30768  size=12.83 MiB
```

Ten segments' dictionaries hold 103,939 term entries between them; the merged segment
holds 30,768 — a **3.38x** reduction in entries and **26.2%** off the on-disk size, for
exactly the same document set. That is the compaction that drives amplification above 2.0,
measured on a collection big enough to be unambiguous about it.

What it does not support: a steady-state amplification figure. This is one merge of one
tier. A long-running size-tiered index re-writes data several times over and the number
keeps climbing; nothing here measures that trajectory.

### 3.2 Tombstones do not reclaim space; merges do

```
$ ./build/meridian_index --corpus /tmp/mb200.tsv --docs-per-segment 25 --delete doc-00000 --merge
  after load       documents=199 segments=8 distinct_terms=8341 size=955.30 KiB generation=8
  tombstoned doc-00000 (space returns at the next merge, not now: the
             segment holding it is immutable)
  after delete     documents=198 segments=8 distinct_terms=8341 size=955.30 KiB generation=9
  after merge      documents=198 segments=1 distinct_terms=5411  size=767.26 KiB generation=10
```

The delete changes `documents` and the generation number and **nothing else** — identical
term count, identical bytes. The merge then takes 8,341 entries to 5,411 and 955.30 KiB to
767.26 KiB. The CLI says this in words rather than implying the delete freed anything.

README cross-check: the README quotes "the merge takes 74 distinct terms to 55 and 8.02
KiB to 6.24 KiB". No corpus producing those figures is in the repository, so they could
not be reproduced. The equivalent run on a four-document corpus gives 37 → 28 terms and
4.40 → 3.36 KiB; on the 199-document corpus above, 8,341 → 5,411 and 955.30 → 767.26 KiB.
Same mechanism, same direction, different corpus.

### 3.3 A merge does not disturb a reader holding the old generation

`./build/merge_test`

```
merged 10 segments into 1: 256 live documents kept, 44 tombstones dropped
1500 queries ran against the superseded generation while 10 segments merged
9 merges consumed 28 segments; 1 segments remain holding 196 documents
```

The middle line is the one the design exists for: a query acquires a generation, all ten
of its segments are merged away underneath it, and the query keeps answering from files
that are not unlinked until the last reader drops its `shared_ptr`.

**The query count is not a reproducible figure.** It is however many queries fit in the
merge's duration, so it moves run to run. Five consecutive runs:

```
1583  1384  1490  1515  1498
```

README cross-check: the README quotes **1,438**, which is inside this range but is a
single sample presented as a result. The reproducible part of the claim is the structure —
10 segments, 0 inconsistencies, files surviving until the last reader lets go — not the
count.

---

## 4. Lexical retrieval and BM25

### 4.1 BM25 against hand computation

`./build/bm25_test` → `Bm25MatchesHandComputedScores`,
`DisjunctiveRetrievalReproducesTheHandComputedRanking`

No printed number: the test carries the arithmetic inline and asserts the engine
reproduces it, including the saturation shape (`Bm25ScoreRisesWithTermFrequencyButSaturates`),
the length penalty, a ubiquitous term's IDF staying tiny-but-positive, and per-field
weights multiplying the contribution. This is the only check in the suite that pins
*absolute* scores rather than an ordering, which is why it is worth having even though it
prints nothing.

### 4.2 Rarest-first ordering: the headline result

`./build/bm25_test` → `ConjunctiveOrderingDecodesFarFewerPostings`

```
leapfrog:    rarest-first 43 postings, commonest-first 44 (ratio 1.02x), skips 22 vs 22
leader-scan: rarest-first 43 postings, commonest-first 203 (ratio 4.72x)
```

Varied: term order (rarest-first vs commonest-first) and the conjunction formulation.
Held fixed: the corpus, the terms, everything else. Deterministic — these are decode
counters, not times, and they reproduce exactly on every run.

| formulation | rarest-first | commonest-first | ratio |
|---|---|---|---|
| leapfrog (what is implemented) | 43 | 44 | **1.02x** |
| leader-scan (written for this comparison) | 43 | 203 | **4.72x** |

The textbook claim — order your terms rarest-first and win big — is a claim about the
leader-scan formulation: pick one list as the leader, walk it, and for each candidate probe
the others. There the leader's length *is* the loop bound, so choosing the longest list as
leader costs you its whole length.

Leapfrog has no leader. Every cursor advances to a target supplied by another cursor, so
the number of advances is a property of the document sets rather than of the order they
were listed in, and the 1.02x is the residual: a couple of extra decodes from the first
cursor having to catch up.

What this supports: in a leapfrog conjunction, term ordering is close to irrelevant, and
the 4.72x is the cost of the formulation the folklore assumes. What it does not support:
a claim that ordering never matters. 4.72x on 203 postings is a small absolute number; the
same experiment at a thousand times the scale would show the leader-scan gap widening with
the leader's length while the leapfrog gap stays near 1. That extrapolation is not measured
here.

Also worth noting: the skip count is identical (22 vs 22) in both leapfrog orderings,
which is the mechanism rather than the result — all cursor movement is target-driven, so
both orders execute the same seeks.

### 4.3 Everything else the lexical path asserts

Structural, not quantitative, but they are the reason the above is trustworthy:
conjunction requires every term; segments missing a term are skipped entirely; phrases
require adjacency and do not span field boundaries; repeated terms in a phrase work;
filters apply per field and in combination; filtering happens **before** the candidate
budget is spent (so a selective filter does not get starved by a budget consumed on
documents it was going to reject); filter-only queries browse the filtered set; deleted
documents are not returned; document ids are global across segments.

### 4.4 BM25 parameters, swept

```
$ ./build/meridian_eval --corpus /tmp/demo/... --mode hybrid --repeat 5 --no-cache \
      --no-per-query --sweep k1=0.6,0.9,1.2,1.6,2.0
```

| config (demo) | NDCG@10 | P@10 | p99 µs | verdict |
|---|---|---|---|---|
| k1=0.60 (baseline) | 0.9215 | 0.3792 | 117.88 | — |
| k1=0.90 | 0.9215 | 0.3792 | 154.58 | worse |
| k1=1.20 | 0.9215 | 0.3792 | 97.71 | better |
| k1=1.60 | 0.9215 | 0.3792 | 94.42 | better |
| k1=2.00 | 0.9215 | 0.3792 | 91.04 | better |

| config (demo) | NDCG@10 | P@10 | p99 µs | verdict |
|---|---|---|---|---|
| b=0.00 (baseline) | 0.9218 | 0.3771 | 115.12 | — |
| b=0.40 | 0.9220 | 0.3792 | 102.12 | better |
| b=0.75 | 0.9215 | 0.3792 | 115.88 | no-change |
| b=1.00 | 0.9215 | 0.3792 | 100.12 | better |

**This table measures nothing about k1 or b.** NDCG is flat to four decimal places across
the entire k1 range and moves 0.0005 across the entire b range, and every "better" verdict
in both tables is awarded on a p99 difference that is smaller than the run-to-run spread of
p99 on this machine. The verdict column is doing exactly what it was built to do — refusing
to call a 0.0000 quality change an improvement — and the correct reading is "the harness
could not distinguish these configurations".

The reason is the collection: 1,200 short documents of near-uniform length give the length
normalisation nothing to normalise. A sweep of b is only informative on a corpus with wide
document-length variance, which this generator does not produce. This is a benchmark with
no shape, and it is included here labelled as such rather than quietly omitted.

---

## 5. Vector search and HNSW

### 5.1 Distance kernels

`./build/vector_test` → `BenchmarkScalarVersusUnrolled`

```
distance kernels: dim=128, 2048 vectors, 200 reps (optimized build)
kernel                    ns/call     Mcalls/s
dot        scalar           29.37         34.1
dot        unrolled         14.44         69.3
l2squared  scalar           32.48         30.8
l2squared  unrolled          9.17        109.1
cosine     scalar           81.51         12.3
cosine     unrolled         32.02         31.2
unrolled speedup: dot 2.03x  l2squared 3.54x  cosine 2.55x
```

Varied: scalar loop vs 8-wide unroll with 4 accumulators. Held fixed: dimension 128, 2,048
vectors, 200 reps, same data.

This is a wall-clock benchmark, so five consecutive runs:

| run | dot | l2squared | cosine |
|---|---|---|---|
| 1 | 2.24x | 3.69x | 2.63x |
| 2 | 2.06x | 3.48x | 2.41x |
| 3 | 2.14x | 3.46x | 2.42x |
| 4 | 2.00x | 3.44x | 2.46x |
| 5 | 2.05x | 3.31x | 2.45x |
| plus the run above | 2.03x | 3.54x | 2.55x |
| **range** | **2.00–2.24x** | **3.31–3.69x** | **2.41–2.63x** |

README cross-check: the README quotes **2.32x (dot)**, 3.69x (L2²), 2.46x (cosine).
3.69x and 2.46x are inside the measured range. **2.32x for dot is above it** — six runs
here top out at 2.24x. Reported correct value: dot is worth **2.0–2.2x**, call it 2.1x.

What this supports: the gain is real and it is specifically the *reduction* that was being
left on the table, not the multiplies. `-O2` vectorizes the element-wise multiply
(`fmul.4s`) and then extracts every lane to sum with serial `fadd`s through one
accumulator, because float addition is not associative and the compiler is not allowed to
reassociate without being told. Writing four accumulators in the source is the author
granting that permission, which is why L2² — the kernel with the most reduction relative
to its other work — gains the most.

What it does not support: the idea that this is free. `ReassociationDivergesUnderCancellation`
exists to make the cost explicit; the two implementations are not required to agree bit for
bit, and on a sum that cancels they can diverge without bound. On the specific cancelling
case the test constructs, both happen to land on exactly 0 on this machine
(`scalar=0 unrolled=0`), which is a property of this input and this target, not a guarantee.

### 5.2 Recall against ef_search, with range

`./build/hnsw_test` → `RecallRisesWithEfSearchAndApproachesOne`

Two graph configurations over the same 20,000 vectors of dimension 128, 50 clusters,
σ=0.50, ef_construction 200, cosine, heuristic neighbour selection. Deterministic.

M=8 / M0=16, 6 layers:

| ef_search | recall@10 | dist/query | hops | vs brute force |
|---|---|---|---|---|
| 10 | 0.6875 | 206.6 | 21.7 | 96.8x |
| 16 | 0.8400 | 260.2 | 28.1 | 76.9x |
| 24 | 0.9087 | 306.1 | 35.2 | 65.3x |
| 32 | 0.9488 | 347.8 | 43.2 | 57.5x |
| 48 | 0.9850 | 402.1 | 59.0 | 49.7x |
| 64 | 0.9975 | 436.6 | 74.8 | 45.8x |
| 96 | 1.0000 | 475.2 | 106.8 | 42.1x |
| 256 | 1.0000 | 562.5 | 266.6 | 35.6x |

M=16 / M0=32, 4 layers:

| ef_search | recall@10 | dist/query | hops | vs brute force |
|---|---|---|---|---|
| 10 | 0.8950 | 305.0 | 17.8 | 65.6x |
| 16 | 0.9662 | 364.3 | 23.5 | 54.9x |
| 24 | 0.9925 | 412.1 | 31.2 | 48.5x |
| 32 | 0.9975 | 441.6 | 39.1 | 45.3x |
| 48 | 1.0000 | 473.5 | 55.0 | 42.2x |
| 256 | 1.0000 | 592.1 | 262.9 | 33.8x |

The point of showing both is the shape of the dial, not the top of the curve. At M=8 the
graph needs ef=96 to saturate; at M=16 it is done by ef=48, and it starts 0.21 higher at
ef=10. Degree buys recall at low beam width — which is also why the *measurement* needs M
held low to have range at all.

That is the lesson the README calls out and it is worth restating as a methodological
point: an earlier dataset reported recall@10 = 1.0000 at every ef_search from 16 up. That
curve cannot distinguish a good graph from a badly connected one, because a graph with
serious connectivity problems would also report 1.0 on it. Raising n and dimension and
lowering M gave the axis range, and the test now **asserts recall at ef=k is below 0.9**,
so a future change that makes this benchmark trivial again fails rather than passes.

Monotonicity is checked separately (`RecallIsMonotoneInEfSearch`): 0.8617 at ef=10 rising
to 1.0000 by ef=40, never decreasing.

Other metrics, same structure:

| metric | ef=10 | ef=32 | ef=128 |
|---|---|---|---|
| inner product (n=10,000, dim 64) | 0.9375 | 0.9925 | 1.0000 |
| squared L2 (n=10,000, dim 64) | 0.7525 | 0.9750 | 1.0000 |

### 5.3 The neighbour-selection heuristic

`./build/hnsw_test` → `HeuristicSelectionBeatsTopM`. n=20,000, dim 128, M=8/M0=16,
recall@10, same data and seed for both arms. Deterministic.

| ef_search | heuristic | top-M | delta | dist heuristic | dist top-M |
|---|---|---|---|---|---|
| 10 | 0.6900 | 0.3163 | +0.3738 | 215.5 | 140.4 |
| 32 | 0.9437 | 0.3825 | +0.5612 | 347.9 | 208.2 |
| 64 | 0.9938 | 0.3913 | +0.6025 | 442.5 | 263.9 |
| 128 | 1.0000 | 0.3925 | +0.6075 | 505.0 | 309.7 |
| 256 | 1.0000 | 0.3925 | +0.6075 | 573.3 | 336.2 |
| 512 | 1.0000 | 0.3925 | +0.6075 | 1353.6 | 357.0 |

This is the clearest result in the suite, and the reason is the second column's *shape*
rather than its level. Top-M **plateaus at 0.3925 from ef=128 onward and never improves
again**, while remaining cheaper per query at every beam width. Keeping the M nearest
candidates as neighbours produces tight clumps with no long-range edges, so a wider beam
explores more of the clump it is already in and cannot reach the clump the answer is in.
The heuristic's whole job is to keep an edge to a candidate that is far from the ones
already kept, and that is what turns beam width into recall.

The all-but-free reading: a benchmark that only measured cost would have *preferred*
top-M. It is cheaper at every row. That is what makes the recall column non-optional.

### 5.4 Level distribution

`./build/hnsw_test` → `LevelDistributionDecaysExponentially`, M=16 so each layer should be
about 16x smaller:

| layer | nodes | ratio to layer below |
|---|---|---|
| 0 | 20,000 | — |
| 1 | 1,268 | 15.8 |
| 2 | 81 | 15.7 |
| 3 | 4 | 20.2 |

The `1/ln(M)` level draw is doing what it claims. The top ratio is noisy because the
denominator is 4.

Graph invariants across three seeds (3,000 nodes each) — symmetric edges, no self-loops,
every node reachable:

```
seed 0x11: 3000 nodes, 4 layers, 44774 directed edges, 14.9 avg degree at layer 0
seed 0x22: 3000 nodes, 4 layers, 44820 directed edges, 14.9 avg degree at layer 0
seed 0x33: 3000 nodes, 4 layers, 44970 directed edges, 15.0 avg degree at layer 0
```

Average degree 14.9 against M0=16 is the heuristic declining to fill every slot, which is
expected rather than a defect.

### 5.5 Where brute force wins, measured properly

```
$ ./build/meridian_eval --corpus $C/corpus.tsv --embeddings $C/embeddings.tsv \
      --queries $C/queries.tsv --qrels $C/qrels.txt \
      --query-embeddings $C/query_embeddings.tsv --repeat 7 --ann-recall 10,64,256
```

The README's crossover table mixes dimensions — its 1,200-vector row is dimension 10 and
its 50,000-vector row is dimension 52 — so it cannot separate "more vectors" from "longer
vectors". Here is the same sweep at **constant dimension 52**, varying only n, with
`--topics 48 --vocabulary-size 28800` held fixed so the embedding dimension does not move:

| vectors (dim 52) | exact p50 | HNSW ef=64 p50 | speedup | recall@10 at ef=64 |
|---|---|---|---|---|
| 1,200 | 10.42 µs | 13.04 µs | **0.80x — slower** | 1.0000 |
| 1,500 | 12.46 µs | 12.17 µs | 1.02x — a tie | 1.0000 |
| 2,000 | 16.08 µs | 10.75 µs | 1.50x | 1.0000 |
| 5,000 | 36.58 µs | 9.62 µs | 3.8x | 1.0000 |
| 10,000 | 70.88 µs | 11.58 µs | 6.1x | 1.0000 |
| 20,000 | 136.88 µs | 15.62 µs | 8.8x | 0.9781 |
| 50,000 | 336.33 µs | 19.83 µs | **17.0x** | 0.9615 |

**The crossover is at about 1,500 vectors** at dimension 52 and ef=64. Below it the graph
loses; above 2,000 it is winning and the margin grows roughly linearly because exact is
linear in n and the graph is close to flat (9.6–19.8 µs across a 25x range of n).

And at dimension 10, which is the README's small-corpus case:

| vectors (dim 10) | exact p50 | HNSW ef=64 p50 | speedup |
|---|---|---|---|
| 1,200 | 5.38 µs | 5.83 µs | **0.92x — slower** |

README cross-check: the README's 1,200-vector row (5.33 vs 5.88 µs) reproduces —
measured 5.38 vs 5.83, same conclusion, brute force wins. Its 2,000 row (15.9 vs 11.5,
1.4x) reproduces as 16.08 vs 10.75, 1.50x. Its 10,000 row (70.2 vs 12.3, 5.7x) reproduces
as 70.88 vs 11.58, 6.1x. Its 50,000 row (330.2 vs 20.6, 16.0x) reproduces as 336.33 vs
19.83, 17.0x; across the several `--ann-recall` runs in this document the ratio ranged
**14.5x to 17.0x**, so 16.0x is a sample from the middle of that band rather than a stable
figure. Its claim that "the graph is ahead by ~2,000 vectors" is right but understated —
at dimension 52 the graph is already ahead at 1,500.

What this supports: there is a corpus size below which the ANN path is a loss, it is
small, and it is measurable. What it does not support: a general crossover. It moves with
dimension, with M, with ef_search, and with whether the exact path is allowed to use the
unrolled kernels (it is). At ef=10 the graph wins much earlier; at ef=256 much later.

### 5.6 Recall and cost at the collection the rest of this document uses

```
$ ./build/meridian_eval ... --ann-recall 10,16,32,64,128,256,512     # /tmp/coll
```

50,000 vectors, dimension 52, cosine, M=16/M0=32, ef_construction 200, 96 queries with
embeddings, top-10:

| ef_search | recall@10 | p50 µs | p99 µs | distances/query |
|---|---|---|---|---|
| exact | 1.0000 | 334.83 | 360.83 | 50,000 |
| 10 | 0.6625 | 5.79 | 18.71 | 320.2 |
| 16 | 0.7698 | 8.00 | 21.33 | 396.8 |
| 32 | 0.8865 | 13.92 | 37.71 | 565.7 |
| 64 | 0.9615 | 23.12 | 65.58 | 787.0 |
| 128 | 0.9760 | 36.00 | 89.46 | 998.4 |
| 256 | 0.9990 | 55.21 | 196.00 | 1,179.6 |
| 512 | 1.0000 | 87.96 | 207.46 | 1,267.8 |

Recalls are deterministic and reproduce exactly. Latencies move a few percent between
runs (a second run gave 336.33 / 4.92 / 19.83 / 48.12 at exact / 10 / 64 / 256).

README cross-check: every recall figure matches. The latency column in the README
(330.2 / 5.0 / 20.6 / 49.5 / 78.5 µs) is a few percent below what reproduces here — the
same measurement on a slightly quieter machine.

The honest framing of this table: the beam buys recall across the whole range **and**
distances per query grow only 4x while recall goes 0.66 → 1.00, because a wider beam
revisits much of what a narrower one already touched. 1,268 distance computations for
recall 1.0 against 50,000 for the same answer is the result; 39x fewer distances, and 3.8x
in wall clock after the graph's pointer-chasing and heap traffic are paid for.

### 5.7 Approximate against exact through the serving pipeline

```
$ ./build/meridian_eval ... --mode hybrid --candidate-k 10 --top-k 10 \
      --no-cache --repeat 5 --semantic exact
$ ... --semantic hnsw --ef-search 640
```

On the 50,000-document collection:

| | NDCG@10 | P@10 | recall@10 | MRR | unjudged | ann stage p50 | total p50 |
|---|---|---|---|---|---|---|---|
| `--semantic exact` | 0.6779 | 0.7659 | 0.2795 | 0.9962 | 0 | 1,186.71 µs | 1,200.71 µs |
| `--semantic hnsw --ef-search 640` | 0.6779 | 0.7659 | 0.2795 | 0.9962 | 0 | 89.21 µs | 100.00 µs |

Identical on **every** quality metric, at **13.3x** less stage time.

README cross-check: the README says 12.7x. Measured 13.3x here; same measurement, timing
spread.

### 5.8 The ANN path through the serving layer agrees with exact

`./build/semantic_ann_test`

```
top-10 agreement with exact search at a beam of 400: 1.0000
distance computations: exact stage 0, hnsw stage 1215, corpus 10000
worst per-query top-10 agreement at the narrowest beam: 0.7000
ef_search   10: top-10 agreement with exact 0.9250
ef_search   40: top-10 agreement with exact 0.9937
ef_search  160: top-10 agreement with exact 1.0000
ef_search  640: top-10 agreement with exact 1.0000
```

The second line is the test that exists because of a specific failure: the `ef_search`
axis was inert for the entire period in which the HNSW suite and the serving suite were
both green, because the pipeline used brute-force exact search and never touched the graph.
`exact stage 0 distance computations, hnsw stage 1215` is the assertion that the backend
switch changes what actually runs. Two components passing their own tests says nothing
about the pair.

Also asserted here: the approximate path honours tombstones and filters (post-filtered
against the pinned generation); concurrent queries through it agree with serial ones; a
corpus with no embeddings is refused rather than built at zero dimensions; mixed embedding
dimensions fail and name the offending document; a rebuild leaves deleted documents out of
the graph.

---

## 6. Fusion

### 6.1 Lexical, semantic, fused

Both paths run in every row — only the fusion weights change — so latency includes the
same work and the comparison is of fusion rather than of two different pipelines.

```
$ ./build/meridian_eval ... --mode hybrid --repeat 5 --no-cache --no-per-query \
      --candidate-k 50 --sweep weights=1/0,1/1,0/1
```

**Demo collection** (1,200 documents):

| configuration | NDCG@10 | P@10 | MRR | p99 µs | verdict |
|---|---|---|---|---|---|
| rrf 1.00/0.00 — lexical only | 0.8161 | 0.2521 | 0.8611 | 52.29 | — |
| **rrf 1.00/1.00 — both** | **0.9509** | **0.3875** | **1.0000** | 42.88 | better |
| rrf 0.00/1.00 — semantic only | 0.2203 | 0.2500 | 0.2500 | 36.12 | worse |

Fusion is worth **+0.1348 NDCG@10** over lexical alone *and* is faster at p99 (0.82x).
Every figure matches the README exactly; this is deterministic on the quality axis.

**Large collection** (50,000 documents), same sweep, candidate_k 50 and 200:

| configuration | NDCG@10 | P@10 | recall@10 | MRR | p99 µs | verdict |
|---|---|---|---|---|---|---|
| lexical only, ck=50 | 0.4737 | 0.3508 | 0.2756 | 0.6111 | 860.08 | — |
| **both, ck=50** | **0.6986** | 0.7348 | 0.2792 | 0.9798 | 896.83 | better |
| semantic only, ck=50 | 0.4365 | 0.7121 | 0.0068 | 0.7121 | 912.04 | worse |
| lexical only, ck=200 | 0.4690 | 0.3508 | 0.2756 | 0.6212 | 1061.71 | — |
| **both, ck=200** | **0.7305** | 0.7568 | 0.2795 | 0.9867 | 1022.46 | better |
| semantic only, ck=200 | 0.4691 | 0.7197 | 0.0069 | 0.7197 | 1011.38 | no-change |

Fusion is worth **+0.2615 NDCG@10** here — nearly twice the demo collection's margin — and
is again not slower at p99. The hybrid result is the most robust finding in this document:
it holds on both collections, at two candidate budgets, and it gets *stronger* as the
collection grows.

The semantic-only row deserves a note that the README does not give it. On the demo
collection it scores 0.2203, which looks like a broken path; on the large collection it
scores 0.4365–0.4691, statistically indistinguishable from lexical-only. The difference is
the judged set's composition: the demo collection's 48 queries are 36 identifiers (75%),
and an identifier has no embedding by construction, so the semantic path cannot answer
three quarters of the queries at all. The large collection's 132 queries are only 27%
identifiers. **0.2203 is mostly a statement about the query mix, not about embeddings.**
The `recall@10` column is the one that exposes the real weakness: 0.0125 and 0.0068 against
the lexical path's 0.755 and 0.276. Topic-level neighbourhoods with no lexical precision.

What it supports: the two paths fail on disjoint queries, and fusing them recovers both.
What it does not support: a transferable quality number. 0.9509 is a statement about this
generator, and the P@10 of 0.3875 against an MRR of 1.0000 is the signature of a judged
pool where topic queries have hundreds of relevant documents and k=10 cannot reach them.

### 6.2 Fusion strategies — and this one reverses

```
$ ./build/meridian_eval ... --mode hybrid --repeat 5 --no-cache --no-per-query \
      --candidate-k 50 --sweep strategy=rrf,norm-linear,zscore-linear
```

**Demo collection:**

| strategy | NDCG@10 | P@10 | MRR | p99 µs | verdict |
|---|---|---|---|---|---|
| rrf (baseline) | 0.9509 | 0.3875 | 1.0000 | 58.33 | — |
| **norm-linear** | **0.9614** | 0.3917 | 1.0000 | 43.21 | better |
| zscore-linear | 0.9505 | 0.3875 | 1.0000 | 45.79 | better |

**Large collection:**

| strategy | NDCG@10 | P@10 | MRR | p99 µs | verdict |
|---|---|---|---|---|---|
| **rrf (baseline)** | **0.6986** | 0.7348 | 0.9798 | 874.42 | — |
| norm-linear | 0.6735 | 0.7455 | 0.9924 | 878.67 | worse |
| zscore-linear | 0.6642 | 0.7538 | 0.9874 | 850.08 | worse |

The ordering **inverts**. On 1,200 documents norm-linear wins by 0.0105; on 50,000 it loses
by 0.0252 and RRF is best. README cross-check: the README's "norm-linear 0.9614, RRF
0.9509, z-score 0.9505" is exactly right *for the demo collection* and does not generalise.

The mechanism is the one that makes rank fusion popular in the first place: min-max and
z-score normalisation both read the score *distribution* of each candidate list, and those
distributions are not comparable between a BM25 list and a cosine list. With 50 candidates
drawn from 1,200 documents the two lists overlap heavily and the normalisation is
approximately monotone in rank anyway; with 50 drawn from 50,000 they overlap much less,
one list's scores are compressed relative to the other's, and the normalisation starts
mattering — badly. RRF never looks at a score.

Note also that P@10 and MRR go the *other* way on the large collection: norm-linear and
zscore-linear both beat RRF on P@10 and MRR while losing on NDCG@10. NDCG is the only one
of the three that is sensitive to graded relevance and to position within the top 10, so
the disagreement means the score-based strategies are retrieving comparably relevant
documents but ordering them worse. Picking a fusion strategy on P@10 alone would have
chosen differently.

`./build/hybrid_test` pins the normalisation behaviour directly:

```
min-max: doc10 1.000000 (rank 0), doc100 1.000000 (rank 1)
z-score: doc10 0.445383 (rank 0), doc100 0.322839 (rank 1)
cluster spread: min-max 0.002008, z-score 0.004642
```

The first line is the trap: min-max maps the top of each list to exactly 1.0, so two
documents that were far apart in raw score become indistinguishable at the head. Z-score
keeps them separated. That is a reason to prefer z-score on principle and the measured
NDCG above does not support doing so, which is the sort of disagreement worth leaving in.

### 6.3 Reranker features

`./build/rank_test`

```
popularity: 0 -> 0.0000, 100 -> 0.8219, 1e8 -> 0.9485
freshness outweighs popularity at weight 0.95
end to end: ml_intro 6.1315, ml_history 5.5831, machine_shop 2.1384
```

Popularity is compressed logarithmically — a jump from 100 to 10^8, six orders of
magnitude, moves the feature 0.8219 → 0.9485. That is deliberate: a linear popularity
feature in a linear model lets one viral document outrank the entire corpus. The end-to-end
line is a hand-checked ranking over three documents where the intended order is known.

---

## 7. Serving and caches

### 7.1 Where the time actually goes

```
$ ./build/meridian_eval ... --mode hybrid --top-k 5 --repeat 15 --cold \
      --query "baguna badoji bajibi"
```

One real query, 50,000-document collection, candidate_k 200, caches dropped before every
sample, 15 samples, per-stage nanoseconds straight out of `QueryTrace`:

| stage | median ns | min | max |
|---|---|---|---|
| parse | 250 | 208 | 4,083 |
| lexical | 54,417 | 52,667 | 273,625 |
| ann | 38,959 | 37,708 | 141,584 |
| merge (fuse) | 14,417 | 14,000 | 42,250 |
| rerank | 24,333 | 23,709 | 43,708 |
| materialize | 42 | 0 | 291 |
| **total** | **107,458** | 104,375 | 379,833 |

Two things to read off it.

**The stages do not sum to the total.** Median stages add to 132,418 ns against a measured
total of 107,458 ns. That is not an accounting error; it is the overlap. `lexical` and
`ann` run concurrently, so the total approaches `parse + max(lexical, ann) + merge +
rerank + materialize` rather than their sum, and the ~25 µs difference is the ANN path
hiding behind the longer lexical path. The trace reports per-stage wall time per stage,
which is the number you want for attribution, and the pipeline reports its own total,
which is the number you want for a latency budget. They are different numbers on purpose.

**Where the 107.5 µs goes, accounting for the overlap.** Because `lexical` and `ann` run
concurrently, elapsed retrieval is `parse + max(lexical, ann)` = 54.7 µs, not their sum:

| | elapsed | share of total |
|---|---|---|
| parse + retrieval (lexical, with ann hidden inside it) | 54.7 µs | 51% |
| fuse + rerank | 38.8 µs | 36% |
| materialize | 0.04 µs | 0.04% |
| attributed to no stage | 14.0 µs | 13% |

The last row is the honest one: 14.0 µs of the measured total belongs to no stage —
queue hand-off, the deadline checks at stage boundaries, and the result-cache probe. Nothing
in the trace breaks it down further. `materialize` is 42 ns because it runs last and touches
only the five documents actually being returned, which is the entire reason it is last.

Warm, same query, result cache enabled:

```
run 1  result cache: miss  stage ns: parse 4250, lexical 302958, ann 168459, merge 45667, rerank 45250, materialize 208, total 417750
run 2  result cache: HIT   stage ns: parse 1459, lexical 0, ann 0, merge 0, rerank 0, materialize 0, total 1541
run 3  result cache: HIT   stage ns: parse  875, lexical 0, ann 0, merge 0, rerank 0, materialize 0, total  916
```

A hit does **no retrieval work at all** — every stage is exactly zero, not merely small,
and the total is 916–1,541 ns against 107 µs cold. A benchmark that ran the same query
three times and reported the mean would be reporting 70–100x optimistic, and would be
measuring a hash lookup. Every quality sweep in this document uses `--no-cache` for that
reason.

### 7.2 Parallelism: when the hand-off is worth paying for

`./build/serve_test`

```
two 20ms paths: serial total 42.9 ms, parallel total 25.1 ms
cheap query median total: serial 0.056 ms, parallel 0.052 ms (hand-off costs 0.000 ms)
stages account for 94.8% of 0.150 ms
```

Two deliberately expensive 20 ms paths: 42.9 ms serial, 25.1 ms parallel — **1.71x**, and
the parallel figure is `max(20, 20)` plus about 5 ms of hand-off rather than `20 + 20`. On
a cheap query where both paths are tens of microseconds the two are within noise of each
other (0.056 vs 0.052 ms).

What it supports: the overlap is real and its break-even is the hand-off cost. On the real
query in 7.1, with lexical at 54 µs and ann at 39 µs, the overlap saved about 25 µs of a
132 µs serial total — so the parallel path is winning at this collection size, which is a
different answer from the one the cheap-query test gives and both are correct for their
input.

### 7.3 Admission control

`OverloadIsRefusedAndAcceptedQueriesStayFast`, three runs:

```
offered 60, accepted 3, refused 57 in 93.2 ms   worst accepted wait 32.3 ms, bound 300.0 ms
offered 60, accepted 3, refused 57 in 97.0 ms   worst accepted wait 33.0 ms, bound 300.0 ms
offered 60, accepted 3, refused 57 in 90.5 ms   worst accepted wait 33.4 ms, bound 300.0 ms
```

60 queries offered against a bounded queue; 3 accepted, 57 refused, and the whole burst is
resolved in 90–97 ms. The number that matters is the **worst accepted wait: 32–33 ms
against a 300 ms bound**. A queue that grew instead of refusing would have admitted all 60
and blown the bound for most of them; refusing 57 immediately is what keeps the 3 that were
accepted inside their deadline.

The counts (60/3/57) are structural and reproduce exactly. README cross-check: 93 ms and
33 ms both reproduce.

### 7.4 Deadlines

Structural assertions, no printed numbers: an impossible deadline is refused up front and
returns no hits rather than a partial list; a deadline expiring mid-query **names the stage
it expired in** and returns nothing rather than a truncated result set. The second is the
same principle as the corrupt-block fix in 1.5 — a partial answer that reports success is
worse than a failure.

### 7.5 SLRU scan resistance

`./build/cache_test` → `ScanDoesNotEvictTheHotSet`. The test asserts rather than prints, so
this is a description of what it pins:

- Capacity is 20 entries; 80% of each shard is the protected segment, so 16 may be
  protected.
- 8 keys are inserted and then **looked up once more**. The second access is what promotes
  out of probation, and one access — exactly what a scan produces — is deliberately not
  enough.
- 500 unique keys are then inserted, a pure scan.
- **All 8 hot keys survive.** `survivors == 8`, asserted exactly.
- `promotions == 8`: the hot set was promoted, the scan was not.
- `evictions == 500 - scan_survivors`, and `survivors + scan_survivors == entries()`.

Under plain LRU the scan would evict all 8. Under SLRU the scan can only occupy the space
the protected segment is not using, which is at most `20 - 8` entries, and that bound is
asserted too.

README cross-check: the README says "`evictions == 500 - survivors`". The assertion is
`evictions == 500 - scan_survivors` — survivors *of the scan*, not of the hot set. With 8
hot survivors and at most 12 scan survivors the two readings give different numbers, so
this is worth correcting in the README even though the property being described is right.

Also asserted in that suite: an oversized entry is refused rather than flushing the cache
to make room; protected-segment overflow demotes rather than evicts; sharding keeps
independent keys apart; concurrent access is safe and the counters stay consistent; the
result cache key distinguishes everything that changes the answer and ignores spacing and
case; the result cache does **not** survive a new generation while the posting cache
**does** (postings are immutable per segment, results are not).

### 7.6 Consistency under concurrent indexing

```
600 queries against 189 flushes and 63 merges     (run 1)
600 queries against 159 flushes and 53 merges     (run 2)
600 queries against 165 flushes and 55 merges     (run 3)
```

Zero inconsistencies in every run — that is the assertion; the flush and merge counts are
whatever the background thread managed in the time available and move run to run. README
cross-check: the README's "171 flushes and 57 merges" is inside this spread (one run during
this session reproduced exactly 171/57); it is a sample, not a result.

`GenerationIsPinnedForTheWholeQuery` is the structural version of the same property.

### 7.7 Concurrency and percentiles

```
800 concurrent queries, result cache hit rate 24.0%, p99 0.224 ms   (0.198 / 0.258 / 0.264 across runs)
mean 10980 ns, p50 1000 ns, p99 500000 ns
p50 2019 (true 1000/2000 boundary), p99 2980 (true ~2990)
stages account for 94.8% of 0.150 ms
```

The hit rate of 24.0% is deterministic (it is a property of the key distribution); p99
varies 0.198–0.264 ms across runs, so the README's 0.240 ms is a sample from that band.

The middle line is the point of reporting percentiles at all: a mean of 10,980 ns over a
distribution whose p50 is 1,000 ns and whose p99 is 500,000 ns. The mean is 11x the median
and 2% of the tail. The last line checks the reservoir keeps percentiles representative
past its capacity — p99 2,980 against a true value near 2,990.

---

## 8. Evaluation harness

### 8.1 Metrics against hand computation

`./build/metrics_test` (30 cases) and `./build/eval_test` (26 cases) print nothing and that
is correct: they pin NDCG, MRR, P@K and R@K against values computed by hand, including
graded judgements, the ideal-DCG denominator, ties, k larger than the result list, empty
result lists, and queries with no relevant documents. Macro-averaging is asserted —
every query counts equally regardless of how densely it was judged, because micro-averaging
hands the aggregate to whichever queries the labelling effort happened to cover.

### 8.2 The candidate_k result, and why it needs two collections

```
$ ./build/meridian_eval ... --mode hybrid --repeat 5 --no-cache --no-per-query \
      --sweep candidate_k=10,50,200,800,2000
```

**Demo collection** (1,200 documents, 48 queries):

| configuration | NDCG@10 | P@10 | recall@10 | MRR | unjudged | p99 µs | cands | verdict |
|---|---|---|---|---|---|---|---|---|
| candidate_k=10 (baseline) | **0.9566** | 0.3896 | 0.7619 | 1.0000 | 0 | 38.21 | 4.0 | — |
| candidate_k=50 | 0.9509 | 0.3875 | 0.7618 | 1.0000 | 0 | 42.17 | 14.0 | worse |
| candidate_k=200 | 0.9215 | 0.3792 | 0.7614 | 1.0000 | 0 | 107.25 | 51.5 | worse |
| candidate_k=800 | 0.8840 | 0.3000 | 0.7575 | 1.0000 | 37 | 387.33 | 201.5 | worse |
| candidate_k=2000 | **0.8827** | 0.2979 | 0.7574 | 1.0000 | **38** | 561.71 | 301.5 | worse |

NDCG@10 **falls** 0.9566 → 0.8827 while p99 rises. The quality column is deterministic and
reproduces exactly on every run. The p99 ratio does not — four runs gave **14.70x, 15.11x,
15.50x, 15.91x**.

README cross-check: 0.9566 → 0.8827 and unjudged 0 → 38 reproduce exactly. The README's
**16.7x** p99 rise does **not** reproduce; the measured value on this machine is
**≈15x (range 14.7–15.9x)**.

Separating the two causes, which is the only way to read this honestly:

- **The judged pool's edge.** `unjudged` is 0 at ck≤200 and 37–38 at ck≥800. An unjudged
  document scores as irrelevant, so part of the decline is the harness penalising documents
  it has no opinion about. The tool says so in a footer rather than leaving it to be
  inferred.
- **Real ranking damage.** But `unjudged` is 0 for the 0.9566 → 0.9215 segment, so the
  first 0.035 of the decline is not the pool. That part is a bigger candidate set giving
  the linear reranker more plausible-but-wrong documents to promote above the right ones.
  `recall@10` barely moves (0.7619 → 0.7574), confirming the right documents were still
  retrieved and were simply ranked lower.

**And now the part the README does not say.** The same sweep on the 50,000-document
collection goes the other way:

| configuration | NDCG@10 | P@10 | MRR | unjudged | p99 µs | cands | verdict |
|---|---|---|---|---|---|---|---|
| candidate_k=10 (baseline) | 0.6748 | 0.7545 | 0.9886 | 15 | 847.96 | 7.8 | — |
| candidate_k=50 | 0.6986 | 0.7348 | 0.9798 | 13 | 867.71 | 36.9 | better |
| candidate_k=200 | 0.7305 | 0.7568 | 0.9867 | 4 | 929.17 | 146.0 | better |
| candidate_k=800 | **0.7306** | 0.7614 | 0.9924 | **0** | 1235.00 | 582.4 | tradeoff |
| candidate_k=2000 | 0.7269 | 0.7614 | 0.9924 | 0 | 1828.21 | 1455.1 | tradeoff |

NDCG@10 **rises** 0.6748 → 0.7306, peaks at candidate_k=800, and only then turns down.
`unjudged` goes the *opposite* direction too — 15 → 0 — because on this collection a wider
candidate set pulls in *more* judged documents, not fewer.

So the README's "more candidates means worse results" is a real, reproducible measurement
**on a 1,200-document collection with 6 topics whose judged pool runs out at around 800
candidates**. On a collection where the judged pool is 56x larger, more candidates is
better up to 800 and the honest verdict is `tradeoff`: +0.0559 NDCG for 1.46x p99. The
transferable finding is not "more candidates is worse" — it is **"the candidate_k curve has
an optimum, the optimum is collection-dependent, and past it a larger budget is pure
cost"**, which is a weaker claim and the one the data supports.

This is also the clearest case for the third verdict existing at all. On the large
collection two rows improve quality and regress p99. A comparison that could only say
better or worse would have labelled them one or the other depending on who wrote the
threshold, and the fact that candidate_k=800 buys +0.0559 NDCG for 1.46x p99 would have
been invisible.

### 8.3 The ef_search axis, and the trap in it

`ef_search` is clamped up to the number of neighbours the semantic stage requests, which is
`candidate_k`, not `top_k`. So an `ef_search` sweep at the default `candidate_k=200`
produces identical rows — not because beam width does not matter, but because every row ran
a beam of 200. The tool emits a note naming the effective beam rather than letting the
table be misread:

```
note: 3 of 5 configuration(s) set ef_search below candidate_k, so the beam they actually
run is candidate_k — up to 2000. [...] ef_search only takes effect above candidate_k.
```

Every sweep below sets `--candidate-k 10` so the axis is live.

**The degenerate case, stated plainly.** On the demo collection the `ef_search` axis cannot
measure anything:

```
$ ./build/meridian_eval ... --repeat 5 --ann-recall 10,32,64,128,256     # demo
  ef_search   recall@k    p50 us      p99 us      dist/query
  exact       1.0000      5.38        7.46        1200
  10          1.0000      1.46        5.00        133.8
  32          1.0000      3.25        7.83        172.5
  64          1.0000      5.83        12.50       194.7
  128         1.0000      10.29       18.08       237.3
  256         1.0000      22.46       38.71       526.8
```

Recall is **1.0000 at every beam width**, because a beam of 10 over a degree-32 graph
already reaches a large fraction of 1,200 documents. And so the pipeline sweep is inert
too:

```
$ ./build/meridian_eval ... --mode semantic --candidate-k 10 --sweep ef_search=10,40,160,640
configuration            NDCG@k     P@k  recall@k     MRR  unjudged  p99 us  verdict
ef_search=10 (baseline)  0.2247  0.2500    0.0125  0.2500         0   26.00  -
ef_search=40             0.2247  0.2500    0.0125  0.2500         0   29.54  worse
ef_search=160            0.2247  0.2500    0.0125  0.2500         0   38.25  worse
ef_search=640            0.2247  0.2500    0.0125  0.2500         0  131.83  worse
```

Four identical quality rows. Every verdict is `worse` and that is the correct reading —
there is no quality left to buy, so a wider beam is pure cost — but **this table cannot
distinguish a good HNSW implementation from a bad one**, and a benchmark that cannot do
that is worse than no benchmark. It is reported here only as the negative control. Every
real ANN measurement in this document uses the 50,000-vector collection, where the axis has
range: recall 0.6625 at ef=10 rising to 1.0000 at ef=512.

### 8.4 ef_search through the full pipeline, on a collection with range

```
$ ./build/meridian_eval ... --no-cache --repeat 5 --candidate-k 10 --top-k 10 \
      --sweep ef_search=10,16,32,64,128,256,512                        # /tmp/coll
```

**Hybrid** (both paths):

| ef_search | index recall@10 (5.6) | NDCG@10 | P@10 | MRR | unjudged | ann-stage p50 | p99 µs |
|---|---|---|---|---|---|---|---|
| 10 | 0.6625 | 0.6726 | 0.7167 | 0.9514 | 65 | 23.50 | 970.25 |
| 16 | 0.7698 | 0.6722 | 0.7212 | 0.9514 | 60 | 27.21 | 944.42 |
| 32 | 0.8865 | 0.6678 | 0.7356 | 0.9697 | 40 | 34.29 | 1006.79 |
| 64 | 0.9615 | 0.6748 | 0.7545 | 0.9886 | 15 | 45.33 | 886.58 |
| 128 | 0.9760 | 0.6722 | 0.7545 | 0.9886 | 15 | 58.83 | 969.83 |
| 256 | 0.9990 | 0.6779 | 0.7659 | 0.9962 | 0 | 78.25 | 994.58 |
| 512 | 1.0000 | 0.6779 | 0.7659 | 0.9962 | 0 | 112.75 | 1042.50 |

**Semantic only** (lexical path removed), same sweep:

| ef_search | NDCG@10 | P@10 | MRR | unjudged | p99 µs | verdict |
|---|---|---|---|---|---|---|
| 10 | 0.3881 | 0.6591 | 0.6591 | 90 | 75.71 | — |
| 16 | 0.3852 | 0.6667 | 0.6667 | 80 | 77.92 | worse |
| 32 | 0.3827 | 0.6894 | 0.6894 | 50 | 128.21 | worse |
| 64 | 0.3925 | 0.7121 | 0.7121 | 20 | 165.42 | tradeoff |
| 128 | 0.3900 | 0.7121 | 0.7121 | 20 | 236.96 | tradeoff |
| 256 | 0.3982 | 0.7273 | 0.7273 | 0 | 262.54 | tradeoff |
| 512 | 0.3980 | 0.7273 | 0.7273 | 0 | 411.33 | tradeoff |

| | NDCG@10 range | absolute | relative |
|---|---|---|---|
| hybrid | 0.6726 → 0.6779 | **0.0053** | 0.78% |
| semantic only | 0.3881 → 0.3982 | **0.0101** | 2.54% |

Dropping index-level recall from 1.00 to 0.66 costs **0.78% of hybrid NDCG@10**. Removing
the lexical path takes the movement to 0.0101 absolute — **1.9x, "roughly double"** — and
2.54% in relative terms, 3.3x.

What this supports: the lexical path independently finds many of the documents the graph
missed, so an ANN recall figure quoted on its own overstates what approximate search costs a
*hybrid* system. That is a real and useful result and it reproduces exactly.

What it does **not** support — and the README overstates this — is a threshold. The README
says quality "is insensitive until index recall falls below about 0.98 and only visibly
degrades below 0.96". The measured curve is **non-monotone**: ef=32 at index recall 0.8865
scores 0.6678, which is *lower* than ef=10 at index recall 0.6625 scoring 0.6726. ef=128 at
recall 0.9760 scores 0.6722, lower than ef=64 at 0.9615 scoring 0.6748. The whole NDCG
range is 0.0053 wide and the non-monotonicity is 0.0070 — larger than the range. There is
no threshold here; **NDCG@10 is flat within noise across the entire recall range**, which is
a stronger version of the same point and does not require inventing a knee.

Read P@10 and MRR instead and the picture is cleaner and monotone — P@10 0.7167 → 0.7659,
MRR 0.9514 → 0.9962, both rising with recall — because `unjudged` falls 65 → 0 over the
same range. That is the pool edge again, and it is pushing NDCG down as fast as better
retrieval pushes it up. The two effects nearly cancel, which is why NDCG@10 looks flat.

### 8.5 The third verdict

The harness's comparison has three outcomes, not two. `kTradeoff` fires when quality
improves and p99 regresses beyond tolerance, and the footer counts them:

```
4 row(s) improve quality and regress p99. Those are decisions about the latency budget,
not results.
```

The fixed case that pins the rule, independent of what any machine measures — NDCG
0.6000 → 0.7000 while p99 goes 1,000 → 1,500 µs — lives in `scripts/demo.sh` step 11 and
is labelled `tradeoff` rather than `better`. A two-outcome comparison calls that a win; it
is a decision, and the explanation carries both numbers so whoever owns the latency budget
can make it.

Every verdict string in this document carries both axes, e.g.

```
candidate_k=800: tradeoff: quality improved and p99 regressed —
  NDCG@10 +0.0559 (0.6748 -> 0.7306) but p99 1.46x (847959ns -> 1235000ns, tolerance 10%)
```

---

## 9. The five claims, re-checked

| # | claim | verdict |
|---|---|---|
| 1 | Rarest-first is worth 1.02x leapfrog, 4.72x leader-scan | **Holds exactly.** Deterministic counters, reproduces byte for byte. |
| 2 | NDCG@10 falls 0.9566 → 0.8827 as candidate_k goes 10 → 2000; p99 +16.7x; unjudged 0 → 38 | **Quality holds exactly. p99 figure is wrong: measured ≈15x (14.7–15.9x over four runs), not 16.7x. And the direction reverses on a larger collection** — on 50,000 documents NDCG *rises* 0.6748 → 0.7306. |
| 3 | Index recall 1.00 → 0.66 costs ~0.8% NDCG@10; removing the lexical path roughly doubles the movement | **Holds exactly.** 0.6726 → 0.6779 = 0.78%; semantic-only moves 0.0101 vs 0.0053 = 1.9x. **But the sub-claim of a knee at recall 0.98 does not hold** — the curve is non-monotone over a range narrower than its own noise. |
| 4 | Brute force wins at 1,200 vectors (5.33 vs 5.88 µs); crossover ~2,000; 16.0x at 50,000 | **Holds.** Measured 5.38 vs 5.83 µs at 1,200 (dim 10); 17.0x at 50,000, with the ratio ranging 14.5–17.0x across runs. At constant dimension 52 the crossover is **~1,500**, slightly earlier than stated. |
| 5 | Write amplification 1.00 → 2.57 after one merge, above the naive 2.0 because the merged segment is more compact | **Holds exactly**, and the explanation is independently confirmed: merging 10 segments of 4,999 documents takes 103,939 dictionary entries to 30,768 and 17.39 MiB to 12.83 MiB. |

Additional README figures checked and **confirmed exactly**: 22,920 C++ / 2,974 Python
lines; 219 tests / 16 suites / 92 Python tests; delta+varint 217,166 vs 800,000 bytes
(3.68x); skipping 200,000/1,563 vs 24,000/188 (8.3x); 12,800 seeks across 80 lists;
heuristic vs top-M at 0.6900/0.3163, 1.0000/0.3925, plateau at 0.3925; every recall figure
in the 50,000-vector ANN table; fusion weights 0.8161 / 0.9509 / 0.2203 and +0.1348;
fusion strategies 0.9614 / 0.9509 / 0.9505; bytes/doc 156.4 → 256.2; admission control
60/3/57 in 93 ms with a 33 ms worst wait; embedding dimension 52; ASAN/UBSAN/TSAN clean.

**Confirmed but not reproducible as a fixed number** (timing-dependent samples the README
presents as results): 1,438 queries against a superseded generation (range 1,384–1,583 over
five runs); 171 flushes and 57 merges (159–189 and 53–63); 800 concurrent queries at p99
0.240 ms (0.198–0.264 ms); the ANN latency column (a few percent faster than reproduces
here).

**Not confirmed:**

- **dot-product kernel speedup 2.32x.** Six runs give 2.00–2.24x. Correct value ≈2.1x.
- **p99 rise of 16.7x on the candidate_k sweep.** Measured ≈15x.
- **"`evictions == 500 - survivors`"** in the SLRU description. The assertion is
  `evictions == 500 - scan_survivors`.
- **"the merge takes 74 distinct terms to 55 and 8.02 KiB to 6.24 KiB."** No corpus in the
  repository produces these figures.
- **Zipf slope −1.086 and intra/inter cosine 0.963 / 0.189.** Nearest configuration
  measured: −1.090 and 0.9645 / 0.1811.
- **"removing the lock produces 7 ThreadSanitizer reports."** Not verified — checking it
  requires removing a lock from `src/`, which was out of scope here. The TSAN build passes
  16/16 *with* the lock in place, which is the half of the claim that can be checked
  without modifying the engine.
- **"the vectorizer... sixteen serial `fadd`s through one accumulator."** The conclusion is
  consistent with the measured speedups, but no disassembly was taken during this run, so
  treat the instruction-level account as an explanation rather than a measurement.

No defects were found in `src/`, `include/`, `tools/` or `python/` during this run.

---

## 10. What is not measured

The limits are structural, not a matter of running more rows.

**No TREC collection, and no real documents at all.** Both corpora come from
`python3 -m meridian_eval generate`. The generator's statistical properties are verified
numerically (§The collections), which is what makes *comparisons between configurations*
meaningful — if norm-linear beats RRF on this corpus, that is a fact about the two fusion
implementations. It does not make NDCG@10 = 0.9509 a statement about documents anyone wrote.
A TREC-style collection with human judgements is the honest next step and it would likely
change several of the rankings in §6 and §8.

**No comparison against Lucene, FAISS, hnswlib, or any vector database.** Every ratio here
is internal: this engine against a different configuration of itself, or against a
brute-force reference written in the same tree with the same kernels. There is no number in
this document that says Meridian is fast. "HNSW is 17x faster than brute force **here**" is
a statement about two code paths that share a distance kernel, a memory layout and a
compiler; it says nothing about how either compares to a tuned library.

**One machine, one architecture, one compiler.** Everything is Apple M4 Pro / arm64 /
clang 21 / `-O2`. The kernel speedups in §5.1 are specifically about what this compiler
does to a reduction on this target; x86-64 with AVX-512, or `-O3` with
`-ffast-math`, would give different numbers and possibly a different conclusion. The
crossover in §5.5 depends on cache sizes.

**Single process, single node.** No distributed sharding, no network, no cross-node fan-out,
no replication. The concurrency measurements are threads in one process sharing one address
space and one page cache.

**The semantic path is serialised.** `HnswIndex::Search` is documented as not thread-safe
and deliberately refuses an internal mutex rather than promise scalability it does not have;
the serving decorator takes a lock. So §7.7's 800 concurrent queries are concurrent in the
lexical path and serialised in the semantic one, and no figure here shows what the ANN path
would do with real parallelism. The right design is one index per shard with per-thread
scratch, and it is not here.

**The HNSW graph is not generation-versioned.** It is built from one snapshot. Deletes are
honoured by post-filtering against the pinned generation's tombstones, but a generation
published after the build is invisible to the semantic path until a rebuild — and the
rebuild cost is measured (5.9–6.9 s for 50,000 vectors of dimension 52) but never amortised
against a continuous indexing stream, because nothing here runs one.

**Filters are post-applied on the vector path**, and the cost of that is not quantified. A
filter selecting 1% of the corpus removes almost everything the graph returned, so the
semantic path contributes a handful of candidates instead of `candidate_k`; the stage
over-fetches 4x to mitigate it. No sweep in this document varies filter selectivity, so the
4x is a guess that happens to be in the code rather than a measured sufficient margin.

**No dictionary compression benchmark.** §2 says so already. Nothing reports dictionary
bytes separately from segment bytes, so the front-coding ratio is unmeasured.

**No steady-state merge trajectory.** §3.1 measures one merge of one tier. Long-run write
amplification under continuous ingestion is not measured.

**No WAND or MaxScore**, so every lexical number here is for exhaustive scoring of the
candidate set. A top-k pruning implementation would change §4 and §7.1 substantially, and
its absence means the `candidate_k` latency curve in §8.2 is steeper than it needs to be.

**Per-field BM25 normalisation is not expressible** against a document record carrying only
a whole-document length, so field-weighted scoring normalises every field by the same
length. That is not BM25F, and §4.4's flat b sweep is partly a consequence.

**Corpus document frequency across per-field dictionaries is a lower bound.** Summing df
over fields double-counts a document holding a term in both title and body, which is the
normal case; the implementation takes the max over fields within a segment and sums across
segments. Exact when one field dominates, an underestimate otherwise — so every IDF here is
very slightly high.
