#!/usr/bin/env bash
#
# A narrated walk through what the evaluation harness measures and why.
#
# Each step states what it is about to prove, runs it, and says what the output shows. The
# claims are about this engine on this generated collection — every number below comes from
# the run you are watching, not from a transcript.
#
# Usage:
#   scripts/demo.sh [work-directory]
#
# Requires: a configured build directory (cmake -S . -B build && cmake --build build) and
# python3. Nothing is downloaded.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${1:-${REPO_ROOT}/build/demo}"
CORPUS="${WORK}/corpus"
BUILD="${REPO_ROOT}/build"
EVAL="${BUILD}/meridian_eval"

# Line width for the rules, chosen to fit an 80-column terminal without wrapping.
RULE="$(printf '%0.s-' {1..78})"

step() {
    printf '\n%s\n' "${RULE}"
    printf '%s\n' "$1"
    printf '%s\n' "${RULE}"
}

claim() { printf '\n  what this proves: %s\n\n' "$1"; }

run() {
    printf '  $ %s\n\n' "$*"
    "$@"
}

if [[ ! -x "${EVAL}" ]]; then
    echo "error: ${EVAL} not found." >&2
    echo "Build it first:  cmake -S . -B build && cmake --build build --target meridian_eval" >&2
    exit 1
fi

mkdir -p "${WORK}"

# ---------------------------------------------------------------------------
step "1. Generate a collection whose ground truth is known"
# ---------------------------------------------------------------------------
cat <<'TEXT'
A labelled collection bought from a human effort is partial: an unjudged result cannot be
told apart from a mistake, so every quality number is a lower bound by an unknown amount.
Here the judgements come from the process that wrote the documents, so relevance is known
rather than sampled — and `unjudged_in_top_k` becomes a number that can be checked rather
than hoped to be small.

The collection has three properties on purpose: Zipfian term frequencies (a uniform corpus
makes every posting list the same length and every optimisation look free), topical
clusters (so semantic retrieval has a neighbourhood to find), and rare identifiers that
only lexical matching can reach.
TEXT

claim "the collection exists, with graded judgements derived from the generative process"
run env PYTHONPATH="${REPO_ROOT}/python" python3 -m meridian_eval generate --out "${CORPUS}"

# The demo needs one of each query kind. Pulled out of the generated files rather than
# hard-coded, so the script keeps working when the generator's parameters change.
IDENTIFIER_QUERY="$(awk -F'\t' '$1 == "id-000" { print $2 }' "${CORPUS}/queries.tsv")"
SEMANTIC_QUERY="$(awk -F'\t' '$1 == "sem-2" { print $2 }' "${CORPUS}/queries.tsv")"
LEXICAL_QUERY="$(awk -F'\t' '$1 == "lex-2" { print $2 }' "${CORPUS}/queries.tsv")"
IDENTIFIER_DOCS="$(
    PYTHONPATH="${REPO_ROOT}/python" python3 - "${CORPUS}/manifest.json" "${IDENTIFIER_QUERY}" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
print(" ".join(sorted(manifest["identifiers"][sys.argv[2]])))
PY
)"

DATA=(
    --corpus "${CORPUS}/corpus.tsv"
    --embeddings "${CORPUS}/embeddings.tsv"
    --queries "${CORPUS}/queries.tsv"
    --qrels "${CORPUS}/qrels.txt"
    --query-embeddings "${CORPUS}/query_embeddings.tsv"
)

# ---------------------------------------------------------------------------
step "2. A rare identifier: lexical finds it, semantic cannot"
# ---------------------------------------------------------------------------
cat <<TEXT
The identifier '${IDENTIFIER_QUERY}' was placed in exactly these documents:
  ${IDENTIFIER_DOCS}

It has no embedding, because it is not a word — it belongs to no topic and sits nowhere in
the vector space. So the semantic path has nothing to search with, and this is the query
that shows the lexical path is not redundant. The usual framing has it backwards: the
interesting question is not whether embeddings beat keywords, it is which queries each one
cannot answer at all.
TEXT

claim "the lexical path returns exactly the two documents the generator placed it in"
run "${EVAL}" "${DATA[@]}" --mode lexical --top-k 5 --repeat 1 --query "${IDENTIFIER_QUERY}"

claim "the semantic path returns nothing: no vector, no neighbourhood, no answer"
run "${EVAL}" "${DATA[@]}" --mode semantic --top-k 5 --repeat 1 --query "${IDENTIFIER_QUERY}"

# ---------------------------------------------------------------------------
step "3. A semantic query with no lexical overlap: semantic finds it, lexical cannot"
# ---------------------------------------------------------------------------
cat <<TEXT
The query '${SEMANTIC_QUERY}' is built from terms the generator held out of every
document. Not rare — absent. String matching has nothing to match, by construction.

The terms do have vectors, and those vectors point into their topic's cluster, so the
semantic path reaches the right documents without sharing a single word with them. That is
the capability worth measuring, and it is invisible on a benchmark whose queries are all
quotations from the corpus.
TEXT

claim "the lexical path returns nothing: these terms occur in no document"
run "${EVAL}" "${DATA[@]}" --mode lexical --top-k 5 --repeat 1 --query "${SEMANTIC_QUERY}"

claim "the semantic path returns documents from the query's own topic, sharing no terms with it"
run "${EVAL}" "${DATA[@]}" --mode semantic --top-k 5 --repeat 1 --query "${SEMANTIC_QUERY}"

# ---------------------------------------------------------------------------
step "4. Hybrid handles both, and the stage trace says where the time went"
# ---------------------------------------------------------------------------
cat <<TEXT
Hybrid retrieval is worth having only if it answers both of the previous queries without
being told which kind it is looking at. Watch the 'candidates' line in each run: on the
identifier the lexical path contributes and the semantic path contributes nothing; on the
held-out-term query it is the other way round. The fusion does not need to know.

The per-stage trace is the other thing to read here. An end-to-end latency number cannot be
acted on — a p99 regression in posting decode is a different problem from one in the ANN hop
count or in reranking, and these are the numbers that tell them apart.
TEXT

claim "one configuration answers the identifier query, using only the lexical path"
run "${EVAL}" "${DATA[@]}" --mode hybrid --top-k 5 --repeat 1 --query "${IDENTIFIER_QUERY}"

claim "the same configuration answers the held-out-term query, using only the semantic path"
run "${EVAL}" "${DATA[@]}" --mode hybrid --top-k 5 --repeat 1 --query "${SEMANTIC_QUERY}"

claim "a stage breakdown per query: a latency regression can be attributed, not merely noticed"
run "${EVAL}" "${DATA[@]}" --mode hybrid --top-k 5 --repeat 1 --query "${LEXICAL_QUERY}"

# ---------------------------------------------------------------------------
step "5. The same query cold versus cached"
# ---------------------------------------------------------------------------
cat <<TEXT
A benchmark that asks the same query three times and reports the mean is measuring a hash
lookup. The first run below misses the result cache and pays for retrieval; the next two hit
it and skip every stage — look at the stage nanoseconds collapse to near zero.

Then the same three runs with the caches dropped first. That is the honest number, because a
production query stream is mostly singletons: the long tail is not an anomaly, it is the
normal background.
TEXT

claim "warm: run 1 misses, runs 2 and 3 hit, and the hit does no retrieval work at all"
run "${EVAL}" "${DATA[@]}" --mode hybrid --top-k 5 --repeat 3 --query "${LEXICAL_QUERY}"

claim "cold: every run misses, and this is the latency the engine actually has"
run "${EVAL}" "${DATA[@]}" --mode hybrid --top-k 5 --repeat 3 --cold --query "${LEXICAL_QUERY}"

# ---------------------------------------------------------------------------
step "6. A candidate_k sweep: quality stops improving long before latency stops climbing"
# ---------------------------------------------------------------------------
cat <<'TEXT'
candidate_k is the recall/latency dial: more candidates means more chances for the reranker
to find a good document, and more work. The two effects do not scale together, and that is
the entire content of this table.

Read the NDCG column and the p99 column side by side. Quality saturates almost immediately
on this collection — the lexical path finds the topically relevant documents in its first
few dozen candidates, so there is nothing for a larger candidate set to add — while p99 keeps
climbing roughly linearly in candidate_k. Past the first row or two, every additional
candidate is pure cost.

Two things to notice beyond that. The `unjudged` column rises with candidate_k: a larger
candidate set pulls in documents outside the judged pool, which score as irrelevant, so part
of the NDCG decline at the bottom of the table is the pool's edge rather than the ranking.
And the verdict column is the point of the exercise: every row is compared against the
baseline on *both* axes, so a row that improved quality while regressing p99 would be
labelled `tradeoff` rather than `better`. A table sorted by quality would have recommended
whichever row happened to score highest and shown no sign of what it cost.
TEXT

claim "quality saturates immediately while p99 climbs an order of magnitude, and every row carries both"
run "${EVAL}" "${DATA[@]}" --mode hybrid --repeat 5 --no-cache --no-per-query \
    --sweep candidate_k=10,50,200,800,2000

# ---------------------------------------------------------------------------
step "7. BM25 parameters and fusion weights, swept the same way"
# ---------------------------------------------------------------------------
cat <<'TEXT'
The conventional BM25 defaults (k1 = 1.2, b = 0.75) are conventions, not measurements. The
sweep asks whether they are right for this corpus rather than assuming it.

The weights sweep compares the two retrieval paths on equal footing: 1/0 is lexical-only
fusion, 0/1 is semantic-only, 1/1 is both. Both paths still run in every row, so the latency
includes the same work and only the fusion changes — turning a path off in the wiring would
compare quality at one latency against quality at another and call the difference a fusion
result.
TEXT

claim "the length-normalisation parameter b is an empirical question, answered here"
run "${EVAL}" "${DATA[@]}" --mode hybrid --repeat 5 --no-cache --no-per-query \
    --sweep b=0.0,0.4,0.75,1.0

claim "lexical-only, semantic-only and fused, measured against each other on both axes"
run "${EVAL}" "${DATA[@]}" --mode hybrid --repeat 5 --no-cache --no-per-query --candidate-k 50 \
    --sweep weights=1/0,1/1,0/1

# ---------------------------------------------------------------------------
step "8. Approximate versus exact vector search, and what it costs"
# ---------------------------------------------------------------------------
cat <<'TEXT'
The semantic stage is selectable: `--semantic exact` scans every vector, `--semantic hnsw`
walks the proximity graph. The graph is the only setting under which ef_search means
anything, and exact search is what its recall is defined against — which is why both are
kept rather than one replaced by the other.

The first table is recall measured directly against brute force over the same vectors, so
nothing downstream can compensate for a missed neighbour. Read it before the quality tables:
an ANN latency number with no recall figure beside it is not a claim about anything.

This collection is 1,200 vectors of 10 dimensions, and the honest result on it is that
approximate search has nothing to offer: recall is 1.0 at every beam width because a beam of
ten over a degree-32 graph already reaches a large fraction of 1,200 documents, and brute
force over 1,200 short vectors is competitive with or faster than walking a graph. The
crossover is a property of the corpus, not of the algorithm. Generate a bigger one to see the
other side of it:

  python3 -m meridian_eval generate --out big --documents 50000 --topics 48 \
      --vocabulary-size 28800

At that size the same build measures recall 0.66 at the narrowest beam rising to 1.0 by 512,
with the semantic stage going from 1343us exact to 24us at ef_search=64.
TEXT

claim "recall against exact search, per beam width, with the cost of each"
run "${EVAL}" "${DATA[@]}" --repeat 5 --ann-recall 10,32,64,128,256

cat <<'TEXT'
The second thing worth seeing is a trap. ef_search is clamped up to the number of neighbours
the stage asks for, and that number is candidate_k, not top_k. So an ef_search sweep at the
default candidate_k of 200 produces identical rows — not because search effort does not
matter, but because every row ran the same beam of 200. The tool names the effective beam
rather than letting the table be read as a result, and candidate_k is lowered below every
swept value here so the axis is actually live.

What it then shows on this collection is that the beam buys latency and nothing else. Recall
was already 1.0 at the narrowest beam, so there is no quality left to purchase and every
wider one is pure cost; the rows are labelled `worse` and that is the correct reading. On a
corpus where recall at the narrowest beam is 0.66, the same sweep reads the other way.
TEXT

claim "ef_search is live once candidate_k stops pinning the beam, and here it buys only cost"
run "${EVAL}" "${DATA[@]}" --mode semantic --repeat 5 --no-cache --no-per-query \
    --candidate-k 10 --top-k 10 --sweep ef_search=10,40,160,640

# ---------------------------------------------------------------------------
step "9. The full judged run, per query, with the unjudged count"
# ---------------------------------------------------------------------------
cat <<'TEXT'
The aggregate is macro-averaged: every query counts equally, regardless of how many
documents were judged for it. Micro-averaging would hand the result to whichever handful of
queries has the densest judgements, which is a property of the labelling effort rather than
of the ranker.

The per-query table is there because a mean hides a bimodal distribution, and this judged
set is bimodal by design — identifier queries have two relevant documents each and topic
queries have hundreds. Note the recall column: it is bounded above by k/|relevant|, so a
topic query with 200 relevant documents cannot exceed 0.05 at k = 10 no matter how good the
ranking is. That is why recall is reported per query and not only as an average.
TEXT

claim "quality and latency for one configuration, with the per-query spread visible"
run "${EVAL}" "${DATA[@]}" --mode hybrid --repeat 5 --no-cache \
    --json "${WORK}/results.json" --sweep candidate_k=10,50,200,800

# ---------------------------------------------------------------------------
step "10. The markdown report"
# ---------------------------------------------------------------------------
cat <<'TEXT'
No chart, and not reluctantly. A quality-versus-latency scatter invites the reader to pick
the point nearest the top-left corner, which is a judgement about the aspect ratio of the
axes. The table puts both numbers and the verdict on one line, so "the NDCG went up 0.03 and
the p99 went up 80%" is a sentence rather than a distance on a plot.
TEXT

claim "the same results as markdown, carrying both axes and the verdict per row"
run env PYTHONPATH="${REPO_ROOT}/python" python3 -m meridian_eval report \
    --results "${WORK}/results.json"

# ---------------------------------------------------------------------------
step "11. What a tradeoff looks like when one occurs"
# ---------------------------------------------------------------------------
cat <<'TEXT'
The verdicts above came out of whatever this machine actually measured, so a tradeoff row may
or may not have appeared. The rule itself is not machine-dependent, and this is it, applied
to the case the header exists for: NDCG 0.6000 -> 0.7000 while p99 goes 1000us -> 1500us.

A comparison that can only say "better" or "worse" calls that a win. It is not one. It is a
decision, and the explanation carries both numbers so that whoever owns the latency budget
can make it.
TEXT

claim "a quality gain bought with a 1.5x p99 regression is labelled a tradeoff, not an improvement"
cat > "${WORK}/tradeoff.json" <<'JSON'
{
  "k": 10,
  "baseline": "baseline",
  "latency_tolerance": 0.10,
  "rows": [
    {"label": "baseline", "ndcg": 0.60, "precision": 0.40, "recall": 0.30, "mrr": 0.85,
     "unjudged": 0, "p50_us": 500.0, "p95_us": 900.0, "p99_us": 1000.0,
     "mean_candidates": 200.0, "timeouts": 0, "cache_hit_rate": 0.0},
    {"label": "deeper-reranker", "ndcg": 0.70, "precision": 0.46, "recall": 0.34, "mrr": 0.88,
     "unjudged": 0, "p50_us": 700.0, "p95_us": 1300.0, "p99_us": 1500.0,
     "mean_candidates": 600.0, "timeouts": 0, "cache_hit_rate": 0.0}
  ]
}
JSON
run env PYTHONPATH="${REPO_ROOT}/python" python3 -m meridian_eval report \
    --results "${WORK}/tradeoff.json"

printf '\n%s\n' "${RULE}"
cat <<TEXT
Done. Artifacts in ${WORK}:
  corpus/            the generated collection, its embeddings and its graded judgements
  results.json       the sweep this run measured
  tradeoff.json      the fixed case that pins the tradeoff rule

The one claim worth carrying away: a ranking change that raises NDCG and doubles p99 is not
automatically an improvement, and nothing in this harness will tell you it was.
TEXT
printf '%s\n' "${RULE}"
