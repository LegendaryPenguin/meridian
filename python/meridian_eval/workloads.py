"""Adversarial query workloads: the queries that find the cliffs.

A benchmark built from clean, mid-frequency, three-term queries flatters a search engine.
Every posting list is a convenient size, the result cache is warm by the second pass, the
candidate set never overflows, and nothing contends. The numbers are real and they describe
a workload nobody has.

Each generator below targets one specific way a retrieval system falls over, and the
comment on each one says which:

  - **High-frequency terms** make disjunctive retrieval scan a posting list containing a
    large fraction of the corpus. This is where candidate_k stops being a tuning parameter
    and becomes the thing standing between the engine and a full scan, and where
    rarest-term-first ordering either works or does not.
  - **Rare terms** are the opposite failure: almost no work to do, so the per-query fixed
    costs — parse, pin, cache lookup, result materialization — are the entire latency. A
    p50 measured only on mid-frequency queries hides them completely.
  - **Long queries** multiply the number of posting iterators in the merge. The cost is
    superlinear in practice because the merge heap grows while the per-term work does not
    shrink, and query length in real traffic has a long tail that benchmarks never have.
  - **Cold-cache queries** are unique by construction, so the result cache cannot help.
    This is the honest latency: a benchmark that re-asks one query measures a hash lookup,
    and a production query stream is mostly singletons.
  - **Bursts** apply the same query concurrently. This finds lock contention, a saturated
    helper pool degrading to serial execution, and admission control either refusing load
    or queueing it into a latency cliff.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .corpus import Collection
from .rng import Rng, derive

#: Workload kinds, in the order they are emitted.
KINDS = ("high_frequency", "rare_term", "long_query", "cold_cache", "burst")


@dataclass(frozen=True)
class AdversarialQuery:
    query_id: str
    kind: str
    text: str
    #: How many copies to issue at once. 1 means a plain sequential query.
    concurrency: int = 1
    #: Whether the caches should be dropped before this query runs.
    cold: bool = False
    #: One line of why this query is in the workload, carried into the file so whoever reads
    #: the output does not have to come back here.
    rationale: str = ""


@dataclass(frozen=True)
class WorkloadConfig:
    high_frequency: int = 6
    rare_term: int = 8
    long_query: int = 6
    long_query_terms: int = 24
    cold_cache: int = 12
    burst: int = 3
    burst_concurrency: int = 32


def _frequent_terms(collection: Collection, count: int) -> list[str]:
    """The highest-ranked document-usable terms, which are the longest posting lists."""
    vocabulary = collection.vocabulary
    return [vocabulary.term(index) for index in vocabulary.document_terms[:count]]


def _rare_terms(collection: Collection, count: int) -> list[str]:
    """The lowest-ranked document-usable terms.

    Taken from the tail of the Zipf ranking rather than from the identifiers, because an
    identifier is rare *and* absent from the term dictionary's hot path; a genuinely rare
    ordinary term exercises the same code the frequent terms do, at the other extreme.
    """
    vocabulary = collection.vocabulary
    return [vocabulary.term(index) for index in vocabulary.document_terms[-count:]]


def build_workload(collection: Collection, config: WorkloadConfig | None = None) -> list[AdversarialQuery]:
    config = config or WorkloadConfig()
    seed = collection.config.seed
    queries: list[AdversarialQuery] = []

    for i, term in enumerate(_frequent_terms(collection, config.high_frequency)):
        queries.append(
            AdversarialQuery(
                query_id=f"adv-hf-{i:02d}",
                kind="high_frequency",
                text=term,
                rationale="single highest-frequency term: longest posting list in the corpus",
            )
        )

    for i, term in enumerate(_rare_terms(collection, config.rare_term)):
        queries.append(
            AdversarialQuery(
                query_id=f"adv-rare-{i:02d}",
                kind="rare_term",
                text=term,
                rationale="tail-frequency term: per-query fixed costs dominate",
            )
        )

    # Long queries draw across every topic, so the terms have no reason to co-occur and the
    # merge cannot collapse them. A long query confined to one topic would retrieve a tight
    # set and understate the cost.
    all_terms = [collection.vocabulary.term(i) for i in collection.vocabulary.document_terms[:1200]]
    for i in range(config.long_query):
        rng = Rng(derive(seed, "workload-long", i))
        terms = rng.shuffled(all_terms)[: config.long_query_terms]
        queries.append(
            AdversarialQuery(
                query_id=f"adv-long-{i:02d}",
                kind="long_query",
                text=" ".join(terms),
                rationale=f"{config.long_query_terms} terms across every topic: iterator count in the merge",
            )
        )

    # Cold-cache queries are each unique, and marked cold so the harness drops the caches
    # first. Uniqueness alone is not enough: a posting cache warmed by an earlier query on
    # the same term still helps.
    for i in range(config.cold_cache):
        rng = Rng(derive(seed, "workload-cold", i))
        terms = rng.shuffled(all_terms)[:3]
        queries.append(
            AdversarialQuery(
                query_id=f"adv-cold-{i:02d}",
                kind="cold_cache",
                text=" ".join(terms),
                cold=True,
                rationale="unique query against dropped caches: latency with no cache help",
            )
        )

    # Bursts reuse a mid-frequency term so the burst is measuring concurrency rather than
    # the cost of one expensive query multiplied by the fan-out.
    mid = len(collection.vocabulary.document_terms) // 2
    for i in range(config.burst):
        term = collection.vocabulary.term(collection.vocabulary.document_terms[mid + i])
        queries.append(
            AdversarialQuery(
                query_id=f"adv-burst-{i:02d}",
                kind="burst",
                text=term,
                concurrency=config.burst_concurrency,
                rationale=f"{config.burst_concurrency} concurrent copies: contention and admission control",
            )
        )

    return queries


def write_workload(queries: Sequence[AdversarialQuery], path: Path) -> int:
    lines = ["# wid\tkind\tconcurrency\tcold\ttext\trationale"]
    for query in queries:
        lines.append(
            "\t".join(
                [
                    query.query_id,
                    query.kind,
                    str(query.concurrency),
                    "1" if query.cold else "0",
                    query.text,
                    query.rationale,
                ]
            )
        )
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for line in lines:
            handle.write(line)
            handle.write("\n")
    return len(queries)


def read_workload(path: Path) -> list[AdversarialQuery]:
    queries: list[AdversarialQuery] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = raw.split("\t")
        if len(fields) < 5:
            raise ValueError(f"{path}:{number}: expected at least 5 tab-separated fields")
        queries.append(
            AdversarialQuery(
                query_id=fields[0],
                kind=fields[1],
                concurrency=int(fields[2]),
                cold=fields[3] == "1",
                text=fields[4],
                rationale=fields[5] if len(fields) > 5 else "",
            )
        )
    return queries


def summarize(queries: Iterable[AdversarialQuery]) -> dict[str, int]:
    counts: dict[str, int] = {kind: 0 for kind in KINDS}
    for query in queries:
        counts[query.kind] = counts.get(query.kind, 0) + 1
    return counts
