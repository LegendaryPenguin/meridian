"""Adversarial workload tests: each kind has to actually be adversarial.

A workload named "high_frequency" that happens to contain a mid-frequency term is worse
than no workload at all, because it produces a number that looks like a stress measurement.
Each test below checks the property the kind's name claims.
"""

from __future__ import annotations

import collections
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from meridian_eval.corpus import GeneratorConfig, generate  # noqa: E402
from meridian_eval.workloads import (  # noqa: E402
    KINDS,
    WorkloadConfig,
    build_workload,
    read_workload,
    summarize,
    write_workload,
)

SMALL = GeneratorConfig(documents=600, topics=5, vocabulary_size=2_000, identifiers=12)


class WorkloadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.collection = generate(SMALL)
        cls.config = WorkloadConfig()
        cls.workload = build_workload(cls.collection, cls.config)
        cls.frequency: collections.Counter[str] = collections.Counter()
        for document in cls.collection.documents:
            cls.frequency.update(document.tokens)

    def test_every_kind_is_present(self) -> None:
        counts = summarize(self.workload)
        for kind in KINDS:
            self.assertGreater(counts[kind], 0, f"no {kind} queries")
        self.assertEqual(
            sum(counts.values()),
            self.config.high_frequency
            + self.config.rare_term
            + self.config.long_query
            + self.config.cold_cache
            + self.config.burst,
        )

    def test_query_ids_are_unique(self) -> None:
        ids = [query.query_id for query in self.workload]
        self.assertEqual(len(ids), len(set(ids)))

    def test_high_frequency_queries_use_the_longest_posting_lists(self) -> None:
        # The claim is specifically about posting list length, so it is checked against the
        # corpus's own term frequencies rather than against the configured Zipf ranks.
        total_documents = len(self.collection.documents)
        for query in self.workload:
            if query.kind != "high_frequency":
                continue
            carriers = sum(
                1 for d in self.collection.documents if query.text in set(d.tokens)
            )
            # A term in a large fraction of the corpus is what makes a disjunctive query
            # approach a full scan.
            self.assertGreater(
                carriers / total_documents,
                0.20,
                f"{query.text} is in only {carriers}/{total_documents} documents",
            )

    def test_rare_queries_use_tail_terms(self) -> None:
        for query in self.workload:
            if query.kind != "rare_term":
                continue
            carriers = sum(1 for d in self.collection.documents if query.text in set(d.tokens))
            # Under 2% of the corpus. Zero is allowed and is itself interesting: a query
            # matching nothing still pays the parse, pin and cache-lookup costs, which is the
            # fixed overhead this kind is here to expose.
            self.assertLessEqual(
                carriers / len(self.collection.documents),
                0.02,
                f"{query.text} is in {carriers} documents",
            )

    def test_high_frequency_terms_are_far_more_common_than_rare_ones(self) -> None:
        # The two extremes have to actually be at opposite ends, or the pair of measurements
        # says nothing about the range.
        frequent = [q.text for q in self.workload if q.kind == "high_frequency"]
        rare = [q.text for q in self.workload if q.kind == "rare_term"]
        lowest_frequent = min(self.frequency[t] for t in frequent)
        highest_rare = max(self.frequency[t] for t in rare)
        self.assertGreater(lowest_frequent, 50 * max(highest_rare, 1))

    def test_long_queries_have_the_configured_term_count(self) -> None:
        for query in self.workload:
            if query.kind != "long_query":
                continue
            terms = query.text.split()
            self.assertEqual(len(terms), self.config.long_query_terms)
            # Distinct terms: a repeated term collapses in the merge and the iterator count
            # would be lower than the query length suggests.
            self.assertEqual(len(set(terms)), len(terms))

    def test_long_queries_span_several_topics(self) -> None:
        # Terms confined to one topic would co-occur and retrieve a tight set, understating
        # the merge cost this kind exists to measure.
        topic_of = {
            self.collection.vocabulary.term(i): self.collection.vocabulary.topic_of[i]
            for i in range(len(self.collection.vocabulary.terms))
        }
        for query in self.workload:
            if query.kind != "long_query":
                continue
            topics = {topic_of[term] for term in query.text.split() if term in topic_of}
            self.assertGreaterEqual(len(topics), SMALL.topics - 1)

    def test_cold_cache_queries_are_unique_and_marked_cold(self) -> None:
        cold = [q for q in self.workload if q.kind == "cold_cache"]
        self.assertTrue(all(q.cold for q in cold))
        self.assertEqual(len({q.text for q in cold}), len(cold))
        # And nothing else is marked cold: a warm measurement polluted by cache drops is not
        # a warm measurement.
        self.assertTrue(all(not q.cold for q in self.workload if q.kind != "cold_cache"))

    def test_bursts_carry_a_concurrency_above_one(self) -> None:
        bursts = [q for q in self.workload if q.kind == "burst"]
        self.assertTrue(all(q.concurrency == self.config.burst_concurrency for q in bursts))
        self.assertGreater(self.config.burst_concurrency, 1)
        self.assertTrue(
            all(q.concurrency == 1 for q in self.workload if q.kind != "burst"),
            "only bursts should be concurrent",
        )

    def test_every_query_explains_why_it_is_there(self) -> None:
        # The rationale travels with the query into the workload file, so whoever reads a
        # latency table does not have to come back to the source to find out what was being
        # stressed.
        for query in self.workload:
            self.assertTrue(query.rationale, query.query_id)

    def test_workload_is_deterministic(self) -> None:
        again = build_workload(generate(SMALL), self.config)
        self.assertEqual([q.text for q in self.workload], [q.text for q in again])

    def test_burst_concurrency_is_configurable(self) -> None:
        custom = build_workload(self.collection, WorkloadConfig(burst_concurrency=4))
        bursts = [q for q in custom if q.kind == "burst"]
        self.assertTrue(bursts)
        self.assertTrue(all(q.concurrency == 4 for q in bursts))

    def test_file_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "workload.tsv"
            written = write_workload(self.workload, path)
            self.assertEqual(written, len(self.workload))

            loaded = read_workload(path)
            self.assertEqual(len(loaded), len(self.workload))
            for original, restored in zip(self.workload, loaded):
                self.assertEqual(original, restored)

    def test_malformed_workload_line_is_reported_with_its_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "workload.tsv"
            path.write_text("# header\nadv-x\tburst\t2\n", encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                read_workload(path)
            self.assertIn(":2:", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
