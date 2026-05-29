"""Generator tests: the corpus has the properties it claims, and it has them repeatably.

A generated collection is only useful if its properties are checkable, so each test here
checks one of the claims the generator's docstring makes rather than checking that the
generator ran.
"""

from __future__ import annotations

import collections
import json
import math
import os
import statistics
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from meridian_eval.corpus import (  # noqa: E402
    GRADE_HIGH_PURITY,
    GRADE_MID_PURITY,
    GeneratorConfig,
    generate,
    grade_for,
    write_collection,
)
from meridian_eval.vocabulary import build_vocabulary, identifier_name, term_name  # noqa: E402

#: A smaller collection than the default, so the suite stays quick while keeping enough
#: documents per topic for the statistical claims to be meaningful.
SMALL = GeneratorConfig(documents=600, topics=5, vocabulary_size=2_000, identifiers=12)


class VocabularyTest(unittest.TestCase):
    def test_term_names_are_unique_and_stable(self) -> None:
        names = [term_name(i) for i in range(5000)]
        self.assertEqual(len(set(names)), 5000)
        self.assertEqual(term_name(0), term_name(0))
        self.assertNotEqual(term_name(0), term_name(1))

    def test_identifiers_tokenize_as_a_single_token(self) -> None:
        # The engine splits on anything that is not a letter or a digit, so an identifier
        # containing punctuation would index as several common tokens and stop being rare.
        for i in (0, 1, 255, 4095):
            token = identifier_name(i)
            self.assertTrue(token.isalnum(), token)
            self.assertEqual(token.lower(), token)

    def test_topics_partition_the_vocabulary_by_stride(self) -> None:
        vocabulary = build_vocabulary(size=600, topics=5, zipf_exponent=1.0, held_out_stride=10)
        # Striding rather than blocking: each topic must span the whole frequency range, or
        # a topic's identity and its terms' frequencies become the same variable.
        for topic in range(5):
            owned = [i for i in range(600) if vocabulary.topic_of[i] == topic]
            self.assertEqual(len(owned), 120)
            self.assertLess(min(owned), 5)
            self.assertGreater(max(owned), 590)

        # Every term belongs to exactly one topic and is either usable by documents or
        # held out, never both.
        document_terms = set(vocabulary.document_terms)
        held_out = {i for terms in vocabulary.topic_query_terms for i in terms}
        self.assertEqual(document_terms & held_out, set())
        self.assertEqual(len(document_terms | held_out), 600)

    def test_weights_are_descending(self) -> None:
        vocabulary = build_vocabulary(size=200, topics=4, zipf_exponent=1.1, held_out_stride=8)
        weights = vocabulary.weights
        self.assertTrue(all(weights[i] > weights[i + 1] for i in range(len(weights) - 1)))

    def test_degenerate_configurations_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_vocabulary(size=10, topics=0, zipf_exponent=1.0, held_out_stride=4)
        with self.assertRaises(ValueError):
            # A stride of 1 would hold out every term, leaving no documents with content.
            build_vocabulary(size=100, topics=4, zipf_exponent=1.0, held_out_stride=1)


class DeterminismTest(unittest.TestCase):
    def test_same_seed_produces_byte_identical_files(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            first = Path(root) / "a"
            second = Path(root) / "b"
            write_collection(generate(SMALL), first)
            write_collection(generate(SMALL), second)

            names = sorted(p.name for p in first.iterdir())
            self.assertIn("corpus.tsv", names)
            self.assertIn("qrels.txt", names)
            self.assertEqual(names, sorted(p.name for p in second.iterdir()))
            for name in names:
                self.assertEqual(
                    (first / name).read_bytes(),
                    (second / name).read_bytes(),
                    f"{name} differs between two runs at the same seed",
                )

    def test_a_different_seed_produces_a_different_corpus(self) -> None:
        # The other half of the claim: the seed has to actually reach the documents.
        with tempfile.TemporaryDirectory() as root:
            first = Path(root) / "a"
            second = Path(root) / "b"
            write_collection(generate(SMALL), first)
            from dataclasses import replace

            write_collection(generate(replace(SMALL, seed=SMALL.seed + 1)), second)
            self.assertNotEqual(
                (first / "corpus.tsv").read_bytes(), (second / "corpus.tsv").read_bytes()
            )

    def test_generate_is_deterministic_in_memory_too(self) -> None:
        a = generate(SMALL)
        b = generate(SMALL)
        self.assertEqual([d.tokens for d in a.documents], [d.tokens for d in b.documents])
        self.assertEqual([q.judgements for q in a.queries], [q.judgements for q in b.queries])
        self.assertEqual(a.term_vectors, b.term_vectors)


class ZipfTest(unittest.TestCase):
    def _rank_frequency(self, collection) -> list[int]:
        counts: collections.Counter[str] = collections.Counter()
        for document in collection.documents:
            counts.update(document.tokens)
        return [count for _, count in counts.most_common()]

    def _log_log_slope(self, ranked: list[int], first: int, last: int) -> float:
        xs = [math.log(rank) for rank in range(first, last + 1)]
        ys = [math.log(ranked[rank - 1]) for rank in range(first, last + 1)]
        mean_x = statistics.mean(xs)
        mean_y = statistics.mean(ys)
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        denominator = sum((x - mean_x) ** 2 for x in xs)
        return numerator / denominator

    def test_rank_frequency_slope_matches_the_configured_exponent(self) -> None:
        # Zipf's law is a straight line of slope -s on log rank against log frequency. The
        # measured slope is the actual claim; "we sampled from a Zipf distribution" is not,
        # because a topical mixture layered on top could easily have flattened it.
        #
        # The stride partition is what preserves the exponent: see vocabulary.py. If topics
        # owned contiguous rank blocks instead, this test is what would fail.
        from dataclasses import replace

        for exponent in (0.85, 1.05, 1.30):
            collection = generate(replace(SMALL, zipf_exponent=exponent, documents=900))
            ranked = self._rank_frequency(collection)
            self.assertGreater(len(ranked), 400)
            slope = self._log_log_slope(ranked, 2, 400)
            self.assertAlmostEqual(
                slope,
                -exponent,
                delta=0.25,
                msg=f"exponent {exponent}: measured log-log slope {slope:.3f}",
            )

    def test_frequencies_are_actually_skewed(self) -> None:
        # The slope test would also pass on a nearly flat curve with a well-fitted line, so
        # the magnitude is checked separately: the commonest term must outnumber the
        # median-ranked one by orders of magnitude, which is what makes posting list lengths
        # vary and skip lists worth having.
        ranked = self._rank_frequency(generate(SMALL))
        self.assertGreater(ranked[0], 100 * ranked[len(ranked) // 2])
        self.assertEqual(ranked[-1], 1)


class HeldOutTermsTest(unittest.TestCase):
    def test_held_out_terms_appear_in_no_document(self) -> None:
        collection = generate(SMALL)
        present = set()
        for document in collection.documents:
            present.update(document.tokens)
        held_out = {
            collection.vocabulary.term(i)
            for terms in collection.vocabulary.topic_query_terms
            for i in terms
        }
        self.assertTrue(held_out)
        self.assertEqual(held_out & present, set())

    def test_semantic_queries_use_only_held_out_terms(self) -> None:
        # This is the property that makes a semantic query a test of the semantic path: the
        # lexical path cannot match any of its terms in any document.
        collection = generate(SMALL)
        present = set()
        for document in collection.documents:
            present.update(document.tokens)

        semantic = [q for q in collection.queries if q.kind == "semantic_topic"]
        self.assertEqual(len(semantic), SMALL.topics)
        for query in semantic:
            for term in query.terms:
                self.assertNotIn(term, present, f"{query.query_id} term {term} is in the corpus")

    def test_lexical_queries_use_terms_that_do_occur(self) -> None:
        collection = generate(SMALL)
        present: collections.Counter[str] = collections.Counter()
        for document in collection.documents:
            present.update(document.tokens)

        lexical = [q for q in collection.queries if q.kind == "lexical_topic"]
        self.assertEqual(len(lexical), SMALL.topics)
        for query in lexical:
            for term in query.terms:
                self.assertGreater(present[term], 0, f"{query.query_id} term {term} matches nothing")


class IdentifierTest(unittest.TestCase):
    def test_identifiers_appear_in_exactly_the_intended_documents(self) -> None:
        collection = generate(SMALL)
        self.assertEqual(len(collection.identifier_placement), SMALL.identifiers)

        for token, expected_ids in collection.identifier_placement.items():
            self.assertEqual(len(expected_ids), SMALL.identifier_documents)
            actual = {d.external_id for d in collection.documents if token in d.tokens}
            self.assertEqual(
                actual,
                set(expected_ids),
                f"{token} is in {sorted(actual)} but was placed in {sorted(expected_ids)}",
            )

    def test_identifier_placement_survives_the_round_trip_to_disk(self) -> None:
        # The manifest is what a test or a demo reads, so it has to agree with the corpus
        # file rather than with the in-memory objects.
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / "corpus"
            collection = generate(SMALL)
            write_collection(collection, out)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))

            tokens_by_doc: dict[str, set[str]] = {}
            for line in (out / "corpus.tsv").read_text(encoding="utf-8").splitlines():
                if line.startswith("#"):
                    continue
                fields = line.split("\t")
                tokens_by_doc[fields[0]] = set(" ".join(fields[1:4]).split())

            for token, doc_ids in manifest["identifiers"].items():
                carrying = {doc for doc, tokens in tokens_by_doc.items() if token in tokens}
                self.assertEqual(carrying, set(doc_ids))

    def test_identifiers_have_no_embedding_and_so_no_semantic_reach(self) -> None:
        # An identifier is not in the term vector table, so a query consisting of one embeds
        # to the zero vector. That is the mechanism by which semantic search cannot find it,
        # and it has to be true rather than assumed.
        collection = generate(SMALL)
        for token in collection.identifier_placement:
            self.assertNotIn(token, collection.term_vectors)

        identifier_queries = [q for q in collection.queries if q.kind == "identifier"]
        self.assertEqual(len(identifier_queries), SMALL.identifiers)
        for query in identifier_queries:
            self.assertTrue(all(v == 0.0 for v in query.embedding), query.query_id)

    def test_identifier_judgements_are_exactly_the_carriers_at_grade_three(self) -> None:
        collection = generate(SMALL)
        for query in collection.queries:
            if query.kind != "identifier":
                continue
            expected = collection.identifier_placement[query.text]
            self.assertEqual(
                sorted(query.judgements), sorted((doc, 3) for doc in expected)
            )


class JudgementTest(unittest.TestCase):
    def test_grades_are_recomputable_from_the_generative_facts(self) -> None:
        # The strongest statement available about a generated ground truth: every grade in
        # the qrels file is reproducible from the recorded facts about the document and the
        # query, through the same public rule. If the grading drifted away from the
        # generative process, this is what would catch it.
        collection = generate(SMALL)
        by_id = {d.external_id: d for d in collection.documents}

        for query in collection.queries:
            if query.kind == "identifier":
                continue
            terms = set(query.terms)
            for doc_id, grade in query.judgements:
                document = by_id[doc_id]
                matches = sum(1 for term in terms if term in set(document.tokens))
                expected = grade_for(document.topic == query.topic, document.purity, matches)
                self.assertEqual(
                    grade,
                    expected,
                    f"{query.query_id}/{doc_id}: topic {document.topic} vs {query.topic}, "
                    f"purity {document.purity:.3f}, matches {matches}",
                )

    def test_grade_rule_thresholds(self) -> None:
        # The rule itself, at its boundaries.
        self.assertEqual(grade_for(True, GRADE_HIGH_PURITY, 0), 3)
        self.assertEqual(grade_for(True, GRADE_HIGH_PURITY - 0.001, 0), 2)
        self.assertEqual(grade_for(True, GRADE_MID_PURITY, 0), 2)
        self.assertEqual(grade_for(True, GRADE_MID_PURITY - 0.001, 0), 1)
        # An off-topic document is marginal only when it uses several of the query's words.
        self.assertEqual(grade_for(False, 0.99, 1), 0)
        self.assertEqual(grade_for(False, 0.0, 2), 1)

    def test_the_judged_pool_covers_every_same_topic_document(self) -> None:
        # The pool is "everything the generator knows could be relevant". Every same-topic
        # document is in it by construction, which is what makes recall a real denominator
        # rather than an artifact of which documents somebody happened to look at.
        collection = generate(SMALL)
        for query in collection.queries:
            if query.kind == "identifier":
                continue
            judged = {doc for doc, _ in query.judgements}
            same_topic = {d.external_id for d in collection.documents if d.topic == query.topic}
            self.assertTrue(same_topic.issubset(judged), query.query_id)

    def test_every_grade_level_is_represented(self) -> None:
        # A ground truth that only ever emits one grade makes NDCG equal to a thresholded
        # metric, and the whole point of grading is lost. The purity spread is what produces
        # the distribution, so this test is really a check on the mixture range.
        collection = generate(SMALL)
        grades: collections.Counter[int] = collections.Counter()
        for query in collection.queries:
            for _, grade in query.judgements:
                grades[grade] += 1
        for level in (0, 1, 2, 3):
            self.assertGreater(grades[level], 0, f"no judgements at grade {level}: {dict(grades)}")

    def test_judgements_have_no_duplicate_documents(self) -> None:
        # A duplicated judgement would double a document's gain and make NDCG exceed 1.
        collection = generate(SMALL)
        for query in collection.queries:
            ids = [doc for doc, _ in query.judgements]
            self.assertEqual(len(ids), len(set(ids)), query.query_id)


class FileFormatTest(unittest.TestCase):
    def test_written_files_match_the_formats_the_engine_reads(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / "corpus"
            collection = generate(SMALL)
            write_collection(collection, out)

            # corpus.tsv: eight tab-separated columns, no embedded tabs or newlines in text.
            lines = [
                line
                for line in (out / "corpus.tsv").read_text(encoding="utf-8").splitlines()
                if not line.startswith("#")
            ]
            self.assertEqual(len(lines), SMALL.documents)
            for line in lines:
                self.assertEqual(len(line.split("\t")), 8)

            # queries.tsv: qid<TAB>text, which is what LoadJudgements parses.
            for line in (out / "queries.tsv").read_text(encoding="utf-8").splitlines():
                if line.startswith("#"):
                    continue
                self.assertEqual(len(line.split("\t")), 2)

            # qrels.txt: TREC's four whitespace-separated columns with an integer grade.
            for line in (out / "qrels.txt").read_text(encoding="utf-8").splitlines():
                if line.startswith("#"):
                    continue
                fields = line.split()
                self.assertEqual(len(fields), 4)
                self.assertEqual(fields[1], "0")
                self.assertIn(int(fields[3]), (0, 1, 2, 3))

            # Every document has an embedding of the declared dimension.
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            dims = manifest["embedding_dimension"]
            embedding_lines = [
                line
                for line in (out / "embeddings.tsv").read_text(encoding="utf-8").splitlines()
                if not line.startswith("#")
            ]
            self.assertEqual(len(embedding_lines), SMALL.documents)
            for line in embedding_lines:
                _, vector = line.split("\t")
                self.assertEqual(len(vector.split(",")), dims)

    def test_query_embeddings_omit_the_zero_vectors(self) -> None:
        # Identifier queries have no usable embedding. Writing a zero vector out would let
        # the semantic path run on a vector that cannot match anything — pure cost, reported
        # as though the semantic path had been given a fair chance.
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / "corpus"
            write_collection(generate(SMALL), out)
            written = {
                line.split("\t")[0]
                for line in (out / "query_embeddings.tsv").read_text(encoding="utf-8").splitlines()
                if not line.startswith("#")
            }
            self.assertEqual(len(written), SMALL.topics * 2)
            self.assertTrue(all(not qid.startswith("id-") for qid in written))

    def test_doc_facts_agree_with_the_documents(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / "corpus"
            collection = generate(SMALL)
            write_collection(collection, out)
            by_id = {d.external_id: d for d in collection.documents}
            for line in (out / "doc_facts.tsv").read_text(encoding="utf-8").splitlines():
                if line.startswith("#"):
                    continue
                fields = line.split("\t")
                document = by_id[fields[0]]
                self.assertEqual(int(fields[1]), document.topic)
                self.assertAlmostEqual(float(fields[2]), document.purity, places=5)
                self.assertEqual(int(fields[3]), document.length)


class ConfigValidationTest(unittest.TestCase):
    def test_invalid_configurations_are_rejected(self) -> None:
        from dataclasses import replace

        with self.assertRaises(ValueError):
            generate(replace(SMALL, documents=0))
        with self.assertRaises(ValueError):
            generate(replace(SMALL, topics=0))
        with self.assertRaises(ValueError):
            generate(replace(SMALL, document_length_min=50, document_length_max=10))
        with self.assertRaises(ValueError):
            generate(replace(SMALL, mixture_min=0.9, mixture_max=0.1))
        with self.assertRaises(ValueError):
            generate(replace(SMALL, query_terms=0))


if __name__ == "__main__":
    unittest.main()
