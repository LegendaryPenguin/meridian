"""Embedding tests: the vector space has to have signal in it, not just numbers."""

from __future__ import annotations

import math
import os
import statistics
import sys
import unittest
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from meridian_eval.corpus import GeneratorConfig, generate  # noqa: E402
from meridian_eval.embeddings import (  # noqa: E402
    build_term_vectors,
    cosine,
    dimension,
    embed_tokens,
    format_vector,
    normalize,
    parse_vector,
    term_weight,
)
from meridian_eval.vocabulary import build_vocabulary  # noqa: E402

SMALL = GeneratorConfig(documents=600, topics=5, vocabulary_size=2_000, identifiers=12)


class VectorArithmeticTest(unittest.TestCase):
    def test_normalize_produces_unit_length(self) -> None:
        unit = normalize([3.0, 4.0])
        self.assertAlmostEqual(math.sqrt(sum(v * v for v in unit)), 1.0)
        self.assertAlmostEqual(unit[0], 0.6)
        self.assertAlmostEqual(unit[1], 0.8)

    def test_normalize_of_zero_is_zero_not_nan(self) -> None:
        # The engine's cosine similarity returns 0 for a zero-length vector rather than NaN
        # because a NaN score sorts unpredictably inside the merge. The same choice here.
        self.assertEqual(normalize([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0])

    def test_cosine_endpoints(self) -> None:
        self.assertAlmostEqual(cosine([1.0, 0.0], [2.0, 0.0]), 1.0)
        self.assertAlmostEqual(cosine([1.0, 0.0], [0.0, 1.0]), 0.0)
        self.assertAlmostEqual(cosine([1.0, 0.0], [-1.0, 0.0]), -1.0)
        self.assertEqual(cosine([0.0, 0.0], [1.0, 1.0]), 0.0)
        with self.assertRaises(ValueError):
            cosine([1.0], [1.0, 2.0])

    def test_vector_text_round_trips(self) -> None:
        # Fixed six decimals so the generated files are byte-comparable; the round trip has
        # to survive that truncation.
        original = [0.5, -0.25, 0.125]
        self.assertEqual(parse_vector(format_vector(original)), original)

    def test_term_weight_rises_with_rarity(self) -> None:
        # Rarer terms count for more in a sum, which is what gives documents in one topic
        # distinct positions inside their cluster.
        self.assertLess(term_weight(0), term_weight(100))
        self.assertLess(term_weight(100), term_weight(3000))


class TermVectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.vocabulary = build_vocabulary(
            size=400, topics=4, zipf_exponent=1.0, held_out_stride=10
        )
        self.vectors = build_term_vectors(self.vocabulary, seed=5)

    def test_every_term_including_held_out_ones_has_a_vector(self) -> None:
        # Held-out terms must have vectors or a semantic query could not reach anything.
        self.assertEqual(len(self.vectors), 400)
        for terms in self.vocabulary.topic_query_terms:
            for index in terms:
                self.assertIn(self.vocabulary.term(index), self.vectors)

    def test_vectors_are_deterministic_and_order_independent(self) -> None:
        again = build_term_vectors(self.vocabulary, seed=5)
        self.assertEqual(self.vectors, again)

        # A term's vector is derived from its own name, so growing the vocabulary must not
        # move the terms that were already in it. Otherwise every parameter change
        # invalidates every previous measurement for a reason unrelated to the change.
        larger = build_vocabulary(size=800, topics=4, zipf_exponent=1.0, held_out_stride=10)
        larger_vectors = build_term_vectors(larger, seed=5)
        shared = self.vocabulary.term(7)
        self.assertEqual(self.vectors[shared], larger_vectors[shared])

    def test_a_different_seed_moves_the_vectors(self) -> None:
        other = build_term_vectors(self.vocabulary, seed=6)
        self.assertNotEqual(self.vectors, other)

    def test_terms_of_one_topic_point_in_a_common_direction(self) -> None:
        # The whole basis of the vector space: topical relatedness is proximity. Checked on
        # the terms before any document is involved, so a failure here localises cleanly.
        dims = dimension(4)
        same = []
        different = []
        for i in range(0, 80):
            for j in range(i + 1, 80):
                a = self.vectors[self.vocabulary.term(i)]
                b = self.vectors[self.vocabulary.term(j)]
                similarity = cosine(a, b)
                if self.vocabulary.topic_of[i] == self.vocabulary.topic_of[j]:
                    same.append(similarity)
                else:
                    different.append(similarity)
        self.assertEqual(len(self.vectors[self.vocabulary.term(0)]), dims)
        self.assertGreater(statistics.mean(same), statistics.mean(different) + 0.3)

    def test_unknown_tokens_contribute_nothing(self) -> None:
        dims = dimension(4)
        known = self.vocabulary.term(3)
        with_unknown = embed_tokens([known, "mrd0000ff"], self.vectors, dims)
        without = embed_tokens([known], self.vectors, dims)
        self.assertEqual(with_unknown, without)

    def test_a_token_list_with_nothing_known_embeds_to_zero(self) -> None:
        dims = dimension(4)
        self.assertEqual(embed_tokens(["mrd0000ff"], self.vectors, dims), [0.0] * dims)


class ClusterSeparabilityTest(unittest.TestCase):
    """Is the topical structure actually recoverable from the vectors alone?

    If it is not, every recall number produced through the semantic path is noise: an ANN
    index's recall measures how faithfully it reproduces brute-force ordering, and when the
    ordering is arbitrary, faithfulness means nothing.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.collection = generate(SMALL)
        cls.dims = cls.collection.dimension
        cls.by_topic: dict[int, list[list[float]]] = {}
        for document in cls.collection.documents:
            cls.by_topic.setdefault(document.topic, []).append(document.embedding)

    def test_documents_are_classified_perfectly_by_nearest_topic_centroid(self) -> None:
        centroids = {}
        for topic, vectors in self.by_topic.items():
            total = [0.0] * self.dims
            for vector in vectors:
                for d in range(self.dims):
                    total[d] += vector[d]
            centroids[topic] = normalize(total)

        correct = 0
        for document in self.collection.documents:
            best = max(centroids, key=lambda t: cosine(document.embedding, centroids[t]))
            correct += int(best == document.topic)
        accuracy = correct / len(self.collection.documents)
        self.assertGreaterEqual(accuracy, 0.99, f"nearest-centroid accuracy {accuracy:.3f}")

    def test_within_topic_similarity_exceeds_across_topic_similarity(self) -> None:
        sample = self.collection.documents[:150]
        intra = []
        inter = []
        for i, a in enumerate(sample):
            for b in sample[i + 1 :]:
                similarity = cosine(a.embedding, b.embedding)
                (intra if a.topic == b.topic else inter).append(similarity)
        self.assertGreater(statistics.mean(intra), statistics.mean(inter) + 0.4)

    def test_clusters_have_internal_structure_to_search(self) -> None:
        # Separable is not sufficient. If every document in a topic had the same vector there
        # would be no ordering inside the cluster, so a top-10 would be an arbitrary tie and
        # ANN recall would be meaningless. The rarity weighting in embeddings.py exists to
        # produce this spread.
        sample = self.by_topic[0][:120]
        similarities = [
            cosine(sample[i], sample[j])
            for i in range(len(sample))
            for j in range(i + 1, len(sample))
        ]
        self.assertLess(min(similarities), 0.995)
        self.assertGreater(max(similarities) - min(similarities), 0.05)

    def test_a_semantic_query_lands_in_its_own_topics_cluster(self) -> None:
        # The payoff: a query whose terms appear in no document still retrieves that topic's
        # documents, which is the thing the lexical path cannot do.
        for query in self.collection.queries:
            if query.kind != "semantic_topic":
                continue
            ranked = sorted(
                self.collection.documents,
                key=lambda d: -cosine(query.embedding, d.embedding),
            )[:10]
            topics = {d.topic for d in ranked}
            self.assertEqual(
                topics,
                {query.topic},
                f"{query.query_id} retrieved topics {sorted(topics)}, wanted {query.topic}",
            )

    def test_an_identifier_query_has_no_semantic_reach(self) -> None:
        # The other half: the semantic path cannot answer an identifier query at all, so the
        # lexical path is not redundant.
        for query in self.collection.queries:
            if query.kind != "identifier":
                continue
            self.assertTrue(all(v == 0.0 for v in query.embedding))
            # A zero query vector has cosine 0 against everything, so there is no ordering
            # for the semantic path to produce.
            similarities = {
                round(cosine(query.embedding, d.embedding), 9)
                for d in self.collection.documents[:50]
            }
            self.assertEqual(similarities, {0.0})

    def test_a_lexical_topic_query_also_lands_in_its_cluster(self) -> None:
        # Not a contradiction: the hybrid path needs the two signals to agree on the easy
        # queries and disagree only where one of them has nothing to say.
        for query in self.collection.queries:
            if query.kind != "lexical_topic":
                continue
            best = max(
                self.collection.documents, key=lambda d: cosine(query.embedding, d.embedding)
            )
            self.assertEqual(best.topic, query.topic)

    def test_embeddings_are_unit_length(self) -> None:
        for document in self.collection.documents[:50]:
            norm = math.sqrt(sum(v * v for v in document.embedding))
            self.assertAlmostEqual(norm, 1.0, places=6)

    def test_dimension_is_topics_plus_the_extra_axes(self) -> None:
        self.assertEqual(self.dims, dimension(SMALL.topics))
        self.assertGreater(self.dims, SMALL.topics)
        bigger = generate(replace(SMALL, topics=8, documents=80))
        self.assertEqual(bigger.dimension, dimension(8))


if __name__ == "__main__":
    unittest.main()
