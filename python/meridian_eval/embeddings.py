"""Embeddings derived from the generative process, not from randomness or a hash.

A vector path can be exercised with random vectors and the latency numbers will be
honest, but every recall number will be noise: with random vectors there is no
neighbourhood to find, so an ANN index's recall measures how well it reproduces an
arbitrary ordering. A hashed bag of words is no better for quality work — two documents
about the same topic in different words land in unrelated places, which is precisely the
case semantic retrieval exists to handle.

So the vectors here are built from the same topic structure the documents are: a term's
vector points along its topic's axis, a document's vector is the normalized sum of its
terms' vectors, and a query's vector is built the same way. The consequences are the ones
that make recall measurable:

  - Documents of one topic form a tight cluster, so nearest-neighbour search has a real
    neighbourhood to find and recall against brute force means something.
  - A query built from a topic's *held-out* terms — terms no document contains — still
    lands in that topic's cluster. Lexical retrieval scores zero on it and semantic
    retrieval finds the right documents, which is the comparison worth making.
  - A rare identifier has no vector at all. Semantic search cannot find it by
    construction, which is the other half of the comparison: the lexical path is not
    redundant.

Nothing is downloaded and nothing is trained. The axes are topic indices, which is a
caricature of a learned embedding space — but it is a caricature with the one property
that matters for evaluating retrieval, namely that proximity means topical relatedness.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from .rng import Rng, derive
from .vocabulary import Vocabulary

#: Extra dimensions beyond one per topic.
#:
#: Without them every term vector would be a scaled one-hot plus noise confined to the
#: topic axes, so two topics would be exactly orthogonal and separability would be an
#: artifact of the dimension count. The extra dimensions give the per-term jitter somewhere
#: to live that is not a topic axis, which makes within-cluster distances vary the way they
#: do in a real embedding space.
EXTRA_DIMENSIONS = 4

#: How far a term may sit from its topic axis, as a fraction of the axis component.
#:
#: Large enough that documents within a topic are not all the same vector — identical
#: vectors make an ANN index's recall trivially 1.0 and tell you nothing. Small enough that
#: the topic signal still dominates, which is what makes the clusters separable at all.
JITTER = 0.35


def dimension(topics: int) -> int:
    return topics + EXTRA_DIMENSIONS


def normalize(vector: Sequence[float]) -> list[float]:
    """L2 normalization, returning zeros for a zero vector rather than dividing by zero.

    The engine's cosine similarity returns 0 for a zero-length vector rather than NaN for
    the same reason: a NaN score sorts unpredictably inside the merge, which is a much
    harder failure to read than a zero.
    """
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return [0.0] * len(vector)
    return [v / norm for v in vector]


def term_weight(rank: int) -> float:
    """How much a term's vector counts in a sum, as a function of its frequency rank.

    Rarer terms count for more, which is the same intuition as IDF and is here for a
    concrete reason: without it, a document embedding is the average of a hundred vectors
    that mostly point at one topic axis, the per-term jitter averages away by a factor of
    ten, and every document in a topic lands on the same point. Identical vectors make an
    ANN index's recall trivially perfect and therefore unmeasurable — there is no
    neighbourhood structure to approximate.

    Weighting by rarity leaves the topic signal intact (the weight does not depend on the
    topic) while letting each document's distinctive terms pull it to its own place inside
    the cluster. That is the structure recall is measured against.
    """
    return math.log2(2.0 + rank)


def build_term_vectors(vocabulary: Vocabulary, seed: int) -> dict[str, list[float]]:
    """A vector per vocabulary term, including the held-out query-only terms.

    Held-out terms are included deliberately: they are the reason a semantic query can
    reach documents it shares no words with.

    The stored vectors are *not* unit length — each is scaled by its rarity weight — so that
    a plain sum in `embed_tokens` is already weighted. Keeping the weight in the vector
    rather than in the summation means the term_vectors file alone is enough to reproduce any
    document or query embedding.
    """
    dims = dimension(vocabulary.topics)
    vectors: dict[str, list[float]] = {}
    for index, term in enumerate(vocabulary.terms):
        topic = vocabulary.topic_of[index]
        # One stream per term, derived from the term's own name, so a vector does not
        # depend on how many terms were generated before it.
        rng = Rng(derive(seed, "term-vector", term))
        vector = [0.0] * dims
        vector[topic] = 1.0
        for d in range(dims):
            vector[d] += rng.uniform(-JITTER, JITTER)
        unit = normalize(vector)
        weight = term_weight(index)
        vectors[term] = [v * weight for v in unit]
    return vectors


def embed_tokens(tokens: Iterable[str], term_vectors: dict[str, list[float]], dims: int) -> list[float]:
    """The normalized sum of the known terms' vectors.

    Unknown tokens — the rare identifiers, chiefly — contribute nothing. That is the
    mechanism by which an identifier is invisible to the semantic path: it has no vector,
    so a query consisting only of an identifier embeds to zero and the semantic path has
    nothing to search with.
    """
    total = [0.0] * dims
    for token in tokens:
        vector = term_vectors.get(token)
        if vector is None:
            continue
        for d in range(dims):
            total[d] += vector[d]
    return normalize(total)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("cosine needs vectors of equal length")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def format_vector(vector: Sequence[float]) -> str:
    """Fixed six decimals, comma separated.

    Fixed precision rather than `repr` because the generated files are compared byte for
    byte in the determinism test, and a shortest-round-trip float repr varies in length
    with the value — which makes a diff of two corpora unreadable even when they agree.
    """
    return ",".join(f"{v:.6f}" for v in vector)


def parse_vector(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part]
