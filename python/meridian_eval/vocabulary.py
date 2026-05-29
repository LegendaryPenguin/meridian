"""The vocabulary, its Zipfian frequencies, and the topic partition over it.

Three properties are built here, and each one exists because its absence makes a
retrieval benchmark too easy in a specific way.

**Zipfian term frequencies.** Real text has a few terms in most documents and most terms
in a few. That is what makes posting-list length vary by four orders of magnitude, what
makes skip lists and rarest-term-first ordering worth implementing, and what makes a
disjunctive query over a stopword expensive. A uniform vocabulary has none of that: every
posting list is the same length, every query costs the same, and every optimisation looks
like it does nothing.

**A topic partition.** Documents have to cluster somewhere for semantic retrieval to have
anything to find. The partition is by *stride* over the frequency ranks — topic t owns
ranks t, t + T, t + 2T, ... — rather than by contiguous blocks. Contiguous blocks would
give one topic all the common words and another all the rare ones, so a topic's identity
and a term's frequency would be the same variable, and no result could be attributed to
either. Striding makes every topic's term-frequency profile identical.

The stride also makes the corpus-wide frequency distribution come out Zipfian with the
exponent that was asked for, which is worth spelling out. A token is drawn from the
document's own topic with probability m and from the whole vocabulary with probability
1 - m. Because each topic holds every T-th rank, a topic's internal weights are the global
weights scaled by roughly T, and each of the T topics is chosen equally often: the topical
contribution to a term of global rank r is m * (1/T) * T * zipf(r) = m * zipf(r). Adding
the background term gives (1 - m) * zipf(r) + m * zipf(r) = zipf(r). The topical structure
is therefore invisible in the aggregate frequency curve and fully present per document,
which is exactly the situation real corpora are in.

**Terms held out of the documents.** A slice of each topic's vocabulary never appears in
any document. Queries built from those terms have zero lexical overlap with their answers,
so they separate the semantic path from the lexical one — a query every document could be
matched on by string equality proves nothing about embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

_CONSONANTS = "bcdfghjklmnprstvwz"
_VOWELS = "aeiou"
_SYLLABLES = tuple(c + v for c in _CONSONANTS for v in _VOWELS)


def term_name(index: int) -> str:
    """A stable, pronounceable, unique name for vocabulary slot `index`.

    Three fixed-width syllables, so the name is a pure function of the index and two
    different indices can never collide. Pronounceable rather than "t0017" because a
    generated corpus gets read by people while they are working out why a query behaved
    oddly, and a page of numbered tokens is unreadable.
    """
    if index < 0:
        raise ValueError("vocabulary index must be non-negative")
    base = len(_SYLLABLES)
    if index >= base**3:
        raise ValueError(f"vocabulary index {index} exceeds the {base ** 3} nameable slots")
    high, rest = divmod(index, base * base)
    mid, low = divmod(rest, base)
    return _SYLLABLES[high] + _SYLLABLES[mid] + _SYLLABLES[low]


def identifier_name(index: int) -> str:
    """A rare identifier: a single token that tokenizes as one token.

    No punctuation, deliberately. The engine's tokenizer splits on anything that is not a
    letter or a digit, so an identifier written as `mrd-00a7` would index as `mrd` plus
    `00a7` — and `mrd` would then appear in every identifier-bearing document, which is
    the opposite of rare. A single alphanumeric run is the only form that stays rare after
    tokenization.
    """
    return f"mrd{index:06x}"


@dataclass(frozen=True)
class Vocabulary:
    """Terms by descending frequency rank, partitioned into topics."""

    terms: tuple[str, ...]
    #: Zipf weight per term, parallel to `terms`.
    weights: tuple[float, ...]
    #: Cumulative weights over the whole vocabulary, for inverse-CDF sampling.
    cumulative: tuple[float, ...]
    #: topic index per term, parallel to `terms`.
    topic_of: tuple[int, ...]
    #: Per topic, the term indices that documents may use.
    topic_document_terms: tuple[tuple[int, ...], ...]
    #: Per topic, the term indices held out of documents entirely.
    topic_query_terms: tuple[tuple[int, ...], ...]
    #: Cumulative weights restricted to each topic's document terms.
    topic_cumulative: tuple[tuple[float, ...], ...]
    #: Cumulative weights over every term a document may use, for the background draw.
    document_cumulative: tuple[float, ...]
    #: The term indices that cumulative-sampling over `document_cumulative` returns.
    document_terms: tuple[int, ...]

    @property
    def topics(self) -> int:
        return len(self.topic_document_terms)

    def term(self, index: int) -> str:
        return self.terms[index]


def _cumulative(values: Sequence[float]) -> tuple[float, ...]:
    total = 0.0
    out = []
    for value in values:
        total += value
        out.append(total)
    return tuple(out)


def build_vocabulary(size: int, topics: int, zipf_exponent: float, held_out_stride: int) -> Vocabulary:
    """Builds the vocabulary, its Zipf weights and the topic partition.

    `held_out_stride` is how often a term within a topic is reserved for queries only; a
    stride of 12 holds out about 8% of each topic's terms.
    """
    if size < topics * held_out_stride:
        raise ValueError("vocabulary is too small to partition into topics with held-out terms")
    if topics <= 0:
        raise ValueError("need at least one topic")
    if held_out_stride < 2:
        raise ValueError("held_out_stride must be at least 2 or every term would be held out")

    terms = tuple(term_name(i) for i in range(size))
    weights = tuple(1.0 / ((rank + 1) ** zipf_exponent) for rank in range(size))
    topic_of = tuple(index % topics for index in range(size))

    document_terms: list[int] = []
    per_topic_document: list[list[int]] = [[] for _ in range(topics)]
    per_topic_query: list[list[int]] = [[] for _ in range(topics)]
    for index in range(size):
        topic = topic_of[index]
        position_in_topic = index // topics
        if position_in_topic % held_out_stride == held_out_stride - 1:
            per_topic_query[topic].append(index)
        else:
            per_topic_document[topic].append(index)
            document_terms.append(index)

    for topic, held_out in enumerate(per_topic_query):
        if not held_out:
            raise ValueError(f"topic {topic} has no held-out query terms")

    return Vocabulary(
        terms=terms,
        weights=weights,
        cumulative=_cumulative(weights),
        topic_of=topic_of,
        topic_document_terms=tuple(tuple(t) for t in per_topic_document),
        topic_query_terms=tuple(tuple(t) for t in per_topic_query),
        topic_cumulative=tuple(
            _cumulative([weights[i] for i in topic_terms]) for topic_terms in per_topic_document
        ),
        document_cumulative=_cumulative([weights[i] for i in document_terms]),
        document_terms=tuple(document_terms),
    )
