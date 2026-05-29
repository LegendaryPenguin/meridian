"""The synthetic collection: documents, embeddings, queries and graded judgements.

Why generate rather than download. An evaluation harness needs ground truth, and ground
truth bought from a labelling effort is partial — it covers the documents somebody pooled
and nothing else, so an unjudged result is indistinguishable from a mistake. Here the
labels come from the process that wrote the documents, so the ground truth is complete by
construction and `unjudged_in_top_k` can be checked against an expected value rather than
hoped to be small.

The properties the generator controls, and the reason each one is controlled:

  - **Zipfian term frequencies**, because a uniform corpus makes every posting list the
    same length and every query the same cost. See vocabulary.py.
  - **Topical clustering**, because semantic retrieval needs a neighbourhood to find.
  - **Rare identifiers**, because a corpus where every query can be answered semantically
    hides the fact that lexical matching is not redundant.
  - **Graded judgements derived from the generative draw**, not from the parameters that
    drove it: a document's grade reflects the topic mix it actually came out with, which
    is what a human labeller would see.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Sequence

from .embeddings import (
    build_term_vectors,
    dimension,
    embed_tokens,
    format_vector,
)
from .rng import Rng, derive
from .vocabulary import Vocabulary, build_vocabulary, identifier_name

#: Purity at or above which a same-topic document is graded 3 ("highly relevant").
GRADE_HIGH_PURITY = 0.75
#: Purity at or above which a same-topic document is graded 2 ("relevant").
GRADE_MID_PURITY = 0.55


@dataclass(frozen=True)
class GeneratorConfig:
    seed: int = 20_260_101
    documents: int = 1_200
    topics: int = 6
    vocabulary_size: int = 3_600

    #: Zipf exponent. Around 1.0 is what English running text shows; the value is a knob
    #: because "how much does skew matter" is a question the harness should be able to ask.
    zipf_exponent: float = 1.05

    #: One term in this many, within each topic, is withheld from every document and used
    #: only in queries.
    held_out_stride: int = 12

    document_length_min: int = 45
    document_length_max: int = 130

    #: Range of the per-document topical mixing weight. The spread is what produces a
    #: distribution of grades instead of two buckets: a document that drew 0.9 of its
    #: tokens from its topic is a better answer than one that drew 0.4, and the grades say so.
    mixture_min: float = 0.35
    mixture_max: float = 0.95

    identifiers: int = 36
    #: Documents each identifier appears in. Small on purpose: an identifier in 50
    #: documents is not a needle.
    identifier_documents: int = 2

    #: Terms per generated query.
    query_terms: int = 3
    #: How many of the most frequent terms in a topic a lexical query may draw from. Drawing
    #: from the whole topic would often produce a query whose terms appear in no document,
    #: which is a fine query and a bad test of lexical retrieval.
    lexical_query_pool: int = 90

    def validate(self) -> None:
        if self.documents <= 0:
            raise ValueError("need at least one document")
        if self.topics <= 0:
            raise ValueError("need at least one topic")
        if self.document_length_min <= 0 or self.document_length_max < self.document_length_min:
            raise ValueError("document length range is empty")
        if not 0.0 <= self.mixture_min <= self.mixture_max <= 1.0:
            raise ValueError("mixture range must lie within [0, 1] and be ordered")
        if self.identifier_documents > self.documents:
            raise ValueError("cannot place an identifier in more documents than exist")
        if self.query_terms <= 0:
            raise ValueError("a query needs at least one term")


@dataclass
class Document:
    external_id: str
    topic: int
    title: list[str]
    body: list[str]
    tags: list[str]
    #: Fraction of the document's vocabulary tokens that belong to its own topic. The
    #: realized mixture, not the parameter that drove it — this is what the grades key off.
    purity: float
    published_at: int
    popularity: float
    embedding: list[float] = field(default_factory=list)
    identifiers: list[str] = field(default_factory=list)

    @property
    def tokens(self) -> list[str]:
        return self.title + self.body + self.tags

    @property
    def length(self) -> int:
        return len(self.tokens)


@dataclass
class GeneratedQuery:
    query_id: str
    text: str
    #: "lexical_topic", "semantic_topic" or "identifier". Carried into the manifest so a
    #: result table can be split by query kind, which is where the interesting differences
    #: between the retrieval paths show up.
    kind: str
    topic: int
    terms: list[str]
    embedding: list[float]
    judgements: list[tuple[str, int]]


@dataclass
class Collection:
    config: GeneratorConfig
    vocabulary: Vocabulary
    documents: list[Document]
    queries: list[GeneratedQuery]
    term_vectors: dict[str, list[float]]
    #: identifier token -> the document ids containing it.
    identifier_placement: dict[str, list[str]]

    @property
    def dimension(self) -> int:
        return dimension(self.config.topics)


def grade_for(same_topic: bool, purity: float, lexical_matches: int) -> int:
    """The grade a judgement gets, as a function of three generative facts.

    One rule for every query kind, so a test can recompute every grade in the qrels file
    from the recorded facts. A per-kind rule would be impossible to check that way, and an
    unverifiable ground truth is not ground truth.

    A same-topic document is relevant because it is about the topic the query is about;
    how relevant depends on how concentrated it is. An off-topic document earns a marginal
    grade only by using several of the query's words — which is the case a lexical ranker
    gets right and a human would call "related but not the answer".
    """
    if same_topic:
        if purity >= GRADE_HIGH_PURITY:
            return 3
        if purity >= GRADE_MID_PURITY:
            return 2
        return 1
    return 1 if lexical_matches >= 2 else 0


def _generate_documents(config: GeneratorConfig, vocabulary: Vocabulary) -> list[Document]:
    documents: list[Document] = []
    length_span = config.document_length_max - config.document_length_min + 1

    for index in range(config.documents):
        # Topic by position rather than by draw, so topic sizes are balanced exactly. An
        # imbalance would be realistic and would also make every per-topic number noisy for
        # a reason unrelated to retrieval.
        topic = index % config.topics
        rng = Rng(derive(config.seed, "document", index))

        mixture = rng.uniform(config.mixture_min, config.mixture_max)
        length = config.document_length_min + rng.below(length_span)

        topic_terms = vocabulary.topic_document_terms[topic]
        topic_cumulative = vocabulary.topic_cumulative[topic]

        token_indices: list[int] = []
        for _ in range(length):
            if rng.random() < mixture:
                token_indices.append(topic_terms[rng.sample_cumulative(topic_cumulative)])
            else:
                # The background draw spans every document-usable term, so a document's own
                # topic also appears here about 1/topics of the time. That is the point: real
                # documents are not pure, and a corpus where off-topic terms never occur
                # makes topic assignment recoverable from a single token.
                token_indices.append(
                    vocabulary.document_terms[rng.sample_cumulative(vocabulary.document_cumulative)]
                )

        own_topic_tokens = sum(1 for i in token_indices if vocabulary.topic_of[i] == topic)
        purity = own_topic_tokens / float(length)

        tokens = [vocabulary.term(i) for i in token_indices]
        # The title gets the document's most topical tokens, which is why field weighting
        # is worth having: a title match is better evidence than a body match, and a
        # generator that scattered tokens uniformly across fields would make the title
        # weight pure overhead.
        title_count = min(5, length)
        title = tokens[:title_count]
        body = tokens[title_count:]
        tags = [vocabulary.term(topic_terms[rng.below(min(24, len(topic_terms)))]) for _ in range(2)]

        documents.append(
            Document(
                external_id=f"doc-{index:05d}",
                topic=topic,
                title=title,
                body=body,
                tags=tags,
                purity=purity,
                # A fixed epoch and a fixed step, so freshness features are exercised and
                # the corpus does not change when it is regenerated on another day.
                published_at=1_700_000_000 + index * 3_600,
                popularity=round((index * 37 % 101) / 100.0, 3),
            )
        )
    return documents


def _place_identifiers(
    config: GeneratorConfig, documents: list[Document]
) -> dict[str, list[str]]:
    """Scatters rare identifiers into a handful of documents each.

    Chosen without replacement across the whole corpus so that a single document rarely
    holds two: an identifier that co-occurs with another is a weaker test, because matching
    either one finds the same document.
    """
    placement: dict[str, list[str]] = {}
    rng = Rng(derive(config.seed, "identifiers"))
    order = rng.shuffled(range(len(documents)))
    cursor = 0

    for i in range(config.identifiers):
        token = identifier_name(i)
        targets: list[str] = []
        for _ in range(config.identifier_documents):
            if cursor >= len(order):
                # Wrapping is better than refusing: a configuration asking for more
                # placements than documents is a usable corpus with duplicated carriers,
                # and the placement map still records the truth.
                cursor = 0
            document = documents[order[cursor]]
            cursor += 1
            document.identifiers.append(token)
            document.body.append(token)
            targets.append(document.external_id)
        placement[token] = targets
    return placement


def _lexical_match_count(document: Document, terms: Sequence[str]) -> int:
    present = set(document.tokens)
    return sum(1 for term in terms if term in present)


def _judge(
    documents: Iterable[Document], topic: int, terms: Sequence[str]
) -> list[tuple[str, int]]:
    """Grades every document the generator knows could be relevant.

    The pool is every same-topic document plus every off-topic document that uses at least
    one query term. Documents outside that pool are left unjudged rather than written out
    as explicit zeroes: the qrels file would be the size of the corpus times the query
    count, and `unjudged_in_top_k` exists precisely so that a sparse pool is visible in the
    output instead of being papered over.
    """
    judgements: list[tuple[str, int]] = []
    for document in documents:
        same_topic = document.topic == topic
        matches = _lexical_match_count(document, terms)
        if not same_topic and matches == 0:
            continue
        judgements.append(
            (document.external_id, grade_for(same_topic, document.purity, matches))
        )
    return judgements


def _generate_queries(
    config: GeneratorConfig,
    vocabulary: Vocabulary,
    documents: list[Document],
    term_vectors: dict[str, list[float]],
    identifier_placement: dict[str, list[str]],
) -> list[GeneratedQuery]:
    dims = dimension(config.topics)
    queries: list[GeneratedQuery] = []

    for topic in range(config.topics):
        # A lexical query: terms drawn from the frequent end of the topic's document
        # vocabulary, so matches certainly exist and BM25 has something to rank.
        rng = Rng(derive(config.seed, "query-lexical", topic))
        pool = vocabulary.topic_document_terms[topic][: config.lexical_query_pool]
        chosen = [vocabulary.term(i) for i in rng.shuffled(pool)[: config.query_terms]]
        queries.append(
            GeneratedQuery(
                query_id=f"lex-{topic}",
                text=" ".join(chosen),
                kind="lexical_topic",
                topic=topic,
                terms=chosen,
                embedding=embed_tokens(chosen, term_vectors, dims),
                judgements=_judge(documents, topic, chosen),
            )
        )

        # A semantic query: terms held out of every document, so the lexical path retrieves
        # nothing at all and whatever is found came from the vector space.
        rng = Rng(derive(config.seed, "query-semantic", topic))
        held_out = vocabulary.topic_query_terms[topic]
        chosen = [vocabulary.term(i) for i in rng.shuffled(held_out)[: config.query_terms]]
        queries.append(
            GeneratedQuery(
                query_id=f"sem-{topic}",
                text=" ".join(chosen),
                kind="semantic_topic",
                topic=topic,
                terms=chosen,
                embedding=embed_tokens(chosen, term_vectors, dims),
                judgements=_judge(documents, topic, chosen),
            )
        )

    # Identifier queries. The judged set is exactly the carrying documents at grade 3, and
    # the query's embedding is the zero vector because an identifier has no term vector —
    # so this is the query the semantic path cannot answer and the lexical path answers
    # exactly.
    for index, (token, carriers) in enumerate(sorted(identifier_placement.items())):
        queries.append(
            GeneratedQuery(
                query_id=f"id-{index:03d}",
                text=token,
                kind="identifier",
                topic=-1,
                terms=[token],
                embedding=embed_tokens([token], term_vectors, dims),
                judgements=[(doc_id, 3) for doc_id in carriers],
            )
        )

    return queries


def generate(config: GeneratorConfig) -> Collection:
    config.validate()
    vocabulary = build_vocabulary(
        size=config.vocabulary_size,
        topics=config.topics,
        zipf_exponent=config.zipf_exponent,
        held_out_stride=config.held_out_stride,
    )
    term_vectors = build_term_vectors(vocabulary, config.seed)

    documents = _generate_documents(config, vocabulary)
    identifier_placement = _place_identifiers(config, documents)

    dims = dimension(config.topics)
    for document in documents:
        document.embedding = embed_tokens(document.tokens, term_vectors, dims)

    queries = _generate_queries(config, vocabulary, documents, term_vectors, identifier_placement)

    return Collection(
        config=config,
        vocabulary=vocabulary,
        documents=documents,
        queries=queries,
        term_vectors=term_vectors,
        identifier_placement=identifier_placement,
    )


# ---------------------------------------------------------------------------
# Writing the collection out
# ---------------------------------------------------------------------------


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    # Explicit newline and encoding: the determinism test compares bytes, and a platform
    # default that turns "\n" into "\r\n" would make the same corpus two different files.
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for line in lines:
            handle.write(line)
            handle.write("\n")


def write_collection(collection: Collection, out_dir: Path) -> dict[str, int]:
    """Writes every file the C++ side reads, plus the facts the tests check against.

    Returns a filename -> line count map, which is what the CLI prints.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    config = collection.config
    counts: dict[str, int] = {}

    # The corpus, in the tab-separated layout tools/corpus_file.hpp reads.
    corpus_lines = [
        "# id\ttitle\tbody\ttags\tlanguage\tcategory\tpublished_at\tpopularity"
    ]
    for document in collection.documents:
        corpus_lines.append(
            "\t".join(
                [
                    document.external_id,
                    " ".join(document.title),
                    " ".join(document.body),
                    " ".join(document.tags),
                    "en",
                    f"topic-{document.topic}",
                    str(document.published_at),
                    f"{document.popularity:.3f}",
                ]
            )
        )
    _write_lines(out_dir / "corpus.tsv", corpus_lines)
    counts["corpus.tsv"] = len(collection.documents)

    _write_lines(
        out_dir / "embeddings.tsv",
        ["# external_id\tcomma-separated vector"]
        + [
            f"{document.external_id}\t{format_vector(document.embedding)}"
            for document in collection.documents
        ],
    )
    counts["embeddings.tsv"] = len(collection.documents)

    _write_lines(
        out_dir / "term_vectors.tsv",
        ["# term\tcomma-separated vector"]
        + [
            f"{term}\t{format_vector(vector)}"
            for term, vector in sorted(collection.term_vectors.items())
        ],
    )
    counts["term_vectors.tsv"] = len(collection.term_vectors)

    _write_lines(
        out_dir / "queries.tsv",
        ["# qid\ttext"] + [f"{query.query_id}\t{query.text}" for query in collection.queries],
    )
    counts["queries.tsv"] = len(collection.queries)

    # Query embeddings only for queries that have one. An identifier query embeds to the
    # zero vector, and writing that out would let the semantic path run with a vector that
    # cannot match anything — a cost with no possible benefit, reported as if the semantic
    # path had been given a fair chance.
    embedded = [q for q in collection.queries if any(v != 0.0 for v in q.embedding)]
    _write_lines(
        out_dir / "query_embeddings.tsv",
        ["# qid\tcomma-separated vector"]
        + [f"{query.query_id}\t{format_vector(query.embedding)}" for query in embedded],
    )
    counts["query_embeddings.tsv"] = len(embedded)

    qrels_lines = ["# qid iteration docid grade"]
    judgement_count = 0
    for query in collection.queries:
        for doc_id, grade in query.judgements:
            qrels_lines.append(f"{query.query_id} 0 {doc_id} {grade}")
            judgement_count += 1
    _write_lines(out_dir / "qrels.txt", qrels_lines)
    counts["qrels.txt"] = judgement_count

    # The realized generative facts, so a test can recompute every grade rather than trust
    # the generator's own output about itself.
    _write_lines(
        out_dir / "doc_facts.tsv",
        ["# external_id\ttopic\tpurity\tlength\tidentifiers"]
        + [
            "\t".join(
                [
                    d.external_id,
                    str(d.topic),
                    f"{d.purity:.6f}",
                    str(d.length),
                    ",".join(d.identifiers),
                ]
            )
            for d in collection.documents
        ],
    )
    counts["doc_facts.tsv"] = len(collection.documents)

    manifest = {
        "seed": config.seed,
        "documents": config.documents,
        "topics": config.topics,
        "vocabulary_size": config.vocabulary_size,
        "zipf_exponent": config.zipf_exponent,
        "held_out_stride": config.held_out_stride,
        "embedding_dimension": collection.dimension,
        "grade_thresholds": {
            "high_purity": GRADE_HIGH_PURITY,
            "mid_purity": GRADE_MID_PURITY,
        },
        "queries": [
            {
                "id": q.query_id,
                "kind": q.kind,
                "topic": q.topic,
                "terms": q.terms,
                "judgements": len(q.judgements),
            }
            for q in collection.queries
        ],
        "identifiers": {token: docs for token, docs in sorted(collection.identifier_placement.items())},
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    counts["manifest.json"] = 1

    return counts


def with_overrides(config: GeneratorConfig, **overrides: object) -> GeneratorConfig:
    """Applies only the overrides that were actually given, ignoring Nones from argparse."""
    supplied = {key: value for key, value in overrides.items() if value is not None}
    return replace(config, **supplied)  # type: ignore[arg-type]
