"""The Python experimentation layer for Meridian's evaluation harness.

Two jobs, and nothing else:

1. **Generate ground truth.** A synthetic collection with controlled properties — Zipfian
   term frequencies, topical clusters, rare identifiers — whose judgements come from the
   generative process, so relevance is known rather than sampled.
2. **Report.** Turn an evaluation run's results into a quality-versus-latency table that
   labels tradeoffs as tradeoffs.

Standard library only, by the same rule the engine follows: a configure-time download makes
the build depend on the network, and nothing here needs more than arithmetic and file I/O.
"""

from __future__ import annotations

from .corpus import (
    Collection,
    Document,
    GeneratedQuery,
    GeneratorConfig,
    generate,
    grade_for,
    write_collection,
)
from .embeddings import cosine, dimension, embed_tokens, normalize
from .report import Results, Row, load_results, render_markdown, verdict_for
from .rng import Rng, derive
from .vocabulary import Vocabulary, build_vocabulary, identifier_name, term_name
from .workloads import AdversarialQuery, WorkloadConfig, build_workload, write_workload

__all__ = [
    "AdversarialQuery",
    "Collection",
    "Document",
    "GeneratedQuery",
    "GeneratorConfig",
    "Results",
    "Rng",
    "Row",
    "Vocabulary",
    "WorkloadConfig",
    "build_vocabulary",
    "build_workload",
    "cosine",
    "derive",
    "dimension",
    "embed_tokens",
    "generate",
    "grade_for",
    "identifier_name",
    "load_results",
    "normalize",
    "render_markdown",
    "term_name",
    "verdict_for",
    "write_collection",
    "write_workload",
]
