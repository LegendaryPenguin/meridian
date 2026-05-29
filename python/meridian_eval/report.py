"""Markdown reporting: the quality-versus-latency table, without a plotting dependency.

No chart, and not reluctantly. A quality-versus-latency scatter is the obvious thing to
draw and it is worse than the table for the decision this harness exists to support: a
reader picks the point nearest the top-left corner, which is a judgement about the aspect
ratio of the axes rather than about the latency budget. The table puts both numbers and the
verdict on one line, so "the NDCG went up 0.03 and the p99 went up 80%" is a sentence
rather than a distance on a plot.

It also keeps the repository dependency-free, which is the same reason the engine has no
third-party libraries and the C++ tests have no framework.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

#: Quality movement below this is noise.
#:
#: Must match kQualityNoise in src/eval/metrics.cpp — the C++ harness is the source of
#: truth for a verdict, and this module only computes one when the input did not carry it.
QUALITY_NOISE = 1e-3

#: Default fractional p99 increase treated as material. Matches the default for
#: `latency_tolerance` on Compare in include/meridian/eval.hpp.
DEFAULT_LATENCY_TOLERANCE = 0.10


@dataclass(frozen=True)
class Row:
    label: str
    ndcg: float
    precision: float
    recall: float
    mrr: float
    unjudged: int
    p50_us: float
    p95_us: float
    p99_us: float
    mean_candidates: float
    timeouts: int
    cache_hit_rate: float
    verdict: str = ""
    explanation: str = ""


@dataclass(frozen=True)
class Results:
    k: int
    baseline: str
    latency_tolerance: float
    rows: tuple[Row, ...]
    queries: int = 0

    @property
    def baseline_row(self) -> Row | None:
        for row in self.rows:
            if row.label == self.baseline:
                return row
        return self.rows[0] if self.rows else None


def verdict_for(
    baseline_ndcg: float,
    candidate_ndcg: float,
    baseline_p99_us: float,
    candidate_p99_us: float,
    latency_tolerance: float = DEFAULT_LATENCY_TOLERANCE,
) -> tuple[str, str]:
    """Mirrors Compare in src/eval/metrics.cpp. Returns (verdict, explanation).

    Duplicated rather than shelled out to: the reporting module has to work on a results
    file produced on another machine, possibly without the binary. The rule is small and
    the duplication is named here so that a change to one side is a change that has to be
    made on both — src/eval/metrics.cpp, function Compare, is the source of truth.
    """
    delta = candidate_ndcg - baseline_ndcg
    have_latency = baseline_p99_us > 0.0 and candidate_p99_us > 0.0
    ratio = candidate_p99_us / baseline_p99_us if baseline_p99_us > 0.0 else 1.0

    quality_up = delta > QUALITY_NOISE
    quality_down = delta < -QUALITY_NOISE
    latency_worse = have_latency and ratio > 1.0 + latency_tolerance
    latency_better = have_latency and ratio < 1.0 - latency_tolerance

    quality_text = f"NDCG {delta:+.4f} ({baseline_ndcg:.4f} -> {candidate_ndcg:.4f})"
    latency_text = (
        f"p99 {ratio:.2f}x ({baseline_p99_us:.2f}us -> {candidate_p99_us:.2f}us, "
        f"tolerance {latency_tolerance * 100:.0f}%)"
        if have_latency
        else "p99 unavailable (no latency samples on one side)"
    )

    if quality_up and latency_worse:
        return "tradeoff", f"quality improved and p99 regressed — {quality_text} but {latency_text}"
    if quality_up:
        return "better", f"{quality_text}, {latency_text}"
    if quality_down:
        return "worse", f"quality regressed — {quality_text}, {latency_text}"
    if latency_worse:
        return "worse", f"p99 regressed with no quality gain — {latency_text}, {quality_text}"
    if latency_better:
        return "better", f"same quality, lower p99 — {latency_text}, {quality_text}"
    return "no-change", f"{quality_text}, {latency_text}"


def _as_float(mapping: dict[str, Any], *names: str, default: float = 0.0) -> float:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return float(mapping[name])
    return default


def _as_int(mapping: dict[str, Any], *names: str, default: int = 0) -> int:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return int(mapping[name])
    return default


def load_results(path: Path) -> Results:
    """Reads the JSON the C++ CLI writes.

    Field names are accepted in both the short form the CLI emits and a longer form, because
    a results file is the thing most likely to be produced by hand while someone is
    investigating, and refusing a hand-written file over a key name is not a useful kind of
    strictness.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "rows" not in payload:
        raise ValueError(f"{path}: expected an object with a 'rows' array")

    rows: list[Row] = []
    for entry in payload["rows"]:
        rows.append(
            Row(
                label=str(entry.get("label", "?")),
                ndcg=_as_float(entry, "ndcg", "ndcg_at_k"),
                precision=_as_float(entry, "precision", "precision_at_k"),
                recall=_as_float(entry, "recall", "recall_at_k"),
                mrr=_as_float(entry, "mrr"),
                unjudged=_as_int(entry, "unjudged", "unjudged_in_top_k"),
                p50_us=_as_float(entry, "p50_us"),
                p95_us=_as_float(entry, "p95_us"),
                p99_us=_as_float(entry, "p99_us"),
                mean_candidates=_as_float(entry, "mean_candidates"),
                timeouts=_as_int(entry, "timeouts", "queries_timed_out"),
                cache_hit_rate=_as_float(entry, "cache_hit_rate"),
                verdict=str(entry.get("verdict", "")),
                explanation=str(entry.get("explanation", "")),
            )
        )

    baseline = str(payload.get("baseline", rows[0].label if rows else ""))
    return Results(
        k=_as_int(payload, "k", default=10),
        baseline=baseline,
        latency_tolerance=_as_float(
            payload, "latency_tolerance", default=DEFAULT_LATENCY_TOLERANCE
        ),
        rows=tuple(rows),
        queries=_as_int(payload, "queries"),
    )


def _markdown_table(headers: Sequence[str], alignments: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def render(cells: Sequence[str]) -> str:
        padded = []
        for i, cell in enumerate(cells):
            padded.append(cell.rjust(widths[i]) if alignments[i] == "right" else cell.ljust(widths[i]))
        return "| " + " | ".join(padded) + " |"

    separators = []
    for i, alignment in enumerate(alignments):
        bar = "-" * widths[i]
        separators.append(bar[:-1] + ":" if alignment == "right" and widths[i] > 0 else bar)

    lines = [render(headers), "| " + " | ".join(separators) + " |"]
    lines.extend(render(row) for row in rows)
    return "\n".join(lines)


def render_markdown(results: Results) -> str:
    """The quality-versus-latency table plus the per-row verdict explanations."""
    baseline = results.baseline_row
    if baseline is None:
        return "# Evaluation report\n\nNo rows.\n"

    headers = [
        "configuration",
        f"NDCG@{results.k}",
        f"P@{results.k}",
        f"R@{results.k}",
        "MRR",
        "unjudged",
        "p50 us",
        "p95 us",
        "p99 us",
        "p99 vs base",
        "cands",
        "cache hit",
        "timeouts",
        "verdict",
    ]
    alignments = ["left"] + ["right"] * 12 + ["left"]

    table_rows: list[list[str]] = []
    verdict_lines: list[str] = []
    for row in results.rows:
        is_baseline = row.label == baseline.label
        verdict = row.verdict
        explanation = row.explanation
        if not verdict:
            verdict, explanation = verdict_for(
                baseline.ndcg, row.ndcg, baseline.p99_us, row.p99_us, results.latency_tolerance
            )
        ratio = row.p99_us / baseline.p99_us if baseline.p99_us > 0 else 0.0

        table_rows.append(
            [
                row.label + (" (baseline)" if is_baseline else ""),
                f"{row.ndcg:.4f}",
                f"{row.precision:.4f}",
                f"{row.recall:.4f}",
                f"{row.mrr:.4f}",
                str(row.unjudged),
                f"{row.p50_us:.2f}",
                f"{row.p95_us:.2f}",
                f"{row.p99_us:.2f}",
                f"{ratio:.2f}x",
                f"{row.mean_candidates:.1f}",
                f"{row.cache_hit_rate:.2f}",
                str(row.timeouts),
                "-" if is_baseline else verdict,
            ]
        )
        if not is_baseline:
            verdict_lines.append(f"- **{row.label}** — {verdict}: {explanation}")

    tradeoffs = sum(1 for line in verdict_lines if "tradeoff" in line)
    parts = [
        "# Evaluation report",
        "",
        f"Baseline: `{baseline.label}`. Cutoff k = {results.k}. "
        f"Latency tolerance: {results.latency_tolerance * 100:.0f}% on p99.",
        "",
        "Quality and latency are separate axes. A row with a higher NDCG and a worse p99 is "
        "labelled a tradeoff rather than an improvement, because whether it is one depends "
        "on a latency budget this table does not know.",
        "",
        _markdown_table(headers, alignments, table_rows),
        "",
    ]

    if verdict_lines:
        parts.extend(["## Verdicts", ""] + verdict_lines + [""])

    if tradeoffs:
        parts.extend(
            [
                f"{tradeoffs} of {len(verdict_lines)} compared configurations are tradeoffs: "
                "they improve quality and regress p99. Picking one is a decision about the "
                "latency budget, not a reading of this table.",
                "",
            ]
        )

    unjudged_total = sum(row.unjudged for row in results.rows)
    if unjudged_total:
        parts.extend(
            [
                f"Across all rows, {unjudged_total} returned documents carried no judgement. "
                "Unjudged results score as irrelevant, so the quality numbers above are a "
                "lower bound wherever that count is large relative to k times the query count.",
                "",
            ]
        )

    return "\n".join(parts)
