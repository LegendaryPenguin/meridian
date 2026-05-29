"""Reporting tests, including that the Python verdict rule agrees with the C++ one.

`verdict_for` duplicates Compare in src/eval/metrics.cpp so that a results file can be
rendered on a machine without the binary. A duplicated rule that drifts is worse than no
duplication, so the cases pinned here are the same cases tests/metrics_test.cpp pins, with
the same hand-computed inputs.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from meridian_eval.report import (  # noqa: E402
    DEFAULT_LATENCY_TOLERANCE,
    QUALITY_NOISE,
    Results,
    Row,
    load_results,
    render_markdown,
    verdict_for,
)


def _row(label: str, ndcg: float, p99_us: float, **overrides: object) -> Row:
    base = dict(
        label=label,
        ndcg=ndcg,
        precision=0.4,
        recall=0.3,
        mrr=0.8,
        unjudged=4,
        p50_us=p99_us / 2,
        p95_us=p99_us * 0.9,
        p99_us=p99_us,
        mean_candidates=200.0,
        timeouts=0,
        cache_hit_rate=0.25,
    )
    base.update(overrides)
    return Row(**base)  # type: ignore[arg-type]


class VerdictTest(unittest.TestCase):
    def test_quality_up_latency_up_is_a_tradeoff(self) -> None:
        # NDCG 0.60 -> 0.70 with p99 1000us -> 1500us: ratio 1.50, well past the 10%
        # tolerance. The same case as CompareQualityUpLatencyUpIsATradeoffNotAWin.
        verdict, explanation = verdict_for(0.60, 0.70, 1000.0, 1500.0)
        self.assertEqual(verdict, "tradeoff")
        self.assertIn("+0.1000", explanation)
        self.assertIn("1.50x", explanation)
        self.assertIn("0.6000", explanation)
        self.assertIn("0.7000", explanation)

    def test_quality_up_latency_flat_is_better(self) -> None:
        # p99 ratio 1.02, inside the tolerance.
        verdict, explanation = verdict_for(0.60, 0.70, 1000.0, 1020.0)
        self.assertEqual(verdict, "better")
        self.assertIn("1.02x", explanation)

    def test_latency_up_quality_flat_is_worse(self) -> None:
        verdict, explanation = verdict_for(0.60, 0.60, 1000.0, 1500.0)
        self.assertEqual(verdict, "worse")
        self.assertIn("no quality gain", explanation)
        self.assertIn("1.50x", explanation)

    def test_within_tolerance_is_no_change(self) -> None:
        # Delta +0.0005 is below the 0.001 noise floor; ratio 1.05 is inside the tolerance.
        verdict, explanation = verdict_for(0.60, 0.6005, 1000.0, 1050.0)
        self.assertEqual(verdict, "no-change")
        self.assertIn("+0.0005", explanation)
        self.assertIn("1.05x", explanation)

    def test_quality_down_is_worse_even_when_faster(self) -> None:
        verdict, explanation = verdict_for(0.70, 0.60, 1000.0, 500.0)
        self.assertEqual(verdict, "worse")
        self.assertIn("-0.1000", explanation)
        self.assertIn("0.50x", explanation)

    def test_same_quality_lower_latency_is_better(self) -> None:
        verdict, explanation = verdict_for(0.60, 0.60, 1000.0, 700.0)
        self.assertEqual(verdict, "better")
        self.assertIn("0.70x", explanation)

    def test_tolerance_changes_the_answer(self) -> None:
        self.assertEqual(verdict_for(0.60, 0.70, 1000.0, 1200.0, 0.50)[0], "better")
        self.assertEqual(verdict_for(0.60, 0.70, 1000.0, 1200.0, 0.10)[0], "tradeoff")

    def test_missing_latency_is_reported_rather_than_assumed_flat(self) -> None:
        verdict, explanation = verdict_for(0.60, 0.70, 0.0, 0.0)
        self.assertIn("p99 unavailable", explanation)
        self.assertEqual(verdict, "better")

    def test_noise_floor_matches_the_cpp_constant(self) -> None:
        # kQualityNoise in src/eval/metrics.cpp. A drift here makes the markdown report
        # disagree with the CLI's own verdict column on the same numbers.
        self.assertEqual(QUALITY_NOISE, 1e-3)
        self.assertEqual(DEFAULT_LATENCY_TOLERANCE, 0.10)


class MarkdownTest(unittest.TestCase):
    def _results(self) -> Results:
        return Results(
            k=10,
            baseline="candidate_k=50",
            latency_tolerance=0.10,
            queries=48,
            rows=(
                _row("candidate_k=50", 0.4693, 100.0),
                _row("candidate_k=100", 0.7654, 140.0),
                _row("candidate_k=400", 0.7654, 560.0),
            ),
        )

    def test_table_carries_quality_latency_and_the_verdict(self) -> None:
        markdown = render_markdown(self._results())
        self.assertIn("NDCG@10", markdown)
        self.assertIn("p99 us", markdown)
        self.assertIn("p99 vs base", markdown)
        self.assertIn("verdict", markdown)
        self.assertIn("candidate_k=50 (baseline)", markdown)
        # ck=100 bought 0.30 NDCG for a 1.4x p99: a tradeoff, not a win.
        self.assertIn("tradeoff", markdown)
        self.assertIn("1.40x", markdown)
        self.assertIn("5.60x", markdown)

    def test_verdicts_section_carries_both_numbers_per_row(self) -> None:
        markdown = render_markdown(self._results())
        self.assertIn("## Verdicts", markdown)
        self.assertIn("**candidate_k=100**", markdown)
        self.assertIn("NDCG +0.2961", markdown)
        self.assertIn("100.00us -> 140.00us", markdown)
        # The baseline does not compare against itself.
        self.assertNotIn("**candidate_k=50**", markdown)

    def test_tradeoff_count_is_called_out(self) -> None:
        markdown = render_markdown(self._results())
        self.assertIn("2 of 2 compared configurations are tradeoffs", markdown)

    def test_unjudged_total_is_called_out(self) -> None:
        # 3 rows at 4 unjudged each. The caveat has to be in the report, not just in the
        # column: a reader who skims the table should still learn that the scores are a
        # lower bound.
        markdown = render_markdown(self._results())
        self.assertIn("12 returned documents carried no judgement", markdown)
        self.assertIn("lower bound", markdown)

    def test_a_supplied_verdict_is_preferred_over_a_recomputed_one(self) -> None:
        # The C++ harness is the source of truth when it was the thing that ran.
        results = Results(
            k=10,
            baseline="base",
            latency_tolerance=0.10,
            rows=(
                _row("base", 0.50, 100.0),
                _row("cand", 0.90, 100.0, verdict="tradeoff", explanation="from the harness"),
            ),
        )
        markdown = render_markdown(results)
        self.assertIn("from the harness", markdown)
        self.assertIn("tradeoff", markdown)

    def test_empty_results_render_without_crashing(self) -> None:
        markdown = render_markdown(
            Results(k=10, baseline="", latency_tolerance=0.1, rows=())
        )
        self.assertIn("No rows", markdown)


class LoadResultsTest(unittest.TestCase):
    def test_reads_the_cli_json(self) -> None:
        payload = {
            "k": 10,
            "baseline": "candidate_k=50",
            "latency_tolerance": 0.1,
            "queries": 48,
            "rows": [
                {
                    "label": "candidate_k=50",
                    "ndcg": 0.4693,
                    "precision": 0.12,
                    "recall": 0.08,
                    "mrr": 0.9,
                    "unjudged": 3,
                    "p50_us": 50.0,
                    "p95_us": 90.0,
                    "p99_us": 100.0,
                    "mean_candidates": 50.0,
                    "timeouts": 0,
                    "cache_hit_rate": 0.0,
                    "verdict": "no-change",
                    "explanation": "baseline",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "results.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            results = load_results(path)

        self.assertEqual(results.k, 10)
        self.assertEqual(results.baseline, "candidate_k=50")
        self.assertEqual(len(results.rows), 1)
        self.assertAlmostEqual(results.rows[0].ndcg, 0.4693)
        self.assertEqual(results.rows[0].verdict, "no-change")

    def test_accepts_the_longer_field_names(self) -> None:
        # A results file is the thing most likely to be written by hand mid-investigation, so
        # the loader accepts the struct-field spellings too.
        payload = {
            "rows": [
                {"label": "x", "ndcg_at_k": 0.5, "precision_at_k": 0.2, "queries_timed_out": 2}
            ]
        }
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "results.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            results = load_results(path)
        self.assertAlmostEqual(results.rows[0].ndcg, 0.5)
        self.assertAlmostEqual(results.rows[0].precision, 0.2)
        self.assertEqual(results.rows[0].timeouts, 2)
        self.assertEqual(results.baseline, "x")

    def test_rejects_a_file_without_rows(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "results.json"
            path.write_text(json.dumps({"k": 10}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_results(path)


if __name__ == "__main__":
    unittest.main()
