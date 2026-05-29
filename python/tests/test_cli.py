"""CLI tests: the two documented entry points work end to end.

    python3 -m meridian_eval generate --out corpus/
    python3 -m meridian_eval report --results results.json
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from meridian_eval.__main__ import main  # noqa: E402

#: Every file the C++ side or the tests read. Named here so that dropping one is a test
#: failure rather than a confusing missing-file error during a demo.
EXPECTED_FILES = (
    "corpus.tsv",
    "doc_facts.tsv",
    "embeddings.tsv",
    "manifest.json",
    "qrels.txt",
    "queries.tsv",
    "query_embeddings.tsv",
    "term_vectors.tsv",
    "workload.tsv",
)


class GenerateCommandTest(unittest.TestCase):
    def test_generate_writes_every_expected_file(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / "corpus"
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = main(["generate", "--out", str(out), "--documents", "120", "--topics", "4"])
            self.assertEqual(code, 0)

            for name in EXPECTED_FILES:
                self.assertTrue((out / name).exists(), f"{name} was not written")

            printed = buffer.getvalue()
            self.assertIn("corpus.tsv", printed)
            self.assertIn("adversarial workload:", printed)
            self.assertIn("judgements:", printed)

    def test_generate_honours_the_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / "corpus"
            with redirect_stdout(io.StringIO()):
                main(
                    [
                        "generate",
                        "--out",
                        str(out),
                        "--seed",
                        "7",
                        "--documents",
                        "90",
                        "--topics",
                        "3",
                        "--vocabulary-size",
                        "900",
                        "--zipf-exponent",
                        "1.2",
                        "--identifiers",
                        "5",
                        "--burst-concurrency",
                        "8",
                    ]
                )
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["seed"], 7)
            self.assertEqual(manifest["documents"], 90)
            self.assertEqual(manifest["topics"], 3)
            self.assertEqual(manifest["vocabulary_size"], 900)
            self.assertAlmostEqual(manifest["zipf_exponent"], 1.2)
            self.assertEqual(len(manifest["identifiers"]), 5)

            bursts = [
                line.split("\t")
                for line in (out / "workload.tsv").read_text(encoding="utf-8").splitlines()
                if line.startswith("adv-burst")
            ]
            self.assertTrue(bursts)
            self.assertTrue(all(fields[2] == "8" for fields in bursts))

    def test_generate_output_is_loadable_by_the_cpp_formats(self) -> None:
        # The qrels file has to be parseable by LoadJudgements, so the column count and the
        # integer grade are checked here as well as in test_generate: this is the path a demo
        # actually takes, and a format break would surface as an unhelpful parse error.
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / "corpus"
            with redirect_stdout(io.StringIO()):
                main(["generate", "--out", str(out), "--documents", "120", "--topics", "4"])

            query_ids = {
                line.split("\t")[0]
                for line in (out / "queries.tsv").read_text(encoding="utf-8").splitlines()
                if not line.startswith("#")
            }
            self.assertTrue(query_ids)
            for line in (out / "qrels.txt").read_text(encoding="utf-8").splitlines():
                if line.startswith("#"):
                    continue
                fields = line.split()
                self.assertEqual(len(fields), 4)
                self.assertIn(fields[0], query_ids)
                self.assertGreaterEqual(int(fields[3]), 0)


class ReportCommandTest(unittest.TestCase):
    PAYLOAD = {
        "k": 10,
        "baseline": "candidate_k=50",
        "latency_tolerance": 0.10,
        "rows": [
            {
                "label": "candidate_k=50",
                "ndcg": 0.60,
                "precision": 0.4,
                "recall": 0.3,
                "mrr": 0.9,
                "unjudged": 0,
                "p50_us": 50.0,
                "p95_us": 90.0,
                "p99_us": 100.0,
                "mean_candidates": 14.0,
                "timeouts": 0,
                "cache_hit_rate": 0.0,
            },
            {
                "label": "candidate_k=400",
                "ndcg": 0.70,
                "precision": 0.46,
                "recall": 0.34,
                "mrr": 0.92,
                "unjudged": 7,
                "p50_us": 70.0,
                "p95_us": 140.0,
                "p99_us": 150.0,
                "mean_candidates": 101.0,
                "timeouts": 0,
                "cache_hit_rate": 0.0,
            },
        ],
    }

    def test_report_prints_the_table_with_the_tradeoff_labelled(self) -> None:
        # NDCG 0.60 -> 0.70 with p99 100us -> 150us: a 1.50x tail regression bought a real
        # quality gain. The report must not call that an improvement.
        with tempfile.TemporaryDirectory() as root:
            results = Path(root) / "results.json"
            results.write_text(json.dumps(self.PAYLOAD), encoding="utf-8")
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = main(["report", "--results", str(results)])
            self.assertEqual(code, 0)
            printed = buffer.getvalue()
            self.assertIn("tradeoff", printed)
            self.assertIn("1.50x", printed)
            self.assertIn("+0.1000", printed)
            self.assertNotIn("| better", printed)

    def test_report_writes_to_a_file_when_asked(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            results = Path(root) / "results.json"
            results.write_text(json.dumps(self.PAYLOAD), encoding="utf-8")
            destination = Path(root) / "nested" / "report.md"
            with redirect_stdout(io.StringIO()):
                code = main(["report", "--results", str(results), "--out", str(destination)])
            self.assertEqual(code, 0)
            self.assertTrue(destination.exists())
            self.assertIn("# Evaluation report", destination.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
