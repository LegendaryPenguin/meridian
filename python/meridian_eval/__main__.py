"""Command line entry point.

    python3 -m meridian_eval generate --out corpus/
    python3 -m meridian_eval report --results results.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .corpus import GeneratorConfig, generate, with_overrides, write_collection
from .report import load_results, render_markdown
from .workloads import WorkloadConfig, build_workload, summarize, write_workload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m meridian_eval",
        description="Corpus generation and reporting for Meridian's evaluation harness.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser(
        "generate", help="write a synthetic collection with known ground truth"
    )
    generate_parser.add_argument("--out", type=Path, required=True, help="output directory")
    generate_parser.add_argument("--seed", type=int, help="generation seed")
    generate_parser.add_argument("--documents", type=int, help="number of documents")
    generate_parser.add_argument("--topics", type=int, help="number of topical clusters")
    generate_parser.add_argument("--vocabulary-size", type=int, dest="vocabulary_size")
    generate_parser.add_argument(
        "--zipf-exponent",
        type=float,
        dest="zipf_exponent",
        help="term frequency skew; around 1.0 matches English running text",
    )
    generate_parser.add_argument("--identifiers", type=int, help="number of rare identifiers")
    generate_parser.add_argument(
        "--burst-concurrency",
        type=int,
        dest="burst_concurrency",
        help="concurrent copies in the burst workload",
    )

    report_parser = subparsers.add_parser(
        "report", help="render an evaluation results file as a markdown table"
    )
    report_parser.add_argument("--results", type=Path, required=True, help="results JSON")
    report_parser.add_argument(
        "--out", type=Path, help="write to this file instead of standard output"
    )

    return parser


def _run_generate(args: argparse.Namespace) -> int:
    config = with_overrides(
        GeneratorConfig(),
        seed=args.seed,
        documents=args.documents,
        topics=args.topics,
        vocabulary_size=args.vocabulary_size,
        zipf_exponent=args.zipf_exponent,
        identifiers=args.identifiers,
    )
    collection = generate(config)
    counts = write_collection(collection, args.out)

    workload_config = WorkloadConfig()
    if args.burst_concurrency is not None:
        workload_config = WorkloadConfig(burst_concurrency=args.burst_concurrency)
    workload = build_workload(collection, workload_config)
    counts["workload.tsv"] = write_workload(workload, args.out / "workload.tsv")

    print(f"wrote {args.out}")
    for name in sorted(counts):
        print(f"  {name:24s} {counts[name]:>8d} record(s)")

    kinds = summarize(workload)
    print("  adversarial workload: " + ", ".join(f"{k}={v}" for k, v in kinds.items()))

    by_kind: dict[str, int] = {}
    for query in collection.queries:
        by_kind[query.kind] = by_kind.get(query.kind, 0) + 1
    print("  judged queries:       " + ", ".join(f"{k}={v}" for k, v in sorted(by_kind.items())))

    # Said out loud because it is the number that decides whether the quality figures mean
    # anything: these judgements are complete over the pool the generator knows about.
    judged = sum(len(q.judgements) for q in collection.queries)
    print(f"  judgements:           {judged} over {len(collection.queries)} queries")
    return 0


def _run_report(args: argparse.Namespace) -> int:
    results = load_results(args.results)
    markdown = render_markdown(results)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(markdown + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(markdown)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "generate":
        return _run_generate(args)
    if args.command == "report":
        return _run_report(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
