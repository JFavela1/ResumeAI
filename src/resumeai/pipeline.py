# Run the full AgentMatch pipeline in order.
import argparse
import os
import sys
import time
from collections.abc import Callable

from .config import CLEAN_PAIRS_PATH


def _header(title: str) -> None:
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def _step(n: int, label: str, fn: Callable[[], None]) -> None:
    _header(f"Step {n} — {label}")
    start = time.time()
    fn()
    elapsed = time.time() - start
    print(f"\n[done in {elapsed:.1f}s]")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full AgentMatch pipeline.")
    parser.add_argument(
        "--rebuild-data",
        action="store_true",
        help="Re-download and rebuild the cleaned dataset even if processed CSVs exist.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip dataset build/download and run analyses against existing processed CSVs.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Run non-interactively; defaults to skipping dataset rebuild when CSVs exist.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.rebuild_data and args.skip_build:
        raise SystemExit("--rebuild-data and --skip-build cannot be used together.")

    from . import baseline_bm25, baseline_tfidf, build_dataset, evaluate, semantic_matcher, skill_gap

    print("AgentMatch — full pipeline")
    print("This will download the dataset and run all analyses.")
    print("SBERT encoding may take a few minutes on first run.")

    steps: list[tuple[int, str, Callable[[], None]]] = [
        (1, "Build dataset",           build_dataset.main),
        (2, "TF-IDF baseline",         baseline_tfidf.main),
        (3, "BM25 baseline",           baseline_bm25.main),
        (4, "SBERT semantic matcher",  semantic_matcher.main),
        (5, "Skill gap analysis",      skill_gap.main),
        (6, "Evaluate & compare",      evaluate.main),
    ]

    if args.skip_build:
        steps = steps[1:]
        print("Skipping dataset download.")
    elif os.path.exists(CLEAN_PAIRS_PATH) and not args.rebuild_data:
        if args.yes:
            steps = steps[1:]
            print("Dataset CSVs already exist. Skipping dataset download.")
        else:
            ans = input("\nDataset CSVs already exist. Re-download? [y/N] ").strip().lower()
            if ans != "y":
                steps = steps[1:]
                print("Skipping dataset download.")

    total_start = time.time()
    for n, label, fn in steps:
        try:
            _step(n, label, fn)
        except Exception as exc:
            print(f"\n[ERROR] Step {n} ({label}) failed: {exc}")
            print("Fix the error above and re-run, or skip by commenting out the step.")
            sys.exit(1)

    total = time.time() - total_start
    print()
    print("=" * 60)
    print(f"  All steps complete in {total:.1f}s")
    print("  Results saved to: data/processed/")
    print("=" * 60)


if __name__ == "__main__":
    main()
