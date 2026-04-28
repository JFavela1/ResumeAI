# Run the full AgentMatch pipeline in order.
#
#   python run_pipeline.py
#
# Steps:
#   1. Build dataset  — download from HuggingFace, clean, save both CSVs
#   2. TF-IDF         — keyword cosine similarity baseline
#   3. BM25           — keyword relevance baseline
#   4. SBERT          — semantic similarity baseline
#   5. Skill gap      — matched / missing skill analysis
#   6. Evaluate       — model comparison table + plots
#
import sys
import os
import time

# Make src/ importable from the project root
SRC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, SRC_PATH)


def _header(title: str) -> None:
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def _step(n: int, label: str, fn) -> None:
    _header(f"Step {n} — {label}")
    start = time.time()
    fn()
    elapsed = time.time() - start
    print(f"\n[done in {elapsed:.1f}s]")


# ── Step functions ─────────────────────────────────────────────────────────────

def build_dataset():
    import importlib.util, types
    # Run data/build_dataset.py as a module
    spec = importlib.util.spec_from_file_location(
        "build_dataset",
        os.path.join(os.path.dirname(__file__), "data", "build_dataset.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


def run_tfidf():
    from baseline_tfidf import main
    main()


def run_bm25():
    from baseline_bm25 import main
    main()


def run_sbert():
    from semantic_matcher import main
    main()


def run_skill_gap():
    from skill_gap import main
    main()


def run_evaluate():
    from evaluate import main
    main()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("AgentMatch — full pipeline")
    print("This will download the dataset and run all analyses.")
    print("SBERT encoding may take a few minutes on first run.")

    steps = [
        (1, "Build dataset",           build_dataset),
        (2, "TF-IDF baseline",         run_tfidf),
        (3, "BM25 baseline",           run_bm25),
        (4, "SBERT semantic matcher",  run_sbert),
        (5, "Skill gap analysis",      run_skill_gap),
        (6, "Evaluate & compare",      run_evaluate),
    ]

    # Allow skipping the dataset download if CSVs already exist
    from config import CLEAN_PAIRS_PATH
    if os.path.exists(CLEAN_PAIRS_PATH):
        ans = input("\nDataset CSVs already exist. Re-download? [y/N] ").strip().lower()
        if ans != "y":
            steps = steps[1:]   # skip step 1
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
    print(f"  Results saved to: data/processed/")
    print("=" * 60)
