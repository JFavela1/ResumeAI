# Central configuration for ResumeAI
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_BASE_DIR = _REPO_ROOT if (_REPO_ROOT / "pyproject.toml").exists() else Path.cwd()
BASE_DIR = os.environ.get("RESUMEAI_BASE_DIR", str(_DEFAULT_BASE_DIR))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# Input datasets
CLEAN_PAIRS_PATH = os.path.join(PROCESSED_DIR, "clean_resume_job_pairs.csv")
CLEAN_PAIRS_WITH_SKILLS_PATH = os.path.join(PROCESSED_DIR, "clean_resume_job_pairs_with_skills.csv")

# Baseline output paths
TFIDF_OUTPUT_PATH = os.path.join(PROCESSED_DIR, "tfidf_results.csv")
BM25_OUTPUT_PATH = os.path.join(PROCESSED_DIR, "bm25_results.csv")
SBERT_OUTPUT_PATH = os.path.join(PROCESSED_DIR, "sbert_results.csv")
SKILL_GAP_OUTPUT_PATH = os.path.join(PROCESSED_DIR, "skill_gap_results.csv")

# Evaluation output paths
MODEL_COMPARISON_PATH = os.path.join(PROCESSED_DIR, "model_comparison.csv")
MODEL_COMPARISON_PLOT = os.path.join(PROCESSED_DIR, "model_comparison_bar.png")

# ── Model settings ─────────────────────────────────────────────────────────────
SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SBERT_BATCH_SIZE = 32

# ── TF-IDF settings ────────────────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 5000
TFIDF_STOP_WORDS = "english"

# ── Hugging Face dataset ───────────────────────────────────────────────────────
HF_DATASET_REPO = "netsol/resume-score-details"

# ── Agent settings ─────────────────────────────────────────────────────────────
AGENT_MODEL = "gpt-4o-mini"   # model used by the AgentMatch orchestrator
AGENT_MAX_TOKENS = 4096

# OpenAI / httpx timeouts (seconds). Tool + SBERT rounds often need a longer read timeout.
OPENAI_HTTP_CONNECT_TIMEOUT = float(os.environ.get("OPENAI_HTTP_CONNECT_TIMEOUT", "30"))
OPENAI_HTTP_READ_TIMEOUT = float(os.environ.get("OPENAI_HTTP_READ_TIMEOUT", "300"))


def _normalize_openai_api_key(key: str) -> str:
    key = key.strip().strip('"').strip("'")
    if "\n" in key or "\r" in key:
        key = key.splitlines()[0].strip()
    # Accidental double-paste: "...Lfy-sk-proj-8j58..." — keep the last full key.
    proj = "sk-proj-"
    if key.startswith(proj) and key.count(proj) > 1:
        key = key[key.rfind(proj) :]
    return key


def bootstrap_env() -> None:
    """
    Load `.env` from the project root (next to pyproject.toml), then cwd.

    Uses override=True so values from these files replace stale shell exports
    (python-dotenv defaults to not overriding, which often leaves OPENAI_API_KEY=sk-...
    from an old `export` and ignores a fixed `.env`).
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    env_file = _REPO_ROOT / ".env"
    if env_file.is_file():
        load_dotenv(env_file, override=True)
    load_dotenv(override=True)

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return
    os.environ["OPENAI_API_KEY"] = _normalize_openai_api_key(key)
