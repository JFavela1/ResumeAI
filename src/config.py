# Central configuration for ResumeAI
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
