import pandas as pd

from resumeai.baseline_bm25 import compute_bm25_pair_scores
from resumeai.baseline_tfidf import compute_tfidf_similarity


def _sample_pairs() -> pd.DataFrame:
    return pd.DataFrame({
        "resume_clean": [
            "python sql data analysis",
            "sales crm negotiation",
        ],
        "job_description_clean": [
            "python data engineering sql",
            "account executive crm sales",
        ],
    })


def test_tfidf_similarity_adds_score_column():
    result = compute_tfidf_similarity(_sample_pairs())

    assert "tfidf_similarity" in result.columns
    assert len(result) == 2
    assert result["tfidf_similarity"].between(0, 1).all()


def test_bm25_pair_scores_adds_score_column():
    result = compute_bm25_pair_scores(_sample_pairs())

    assert "bm25_score" in result.columns
    assert len(result) == 2
