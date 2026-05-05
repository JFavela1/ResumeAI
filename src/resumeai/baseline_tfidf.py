# Baseline TF-IDF similarity matcher
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import (
    CLEAN_PAIRS_PATH,
    TFIDF_OUTPUT_PATH,
    TFIDF_MAX_FEATURES,
    TFIDF_STOP_WORDS,
)


def load_clean_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def compute_tfidf_similarity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    vectorizer = TfidfVectorizer(
        stop_words=TFIDF_STOP_WORDS,
        max_features=TFIDF_MAX_FEATURES,
    )

    combined_text = pd.concat(
        [df["resume_clean"], df["job_description_clean"]],
        axis=0
    ).fillna("")

    vectorizer.fit(combined_text)

    resume_vectors = vectorizer.transform(df["resume_clean"].fillna(""))
    job_vectors = vectorizer.transform(df["job_description_clean"].fillna(""))

    # Vectorized row-wise dot product on sparse matrices (equivalent to
    # cosine similarity when vectors are already L2-normalized by TF-IDF).
    # Much faster than looping with cosine_similarity() one row at a time.
    dot_products = resume_vectors.multiply(job_vectors).sum(axis=1)
    similarities = np.asarray(dot_products).flatten()

    df["tfidf_similarity"] = similarities
    return df


def main():
    print("Loading cleaned dataset...")
    df = load_clean_data(CLEAN_PAIRS_PATH)

    print(f"Dataset shape: {df.shape}")
    print("Computing TF-IDF similarity...")

    results_df = compute_tfidf_similarity(df)
    results_df.to_csv(TFIDF_OUTPUT_PATH, index=False)

    print(f"\nSaved TF-IDF results to: {TFIDF_OUTPUT_PATH}")
    print(f"Shape: {results_df.shape}")

    print("\nPreview:")
    print(results_df[["micro_score", "macro_score", "tfidf_similarity"]].head(10))

    print("\nTF-IDF similarity summary:")
    print(results_df["tfidf_similarity"].describe())

    corr_micro = results_df["tfidf_similarity"].corr(results_df["micro_score"])
    corr_macro = results_df["tfidf_similarity"].corr(results_df["macro_score"])

    print(f"\nCorrelation with micro_score: {corr_micro:.4f}")
    print(f"Correlation with macro_score: {corr_macro:.4f}")


if __name__ == "__main__":
    main()
