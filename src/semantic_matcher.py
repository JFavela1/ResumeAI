# Build a semantic matcher using SBERT
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import (
    CLEAN_PAIRS_PATH,
    SBERT_OUTPUT_PATH,
    SBERT_MODEL_NAME,
    SBERT_BATCH_SIZE,
)


def load_clean_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def compute_sbert_similarity(
    df: pd.DataFrame,
    model_name: str = SBERT_MODEL_NAME,
) -> pd.DataFrame:
    df = df.copy()

    model = SentenceTransformer(model_name)

    resume_texts = df["resume_clean"].fillna("").tolist()
    job_texts = df["job_description_clean"].fillna("").tolist()

    print("Encoding resumes...")
    resume_embeddings = model.encode(
        resume_texts,
        batch_size=SBERT_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalize so dot product == cosine sim
    )

    print("Encoding job descriptions...")
    job_embeddings = model.encode(
        job_texts,
        batch_size=SBERT_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Vectorized element-wise dot product across all pairs at once —
    # equivalent to cosine similarity after L2 normalization above.
    similarities = (resume_embeddings * job_embeddings).sum(axis=1)

    df["sbert_similarity"] = similarities
    return df


def main():
    print("Loading cleaned dataset...")
    df = load_clean_data(CLEAN_PAIRS_PATH)

    print(f"Dataset shape: {df.shape}")
    print("Computing SBERT similarity...")

    results_df = compute_sbert_similarity(df)
    results_df.to_csv(SBERT_OUTPUT_PATH, index=False)

    print(f"\nSaved SBERT results to: {SBERT_OUTPUT_PATH}")
    print(f"Shape: {results_df.shape}")

    print("\nPreview:")
    print(results_df[["micro_score", "macro_score", "sbert_similarity"]].head(10))

    print("\nSBERT similarity summary:")
    print(results_df["sbert_similarity"].describe())

    corr_micro = results_df["sbert_similarity"].corr(results_df["micro_score"])
    corr_macro = results_df["sbert_similarity"].corr(results_df["macro_score"])

    print(f"\nCorrelation with micro_score: {corr_micro:.4f}")
    print(f"Correlation with macro_score: {corr_macro:.4f}")


if __name__ == "__main__":
    main()
