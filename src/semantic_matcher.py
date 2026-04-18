import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "clean_resume_job_pairs.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "sbert_results.csv")


def load_clean_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def compute_sbert_similarity(
    df: pd.DataFrame,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> pd.DataFrame:
    df = df.copy()

    model = SentenceTransformer(model_name)

    resume_texts = df["resume_clean"].fillna("").tolist()
    job_texts = df["job_description_clean"].fillna("").tolist()

    print("Encoding resumes...")
    resume_embeddings = model.encode(
        resume_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print("Encoding job descriptions...")
    job_embeddings = model.encode(
        job_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    similarities = []
    for i in range(len(df)):
        sim = cosine_similarity(
            resume_embeddings[i].reshape(1, -1),
            job_embeddings[i].reshape(1, -1)
        )[0][0]
        similarities.append(sim)

    df["sbert_similarity"] = similarities
    return df


def main():
    print("Loading cleaned dataset...")
    df = load_clean_data(INPUT_PATH)

    print(f"Dataset shape: {df.shape}")
    print("Computing SBERT similarity...")

    results_df = compute_sbert_similarity(df)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved SBERT results to: {OUTPUT_PATH}")
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