#Implment a baseline BM25 similarity matcher
import os
import pandas as pd
from rank_bm25 import BM25Okapi


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "clean_resume_job_pairs.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "bm25_results.csv")


def load_clean_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return text.split()


def compute_bm25_pair_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    job_tokenized = df["job_description_clean"].fillna("").apply(tokenize).tolist()
    resume_tokenized = df["resume_clean"].fillna("").apply(tokenize).tolist()

    bm25 = BM25Okapi(job_tokenized)

    bm25_scores = []
    for i, query_tokens in enumerate(resume_tokenized):
        scores = bm25.get_scores(query_tokens)
        bm25_scores.append(scores[i])

    df["bm25_score"] = bm25_scores
    return df


def main():
    print("Loading cleaned dataset...")
    df = load_clean_data(INPUT_PATH)

    print(f"Dataset shape: {df.shape}")
    print("Computing BM25 scores...")

    results_df = compute_bm25_pair_scores(df)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved BM25 results to: {OUTPUT_PATH}")
    print(f"Shape: {results_df.shape}")

    print("\nPreview:")
    print(results_df[["micro_score", "macro_score", "bm25_score"]].head(10))

    print("\nBM25 score summary:")
    print(results_df["bm25_score"].describe())

    corr_micro = results_df["bm25_score"].corr(results_df["micro_score"])
    corr_macro = results_df["bm25_score"].corr(results_df["macro_score"])

    print(f"\nCorrelation with micro_score: {corr_micro:.4f}")
    print(f"Correlation with macro_score: {corr_macro:.4f}")


if __name__ == "__main__":
    main()