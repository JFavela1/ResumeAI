import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "clean_resume_job_pairs.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "tfidf_results.csv")


def load_clean_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def compute_tfidf_similarity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    combined_text = pd.concat(
        [df["resume_clean"], df["job_description_clean"]],
        axis=0
    ).fillna("")

    vectorizer.fit(combined_text)

    resume_vectors = vectorizer.transform(df["resume_clean"].fillna(""))
    job_vectors = vectorizer.transform(df["job_description_clean"].fillna(""))

    similarities = []
    for i in range(len(df)):
        sim = cosine_similarity(resume_vectors[i], job_vectors[i])[0][0]
        similarities.append(sim)

    df["tfidf_similarity"] = similarities
    return df


def main():
    print("Loading cleaned dataset...")
    df = load_clean_data(INPUT_PATH)

    print(f"Dataset shape: {df.shape}")
    print("Computing TF-IDF similarity...")

    results_df = compute_tfidf_similarity(df)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved TF-IDF results to: {OUTPUT_PATH}")
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