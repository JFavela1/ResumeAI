# Evaluated our findings
import os
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TFIDF_PATH = os.path.join(BASE_DIR, "data", "processed", "tfidf_results.csv")
BM25_PATH = os.path.join(BASE_DIR, "data", "processed", "bm25_results.csv")
SBERT_PATH = os.path.join(BASE_DIR, "data", "processed", "sbert_results.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_results():
    tfidf = pd.read_csv(TFIDF_PATH)
    bm25 = pd.read_csv(BM25_PATH)
    sbert = pd.read_csv(SBERT_PATH)

    return tfidf, bm25, sbert


def compute_correlations(tfidf, bm25, sbert):
    results = []

    models = {
        "TF-IDF": tfidf["tfidf_similarity"],
        "BM25": bm25["bm25_score"],
        "SBERT": sbert["sbert_similarity"],
    }

    for name, scores in models.items():
        micro_corr = scores.corr(tfidf["micro_score"])
        macro_corr = scores.corr(tfidf["macro_score"])

        results.append({
            "Model": name,
            "Micro Score Correlation": micro_corr,
            "Macro Score Correlation": macro_corr
        })

    return pd.DataFrame(results)


def plot_correlations(df):
    df.set_index("Model")[[
        "Micro Score Correlation",
        "Macro Score Correlation"
    ]].plot(kind="bar")

    plt.title("Model Performance Comparison")
    plt.ylabel("Correlation")
    plt.xticks(rotation=0)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "model_comparison_bar.png")
    plt.savefig(path)
    plt.show()

    print(f"Saved plot to: {path}")


def scatter_plot(df, column_name, title):
    plt.scatter(df[column_name], df["micro_score"], alpha=0.5)
    plt.xlabel(column_name)
    plt.ylabel("Micro Score")
    plt.title(title)
    plt.tight_layout()

    filename = f"{column_name}_scatter.png"
    path = os.path.join(OUTPUT_DIR, filename)

    plt.savefig(path)
    plt.show()

    print(f"Saved plot to: {path}")


def main():
    tfidf, bm25, sbert = load_results()

    print("Computing correlations...")
    comparison_df = compute_correlations(tfidf, bm25, sbert)

    print("\nModel Comparison:")
    print(comparison_df)

    comparison_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nSaved comparison table to: {comparison_path}")

    print("\nGenerating bar chart...")
    plot_correlations(comparison_df)

    print("\nGenerating scatter plots...")
    scatter_plot(tfidf, "tfidf_similarity", "TF-IDF vs Micro Score")
    scatter_plot(bm25, "bm25_score", "BM25 vs Micro Score")
    scatter_plot(sbert, "sbert_similarity", "SBERT vs Micro Score")


if __name__ == "__main__":
    main()