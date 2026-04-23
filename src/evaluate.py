# Evaluate and compare baseline model performance
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for scripted runs
import matplotlib.pyplot as plt

from config import (
    TFIDF_OUTPUT_PATH,
    BM25_OUTPUT_PATH,
    SBERT_OUTPUT_PATH,
    MODEL_COMPARISON_PATH,
    MODEL_COMPARISON_PLOT,
    PROCESSED_DIR,
)


def load_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tfidf = pd.read_csv(TFIDF_OUTPUT_PATH)
    bm25 = pd.read_csv(BM25_OUTPUT_PATH)
    sbert = pd.read_csv(SBERT_OUTPUT_PATH)
    return tfidf, bm25, sbert


def compute_correlations(
    tfidf: pd.DataFrame,
    bm25: pd.DataFrame,
    sbert: pd.DataFrame,
) -> pd.DataFrame:
    # Use a shared ground-truth series rather than always referencing tfidf
    micro_gt = tfidf["micro_score"]
    macro_gt = tfidf["macro_score"]

    models = {
        "TF-IDF": tfidf["tfidf_similarity"],
        "BM25": bm25["bm25_score"],
        "SBERT": sbert["sbert_similarity"],
    }

    results = []
    for name, scores in models.items():
        results.append({
            "Model": name,
            "Micro Score Correlation": scores.corr(micro_gt),
            "Macro Score Correlation": scores.corr(macro_gt),
        })

    return pd.DataFrame(results)


def plot_correlations(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots()
    df.set_index("Model")[["Micro Score Correlation", "Macro Score Correlation"]].plot(
        kind="bar", ax=ax
    )
    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Correlation")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(MODEL_COMPARISON_PLOT)
    plt.close(fig)
    print(f"Saved plot to: {MODEL_COMPARISON_PLOT}")


def scatter_plot(df: pd.DataFrame, column_name: str, title: str) -> None:
    fig, ax = plt.subplots()     # new figure each call — prevents bleed-across
    ax.scatter(df[column_name], df["micro_score"], alpha=0.5)
    ax.set_xlabel(column_name)
    ax.set_ylabel("Micro Score")
    ax.set_title(title)
    fig.tight_layout()

    filename = f"{column_name}_scatter.png"
    import os
    path = os.path.join(PROCESSED_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved plot to: {path}")


def main():
    tfidf, bm25, sbert = load_results()

    print("Computing correlations...")
    comparison_df = compute_correlations(tfidf, bm25, sbert)

    print("\nModel Comparison:")
    print(comparison_df)

    comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False)
    print(f"\nSaved comparison table to: {MODEL_COMPARISON_PATH}")

    print("\nGenerating bar chart...")
    plot_correlations(comparison_df)

    print("\nGenerating scatter plots...")
    scatter_plot(tfidf, "tfidf_similarity", "TF-IDF vs Micro Score")
    scatter_plot(bm25, "bm25_score", "BM25 vs Micro Score")
    scatter_plot(sbert, "sbert_similarity", "SBERT vs Micro Score")


if __name__ == "__main__":
    main()
