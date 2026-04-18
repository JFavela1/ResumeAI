import os
import sys
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(CURRENT_DIR, "..", "src")
sys.path.append(SRC_PATH)

from data_loader import load_data
from preprocess import build_clean_dataframe


def main():
    print("Loading raw dataset...")
    raw_df = load_data()

    print("Preprocessing dataset...")
    clean_df = build_clean_dataframe(raw_df)

    # Create processed folder inside data/
    processed_dir = os.path.join(CURRENT_DIR, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    output_path = os.path.join(processed_dir, "clean_resume_job_pairs.csv")

    print("Saving cleaned dataset...")
    clean_df.to_csv(output_path, index=False)

    print(f"\nSaved cleaned dataset to: {output_path}")
    print(f"Shape: {clean_df.shape}")

    print("\nPreview:")
    print(clean_df.head(3))


if __name__ == "__main__":
    main()