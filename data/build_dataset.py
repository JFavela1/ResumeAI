# Build and save the cleaned dataset from Hugging Face
# Run this script once before running any baselines.
#
#   python data/build_dataset.py
#
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(CURRENT_DIR, "..", "src")
sys.path.append(SRC_PATH)

from config import CLEAN_PAIRS_WITH_SKILLS_PATH, PROCESSED_DIR
from data_loader import load_data
from preprocess import build_clean_dataframe


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Loading raw dataset from Hugging Face...")
    raw_df = load_data()

    print("Preprocessing dataset...")
    clean_df = build_clean_dataframe(raw_df)

    print("Saving cleaned dataset...")
    clean_df.to_csv(CLEAN_PAIRS_WITH_SKILLS_PATH, index=False)

    print(f"\nSaved to: {CLEAN_PAIRS_WITH_SKILLS_PATH}")
    print(f"Shape: {clean_df.shape}")

    print("\nPreview:")
    print(clean_df.head(3))


if __name__ == "__main__":
    main()
