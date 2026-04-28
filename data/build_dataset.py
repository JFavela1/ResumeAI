# Build and save the cleaned dataset from Hugging Face.
# Run this script once before running any baselines.
#
#   python data/build_dataset.py
#
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(CURRENT_DIR, "..", "src")
sys.path.append(SRC_PATH)

from config import CLEAN_PAIRS_PATH, CLEAN_PAIRS_WITH_SKILLS_PATH, PROCESSED_DIR
from data_loader import load_data
from preprocess import build_clean_dataframe


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Loading raw dataset from Hugging Face...")
    raw_df = load_data()

    print("Preprocessing dataset...")
    clean_df = build_clean_dataframe(raw_df)

    # Save the full version (with micro_dict skill columns) — used by skill_gap.py
    clean_df.to_csv(CLEAN_PAIRS_WITH_SKILLS_PATH, index=False)
    print(f"Saved (with skills): {CLEAN_PAIRS_WITH_SKILLS_PATH}")

    # Save the slim version (core columns only) — used by the three baselines
    slim_cols = ["resume", "job_description", "resume_clean", "job_description_clean",
                 "micro_score", "macro_score"]
    available_slim = [c for c in slim_cols if c in clean_df.columns]
    clean_df[available_slim].to_csv(CLEAN_PAIRS_PATH, index=False)
    print(f"Saved (slim):        {CLEAN_PAIRS_PATH}")

    print(f"\nShape: {clean_df.shape}")
    print("\nPreview:")
    print(clean_df[available_slim].head(3))


if __name__ == "__main__":
    main()
