#Load the dataset
from huggingface_hub import snapshot_download
import os
import json
import pandas as pd
from preprocess import build_clean_dataframe

def load_data():
    repo_path = snapshot_download(
        repo_id="netsol/resume-score-details",
        repo_type="dataset"
    )

    rows = []
    for filename in os.listdir(repo_path):
        if filename.endswith(".json"):
            path = os.path.join(repo_path, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                    rows.append(obj)
            except Exception as e:
                print(f"Skipping {filename}: {e}")

    df = pd.json_normalize(rows)
    return df

if __name__ == "__main__":
    raw_df = load_data()
    clean_df = build_clean_dataframe(raw_df)

    print("Raw shape:", raw_df.shape)
    print("Clean shape:", clean_df.shape)
    print("\nColumns:")
    print(clean_df.columns.tolist())

    print("\nPreview:")
    print(clean_df[[
        "resume_clean",
        "job_description_clean",
        "micro_score",
        "macro_score"
    ]].head(3))