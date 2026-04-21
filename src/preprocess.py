# Preprocess the data for training and evaluation
import re
import pandas as pd


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "input.resume",
        "input.job_description",
        "output.scores.aggregated_scores.micro_scores",
        "output.scores.aggregated_scores.macro_scores",
        "output.justification",
        "output.valid_resume_and_jd",
    ]

    micro_cols = [c for c in df.columns if c.startswith("input.micro_dict.")]
    macro_cols = [c for c in df.columns if c.startswith("input.macro_dict.")]

    all_cols = base_cols + micro_cols + macro_cols
    available_cols = [c for c in all_cols if c in df.columns]

    clean_df = df[available_cols].copy()

    clean_df = clean_df.rename(columns={
        "input.resume": "resume",
        "input.job_description": "job_description",
        "output.scores.aggregated_scores.micro_scores": "micro_score",
        "output.scores.aggregated_scores.macro_scores": "macro_score",
        "output.justification": "justification",
        "output.valid_resume_and_jd": "valid_pair",
    })

    clean_df["resume_clean"] = clean_df["resume"].apply(clean_text)
    clean_df["job_description_clean"] = clean_df["job_description"].apply(clean_text)

    clean_df = clean_df.dropna(subset=[
        "resume_clean",
        "job_description_clean",
        "micro_score",
        "macro_score",
    ])

    clean_df = clean_df[
        clean_df["resume_clean"] != clean_df["job_description_clean"]
    ]

    return clean_df