import os
import re
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "clean_resume_job_pairs_with_skills.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "skill_gap_results.csv")


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def normalize_text(text: str) -> str:
    text = str(text).lower()

    # Normalize common variants
    text = text.replace("powerbi", "power bi")
    text = text.replace("ms excel", "excel")
    text = text.replace("microsoft excel", "excel")
    text = text.replace("ms word", "word")
    text = text.replace("microsoft word", "word")
    text = text.replace("seo/sem", "seo sem")

    # Remove punctuation but keep letters/numbers/spaces
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_required_skills(row: pd.Series) -> list[str]:
    skills = []

    for col in row.index:
        if col.startswith("input.micro_dict."):
            value = row[col]
            if pd.notna(value) and value != 0:
                skill_name = col.replace("input.micro_dict.", "").strip().lower()
                skills.append(skill_name)

    return sorted(set(skills))


def skill_in_resume(skill: str, resume_text: str) -> bool:
    skill_norm = normalize_text(skill)
    resume_norm = normalize_text(resume_text)

    if not skill_norm:
        return False

    # Exact normalized phrase match
    if skill_norm in resume_norm:
        return True

    # Looser matching for multi-word skills
    skill_words = skill_norm.split()
    resume_words = set(resume_norm.split())

    if len(skill_words) == 1:
        return skill_words[0] in resume_words

    overlap_count = sum(1 for word in skill_words if word in resume_words)

    # Require most words in the skill phrase to appear
    if overlap_count >= max(1, len(skill_words) - 1):
        return True

    return False


def analyze_skill_gap(row: pd.Series):
    required_skills = extract_required_skills(row)
    resume_text = str(row["resume_clean"])

    matched = []
    missing = []

    for skill in required_skills:
        if skill_in_resume(skill, resume_text):
            matched.append(skill)
        else:
            missing.append(skill)

    return matched, missing


def build_explanation(matched: list[str], missing: list[str]) -> str:
    matched_text = ", ".join(matched[:5]) if matched else "no clearly matched required skills"
    missing_text = ", ".join(missing[:5]) if missing else "no major missing required skills"

    return (
        f"The resume matches the role through skills such as {matched_text}. "
        f"Missing or less visible required skills include {missing_text}."
    )


def build_recommendation(missing: list[str]) -> str:
    if not missing:
        return "The resume appears to cover the main required skills for this job."

    return (
        "Consider revising the resume to better highlight these required skills if applicable: "
        + ", ".join(missing[:5])
        + "."
    )


def main():
    df = load_data(INPUT_PATH).copy()

    matched_col = []
    missing_col = []
    explanation_col = []
    recommendation_col = []

    for _, row in df.iterrows():
        matched, missing = analyze_skill_gap(row)

        matched_col.append(", ".join(matched))
        missing_col.append(", ".join(missing))
        explanation_col.append(build_explanation(matched, missing))
        recommendation_col.append(build_recommendation(missing))

    extra_df = pd.DataFrame({
        "matched_skills": matched_col,
        "missing_skills": missing_col,
        "explanation": explanation_col,
        "recommendation": recommendation_col,
    })

    results_df = pd.concat([df.reset_index(drop=True), extra_df], axis=1)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved skill gap results to: {OUTPUT_PATH}")

    preview_df = results_df[
        ["micro_score", "macro_score", "matched_skills", "missing_skills", "explanation", "recommendation"]
    ]

    print("\nDetailed Preview:\n")
    for i in range(min(5, len(preview_df))):
        print(f"--- Example {i+1} ---")
        print(f"Micro Score: {preview_df.loc[i, 'micro_score']}")
        print(f"Macro Score: {preview_df.loc[i, 'macro_score']}")
        print(f"Matched Skills: {preview_df.loc[i, 'matched_skills']}")
        print(f"Missing Skills: {preview_df.loc[i, 'missing_skills']}")
        print(f"Explanation: {preview_df.loc[i, 'explanation']}")
        print(f"Recommendation: {preview_df.loc[i, 'recommendation']}")
        print()


if __name__ == "__main__":
    main()