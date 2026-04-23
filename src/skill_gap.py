import pandas as pd

from config import CLEAN_PAIRS_WITH_SKILLS_PATH, SKILL_GAP_OUTPUT_PATH
from preprocess import normalize_text


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


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

    # 1. Exact normalized phrase match (fastest, most reliable)
    if skill_norm in resume_norm:
        return True

    skill_words = skill_norm.split()

    # 2. Single-word skill: exact token match
    if len(skill_words) == 1:
        return skill_words[0] in set(resume_norm.split())

    # 3. Multi-word skill — consecutive phrase match
    #    Requires all words to appear in order and adjacent.
    #    Prevents false positives from the same words scattered in unrelated
    #    parts of the document (e.g. "machine" + "learning" far apart).
    resume_word_list = resume_norm.split()
    for i in range(len(resume_word_list) - len(skill_words) + 1):
        if resume_word_list[i:i + len(skill_words)] == skill_words:
            return True

    # 4. Concatenated-word fallback: many resumes have words run together
    #    (e.g. "marketresearch" instead of "market research").
    skill_compact = "".join(skill_words)
    resume_compact = resume_norm.replace(" ", "")
    if skill_compact in resume_compact:
        return True

    return False


def analyze_skill_gap(row: pd.Series) -> tuple[list[str], list[str]]:
    required_skills = extract_required_skills(row)
    resume_text = str(row["resume_clean"])

    matched = [s for s in required_skills if skill_in_resume(s, resume_text)]
    missing = [s for s in required_skills if not skill_in_resume(s, resume_text)]

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


def _process_row(row: pd.Series) -> pd.Series:
    matched, missing = analyze_skill_gap(row)
    return pd.Series({
        "matched_skills": ", ".join(matched),
        "missing_skills": ", ".join(missing),
        "explanation": build_explanation(matched, missing),
        "recommendation": build_recommendation(missing),
    })


def main():
    df = load_data(CLEAN_PAIRS_WITH_SKILLS_PATH).copy()

    print(f"Processing {len(df)} rows...")
    extra_df = df.apply(_process_row, axis=1)

    results_df = pd.concat([df.reset_index(drop=True), extra_df], axis=1)
    results_df.to_csv(SKILL_GAP_OUTPUT_PATH, index=False)

    print(f"Saved skill gap results to: {SKILL_GAP_OUTPUT_PATH}")

    preview_cols = ["micro_score", "macro_score", "matched_skills", "missing_skills", "explanation", "recommendation"]
    preview_df = results_df[preview_cols]

    print("\nDetailed Preview:\n")
    for i in range(5, min(10, len(preview_df))):
        print(f"--- Example {i + 1} ---")
        print(f"Micro Score:    {preview_df.loc[i, 'micro_score']}")
        print(f"Macro Score:    {preview_df.loc[i, 'macro_score']}")
        print(f"Matched Skills: {preview_df.loc[i, 'matched_skills']}")
        print(f"Missing Skills: {preview_df.loc[i, 'missing_skills']}")
        print(f"Explanation:    {preview_df.loc[i, 'explanation']}")
        print(f"Recommendation: {preview_df.loc[i, 'recommendation']}")
        print()


if __name__ == "__main__":
    main()
