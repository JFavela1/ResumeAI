from resumeai.preprocess import clean_text, normalize_text
from resumeai.skill_gap import skill_in_resume


def test_clean_text_lowercases_and_collapses_whitespace():
    assert clean_text("  Python\nDeveloper\t ") == "python developer"


def test_normalize_text_canonicalizes_common_variants():
    assert normalize_text("PowerBI + MS Excel") == "power bi excel"


def test_skill_in_resume_matches_normalized_phrase():
    resume = "Built dashboards with Power BI and SQL for revenue reporting."
    assert skill_in_resume("powerbi", resume)


def test_skill_in_resume_avoids_scattered_multiword_false_positive():
    resume = "Used machine controls. Enjoyed learning new systems."
    assert not skill_in_resume("machine learning", resume)
