"""resumeai.config helpers."""
import resumeai.config as cfg


def test_normalize_openai_api_key_strips_and_quotes():
    assert cfg._normalize_openai_api_key('  sk-proj-abc  ') == "sk-proj-abc"
    assert cfg._normalize_openai_api_key('"sk-proj-abc"') == "sk-proj-abc"


def test_normalize_openai_api_key_double_sk_proj_paste():
    bad = "sk-proj-part1truncated-Lfy-sk-proj-part1rest-realTailEND"
    assert cfg._normalize_openai_api_key(bad) == "sk-proj-part1rest-realTailEND"
