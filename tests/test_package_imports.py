import resumeai
from resumeai import pipeline


def test_public_analyze_api_is_exposed():
    assert callable(resumeai.analyze)


def test_pipeline_main_is_importable():
    assert callable(pipeline.main)
