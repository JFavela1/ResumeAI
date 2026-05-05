import pytest

from resumeai.web.extract import extract_resume_text


def test_extract_txt():
    text = extract_resume_text("resume.txt", b"Hello world\nSkills: Python\n")
    assert "Hello world" in text
    assert "Python" in text


def test_extract_empty_bytes():
    with pytest.raises(ValueError, match="Empty"):
        extract_resume_text("a.txt", b"")


def test_extract_unsupported_extension():
    with pytest.raises(ValueError, match="Unsupported"):
        extract_resume_text("doc.docx", b"not really")


def test_extract_txt_empty_whitespace_only():
    with pytest.raises(ValueError, match="No text"):
        extract_resume_text("a.txt", b"   \n\n  ")
