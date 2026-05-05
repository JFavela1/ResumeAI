"""Extract plain text from uploaded resume files."""
from __future__ import annotations

import io
import re
from pathlib import Path

_ALLOWED_EXTENSIONS = {".pdf", ".txt"}


def extract_resume_text(filename: str, content: bytes) -> str:
    """
    Extract UTF-8-ish plain text from a resume file.

    Supported: .txt, .pdf
    """
    if not content:
        raise ValueError("Empty file.")

    ext = Path(filename or "").suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type {ext or '(none)'}. Use a .pdf or .txt file."
        )

    if ext == ".txt":
        text = content.decode("utf-8", errors="replace")
    else:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        parts: list[str] = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        text = "\n".join(parts)

    text = text.strip()
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    if not text:
        raise ValueError(
            "No text could be extracted from this file. Try another PDF or paste your resume."
        )
    return text
