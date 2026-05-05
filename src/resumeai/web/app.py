"""FastAPI web app: upload resume + paste job posting -> AgentMatch analysis."""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import APIError, APITimeoutError

from resumeai import analyze

from .extract import extract_resume_text

logger = logging.getLogger(__name__)

_WEB_ROOT = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(_WEB_ROOT / "templates"))

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_TEXT_CHARS = 50_000
MIN_JOB_CHARS = 30


@asynccontextmanager
async def _lifespan(app: FastAPI):
    from resumeai.config import bootstrap_env

    bootstrap_env()
    _warn_openai_key_setting()
    yield


def _warn_openai_key_setting() -> None:
    import os

    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        logger.warning(
            "OPENAI_API_KEY is not set. Put it in %s or export it, then restart the server.",
            "the project .env file",
        )
        return
    placeholders = {"sk-...", "sk-ant-...", "your-api-key", "changeme"}
    if key.lower() in placeholders or key == "sk-...":
        logger.warning(
            "OPENAI_API_KEY still looks like a placeholder from .env.example — "
            "replace it with a real key from https://platform.openai.com/api-keys"
        )
    if len(key) < 20:
        logger.warning(
            "OPENAI_API_KEY seems too short — OpenAI secret keys are normally longer. "
            "Check for typos, quotes, or copy-paste errors."
        )


app = FastAPI(
    title="ResumeAI",
    description="Upload a resume and paste a job posting for AgentMatch analysis.",
    lifespan=_lifespan,
)

app.mount(
    "/static",
    StaticFiles(directory=str(_WEB_ROOT / "static")),
    name="static",
)


async def _read_upload_limit(upload: UploadFile) -> bytes:
    total = 0
    chunks: list[bytes] = []
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _clip(text: str, label: str) -> str:
    text = text.strip()
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"{label} is too long after extraction (max {MAX_TEXT_CHARS} characters).",
        )
    return text


def _form_error_response(
    request: Request,
    *,
    error: str,
    job_text: str,
    resume_text: str,
) -> HTMLResponse:
    """Redisplay the form. Use 200 so the browser does not log a failed POST for validation issues."""
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "error": error,
            "job_text": job_text,
            "resume_text": resume_text,
            "min_job_chars": MIN_JOB_CHARS,
            "max_text_chars": MAX_TEXT_CHARS,
        },
        status_code=200,
    )


def _validate_api_key(raw: str) -> str:
    key = raw.strip()
    if not key:
        raise HTTPException(status_code=400, detail="Please enter your OpenAI API key.")
    if not key.startswith("sk-"):
        raise HTTPException(
            status_code=400,
            detail="That doesn't look like an OpenAI API key (should start with 'sk-').",
        )
    return key


async def _resolve_resume_text(
    resume_text: str,
    resume_file: UploadFile | None,
) -> str:
    has_file = resume_file and getattr(resume_file, "filename", None)
    raw_text = (resume_text or "").strip()

    if has_file:
        content = await _read_upload_limit(resume_file)
        if content:
            try:
                extracted = extract_resume_text(resume_file.filename, content)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return _clip(extracted, "Resume")

    if raw_text:
        return _clip(raw_text, "Resume")

    raise HTTPException(
        status_code=400,
        detail="Provide a resume file (.pdf or .txt) or paste your resume text.",
    )


def _run_analysis(resume: str, job: str, api_key: str | None = None) -> dict:
    try:
        return analyze(resume, job, verbose=False, api_key=api_key)
    except APITimeoutError as exc:
        logger.warning("OpenAI request timed out: %s", exc)
        raise HTTPException(
            status_code=504,
            detail="The analysis service timed out. Try again or increase OPENAI_HTTP_READ_TIMEOUT.",
        ) from exc
    except APIError as exc:
        logger.warning("OpenAI API error: %s", exc)
        status = getattr(exc, "status_code", None) or getattr(
            getattr(exc, "response", None), "status_code", None
        )
        err_type = None
        err_body = getattr(exc, "body", None)
        if isinstance(err_body, dict):
            err = err_body.get("error")
            if isinstance(err, dict):
                err_type = err.get("code") or err.get("type")
        if status == 401 or err_type in ("invalid_api_key", "invalid_api_key_error"):
            raise HTTPException(
                status_code=502,
                detail="OpenAI rejected the API key (401). Double-check that you entered a valid key.",
            ) from exc
        raise HTTPException(
            status_code=502,
            detail="The analysis provider returned an error. Check your API key, billing, and quota.",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected analysis error")
        raise HTTPException(
            status_code=500,
            detail="Analysis failed unexpectedly.",
        ) from exc


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "error": None,
            "job_text": "",
            "resume_text": "",
            "min_job_chars": MIN_JOB_CHARS,
            "max_text_chars": MAX_TEXT_CHARS,
        },
    )


@app.post("/analyze", response_class=HTMLResponse)
async def analyze_form(
    request: Request,
    api_key: str = Form(""),
    job_text: str = Form(...),
    resume_text: str = Form(""),
    resume_file: UploadFile | None = File(None),
) -> HTMLResponse:
    try:
        key = _validate_api_key(api_key)
    except HTTPException as exc:
        return _form_error_response(
            request,
            error=exc.detail,
            job_text=job_text,
            resume_text=resume_text,
        )

    try:
        job = _clip(job_text.strip(), "Job description")
    except HTTPException as exc:
        detail = exc.detail
        if not isinstance(detail, str):
            detail = str(detail)
        return _form_error_response(
            request,
            error=detail,
            job_text=job_text,
            resume_text=resume_text,
        )

    if len(job) < MIN_JOB_CHARS:
        return _form_error_response(
            request,
            error=(
                f"Job description is too short (min {MIN_JOB_CHARS} characters "
                f"after trimming spaces). You entered {len(job)}."
            ),
            job_text=job_text,
            resume_text=resume_text,
        )

    try:
        resume = await _resolve_resume_text(resume_text, resume_file)
    except HTTPException as exc:
        detail = exc.detail
        if not isinstance(detail, str):
            detail = str(detail)
        return _form_error_response(
            request,
            error=detail,
            job_text=job_text,
            resume_text=resume_text,
        )

    try:
        result = _run_analysis(resume, job, api_key=key)
    except HTTPException as exc:
        detail = exc.detail
        if not isinstance(detail, str):
            detail = str(detail)
        return _form_error_response(
            request,
            error=detail,
            job_text=job_text,
            resume_text=resume_text,
        )

    return templates.TemplateResponse(
        request,
        "result.html",
        {
            "result": result,
            "result_json": json.dumps(result, indent=2, ensure_ascii=False),
        },
    )


@app.post("/api/analyze")
async def api_analyze(
    api_key: str = Form(""),
    job_text: str = Form(...),
    resume_text: str = Form(""),
    resume_file: UploadFile | None = File(None),
) -> JSONResponse:
    key = _validate_api_key(api_key)
    job = _clip(job_text.strip(), "Job description")
    if len(job) < MIN_JOB_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Job description is too short (min {MIN_JOB_CHARS} characters).",
        )

    resume = await _resolve_resume_text(resume_text, resume_file)
    result = _run_analysis(resume, job, api_key=key)
    return JSONResponse(content=result)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
