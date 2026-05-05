# AgentMatch — LLM-powered resume × job-description analysis agent
#
# Usage:
#   from resumeai import analyze
#   result = analyze(resume_text, job_description_text)
#
# Requirements:
#   pip install openai
#   export OPENAI_API_KEY=sk-...

import json
import logging
from functools import lru_cache

import httpx
import numpy as np
from openai import OpenAI

from .config import (
    AGENT_MODEL,
    AGENT_MAX_TOKENS,
    OPENAI_HTTP_CONNECT_TIMEOUT,
    OPENAI_HTTP_READ_TIMEOUT,
    SBERT_MODEL_NAME,
)
from .preprocess import clean_text
from .baseline_tfidf import compute_tfidf_similarity
from .baseline_bm25 import compute_bm25_pair_scores
from .skill_gap import skill_in_resume

logger = logging.getLogger(__name__)
_MAX_TOOL_ITERATIONS = 10


def _load_dotenv() -> None:
    from .config import bootstrap_env

    bootstrap_env()


# ── Lazy SBERT model (loaded once, reused across calls) ───────────────────────

@lru_cache(maxsize=1)
def _sbert_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(SBERT_MODEL_NAME)


# ── Tool implementations ───────────────────────────────────────────────────────

def _single_row_df(resume: str, job_description: str):
    import pandas as pd
    return pd.DataFrame({
        "resume_clean": [clean_text(resume)],
        "job_description_clean": [clean_text(job_description)],
    })


def _run_tfidf(resume: str, job_description: str) -> float:
    df = _single_row_df(resume, job_description)
    result = compute_tfidf_similarity(df)
    return round(float(result["tfidf_similarity"].iloc[0]), 4)


def _run_bm25(resume: str, job_description: str) -> float:
    df = _single_row_df(resume, job_description)
    result = compute_bm25_pair_scores(df)
    return round(float(result["bm25_score"].iloc[0]), 4)


def _run_sbert(resume: str, job_description: str) -> float:
    model = _sbert_model()
    embeddings = model.encode(
        [clean_text(resume), clean_text(job_description)],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return round(float(np.dot(embeddings[0], embeddings[1])), 4)


def _check_skill_match(resume: str, skills: list[str]) -> dict:
    matched = [s for s in skills if skill_in_resume(s, resume)]
    missing = [s for s in skills if not skill_in_resume(s, resume)]
    return {"matched": matched, "missing": missing}


# ── Tool schemas (passed to OpenAI) ────────────────────────────────────────────

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "compute_tfidf_similarity",
            "strict": True,
            "description": (
                "Computes TF-IDF cosine similarity between a resume and a job description. "
                "Measures keyword overlap. Returns a score between 0 and 1."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "resume": {"type": "string", "description": "Full resume text"},
                    "job_description": {"type": "string", "description": "Full job description text"},
                },
                "required": ["resume", "job_description"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_bm25_score",
            "strict": True,
            "description": (
                "Computes a BM25 keyword relevance score between a resume and a job description. "
                "Accounts for term frequency and document length. Higher is better."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "resume": {"type": "string", "description": "Full resume text"},
                    "job_description": {"type": "string", "description": "Full job description text"},
                },
                "required": ["resume", "job_description"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_sbert_similarity",
            "strict": True,
            "description": (
                "Computes semantic similarity using SBERT sentence embeddings. "
                "Captures meaning beyond exact keywords. Returns a score between -1 and 1 "
                "(typically 0.2-0.9 for real resume/JD pairs; higher is better)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "resume": {"type": "string", "description": "Full resume text"},
                    "job_description": {"type": "string", "description": "Full job description text"},
                },
                "required": ["resume", "job_description"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_skill_match",
            "strict": True,
            "description": (
                "Given a list of skills extracted from the job description, checks which ones "
                "appear explicitly in the resume. Returns matched and missing skill lists."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "resume": {"type": "string", "description": "Full resume text"},
                    "skills": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of required skills extracted from the job description. "
                            "Use short noun phrases, e.g. ['python', 'machine learning', 'sql']."
                        ),
                    },
                },
                "required": ["resume", "skills"],
                "additionalProperties": False,
            },
        },
    },
]

# ── Tool dispatcher ────────────────────────────────────────────────────────────

def _dispatch(name: str, inputs: dict) -> str:
    try:
        if name == "compute_tfidf_similarity":
            score = _run_tfidf(inputs["resume"], inputs["job_description"])
            return json.dumps({"tfidf_similarity": score})

        if name == "compute_bm25_score":
            score = _run_bm25(inputs["resume"], inputs["job_description"])
            return json.dumps({"bm25_score": score})

        if name == "compute_sbert_similarity":
            score = _run_sbert(inputs["resume"], inputs["job_description"])
            return json.dumps({"sbert_similarity": score})

        if name == "check_skill_match":
            result = _check_skill_match(inputs["resume"], inputs["skills"])
            return json.dumps(result)

        return json.dumps({"error": f"Unknown tool: {name}"})

    except Exception:
        logger.exception("Tool %s failed", name)
        return json.dumps({"error": f"Tool '{name}' failed. Please try again."})


# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM = """You are AgentMatch, an expert AI career advisor that evaluates how well \
a resume matches a job description.

## Your workflow
1. Run all three similarity tools (TF-IDF, BM25, SBERT) on the full resume and JD.
2. Read the job description carefully and extract a concise list of required/preferred \
skills (technical skills, tools, soft skills). Then call check_skill_match.
3. Weigh the evidence:
   - SBERT is the most semantically rich signal.
   - TF-IDF and BM25 confirm keyword presence.
   - Skill match shows explicit gaps.
4. Return your final answer as a single JSON object — no extra text, no markdown fences.

## Output schema (return exactly this structure)
{
  "scores": {
    "tfidf_similarity": <float>,
    "bm25_score": <float>,
    "sbert_similarity": <float>,
    "overall_fit_pct": <integer 0–100>
  },
  "skill_analysis": {
    "matched_skills": ["...", ...],
    "missing_skills": ["...", ...]
  },
  "fit_level": "<Poor | Fair | Good | Strong>",
  "recommendation": "<2–3 sentence plain-English summary with specific, actionable advice>"
}

## Fit level guide
- Strong  → overall_fit_pct >= 70
- Good    → 50-69
- Fair    → 30-49
- Poor    → < 30
"""


# ── Public API ─────────────────────────────────────────────────────────────────

def analyze(
    resume: str,
    job_description: str,
    model: str = AGENT_MODEL,
    verbose: bool = True,
    api_key: str | None = None,
) -> dict:
    """
    Analyze how well a resume matches a job description.

    Parameters
    ----------
    resume : str
        Full text of the candidate's resume.
    job_description : str
        Full text of the job posting.
    model : str
        OpenAI model to use (default: config.AGENT_MODEL).
    verbose : bool
        If True, prints each tool call and its result.
    api_key : str | None
        OpenAI API key. Falls back to OPENAI_API_KEY env var if not provided.

    Returns
    -------
    dict with keys: scores, skill_analysis, fit_level, recommendation
    """
    _load_dotenv()
    client_timeout = httpx.Timeout(
        connect=OPENAI_HTTP_CONNECT_TIMEOUT,
        read=OPENAI_HTTP_READ_TIMEOUT,
        write=120.0,
        pool=OPENAI_HTTP_CONNECT_TIMEOUT,
    )
    client = OpenAI(timeout=client_timeout, max_retries=3, **({"api_key": api_key} if api_key else {}))

    messages = [
        {
            "role": "system",
            "content": _SYSTEM,
        },
        {
            "role": "user",
            "content": (
                "Please analyze this resume against the job description.\n\n"
                f"RESUME:\n{resume}\n\n"
                f"JOB DESCRIPTION:\n{job_description}"
            ),
        }
    ]

    # Agentic tool-use loop
    for iteration in range(_MAX_TOOL_ITERATIONS):
        request = {
            "model": model,
            "max_tokens": AGENT_MAX_TOKENS,
            "tools": _TOOLS,
            "tool_choice": "required" if iteration == 0 else "auto",
            "messages": messages,
        }
        if iteration > 0:
            request["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**request)
        message = response.choices[0].message

        if not message.tool_calls:
            text = message.content or ""
            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Agent returned no JSON.\nRaw response:\n{text}") from exc

        messages.append(message.model_dump(exclude_unset=True))
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            raw_arguments = tool_call.function.arguments or "{}"
            inputs = json.loads(raw_arguments)
            if verbose:
                print(f"[tool] {name}({json.dumps(inputs)[:80]}...)")
            result = _dispatch(name, inputs)
            if verbose:
                print(f"       -> {result[:120]}")
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    raise RuntimeError(
        f"Agent exceeded {_MAX_TOOL_ITERATIONS} tool-calling iterations without a final answer."
    )


# ── CLI demo ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _SAMPLE_RESUME = """
    Jane Doe | jane@email.com | github.com/janedoe

    SKILLS
    Python (Django, FastAPI), SQL, PostgreSQL, scikit-learn, pandas,
    Git, Docker, agile methodologies, REST APIs

    EXPERIENCE
    Software Engineer — Acme Corp (2021–present)
    • Built and maintained Python/Django REST APIs serving 50k daily users
    • Optimized PostgreSQL queries, reducing p95 latency by 40%
    • Collaborated with cross-functional teams using agile/scrum

    Junior Developer — Startup XYZ (2019–2021)
    • Developed data pipelines with pandas and scikit-learn
    • Wrote unit tests, maintained CI/CD on GitHub Actions

    EDUCATION
    B.S. Computer Science, State University, 2019
    """

    _SAMPLE_JD = """
    Senior Python Engineer

    We are looking for a Senior Python Engineer to join our backend team.

    Requirements:
    - 4+ years of Python development
    - Strong experience with Django or FastAPI
    - PostgreSQL or similar relational database
    - Machine learning experience (scikit-learn, TensorFlow, or PyTorch)
    - Docker and CI/CD pipelines
    - Excellent communication and teamwork skills

    Nice to have:
    - Experience with Kubernetes
    - Familiarity with Redis or Celery
    """

    print("Running AgentMatch analysis...\n")
    result = analyze(_SAMPLE_RESUME, _SAMPLE_JD, verbose=True)
    print("\n" + "=" * 60)
    print(json.dumps(result, indent=2))
