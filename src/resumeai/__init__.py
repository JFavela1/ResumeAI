"""ResumeAI public package API."""

from .config import AGENT_MODEL


def analyze(
    resume: str,
    job_description: str,
    model: str = AGENT_MODEL,
    verbose: bool = True,
    api_key: str | None = None,
) -> dict:
    """Analyze a resume/job pair using the AgentMatch LLM orchestrator."""
    from .agent import analyze as _analyze

    return _analyze(
        resume=resume,
        job_description=job_description,
        model=model,
        verbose=verbose,
        api_key=api_key,
    )

__all__ = ["analyze"]
