import json
from types import SimpleNamespace

import pytest

pytest.importorskip("openai")

from resumeai import agent


class _FakeToolCall:
    def __init__(self, name: str, arguments: dict):
        self.id = "call_1"
        self.function = SimpleNamespace(
            name=name,
            arguments=json.dumps(arguments),
        )

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments,
            },
        }


class _FakeMessage:
    def __init__(self, content: str | None = None, tool_calls: list[_FakeToolCall] | None = None):
        self.content = content
        self.tool_calls = tool_calls or []

    def model_dump(self, exclude_unset: bool = True) -> dict:
        message = {
            "role": "assistant",
            "content": self.content,
        }
        if self.tool_calls:
            message["tool_calls"] = [tool_call.as_dict() for tool_call in self.tool_calls]
        return message


class _FakeCompletions:
    def __init__(self, responses: list[_FakeMessage]):
        self.responses = responses
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=self.responses.pop(0))]
        )


class _FakeOpenAIClient:
    def __init__(self, responses: list[_FakeMessage]):
        self.chat = SimpleNamespace(completions=_FakeCompletions(responses))


def test_tool_schemas_use_strict_mode():
    assert all(tool["function"]["strict"] is True for tool in agent._TOOLS)


def test_dispatch_unknown_tool_returns_error():
    result = json.loads(agent._dispatch("unknown_tool", {}))

    assert result == {"error": "Unknown tool: unknown_tool"}


def test_dispatch_routes_known_tools(monkeypatch):
    monkeypatch.setattr(agent, "_run_tfidf", lambda resume, job: 0.1)
    monkeypatch.setattr(agent, "_run_bm25", lambda resume, job: 2.3)
    monkeypatch.setattr(agent, "_run_sbert", lambda resume, job: 0.8)
    monkeypatch.setattr(
        agent,
        "_check_skill_match",
        lambda resume, skills: {"matched": ["python"], "missing": ["sql"]},
    )

    inputs = {"resume": "resume", "job_description": "job"}

    assert json.loads(agent._dispatch("compute_tfidf_similarity", inputs)) == {
        "tfidf_similarity": 0.1
    }
    assert json.loads(agent._dispatch("compute_bm25_score", inputs)) == {
        "bm25_score": 2.3
    }
    assert json.loads(agent._dispatch("compute_sbert_similarity", inputs)) == {
        "sbert_similarity": 0.8
    }
    assert json.loads(agent._dispatch("check_skill_match", {"resume": "resume", "skills": ["python", "sql"]})) == {
        "matched": ["python"],
        "missing": ["sql"],
    }


def test_analyze_runs_tool_loop_without_network(monkeypatch):
    final = {
        "scores": {
            "tfidf_similarity": 0.5,
            "bm25_score": 1.2,
            "sbert_similarity": 0.8,
            "overall_fit_pct": 75,
        },
        "skill_analysis": {
            "matched_skills": ["python"],
            "missing_skills": [],
        },
        "fit_level": "Strong",
        "recommendation": "Good fit.",
    }
    fake_client = _FakeOpenAIClient([
        _FakeMessage(tool_calls=[
            _FakeToolCall(
                "compute_tfidf_similarity",
                {"resume": "resume", "job_description": "job"},
            )
        ]),
        _FakeMessage(content=json.dumps(final)),
    ])
    monkeypatch.setattr(agent, "OpenAI", lambda **kwargs: fake_client)
    monkeypatch.setattr(agent, "_dispatch", lambda name, inputs: json.dumps({"tfidf_similarity": 0.5}))

    result = agent.analyze("resume", "job", verbose=False)

    assert result == final
    calls = fake_client.chat.completions.calls
    assert calls[0]["tool_choice"] == "required"
    assert "response_format" not in calls[0]
    assert calls[1]["tool_choice"] == "auto"
    assert calls[1]["response_format"] == {"type": "json_object"}
    assert any(message["role"] == "tool" for message in calls[1]["messages"])


def test_analyze_stops_after_max_tool_iterations(monkeypatch):
    fake_client = _FakeOpenAIClient([
        _FakeMessage(tool_calls=[
            _FakeToolCall(
                "compute_tfidf_similarity",
                {"resume": "resume", "job_description": "job"},
            )
        ]),
        _FakeMessage(tool_calls=[
            _FakeToolCall(
                "compute_tfidf_similarity",
                {"resume": "resume", "job_description": "job"},
            )
        ]),
    ])
    monkeypatch.setattr(agent, "OpenAI", lambda **kwargs: fake_client)
    monkeypatch.setattr(agent, "_dispatch", lambda name, inputs: json.dumps({"tfidf_similarity": 0.5}))
    monkeypatch.setattr(agent, "_MAX_TOOL_ITERATIONS", 2)

    with pytest.raises(RuntimeError, match="tool-calling iterations"):
        agent.analyze("resume", "job", verbose=False)
