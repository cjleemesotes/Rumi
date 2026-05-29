"""End-to-end integration tests — the hello-world from the design doc, mocked."""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from rumi import (
    BaseModelIdea,
    HeartConfig,
    Voice,
    IIdea,
    JsonIdea,
    Dervish,
    World,
    TextIdea,
    ToolRequestIdea,
    ToolResultIdea,
    inscribe,
    tool,
)
from rumi.agent import registry as _multiton_mod
from rumi.tools.decorator import _reset_for_tests as _reset_tools


@pytest.fixture(autouse=True)
def _reset_all():
    _reset_tools()
    _multiton_mod._reset_for_tests()
    yield
    _reset_tools()
    _multiton_mod._reset_for_tests()


def _canned(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}], "usage": {"total_tokens": 1, "cost": 0.0001}}


def _canned_tool(name: str, args: str) -> dict:
    return {
        "choices": [{
            "message": {
                "content": None,
                "tool_calls": [{
                    "id": "c1", "type": "function",
                    "function": {"name": name, "arguments": args},
                }],
            }
        }]
    }


def test_public_surface():
    """Confirm all advertised exports are importable via `from rumi import ...`."""
    import rumi
    for name in [
        "Dervish", "World", "HeartConfig", "Voice",
        "tool", "inscribe",
        "IIdea", "TextIdea", "ToolRequestIdea",
        "ToolResultIdea", "JsonIdea", "BaseModelIdea",
    ]:
        assert hasattr(rumi, name), f"missing public export: {name}"


def test_hello_world_end_to_end(tmp_path: Path):
    """The hello-world script from the design doc, fully mocked."""

    @tool(["Researcher"])
    def search_web(query: str) -> str:
        """Search the web."""
        return f"results for: {query}"

    @tool(["Researcher", "Summarizer"])
    def list_papers(topic: str) -> list[str]:
        """List recent papers on topic."""
        return ["paper-a", "paper-b"]

    @inscribe
    class ResearchState(BaseModel):
        last_query: str = ""
        findings: list[str] = []

    class Researcher(Dervish):
        heart_config = HeartConfig(voices=[Voice(
            model_name="~openai/gpt-latest",
            instructions="You are a researcher.",
            context_window=16,
        )])

        def _on_heart_call_ended_impl(self, payload_in, payload_out):
            usage = payload_out.get("usage") or {}
            self.log.info(f"call cost: ${usage.get('cost')}")

    class _HelloSummarizer(Dervish):
        heart_config = HeartConfig(voices=[Voice(model_name="~openai/gpt-latest")])

    with patch("rumi.world.openai.OpenAI") as MockOpenAI:
        MockOpenAI.return_value = MagicMock()
        world = World(openrouter_api_key="sk-test", data_dir=tmp_path)
        agent = Researcher(world=world, agent_id="r-1")

        # Free-form inscribe:
        agent.inscribe("started_at", 12345.0)
        assert agent.recall("started_at") == 12345.0

        # Typed inscribe:
        state = agent.recall(ResearchState, ResearchState())
        state.last_query = "saturn"
        agent.inscribe(ResearchState, state)
        re_read = agent.recall(ResearchState)
        assert re_read.last_query == "saturn"

        # Basic whirl (mocked):
        with patch("rumi.heart.openrouter.OpenRouterHeart.call",
                   return_value=_canned("Saturn has many moons.")):
            result = agent.whirl("What are Saturn's moons?")
        assert isinstance(result, TextIdea)
        assert result.content == "Saturn has many moons."
        assert result.metadata["cost_usd"] == 0.0001

        # Structured output:
        class Summary(BaseModel):
            title: str
            bullets: list[str]

        structured_response = {
            "choices": [{
                "message": {
                    "parsed": {"title": "Findings", "bullets": ["a", "b"]},
                    "content": '{"title":"Findings","bullets":["a","b"]}',
                }
            }],
            "usage": {"total_tokens": 5, "cost": 0.0002},
        }
        with patch("rumi.heart.openrouter.OpenRouterHeart.call", return_value=structured_response):
            summary = agent.whirl("Summarize the findings.", response_model=Summary)
        assert isinstance(summary, BaseModelIdea)

        # Direct tool invocation (skip LLM):
        papers = agent.use_tool("list_papers", {"topic": "saturn"})
        assert papers == ["paper-a", "paper-b"]

        # Cross-Dervish:
        sum_mind = agent.get_agent("_HelloSummarizer", "s-1")
        assert sum_mind.agent_type.endswith("_HelloSummarizer")
        assert sum_mind.agent_id == "s-1"


def test_tool_chain_with_research_state(tmp_path: Path):
    """A more realistic flow: tool request → tool result → final answer."""

    @tool(["Investigator"])
    def lookup(topic: str) -> str:
        """Look up info on a topic."""
        return f"info-about-{topic}"

    class Investigator(Dervish):
        heart_config = HeartConfig(voices=[Voice()])

    responses = [
        _canned_tool("lookup", '{"topic":"saturn"}'),
        _canned("Based on lookup, here's the answer."),
    ]
    response_iter = iter(responses)

    with patch("rumi.world.openai.OpenAI"):
        world = World(openrouter_api_key="k", data_dir=tmp_path)

    with patch("rumi.heart.openrouter.OpenRouterHeart.call",
               side_effect=lambda *a, **kw: next(response_iter)):
        agent = Investigator(world=world, agent_id="i-1")
        result = agent.whirl("Tell me about saturn")

    assert isinstance(result, TextIdea)
    assert result.content == "Based on lookup, here's the answer."

    # 4 ideas: user msg, tool_request, tool_result, final
    assert len(agent.ideas) == 4
    assert agent.ideas[0].role == "user"
    assert isinstance(agent.ideas[1], ToolRequestIdea)
    assert isinstance(agent.ideas[2], ToolResultIdea)
    assert agent.ideas[2].content == "info-about-saturn"
    assert isinstance(agent.ideas[3], TextIdea)
