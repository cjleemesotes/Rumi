"""Tests for OpenRouterHeart (mocked openai SDK)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from rumi.heart.config import Voice
from rumi.heart.openrouter import OpenRouterHeart, _metadata_from
from rumi.ideas.types import (
    BaseModelIdea,
    JsonIdea,
    TextIdea,
    ToolRequestIdea,
)


@pytest.fixture
def mock_world():
    world = MagicMock()
    world.openrouter_client = MagicMock()
    return world


@pytest.fixture
def heart(mock_world):
    return OpenRouterHeart(mock_world)


# ---- call() ----


def test_call_basic_passes_through(heart, mock_world):
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {"choices": [{"message": {"content": "hi"}}]}
    mock_world.openrouter_client.chat.completions.create.return_value = mock_response

    cfg = Voice(model_name="~openai/gpt-latest")
    out = heart.call(cfg, [{"role": "user", "content": "ping"}])

    create_call = mock_world.openrouter_client.chat.completions.create
    assert create_call.call_count == 1
    kwargs = create_call.call_args.kwargs
    assert kwargs["model"] == "~openai/gpt-latest"
    assert kwargs["messages"] == [{"role": "user", "content": "ping"}]
    # No temperature/max_tokens by default
    assert "temperature" not in kwargs
    assert "max_tokens" not in kwargs
    assert "tools" not in kwargs
    assert out["choices"][0]["message"]["content"] == "hi"


def test_call_threads_temperature_and_max_tokens(heart, mock_world):
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {}
    mock_world.openrouter_client.chat.completions.create.return_value = mock_response

    cfg = Voice(temperature=0.5, max_tokens=200)
    heart.call(cfg, [])
    kwargs = mock_world.openrouter_client.chat.completions.create.call_args.kwargs
    assert kwargs["temperature"] == 0.5
    assert kwargs["max_tokens"] == 200


def test_call_threads_tools(heart, mock_world):
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {}
    mock_world.openrouter_client.chat.completions.create.return_value = mock_response

    cfg = Voice()
    tool_schema = [{"type": "function", "function": {"name": "t1", "description": "", "parameters": {}}}]
    heart.call(cfg, [], tools=tool_schema)
    kwargs = mock_world.openrouter_client.chat.completions.create.call_args.kwargs
    assert kwargs["tools"] == tool_schema


def test_call_threads_extra_body(heart, mock_world):
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {}
    mock_world.openrouter_client.chat.completions.create.return_value = mock_response

    cfg = Voice(extra_body={"provider": {"require_parameters": True}})
    heart.call(cfg, [])
    kwargs = mock_world.openrouter_client.chat.completions.create.call_args.kwargs
    assert kwargs["extra_body"] == {"provider": {"require_parameters": True}}


def test_call_response_model_uses_parse(heart, mock_world):
    class Summary(BaseModel):
        title: str

    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "choices": [{"message": {"parsed": {"title": "t"}, "content": '{"title":"t"}'}}]
    }
    mock_world.openrouter_client.chat.completions.parse.return_value = mock_response

    cfg = Voice()
    heart.call(cfg, [], response_model=Summary)

    assert mock_world.openrouter_client.chat.completions.parse.call_count == 1
    parse_kwargs = mock_world.openrouter_client.chat.completions.parse.call_args.kwargs
    assert parse_kwargs["response_format"] is Summary


def test_call_response_format_dict_passes_through(heart, mock_world):
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {}
    mock_world.openrouter_client.chat.completions.create.return_value = mock_response

    cfg = Voice(response_format={"type": "json_object"})
    heart.call(cfg, [])
    kwargs = mock_world.openrouter_client.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}


# ---- parse_response() ----


def test_parse_text_response(heart):
    response = {
        "choices": [{"message": {"content": "hello world"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
    }
    idea = heart.parse_response(response)
    assert isinstance(idea, TextIdea)
    assert idea.content == "hello world"
    assert idea.metadata["prompt_tokens"] == 10
    assert idea.metadata["completion_tokens"] == 3
    assert idea.metadata["total_tokens"] == 13


def test_parse_tool_request(heart):
    response = {
        "choices": [{
            "message": {
                "content": None,
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "search", "arguments": '{"q":"saturn"}'}}
                ],
            }
        }]
    }
    idea = heart.parse_response(response)
    assert isinstance(idea, ToolRequestIdea)
    assert idea.content[0]["id"] == "c1"


def test_parse_basemodel_response_via_parsed(heart):
    """BaseModelIdea.content must be a real pydantic instance, not the dict
    that _to_dict's model_dump() leaves behind. (The openai SDK's `.parse()`
    returns a pydantic instance under .parsed, but our normalization
    flattens it to a dict; parse_response re-validates so the contract
    holds.)
    """
    class Summary(BaseModel):
        title: str
        bullets: list[str]

    parsed = Summary(title="t", bullets=["a", "b"])
    response = {
        "choices": [{"message": {"parsed": parsed.model_dump(), "content": parsed.model_dump_json()}}]
    }
    idea = heart.parse_response(response, response_model=Summary)
    assert isinstance(idea, BaseModelIdea)
    # Content is the typed instance — attribute access works, not just subscript
    assert isinstance(idea.content, Summary)
    assert idea.content.title == "t"
    assert idea.content.bullets == ["a", "b"]


def test_parse_basemodel_fallback_from_content(heart):
    """When .parsed is missing, parse content as JSON."""
    class Summary(BaseModel):
        title: str

    response = {
        "choices": [{"message": {"content": '{"title":"hello"}'}}]
    }
    idea = heart.parse_response(response, response_model=Summary)
    assert isinstance(idea, BaseModelIdea)
    assert idea.content.title == "hello"


def test_parse_json_object_response(heart):
    """Plain json_object response (no response_model) → JsonIdea."""
    response = {
        "choices": [{"message": {"content": '{"answer": 42}'}}]
    }
    idea = heart.parse_response(response)
    assert isinstance(idea, JsonIdea)
    assert idea.content == {"answer": 42}


def test_parse_json_array_response(heart):
    response = {
        "choices": [{"message": {"content": '[1, 2, 3]'}}]
    }
    idea = heart.parse_response(response)
    assert isinstance(idea, JsonIdea)
    assert idea.content == [1, 2, 3]


def test_parse_empty_choices_returns_text(heart):
    response = {"choices": []}
    idea = heart.parse_response(response)
    assert isinstance(idea, TextIdea)
    assert idea.content == ""


def test_metadata_extracts_cost_and_provider():
    response = {
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cost": 0.0001,
        },
        "provider": "openai",
    }
    meta = _metadata_from(response)
    assert meta["cost_usd"] == 0.0001
    assert meta["resolved_upstream"] == "openai"
    assert meta["total_tokens"] == 15


def test_metadata_missing_fields_handled():
    meta = _metadata_from({})
    assert meta == {}


def test_call_rejects_extra_body_with_reserved_keys(heart):
    """Regression: Voice.extra_body must not override rumi-managed fields.

    The openai SDK merges extra_body into the request JSON with extra_body
    WINNING on collisions, so a user setting extra_body={"model": "X"} would
    silently override the voice's model_name — billed to the wrong upstream
    with no warning. Reserved keys: model, messages, tools, tool_choice,
    temperature, max_tokens, response_format, stream. extra_body is for
    OpenRouter EXTENSIONS only (provider, transforms, etc.).
    """
    voice = Voice(
        model_name="anthropic/claude-3-opus",
        extra_body={"model": "openai/gpt-4o-mini", "provider": {"order": ["X"]}},
    )
    with pytest.raises(ValueError, match="rumi-managed fields"):
        heart.call(
            voice=voice,
            messages=[{"role": "user", "content": "hi"}],
        )

    # Same hole for other reserved keys
    for reserved in ("messages", "tools", "tool_choice",
                     "temperature", "max_tokens", "response_format", "stream"):
        bad_voice = Voice(
            model_name="m",
            extra_body={reserved: "anything"},
        )
        with pytest.raises(ValueError, match=reserved):
            heart.call(
                voice=bad_voice,
                messages=[{"role": "user", "content": "hi"}],
            )

    # Extensions-only extra_body still passes through
    ok_voice = Voice(
        model_name="m",
        extra_body={"provider": {"order": ["DeepInfra"]}, "transforms": ["middle-out"]},
    )
    # We can't easily run the real client; just confirm the check doesn't raise.
    heart.world.openrouter_client.chat.completions.create.return_value = {"choices": []}
    heart.call(voice=ok_voice, messages=[{"role": "user", "content": "hi"}])


def test_call_rejects_response_model_with_tools(heart):
    """response_model + tools collide because openai's .parse() needs strict
    tool schemas, which rumi's @tool decorator doesn't generate. Should raise
    a clear ValueError before reaching the SDK."""
    class _Out(BaseModel):
        result: str

    with pytest.raises(ValueError, match="response_model \\+ tools"):
        heart.call(
            voice=Voice(model_name="~openai/gpt-latest"),
            messages=[{"role": "user", "content": "test"}],
            tools=[{"type": "function", "function": {"name": "any_tool"}}],
            response_model=_Out,
        )
