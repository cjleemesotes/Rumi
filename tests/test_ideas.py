"""Tests for the 5 IIdea subtypes."""
from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from rumi.ideas import (
    BaseModelIdea,
    IIdea,
    JsonIdea,
    TextIdea,
    ToolRequestIdea,
    ToolResultIdea,
)


def test_ithought_defaults():
    t = IIdea()
    assert t.role == "assistant"
    assert t.metadata == {}
    assert len(t.id) > 0
    assert t.timestamp > 0


def test_text_thought_message():
    t = TextIdea(content="hello")
    msg = t.to_openai_message()
    assert msg == {"role": "assistant", "content": "hello"}


def test_text_thought_user_role():
    t = TextIdea(role="user", content="hi")
    assert t.to_openai_message() == {"role": "user", "content": "hi"}


def test_tool_request_thought_message():
    t = ToolRequestIdea(
        content=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": '{"q":"saturn"}'},
            }
        ],
        text="",
    )
    msg = t.to_openai_message()
    assert msg["role"] == "assistant"
    assert msg["tool_calls"][0]["id"] == "call_1"


def test_tool_request_parsed_calls():
    t = ToolRequestIdea(
        content=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": '{"q":"saturn"}'},
            }
        ]
    )
    parsed = t.parsed_calls()
    assert parsed[0]["name"] == "search"
    assert parsed[0]["arguments"] == {"q": "saturn"}


def test_tool_request_bad_json_args():
    t = ToolRequestIdea(
        content=[
            {"id": "c1", "type": "function", "function": {"name": "x", "arguments": "not-json"}}
        ]
    )
    parsed = t.parsed_calls()
    assert parsed[0]["arguments"] == {}


def test_tool_result_thought_role():
    t = ToolResultIdea(tool_call_id="call_1", content="result")
    assert t.role == "tool"
    msg = t.to_openai_message()
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "call_1"
    assert msg["content"] == "result"


def test_tool_result_with_name():
    t = ToolResultIdea(tool_call_id="c1", content="ok", name="search")
    assert t.to_openai_message()["name"] == "search"


def test_json_thought():
    t = JsonIdea(content={"answer": 42})
    msg = t.to_openai_message()
    assert json.loads(msg["content"]) == {"answer": 42}


class _SampleModel(BaseModel):
    title: str
    bullets: list[str]


def test_basemodel_thought():
    m = _SampleModel(title="t", bullets=["a", "b"])
    t = BaseModelIdea(content=m)
    msg = t.to_openai_message()
    parsed = json.loads(msg["content"])
    assert parsed == {"title": "t", "bullets": ["a", "b"]}


def test_dispatch_from_dict_text():
    raw = TextIdea(content="x").to_dict()
    out = IIdea.from_dict(raw)
    assert isinstance(out, TextIdea)
    assert out.content == "x"


def test_dispatch_from_dict_tool_request():
    raw = ToolRequestIdea(content=[{"id": "1", "type": "function", "function": {"name": "n", "arguments": "{}"}}]).to_dict()
    out = IIdea.from_dict(raw)
    assert isinstance(out, ToolRequestIdea)


def test_dispatch_from_dict_tool_result():
    raw = ToolResultIdea(tool_call_id="c", content="r").to_dict()
    out = IIdea.from_dict(raw)
    assert isinstance(out, ToolResultIdea)
    assert out.role == "tool"


def test_dispatch_unknown_falls_back_to_text():
    out = IIdea.from_dict({"kind": "weird", "content": "x"})
    assert isinstance(out, TextIdea)
