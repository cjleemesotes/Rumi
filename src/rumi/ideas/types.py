"""The five subtypes an Idea can take.

- `TextIdea` — plain text (system message, user input, assistant reply).
- `ToolRequestIdea` — an assistant message that wants to call one or more tools.
- `ToolResultIdea` — the return value of a tool, in role="tool".
- `JsonIdea` — JSON-shaped output (when response_format is json_object).
- `BaseModelIdea` — a parsed pydantic instance (via openai SDK's .parse).

Every subtype carries a `kind` discriminator so a persisted history can be
re-hydrated into the right shape via `IIdea.from_dict`.
"""
from __future__ import annotations

import json
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field

from rumi.ideas.base import IIdea

M = TypeVar("M", bound=BaseModel)


class TextIdea(IIdea):
    """Plain text content."""

    kind: Literal["text"] = "text"
    content: str = ""

    def to_openai_message(self) -> dict:
        return {"role": self.role, "content": self.content}


class ToolCall(BaseModel):
    """OpenAI tool-call shape: {id, type: "function", function: {name, arguments}}."""

    id: str
    type: Literal["function"] = "function"
    function: dict  # {"name": str, "arguments": str (JSON-encoded)}


class ToolRequestIdea(IIdea):
    """LLM-emitted tool call request. `content` is a list of ToolCall-shaped dicts."""

    kind: Literal["tool_request"] = "tool_request"
    content: list[dict] = Field(default_factory=list)
    text: str = ""  # textual content alongside tool_calls (often empty)

    def to_openai_message(self) -> dict:
        return {
            "role": "assistant",
            "content": self.text or None,
            "tool_calls": self.content,
        }

    def parsed_calls(self) -> list[dict]:
        """Return tool calls with `arguments` JSON-parsed."""
        out = []
        for call in self.content:
            fn = call.get("function", {})
            args_raw = fn.get("arguments", "{}")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except json.JSONDecodeError:
                args = {}
            out.append({
                "id": call.get("id"),
                "name": fn.get("name"),
                "arguments": args,
            })
        return out


class ToolResultIdea(IIdea):
    """Tool execution result. role is always 'tool'."""

    kind: Literal["tool_result"] = "tool_result"
    role: Literal["tool"] = "tool"
    content: str = ""
    tool_call_id: str = ""
    name: str = ""  # tool name (for logging; OpenAI accepts in tool role messages)

    def to_openai_message(self) -> dict:
        msg = {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "content": self.content,
        }
        if self.name:
            msg["name"] = self.name
        return msg


class JsonIdea(IIdea):
    """Structured JSON content (when response_format={"type": "json_object"})."""

    kind: Literal["json"] = "json"
    content: Any = None  # parsed dict / list

    def to_openai_message(self) -> dict:
        return {"role": self.role, "content": json.dumps(self.content)}


class BaseModelIdea(IIdea, Generic[M]):
    """Structured output parsed to a pydantic BaseModel via openai SDK .parse()."""

    kind: Literal["basemodel"] = "basemodel"
    content: Any = None  # pydantic instance (typed as Any for serialization simplicity)

    def to_openai_message(self) -> dict:
        if isinstance(self.content, BaseModel):
            return {"role": self.role, "content": self.content.model_dump_json()}
        return {"role": self.role, "content": json.dumps(self.content)}


# --- Dispatch table for IIdea.from_dict (used by tablet replay) ---

_TYPE_MAP: dict[str, type[IIdea]] = {
    "text": TextIdea,
    "tool_request": ToolRequestIdea,
    "tool_result": ToolResultIdea,
    "json": JsonIdea,
    "basemodel": BaseModelIdea,
}


def _dispatch_from_dict(data: dict) -> IIdea:
    """Hydrate the right subtype from a serialized dict.

    If the `kind` field is unknown, fall back to TextIdea (stripping the
    unknown kind so validation succeeds).
    """
    t = data.get("kind", "text")
    cls = _TYPE_MAP.get(t)
    if cls is None:
        data = {**data, "kind": "text"}
        cls = TextIdea
    return cls.model_validate(data)
