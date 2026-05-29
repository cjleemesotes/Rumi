"""Base IIdea — the unit of conversation in an Dervish.

An Idea is what an Dervish has: a sentence, a tool request, a tool result,
a JSON object, a pydantic instance. Five subtypes live in `rumi.ideas.types`.
Each Idea carries a `kind` discriminator so a stored history can be re-hydrated
into the right subtype on replay.
"""
from __future__ import annotations

import itertools
import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant", "tool"]

# Monotonic counter for tiebreaking when `time.time()` returns identical values
# in quick succession (common on Windows during a tool-call burst — observed:
# 5 calls in microseconds yielding only 3 unique timestamps). Without this,
# `_replay_history` could put a ToolResultIdea before its ToolRequestIdea on
# Dervish restart, since stable sort on equal timestamps preserves the random
# alphabetical-UUID order returned by `list_keys`. OpenAI then rejects the
# next whirl() with "messages with role 'tool' must be a response to a
# preceding message with 'tool_calls'."
_seq_counter = itertools.count()


class IIdea(BaseModel):
    """A single idea in a Dervish's history.

    Subtypes specialize `content` and `to_openai_message()`.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    role: Role = "assistant"
    content: Any = None
    timestamp: float = Field(default_factory=time.time)
    # Monotonic insertion counter (per-process). Used as a sort tiebreaker
    # when timestamps collide. See `_seq_counter` above for the why.
    seq: int = Field(default_factory=lambda: next(_seq_counter))
    metadata: dict = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    def to_openai_message(self) -> dict:
        """Convert this idea to an OpenAI chat-completions message dict.

        Subtypes override for tool calls / tool results / structured content.
        Default: role + str(content).
        """
        return {"role": self.role, "content": "" if self.content is None else str(self.content)}

    def to_dict(self) -> dict:
        """Serialize to plain dict (for tablet)."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict) -> "IIdea":
        """Re-hydrate from a stored dict. Dispatches to the right subtype by `_type`."""
        from rumi.ideas.types import _dispatch_from_dict

        return _dispatch_from_dict(data)
