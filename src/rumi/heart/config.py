"""HeartConfig + Voice — what the Heart will say, in how many steps.

A `Voice` configures one LLM call: model, instructions, temperature, the
context window of recent ideas to send, and any OpenRouter extension fields.
A `HeartConfig` is an ordered list of Voices — for multi-step pipelines where
each voice's output seeds the next. Most Dervishes need just one Voice.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Voice(BaseModel):
    """One step of a heart pipeline.

    Field constraints catch the typos that would otherwise burn a partially-
    persisted LLM call before failing at the provider: `model_name` must be
    non-empty / non-whitespace, `instructions` may be empty (no system prompt)
    but not whitespace-only, `context_window` must be >= 0 (0 = unlimited),
    `temperature` must be within OpenAI's documented 0..2 range when set,
    `max_tokens` must be > 0 when set.

    `extra_body` flows through to OpenRouter's extension fields (provider,
    models, transforms, plugins, reasoning, cache_control, etc.) via the openai
    SDK's `extra_body` kwarg.
    """

    model_name: str = Field(default="~openai/gpt-latest", min_length=1)
    instructions: str = ""
    context_window: int = Field(default=16, ge=0)  # 0 = unlimited
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    response_format: dict | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")

    @field_validator("model_name", "instructions")
    @classmethod
    def _no_whitespace_only(cls, v: str) -> str:
        # Empty string is allowed (instructions="" means "no system prompt");
        # whitespace-only is rejected because it sends "   " downstream and
        # burns a system-message slot or a model-name lookup that the LLM
        # cannot satisfy.
        if v and not v.strip():
            raise ValueError("must not be whitespace-only")
        return v


class HeartConfig(BaseModel):
    """A multi-step heart pipeline. Defaults to one step with default Voice.

    `voices` must contain at least one Voice — a Heart with zero voices
    cannot speak. An empty list at construction raises ValidationError;
    the whirl pipeline then never has to silently degenerate to an empty
    TextIdea, which would look like the model said nothing (and burn
    no API call, with no warning to the caller).
    """

    voices: list[Voice] = Field(
        default_factory=lambda: [Voice()],
        min_length=1,
    )

    model_config = ConfigDict(extra="allow")
