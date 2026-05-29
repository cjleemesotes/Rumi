"""Tests for HeartConfig + Voice."""
from __future__ import annotations

import pytest

from rumi.heart.config import HeartConfig, Voice


def test_voice_defaults():
    voice = Voice()
    assert voice.model_name == "~openai/gpt-latest"
    assert voice.instructions == ""
    assert voice.context_window == 16
    assert voice.temperature is None
    assert voice.max_tokens is None
    assert voice.response_format is None
    assert voice.extra_body == {}


def test_voice_extra_pass_through():
    voice = Voice(model_name="x", arbitrary="ok")
    assert voice.model_name == "x"
    # extra="allow" means arbitrary fields survive
    assert voice.model_dump()["arbitrary"] == "ok"


def test_heart_config_default_has_one_voice():
    cfg = HeartConfig()
    assert len(cfg.voices) == 1
    assert isinstance(cfg.voices[0], Voice)


def test_heart_config_multi_step():
    cfg = HeartConfig(voices=[
        Voice(model_name="~openai/gpt-latest"),
        Voice(model_name="openai/gpt-5.5", instructions="step 2"),
    ])
    assert len(cfg.voices) == 2
    assert cfg.voices[1].instructions == "step 2"


def test_extra_body_threading():
    c = Voice(extra_body={"provider": {"require_parameters": True}})
    assert c.extra_body == {"provider": {"require_parameters": True}}


def test_no_factory_methods():
    # Confirm we did not accidentally add .auto() / .create_simple_config()
    assert not hasattr(HeartConfig, "auto")
    assert not hasattr(HeartConfig, "create_simple_config")


def test_voice_rejects_degenerate_field_values():
    """Regression: Voice field constraints catch typos that would otherwise
    burn a partially-persisted LLM call before failing at the provider.

    Each of these would have silently corrupted a request mid-pipeline
    (after the user TextIdea was already appended to history) and then
    failed at the OpenRouter/OpenAI HTTP layer with an opaque error.
    Catching at construction surfaces the typo before any Dervish is
    instantiated with the Voice.
    """
    from pydantic import ValidationError

    bad_kwargs = [
        {"model_name": ""},          # empty model name
        {"model_name": "   "},       # whitespace-only model name
        {"instructions": "   "},     # whitespace-only system prompt
        {"context_window": -1},      # negative window
        {"temperature": -0.1},       # below OpenAI's 0..2 range
        {"temperature": 2.5},        # above OpenAI's 0..2 range
        {"max_tokens": 0},           # zero tokens — no output
        {"max_tokens": -100},        # negative tokens
    ]
    for kw in bad_kwargs:
        with pytest.raises(ValidationError):
            Voice(**kw)


def test_voice_accepts_legitimate_field_values():
    """Confirm the constraints don't accidentally reject valid configs."""
    # Default Voice still works
    Voice()
    # context_window=0 is documented as "unlimited" — must remain valid
    Voice(context_window=0)
    # temperature at the boundaries
    Voice(temperature=0.0)
    Voice(temperature=2.0)
    # max_tokens=1 is valid (minimum sensible)
    Voice(max_tokens=1)
    # instructions="" means "no system prompt" — must remain valid
    Voice(instructions="")
    # None for optional fields
    Voice(temperature=None, max_tokens=None)


def test_heart_config_rejects_empty_voices():
    """Regression: a Heart with zero voices cannot speak.

    `HeartConfig(voices=[])` was previously accepted. The whirl pipeline would
    skip the per-voice loop entirely, fall through to the empty-final defensive
    branch, append an empty TextIdea, and return it. No LLM
    call, no warning, no error — the caller sees `agent.whirl("hello")` come
    back with empty content as if the model said nothing. This is the same
    silent-degeneration shape closed for inscribed keys, agent classes, and
    the tool registry. Raise at
    HeartConfig construction so misconfiguration surfaces before any Dervish
    is instantiated with it.
    """
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        HeartConfig(voices=[])
