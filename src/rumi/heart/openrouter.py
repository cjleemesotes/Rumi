"""OpenRouterHeart — the only heart. Uses the openai SDK pointed at
openrouter.ai/api/v1 (via World.openrouter_client).

`call()` dispatches one LLM call; `parse_response()` converts the raw response
dict into one of the 5 IIdea subtypes (Text / ToolRequest / Json / BaseModel).
Tool-result ideas are NOT produced here — those come from the tool loop in
`agent.pipeline`.

OpenRouter usage/cost metadata (when returned) lands on the resulting idea's
`metadata` dict as: `cost_usd`, `prompt_tokens`, `completion_tokens`,
`total_tokens`, `resolved_upstream`.
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from rumi.heart.config import Voice
from rumi.ideas.base import IIdea
from rumi.ideas.types import (
    BaseModelIdea,
    JsonIdea,
    TextIdea,
    ToolRequestIdea,
)

if TYPE_CHECKING:
    from rumi.world import World

log = logging.getLogger("rumi.heart")

# Reserved keys that rumi sets on the openai SDK call kwargs. Voice.extra_body
# is for OpenRouter EXTENSION fields only (provider routing, transforms, etc.);
# if a user tries to override one of these via extra_body, the SDK would
# silently win the merge — see the explanatory comment near the check site.
_RESERVED_BODY_KEYS = frozenset({
    "model", "messages", "tools", "tool_choice",
    "temperature", "max_tokens", "response_format", "stream",
})


class OpenRouterHeart:
    """Single-provider heart backed by the openai SDK, pointed at OpenRouter."""

    def __init__(self, world: "World") -> None:
        self.world = world

    # ---- Dispatch ----

    def call(
        self,
        voice: Voice,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> dict:
        """One LLM call. Returns the raw response dict.

        Routing:
        - `response_model` set → use `client.chat.completions.parse(response_format=Model, ...)`
          for structured pydantic output.
        - `voice.response_format` set (e.g. `{"type": "json_object"}`)
          → pass through unchanged.
        - Else → plain `.create(...)`.
        """
        # Pre-emptive check: response_model + tools collide because the openai
        # SDK's .parse() requires `strict: true` on all tool schemas, and
        # rumi's @tool decorator doesn't generate strict-mode schemas. Surface
        # the constraint clearly instead of letting the SDK raise a cryptic
        # "is not strict" error mid-call.
        if response_model is not None and tools:
            raise ValueError(
                "rumi does not currently support response_model + tools in the "
                "same whirl() call — the openai SDK's .parse() requires "
                "strict-mode tool schemas, which rumi's @tool decorator does "
                "not generate. Either: (a) drop response_model and let the "
                "Heart return text/tool-calls, or (b) call whirl() on an "
                "Dervish without registered tools when you need structured "
                "output."
            )

        client = self.world.openrouter_client

        kwargs: dict[str, Any] = {
            "model": voice.model_name,
            "messages": messages,
        }
        if voice.temperature is not None:
            kwargs["temperature"] = voice.temperature
        if voice.max_tokens is not None:
            kwargs["max_tokens"] = voice.max_tokens
        if tools:
            kwargs["tools"] = tools
        if voice.extra_body:
            # The openai SDK MERGES extra_body into the request JSON with
            # extra_body WINNING on key collisions
            # (`_merge_mappings(json_data, options.extra_json)` in
            # BaseClient._build_request). A user setting `extra_body={"model":
            # "X"}` would silently override the model rumi just set — billed to
            # the wrong upstream with no warning. Same hole for messages, tools,
            # temperature, etc. extra_body is meant for OpenRouter EXTENSIONS
            # (provider, models, transforms, plugins, reasoning, cache_control,
            # route) — not rumi-managed primary fields. Surface collisions loud.
            clash = _RESERVED_BODY_KEYS & voice.extra_body.keys()
            if clash:
                raise ValueError(
                    f"Voice.extra_body cannot set {sorted(clash)} — those are "
                    "rumi-managed fields (set them on the Voice itself). "
                    "extra_body is for OpenRouter extensions only: provider, "
                    "models, transforms, plugins, reasoning, cache_control, "
                    "route."
                )
            kwargs["extra_body"] = dict(voice.extra_body)

        log.debug("OPENROUTER.SEND model=%s tools=%d msgs=%d",
                  voice.model_name, len(tools or []), len(messages))

        try:
            if response_model is not None:
                # Structured output via openai SDK .parse() — returns ParsedChatCompletion
                response = client.chat.completions.parse(
                    response_format=response_model, **kwargs
                )
            elif voice.response_format is not None:
                response = client.chat.completions.create(
                    response_format=voice.response_format, **kwargs
                )
            else:
                response = client.chat.completions.create(**kwargs)
        except Exception as exc:
            log.error("OPENROUTER.ERROR: %s", exc)
            raise

        return _to_dict(response)

    # ---- Parse ----

    def parse_response(
        self,
        response: dict,
        response_model: type[BaseModel] | None = None,
    ) -> IIdea:
        """Convert raw response dict → IIdea subtype.

        Branching:
        - tool_calls present → ToolRequestIdea
        - response_model set → BaseModelIdea (content = parsed pydantic instance)
        - JSON-shaped content (json_object or json_schema response_format) → JsonIdea
        - else → TextIdea
        """
        choices = response.get("choices") or []
        if not choices:
            return TextIdea(content="", metadata=_metadata_from(response))

        msg = choices[0].get("message") or {}
        tool_calls = msg.get("tool_calls")
        content = msg.get("content")
        parsed = msg.get("parsed")

        meta = _metadata_from(response)

        if tool_calls:
            return ToolRequestIdea(
                content=list(tool_calls),
                text=content or "",
                metadata=meta,
            )

        if response_model is not None and parsed is not None:
            # openai SDK's .parse() puts a real pydantic instance under
            # "parsed" on the SDK object. But `_to_dict` calls model_dump()
            # to normalize the response, and model_dump() recursively
            # flattens every nested pydantic field — including parsed — to
            # a plain dict. So by the time we read it here, the instance is
            # gone. Re-validate so BaseModelIdea.content actually holds a
            # pydantic instance (as its docstring promises and to_openai_message
            # assumes), instead of a dict that would force every caller to
            # do model_validate(result.content) themselves.
            if not isinstance(parsed, response_model):
                parsed = response_model.model_validate(parsed)
            return BaseModelIdea(content=parsed, metadata=meta)

        if response_model is not None and content:
            # Fallback: parse JSON content manually
            try:
                instance = response_model.model_validate_json(content)
                return BaseModelIdea(content=instance, metadata=meta)
            except Exception as exc:
                log.warning("response_model parse failed, returning text: %s", exc)

        # JSON-shaped content?
        if isinstance(content, str) and content.strip().startswith(("{", "[")):
            try:
                return JsonIdea(content=json.loads(content), metadata=meta)
            except json.JSONDecodeError:
                pass

        return TextIdea(content=content or "", metadata=meta)


# --- Helpers ---


def _to_dict(response: Any) -> dict:
    """Convert an openai-SDK response object into a plain dict.

    openai>=1.0 returns pydantic-based ChatCompletion objects that have
    `model_dump()`. For mocked responses (test path) we accept dict already.
    """
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "to_dict"):
        return response.to_dict()
    # Last resort — best-effort attribute scrape
    return {"_raw": str(response)}


def _metadata_from(response: dict) -> dict:
    """Extract usage + cost + upstream-provider metadata for IIdea.metadata."""
    usage = response.get("usage") or {}
    meta: dict[str, Any] = {}
    if "prompt_tokens" in usage:
        meta["prompt_tokens"] = usage["prompt_tokens"]
    if "completion_tokens" in usage:
        meta["completion_tokens"] = usage["completion_tokens"]
    if "total_tokens" in usage:
        meta["total_tokens"] = usage["total_tokens"]
    # OpenRouter-specific: usage.cost (USD)
    if "cost" in usage:
        meta["cost_usd"] = usage["cost"]
    # OpenRouter-specific: top-level "provider" field with resolved upstream
    if "provider" in response:
        meta["resolved_upstream"] = response["provider"]
    return meta
