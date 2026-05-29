"""Tools facade + execute_tool_loop.

Per Dervish: `Tools(agent_type)` takes a snapshot of the global tool registry
filtered by `agent_type`. `execute_tool_loop` calls each requested tool
sequentially, catching per-tool exceptions and wrapping each result (or error)
in a `ToolResultIdea`.
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from rumi.ideas.types import ToolRequestIdea, ToolResultIdea
from rumi.tools.decorator import ToolEntry, get_tools_for

if TYPE_CHECKING:
    pass

log = logging.getLogger("rumi.tools")


class Tools:
    """Per-Dervish tool facade. Snapshot of the global registry, scoped by agent_type."""

    def __init__(self, agent_type: str) -> None:
        self._agent_type = agent_type
        # Snapshot at construction; internal+public both available
        self._all: tuple[ToolEntry, ...] = tuple(
            get_tools_for(agent_type, include_internal=True)
        )
        # Two tools registered with the same `fn.__name__` for the same
        # agent_type would land in `_all` together, advertise two identical
        # schemas to the LLM (which has no way to disambiguate them), and
        # silently drop one when routed through `_by_name`. Raise at Dervish
        # construction instead of waiting for a confusing LLM call to a
        # dead function.
        self._by_name: dict[str, ToolEntry] = {}
        for t in self._all:
            if t.name in self._by_name:
                prev = self._by_name[t.name]
                raise ValueError(
                    f"Two tools registered as {t.name!r} for agent_type "
                    f"{agent_type!r}: "
                    f"{prev.function.__module__}.{prev.function.__qualname__} "
                    f"and {t.function.__module__}.{t.function.__qualname__}. "
                    "Rename one — the LLM cannot disambiguate same-named tools."
                )
            self._by_name[t.name] = t

    @property
    def agent_type(self) -> str:
        return self._agent_type

    def public(self) -> list[ToolEntry]:
        """Tools exposed to the LLM (excludes internal)."""
        return [t for t in self._all if not t.internal]

    def all(self) -> list[ToolEntry]:
        """All tools including internal."""
        return list(self._all)

    def get(self, name: str) -> ToolEntry | None:
        return self._by_name.get(name)

    def schemas(self) -> list[dict]:
        """OpenAI tool dicts for the LLM payload — public tools only."""
        return [t.schema for t in self.public()]


def execute_tool_loop(
    tool_request: ToolRequestIdea,
    tools: Tools,
) -> list[ToolResultIdea]:
    """Run each requested tool sequentially.

    Per-tool exceptions are caught and rendered as ToolResultIdea content
    starting with "ERROR: ". The LLM sees the error string; rumi caller does
    not get an exception propagated up (per axiom 26).
    """
    results: list[ToolResultIdea] = []
    for call in tool_request.parsed_calls():
        name = call.get("name") or ""
        args = call.get("arguments") or {}
        call_id = call.get("id") or ""

        entry = tools.get(name)
        if entry is None:
            log.warning("tool not found: %s", name)
            results.append(ToolResultIdea(
                tool_call_id=call_id,
                name=name,
                content=f"ERROR: tool '{name}' not registered for agent_type '{tools.agent_type}'",
            ))
            continue

        try:
            result = entry.function(**args)
            content = _stringify_result(result)
        except Exception as exc:
            log.exception("tool %s raised", name)
            content = f"ERROR: {type(exc).__name__}: {exc}"

        results.append(ToolResultIdea(
            tool_call_id=call_id,
            name=name,
            content=content,
        ))

    return results


def _stringify_result(result) -> str:
    """Coerce a tool result into a string for the LLM."""
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(result)
