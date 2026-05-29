"""@tool decorator + global MVCC tool registry.

Registration is global (`@tool(["Researcher"])` registers worldwide). Each Dervish
filters the snapshot by its `agent_type` at instantiation. The snapshot is an
immutable tuple, swapped under a lock — MVCC readers see a consistent view
without locking.
"""
from __future__ import annotations

import enum
import inspect
import threading
import typing
from dataclasses import dataclass, field
from typing import Any, Callable

# --- Global MVCC snapshot ---

_TOOLS_SNAPSHOT: tuple["ToolEntry", ...] = ()
_TOOLS_LOCK = threading.Lock()


@dataclass(frozen=True)
class ToolEntry:
    function: Callable[..., Any]
    agent_types: tuple[str, ...]
    internal: bool
    name: str
    schema: dict = field(default_factory=dict)  # OpenAI tool dict

    def __hash__(self) -> int:
        return hash((self.name, self.agent_types, self.internal))


def tool(
    agent_types: str | list[str],
    *,
    internal: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a function as a tool, scoped to one or more `agent_type` strings.

    Async tools are rejected at registration time — rumi is sync-only.
    Internal tools are hidden from the LLM (used via `agent.use_tool` only).
    """
    types_tuple: tuple[str, ...] = (
        (agent_types,) if isinstance(agent_types, str) else tuple(agent_types)
    )

    # Reject degenerate agent_types at decoration time. A tool registered for
    # zero agent types (or for empty / whitespace strings) is silent dead
    # code: `get_tools_for(...)` can never return it because `Dervish._agent_type`
    # is built from `cls.__name__` which Python guarantees non-empty. Catch it
    # here so an unreachable registration fails loudly at import, not silently.
    if not types_tuple:
        raise ValueError(
            "@tool([]) registers a function for no agent_type — it would "
            "never be reachable from any Dervish. Pass at least one agent_type "
            "string."
        )
    for t in types_tuple:
        if not isinstance(t, str) or not t.strip():
            raise ValueError(
                f"@tool agent_types must be non-empty, non-whitespace "
                f"strings. Got {t!r}. No Dervish class can match a blank "
                "agent_type."
            )

    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(fn):
            raise TypeError(
                f"{fn.__name__} is a coroutine, and rumi tools are sync — "
                "the reed sings while standing still. Wrap with a sync "
                "function (e.g., `asyncio.run(...)` inside) or use a thread."
            )
        schema = _schema_from_signature(fn)
        entry = ToolEntry(
            function=fn,
            agent_types=types_tuple,
            internal=internal,
            name=fn.__name__,
            schema=schema,
        )
        global _TOOLS_SNAPSHOT
        with _TOOLS_LOCK:
            # Idempotent registration: same (name, module.qualname, agent_types)
            # is a re-registration — typically `importlib.reload` during dev or
            # tests. Drop the prior entry; the new function object replaces it.
            # A genuinely different module.qualname registering the same name
            # for the same agent_type is a real collision, NOT deduped here —
            # `Tools.__init__` raises at Dervish construction so the user sees
            # the conflict instead of silently losing one function.
            new_qual = f"{fn.__module__}.{fn.__qualname__}"
            _TOOLS_SNAPSHOT = tuple(
                t for t in _TOOLS_SNAPSHOT
                if not (
                    t.name == fn.__name__
                    and t.agent_types == types_tuple
                    and f"{t.function.__module__}.{t.function.__qualname__}"
                        == new_qual
                )
            ) + (entry,)
        return fn

    return _decorator


def get_tools_for(agent_type: str, *, include_internal: bool = False) -> list[ToolEntry]:
    """MVCC snapshot read — no lock. Returns tools matching `agent_type`.

    Accepts either the qualified Dervish name (`module.ClassName`, the form
    `Dervish._agent_type` carries) or the bare class name (the
    form most `@tool([...])` users write). A tool registered with the bare
    name like `@tool(["Researcher"])` matches both `pkg_a.research.Researcher`
    and `pkg_b.research.Researcher` — the bare name is class-wide. A tool
    registered with the qualified name `@tool(["pkg_a.research.Researcher"])`
    matches only that exact qualified class.

    By default, internal tools are excluded (they're hidden from the LLM).
    """
    snap = _TOOLS_SNAPSHOT
    bare = agent_type.rsplit(".", 1)[-1] if "." in agent_type else agent_type
    return [
        t for t in snap
        if (agent_type in t.agent_types or bare in t.agent_types)
        and (include_internal or not t.internal)
    ]


def _reset_for_tests() -> None:
    """Test-only: clear the global snapshot. Not part of the public API."""
    global _TOOLS_SNAPSHOT
    with _TOOLS_LOCK:
        _TOOLS_SNAPSHOT = ()


# --- Schema generation from function signature ---

_PY_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _type_to_json_schema(annotation: Any) -> dict:
    """Best-effort mapping from a Python type annotation to a JSON schema fragment."""
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    # Enum subclasses → "enum" constraint so the LLM sees the allowed values
    # (e.g. `class Practice(str, Enum)` with three members surfaces as
    # `{"type": "string", "enum": ["contemplate", "whirl", "compose"]}`).
    # Without this, the LLM sees a plain string and may pass an invalid value,
    # which the Enum constructor would reject at tool-call time.
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        values = [m.value for m in annotation]
        # Infer JSON type from the first member's value
        json_type = "string"
        if values and isinstance(values[0], bool):
            json_type = "boolean"
        elif values and isinstance(values[0], int):
            json_type = "integer"
        elif values and isinstance(values[0], float):
            json_type = "number"
        return {"type": json_type, "enum": values}

    if annotation in _PY_TO_JSON:
        return {"type": _PY_TO_JSON[annotation]}

    if origin is list:
        item = args[0] if args else Any
        return {"type": "array", "items": _type_to_json_schema(item)}

    if origin is dict:
        return {"type": "object"}

    if origin is typing.Union or origin is type(None | int):  # PEP 604 X | Y
        # Strip None for optional handling, keep first non-None type
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _type_to_json_schema(non_none[0])
        return {}  # too complex; let the LLM figure it out

    # Fallback: untyped string
    return {"type": "string"}


def _schema_from_signature(fn: Callable[..., Any]) -> dict:
    """Build an OpenAI tool dict from a function's signature + docstring.

    Uses `typing.get_type_hints` so that PEP 563 / `from __future__ import
    annotations` string-annotations are resolved to real types.
    """
    sig = inspect.signature(fn)
    description = (inspect.getdoc(fn) or fn.__name__).split("\n\n")[0].strip()

    try:
        type_hints = typing.get_type_hints(fn)
    except Exception:
        type_hints = {}

    properties: dict[str, dict] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        # JSON schema has no representation for *args / **kwargs — exposing them
        # as required scalars (`{"type": "string"}` is the fallback) would publish
        # a parameter the function does not accept by that name. The LLM would
        # send `{"kwargs": "..."}` and `entry.function(**args)` would land it as
        # a keyword arg, captured by the VAR_KEYWORD slot as `kwargs={"kwargs": "..."}`
        # — silent semantic corruption. Skip both kinds.
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        annotation = type_hints.get(name, str)
        properties[name] = _type_to_json_schema(annotation)
        if param.default is inspect.Parameter.empty:
            required.append(name)

    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        parameters_schema["required"] = required

    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": description,
            "parameters": parameters_schema,
        },
    }
