"""Dervish — the entry point of rumi.

Subclass `Dervish`, set a `heart_config`, register a few `@tool` functions, and
call `agent.whirl(...)`. The Dervish listens through its Heart, speaks through
its tools, and remembers through its Tablet.

The public surface (mirrors nodeai's where it makes sense):

- `(world, agent_type=cls.__name__, agent_id)` is the identity. One instance
  per triple, kept in a per-World registry.
- Four lifecycle hooks (override in subclass): `_on_whirl_start_impl`,
  `_on_whirl_ended_impl`, `_on_heart_call_start_impl`, `_on_heart_call_ended_impl`.
- `inscribe(key, value)` / `recall(key, default=None)` for persistent state —
  either a string key or a pydantic class as the key.
- `get_agent(type, id)` / `use_tool(name, args)` for cross-Dervish + direct-tool
  calls.

What's not here, deliberately: `send()`, `whirl_once()`, `register_internal_tool`,
streaming, hot reload. See the DROPPED list in the README.
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Callable, ClassVar

from pydantic import BaseModel

from rumi.agent import registry
from rumi.heart.config import HeartConfig
from rumi.heart.openrouter import OpenRouterHeart
from rumi.ideas.base import IIdea
from rumi.ideas.types import TextIdea
from rumi.tablet.filesystem import _snake_case
from rumi.tools.registry import Tools

if TYPE_CHECKING:
    from rumi.world import World

# Global registry of Dervish subclasses, keyed by FULLY-QUALIFIED class name
# (`module.ClassName`) — Populated via `__init_subclass__` — used by `get_agent`
# to resolve cross-Dervish lookups. The qualified key prevents a silent-aliasing
# bug: without `__module__` in the key, two same-named Dervish classes from different
# packages would overwrite each other on import, and `get_agent("Tool", ...)`
# would silently return whichever was imported last.
_AGENT_SUBCLASSES: dict[str, type] = {}


def _qualified_agent_name(cls: type) -> str:
    """`module.ClassName` if module is set, else bare `ClassName` (for tests)."""
    return f"{cls.__module__}.{cls.__name__}" if cls.__module__ else cls.__name__


class Dervish:
    """A being that whirls, calls tools, and remembers.

    Subclass and set `heart_config` to bring an Dervish into the world.
    Each `(world, agent_type, agent_id)` triple maps to exactly one instance.
    """

    heart_config: ClassVar[HeartConfig | None] = None

    # ---- Subclass registration ----

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _AGENT_SUBCLASSES[_qualified_agent_name(cls)] = cls

    # ---- Multiton via __new__ ----

    def __new__(cls, *, world: "World", agent_id: str, **kwargs: Any) -> "Dervish":
        # Use the qualified name as the multiton key — two same-named classes
        # from different modules must map to DIFFERENT instances (the same
        # qualification _AGENT_SUBCLASSES uses, applied to the per-World
        # instance registry).
        return registry.get_or_create(
            cls, world=world, agent_type=_qualified_agent_name(cls),
            agent_id=agent_id, **kwargs,
        )

    def __init__(self, *, world: "World", agent_id: str, **kwargs: Any) -> None:
        # Idempotent: re-init on registry cache hit is a no-op
        if getattr(self, "_initialized", False):
            return

        if type(self).heart_config is None:
            raise ValueError(
                f"{type(self).__name__} has no Heart — set "
                f"{type(self).__name__}.heart_config = HeartConfig(...) "
                "as a class attribute before instantiating. An Dervish without "
                "a Heart cannot whirl."
            )

        # agent_id is the tablet scope. Reject inputs that would silently
        # alias instances on disk (empty / whitespace) or escape the World's
        # data_dir entirely (path separators / parent-dir refs). The scope
        # dir is built by raw `Path` join (data_dir / agent_type / agent_id),
        # so `_safe_filename` never gets to sanitize this layer — the check
        # has to happen at the boundary. The twist beyond mere disk garbage:
        # agent_id="../escape" is a literal sandbox-escape (writes outside
        # data_dir).
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ValueError(
                f"{type(self).__name__}(agent_id=...) must be a non-empty, "
                f"non-whitespace string. Got {agent_id!r}. The agent_id is "
                "the tablet scope — a blank id would alias every instance "
                "onto the agent_type directory itself."
            )
        if "/" in agent_id or "\\" in agent_id or agent_id in (".", ".."):
            raise ValueError(
                f"{type(self).__name__}(agent_id={agent_id!r}) cannot "
                "contain path separators ('/' or '\\\\') or be '.'/'..' — "
                "the tablet scope dir is built by Path-joining agent_id, "
                "so these would escape the World's data_dir."
            )

        self._world = world
        # `_agent_type` is the FULLY QUALIFIED identity (`module.ClassName`).
        # The qualification that keeps _AGENT_SUBCLASSES from aliasing two
        # same-named Dervish classes on import is carried through to three
        # downstream consumers:
        #   - `world.tablet_for(agent_type, ...)` → on-disk directory path
        #   - `world.logger_for(agent_type, ...)` → logger namespace
        #   - `Tools(agent_type)` → tool registry filter
        # All three previously used the bare class name and would have
        # silently shared state between distinct same-named classes.
        self._agent_type = _qualified_agent_name(type(self))
        self._agent_id = agent_id

        # Per-Dervish resources (now keyed by qualified agent_type)
        self._tablet = world.tablet_for(self._agent_type, agent_id)
        self._logger = world.logger_for(self._agent_type, agent_id)
        self._tools = Tools(self._agent_type)
        self._heart = OpenRouterHeart(world)

        # In-memory idea history
        self._ideas: list[IIdea] = []
        self._current_idea: IIdea | None = None

        # Replay persisted ideas (best-effort)
        self._replay_history()

        self._initialized = True
        self._logger.debug(
            "rumi.agent.init agent_type=%s agent_id=%s tools=%d voices=%d",
            self._agent_type, agent_id, len(self._tools.public()),
            len(type(self).heart_config.voices),
        )

    # ---- Properties ----

    @property
    def world(self) -> "World":
        return self._world

    @property
    def agent_type(self) -> str:
        return self._agent_type

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def log(self) -> logging.Logger:
        return self._logger

    @property
    def tablet(self):
        return self._tablet

    @property
    def tools(self) -> Tools:
        return self._tools

    @property
    def heart(self) -> OpenRouterHeart:
        return self._heart

    @property
    def ideas(self) -> list[IIdea]:
        """A copy of the in-memory idea history."""
        return list(self._ideas)

    @property
    def current_idea(self) -> IIdea | None:
        """The most recent Idea — the last thing said, or None if nothing yet."""
        return self._current_idea

    # ---- Public surface ----

    def whirl(
        self,
        message: str | IIdea | list[IIdea],
        *,
        response_model: type[BaseModel] | None = None,
        cancel_check: "Callable[[], bool] | None" = None,
    ) -> IIdea:
        """Run one whirl cycle and return the final Idea.

        `cancel_check`, if given, is a no-argument callable polled at each
        tool-loop boundary; when it returns True the cycle raises
        `WhirlCancelled` (stopping the agent before its next LLM call). It
        defaults to None — an ordinary `agent.whirl("...")` is unaffected.
        The rumi-edge daemon supplies one so A2A `tasks/cancel` can
        cooperatively stop a running whirl across the process seam.
        """
        from rumi._core import run_whirl_pipeline  # the native whirl pipeline
        return run_whirl_pipeline(self, message, response_model, cancel_check)

    def inscribe(self, key: str | type[BaseModel], value: Any) -> None:
        """Write `value` onto this Dervish's tablet, under `key`.

        `key` may be a string OR a pydantic BaseModel subclass. With a class
        key, the value must be an instance of that class — its model_dump()
        is persisted, and `recall(KeyClass)` returns a fresh instance.
        """
        if isinstance(key, type) and issubclass(key, BaseModel):
            if not isinstance(value, key):
                raise TypeError(
                    "When inscribing under a class key, the value must be an "
                    f"instance of that class. Got {type(value).__name__}; "
                    f"expected {key.__name__}."
                )
            self._tablet.set(_inscribed_key_for(key), json.loads(value.model_dump_json()))
        else:
            if not isinstance(key, str):
                raise TypeError(
                    "An inscribe key must be a string or a pydantic class. "
                    f"Got {type(key).__name__}."
                )
            self._tablet.set(key, _coerce_for_json(value))

    def recall(
        self,
        key: str | type[BaseModel],
        default: Any = None,
    ) -> Any:
        """Read back what was inscribed under `key`. Returns `default` if missing.

        If `key` is a pydantic class, the stored dict is re-hydrated to an
        instance via `model_validate`.

        For class-keyed reads, `default` must be `None` or an instance of `key`
        — `inscribe(key, value)` already enforces `isinstance(value, key)`, and
        the symmetric recall contract follows the same shape. A wrong-type
        default would otherwise return on an empty tablet and surface far
        away as an AttributeError on first field access, blaming the field
        rather than the recall call site.
        """
        if isinstance(key, type) and issubclass(key, BaseModel):
            if default is not None and not isinstance(default, key):
                raise TypeError(
                    f"recall({key.__name__}, default=...) expects a "
                    f"{key.__name__} instance or None — got "
                    f"{type(default).__name__}. The default must satisfy the "
                    "same type contract as the recalled value."
                )
            raw = self._tablet.get(_inscribed_key_for(key), None)
            if raw is None:
                return default
            return key.model_validate(raw)
        if not isinstance(key, str):
            raise TypeError(
                "A recall key must be a string or a pydantic class. "
                f"Got {type(key).__name__}."
            )
        return self._tablet.get(key, default)

    def get_agent(self, agent_type: str, agent_id: str) -> "Dervish":
        """Look up (or create) a sibling Dervish on the same World.

        `agent_type` may be either the qualified form (`module.ClassName`) or
        the bare class name (`ClassName`). Bare-name lookups fall through to
        a suffix scan: exactly one match → use it (preserves the simple call
        site `get_agent("Researcher", ...)`); multiple matches → raise with
        a disambiguation hint so the caller can pass the qualified name.
        """
        cls = _AGENT_SUBCLASSES.get(agent_type)
        if cls is None:
            # Try bare-name fallback for backward compat with un-qualified calls.
            matches = [
                c for k, c in _AGENT_SUBCLASSES.items()
                if k.rsplit(".", 1)[-1] == agent_type
            ]
            if len(matches) == 1:
                cls = matches[0]
            elif len(matches) > 1:
                qualified = sorted(
                    k for k, c in _AGENT_SUBCLASSES.items()
                    if k.rsplit(".", 1)[-1] == agent_type
                )
                raise KeyError(
                    f"Dervish name {agent_type!r} is ambiguous — multiple classes "
                    f"with that bare name have been imported: {qualified}. Pass "
                    "the module-qualified name (e.g. \"pkg.module.ClassName\")."
                )
        if cls is None:
            raise KeyError(
                f"No Dervish named {agent_type!r} has been imported into this "
                "World. Import the class first — registration happens at "
                "class-definition time."
            )
        return cls(world=self._world, agent_id=agent_id)

    def use_tool(self, name: str, args: dict | None = None) -> Any:
        """Invoke a tool directly (skip the LLM). Internal tools are reachable.

        `args` may be `None` (no arguments) or a dict of keyword arguments.
        Non-dict / non-None inputs are rejected at the boundary — they would
        otherwise crash inside `function(**args)` with an opaque framework-
        internal `TypeError`, which names an internal variable rather than
        explaining the contract. Loud rejection at the user-facing surface.
        """
        if args is None:
            args = {}
        elif not isinstance(args, dict):
            raise TypeError(
                f"Dervish.use_tool(name={name!r}, args=...) expects a dict of "
                f"keyword arguments or None. Got {type(args).__name__}. "
                "Tools are called as `function(**args)`, so `args` must be a "
                "mapping of param-name -> value."
            )
        entry = self._tools.get(name)
        if entry is None:
            raise KeyError(
                f"A {self._agent_type} has no tool named {name!r}. "
                "Register it with @tool([\"" + self._agent_type + "\"]) or "
                "check the spelling."
            )
        return entry.function(**args)

    # ---- Hooks (override in subclass; default no-op) ----

    def _on_whirl_start_impl(self, incoming_ideas: list[IIdea]) -> None:
        """Fires once at the top of a whirl() cycle, after the incoming
        message has been normalized and persisted. Sees the fresh ideas."""

    def _on_whirl_ended_impl(self, final_idea: IIdea) -> None:
        """Fires once after a whirl() cycle completes. Sees the final
        IIdea about to be returned. Override to update inscribed state
        based on which tools fired during the cycle (see `Mystic` in
        `rumi/example.py` for a worked example)."""

    def _on_heart_call_start_impl(self, payload_in: dict) -> None:
        """Fires before each LLM call. Called once per voice step in the
        Heart pipeline. `payload_in` holds the model name, messages,
        tools, and response_model (if any) about to be sent."""

    def _on_heart_call_ended_impl(self, payload_in: dict, payload_out: dict) -> None:
        """Fires after each LLM call returns. `payload_out` is the raw
        response dict — usage tokens and OpenRouter cost metadata live
        there if the upstream provider returned them."""

    # ---- Internal: history + persistence ----

    def _append_idea(self, idea: IIdea) -> None:
        """Append an idea to in-memory history and persist it to the Tablet.

        Persists FIRST, then commits to in-memory state. This is the write-side
        counterpart of `_replay_history`'s read-side invariants (the seq
        tiebreak and the pipeline's leading-orphan skip): if disk can't see an
        idea, neither can memory. A transient persistence failure (Windows file lock, ENOSPC,
        etc.) now surfaces as a propagated exception at the call site, instead
        of silently desyncing memory and disk and showing up later as a
        confusing OpenAI HTTP 400 after restart (orphaned ToolRequestIdea with
        no matching ToolResultIdea on the next whirl).
        """
        self._tablet.set(f"history/{idea.id}", idea.to_dict())
        self._ideas.append(idea)
        self._current_idea = idea

    def _replay_history(self) -> None:
        """Read back the past — re-hydrate the in-memory idea list from the Tablet."""
        try:
            keys = sorted(self._tablet.list_keys("history/"))
        except Exception as exc:
            self._logger.debug("could not replay history: %s", exc)
            return

        for key in keys:
            raw = self._tablet.get(key)
            if not raw:
                # `Tablet.get(key)` returns None / falsy for:
                #   - corrupt JSON (JSONDecodeError swallowed)
                #   - OS errors (PermissionError, file vanished, ENOSPC mid-read)
                #   - missing payload (the key existed in list_keys but the
                #     file is empty)
                # The next `from_dict` branch below warns on parseable-but-
                # malformed data; THIS branch silently dropped truly unreadable
                # files, leaving a ghost gap in history the user could not
                # detect (an orphaned ToolResultIdea whose ToolRequestIdea got
                # corrupted would silently disappear from replay). Match the
                # observability of the from_dict branch with a parallel
                # warning so the failure is visible.
                self._logger.warning(
                    "history entry %s unreadable on disk (corrupt JSON, "
                    "OS error, or empty payload) — skipping during replay",
                    key,
                )
                continue
            try:
                idea = IIdea.from_dict(raw)
            except Exception as exc:
                self._logger.warning("could not rehydrate %s: %s", key, exc)
                continue
            self._ideas.append(idea)

        # Sort by (timestamp, seq, id) — the tablet doesn't guarantee key
        # order, and stable sort on equal timestamps would otherwise preserve
        # the random alphabetical-UUID order from list_keys. The seq tiebreaker
        # preserves the original write-order within a session, which is what
        # we need so a ToolResultIdea never replays before its ToolRequestIdea.
        self._ideas.sort(key=lambda t: (t.timestamp, getattr(t, "seq", 0), t.id))
        if self._ideas:
            self._current_idea = self._ideas[-1]


# --- Helpers ---


def _inscribed_key_for(cls: type[BaseModel]) -> str:
    """Filesystem-safe key for a class-keyed inscribe value.

    Qualified by the class's module path so distinct classes never alias on
    the tablet. Two known collision patterns the unqualified form silently
    suffered:
      - acronym-cased pairs:  APIKey vs ApiKey  → both snake_case to "api_key"
      - same-named classes in different modules:  module_a.Settings vs
        module_b.Settings  → both snake_case to "settings"
    Both would silently overwrite each other's inscribed state. The module
    prefix (`.`-separated, which `_safe_filename` preserves) closes the seam
    while keeping tablet filenames human-greppable.
    """
    snake = _snake_case(cls.__name__)
    mod = cls.__module__ or ""
    return f"{mod}.{snake}" if mod else snake


def _coerce_for_json(value: Any) -> Any:
    """Make a value JSON-serializable. Pydantic instances → dict; others pass through."""
    if isinstance(value, BaseModel):
        return json.loads(value.model_dump_json())
    return value


def inscribe(cls: type[BaseModel]) -> type[BaseModel]:
    """A marker decorator: 'this class is meant to be inscribed.'

    The class identity serves as the inscribe key; the round-trip uses
    `model_dump_json()` and `model_validate_json(...)` so a non-BaseModel
    target cannot be serialized. The decorator validates at decoration
    time — moving the rejection from runtime-distant (inside `Dervish.inscribe`,
    blaming the call site) to the actual line the user wrote wrong: catch the
    misuse at the decorator, not at first use.
    """
    if not (isinstance(cls, type) and issubclass(cls, BaseModel)):
        raise TypeError(
            f"@inscribe expects a pydantic BaseModel subclass — got "
            f"{getattr(cls, '__name__', type(cls).__name__)!r}. "
            "Inheriting from pydantic.BaseModel is required so the "
            "instance can be serialized via model_dump_json()."
        )
    return cls
