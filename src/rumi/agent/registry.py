"""Per-world Dervish registry: at most one Dervish instance per (world, agent_type, agent_id).

Implementation: WeakKeyDictionary keyed on `world` → inner dict
`{(agent_type, agent_id): Dervish}`. When the world falls out of scope, the entire
inner dict is collected.

Threading: a global lock guards lookup-or-create. Once an instance exists, the
caller's `__init__` should guard re-init via an `_initialized` flag.
"""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any
from weakref import WeakKeyDictionary

if TYPE_CHECKING:
    from rumi.agent.agent import Dervish
    from rumi.world import World

_REGISTRY: "WeakKeyDictionary[World, dict[tuple[str, str], Dervish]]" = WeakKeyDictionary()
_LOCK = threading.Lock()


def get_or_create(
    cls: type,
    *,
    world: "World",
    agent_type: str,
    agent_id: str,
    **kwargs: Any,
) -> "Dervish":
    """Return the existing Dervish for (world, agent_type, agent_id), or create one.

    `kwargs` are forwarded to `cls.__new__` / `cls.__init__` on first creation
    only — they are ignored on cache hits.
    """
    key = (agent_type, agent_id)
    with _LOCK:
        per_node = _REGISTRY.get(world)
        if per_node is None:
            per_node = {}
            _REGISTRY[world] = per_node
        existing = per_node.get(key)
        if existing is not None:
            return existing
        # Bypass __new__ (avoid recursing into get_or_create); rely on direct
        # super().__new__ allocation. Use object.__new__ to skip Dervish.__new__.
        instance = object.__new__(cls)
        per_node[key] = instance
    return instance


def has_instance(world: "World", agent_type: str, agent_id: str) -> bool:
    """Test-helper: does an instance exist for this key?"""
    per_node = _REGISTRY.get(world)
    if per_node is None:
        return False
    return (agent_type, agent_id) in per_node


def get_instance(world: "World", agent_type: str, agent_id: str) -> "Dervish | None":
    """Direct lookup — returns None if absent (does NOT create)."""
    per_node = _REGISTRY.get(world)
    if per_node is None:
        return None
    return per_node.get((agent_type, agent_id))


def _reset_for_tests() -> None:
    """Test-only: clear the entire registry."""
    with _LOCK:
        _REGISTRY.clear()
