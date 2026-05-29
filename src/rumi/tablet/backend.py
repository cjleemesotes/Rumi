"""Tablet — the surface an Dervish writes on.

An ITablet is a small key/value store scoped to one `(agent_type, agent_id)`.
Implementations decide where the bytes go (filesystem, memory, remote);
rumi ships `FilesystemTablet`, which stores JSON files at
`{data_dir}/{agent_type}/{agent_id}/{key}.json`.

The Lawh al-Mahfuz — *the Preserved Tablet* — is the Sufi image: a record
that survives the writing.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ITablet(ABC):
    """Abstract tablet backend. CRUD + listing on string keys."""

    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any: ...

    @abstractmethod
    def set(self, key: str, value: Any) -> None: ...

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a key. Returns True if the key existed and was deleted."""
        ...

    @abstractmethod
    def list_keys(self, prefix: str = "") -> list[str]:
        """List keys (unscoped — backend implementations apply their own scope)."""
        ...


class TabletScopeMixin:
    """Scopes all keys under `{agent_type}/{agent_id}/`.

    Used by backends that need per-(agent_type, agent_id) namespacing. No account
    or world prefix — rumi identity is just `(agent_type, agent_id)`.
    """

    def __init__(self, agent_type: str, agent_id: str) -> None:
        if not agent_type or not agent_id:
            raise ValueError("agent_type and agent_id must both be non-empty")
        if "/" in agent_type or "/" in agent_id:
            raise ValueError("agent_type and agent_id must not contain '/'")
        self._agent_type = agent_type
        self._agent_id = agent_id
        self._scope = f"{agent_type}/{agent_id}"

    @property
    def agent_type(self) -> str:
        return self._agent_type

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def scope(self) -> str:
        return self._scope

    def _scoped(self, key: str) -> str:
        """Prepend the scope to a key (`agent_type/agent_id/key`)."""
        return f"{self._scope}/{key}"

    def _unscoped(self, full_key: str) -> str:
        """Strip the scope prefix from a fully-qualified key."""
        prefix = f"{self._scope}/"
        if full_key.startswith(prefix):
            return full_key[len(prefix):]
        return full_key
