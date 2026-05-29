"""FilesystemTablet — a JSON-on-disk store backed by the native `rumi._core`.

The Rust core (`rumi._core.Tablet`) owns the filename codec, atomic byte I/O,
read, and listing; JSON (de)serialization stays here in Python — the tablet
stores opaque bytes. One file per key at
`{data_dir}/{agent_type}/{agent_id}/{key}.json`.

The Lawh al-Mahfuz — *the Preserved Tablet* — is the Sufi image: a record
that survives the writing.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rumi.tablet.backend import ITablet, TabletScopeMixin


def _snake_case(name: str) -> str:
    """Convert CamelCase → snake_case. e.g. ResearchState → research_state.

    Used to derive a stable basename for class-keyed inscribes
    (`Dervish.inscribe(SomeModel, ...)`).
    """
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


class FilesystemTablet(ITablet, TabletScopeMixin):
    """JSON-on-disk tablet. The fs + filename-codec layer is the Rust core.

    JSON serialization stays in Python so the on-disk bytes are exactly what
    `json.dumps(value, ensure_ascii=False)` produces; the Rust core writes
    those bytes atomically (tempfile + atomic rename) and owns the key→filename
    codec, read, and listing.
    """

    def __init__(self, data_dir: str | Path, agent_type: str, agent_id: str) -> None:
        # Python-side scope validation (identical messages); the Rust core
        # validates again defensively.
        TabletScopeMixin.__init__(self, agent_type, agent_id)
        from rumi._core import Tablet as _RustTablet

        self._rust = _RustTablet(str(data_dir), agent_type, agent_id)
        self._scope_dir = Path(self._rust.scope_dir())

    @property
    def scope_dir(self) -> Path:
        return self._scope_dir

    def _path(self, key: str) -> Path:
        return Path(self._rust.path(key))

    def get(self, key: str, default: Any = None) -> Any:
        raw = self._rust.get(key)
        if raw is None:
            return default
        try:
            return json.loads(raw)
        except (ValueError, json.JSONDecodeError):
            return default

    def set(self, key: str, value: Any) -> None:
        self._rust.set(key, json.dumps(value, ensure_ascii=False))

    def delete(self, key: str) -> bool:
        return self._rust.delete(key)

    def list_keys(self, prefix: str = "") -> list[str]:
        return list(self._rust.list_keys(prefix))
