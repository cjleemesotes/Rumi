"""Dervish-scoped logger factory. Stdlib logging only — no Rich, no JSON mode.

Logger name convention: `rumi.<agent_type>.<agent_id>`. Handlers attached idempotently
so calling `agent_logger(...)` more than once doesn't duplicate output.
"""
from __future__ import annotations

import logging
import sys

_FORMAT = "[%(name)s] %(levelname)s %(message)s"
_HANDLER_TAG = "_rumi_handler"


def _ensure_root_handler(level: str = "INFO") -> None:
    root = logging.getLogger("rumi")
    has_ours = any(getattr(h, _HANDLER_TAG, False) for h in root.handlers)
    if has_ours:
        return
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT))
    setattr(handler, _HANDLER_TAG, True)
    root.addHandler(handler)
    root.setLevel(level.upper())
    # Don't propagate up to the root root; we own this namespace.
    root.propagate = False


def agent_logger(agent_type: str, agent_id: str, level: str = "INFO") -> logging.Logger:
    """Return the logger for a given (agent_type, agent_id).

    Idempotent — repeated calls return the same logger and do not attach
    duplicate handlers.
    """
    _ensure_root_handler(level)
    return logging.getLogger(f"rumi.{agent_type}.{agent_id}")


def set_log_level(level: str) -> None:
    """Update the root rumi logger level (affects all Dervish loggers)."""
    logging.getLogger("rumi").setLevel(level.upper())
