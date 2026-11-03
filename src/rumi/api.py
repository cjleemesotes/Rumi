"""Public API surface — `from rumi import ...`.

Stable exports (mirroring nodeai's surface for cognitive parity):
- Identity / runtime: `Dervish`, `World`
- Heart: `HeartConfig`, `Voice`, `OpenRouterHeart`
- Tools: `tool`, `inscribe`
- Thoughts: `IIdea`, `TextIdea`, `ToolRequestIdea`, `ToolResultIdea`,
  `JsonIdea`, `BaseModelIdea`
- Recovery (an interrupted tool call): `PendingToolCall`, `RecoveryError`,
  `RecoveryNeededError`, `UnsupportedRecoveryError`, `ToolCallNotFoundError`,
  `RecoveryConflictError`
"""
from __future__ import annotations

from rumi.heart.config import HeartConfig, Voice
from rumi.heart.openrouter import OpenRouterHeart
from rumi.agent.agent import Dervish, inscribe
from rumi.agent.recovery import (
    PendingToolCall,
    RecoveryConflictError,
    RecoveryError,
    RecoveryNeededError,
    ToolCallNotFoundError,
    UnsupportedRecoveryError,
)
from rumi.world import World
from rumi.ideas.base import IIdea
from rumi.ideas.types import (
    BaseModelIdea,
    JsonIdea,
    TextIdea,
    ToolRequestIdea,
    ToolResultIdea,
)
from rumi.tools.decorator import tool

__all__ = [
    # runtime
    "Dervish",
    "World",
    # heart
    "HeartConfig",
    "Voice",
    "OpenRouterHeart",
    # tools
    "tool",
    "inscribe",
    # ideas
    "IIdea",
    "TextIdea",
    "ToolRequestIdea",
    "ToolResultIdea",
    "JsonIdea",
    "BaseModelIdea",
    # recovery
    "PendingToolCall",
    "RecoveryError",
    "RecoveryNeededError",
    "UnsupportedRecoveryError",
    "ToolCallNotFoundError",
    "RecoveryConflictError",
]
