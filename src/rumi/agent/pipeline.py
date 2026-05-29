"""Python-side helpers for the whirl-pipeline.

The orchestration itself — input append, the per-voice loop, the tool loop,
the tool-iteration cap, the atomic commit + rollback, cooperative cancel, and
message building — lives in the Rust core (`rumi._core.run_whirl_pipeline`,
called from `Dervish.whirl`). What remains here are the leaves the Rust pipeline
calls back into:

- `_normalize_incoming` / `_validate_idea` — the once-per-whirl input boundary
  (pydantic Idea construction + the verbatim rejection messages).
- `_safe_hook` — lifecycle-hook invocation with exception isolation + logging.
- `WhirlCancelled` — the exception Rust raises (and callers `except` against)
  when a `cancel_check` returns True.
- `MAX_TOOL_ITERATIONS` — the authoritative tool-loop cap the Rust core reads.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rumi.ideas.base import IIdea
from rumi.ideas.types import TextIdea

if TYPE_CHECKING:
    from rumi.agent.agent import Dervish

log = logging.getLogger("rumi.pipeline")

MAX_TOOL_ITERATIONS = 10


class WhirlCancelled(Exception):
    """Raised inside whirl() when a `cancel_check` callable returns True.

    Cooperative cancel: the Rust pipeline polls `cancel_check` only at the top
    of each tool-loop iteration — i.e. *before* each LLM call. This guarantees
    a complete tool cycle (the ToolRequestIdea and ALL its ToolResultIdeas) is
    never interrupted halfway, so cancel can't strand an orphaned
    ToolRequestIdea on disk. The effect is "stop the agent before its next LLM
    call (and before any further side-effecting tools)."

    An in-flight LLM call is NOT interruptible — the openai SDK call has no
    cancellation hook — so cancel observed during a call lands at the next
    boundary, or (for a single-call whirl) is surfaced by the caller after
    whirl() returns. Default `cancel_check=None` makes all of this a no-op:
    the pipeline is byte-for-byte unchanged for callers that don't pass one.
    """


def _safe_hook(agent: "Dervish", name: str, *args) -> None:
    """Invoke a user lifecycle hook, isolating its exceptions from whirl().

    Hooks are documented as observation/bookkeeping — they sit at the bottom
    of the value chain (LLM call already billed, ideas persisted, tool cycle
    closed). A raise in a user override should NOT erase a successful whirl().

    The most concrete failure mode this isolates is `_on_heart_call_ended_impl`:
    it fires AFTER `heart.call(...)` returns the response but BEFORE
    `parse_response` + `_append_idea` run. If the hook raises uncaught, the
    response is received but never parsed or persisted — the user has paid
    for an LLM call that produces nothing reachable. Logging the hook error
    preserves observability for the override author without poisoning callers.

    Core operations (persistence, atomicity, rollback) remain LOUD. Only
    user-supplied hooks are wrapped here.
    """
    try:
        getattr(agent, name)(*args)
    except Exception as exc:
        log.warning(
            "Hook %s on %s raised %s: %s — whirl() continuing. "
            "Override `%s` is observation/bookkeeping; if a raise is intended "
            "as control flow, refactor it out of the hook.",
            name, type(agent).__name__, type(exc).__name__, exc, name,
        )


def _validate_idea(m: IIdea) -> IIdea:
    """Reject IIdeas with empty/whitespace-only content OR non-user role.

    Two failure modes, both caught at the boundary:

    1. Empty content: an empty IIdea persists to disk forever AND ships
       `{"role":"user","content":""}` to the provider, which most return
       as an opaque 400.

    2. Non-user role: each non-user role has a documented owner
       in rumi's architecture:
         - "system" is owned by Voice.instructions
         - "assistant" is owned by Heart.parse_response
         - "tool" is owned by execute_tool_loop
       A caller-injected non-user IIdea would persist forever AND produce
       illegal message sequences (double system messages fighting
       Voice.instructions; "assistant" injection orphaning future tool
       cycles; "tool" without tool_call_id → 400).

    Only the content check is enforced when `content` is a string.
    ToolRequestIdea and ToolResultIdea carry list / tool-payload content
    (not user-typed text) so the content check exempts them naturally,
    but those Ideas are also never reached via `_normalize_incoming` —
    they're produced by the pipeline itself, not passed to whirl().
    """
    c = getattr(m, "content", None)
    if isinstance(c, str) and not c.strip():
        raise ValueError(
            "whirl() got an IIdea with empty/whitespace-only content. "
            "An empty user message persists to disk and is rejected by "
            "most providers with an opaque 400. Pass a non-empty Idea."
        )
    role = getattr(m, "role", None)
    if role != "user":
        raise ValueError(
            f"whirl() got an IIdea with role={role!r}. Caller-supplied "
            "messages must use role='user'. 'system' belongs on "
            "Voice.instructions; 'assistant' and 'tool' are produced by "
            "the Heart and the tool loop, not the caller."
        )
    return m


def _normalize_incoming(
    message: str | IIdea | list[IIdea],
) -> list[IIdea]:
    # Reject degenerate inputs at the boundary, before they silently persist
    # garbage to the tablet AND burn an LLM call. `whirl("")` would otherwise
    # append an empty user TextIdea (lives on disk forever) and send a
    # `{"role": "user", "content": ""}` payload that most providers reject
    # with an opaque error. `whirl([])` would skip the append entirely and
    # call the LLM with whatever history is — on a fresh Dervish, just the
    # system message — billing for a no-op-shaped turn. The rule: a silent
    # degenerate input becomes a loud rejection at the entrypoint.
    #
    # The IIdea path (`whirl(TextIdea(content=""))`, and the same inside a
    # list) falls through `_validate_idea` to the same ValueError, so all
    # three documented `_normalize_incoming` paths (string / IIdea / list)
    # reject empty content uniformly.
    if isinstance(message, str):
        if not message.strip():
            raise ValueError(
                "whirl() needs something to think about — got an empty or "
                "whitespace-only string. Pass a non-empty message, an IIdea, "
                "or a non-empty list."
            )
        return [TextIdea(role="user", content=message)]
    if isinstance(message, IIdea):
        return [_validate_idea(message)]
    if isinstance(message, list):
        if not message:
            raise ValueError(
                "whirl() needs something to think about — got an empty list. "
                "Pass at least one message string or IIdea."
            )
        out = []
        for m in message:
            if isinstance(m, IIdea):
                out.append(_validate_idea(m))
            elif isinstance(m, str):
                if not m.strip():
                    raise ValueError(
                        "whirl() got an empty/whitespace-only string inside "
                        "the message list. Pass non-empty strings or Ideas."
                    )
                out.append(TextIdea(role="user", content=m))
            else:
                raise TypeError(f"whirl(): unexpected message item type {type(m).__name__}")
        return out
    raise TypeError(f"whirl(): unexpected message type {type(message).__name__}")
