"""Recovering a tool call that was interrupted before its result was saved.

The whirl pipeline saves a `ToolRequestIdea`, runs the tool, then saves one
`ToolResultIdea` per call. If the process dies in between — after the tool
acted on the outside world, before its result reached the Tablet — the saved
history ends on a tool request nobody answered. Re-running the tool could
repeat its effect, and sending that history to a model is an error
(OpenAI-format providers reject an assistant `tool_calls` message that has no
matching tool messages).

Three `Dervish` methods handle that state; this module holds their pieces:

- `pending_tool_calls()` — read-only: which tool call is unfinished?
- `record_recovered_result(request_idea_id, tool_call_id, result)` — save an
  outcome the caller has confirmed in the external system, without running
  the tool again.
- `whirl()` — raises `RecoveryNeededError` while a call is unfinished, before
  saving the new input or calling the model.

Supported shape (one writer per agent): exactly ONE unfinished call, in a
request that holds exactly one tool call, and that request is the last idea
in history — what an interruption during a single-call tool cycle leaves.
Every other unfinished shape raises `UnsupportedRecoveryError` with a
`reason` code instead of being guessed at:

- `multiple_unfinished_requests` — more than one request lacks results.
- `multi_call_request` — the unfinished request holds several tool calls
  (answered partially or not at all; there is no record of which ran).
- `history_continues` — later ideas follow the unfinished request, so a
  result appended now would not sit next to its request.
- `invalid_tool_call` — the stored call has no id or tool name, its id is
  duplicated, or its arguments are not a JSON object.
- `duplicate_results` — the call already has more than one saved result.
- `stale_history` — this instance's history no longer matches the saved
  history (another writer changed it); reconstruct the agent and retry.

Pending-call shape (plain, JSON-serializable dict — see `PendingToolCall`)::

    {"request_idea_id": "4f0c…",            # id of the saved ToolRequestIdea
     "tool_call_id":    "call_pay_1",       # the provider's id for the call
     "tool_name":       "record_payment",
     "arguments":       {"order_id": "A-1", "amount": 42}}   # decoded JSON object

`arguments` is decoded the way the tool loop decoded it before calling the
tool (a missing or blank arguments string means `{}`). In the `calls` of an
`UnsupportedRecoveryError` — diagnostics only — a field that could not be
decoded holds the raw stored value.

Result normalization (`record_recovered_result`):

- a `str` is saved verbatim;
- any other JSON-compatible value (dict, list, int, float, bool, None) is
  saved as `json.dumps(value, ensure_ascii=False)` — the same text the tool
  loop saves when a tool returns that value. NaN/Infinity and non-JSON types
  (bytes, sets, pydantic models, …) are rejected with `ValueError` /
  `TypeError` before anything is looked up or written.

Two results are "the same" when their saved texts are equal, or when both
parse as JSON and are equal after canonical re-serialization (sorted keys, no
insignificant whitespace; `1`, `1.0` and `true` stay distinct). Recording the
same result again returns the saved `ToolResultIdea` and writes nothing; a
different one raises `RecoveryConflictError`.
"""
from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypedDict

from rumi.ideas.base import IIdea
from rumi.ideas.types import ToolRequestIdea, ToolResultIdea

if TYPE_CHECKING:
    from rumi.tablet.backend import ITablet

# `reason` codes carried by UnsupportedRecoveryError / RecoveryNeededError.
MULTIPLE_UNFINISHED_REQUESTS = "multiple_unfinished_requests"
MULTI_CALL_REQUEST = "multi_call_request"
HISTORY_CONTINUES = "history_continues"
INVALID_TOOL_CALL = "invalid_tool_call"
DUPLICATE_RESULTS = "duplicate_results"
STALE_HISTORY = "stale_history"


class PendingToolCall(TypedDict):
    """One unfinished tool call, as returned by `Dervish.pending_tool_calls()`."""

    request_idea_id: str
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]


# ---- Errors ----


class RecoveryError(Exception):
    """Base class for the tool-call recovery errors."""


class RecoveryNeededError(RecoveryError):
    """`whirl()` refused: saved history has an unfinished tool call.

    Raised before the new input is saved and before any model call, so the
    history is exactly as it was.

    Attributes:
        pending: every unfinished call (pending-call dicts, history order).
        reason: None when `record_recovered_result` can resolve `pending`;
            otherwise the `UnsupportedRecoveryError` reason code.
        recoverable: `reason is None`.
        agent_type, agent_id: the Dervish that refused.
    """

    def __init__(
        self,
        message: str,
        *,
        agent_type: str,
        agent_id: str,
        pending: list[dict[str, Any]],
        reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.agent_type = agent_type
        self.agent_id = agent_id
        self.pending = pending
        self.reason = reason

    @property
    def recoverable(self) -> bool:
        return self.reason is None


class UnsupportedRecoveryError(RecoveryError):
    """History has unfinished tool calls outside the supported shape.

    Attributes:
        reason: a reason code (see the module docstring).
        calls: the unfinished calls found, for diagnostics.
    """

    def __init__(
        self,
        reason: str,
        message: str,
        *,
        calls: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.calls = calls or []


class ToolCallNotFoundError(RecoveryError):
    """No saved tool request/tool call matches the given identifiers."""

    def __init__(self, message: str, *, request_idea_id: str, tool_call_id: str) -> None:
        super().__init__(message)
        self.request_idea_id = request_idea_id
        self.tool_call_id = tool_call_id


class RecoveryConflictError(RecoveryError):
    """The call already has a saved result that differs from the one supplied.

    Attributes:
        existing: the saved `ToolResultIdea` (left untouched).
        supplied: the normalized text that was rejected.
    """

    def __init__(
        self,
        message: str,
        *,
        request_idea_id: str,
        tool_call_id: str,
        existing: ToolResultIdea,
        supplied: str,
    ) -> None:
        super().__init__(message)
        self.request_idea_id = request_idea_id
        self.tool_call_id = tool_call_id
        self.existing = existing
        self.supplied = supplied


# ---- History scan ----


@dataclass(frozen=True)
class _Scan:
    """What `_scan_history` found. `reason is None` → supported (0 or 1 call)."""

    unfinished: list[dict[str, Any]] = field(default_factory=list)
    reason: str | None = None
    detail: str = ""


def _decode_arguments(raw: Any) -> tuple[Any, bool]:
    """Decode stored tool-call arguments as the tool loop did before calling.

    Missing/blank → `{}` (the tool loop calls `fn(**{})`); a dict passes
    through; a string must decode to a JSON object. Returns (value, ok); on
    failure `value` is the raw stored value.
    """
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}, True
    if isinstance(raw, dict):
        return copy.deepcopy(raw), True
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except ValueError:
            return raw, False
        if isinstance(value, dict):
            return value, True
    return raw, False


def _describe_call(request: ToolRequestIdea, call: dict) -> tuple[dict[str, Any], str | None]:
    """The pending-call dict for one stored call, plus why it is unusable (or None)."""
    fn = call.get("function")
    fn = fn if isinstance(fn, dict) else {}
    call_id, name = call.get("id"), fn.get("name")
    arguments, args_ok = _decode_arguments(fn.get("arguments"))
    problem = None
    if not (isinstance(call_id, str) and call_id):
        problem = "the tool call has no id"
    elif not (isinstance(name, str) and name):
        problem = "the tool call has no tool name"
    elif not args_ok:
        problem = "the tool call's arguments are not a JSON object"
    described = {
        "request_idea_id": request.id,
        "tool_call_id": call_id,
        "tool_name": name,
        "arguments": arguments,
    }
    return described, problem


def _results_after(ideas: Sequence[IIdea], index: int) -> list[ToolResultIdea]:
    """The contiguous ToolResultIdeas right after `ideas[index]` — its answers.

    Positional on purpose: tool-call ids are only unique within one request
    (mocked and some real providers reuse ids such as "c1"), and a model only
    accepts tool messages directly after the request they answer.
    """
    out: list[ToolResultIdea] = []
    for j in range(index + 1, len(ideas)):  # no slice: this runs on every whirl()
        if not isinstance(ideas[j], ToolResultIdea):
            break
        out.append(ideas[j])
    return out


def _scan_history(ideas: Sequence[IIdea]) -> _Scan:
    """Find unfinished tool calls and classify the history. Pure; no I/O.

    Runs at the top of every whirl(), so the consistent-history path is one
    linear pass that never decodes arguments.
    """
    open_requests: list[tuple[int, ToolRequestIdea, int, list[tuple[dict, str | None]]]] = []
    for i, idea in enumerate(ideas):
        if not isinstance(idea, ToolRequestIdea):
            continue
        answered = {r.tool_call_id for r in _results_after(ideas, i)}
        # `or ""` mirrors the tool loop, which saves results for an id-less
        # call under tool_call_id="" — such a call counts as answered.
        missing = [c for c in idea.content if (c.get("id") or "") not in answered]
        if missing:
            described = [_describe_call(idea, c) for c in missing]
            open_requests.append((i, idea, len(idea.content), described))

    if not open_requests:
        return _Scan()
    unfinished = [d for *_, missing in open_requests for d, _ in missing]
    if len(open_requests) > 1:
        ids = [req.id for _, req, _, _ in open_requests]
        return _Scan(unfinished, MULTIPLE_UNFINISHED_REQUESTS,
                     f"{len(open_requests)} tool requests have unanswered calls ({ids}).")
    index, request, n_calls, missing = open_requests[0]
    if n_calls != 1:
        return _Scan(unfinished, MULTI_CALL_REQUEST,
                     f"tool request {request.id!r} holds {n_calls} tool calls, "
                     f"{len(missing)} without a saved result; only single-call "
                     "requests can be recovered (there is no record of which calls ran).")
    if index != len(ideas) - 1:
        return _Scan(unfinished, HISTORY_CONTINUES,
                     f"tool request {request.id!r} is followed by "
                     f"{len(ideas) - 1 - index} later idea(s); a result saved now "
                     "would not follow its request.")
    _, problem = missing[0]
    if problem:
        return _Scan(unfinished, INVALID_TOOL_CALL, f"tool request {request.id!r}: {problem}.")
    return _Scan(unfinished)


def _needed_message(subject: str, scan: _Scan) -> str:
    if scan.reason is None:
        call = scan.unfinished[0]
        return (
            f"{subject} cannot whirl: its saved history ends with an unfinished "
            f"tool call (request_idea_id={call['request_idea_id']!r}, "
            f"tool_call_id={call['tool_call_id']!r}, tool={call['tool_name']!r}). "
            "The tool may already have acted. Confirm the outcome in the system "
            "it acted on, then call record_recovered_result(request_idea_id, "
            "tool_call_id, result) and whirl again. Nothing was saved and no "
            "model call was made."
        )
    return (
        f"{subject} cannot whirl: its saved history has unfinished tool calls "
        f"that record_recovered_result() cannot resolve [{scan.reason}]: "
        f"{scan.detail} Nothing was saved and no model call was made."
    )


# ---- Results ----


def _normalize_result(result: Any) -> str:
    """Caller-supplied result → the text saved as ToolResultIdea.content."""
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False, allow_nan=False)
    except TypeError as exc:
        raise TypeError(
            "record_recovered_result() takes a str or a JSON-compatible value "
            "(dict, list, int, float, bool, None) as the result; got "
            f"{type(result).__name__} ({exc})."
        ) from exc
    except ValueError as exc:
        raise ValueError(
            f"record_recovered_result(): the result is not valid JSON ({exc})."
        ) from exc


def _canonical_json(text: str) -> str | None:
    try:
        return json.dumps(json.loads(text), ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        return None


def _same_result(saved: str, supplied: str) -> bool:
    if saved == supplied:
        return True
    canonical = _canonical_json(saved)
    return canonical is not None and canonical == _canonical_json(supplied)


def _sort_after(idea: IIdea, previous: IIdea) -> None:
    """Make `idea` replay after `previous` (sort key: timestamp, seq, id).

    A recovered result is written by a different process than its request:
    `seq` restarts per process and the wall clock can step backwards, so the
    default (timestamp, seq) could sort the result before its request.
    """
    if (idea.timestamp, idea.seq) <= (previous.timestamp, previous.seq):
        idea.timestamp = previous.timestamp
        idea.seq = previous.seq + 1


def _check_in_sync(tablet: "ITablet", ideas: Sequence[IIdea], subject: str) -> None:
    """Refuse to write unless this instance's history is the saved history.

    `Dervish` instances are cached per World, so an instance can predate
    another process's writes (e.g. a recovery recorded elsewhere). Writing from
    that stale view could duplicate a result. Entries that replay skips as
    unreadable are ignored here too.
    """
    on_disk = set(tablet.list_keys("history/"))
    loaded = {f"history/{idea.id}" for idea in ideas}
    missing = loaded - on_disk
    extra = [k for k in on_disk - loaded if _is_loadable(tablet.get(k))]
    if missing or extra:
        raise UnsupportedRecoveryError(
            STALE_HISTORY,
            f"{subject}: this instance's history differs from the saved history "
            f"({len(extra)} saved idea(s) not loaded, {len(missing)} loaded idea(s) "
            "not saved) — another writer has changed it. Reconstruct the agent "
            "in a fresh World (or process) and retry. Nothing was written.",
        )


def _is_loadable(raw: Any) -> bool:
    if not raw:
        return False
    try:
        IIdea.from_dict(raw)
    except Exception:
        return False
    return True
