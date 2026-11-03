"""Recovering a tool call interrupted before its result was saved.

`Dervish.pending_tool_calls()`, `Dervish.record_recovered_result(...)` and the
`whirl()` guard (`RecoveryNeededError`). All offline: the model is mocked.

Most tests stand in for "the process died inside the tool" with a
BaseException raised from the tool — it escapes the tool loop's
`except Exception`, so no result is saved, leaving the same history a real
crash leaves. `test_interrupted_tool_call_recovers_across_real_processes`
does the real thing: a worker process commits a SQLite row and `os._exit`s
inside the tool, and fresh processes recover it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

import rumi
from rumi import (
    Dervish,
    HeartConfig,
    RecoveryConflictError,
    RecoveryError,
    RecoveryNeededError,
    TextIdea,
    ToolCallNotFoundError,
    ToolRequestIdea,
    ToolResultIdea,
    UnsupportedRecoveryError,
    Voice,
    World,
    tool,
)
from rumi.agent import recovery
from rumi.agent import registry as _multiton_mod
from rumi.tablet.filesystem import FilesystemTablet
from rumi.tools.decorator import _reset_for_tests as _reset_tools

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
import recovery_agent  # noqa: E402  (the subprocess worker; see its docstring)

HEART_CALL = "rumi.heart.openrouter.OpenRouterHeart.call"


@pytest.fixture(autouse=True)
def _reset_all():
    _reset_tools()
    _multiton_mod._reset_for_tests()
    yield
    _reset_tools()
    _multiton_mod._reset_for_tests()


@pytest.fixture
def world(tmp_path: Path) -> World:
    return World(openrouter_api_key="sk-test", data_dir=tmp_path)


class Clerk(Dervish):
    heart_config = HeartConfig(voices=[Voice(model_name="mock/clerk")])


class _ProcessStopped(BaseException):
    """Stands in for the process dying inside a tool (see module docstring)."""


@pytest.fixture
def ledger():
    """Clerk's tools. `rows` records every committed effect; set `stop` to
    make a tool stop the 'process' right after its effect commits."""
    state: dict = {"rows": [], "stop": False}

    @tool(["Clerk"])
    def charge(order_id: str, amount: int) -> dict:
        """Charge an order."""
        state["rows"].append(("charge", order_id, amount))
        if state["stop"]:
            raise _ProcessStopped("stopped after the charge committed")
        return {"order_id": order_id, "amount": amount, "status": "charged"}

    @tool(["Clerk"])
    def refund(order_id: str) -> str:
        """Refund an order."""
        state["rows"].append(("refund", order_id))
        if state["stop"]:
            raise _ProcessStopped("stopped after the refund committed")
        return f"refunded {order_id}"

    return state


# ---- helpers ----


def _call(call_id, name, args) -> dict:
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def _tool_calls(*calls) -> dict:
    return {"choices": [{"message": {"content": None, "tool_calls": [_call(*c) for c in calls]}}]}


def _text(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


CHARGE = ("c1", "charge", {"order_id": "A-1", "amount": 42})
REFUND = ("c2", "refund", {"order_id": "A-0"})


def _interrupt(agent: Dervish, ledger: dict, *calls) -> ToolRequestIdea:
    """Drive a whirl whose tool cycle is cut off after the first tool acts."""
    ledger["stop"] = True
    try:
        with patch(HEART_CALL, return_value=_tool_calls(*calls)):
            with pytest.raises(_ProcessStopped):
                agent.whirl("charge order A-1")
    finally:
        ledger["stop"] = False
    request = agent.ideas[-1]
    assert isinstance(request, ToolRequestIdea)
    return request


def _reopen(world: World, agent_id: str) -> Clerk:
    """A fresh instance rebuilt from the Tablet, as after a restart."""
    _multiton_mod._reset_for_tests()
    return Clerk(world=world, agent_id=agent_id)


def _saved(agent: Dervish) -> dict:
    """The saved history exactly as stored, plus the in-memory id order."""
    t = agent.tablet
    return {"disk": {k: t.get(k) for k in sorted(t.list_keys("history/"))},
            "memory": [i.id for i in agent.ideas]}


def _results_on_disk(agent: Dervish, call_id: str) -> list[dict]:
    return [v for v in _saved(agent)["disk"].values()
            if v["kind"] == "tool_result" and v["tool_call_id"] == call_id]


def _pending_for(request: ToolRequestIdea, call=CHARGE) -> dict:
    return {"request_idea_id": request.id, "tool_call_id": call[0],
            "tool_name": call[1], "arguments": call[2]}


# ---- detection ----


def test_completed_runs_have_no_pending_calls(world, ledger):
    """Single-tool, multi-call and repeated-id ("c1" twice) runs are complete."""
    agent = Clerk(world=world, agent_id="complete")
    script = iter([
        _tool_calls(CHARGE), _text("charged"),                  # single call
        _tool_calls(CHARGE, REFUND), _text("both done"),        # two calls, one request
        _tool_calls(CHARGE), _tool_calls(CHARGE), _text("x2"),  # id "c1" reused
    ])
    with patch(HEART_CALL, side_effect=lambda *a, **k: next(script)):
        agent.whirl("one")
        agent.whirl("two")
        agent.whirl("three")
    assert agent.pending_tool_calls() == []
    assert _reopen(world, "complete").pending_tool_calls() == []

    agent = _reopen(world, "complete")
    with patch(HEART_CALL, return_value=_text("still fine")) as model:
        assert agent.whirl("four").content == "still fine"
    assert model.call_count == 1


def test_pending_tool_call_after_interruption(world, ledger):
    agent = Clerk(world=world, agent_id="till")
    script = iter([_tool_calls(CHARGE), _text("charged")])
    with patch(HEART_CALL, side_effect=lambda *a, **k: next(script)):
        agent.whirl("charge A-1")  # a completed cycle that already answered "c1"
    request = _interrupt(agent, ledger, CHARGE)  # same call id, never answered
    agent = _reopen(world, "till")

    pending = agent.pending_tool_calls()
    assert pending == [_pending_for(request)]
    assert json.loads(json.dumps(pending)) == pending  # JSON-serializable as-is
    assert ledger["rows"] == [("charge", "A-1", 42)] * 2


def test_pending_tool_calls_is_read_only(world, ledger):
    _interrupt(Clerk(world=world, agent_id="ro"), ledger, CHARGE)
    agent = _reopen(world, "ro")
    before = _saved(agent)

    with patch.object(agent.tablet, "set", side_effect=AssertionError("write")), \
         patch.object(agent.tablet, "delete", side_effect=AssertionError("delete")), \
         patch(HEART_CALL, side_effect=AssertionError("model call")):
        first = agent.pending_tool_calls()
        first[0]["arguments"]["amount"] = 999  # callers own the returned dicts
        first[0]["tool_call_id"] = "changed"
        second = agent.pending_tool_calls()

    assert second[0]["arguments"] == {"order_id": "A-1", "amount": 42}
    assert second[0]["tool_call_id"] == "c1"
    assert _saved(agent) == before
    assert ledger["rows"] == [("charge", "A-1", 42)]  # no tool execution


def test_whirl_refuses_unfinished_history_before_saving_input(world, ledger):
    first = Clerk(world=world, agent_id="guard")
    request = _interrupt(first, ledger, CHARGE)

    # Same instance (e.g. after Ctrl-C) and a restarted one both refuse.
    for agent in (first, _reopen(world, "guard")):
        before = _saved(agent)
        with patch(HEART_CALL) as model, \
             patch.object(Clerk, "_on_whirl_start_impl") as start_hook, \
             pytest.raises(RecoveryNeededError) as info:
            agent.whirl("did it go through?")
        model.assert_not_called()
        start_hook.assert_not_called()
        assert _saved(agent) == before  # input not saved, nothing else written

        err = info.value
        assert isinstance(err, RecoveryError)
        assert err.pending == [_pending_for(request)] == agent.pending_tool_calls()
        assert err.recoverable is True and err.reason is None
        assert (err.agent_type, err.agent_id) == (agent.agent_type, "guard")
        assert request.id in str(err) and "record_recovered_result" in str(err)

    assert rumi.RecoveryNeededError is RecoveryNeededError
    assert ledger["rows"] == [("charge", "A-1", 42)]  # the tool never re-ran


# ---- unsupported / ambiguous shapes ----


def _shape_multi_call_unanswered(agent, ledger):
    request = _interrupt(agent, ledger, CHARGE, REFUND)
    return recovery.MULTI_CALL_REQUEST, (request.id, "c1")


def _shape_multi_call_partially_answered(agent, ledger):
    request = ToolRequestIdea(content=[_call(*CHARGE), _call(*REFUND)])
    agent._append_idea(TextIdea(role="user", content="charge and refund"))
    agent._append_idea(request)
    agent._append_idea(ToolResultIdea(tool_call_id="c1", name="charge", content="charged"))
    return recovery.MULTI_CALL_REQUEST, (request.id, "c2")


def _shape_history_continues(agent, ledger):
    request = _interrupt(agent, ledger, CHARGE)
    # Only reachable from a crash before this guard existed (or a failed
    # rollback): the conversation went on past the unanswered request.
    agent._append_idea(TextIdea(role="user", content="hello again"))
    return recovery.HISTORY_CONTINUES, (request.id, "c1")


def _shape_multiple_unfinished_requests(agent, ledger):
    first = ToolRequestIdea(content=[_call("a", "charge", {"order_id": "A", "amount": 1})])
    agent._append_idea(first)
    agent._append_idea(TextIdea(role="user", content="next"))
    agent._append_idea(ToolRequestIdea(content=[_call("b", "charge", {"order_id": "B", "amount": 2})]))
    return recovery.MULTIPLE_UNFINISHED_REQUESTS, (first.id, "a")


def _shape_call_without_id(agent, ledger):
    agent._append_idea(ToolRequestIdea(content=[
        {"type": "function", "function": {"name": "charge", "arguments": "{}"}}]))
    return recovery.INVALID_TOOL_CALL, None


def _shape_unparseable_arguments(agent, ledger):
    request = ToolRequestIdea(content=[
        {"id": "c1", "type": "function", "function": {"name": "charge", "arguments": "{oops"}}])
    agent._append_idea(request)
    return recovery.INVALID_TOOL_CALL, (request.id, "c1")


@pytest.mark.parametrize("shape", [
    _shape_multi_call_unanswered,
    _shape_multi_call_partially_answered,
    _shape_history_continues,
    _shape_multiple_unfinished_requests,
    _shape_call_without_id,
    _shape_unparseable_arguments,
], ids=lambda f: f.__name__.removeprefix("_shape_"))
def test_unsupported_shapes_are_reported_not_guessed(world, ledger, shape):
    agent_id = shape.__name__.strip("_")
    reason, target = shape(Clerk(world=world, agent_id=agent_id), ledger)
    agent = _reopen(world, agent_id)
    before = _saved(agent)

    with pytest.raises(UnsupportedRecoveryError) as listed:
        agent.pending_tool_calls()
    assert listed.value.reason == reason and listed.value.calls

    with patch(HEART_CALL) as model, pytest.raises(RecoveryNeededError) as refused:
        agent.whirl("go on")
    model.assert_not_called()
    assert refused.value.reason == reason and refused.value.recoverable is False
    assert refused.value.pending == listed.value.calls

    if target is not None:
        with pytest.raises(UnsupportedRecoveryError) as rejected:
            agent.record_recovered_result(*target, "done")
        assert rejected.value.reason == reason

    assert _saved(agent) == before


# ---- recording ----


def test_record_saves_result_and_whirl_continues(world, ledger):
    request = _interrupt(Clerk(world=world, agent_id="rec"), ledger, CHARGE)
    agent = _reopen(world, "rec")
    confirmed = {"order_id": "A-1", "amount": 42, "status": "charged"}

    saved = agent.record_recovered_result(request.id, "c1", confirmed)

    assert isinstance(saved, ToolResultIdea)
    assert (saved.tool_call_id, saved.name, saved.role) == ("c1", "charge", "tool")
    assert saved.content == json.dumps(confirmed, ensure_ascii=False)
    assert saved.metadata == {"recovered": True}
    assert agent.tablet.get(f"history/{saved.id}") == saved.to_dict()  # on disk
    assert agent.ideas[-1] is saved                                     # in memory
    assert agent.pending_tool_calls() == []
    assert ledger["rows"] == [("charge", "A-1", 42)]  # recorded, not re-run

    sent = []
    with patch(HEART_CALL, side_effect=lambda voice, messages, **kw:
               sent.append(messages) or _text("A-1 is charged.")):
        assert _reopen(world, "rec").whirl("did it go through?").content == "A-1 is charged."
    assert [m["role"] for m in sent[0]] == ["user", "assistant", "tool", "user"]
    assert sent[0][1]["tool_calls"][0]["id"] == "c1"
    assert sent[0][2] == {"role": "tool", "tool_call_id": "c1",
                          "content": saved.content, "name": "charge"}


def test_record_string_result_verbatim(world, ledger):
    request = _interrupt(Clerk(world=world, agent_id="verbatim"), ledger, CHARGE)
    agent = _reopen(world, "verbatim")
    text = "ERROR: no charge found for A-1 — the tool never took effect"
    assert agent.record_recovered_result(request.id, "c1", text).content == text


def test_record_is_idempotent_across_restarts(world, ledger):
    request = _interrupt(Clerk(world=world, agent_id="idem"), ledger, CHARGE)
    agent = _reopen(world, "idem")
    saved = agent.record_recovered_result(request.id, "c1", {"status": "charged", "amount": 42})
    keys = set(agent.tablet.list_keys("history/"))

    # Same instance; then fresh instances, with key order / formatting changed.
    again = agent.record_recovered_result(request.id, "c1", {"status": "charged", "amount": 42})
    assert again is saved
    for same in ({"amount": 42, "status": "charged"}, '{"amount":42,"status":"charged"}'):
        agent = _reopen(world, "idem")
        assert agent.pending_tool_calls() == []  # no longer pending, yet accepted
        assert agent.record_recovered_result(request.id, "c1", same).id == saved.id

    assert set(agent.tablet.list_keys("history/")) == keys
    assert len(_results_on_disk(agent, "c1")) == 1


def test_record_conflicting_result_raises(world, ledger):
    request = _interrupt(Clerk(world=world, agent_id="conflict"), ledger, CHARGE)
    saved = _reopen(world, "conflict").record_recovered_result(request.id, "c1", {"status": "charged"})

    for agent in (Clerk(world=world, agent_id="conflict"), _reopen(world, "conflict")):
        before = _saved(agent)
        for different in ({"status": "failed"}, "charged", {"status": "charged", "amount": 42}):
            with pytest.raises(RecoveryConflictError) as info:
                agent.record_recovered_result(request.id, "c1", different)
            assert info.value.existing.id == saved.id
            assert info.value.supplied == recovery._normalize_result(different)
        assert _saved(agent) == before


def test_record_against_a_normally_completed_call(world, ledger):
    """A call that completed normally: same outcome → its result; else conflict."""
    agent = Clerk(world=world, agent_id="normal")
    script = iter([_tool_calls(CHARGE), _text("charged")])
    with patch(HEART_CALL, side_effect=lambda *a, **k: next(script)):
        agent.whirl("charge A-1")
    request = next(i for i in agent.ideas if isinstance(i, ToolRequestIdea))
    normal = next(i for i in agent.ideas if isinstance(i, ToolResultIdea))
    before = _saved(agent)

    same = {"status": "charged", "amount": 42, "order_id": "A-1"}  # other key order
    assert agent.record_recovered_result(request.id, "c1", same) is normal
    with pytest.raises(RecoveryConflictError):
        agent.record_recovered_result(request.id, "c1", {"status": "failed"})
    assert _saved(agent) == before


def test_record_rejects_invalid_identifiers(world, ledger):
    agent = Clerk(world=world, agent_id="ids")
    script = iter([_tool_calls(("c-old", "refund", {"order_id": "Z"})), _text("refunded")])
    with patch(HEART_CALL, side_effect=lambda *a, **k: next(script)):
        agent.whirl("refund Z")
    old_request = next(i for i in agent.ideas if isinstance(i, ToolRequestIdea))
    request = _interrupt(agent, ledger, CHARGE)
    agent = _reopen(world, "ids")
    user_text = next(i for i in agent.ideas if isinstance(i, TextIdea))
    before, pending = _saved(agent), agent.pending_tool_calls()

    bad = [
        ("no-such-idea", "c1", "no saved idea"),
        (user_text.id, "c1", "not a tool request"),
        (request.id, "c9", "has no tool call 'c9'"),
        (request.id, "c-old", old_request.id),   # call of a different request
        (old_request.id, "c1", "has no tool call 'c1'"),
        ("", "", "no saved idea"),
    ]
    for request_idea_id, call_id, message in bad:
        with pytest.raises(ToolCallNotFoundError, match=message) as info:
            agent.record_recovered_result(request_idea_id, call_id, "done")
        assert (info.value.request_idea_id, info.value.tool_call_id) == (request_idea_id, call_id)
    for request_idea_id, call_id in ((None, "c1"), (request.id, 1)):
        with pytest.raises(TypeError, match="ids must be strings"):
            agent.record_recovered_result(request_idea_id, call_id, "done")

    assert _saved(agent) == before
    assert agent.pending_tool_calls() == pending


@pytest.mark.parametrize("result, error", [
    (object(), TypeError),
    (b"bytes", TypeError),
    ({1, 2}, TypeError),
    (float("nan"), ValueError),
    ({"amount": float("inf")}, ValueError),
])
def test_record_rejects_non_json_results(world, ledger, result, error):
    request = _interrupt(Clerk(world=world, agent_id="json"), ledger, CHARGE)
    agent = _reopen(world, "json")
    before = _saved(agent)
    with pytest.raises(error):
        agent.record_recovered_result(request.id, "c1", result)
    assert _saved(agent) == before
    assert agent.pending_tool_calls() == [_pending_for(request)]


def test_record_persistence_failure_leaves_history_unchanged(world, ledger):
    request = _interrupt(Clerk(world=world, agent_id="disk"), ledger, CHARGE)
    agent = _reopen(world, "disk")
    before = _saved(agent)

    with patch.object(agent.tablet, "set", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            agent.record_recovered_result(request.id, "c1", "charged")

    assert _saved(agent) == before  # neither disk nor memory moved
    assert agent.pending_tool_calls() == [_pending_for(request)]
    with patch(HEART_CALL) as model, pytest.raises(RecoveryNeededError):
        agent.whirl("well?")
    model.assert_not_called()

    saved = agent.record_recovered_result(request.id, "c1", "charged")  # retry
    assert _reopen(world, "disk").ideas[-1].id == saved.id
    assert len(_results_on_disk(agent, "c1")) == 1


def test_record_refuses_a_stale_instance(world, ledger, tmp_path):
    """Instances are cached per World. One loaded before another writer
    recorded the result must not write a second result from its stale view."""
    request = _interrupt(Clerk(world=world, agent_id="stale"), ledger, CHARGE)
    console = Clerk(world=World(openrouter_api_key="sk-test", data_dir=tmp_path),
                    agent_id="stale")
    saved = Clerk(world=world, agent_id="stale").record_recovered_result(request.id, "c1", "charged")

    with pytest.raises(UnsupportedRecoveryError) as info:
        console.record_recovered_result(request.id, "c1", "charged")
    assert info.value.reason == recovery.STALE_HISTORY
    assert len(_results_on_disk(console, "c1")) == 1

    fresh = Clerk(world=World(openrouter_api_key="sk-test", data_dir=tmp_path), agent_id="stale")
    assert fresh.record_recovered_result(request.id, "c1", "charged").id == saved.id


def test_recovered_result_replays_after_its_request_when_clock_is_behind(world):
    """The result is written by another process than its request: `seq`
    restarts per process and the clock can step back. It must still replay
    right after its request."""
    agent = Clerk(world=world, agent_id="clock")
    agent._append_idea(TextIdea(role="user", content="charge A-1"))
    request = ToolRequestIdea(content=[_call(*CHARGE)], timestamp=time.time() + 3600, seq=10**9)
    agent._append_idea(request)

    saved = _reopen(world, "clock").record_recovered_result(request.id, "c1", "charged")
    assert (saved.timestamp, saved.seq) > (request.timestamp, request.seq)
    replayed = _reopen(world, "clock")
    assert [i.id for i in replayed.ideas][-2:] == [request.id, saved.id]
    assert replayed.pending_tool_calls() == []


# ---- the real interruption: separate processes, os._exit inside the tool ----


def _run_phase(phase: str, *args, env: dict, expect_exit: int = 0) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys, recovery_agent; sys.exit(recovery_agent.main(sys.argv[1:]))",
         phase, *map(str, args)],
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == expect_exit, (
        f"phase {phase!r} exited {proc.returncode}, expected {expect_exit}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    return proc


def _phase_json(phase: str, *args, env: dict) -> dict:
    return json.loads(_run_phase(phase, *args, env=env).stdout)


def _files(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_interrupted_tool_call_recovers_across_real_processes(tmp_path):
    data_dir, ledger = tmp_path / "data", tmp_path / "ledger.sqlite3"
    recovery_agent.create_ledger(ledger)
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(filter(None, [str(TESTS_DIR), os.environ.get("PYTHONPATH")])),
        recovery_agent.LEDGER_ENV: str(ledger),
    }
    env.pop(recovery_agent.CRASH_ENV, None)
    tablet = FilesystemTablet(data_dir, "recovery_agent.Cashier", recovery_agent.AGENT_ID)

    def saved_entries() -> list[dict]:
        return [tablet.get(k) for k in tablet.list_keys("history/")]

    # 1-2. The tool commits a row, then the worker dies before rumi saves the result.
    _run_phase("crash", data_dir, env={**env, recovery_agent.CRASH_ENV: "1"},
               expect_exit=recovery_agent.CRASH_EXIT_CODE)
    assert recovery_agent.ledger_rows(ledger) == [("A-1", 42)]
    assert sorted(e["kind"] for e in saved_entries()) == ["text", "tool_request"]
    request = next(e for e in saved_entries() if e["kind"] == "tool_request")
    expected = {"request_idea_id": request["id"], "tool_call_id": recovery_agent.CALL_ID,
                "tool_name": "record_payment", "arguments": recovery_agent.PAYMENT_ARGS}

    # 3-5. A fresh process finds the call; whirl() refuses without a model call.
    files_before = _files(data_dir)
    out = _phase_json("inspect", data_dir, env=env)
    assert out["pending"] == [expected]
    assert out["error"] is not None, "whirl() should have raised RecoveryNeededError"
    assert out["error"]["type"] == "RecoveryNeededError"
    assert out["error"]["pending"] == [expected] and out["error"]["recoverable"] is True
    assert out["model_calls"] == 0 and out["whirl_start_hooks"] == 0
    assert out["history_unchanged"] is True
    assert _files(data_dir) == files_before  # byte-for-byte

    # 6. Confirm the effect in the ledger; record it through the API (this
    #    test process is a fresh one too).
    [(order_id, amount)] = recovery_agent.ledger_rows(ledger)
    confirmed = {"order_id": order_id, "amount": amount, "status": "recorded"}
    agent = recovery_agent.open_agent(data_dir)
    [pending] = agent.pending_tool_calls()
    assert pending == expected
    saved = agent.record_recovered_result(pending["request_idea_id"], pending["tool_call_id"], confirmed)
    assert isinstance(saved, ToolResultIdea)
    assert (saved.tool_call_id, saved.name) == (recovery_agent.CALL_ID, "record_payment")
    assert json.loads(saved.content) == confirmed
    assert agent.pending_tool_calls() == []

    # 7. Another fresh process submits the same result again.
    out = _phase_json("record", data_dir, request["id"], json.dumps(confirmed), env=env)
    assert out["pending_before"] == [] and out["pending_after"] == []
    assert out["saved"]["id"] == saved.id and out["keys_added"] == []

    # 8. One external effect, one saved result, nothing pending.
    assert recovery_agent.ledger_rows(ledger) == [("A-1", 42)]
    results = [e for e in saved_entries() if e["kind"] == "tool_result"]
    assert [(r["id"], r["tool_call_id"]) for r in results] == [(saved.id, recovery_agent.CALL_ID)]

    # 9. A normal invocation in yet another fresh process.
    out = _phase_json("whirl", data_dir, env=env)
    assert out["content"] == "Payment A-1 for 42 is recorded."
    assert out["model_calls"] == 1 and out["pending"] == []
    messages = out["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "user"]
    assert messages[1]["tool_calls"][0]["id"] == recovery_agent.CALL_ID
    assert messages[2] == {"role": "tool", "tool_call_id": recovery_agent.CALL_ID,
                           "content": saved.content, "name": "record_payment"}
    assert recovery_agent.ledger_rows(ledger) == [("A-1", 42)]
