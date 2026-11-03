"""Subprocess side of the interrupted-tool-call regression test.

Not a test module (pytest collects only `test_*.py`). `tests/test_recovery.py`
runs each phase below in a FRESH interpreter, so the Dervish is rebuilt from
storage exactly as a restarted application would rebuild it:

    python -c "import recovery_agent, sys; ..." <phase> <data_dir> [args...]

The module is always imported by name — never run as `__main__` — so the
Dervish's qualified agent_type, and with it the Tablet scope on disk, is
`recovery_agent.Cashier` in every process, the test process included.

The model is mocked in every phase; nothing touches the network. Each phase
prints one JSON object on stdout (rumi logs to stderr).
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

from rumi import Dervish, HeartConfig, RecoveryNeededError, Voice, World, tool

AGENT_ID = "till-1"
CALL_ID = "call_pay_1"
CRASH_EXIT_CODE = 17
PAYMENT_ARGS = {"order_id": "A-1", "amount": 42}
LEDGER_ENV = "RECOVERY_LEDGER"
CRASH_ENV = "RECOVERY_CRASH_AFTER_COMMIT"


class Cashier(Dervish):
    heart_config = HeartConfig(voices=[Voice(model_name="mock/cashier")])


@tool(["Cashier"])
def record_payment(order_id: str, amount: int) -> dict:
    """Record a payment in the ledger."""
    conn = sqlite3.connect(os.environ[LEDGER_ENV])
    try:
        conn.execute(
            "INSERT INTO payments (order_id, amount) VALUES (?, ?)", (order_id, amount)
        )
        conn.commit()
    finally:
        conn.close()
    if os.environ.get(CRASH_ENV) == "1":
        # The effect is committed. Die now — no exception for rumi to catch
        # and save as an error result, no cleanup, no result returned.
        os._exit(CRASH_EXIT_CODE)
    return {"order_id": order_id, "amount": amount, "status": "recorded"}


# ---- helpers shared with the test process ----


def create_ledger(path: str | Path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE payments (order_id TEXT NOT NULL, amount INTEGER NOT NULL)")
        conn.commit()
    finally:
        conn.close()


def ledger_rows(path: str | Path) -> list[tuple[str, int]]:
    conn = sqlite3.connect(str(path))
    try:
        return conn.execute("SELECT order_id, amount FROM payments ORDER BY rowid").fetchall()
    finally:
        conn.close()


def open_agent(data_dir: str | Path) -> Cashier:
    """Construct the agent from storage, as any restarted process would."""
    world = World(openrouter_api_key="sk-test", data_dir=data_dir)
    return Cashier(world=world, agent_id=AGENT_ID)


def history_snapshot(agent: Dervish) -> dict[str, Any]:
    """Every saved history entry, as stored."""
    tablet = agent.tablet
    return {k: tablet.get(k) for k in sorted(tablet.list_keys("history/"))}


def _tool_request_response() -> dict:
    return {"choices": [{"message": {"content": None, "tool_calls": [{
        "id": CALL_ID, "type": "function",
        "function": {"name": "record_payment", "arguments": json.dumps(PAYMENT_ARGS)},
    }]}}]}


def _text_response(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


# ---- phases ----


def phase_crash(data_dir: str) -> dict:
    agent = open_agent(data_dir)
    with patch("rumi.heart.openrouter.OpenRouterHeart.call",
               return_value=_tool_request_response()):
        agent.whirl("Charge order A-1 42 dollars.")
    return {"error": "the tool returned; this worker should have died inside it"}


def phase_inspect(data_dir: str) -> dict:
    agent = open_agent(data_dir)
    pending = agent.pending_tool_calls()
    snapshot, idea_ids = history_snapshot(agent), [i.id for i in agent.ideas]
    error = None
    with patch("rumi.heart.openrouter.OpenRouterHeart.call") as model, \
         patch.object(Cashier, "_on_whirl_start_impl") as start_hook:
        try:
            agent.whirl("Did the payment go through?")
        except RecoveryNeededError as exc:
            error = {"type": type(exc).__name__, "pending": exc.pending,
                     "recoverable": exc.recoverable, "message": str(exc)}
    return {
        "pending": pending,
        "error": error,
        "model_calls": model.call_count,
        "whirl_start_hooks": start_hook.call_count,
        "history_unchanged": (history_snapshot(agent) == snapshot
                              and [i.id for i in agent.ideas] == idea_ids),
    }


def phase_record(data_dir: str, request_idea_id: str, result_json: str) -> dict:
    agent = open_agent(data_dir)
    keys_before = set(agent.tablet.list_keys("history/"))
    pending_before = agent.pending_tool_calls()
    saved = agent.record_recovered_result(request_idea_id, CALL_ID, json.loads(result_json))
    return {
        "pending_before": pending_before,
        "saved": saved.to_dict(),
        "keys_added": sorted(set(agent.tablet.list_keys("history/")) - keys_before),
        "pending_after": agent.pending_tool_calls(),
    }


def phase_whirl(data_dir: str) -> dict:
    agent = open_agent(data_dir)
    sent: list[list[dict]] = []

    def model(voice, messages, **kwargs):
        sent.append(messages)
        return _text_response("Payment A-1 for 42 is recorded.")

    with patch("rumi.heart.openrouter.OpenRouterHeart.call", side_effect=model):
        final = agent.whirl("Did the payment go through?")
    return {
        "content": final.content,
        "model_calls": len(sent),
        "messages": sent[0] if sent else None,
        "pending": agent.pending_tool_calls(),
    }


PHASES = {
    "crash": phase_crash,
    "inspect": phase_inspect,
    "record": phase_record,
    "whirl": phase_whirl,
}


def main(argv: list[str]) -> int:
    phase, *args = argv
    print(json.dumps(PHASES[phase](*args)))
    return 0
