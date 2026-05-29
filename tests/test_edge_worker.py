"""Seam test for the rumi-edge Python worker.

Proves the Go↔Python IPC contract in pure Python (fast iteration) before the Go
daemon exists: start the worker on a background thread, drive it over a loopback
socket with newline-delimited JSON-RPC, and assert invoke / call_tool /
list_tools round-trip. The LLM is mocked, so no network/key is needed.
"""
from __future__ import annotations

import json
import socket
import threading
from unittest.mock import patch

import pytest

import rumi_worker  # importable via pytest pythonpath = [".", "rumi-edge"]


@pytest.fixture(autouse=True)
def _register_agents():
    """Ensure Poet/Sage tools are in the global registry for each test.

    Other test modules reset the tool snapshot in their teardown, so by the
    time this module runs the registry may be empty — reload the reference
    agents to re-register their @tool functions (the registration is
    idempotent on reload).
    """
    import importlib

    from rumi.tools.decorator import _reset_for_tests
    import examples.poet
    import examples.sage

    _reset_for_tests()
    importlib.reload(examples.poet)
    importlib.reload(examples.sage)
    yield


def _rpc(host: str, port: int, method: str, params: dict) -> dict:
    """One newline-delimited JSON-RPC round-trip over a fresh connection."""
    with socket.create_connection((host, port), timeout=10) as s:
        req = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        s.sendall((json.dumps(req) + "\n").encode("utf-8"))
        line = s.makefile("r", encoding="utf-8").readline()
    return json.loads(line)


def _rpc_stream(host: str, port: int, params: dict) -> list[dict]:
    """Drive `invoke_stream`: collect bare NDJSON frames until a terminal frame.

    Unlike `_rpc`, the worker replies with N progress frames then exactly one
    terminal frame (type in {final, error, canceled}) — no JSON-RPC envelope.
    """
    frames: list[dict] = []
    with socket.create_connection((host, port), timeout=15) as s:
        req = {"jsonrpc": "2.0", "id": 1, "method": "invoke_stream", "params": params}
        s.sendall((json.dumps(req) + "\n").encode("utf-8"))
        for raw in s.makefile("r", encoding="utf-8"):
            line = raw.strip()
            if not line:
                continue
            frame = json.loads(line)
            frames.append(frame)
            if frame.get("type") in ("final", "error", "canceled"):
                break
    return frames


_TOOL_REQ = {
    "choices": [{"message": {"content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {
            "name": "enrich_company",
            "arguments": json.dumps({"company_name": "Northwind Analytics"}),
        }},
    ]}}],
    "usage": {"cost": 0.0},
}


@pytest.fixture
def worker_addr(tmp_path, monkeypatch):
    # Isolate tablet writes to a temp dir.
    monkeypatch.setenv("RUMI_DATA_DIR", str(tmp_path))
    server = rumi_worker._Server(("127.0.0.1", 0), rumi_worker.Worker())
    host, port = server.server_address
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        yield host, port
    finally:
        server.shutdown()
        server.server_close()


def test_list_tools(worker_addr):
    host, port = worker_addr
    resp = _rpc(host, port, "list_tools", {"agent_type": "Poet"})
    names = {t["name"] for t in resp["result"]["tools"]}
    assert "find_leads_for_icp" in names
    assert "enrich_company" in names
    # the schema is the OpenAI function dict the daemon will translate to MCP
    one = resp["result"]["tools"][0]
    assert one["schema"]["type"] == "function"


def test_call_tool(worker_addr):
    host, port = worker_addr
    resp = _rpc(
        host,
        port,
        "call_tool",
        {
            "agent_type": "Poet",
            "agent_id": "p1",
            "name": "enrich_company",
            "arguments": {"company_name": "Northwind Analytics"},
        },
    )
    assert resp["result"]["result"]["business"] == "Northwind Analytics"


def test_invoke_with_mocked_llm(worker_addr):
    host, port = worker_addr
    canned = {
        "choices": [{"message": {"content": "I found two that hear the drum."}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cost": 0.0001},
    }
    # The worker runs in a background thread of THIS process, so patching the
    # class method is visible to it for the duration of the synchronous call.
    with patch("rumi.heart.openrouter.OpenRouterHeart.call", return_value=canned):
        resp = _rpc(host, port, "invoke", {"agent_type": "Poet", "agent_id": "p1", "message": "who is ready?"})
    assert resp["result"]["content"] == "I found two that hear the drum."
    assert resp["result"]["idea"]["kind"] == "text"
    assert resp["result"]["metadata"]["cost_usd"] == 0.0001


def test_unknown_method_is_jsonrpc_error(worker_addr):
    host, port = worker_addr
    resp = _rpc(host, port, "nope", {})
    assert resp["error"]["code"] == -32601


def test_unknown_agent_is_jsonrpc_error(worker_addr):
    host, port = worker_addr
    resp = _rpc(host, port, "list_tools", {"agent_type": "DoesNotExist"})
    # get_tools_for returns [] for an unknown agent_type (no error) — that's fine;
    # but resolving an unknown agent for invoke/call_tool IS an error:
    resp2 = _rpc(host, port, "call_tool", {"agent_type": "Ghost", "agent_id": "x", "name": "t"})
    assert resp2["error"]["code"] == -32602


# ---- Phase 2b: streaming + cancel ----


def test_invoke_stream_progress_then_final(worker_addr):
    """invoke_stream emits lifecycle progress frames then one terminal `final`."""
    host, port = worker_addr
    canned = {
        "choices": [{"message": {"content": "I found two that hear the drum."}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cost": 0.0001},
    }
    with patch("rumi.heart.openrouter.OpenRouterHeart.call", return_value=canned):
        frames = _rpc_stream(host, port, {
            "task_id": "s1", "agent_type": "Poet", "agent_id": "p1", "message": "who is ready?",
        })

    assert frames[-1]["type"] == "final"
    assert frames[-1]["content"] == "I found two that hear the drum."
    assert frames[-1]["idea"]["kind"] == "text"
    assert frames[-1]["metadata"]["cost_usd"] == 0.0001
    stages = [f.get("stage") for f in frames if f["type"] == "status"]
    assert "whirl_start" in stages
    assert "llm_call" in stages
    assert "llm_done" in stages


def test_invoke_stream_tool_frame(worker_addr):
    """A tool-request cycle surfaces a `tools` frame naming the called tool."""
    host, port = worker_addr
    text = {"choices": [{"message": {"content": "done"}}], "usage": {"cost": 0.0}}
    seq = [_TOOL_REQ, text]  # iter 1 asks for a tool, iter 2 returns text
    with patch("rumi.heart.openrouter.OpenRouterHeart.call", side_effect=lambda *a, **k: seq.pop(0)):
        frames = _rpc_stream(host, port, {
            "task_id": "s2", "agent_type": "Poet", "agent_id": "p1", "message": "enrich it",
        })

    assert frames[-1]["type"] == "final"
    tool_frames = [f for f in frames if f.get("stage") == "tools"]
    assert tool_frames, f"no tools frame in {frames}"
    assert "enrich_company" in tool_frames[0]["tools"]


def test_invoke_stream_cancel(worker_addr):
    """A `cancel` mid-run stops the agent before its next LLM call → `canceled`.

    The mocked LLM blocks on the first call so the test has a window to cancel
    from a second connection; the first tool cycle still completes (cancel
    lands at the next loop boundary), and the run terminates as canceled.
    """
    host, port = worker_addr
    call_entered = threading.Event()
    release = threading.Event()

    def fake_call(*a, **k):
        call_entered.set()
        release.wait(timeout=5)
        return _TOOL_REQ  # forces a 2nd iteration, whose top sees the cancel

    frames: list[dict] = []

    def run():
        frames.extend(_rpc_stream(host, port, {
            "task_id": "cx", "agent_type": "Poet", "agent_id": "p1", "message": "go",
        }))

    with patch("rumi.heart.openrouter.OpenRouterHeart.call", side_effect=fake_call):
        th = threading.Thread(target=run)
        th.start()
        assert call_entered.wait(timeout=5), "whirl() never reached the LLM call"
        cresp = _rpc(host, port, "cancel", {"task_id": "cx"})
        assert cresp["result"]["found"] is True
        release.set()
        th.join(timeout=10)

    assert frames, "no frames received"
    assert frames[-1]["type"] == "canceled"


def test_cancel_unknown_task_reports_not_found(worker_addr):
    host, port = worker_addr
    resp = _rpc(host, port, "cancel", {"task_id": "nope"})
    assert resp["result"]["found"] is False
    assert resp["result"]["canceled"] is False


def test_hook_wrap_preserves_override_and_restores(tmp_path, monkeypatch):
    """_wrap_hooks emits a frame THEN calls the real override; _restore_hooks
    leaves the instance __dict__ clean so the multiton isn't poisoned."""
    monkeypatch.setenv("RUMI_DATA_DIR", str(tmp_path))
    from rumi import HeartConfig, Voice, World
    from rumi.agent.agent import Dervish

    fired: list[str] = []

    class _Probe(Dervish):
        heart_config = HeartConfig(voices=[Voice(model_name="x", instructions="y")])

        def _on_whirl_start_impl(self, incoming):
            fired.append("real")

    agent = _Probe(world=World(), agent_id="probe1")
    emitted: list[dict] = []
    saved = rumi_worker._wrap_hooks(agent, emitted.append)
    assert "_on_whirl_start_impl" in agent.__dict__  # instance shadow installed

    agent._on_whirl_start_impl([])  # wrapper: emit THEN call the real override
    assert emitted[0]["stage"] == "whirl_start"
    assert fired == ["real"]

    rumi_worker._restore_hooks(agent, saved)
    for h in rumi_worker._HOOKS:
        assert h not in agent.__dict__  # shadow dropped; class method resumes
    agent._on_whirl_start_impl([])  # real override still works post-restore
    assert fired == ["real", "real"]


# ---- Phase 2c: threads / messages-history / resources ----
#
# These read-only methods need on-disk state. We seed it through the REAL
# library (a test-side Poet writing into the same RUMI_DATA_DIR the worker reads,
# set by the worker_addr fixture) so files land exactly where the worker looks —
# under the qualified dir `examples.poet.Poet/{agent_id}/`. The worker constructs
# its OWN agents in its OWN World and replays that same disk, so the two agree.


def _make_poet(agent_id: str = "p1"):
    """A test-side Poet writing into the worker's data dir (via RUMI_DATA_DIR)."""
    import examples.poet
    from rumi import World

    return examples.poet.Poet(world=World(), agent_id=agent_id)


def _whirl(agent, message: str, reply: str) -> None:
    canned = {"choices": [{"message": {"content": reply}}], "usage": {"cost": 0.0}}
    with patch("rumi.heart.openrouter.OpenRouterHeart.call", return_value=canned):
        agent.whirl(message)


def test_history_returns_visible_turns(worker_addr):
    host, port = worker_addr
    _whirl(_make_poet("p1"), "hello there", "first reply")
    resp = _rpc(host, port, "history", {"agent_type": "Poet", "agent_id": "p1"})
    msgs = resp["result"]["messages"]
    assert [(m["role"], m["parts"][0]["text"]) for m in msgs] == [
        ("user", "hello there"),
        ("agent", "first reply"),
    ]
    assert resp["result"]["hasMore"] is False


def test_history_filters_internal_tool_ideas(worker_addr):
    """tool_request/tool_result ideas are not user-visible messages."""
    host, port = worker_addr
    poet = _make_poet("pt")
    seq = [_TOOL_REQ, {"choices": [{"message": {"content": "done"}}], "usage": {"cost": 0.0}}]
    with patch("rumi.heart.openrouter.OpenRouterHeart.call", side_effect=lambda *a, **k: seq.pop(0)):
        poet.whirl("enrich it")
    resp = _rpc(host, port, "history", {"agent_type": "Poet", "agent_id": "pt"})
    # user + final assistant text only — the tool cycle in between is filtered out
    assert [m["role"] for m in resp["result"]["messages"]] == ["user", "agent"]
    assert resp["result"]["messages"][-1]["parts"][0]["text"] == "done"


def test_history_cursor_pages_backward(worker_addr):
    """`before` is an opaque message-id cursor (collision-proof, not a timestamp)."""
    host, port = worker_addr
    poet = _make_poet("pc")
    _whirl(poet, "q1", "a1")
    _whirl(poet, "q2", "a2")  # 4 visible turns: q1, a1, q2, a2

    p1 = _rpc(host, port, "history", {"agent_type": "Poet", "agent_id": "pc", "limit": 2})["result"]
    assert [m["parts"][0]["text"] for m in p1["messages"]] == ["q2", "a2"]
    assert p1["hasMore"] is True and p1["nextCursor"]

    p2 = _rpc(host, port, "history", {
        "agent_type": "Poet", "agent_id": "pc", "limit": 2, "before": p1["nextCursor"],
    })["result"]
    assert [m["parts"][0]["text"] for m in p2["messages"]] == ["q1", "a1"]
    assert p2["hasMore"] is False


def test_list_threads_enumerates(worker_addr):
    host, port = worker_addr
    _whirl(_make_poet("p1"), "hello there", "r1")
    _whirl(_make_poet("p2"), "second convo", "r2")
    resp = _rpc(host, port, "list_threads", {"agent_type": "Poet"})
    threads = {t["threadId"]: t for t in resp["result"]["threads"]}
    assert "Poet/p1" in threads and "Poet/p2" in threads
    assert threads["Poet/p1"]["messageCount"] == 2
    assert threads["Poet/p1"]["firstMessagePreview"] == "hello there"
    assert threads["Poet/p1"]["title"] == "p1"
    assert threads["Poet/p1"]["createdAt"]  # ISO timestamp present


def test_list_threads_unknown_type_is_error(worker_addr):
    host, port = worker_addr
    resp = _rpc(host, port, "list_threads", {"agent_type": "Ghost"})
    assert resp["error"]["code"] == -32602


def test_list_resources_excludes_history(worker_addr):
    host, port = worker_addr
    poet = _make_poet("p1")
    _whirl(poet, "hello there", "r1")  # writes history/ keys
    poet.inscribe("demo.key", {"hello": "world"})  # writes a box key
    resp = _rpc(host, port, "list_resources", {"agent_type": "Poet", "agent_id": "p1"})
    uris = [r["uri"] for r in resp["result"]["resources"]]
    assert "rumi://box/demo.key" in uris
    assert all(not u.startswith("rumi://box/history/") for u in uris)
    one = next(r for r in resp["result"]["resources"] if r["name"] == "demo.key")
    assert one["mimeType"] == "application/json"


def test_read_resource_round_trip(worker_addr):
    host, port = worker_addr
    _make_poet("p1").inscribe("demo.key", {"hello": "world"})
    resp = _rpc(host, port, "read_resource", {
        "agent_type": "Poet", "agent_id": "p1", "uri": "rumi://box/demo.key",
    })
    contents = resp["result"]["contents"]
    assert len(contents) == 1
    assert contents[0]["uri"] == "rumi://box/demo.key"
    assert contents[0]["mimeType"] == "application/json"
    assert json.loads(contents[0]["text"]) == {"hello": "world"}


def test_read_resource_unknown_is_error(worker_addr):
    host, port = worker_addr
    _make_poet("p1")
    resp = _rpc(host, port, "read_resource", {
        "agent_type": "Poet", "agent_id": "p1", "uri": "rumi://box/nope",
    })
    assert resp["error"]["code"] == -32602


def test_read_resource_rejects_history_key(worker_addr):
    """History is served by messages/history, not as a resource."""
    host, port = worker_addr
    _make_poet("p1")
    resp = _rpc(host, port, "read_resource", {
        "agent_type": "Poet", "agent_id": "p1", "uri": "rumi://box/history/abc",
    })
    assert resp["error"]["code"] == -32602
