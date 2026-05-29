"""rumi-edge worker — a loopback-TCP JSON-RPC server that runs a rumi Dervish.

The Go daemon (`rumi-edge`) owns the A2A/MCP protocol and the network edge; this
worker owns the actual Dervish. It constructs one `World`, resolves agents by
`(agent_type, agent_id)`, and runs `whirl()` / `use_tool()` on request. Frames
are newline-delimited JSON-RPC over 127.0.0.1; `whirl` is serialized (one at a
time) by a lock.

Run standalone:
    python rumi-edge/rumi_worker.py [--host 127.0.0.1] [--port 0]
It prints `RUMI_WORKER_LISTENING <host>:<port>` once bound (port 0 = ephemeral),
so a launcher can read the chosen port.
"""
from __future__ import annotations

import argparse
import json
import os
import socketserver
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Make the repo root importable so the reference `examples` package resolves when
# this file is run as a standalone script (its own dir is rumi-edge/, not root).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Import the reference examples so they self-register with the Dervish registry.
# A real deployment would import whichever agents it serves.
try:  # best-effort — botasaurus (Sage's optional dep) imports lazily, so this is safe
    import examples.poet  # noqa: F401
    import examples.sage  # noqa: F401
except Exception:  # pragma: no cover - depends on the deployment's agents
    pass

from rumi import World
from rumi.agent.agent import _AGENT_SUBCLASSES, _qualified_agent_name
from rumi.agent.pipeline import WhirlCancelled
from rumi.tools.decorator import get_tools_for


class _RpcError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


# --- streaming + cancel plumbing ---------------------------------------------
#
# `invoke_stream` runs whirl() while emitting NEUTRAL progress frames (NOT A2A —
# the Go daemon translates them). The 4 lifecycle hooks are the only observation
# seam, so we shadow them on the agent INSTANCE for the duration of one run and
# restore them after. A per-task `threading.Event` (looked up by the daemon's
# task_id) lets a separate `cancel` request cooperatively stop the run via the
# library's `cancel_check`.

_CANCELS: dict[str, threading.Event] = {}
_CANCELS_LOCK = threading.Lock()

_HOOKS = (
    "_on_whirl_start_impl",
    "_on_heart_call_start_impl",
    "_on_heart_call_ended_impl",
    "_on_whirl_ended_impl",
)
_MISSING = object()


def _tool_names(response: dict) -> list[str]:
    """Pull tool-call function names off a raw heart.call() response dict."""
    if not isinstance(response, dict):
        return []
    choices = response.get("choices") or []
    if not choices:
        return []
    msg = choices[0].get("message") or {}
    calls = msg.get("tool_calls") or []
    return [c.get("function", {}).get("name", "") for c in calls if c.get("function")]


def _wrap_hooks(agent, emit) -> dict:
    """Shadow the agent's 4 lifecycle hooks to also emit progress frames.

    The library's `_safe_hook` calls `getattr(agent, name)(*args)`, so an
    instance attribute shadows the class method and is picked up. Each wrapper
    emits its frame THEN calls the original (preserving real subclass overrides
    like Mystic's bookkeeping). Returns the prior instance attrs (or _MISSING)
    so `_restore_hooks` can put things back exactly — the agent is a multiton,
    so a leaked wrapper would persist across runs.
    """
    saved = {h: agent.__dict__.get(h, _MISSING) for h in _HOOKS}

    def chain(orig, frame_fn):
        def wrapped(*args):
            try:
                frame_fn(*args)
            except Exception:  # observation must never break whirl()
                pass
            return orig(*args)
        return wrapped

    def _ended(_payload_in, response):
        usage = (response.get("usage") or {}) if isinstance(response, dict) else {}
        frame = {"type": "status", "stage": "llm_done"}
        if "cost" in usage:
            frame["cost_usd"] = usage["cost"]
        emit(frame)
        names = _tool_names(response)
        if names:
            emit({"type": "status", "stage": "tools", "tools": names})

    agent._on_whirl_start_impl = chain(
        agent._on_whirl_start_impl,
        lambda _ideas: emit({"type": "status", "stage": "whirl_start"}),
    )
    agent._on_heart_call_start_impl = chain(
        agent._on_heart_call_start_impl,
        lambda p: emit({"type": "status", "stage": "llm_call", "model": p.get("model")}),
    )
    agent._on_heart_call_ended_impl = chain(agent._on_heart_call_ended_impl, _ended)
    agent._on_whirl_ended_impl = chain(
        agent._on_whirl_ended_impl,
        lambda _idea: emit({"type": "status", "stage": "whirl_ended"}),
    )
    return saved


def _restore_hooks(agent, saved: dict) -> None:
    """Undo `_wrap_hooks`: drop our shadow (class method resumes) or restore the
    prior instance attr, per the _MISSING sentinel."""
    for h, original in saved.items():
        if original is _MISSING:
            agent.__dict__.pop(h, None)
        else:
            agent.__dict__[h] = original


def _resolve_agent_class(agent_type: str):
    """Resolve a registered Dervish class by qualified or bare name.

    Mirrors `Dervish.get_agent`'s resolution against the global registry so the
    worker can construct any imported Dervish with only a World.
    """
    cls = _AGENT_SUBCLASSES.get(agent_type)
    if cls is not None:
        return cls
    matches = {k: c for k, c in _AGENT_SUBCLASSES.items() if k.rsplit(".", 1)[-1] == agent_type}
    if len(matches) == 1:
        return next(iter(matches.values()))
    if len(matches) > 1:
        raise _RpcError(-32602, f"agent_type {agent_type!r} is ambiguous: {sorted(matches)}")
    raise _RpcError(-32602, f"no Dervish named {agent_type!r} is registered")


# --- Phase 2c read helpers (threads / messages-history / resources) -----------
#
# These back the five read-only convenience methods. They use ONLY public rumi
# surface (`agent.ideas`, `world.tablet_for(...).list_keys()/.get()`,
# `world.data_dir`) — the library is untouched by the edge layer.


def _qualified(agent_type: str) -> str:
    """The on-disk/scope name for an agent_type. `Dervish.__init__` writes its
    Tablet under `_qualified_agent_name(type(self))` (module.ClassName), so any
    direct `tablet_for`/`data_dir` access MUST qualify or it hits the wrong dir.
    Raises _RpcError if the type isn't registered."""
    return _qualified_agent_name(_resolve_agent_class(agent_type))


def _iso(ts: float) -> str:
    """Epoch seconds → RFC3339 UTC with a 'Z' suffix (matches the Go daemon)."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _visible(ideas: list) -> list:
    """The user-visible turns of a history: drop internal tool cycles and empty
    content. (No `system` filter — system messages are injected at build time and
    never persisted, so none ever reach history.)"""
    out = []
    for idea in ideas:
        if getattr(idea, "kind", "text") in ("tool_request", "tool_result"):
            continue
        content = idea.content
        if content is None or (isinstance(content, str) and content == ""):
            continue
        out.append(idea)
    return out


def _to_a2a_message(idea) -> dict:
    """Map one IIdea → an A2A Message (role + parts). assistant→agent; text vs
    data part by idea kind. Post-replay BaseModel/Json content is a plain dict."""
    role = "user" if idea.role == "user" else "agent"
    if getattr(idea, "kind", "text") in ("json", "basemodel"):
        part = {"type": "data", "data": idea.content}
    else:
        content = idea.content
        part = {"type": "text", "text": content if isinstance(content, str) else str(content)}
    return {"role": role, "parts": [part]}


class Worker:
    """Holds one World; runs whirl()/use_tool() for resolved agents."""

    def __init__(self) -> None:
        self._world = World()
        self._whirl_lock = threading.Lock()  # one whirl() at a time

    def _agent(self, params: dict):
        agent_type, agent_id = self._identity(params)
        cls = _resolve_agent_class(agent_type)
        return cls(world=self._world, agent_id=agent_id)

    def handle(self, method: str, params: dict) -> Any:
        # NB: `invoke_stream` is NOT routed here — it needs the connection's
        # emit callback, so `_Handler` drives it directly via `stream_invoke`.
        if method == "invoke":
            return self._invoke(params)
        if method == "call_tool":
            return self._call_tool(params)
        if method == "list_tools":
            return self._list_tools(params)
        if method == "cancel":
            return self._cancel(params)
        if method == "list_threads":
            return self._list_threads(params)
        if method == "history":
            return self._history(params)
        if method == "list_resources":
            return self._list_resources(params)
        if method == "read_resource":
            return self._read_resource(params)
        raise _RpcError(-32601, f"method not found: {method!r}")

    def _invoke(self, params: dict) -> dict:
        agent = self._agent(params)
        if "message" not in params:
            raise _RpcError(-32602, "missing param: 'message'")
        with self._whirl_lock:
            idea = agent.whirl(params["message"])
        return {"idea": idea.to_dict(), "content": idea.content, "metadata": idea.metadata}

    def stream_invoke(self, params: dict, emit) -> dict:
        """Run whirl() while emitting progress frames via `emit`; return the
        single terminal frame ({"type": "final"|"error"|"canceled", ...}).

        The daemon supplies `task_id` so a concurrent `cancel` can find this
        run's Event. Progress frames flow synchronously from the wrapped hooks
        on THIS thread (the handler thread), which keeps them in exact pipeline
        order with no queue. `cancel_check=ev.is_set` makes the library stop
        before its next LLM call; a cancel that lands during the final LLM call
        is caught by the post-whirl `ev.is_set()` re-check (the computed idea is
        discarded — already persisted to the tablet, but never returned).
        """
        agent = self._agent(params)
        if "message" not in params:
            raise _RpcError(-32602, "missing param: 'message'")
        task_id = params.get("task_id") or ""
        ev = threading.Event()
        if task_id:
            with _CANCELS_LOCK:
                _CANCELS[task_id] = ev
        saved = _wrap_hooks(agent, emit)
        try:
            with self._whirl_lock:
                idea = agent.whirl(params["message"], cancel_check=ev.is_set)
            if ev.is_set():
                return {"type": "canceled"}
            return {
                "type": "final",
                "idea": idea.to_dict(),
                "content": idea.content,
                "metadata": idea.metadata,
            }
        except WhirlCancelled:
            return {"type": "canceled"}
        except Exception as e:  # a failed run is a terminal frame, not a transport error
            return {"type": "error", "message": str(e)}
        finally:
            _restore_hooks(agent, saved)
            if task_id:
                with _CANCELS_LOCK:
                    _CANCELS.pop(task_id, None)

    def _cancel(self, params: dict) -> dict:
        task_id = params.get("task_id")
        if not task_id:
            raise _RpcError(-32602, "missing param: 'task_id'")
        with _CANCELS_LOCK:
            ev = _CANCELS.get(task_id)
        if ev is not None:
            ev.set()
            return {"canceled": True, "found": True}
        return {"canceled": False, "found": False}

    def _call_tool(self, params: dict) -> dict:
        agent = self._agent(params)
        if "name" not in params:
            raise _RpcError(-32602, "missing param: 'name'")
        result = agent.use_tool(params["name"], params.get("arguments") or {})
        return {"result": result}

    def _list_tools(self, params: dict) -> dict:
        agent_type = params.get("agent_type")
        if not agent_type:
            raise _RpcError(-32602, "missing param: 'agent_type'")
        tools = get_tools_for(agent_type)  # accepts bare or qualified
        return {"tools": [{"name": t.name, "schema": t.schema} for t in tools]}

    # --- Phase 2c: threads / messages-history / resources --------------------

    def _list_threads(self, params: dict) -> dict:
        """Enumerate the conversations (agent_ids) under one agent_type. A
        'thread' is an (agent_type, agent_id); threadId == the contextId. Each
        thread is built by constructing the agent and reading `.ideas` — the
        SAME replay `_history` uses, so the two never disagree on ordering."""
        agent_type = params.get("agent_type")
        if not agent_type:
            raise _RpcError(-32602, "missing param: 'agent_type'")
        cls = _resolve_agent_class(agent_type)
        base = Path(self._world.data_dir) / _qualified_agent_name(cls)
        threads = []
        if base.is_dir():
            for sub in sorted(base.iterdir()):
                if not sub.is_dir():
                    continue
                agent_id = sub.name
                try:
                    vis = _visible(cls(world=self._world, agent_id=agent_id).ideas)
                except Exception:  # stray dir / unreplayable history — skip it
                    continue
                threads.append({
                    "threadId": f"{agent_type}/{agent_id}",
                    "title": agent_id,
                    "messageCount": len(vis),
                    "createdAt": _iso(vis[0].timestamp) if vis else "",
                    "updatedAt": _iso(vis[-1].timestamp) if vis else "",
                    "firstMessagePreview": str(vis[0].content)[:120] if vis else "",
                })
        threads.sort(key=lambda t: t["updatedAt"], reverse=True)
        return {"threads": threads}

    def _history(self, params: dict) -> dict:
        """Replayed user-visible turns of one thread, paged by an opaque message-
        id cursor (`before`) — NOT a timestamp, which would drop messages on the
        routine same-timestamp collisions rumi's `seq` tiebreak exists for."""
        agent = self._agent(params)  # offline replay; no network (lazy heart)
        vis = _visible(agent.ideas)
        try:
            limit = int(params.get("limit", 50))
        except (TypeError, ValueError):
            raise _RpcError(-32602, "limit must be an integer")
        if limit <= 0:
            limit = 50
        limit = min(limit, 200)
        before = params.get("before")
        if before:
            idx = next((i for i, m in enumerate(vis) if m.id == before), None)
            if idx is None:
                raise _RpcError(-32602, f"unknown cursor: {before!r}")
            vis = vis[:idx]  # everything strictly older than the cursor message
        has_more = len(vis) > limit
        window = vis[-limit:] if has_more else vis
        out = {"messages": [_to_a2a_message(m) for m in window], "hasMore": has_more}
        if has_more and window:
            out["nextCursor"] = window[0].id  # page further back with this
        return out

    def _list_resources(self, params: dict) -> dict:
        """Inscribed box keys (NOT history) as MCP resources. Reads the tablet
        directly via `world.tablet_for` — no agent construction needed."""
        agent_type, agent_id = self._identity(params)
        tablet = self._world.tablet_for(_qualified(agent_type), agent_id)
        resources = [
            {
                "uri": "rumi://box/" + key,
                "name": key,
                "description": "Box key: " + key,
                "mimeType": "application/json",
            }
            for key in sorted(tablet.list_keys())
            if not key.startswith("history/")
        ]
        return {"resources": resources}

    def _read_resource(self, params: dict) -> dict:
        """Read one box key by `rumi://box/{key}` URI (a bare key is accepted
        too). History keys are out of resource scope."""
        agent_type, agent_id = self._identity(params)
        uri = params.get("uri") or ""
        key = uri[len("rumi://box/"):] if uri.startswith("rumi://box/") else uri
        if not key:
            raise _RpcError(-32602, "missing param: 'uri'")
        if key.startswith("history/"):
            raise _RpcError(-32602, f"not a resource: {key!r}")
        val = self._world.tablet_for(_qualified(agent_type), agent_id).get(key, _MISSING)
        if val is _MISSING:
            raise _RpcError(-32602, f"unknown resource: {key!r}")
        return {"contents": [{
            "uri": "rumi://box/" + key,
            "text": json.dumps(val, ensure_ascii=False),
            "mimeType": "application/json",
        }]}

    @staticmethod
    def _identity(params: dict) -> tuple[str, str]:
        try:
            return params["agent_type"], params["agent_id"]
        except KeyError as e:
            raise _RpcError(-32602, f"missing param: {e}")


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        self._peer_gone = False
        for raw in self.rfile:  # one newline-delimited JSON-RPC request per line
            line = raw.strip()
            if not line:
                continue
            rid = None
            try:
                req = json.loads(line)
                rid = req.get("id")
                method = req.get("method")
                params = req.get("params") or {}
                if method == "invoke_stream":
                    # Streaming: bare progress frames then exactly one terminal
                    # frame — NOT a JSON-RPC envelope (the Go client reads frames
                    # by `type` until terminal). All other methods stay one-shot.
                    self._run_stream(params)
                    continue
                result = self.server.worker.handle(method, params)
                resp = {"jsonrpc": "2.0", "id": rid, "result": result}
            except _RpcError as e:
                resp = {"jsonrpc": "2.0", "id": rid, "error": {"code": e.code, "message": e.message}}
            except Exception as e:  # surface as a JSON-RPC internal error, keep serving
                resp = {"jsonrpc": "2.0", "id": rid, "error": {"code": -32603, "message": str(e)}}
            self._write(resp)

    def _run_stream(self, params: dict) -> None:
        """Drive one `invoke_stream`: progress frames via `emit`, one terminal."""
        def emit(frame: dict) -> None:
            self._write_frame(frame)
        try:
            terminal = self.server.worker.stream_invoke(params, emit)
        except _RpcError as e:  # e.g. missing/unknown agent → terminal error frame
            terminal = {"type": "error", "message": e.message, "code": e.code}
        except Exception as e:  # pragma: no cover - defensive
            terminal = {"type": "error", "message": str(e)}
        self._write_frame(terminal)

    def _write_frame(self, frame: dict) -> None:
        """Write one NDJSON frame; on a dead peer, latch _peer_gone and no-op
        thereafter so a half-written frame can't interleave."""
        if self._peer_gone:
            return
        try:
            self.wfile.write((json.dumps(frame, ensure_ascii=False) + "\n").encode("utf-8"))
            self.wfile.flush()
        except (BrokenPipeError, OSError):
            self._peer_gone = True

    def _write(self, obj: dict) -> None:
        self.wfile.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
        self.wfile.flush()


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, worker: Worker) -> None:
        super().__init__(addr, _Handler)
        self.worker = worker


def main() -> None:
    ap = argparse.ArgumentParser(description="rumi-edge Python worker")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0)
    args = ap.parse_args()

    server = _Server((args.host, args.port), Worker())
    host, port = server.server_address
    print(f"RUMI_WORKER_LISTENING {host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
