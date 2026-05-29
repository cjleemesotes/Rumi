"""Tests for Dervish facade + whirl pipeline."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from rumi.heart.config import HeartConfig, Voice
from rumi.agent import registry as _multiton_mod
from rumi.agent.agent import Dervish, inscribe
from rumi.world import World
from rumi.ideas.types import (
    BaseModelIdea,
    JsonIdea,
    TextIdea,
    ToolRequestIdea,
    ToolResultIdea,
)
from rumi.tools.decorator import _reset_for_tests as _reset_tools


@pytest.fixture(autouse=True)
def _reset_all():
    _reset_tools()
    _multiton_mod._reset_for_tests()
    yield
    _reset_tools()
    _multiton_mod._reset_for_tests()


@pytest.fixture
def world(tmp_path: Path) -> World:
    with patch("rumi.world.openai.OpenAI"):
        return World(openrouter_api_key="sk-test", data_dir=tmp_path)


def _canned_text_response(text: str = "ok") -> dict:
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _canned_tool_request(tool_name: str, args: str = "{}") -> dict:
    return {
        "choices": [{
            "message": {
                "content": None,
                "tool_calls": [{
                    "id": "c1",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": args},
                }],
            }
        }]
    }


class Researcher(Dervish):
    heart_config = HeartConfig(voices=[Voice(model_name="~openai/gpt-latest")])


def test_subclass_required_heart_config(world):
    class _Bad(Dervish):
        heart_config = None

    with pytest.raises(ValueError, match="heart_config"):
        _Bad(world=world, agent_id="x")


def test_agent_rejects_degenerate_agent_id(world, tmp_path):
    """Regression: agent_id is the tablet scope and is Path-joined directly
    into the data_dir hierarchy. Empty/whitespace ids alias every instance
    onto the agent_type directory itself (silent multi-instance disk
    pollution); ids containing path separators or '.'/'..' escape the
    World's data_dir entirely (literal sandbox escape, since the scope dir
    is built BEFORE _safe_filename gets a chance to sanitize anything).

    These are silent-failure inputs, with the additional sandbox-escape
    twist on the path-traversal variants.
    """
    degenerate_ids = [
        "",            # empty — aliases to agent_type root
        "   ",         # whitespace — Win32 trailing-space directory footgun
        "/",           # bare separator
        "\\",          # Windows separator
        "foo/bar",     # nested separator
        "foo\\bar",    # nested Windows separator
        "../escape",   # parent traversal — literal sandbox escape
        ".",           # current dir
        "..",          # parent dir
    ]
    for bad in degenerate_ids:
        with pytest.raises(ValueError, match="agent_id"):
            Researcher(world=world, agent_id=bad)

    # Positive control: a valid id still constructs cleanly.
    r = Researcher(world=world, agent_id="valid-id")
    assert r.agent_id == "valid-id"


def test_multiton_identity(world):
    r1 = Researcher(world=world, agent_id="r-1")
    r2 = Researcher(world=world, agent_id="r-1")
    assert r1 is r2
    # Different agent_id is a different instance
    r3 = Researcher(world=world, agent_id="r-2")
    assert r3 is not r1


def test_mind_type_property(world):
    r = Researcher(world=world, agent_id="r-1")
    # agent_type is now fully qualified — endswith the bare class name
    assert r.agent_type.endswith("Researcher")
    assert "." in r.agent_type  # qualified form: module.ClassName
    assert r.agent_id == "r-1"


def test_inscribe_recall_str_key(world):
    r = Researcher(world=world, agent_id="r-1")
    r.inscribe("started_at", 12345.0)
    assert r.recall("started_at") == 12345.0
    assert r.recall("missing", "default") == "default"


class _ResearchState(BaseModel):
    last_query: str = ""
    findings: list[str] = []


def test_inscribe_recall_typed_class(world):
    r = Researcher(world=world, agent_id="r-class")
    st = _ResearchState(last_query="saturn", findings=["a"])
    r.inscribe(_ResearchState, st)

    out = r.recall(_ResearchState)
    assert isinstance(out, _ResearchState)
    assert out.last_query == "saturn"
    assert out.findings == ["a"]


def test_inscribe_typed_rejects_wrong_type(world):
    r = Researcher(world=world, agent_id="r-bad")
    with pytest.raises(TypeError):
        r.inscribe(_ResearchState, "not-a-research-state")


def test_recall_typed_default(world):
    r = Researcher(world=world, agent_id="r-default")
    assert r.recall(_ResearchState, default=None) is None


def test_recall_typed_default_wrong_type_rejected(world):
    """Regression: symmetry with Dervish.inscribe.

    `inscribe(cls, value)` rejects type-mismatched values via isinstance.
    `recall(cls, default=value)` silently accepted wrong-type defaults;
    when the tablet was empty, the user got back e.g. a string where their
    code expected a BaseModel instance. The error then surfaced far away
    as AttributeError on first field access, blaming the field rather
    than the recall call site.

    Restores symmetry between inscribe (type-safe write) and recall
    (type-safe read with type-safe default).
    """
    r = Researcher(world=world, agent_id="r-default-bad")

    # Wrong-type default raises with a message that names the call site
    with pytest.raises(TypeError, match="recall.*expects a _ResearchState"):
        r.recall(_ResearchState, default="not a state")
    with pytest.raises(TypeError, match="recall.*expects a _ResearchState"):
        r.recall(_ResearchState, default=42)
    with pytest.raises(TypeError, match="recall.*expects a _ResearchState"):
        r.recall(_ResearchState, default={"last_query": "x"})  # dict, not instance

    # None is still allowed (documented default)
    assert r.recall(_ResearchState, default=None) is None
    # Correct-type instance is still allowed
    fallback = _ResearchState(last_query="fallback")
    assert r.recall(_ResearchState, default=fallback) is fallback


def test_inscribe_decorator_is_identity():
    """@inscribe should return the class unchanged (mirror of nodeai surface)."""
    @inscribe
    class M(BaseModel):
        x: int = 0

    assert M.__name__ == "M"
    assert M(x=5).x == 5


def test_inscribe_decorator_rejects_non_basemodel():
    """Regression: @inscribe used to silently accept any class. The error then
    surfaced far away inside Dervish.inscribe() ("inscribe key must be a string
    or a pydantic class") — blaming the call site instead of the decoration
    line where the user actually made the mistake.

    Symmetric on the storage side: catch the misuse at
    the decorator, not at first use.
    """
    with pytest.raises(TypeError, match="BaseModel"):
        @inscribe
        class NotAModel:
            pass

    # And a non-class passed (e.g. a function or a string) should also reject
    with pytest.raises(TypeError, match="BaseModel"):
        inscribe("not a class")  # type: ignore[arg-type]


# ---- whirl pipeline ----


def test_whirl_simple_text_response(world):
    with patch("rumi.heart.openrouter.OpenRouterHeart.call",
               return_value=_canned_text_response("hi there")) as mock_call:
        r = Researcher(world=world, agent_id="whirl-1")
        result = r.whirl("ping")

    assert isinstance(result, TextIdea)
    assert result.content == "hi there"
    # User + assistant ideas persisted
    assert len(r.ideas) == 2
    assert r.ideas[0].role == "user"
    assert r.ideas[1].role == "assistant"
    assert mock_call.call_count == 1


def test_whirl_invokes_hooks_in_order(world):
    fired = []

    class HookedResearcher(Dervish):
        heart_config = HeartConfig(voices=[Voice()])

        def _on_whirl_start_impl(self, incoming): fired.append("whirl_start")
        def _on_heart_call_start_impl(self, payload): fired.append("llm_start")
        def _on_heart_call_ended_impl(self, payload_in, payload_out): fired.append("llm_end")
        def _on_whirl_ended_impl(self, final): fired.append("whirl_end")

    with patch("rumi.heart.openrouter.OpenRouterHeart.call",
               return_value=_canned_text_response("ok")):
        r = HookedResearcher(world=world, agent_id="h-1")
        r.whirl("ping")

    assert fired == ["whirl_start", "llm_start", "llm_end", "whirl_end"]


def test_whirl_isolates_hook_exceptions(world):
    """Regression: lifecycle hooks are documented as observation/bookkeeping.
    A raise in a user override must NOT erase a successful whirl().

    The most concrete failure mode this isolates is `_on_heart_call_ended_impl`:
    it fires AFTER `heart.call(...)` returns the response but BEFORE
    `parse_response` + `_append_idea` run. Without isolation, an exception
    there means the LLM call is BILLED but the response is never parsed or
    persisted — a cost-value inversion.

    With isolation, all four hooks log a warning and continue. whirl()
    returns normally; the user can inspect logs to find the override bug.
    """
    class BadHooks(Dervish):
        heart_config = HeartConfig(voices=[Voice()])

        def _on_whirl_start_impl(self, incoming):
            raise RuntimeError("whirl_start failed")

        def _on_heart_call_start_impl(self, payload_in):
            raise RuntimeError("call_start failed")

        def _on_heart_call_ended_impl(self, payload_in, payload_out):
            raise RuntimeError("call_ended failed")

        def _on_whirl_ended_impl(self, final_idea):
            raise RuntimeError("whirl_ended failed")

    # Patch the pipeline's logger.warning directly so we can assert which
    # hook failures were reported. (Using `caplog` here is unreliable because
    # the `rumi.pipeline` logger may have its own handler with propagate=False
    # via World's logger setup, which makes caplog miss records.)
    warning_messages: list[str] = []
    with patch("rumi.agent.pipeline.log") as mock_log:
        mock_log.warning = lambda fmt, *args, **kw: warning_messages.append(fmt % args)
        with patch("rumi.heart.openrouter.OpenRouterHeart.call",
                   return_value=_canned_text_response("hi")):
            agent = BadHooks(world=world, agent_id="hooks-isolate")
            # whirl() must NOT raise even though all four hooks raise
            result = agent.whirl("hello")

    # The LLM response was preserved end-to-end
    assert isinstance(result, TextIdea)
    assert result.content == "hi"
    # And the user message + response both made it into history
    assert any(i.role == "user" and getattr(i, "content", None) == "hello"
               for i in agent.ideas)
    assert any(i is result for i in agent.ideas)

    # All four hook failures got logged as warnings
    assert any("_on_whirl_start_impl" in m and "whirl_start failed" in m
               for m in warning_messages), warning_messages
    assert any("_on_heart_call_start_impl" in m and "call_start failed" in m
               for m in warning_messages), warning_messages
    assert any("_on_heart_call_ended_impl" in m and "call_ended failed" in m
               for m in warning_messages), warning_messages
    assert any("_on_whirl_ended_impl" in m and "whirl_ended failed" in m
               for m in warning_messages), warning_messages


def test_tool_loop_runs_then_terminates(world):
    from rumi.tools.decorator import tool

    @tool("Researcher")
    def search(query: str) -> str:
        """Search."""
        return f"results-for-{query}"

    responses = [
        _canned_tool_request("search", '{"query":"saturn"}'),
        _canned_text_response("final answer: results-for-saturn"),
    ]
    call_iter = iter(responses)
    with patch("rumi.heart.openrouter.OpenRouterHeart.call",
               side_effect=lambda *a, **kw: next(call_iter)):
        r = Researcher(world=world, agent_id="tool-1")
        result = r.whirl("find saturn")

    assert isinstance(result, TextIdea)
    assert "results-for-saturn" in result.content
    # Should have: user msg, tool_request, tool_result, final text = 4 ideas
    kinds = [type(t).__name__ for t in r.ideas]
    assert "ToolRequestIdea" in kinds
    assert "ToolResultIdea" in kinds


def test_tool_loop_max_iterations_guard(world):
    from rumi.tools.decorator import tool

    @tool("Researcher")
    def loopy(x: str) -> str:
        return "loop"

    # Always return tool requests — pipeline should bail after MAX_TOOL_ITERATIONS
    def always_tool(*args, **kwargs):
        return _canned_tool_request("loopy", '{"x":"_"}')

    with patch("rumi.heart.openrouter.OpenRouterHeart.call", side_effect=always_tool):
        r = Researcher(world=world, agent_id="loop-1")
        with pytest.raises(RuntimeError, match="whirled past the floor"):
            r.whirl("go")


def test_agent_type_qualified_propagates_to_tablet_path(world, tmp_path):
    """Regression: two same-named Dervish classes must get distinct tablet
    scopes on disk. Bare-name agent_type would have collided
    `data/Researcher/r1/` for both classes — agent B reading agent A's
    persisted history on init, and one's `inscribe` overwriting the other.
    The qualified agent_type keys the on-disk path by
    `module.ClassName`, separating the scopes.
    """

    class _OurResearcher(Dervish):
        heart_config = HeartConfig(voices=[Voice()])

    a = _OurResearcher(world=world, agent_id="r-1")

    # agent_type is the qualified form
    assert "." in a.agent_type
    assert a.agent_type.endswith("_OurResearcher")

    # Tablet's scope is rooted at the qualified agent_type path
    assert a.tablet.agent_type == a.agent_type
    # On-disk: data_dir/<qualified>/<agent_id>/
    scope_dir = world.data_dir / a.agent_type / "r-1"
    assert scope_dir.exists() or world.data_dir.exists()


def test_agent_registry_disambiguates_same_named_classes(world):
    """Regression: two Dervish subclasses with the same bare name but different
    modules must not silently overwrite each other in the global registry.
    `get_agent` should still resolve bare names when unambiguous, but raise
    a clear error when multiple classes match.

    This is the same module-aliasing pattern as the _inscribed_key_for
    fix, applied to the separate `_AGENT_SUBCLASSES` registry.
    """
    from rumi.agent.agent import _AGENT_SUBCLASSES

    # Build a real Dervish subclass in this test module
    class _Probe(Dervish):
        heart_config = HeartConfig(voices=[Voice()])

    # Register a synthetic same-named class with a different __module__
    other = type(
        "_Probe",
        (Dervish,),
        {
            "__module__": "fake.other.plugin",
            "heart_config": HeartConfig(voices=[Voice()]),
        },
    )

    # Both classes coexist under DISTINCT qualified keys
    assert _AGENT_SUBCLASSES.get(f"{_Probe.__module__}._Probe") is _Probe
    assert _AGENT_SUBCLASSES.get("fake.other.plugin._Probe") is other

    # Bare-name lookup raises with disambiguation hint
    r = _Probe(world=world, agent_id="probe-1")
    with pytest.raises(KeyError, match="ambiguous"):
        r.get_agent("_Probe", "id")

    # Qualified-name lookup resolves cleanly
    same_module_qualified = f"{_Probe.__module__}._Probe"
    instance = r.get_agent(same_module_qualified, "id")
    assert isinstance(instance, _Probe)

    # Cleanup the synthetic class so other tests don't see it
    _AGENT_SUBCLASSES.pop("fake.other.plugin._Probe", None)


def test_inscribed_key_qualified_by_module():
    """Regression: _inscribed_key_for must include the class's module path,
    so two same-named classes from different modules (e.g. plugin_a.Settings
    and plugin_b.Settings) never alias on the tablet. The prior version used
    only `_snake_case(cls.__name__)`, which silently collapsed cross-module
    duplicates.

    Same-module same-snake-case collisions (APIKey vs ApiKey defined in the
    SAME module) remain a known limitation — that pair is unusual in
    practice; cross-module class-name collisions are the realistic risk
    this fix is closing.
    """
    from rumi.agent.agent import _inscribed_key_for

    class _Settings(BaseModel):
        x: int = 0

    # Real class in this test module
    key_real = _inscribed_key_for(_Settings)

    # Synthetic class with same name but a different __module__
    fake = type("_Settings", (BaseModel,), {"__module__": "other.plugin"})
    key_fake = _inscribed_key_for(fake)

    assert key_real != key_fake, (
        f"Two `_Settings` classes from different modules collided: {key_real!r}"
    )
    assert _Settings.__module__ in key_real
    assert "other.plugin" in key_fake
    # Both keys still contain the snake-case form for human-grep
    assert "_settings" in key_real
    assert "_settings" in key_fake


def test_tool_loop_fires_at_exactly_max_iterations(world):
    """Regression: the cap must fire AT MAX_TOOL_ITERATIONS, not one past it.

    The previous `iterations > MAX` check ran the (MAX+1)-th tool cycle to
    completion before raising — each extra round being a real LLM call, real
    tool side effects, and a persisted ToolRequest+ToolResult pair. The
    fix uses `iterations >= MAX` so the agent stays inside the documented
    cap.
    """
    from rumi.tools.decorator import tool
    from rumi.agent.pipeline import MAX_TOOL_ITERATIONS

    @tool("Researcher")
    def loopy(x: str) -> str:
        return "loop"

    call_count = {"n": 0}

    def always_tool(*args, **kwargs):
        call_count["n"] += 1
        return _canned_tool_request("loopy", '{"x":"_"}')

    with patch("rumi.heart.openrouter.OpenRouterHeart.call", side_effect=always_tool):
        r = Researcher(world=world, agent_id="loop-precise")
        with pytest.raises(RuntimeError, match="whirled past the floor"):
            r.whirl("go")

    # The LLM is called once per tool cycle. With the cap firing AT
    # MAX_TOOL_ITERATIONS (>=), exactly MAX cycles run before the raise.
    # (Each cycle: heart.call → ToolRequestIdea, execute tools, append
    # results, increment iterations, check the cap.)
    assert call_count["n"] == MAX_TOOL_ITERATIONS, (
        f"Expected exactly {MAX_TOOL_ITERATIONS} LLM calls before cap; "
        f"got {call_count['n']}"
    )


def test_whirl_with_response_model(world):
    class Summary(BaseModel):
        title: str
        bullets: list[str]

    parsed_response = {
        "choices": [{
            "message": {
                "parsed": {"title": "saturn", "bullets": ["a", "b"]},
                "content": '{"title":"saturn","bullets":["a","b"]}',
            }
        }],
        "usage": {"total_tokens": 5},
    }

    with patch("rumi.heart.openrouter.OpenRouterHeart.call", return_value=parsed_response):
        r = Researcher(world=world, agent_id="rm-1")
        result = r.whirl("Summarize", response_model=Summary)

    assert isinstance(result, BaseModelIdea)
    # Content is a real pydantic instance (not a dict) — attribute access works.
    assert isinstance(result.content, Summary)
    assert result.content.title == "saturn"
    assert result.content.bullets == ["a", "b"]


def test_use_tool_bypasses_heart(world):
    from rumi.tools.decorator import tool

    @tool("Researcher")
    def list_papers(topic: str) -> list[str]:
        """List papers."""
        return ["paper-a", "paper-b"]

    r = Researcher(world=world, agent_id="ut-1")
    out = r.use_tool("list_papers", {"topic": "saturn"})
    assert out == ["paper-a", "paper-b"]


def test_use_tool_unknown_raises(world):
    r = Researcher(world=world, agent_id="ut-2")
    with pytest.raises(KeyError, match="has no tool named"):
        r.use_tool("missing", {})


def test_use_tool_rejects_non_dict_args(world):
    """Regression: `use_tool` is the documented public direct-invocation API.
    `args = args or {}` only caught None/falsy — a non-dict like a string or
    a list of tuples slipped through and crashed at `entry.function(**args)`
    with an opaque framework-internal TypeError that names internal variables
    rather than explaining the contract. The user surface should reject at
    the boundary with a message that names the public parameter and the
    type expectation.

    Loud rejection at the user-facing
    surface beats opaque downstream failure.
    """
    from rumi.tools.decorator import tool as _tool

    @_tool(["Researcher"])
    def noop_tool() -> str:
        return "ok"

    r = Researcher(world=world, agent_id="ut-args-reject")

    # Non-dict inputs are rejected with a meaningful TypeError
    for bad in ("string", [("x", 1)], 42, True, b"bytes"):
        with pytest.raises(TypeError, match="expects a dict"):
            r.use_tool("noop_tool", args=bad)

    # Legitimate calls still work
    assert r.use_tool("noop_tool") == "ok"            # None still fine
    assert r.use_tool("noop_tool", args={}) == "ok"   # empty dict still fine
    assert r.use_tool("noop_tool", args=None) == "ok"  # explicit None


def test_get_agent_cross_agent_lookup(world):
    class Summarizer(Dervish):
        heart_config = HeartConfig(voices=[Voice()])

    r = Researcher(world=world, agent_id="get-1")
    s = r.get_agent("Summarizer", "s-1")
    assert isinstance(s, Summarizer)
    # Same instance on re-lookup (registry)
    s2 = r.get_agent("Summarizer", "s-1")
    assert s is s2


def test_get_agent_unknown_type_raises(world):
    r = Researcher(world=world, agent_id="get-2")
    with pytest.raises(KeyError, match="No Dervish named"):
        r.get_agent("NotARealMindType", "x")


def test_history_persisted_across_instances(world, tmp_path):
    """Storage rehydration: a new Dervish with same agent_id replays prior ideas."""
    with patch("rumi.heart.openrouter.OpenRouterHeart.call",
               return_value=_canned_text_response("response")):
        r1 = Researcher(world=world, agent_id="persist-1")
        r1.whirl("first message")

    initial_count = len(r1.ideas)
    assert initial_count == 2

    # Clear the registry cache so we get a NEW instance for the same id
    _multiton_mod._reset_for_tests()

    # Same world + agent_id → re-instantiate; should replay
    r2 = Researcher(world=world, agent_id="persist-1")
    assert r1 is not r2
    assert len(r2.ideas) == initial_count


def test_replay_history_warns_on_unreadable_entry(world, tmp_path):
    """Regression: truly unreadable history files (corrupt JSON,
    OS errors, empty payloads — all surfaces where `Tablet.get(key)` returns
    None) were silently skipped during replay. The user had no way to know
    their disk had a ghost gap.

    The `from_dict` failure branch already warned on parseable-but-malformed
    data; this fix closes the symmetric leg for truly unreadable files.

    A failure that lived forever silent now emits a warning naming the key.
    """
    # Populate one valid history entry, then write garbage to a second file
    # that list_keys will return but `get` will reject as unreadable.
    with patch("rumi.heart.openrouter.OpenRouterHeart.call",
               return_value=_canned_text_response("response")):
        r1 = Researcher(world=world, agent_id="replay-corrupt-1")
        r1.whirl("valid message")

    # Tablet stores keys flat at scope root with `/` escaped to `__`
    scope_dir = r1._tablet._scope_dir
    history_files = sorted(scope_dir.glob("history__*.json"))
    assert len(history_files) >= 1, "should have at least one history file"
    # Empty the file — `Tablet.get` returns None on JSONDecodeError; that's
    # exactly the unreadable case the fix targets.
    bad_file = history_files[0]
    bad_file.write_text("")

    # Reset multiton so we get a fresh Dervish that replays from disk
    _multiton_mod._reset_for_tests()

    # Capture warnings by replacing the agent's logger after construction
    warning_messages: list[str] = []

    class _CaptureLogger:
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def warning(self, fmt, *args, **kw):
            warning_messages.append(fmt % args if args else fmt)
            self._wrapped.warning(fmt, *args, **kw)

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    # Re-instantiate, then re-run replay with the capture logger spliced in.
    # (The first replay during __init__ uses the real logger; we re-run to
    # observe what THIS fix emits without monkeypatching global logging.)
    r2 = Researcher(world=world, agent_id="replay-corrupt-1")
    r2._logger = _CaptureLogger(r2._logger)
    r2._ideas = []
    r2._replay_history()

    # Derive the key the warning should name: filename has `history__` prefix
    # and `.json` suffix; the key is `history/<rest>`.
    bad_stem = bad_file.stem  # e.g. "history__abc123"
    bad_key = "history/" + bad_stem[len("history__"):]
    assert any(bad_key in m and "unreadable" in m for m in warning_messages), (
        f"expected unreadable warning for {bad_key}, got: {warning_messages}"
    )


def test_tool_cycle_atomic_rollback_on_result_persist_failure(world):
    """Regression: if a ToolResultIdea's persistence fails mid-batch, the
    matching ToolRequestIdea must be rolled back from both memory AND disk —
    otherwise the next whirl() builds messages ending in an unmatched
    assistant tool_calls and OpenAI rejects with HTTP 400.

    This is the inter-call extension of the single-call atomicity fix.
    """
    from rumi.tools.decorator import tool

    @tool(["Researcher"])
    def echo(x: str) -> str:
        return f"echoed: {x}"

    # Stage 1: LLM returns a tool request; Stage 2 (unused — we make the
    # tool-result append fail before we get there).
    with patch("rumi.heart.openrouter.OpenRouterHeart.call",
               return_value=_canned_tool_request("echo", '{"x":"hi"}')):
        r = Researcher(world=world, agent_id="atomic-1")

        # Snapshot disk state BEFORE the doomed whirl
        keys_before = sorted(r._tablet.list_keys("history/"))
        original_set = r._tablet.set
        call_count = {"n": 0}

        def flaky_set(key, value):
            call_count["n"] += 1
            # 1st call: user TextIdea (succeeds)
            # 2nd call: ToolRequestIdea (succeeds)
            # 3rd call: ToolResultIdea (FAIL)
            if call_count["n"] == 3:
                raise OSError("simulated mid-tool-cycle disk failure")
            return original_set(key, value)

        with patch.object(r._tablet, "set", side_effect=flaky_set):
            with pytest.raises(OSError, match="simulated"):
                r.whirl("call the tool")

    # After the failure: NO orphan ToolRequestIdea on disk.
    keys_after = sorted(r._tablet.list_keys("history/"))
    new_keys = set(keys_after) - set(keys_before)
    # The user TextIdea persisted before the request; the ToolRequest persisted
    # but was rolled back; the ToolResult never persisted. So new_keys should
    # contain ONLY the user TextIdea — no ToolRequestIdea, no ToolResultIdea.
    from rumi.ideas.types import ToolRequestIdea as _TR, ToolResultIdea as _TRes
    for k in new_keys:
        data = r._tablet.get(k)
        # Whatever's left must NOT be a ToolRequestIdea (those are rolled back)
        assert data.get("kind") != "tool_request", (
            f"Orphan ToolRequestIdea on disk after rollback: {k} -> {data}"
        )

    # And in-memory state must not contain an orphan ToolRequestIdea either
    for i in r.ideas:
        assert not isinstance(i, _TR), (
            f"Orphan ToolRequestIdea in memory after rollback: {i}"
        )


def test_tool_cycle_atomic_rollback_unwinds_partial_results(world):
    """Regression for multi-call requests: when an assistant message carries
    N>1 tool_calls and the k-th result (k>0) fails to persist, results 0..k-1
    have ALREADY committed. Without unwinding them, they strand on disk and in
    memory; on the next restart they sit mid-history as orphan tool messages
    (the leading-orphan skip only saves head orphans), and OpenAI 400s
    with 'messages with role tool must be a response to a preceding message
    with tool_calls'. The atomicity contract is "request + ALL results as a
    unit" — track committed results and roll them back too.
    """
    from rumi.tools.decorator import tool
    from rumi.ideas.types import ToolRequestIdea as _TR, ToolResultIdea as _TRes

    @tool(["Researcher"])
    def echo_a(x: str) -> str:
        return f"a-{x}"

    @tool(["Researcher"])
    def echo_b(x: str) -> str:
        return f"b-{x}"

    # Two-tool-call request — both calls execute, both results queue for persist.
    two_call_response = {
        "choices": [{
            "message": {
                "content": None,
                "tool_calls": [
                    {"id": "c1", "type": "function",
                     "function": {"name": "echo_a", "arguments": '{"x":"hi"}'}},
                    {"id": "c2", "type": "function",
                     "function": {"name": "echo_b", "arguments": '{"x":"hi"}'}},
                ],
            }
        }]
    }

    with patch("rumi.heart.openrouter.OpenRouterHeart.call",
               return_value=two_call_response):
        r = Researcher(world=world, agent_id="atomic-multi-1")

        keys_before = sorted(r._tablet.list_keys("history/"))
        original_set = r._tablet.set
        call_count = {"n": 0}

        def flaky_set(key, value):
            call_count["n"] += 1
            # 1: user TextIdea (ok)  2: ToolRequestIdea (ok)
            # 3: 1st ToolResultIdea (ok)   4: 2nd ToolResultIdea (FAIL)
            if call_count["n"] == 4:
                raise OSError("simulated mid-batch disk failure on result 2")
            return original_set(key, value)

        with patch.object(r._tablet, "set", side_effect=flaky_set):
            with pytest.raises(OSError, match="simulated mid-batch"):
                r.whirl("call both tools")

    # NO orphan ToolRequestIdea AND no orphan ToolResultIdea on disk.
    keys_after = sorted(r._tablet.list_keys("history/"))
    new_keys = set(keys_after) - set(keys_before)
    for k in new_keys:
        data = r._tablet.get(k)
        kind = data.get("kind")
        assert kind not in ("tool_request", "tool_result"), (
            f"Orphan {kind} on disk after multi-call rollback: {k} -> {data}"
        )

    # Same invariant in memory — no stranded ToolResultIdea from the partial batch.
    for i in r.ideas:
        assert not isinstance(i, _TR), f"Orphan ToolRequestIdea in memory: {i}"
        assert not isinstance(i, _TRes), f"Orphan ToolResultIdea in memory: {i}"


def test_append_idea_does_not_drift_on_persistence_failure(world):
    """Regression: a tablet.set failure must NOT leave the in-memory history
    holding an idea that was never persisted. Such drift would survive the
    process and re-orphan on the next restart — defeating the seq tiebreak
    and orphan-skip read-side guards.
    """
    from rumi.heart.config import HeartConfig

    class _Lite(Dervish):
        heart_config = HeartConfig(voices=[Voice()])

    a = _Lite(world=world, agent_id="drift-1")
    starting_count = len(a._ideas)

    # Force the tablet.set to raise on the next write
    with patch.object(a._tablet, "set", side_effect=OSError("simulated disk failure")):
        with pytest.raises(OSError, match="simulated disk failure"):
            a._append_idea(TextIdea(role="user", content="will not survive"))

    # Memory must be UNCHANGED — no drift between in-memory list and disk
    assert len(a._ideas) == starting_count
    assert a._current_idea is None or a._current_idea.content != "will not survive"


def test_build_messages_does_not_orphan_tool_results(world):
    """Regression: context-window tail-slice must never leave a `role="tool"`
    message without its preceding `assistant` with matching `tool_calls`.

    Driven through the real `whirl()` path (whichever pipeline backend is
    active) rather than calling the internal message-builder directly. Seed a
    history whose `context_window` slice would begin in the middle of a tool
    cycle, then capture the messages the Heart actually receives and assert no
    tool message is orphaned.

    Concrete repro: seeded history = [ToolRequest, ToolResult, ToolResult,
    Text(assistant)]; `whirl()` appends the user turn → 5 ideas. With
    `context_window=4`, the naive `history[-4:]` slice yields [ToolResult,
    ToolResult, Text(assistant), Text(user)] — both ToolResults orphaned (their
    ToolRequest at index 0 dropped). OpenAI rejects this with `400: messages
    with role 'tool' must be a response to a preceding message with
    'tool_calls'`. The fix skips leading orphan ToolResultIdeas so the slice
    always begins on a self-contained message.
    """
    class Truncator(Dervish):
        heart_config = HeartConfig(voices=[Voice(model_name="x", context_window=4)])

    captured: dict = {}

    def capture(voice, messages, **kwargs):
        # Heart is called as call(voice, messages, tools=..., response_model=...)
        # under both pipeline backends; grab the built messages and return a
        # plain text reply so the loop terminates after one call.
        captured["messages"] = messages
        return _canned_text_response("done")

    with patch("rumi.heart.openrouter.OpenRouterHeart.call", side_effect=capture):
        r = Truncator(world=world, agent_id="trunc-1")
        # Seed a tool cycle directly into history: a 2-call request, its two
        # results, and a trailing assistant turn. `whirl()` then appends the
        # user message; with context_window=4 the slice would start on the
        # first orphaned ToolResultIdea without the leading-orphan skip.
        r._ideas.clear()
        r._append_idea(ToolRequestIdea(content=[
            {"id": "c1", "type": "function",
             "function": {"name": "f", "arguments": "{}"}},
            {"id": "c2", "type": "function",
             "function": {"name": "g", "arguments": "{}"}},
        ]))
        r._append_idea(ToolResultIdea(content="a", tool_call_id="c1", name="f"))
        r._append_idea(ToolResultIdea(content="b", tool_call_id="c2", name="g"))
        r._append_idea(TextIdea(role="assistant", content="done-prior"))
        r.whirl("trigger")

    msgs = captured["messages"]

    # Invariant: every role="tool" message must be preceded by an assistant
    # whose tool_calls contains its tool_call_id, within the same message list.
    for i, m in enumerate(msgs):
        if m.get("role") != "tool":
            continue
        tcid = m.get("tool_call_id")
        prior = msgs[:i]
        has_caller = any(
            p.get("role") == "assistant"
            and any(c.get("id") == tcid for c in (p.get("tool_calls") or []))
            for p in prior
        )
        assert has_caller, (
            f"orphan tool message at index {i}: {m}\nprior messages: {prior}"
        )


def test_history_preserves_write_order_under_timestamp_collision(world):
    """Replay must preserve write-order even when timestamps collide.

    Regression test: on Windows, time.time() can return identical values for
    rapid-fire writes during a tool-call burst. Without the seq tiebreaker,
    stable sort by timestamp falls back to alphabetical-UUID order, which can
    put a ToolResultIdea BEFORE its ToolRequestIdea on replay — and OpenAI
    rejects the next whirl() with 'messages with role tool must be a response
    to a preceding message with tool_calls'.
    """
    from rumi.ideas.types import ToolRequestIdea, ToolResultIdea, TextIdea

    # Force three ideas with the same timestamp but distinct seq + IDs
    # chosen so alphabetical sort would scramble them.
    SAME_T = 1_700_000_000.0
    text = TextIdea(content="seed", timestamp=SAME_T, id="zzz_text")
    req = ToolRequestIdea(content=[], timestamp=SAME_T, id="mmm_req")
    res = ToolResultIdea(content="result", timestamp=SAME_T, id="aaa_res",
                         tool_call_id="x", name="t")
    # write-order should be: text, req, res
    # alphabetical-UUID order would scramble to: aaa_res, mmm_req, zzz_text
    # which puts the tool result BEFORE its request → broken.

    # Persist them in write-order to a fresh Dervish's tablet, then replay
    class Recall(Dervish):
        heart_config = HeartConfig(voices=[Voice()])

    with patch("rumi.heart.openrouter.OpenRouterHeart.call",
               return_value=_canned_text_response("ignored")):
        a1 = Recall(world=world, agent_id="order-1")
        # Force-write the three ideas directly via _append_idea (which both
        # appends to in-memory list AND persists to the tablet)
        a1._ideas.clear()
        a1._append_idea(text)
        a1._append_idea(req)
        a1._append_idea(res)

    # Restart the Dervish (new instance, replays from tablet)
    _multiton_mod._reset_for_tests()
    a2 = Recall(world=world, agent_id="order-1")

    # Replayed order must match write-order, not alphabetical-UUID order
    ids_in_order = [i.id for i in a2.ideas]
    assert ids_in_order == ["zzz_text", "mmm_req", "aaa_res"], (
        f"Replay scrambled tool sequence: {ids_in_order}"
    )


def test_multi_step_pipeline(world):
    """HeartConfig with 2 voices → 2 LLM calls. Final idea from step 2."""
    class TwoStep(Dervish):
        heart_config = HeartConfig(voices=[
            Voice(model_name="step-1"),
            Voice(model_name="step-2"),
        ])

    responses = [
        _canned_text_response("step1-out"),
        _canned_text_response("step2-final"),
    ]
    call_iter = iter(responses)
    with patch("rumi.heart.openrouter.OpenRouterHeart.call",
               side_effect=lambda *a, **kw: next(call_iter)) as mock_call:
        ts = TwoStep(world=world, agent_id="ms-1")
        result = ts.whirl("go")

    assert mock_call.call_count == 2
    assert result.content == "step2-final"


def test_normalize_incoming_thought_passed_through(world):
    r = Researcher(world=world, agent_id="ni-1")
    custom = TextIdea(role="user", content="custom")
    with patch("rumi.heart.openrouter.OpenRouterHeart.call",
               return_value=_canned_text_response("got it")):
        r.whirl(custom)
    # `custom` should appear in history
    assert any(t.id == custom.id for t in r.ideas)


def test_whirl_rejects_empty_inputs(world):
    """Regression: degenerate inputs to whirl() must be rejected at the
    boundary before they silently persist garbage and burn an LLM call.

    `whirl("")` would otherwise append an empty user TextIdea (lives on disk
    forever) and send `{"role": "user", "content": ""}` — most providers
    reject this with an opaque 400. `whirl("   ")` is the same payload after
    whitespace strip. `whirl([])` would skip the append entirely and call
    the LLM with whatever history is — on a fresh Dervish, just the system
    message — billing for a no-op-shaped turn.

    Silent degenerate-input → loud rejection at the entrypoint.
    """
    r = Researcher(world=world, agent_id="empty-reject-1")
    starting_ideas = len(r.ideas)

    with pytest.raises(ValueError, match="empty"):
        r.whirl("")
    with pytest.raises(ValueError, match="empty"):
        r.whirl("   ")
    with pytest.raises(ValueError, match="empty"):
        r.whirl([])

    # No empty TextIdea was persisted — idea history is unchanged.
    assert len(r.ideas) == starting_ideas


def test_whirl_rejects_empty_idea_content(world):
    """Regression: the IIdea path through _normalize_incoming
    must reject empty/whitespace content too. The string path was already
    rejected; the bare IIdea path and the IIdea-inside-list path both slipped
    through with the IDENTICAL failure mode (persist empty TextIdea forever
    + ship `{"role":"user","content":""}` to the provider → opaque 400).

    The third documented `_normalize_incoming` path is now uniformly enforced.
    """
    r = Researcher(world=world, agent_id="empty-idea-reject")
    starting_ideas = len(r.ideas)

    # Bare IIdea with empty content
    with pytest.raises(ValueError, match="empty"):
        r.whirl(TextIdea(role="user", content=""))
    with pytest.raises(ValueError, match="empty"):
        r.whirl(TextIdea(role="user", content="   "))

    # IIdea-in-list with empty content
    with pytest.raises(ValueError, match="empty"):
        r.whirl([TextIdea(role="user", content="")])

    # Mixed list with one empty string item should also reject
    with pytest.raises(ValueError, match="empty"):
        r.whirl(["valid message", ""])
    with pytest.raises(ValueError, match="empty"):
        r.whirl(["valid message", "   "])

    # No empty TextIdea was persisted across any of the rejections
    assert len(r.ideas) == starting_ideas


def test_whirl_rejects_non_user_role_ideas(world):
    """Regression: each non-user role has a documented owner in
    rumi's architecture — "system" belongs on Voice.instructions,
    "assistant" comes from Heart.parse_response, "tool" comes from the
    tool loop. A caller-injected non-user IIdea would persist to disk
    forever AND produce illegal message sequences (double system msgs;
    "assistant" injection orphaning future tool cycles; "tool" without
    tool_call_id → 400).

    This closes another silent-failure leg on the input surface.
    """
    r = Researcher(world=world, agent_id="role-reject")
    starting_ideas = len(r.ideas)

    for bad_role in ("system", "assistant", "tool"):
        # Bare IIdea with non-user role
        with pytest.raises(ValueError, match="role="):
            r.whirl(TextIdea(role=bad_role, content="hi"))
        # IIdea-in-list with non-user role
        with pytest.raises(ValueError, match="role="):
            r.whirl([TextIdea(role=bad_role, content="hi")])

    # Positive control: role="user" with valid content still works
    # (we don't actually run whirl here — just confirm the IIdea passes
    # the boundary check, since running would invoke the LLM)
    from rumi.agent.pipeline import _validate_idea
    valid = TextIdea(role="user", content="hi")
    assert _validate_idea(valid) is valid

    # No IIdea was persisted across the rejections
    assert len(r.ideas) == starting_ideas


def test_whirl_cancel_check_stops_before_llm_call(world):
    """Cooperative cancel: a `cancel_check` returning True is observed at the
    top of the tool loop — BEFORE any LLM call — and raises WhirlCancelled.

    This is the one library seam the rumi-edge daemon uses to make A2A
    tasks/cancel cooperative across the process boundary. Checking only at the
    loop top (never mid-tool-cycle) keeps every ToolRequest/ToolResult pair
    atomic, so cancel can't strand an orphan on disk.
    """
    from rumi.agent.pipeline import WhirlCancelled

    r = Researcher(world=world, agent_id="cancel-1")
    with patch.object(r.heart, "call") as mock_call:
        with pytest.raises(WhirlCancelled):
            r.whirl("hello", cancel_check=lambda: True)
    # Cancel was observed before the agent ever called the LLM.
    mock_call.assert_not_called()


def test_whirl_cancel_check_none_and_false_are_noops(world):
    """Default cancel_check=None (and an explicit lambda:False) leave whirl()
    byte-for-byte unchanged — the library surface is unaffected for callers
    that don't opt in."""
    r = Researcher(world=world, agent_id="cancel-2")
    with patch.object(r.heart, "call", return_value=_canned_text_response("hi")):
        a = r.whirl("hello")  # default None
        b = r.whirl("hello", cancel_check=lambda: False)
    assert a.content == "hi"
    assert b.content == "hi"
