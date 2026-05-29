"""Tests for tool decorator + registry + execute_tool_loop."""
from __future__ import annotations

import enum

import pytest

from rumi.ideas.types import ToolRequestIdea, ToolResultIdea
from rumi.tools.decorator import (
    ToolEntry,
    _reset_for_tests,
    get_tools_for,
    tool,
)
from rumi.tools.registry import Tools, execute_tool_loop


# Module-level Enum classes for schema-resolution tests (typing.get_type_hints
# can only resolve names visible in the function's __globals__, which is the
# module — locally-defined classes inside a test function would silently
# fall through to the schema generator's `str` default).
class _Practice(str, enum.Enum):
    CONTEMPLATE = "contemplate"
    WHIRL = "whirl"
    COMPOSE = "compose"


class _Level(int, enum.Enum):
    LOW = 0
    HIGH = 1


@pytest.fixture(autouse=True)
def _reset_registry():
    """Each test starts with a clean global snapshot."""
    _reset_for_tests()
    yield
    _reset_for_tests()


def test_register_single_mind_type():
    @tool("Researcher")
    def search(query: str) -> str:
        """Search the web."""
        return f"r:{query}"

    tools = get_tools_for("Researcher")
    assert len(tools) == 1
    assert tools[0].name == "search"
    assert tools[0].agent_types == ("Researcher",)


def test_register_multiple_mind_types():
    @tool(["Researcher", "Summarizer"])
    def fetch(url: str) -> str:
        """Fetch a URL."""
        return url

    assert len(get_tools_for("Researcher")) == 1
    assert len(get_tools_for("Summarizer")) == 1
    assert len(get_tools_for("Unknown")) == 0


def test_async_tool_rejected():
    with pytest.raises(TypeError, match="async"):
        @tool("X")
        async def async_tool(q: str) -> str:  # type: ignore[no-redef]
            return q


def test_internal_tool_hidden_from_get_tools_for():
    @tool("X", internal=True)
    def hidden(q: str) -> str:
        """Hidden tool."""
        return q

    assert get_tools_for("X") == []
    assert len(get_tools_for("X", include_internal=True)) == 1


def test_schema_includes_description_from_docstring():
    @tool("X")
    def my_tool(query: str, limit: int = 5) -> str:
        """Search for things."""
        return ""

    entry = get_tools_for("X")[0]
    fn_schema = entry.schema["function"]
    assert fn_schema["name"] == "my_tool"
    assert fn_schema["description"] == "Search for things."
    assert "query" in fn_schema["parameters"]["properties"]
    assert "limit" in fn_schema["parameters"]["properties"]
    assert fn_schema["parameters"]["required"] == ["query"]


def test_schema_param_types():
    @tool("X")
    def typed(s: str, i: int, f: float, b: bool, lst: list[str]) -> None:
        return None

    props = get_tools_for("X")[0].schema["function"]["parameters"]["properties"]
    assert props["s"]["type"] == "string"
    assert props["i"]["type"] == "integer"
    assert props["f"]["type"] == "number"
    assert props["b"]["type"] == "boolean"
    assert props["lst"]["type"] == "array"
    assert props["lst"]["items"]["type"] == "string"


def test_schema_enum_param_surfaces_allowed_values():
    """An Enum-typed parameter must emit `enum: [values]` in the schema so the
    LLM sees the constraint. Without this the LLM would treat the param as a
    plain string and could pass invalid values that the Enum constructor
    then rejects at call time."""

    @tool("E")
    def chooser(practice: _Practice, level: _Level) -> None:
        return None

    props = get_tools_for("E")[0].schema["function"]["parameters"]["properties"]
    assert props["practice"]["type"] == "string"
    assert props["practice"]["enum"] == ["contemplate", "whirl", "compose"]
    assert props["level"]["type"] == "integer"
    assert props["level"]["enum"] == [0, 1]


def test_schema_skips_var_positional_and_var_keyword():
    """Regression: *args and **kwargs have no JSON-schema representation.
    Including them as scalar params publishes a parameter the function does
    NOT accept by that name — e.g. `def t(name, **kwargs)` would emit a
    required `"kwargs": {"type": "string"}`. The LLM would dutifully send
    `{"name": "...", "kwargs": "..."}`; `entry.function(**args)` then lands
    `kwargs="..."` as a keyword arg captured by the VAR_KEYWORD slot as
    `kwargs={"kwargs": "..."}`. The intended payload becomes reachable only
    via double-key; the real arguments are lost. Skip both VAR_POSITIONAL
    and VAR_KEYWORD entirely.
    """
    @tool("V")
    def variadic(name: str, *args, **kwargs) -> str:
        """A tool with variadic params."""
        return name

    schema = get_tools_for("V")[0].schema["function"]["parameters"]
    props = schema["properties"]
    # Only `name` should appear — `args` and `kwargs` are stripped
    assert set(props.keys()) == {"name"}
    assert schema.get("required") == ["name"]
    # And the schema does NOT falsely claim args/kwargs are typed scalars
    assert "args" not in props
    assert "kwargs" not in props


def test_tool_rejects_empty_agent_types_list():
    """Regression: @tool([]) registers for zero agent_types — the entry
    can never be returned by `get_tools_for(...)` since `Dervish._agent_type`
    is built from a non-empty class name. Silent dead code.

    Symmetric with HeartConfig(voices=[]): both are empty-
    collection boundary inputs that would otherwise produce unreachable
    state. Reject at decoration time, the earliest possible point.
    """
    with pytest.raises(ValueError, match="no agent_type"):
        @tool([])
        def f1() -> str:
            return "x"


def test_tool_rejects_empty_or_whitespace_agent_type_string():
    """Regression: @tool([""]) or @tool(["   "]) registers for an agent_type
    no Dervish class can ever equal (Python guarantees `cls.__name__` non-
    empty). Silent dead code — same shape as Voice instructions /
    model_name whitespace rejection.
    """
    with pytest.raises(ValueError, match="non-empty"):
        @tool([""])
        def f2() -> str:
            return "x"

    with pytest.raises(ValueError, match="non-empty"):
        @tool(["   "])
        def f3() -> str:
            return "x"

    # Mixed valid + empty: a single blank agent_type pollutes the registration.
    # If the user intends "Researcher" only, the empty entry is a typo and
    # should be caught loudly rather than registered as silent dead code.
    with pytest.raises(ValueError, match="non-empty"):
        @tool(["Researcher", ""])
        def f4() -> str:
            return "x"


def test_tool_accepts_legitimate_agent_types():
    """Confirm the constraints don't accidentally reject any valid form."""
    # Single string (becomes one-tuple)
    @tool("ValidA")
    def g1() -> str:
        return "a"

    # List with one entry
    @tool(["ValidB"])
    def g2() -> str:
        return "b"

    # List with multiple entries
    @tool(["ValidC", "ValidD"])
    def g3() -> str:
        return "c"

    # Confirm they ended up in the registry
    assert get_tools_for("ValidA")[0].name == "g1"
    assert get_tools_for("ValidB")[0].name == "g2"
    assert get_tools_for("ValidC")[0].name == "g3"


def test_tools_snapshot_isolation():
    """A Tools instance snapshots the registry at construction; later
    registrations don't appear in the existing instance."""
    @tool("X")
    def t1(q: str) -> str:
        return q

    tools_x = Tools("X")
    assert len(tools_x.public()) == 1

    @tool("X")
    def t2(q: str) -> str:
        return q

    # tools_x is a snapshot — still has only 1
    assert len(tools_x.public()) == 1

    tools_x_fresh = Tools("X")
    assert len(tools_x_fresh.public()) == 2


def test_duplicate_tool_name_raises_at_agent_construct():
    """Regression: two tools registered with the same `fn.__name__` for the
    SAME agent_type would land in `_all` together, advertise two identical
    schemas to the LLM (which has no way to disambiguate them), and silently
    drop one when routed through `_by_name` (dict overwrite). One function
    becomes dead code; the user has no warning.

    Same silent-aliasing shape closed for inscribed keys
    (_inscribed_key_for) and for agent classes
    (_AGENT_SUBCLASSES). The tool registry was the third parallel registry
    and was not previously given the same audit.

    Raise at Dervish construction (eager) instead of at whirl-time after a
    confusing LLM call to a dead function.
    """
    @tool(["DupTest"])
    def search(q: str) -> str:
        return "a"

    # Second decorator with same fn.__name__ targeting same agent_type
    def _make_b():
        @tool(["DupTest"])
        def search(q: str) -> str:  # noqa: F811
            return "b"

    _make_b()

    with pytest.raises(ValueError, match="Two tools registered as 'search'"):
        Tools("DupTest")


def test_duplicate_tool_idempotent_when_same_origin():
    """Module reload (importlib.reload) re-fires the @tool decorator on the
    SAME logical function — same (name, module, qualname, agent_types). This
    must NOT collide; otherwise every test fixture that uses reload to refresh
    a preset Dervish's tools would fail. Dedupe at snapshot append time keeps
    the registry idempotent for legitimate reload patterns while preserving
    the cross-module collision check.
    """
    @tool(["ReloadTest"])
    def search(q: str) -> str:
        return q

    # Re-decorate the SAME logical function (simulating a module reload).
    # In practice this is what importlib.reload triggers — Python rebinds
    # `search` to a fresh function object with identical __module__ and
    # __qualname__.
    @tool(["ReloadTest"])
    def search(q: str) -> str:  # noqa: F811
        return q

    # No collision — snapshot has exactly one entry, and Tools("ReloadTest")
    # constructs without error.
    tools = Tools("ReloadTest")
    assert tools.get("search") is not None
    # And exactly one schema is advertised to the LLM, not two duplicates.
    assert len([s for s in tools.schemas() if s["function"]["name"] == "search"]) == 1


def test_tools_facade_public_vs_all():
    @tool("X")
    def public_tool(q: str) -> str:
        return q

    @tool("X", internal=True)
    def internal_tool(q: str) -> str:
        return q

    tools = Tools("X")
    assert len(tools.public()) == 1
    assert len(tools.all()) == 2
    assert tools.get("internal_tool") is not None


def test_execute_tool_loop_success():
    @tool("X")
    def echo(msg: str) -> str:
        return f"got:{msg}"

    tools = Tools("X")
    req = ToolRequestIdea(content=[
        {"id": "call_1", "type": "function", "function": {"name": "echo", "arguments": '{"msg":"hi"}'}}
    ])
    results = execute_tool_loop(req, tools)
    assert len(results) == 1
    assert results[0].content == "got:hi"
    assert results[0].tool_call_id == "call_1"
    assert results[0].role == "tool"


def test_execute_tool_loop_unknown_tool():
    tools = Tools("X")
    req = ToolRequestIdea(content=[
        {"id": "c", "type": "function", "function": {"name": "missing", "arguments": "{}"}}
    ])
    results = execute_tool_loop(req, tools)
    assert results[0].content.startswith("ERROR:")
    assert "missing" in results[0].content


def test_execute_tool_loop_catches_exceptions():
    @tool("X")
    def boom(q: str) -> str:
        raise ValueError("kaboom")

    tools = Tools("X")
    req = ToolRequestIdea(content=[
        {"id": "c", "type": "function", "function": {"name": "boom", "arguments": '{"q":"x"}'}}
    ])
    results = execute_tool_loop(req, tools)
    assert results[0].content.startswith("ERROR:")
    assert "kaboom" in results[0].content


def test_execute_tool_loop_serializes_complex_result():
    @tool("X")
    def list_tool(q: str) -> list:
        return [1, 2, 3]

    tools = Tools("X")
    req = ToolRequestIdea(content=[
        {"id": "c", "type": "function", "function": {"name": "list_tool", "arguments": '{"q":"x"}'}}
    ])
    results = execute_tool_loop(req, tools)
    assert results[0].content == "[1, 2, 3]"


def test_execute_tool_loop_multiple_calls_sequential():
    call_order = []

    @tool("X")
    def step_a(n: int) -> str:
        call_order.append(("a", n))
        return f"a:{n}"

    @tool("X")
    def step_b(n: int) -> str:
        call_order.append(("b", n))
        return f"b:{n}"

    tools = Tools("X")
    req = ToolRequestIdea(content=[
        {"id": "1", "type": "function", "function": {"name": "step_a", "arguments": '{"n":1}'}},
        {"id": "2", "type": "function", "function": {"name": "step_b", "arguments": '{"n":2}'}},
    ])
    results = execute_tool_loop(req, tools)
    assert call_order == [("a", 1), ("b", 2)]
    assert [r.content for r in results] == ["a:1", "b:2"]


def test_tools_schemas_for_llm_payload():
    @tool("X")
    def t1(q: str) -> str:
        return q

    tools = Tools("X")
    schemas = tools.schemas()
    assert len(schemas) == 1
    assert schemas[0]["function"]["name"] == "t1"
