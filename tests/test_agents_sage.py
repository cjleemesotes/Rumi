"""Tests for the Sage preset Dervish. Mocks botasaurus entirely — no live browser."""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rumi.tools.decorator import _reset_for_tests as _reset_tools
from rumi.agent import registry as _agent_registry


# --- Fixtures ---


@pytest.fixture(autouse=True)
def _reset_state():
    _reset_tools()
    _agent_registry._reset_for_tests()
    yield
    _reset_tools()
    _agent_registry._reset_for_tests()


@pytest.fixture
def fake_botasaurus(monkeypatch):
    """Inject a fake botasaurus.browser module BEFORE Sage's tools run.

    Returns the mock Driver class so tests can configure return values.
    """
    # Build a fake driver class with the methods Sage uses
    driver_mock = MagicMock()
    driver_mock.title = "Fake Page"
    driver_mock.get_text = MagicMock(return_value="Hello from the fake page.")
    driver_mock.select_all = MagicMock(return_value=[])
    driver_mock.type = MagicMock(return_value=None)
    driver_mock.click = MagicMock(return_value=None)
    driver_mock.current_url = "https://example.com/after-click"
    driver_mock.get = MagicMock(return_value=None)
    driver_mock.google_get = MagicMock(return_value=None)

    # Build the fake @browser decorator: returns a callable that runs `fn`
    # synchronously with the driver_mock + data dict, exactly like botasaurus.
    def fake_browser(**decorator_kwargs):
        def wrap(fn):
            def runner(data):
                return fn(driver_mock, data)
            return runner
        return wrap

    fake_module = types.ModuleType("botasaurus.browser")
    fake_module.browser = fake_browser
    fake_module.Driver = MagicMock
    monkeypatch.setitem(sys.modules, "botasaurus", types.ModuleType("botasaurus"))
    monkeypatch.setitem(sys.modules, "botasaurus.browser", fake_module)

    return driver_mock


# --- Tests ---


def test_sage_imports_and_has_heart_config():
    from examples import Sage
    assert Sage.heart_config is not None
    assert len(Sage.heart_config.voices) == 1


def test_sage_can_be_instantiated(tmp_path):
    from rumi import World
    from examples import Sage

    with patch("rumi.world.openai.OpenAI"):
        world = World(openrouter_api_key="sk-test", data_dir=tmp_path)
        w = Sage(world=world, agent_id="walker-1")
    assert w.agent_type.endswith("Sage")  # qualified: examples.sage.Sage
    assert w.agent_id == "walker-1"


def test_tools_registered_for_sage():
    """All 7 default tools should be registered under agent_type 'Sage'."""
    # Force a fresh import so the @tool registrations land in the now-empty snapshot
    import importlib
    import examples.sage
    importlib.reload(examples.sage)

    from rumi.tools.decorator import get_tools_for
    names = {t.name for t in get_tools_for("Sage")}
    expected = {
        "visit_page", "extract_text", "extract_all_text", "extract_links",
        "fill_field", "click_element", "search_google",
    }
    assert expected.issubset(names)


def test_visit_page_calls_driver(fake_botasaurus):
    import importlib
    import examples.sage
    importlib.reload(examples.sage)
    from examples.sage import visit_page

    out = visit_page("https://example.com")
    fake_botasaurus.get.assert_called_with("https://example.com")
    assert "Fake Page" in out
    assert "Hello from the fake page." in out


def test_extract_text_returns_selector_content(fake_botasaurus):
    import importlib
    import examples.sage
    importlib.reload(examples.sage)
    from examples.sage import extract_text

    fake_botasaurus.get_text.return_value = "Heading text"
    out = extract_text("https://example.com", "h1")
    # The driver's get_text should have been called with the selector
    assert fake_botasaurus.get_text.call_args.args[0] == "h1"
    assert out == "Heading text"


def test_extract_all_text(fake_botasaurus):
    import importlib
    import examples.sage
    importlib.reload(examples.sage)
    from examples.sage import extract_all_text

    fake_botasaurus.get_text.return_value = "Body content"
    out = extract_all_text("https://example.com")
    assert out == "Body content"


def test_extract_links(fake_botasaurus):
    import importlib
    import examples.sage
    importlib.reload(examples.sage)
    from examples.sage import extract_links

    a1 = MagicMock()
    a1.get_attribute = MagicMock(return_value="https://a.example/")
    a1.text = "Alpha"
    a2 = MagicMock()
    a2.get_attribute = MagicMock(return_value="https://b.example/")
    a2.text = "  Beta  "
    fake_botasaurus.select_all.return_value = [a1, a2]

    out = extract_links("https://example.com")
    assert out == [
        {"text": "Alpha", "href": "https://a.example/"},
        {"text": "Beta", "href": "https://b.example/"},
    ]


def test_fill_field(fake_botasaurus):
    import importlib
    import examples.sage
    importlib.reload(examples.sage)
    from examples.sage import fill_field

    out = fill_field("https://example.com", "input[name=q]", "saturn")
    fake_botasaurus.type.assert_called_with("input[name=q]", "saturn")
    assert out == "ok"


def test_click_element(fake_botasaurus):
    import importlib
    import examples.sage
    importlib.reload(examples.sage)
    from examples.sage import click_element

    fake_botasaurus.current_url = "https://example.com/next"
    out = click_element("https://example.com", "a.next")
    fake_botasaurus.click.assert_called_with("a.next")
    assert out == "https://example.com/next"


def test_search_google_uses_google_get(fake_botasaurus):
    import importlib
    import examples.sage
    importlib.reload(examples.sage)
    from examples.sage import search_google

    r1 = MagicMock()
    r1.get_attribute = MagicMock(return_value="https://result1.example/")
    r1.text = "Result 1"
    r2 = MagicMock()
    r2.get_attribute = MagicMock(return_value="https://result2.example/")
    r2.text = "Result 2"
    fake_botasaurus.select_all.return_value = [r1, r2]

    out = search_google("saturn moons", max_results=5)
    fake_botasaurus.google_get.assert_called_with("saturn moons")
    assert out == [
        {"title": "Result 1", "href": "https://result1.example/"},
        {"title": "Result 2", "href": "https://result2.example/"},
    ]


def test_search_google_caps_max_results(fake_botasaurus):
    import importlib
    import examples.sage
    importlib.reload(examples.sage)
    from examples.sage import search_google

    # 15 fake results, but tool should cap at 10 and we ask for 5
    results = []
    for i in range(15):
        r = MagicMock()
        r.get_attribute = MagicMock(return_value=f"https://r{i}.example/")
        r.text = f"r{i}"
        results.append(r)
    fake_botasaurus.select_all.return_value = results

    out = search_google("query", max_results=5)
    assert len(out) == 5


def test_botasaurus_missing_raises_informative_error(monkeypatch):
    """If botasaurus is not importable, the tool should raise ImportError with a hint."""
    # Make sure botasaurus is not in sys.modules
    monkeypatch.delitem(sys.modules, "botasaurus", raising=False)
    monkeypatch.delitem(sys.modules, "botasaurus.browser", raising=False)

    # Stub the import system so botasaurus appears unimportable
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("botasaurus"):
            raise ImportError("simulated missing botasaurus")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    import importlib
    import examples.sage
    importlib.reload(examples.sage)
    from examples.sage import visit_page

    with pytest.raises(ImportError, match="botasaurus"):
        visit_page("https://example.com")
