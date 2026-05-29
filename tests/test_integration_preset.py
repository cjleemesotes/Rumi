"""Cross-Dervish integration: Poet enriches, Sage scouts the web.

Both Dervishes are mocked at their external surfaces: botasaurus is replaced
with a fake module, openai.OpenAI is patched. No network, no browser.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rumi.agent import registry as _agent_registry
from rumi.tools.decorator import _reset_for_tests as _reset_tools


@pytest.fixture(autouse=True)
def _reset_state():
    _reset_tools()
    _agent_registry._reset_for_tests()
    # Fresh poet + sage registrations — reload the leaf modules AND the
    # examples package itself so its re-exports point at the fresh classes.
    import importlib
    import examples
    import examples.poet
    import examples.sage
    importlib.reload(examples.poet)
    importlib.reload(examples.sage)
    importlib.reload(examples)
    yield
    _reset_tools()
    _agent_registry._reset_for_tests()


@pytest.fixture
def fake_botasaurus(monkeypatch):
    driver_mock = MagicMock()
    driver_mock.title = "Northwind Analytics"
    driver_mock.get_text = MagicMock(return_value="We help B2B sales teams scale outbound.")
    driver_mock.select_all = MagicMock(return_value=[])
    driver_mock.type = MagicMock()
    driver_mock.click = MagicMock()
    driver_mock.current_url = "https://northwindanalytics.com/"
    driver_mock.get = MagicMock()
    driver_mock.google_get = MagicMock()

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

    import importlib
    import examples
    import examples.sage
    importlib.reload(examples.sage)
    importlib.reload(examples)
    return driver_mock


def test_prospect_demo_full_flow(tmp_path, fake_botasaurus, monkeypatch):
    """Poet enriches → Sage visits the website. Schema fidelity end-to-end."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("RUMI_DATA_DIR", str(tmp_path))

    with patch("rumi.world.openai.OpenAI"):
        from examples.prospect_demo import prospect_demo
        out = prospect_demo("Northwind Analytics", "https://northwindanalytics.com")

    # Poet's enrichment — LeadPoet shape preserved
    company = out["poet_company"]
    required = {
        "business", "company_linkedin", "company_website", "employee_count",
        "industry", "sub_industry", "country", "city", "state", "intent_signals",
    }
    assert required.issubset(company.keys())
    assert company["business"] == "Northwind Analytics"

    # Poet's score
    score = out["poet_score"]
    assert "score" in score
    assert 0 <= score["score"] <= 100

    # Sage's observation
    assert "sage_observation" in out
    assert "Northwind Analytics" in out["sage_observation"]


def test_cross_agent_dispatch_via_get_agent(tmp_path, fake_botasaurus, monkeypatch):
    """Verify Poet.get_agent('Sage', ...) returns a Sage on the same World."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    with patch("rumi.world.openai.OpenAI"):
        from rumi import World
        from examples import Poet, Sage

        world = World(data_dir=tmp_path)
        poet = Poet(world=world, agent_id="p1")
        sage = poet.get_agent("Sage", "s1")

    assert isinstance(sage, Sage)
    assert sage.world is world
    # Re-lookup returns the same instance
    sage2 = poet.get_agent("Sage", "s1")
    assert sage is sage2


def test_prospect_demo_unknown_company(tmp_path, fake_botasaurus, monkeypatch):
    """Poet gracefully returns a stub for unknown companies."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    with patch("rumi.world.openai.OpenAI"):
        from examples.prospect_demo import prospect_demo
        out = prospect_demo("Nonexistent Inc", "https://unknown.example")

    company = out["poet_company"]
    assert company["business"] == "Nonexistent Inc"
    # Falsy "not on record" stub — empty strings, not "unknown" literals
    assert company["industry"] == ""
    assert "_note" in company
    # Score must be near-zero for unknowns (honesty guarantee — see
    # test_score_lead_quality_zero_for_unknown_company in test_agents_poet.py)
    assert out["poet_score"]["score"] <= 4
