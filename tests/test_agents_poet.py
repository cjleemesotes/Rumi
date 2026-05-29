"""Tests for the Poet preset Dervish + LeadPoet-shape tools."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from rumi.tools.decorator import _reset_for_tests as _reset_tools
from rumi.agent import registry as _agent_registry


# Schema constants (mirror what's in poet.py — schema fidelity is the contract).
REQUIRED_COMPANY_FIELDS = {
    "business", "company_linkedin", "company_website", "employee_count",
    "industry", "sub_industry", "country", "city", "state", "intent_signals",
}
REQUIRED_SIGNAL_FIELDS = {"source", "description", "url", "date", "snippet"}


@pytest.fixture(autouse=True)
def _reset_state():
    _reset_tools()
    _agent_registry._reset_for_tests()
    # Ensure poet.py registers fresh after the reset
    import importlib
    import examples.poet
    importlib.reload(examples.poet)
    yield
    _reset_tools()
    _agent_registry._reset_for_tests()


def test_poet_imports_and_has_heart_config():
    from examples import Poet
    assert Poet.heart_config is not None
    assert len(Poet.heart_config.voices) == 1


def test_poet_can_be_instantiated(tmp_path):
    from rumi import World
    from examples import Poet

    with patch("rumi.world.openai.OpenAI"):
        world = World(openrouter_api_key="sk-test", data_dir=tmp_path)
        s = Poet(world=world, agent_id="advisor-1")
    assert s.agent_type.endswith("Poet")  # qualified: examples.poet.Poet
    assert s.agent_id == "advisor-1"


def test_poet_tools_registered():
    from rumi.tools.decorator import get_tools_for
    names = {t.name for t in get_tools_for("Poet")}
    expected = {
        "find_leads_for_icp",
        "enrich_company",
        "verify_intent_signals",
        "score_lead_quality",
        "summarize_top_intent",
    }
    assert expected.issubset(names)


def test_find_leads_returns_company_output_shape():
    from examples.poet import find_leads_for_icp

    leads = find_leads_for_icp("B2B SaaS companies hiring sales", max_results=3)
    assert isinstance(leads, list)
    assert 1 <= len(leads) <= 3
    for lead in leads:
        # Every required field present
        missing = REQUIRED_COMPANY_FIELDS - set(lead.keys())
        assert not missing, f"{lead.get('business')} missing fields: {missing}"
        # intent_signals is a list of dicts with the required fields
        for signal in lead["intent_signals"]:
            signal_missing = REQUIRED_SIGNAL_FIELDS - set(signal.keys())
            assert not signal_missing, f"signal missing fields: {signal_missing}"


def test_find_leads_max_results_cap():
    """Requesting >10 should cap at 10 (or however many mocks exist)."""
    from examples.poet import find_leads_for_icp

    leads = find_leads_for_icp("anything", max_results=50)
    assert len(leads) <= 10


def test_find_leads_min_one():
    """Requesting <1 should still return at least 1 if any leads exist."""
    from examples.poet import find_leads_for_icp

    leads = find_leads_for_icp("anything", max_results=0)
    assert len(leads) >= 1


def test_enrich_company_finds_known():
    from examples.poet import enrich_company

    result = enrich_company("Northwind Analytics")
    assert result["business"] == "Northwind Analytics"
    assert result["industry"] == "Software"
    assert result["company_website"].startswith("https://")
    assert REQUIRED_COMPANY_FIELDS.issubset(result.keys())


def test_enrich_company_case_insensitive():
    from examples.poet import enrich_company

    result = enrich_company("CORAL REEF ROBOTICS")
    assert result["business"] == "Coral Reef Robotics"


def test_enrich_company_unknown_returns_stub():
    """Unknown companies return a stub with required fields + a note.

    Field values are FALSY (empty strings, 0) — not "unknown" literals —
    so the Heart has nothing to relay as fact and score_lead_quality's
    truthy checks naturally short-circuit (see test_score_lead_quality_zero_for_unknown).
    """
    from examples.poet import enrich_company

    result = enrich_company("Nonexistent Inc")
    assert result["business"] == "Nonexistent Inc"
    assert REQUIRED_COMPANY_FIELDS.issubset(result.keys())
    # Falsy values on the "fact" fields — nothing to invent from
    assert result["industry"] == ""
    assert result["country"] == ""
    assert result["intent_signals"] == []
    assert "_note" in result
    assert "Nonexistent Inc" in result["_note"]


def test_score_lead_quality_zero_for_unknown_company():
    """Honesty guarantee: a not-on-record stub from enrich_company must score
    near-zero, not earn icp_fit points just by being asked about."""
    from examples.poet import enrich_company, score_lead_quality

    stub = enrich_company("Definitely Nonexistent Co")
    result = score_lead_quality(stub)
    # business field is present and truthy (the company_name we passed in),
    # so icp_fit gets at most 4 points (1 of 10 fields). Everything else
    # must be falsy and earn 0.
    assert result["icp_fit"] <= 4
    assert result["accuracy"] == 0
    assert result["intent"] == 0
    assert result["score"] <= 4


def test_verify_intent_signals_returns_list_of_signals():
    from examples.poet import verify_intent_signals

    signals = verify_intent_signals("Northwind Analytics")
    assert isinstance(signals, list)
    assert len(signals) >= 1
    for s in signals:
        assert REQUIRED_SIGNAL_FIELDS.issubset(s.keys())


def test_verify_intent_signals_unknown_returns_empty():
    from examples.poet import verify_intent_signals

    assert verify_intent_signals("Nonexistent Inc") == []


def test_score_lead_quality_returns_breakdown():
    from examples.poet import find_leads_for_icp, score_lead_quality

    leads = find_leads_for_icp("x", max_results=1)
    score = score_lead_quality(leads[0])
    assert "score" in score
    assert "icp_fit" in score
    assert "accuracy" in score
    assert "intent" in score
    assert 0 <= score["score"] <= 100


def test_score_lead_quality_handles_bad_input():
    from examples.poet import score_lead_quality

    result = score_lead_quality("not a dict")
    assert result["score"] == 0
    assert "error" in result


def test_summarize_top_intent_one_paragraph():
    from examples.poet import find_leads_for_icp, summarize_top_intent

    leads = find_leads_for_icp("x", max_results=3)
    summary = summarize_top_intent(leads)
    assert isinstance(summary, str)
    assert len(summary) > 0
    # Should mention at least one of the lead companies' names
    assert any(lead["business"] in summary for lead in leads)


def test_summarize_top_intent_empty():
    from examples.poet import summarize_top_intent

    assert summarize_top_intent([]) == "No leads to summarize."


def test_real_network_signaled_by_env(monkeypatch):
    """When LEADPOET_NETWORK_URL is set, the find_leads path goes through the
    network helper (which currently still returns mocks but documents the seam)."""
    from examples.poet import find_leads_for_icp, _is_real_network_enabled

    monkeypatch.delenv("LEADPOET_NETWORK_URL", raising=False)
    assert _is_real_network_enabled() is False

    monkeypatch.setenv("LEADPOET_NETWORK_URL", "https://example.com/leadpoet")
    assert _is_real_network_enabled() is True
    # And calling still returns CompanyOutput-shaped dicts
    leads = find_leads_for_icp("anything", max_results=2)
    for lead in leads:
        assert REQUIRED_COMPANY_FIELDS.issubset(lead.keys())
