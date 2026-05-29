"""Tests for World container."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rumi.world import OPENROUTER_BASE_URL, World
from rumi.tablet.backend import ITablet


def test_node_defaults_from_no_env(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("RUMI_DATA_DIR", raising=False)
    monkeypatch.delenv("RUMI_LOG_LEVEL", raising=False)
    monkeypatch.delenv("OPENROUTER_HTTP_REFERER", raising=False)
    monkeypatch.delenv("OPENROUTER_X_TITLE", raising=False)

    world = World()
    assert world.api_key is None
    assert world.data_dir == Path("./data")
    assert world.log_level == "INFO"
    assert world.http_referer is None
    assert world.x_title is None
    assert world.base_url == OPENROUTER_BASE_URL


def test_node_env_resolution(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("RUMI_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RUMI_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("OPENROUTER_HTTP_REFERER", "https://example.com")
    monkeypatch.setenv("OPENROUTER_X_TITLE", "my-app")

    world = World()
    assert world.api_key == "sk-or-test"
    assert world.data_dir == tmp_path
    assert world.log_level == "DEBUG"
    assert world.http_referer == "https://example.com"
    assert world.x_title == "my-app"


def test_node_kwargs_override_env(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    world = World(openrouter_api_key="kwarg-key", data_dir=tmp_path)
    assert world.api_key == "kwarg-key"
    assert world.data_dir == tmp_path


def test_storage_for_returns_filesystem(tmp_path):
    world = World(data_dir=tmp_path)
    tablet = world.tablet_for("Researcher", "r-1")
    # The World contract is "returns a scoped ITablet" — assert the interface,
    # not the concrete class.
    assert isinstance(tablet, ITablet)
    assert tablet.agent_type == "Researcher"
    assert tablet.agent_id == "r-1"


def test_custom_storage_factory(tmp_path):
    calls = []

    def fake_factory(agent_type: str, agent_id: str):
        calls.append((agent_type, agent_id))
        return MagicMock()

    world = World(data_dir=tmp_path, tablet_factory=fake_factory)
    world.tablet_for("X", "y")
    assert calls == [("X", "y")]


def test_openrouter_client_lazy_construction(tmp_path):
    """Mock openai.OpenAI; verify it's only built on first .openrouter_client access."""
    with patch("rumi.world.openai.OpenAI") as MockOpenAI:
        MockOpenAI.return_value = MagicMock()
        world = World(
            openrouter_api_key="sk-or-test",
            data_dir=tmp_path,
            http_referer="https://example.com",
            x_title="ttl",
        )
        # Not yet called
        assert MockOpenAI.call_count == 0

        client = world.openrouter_client
        assert MockOpenAI.call_count == 1
        kwargs = MockOpenAI.call_args.kwargs
        assert kwargs["base_url"] == OPENROUTER_BASE_URL
        assert kwargs["api_key"] == "sk-or-test"
        assert kwargs["default_headers"]["HTTP-Referer"] == "https://example.com"
        assert kwargs["default_headers"]["X-Title"] == "ttl"

        # Second access: cached
        client2 = world.openrouter_client
        assert client is client2
        assert MockOpenAI.call_count == 1


def test_openrouter_client_no_attribution_headers(tmp_path):
    """When no http_referer/x_title, default_headers should be absent (or empty)."""
    with patch("rumi.world.openai.OpenAI") as MockOpenAI:
        world = World(openrouter_api_key="k", data_dir=tmp_path)
        _ = world.openrouter_client
        kwargs = MockOpenAI.call_args.kwargs
        assert "default_headers" not in kwargs or not kwargs.get("default_headers")


def test_openrouter_client_missing_key_does_not_raise_yet(tmp_path, monkeypatch):
    """No api key → openai SDK constructed with 'missing' sentinel; failure at call time."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with patch("rumi.world.openai.OpenAI") as MockOpenAI:
        world = World(data_dir=tmp_path)
        world.openrouter_client  # should not raise
        assert MockOpenAI.call_args.kwargs["api_key"] == "missing"


def test_logger_for_returns_scoped(tmp_path):
    world = World(data_dir=tmp_path)
    log = world.logger_for("X", "y")
    assert log.name == "rumi.X.y"
