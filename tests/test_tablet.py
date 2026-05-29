"""Tests for FilesystemTablet + TabletScopeMixin."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rumi.tablet import FilesystemTablet, TabletScopeMixin
from rumi.tablet.filesystem import _snake_case


@pytest.fixture
def tmp_storage(tmp_path: Path) -> FilesystemTablet:
    return FilesystemTablet(tmp_path, agent_type="Researcher", agent_id="r-1")


def test_set_get_roundtrip(tmp_storage):
    tmp_storage.set("last_query", "saturn")
    assert tmp_storage.get("last_query") == "saturn"


def test_get_missing_returns_default(tmp_storage):
    assert tmp_storage.get("nonexistent") is None
    assert tmp_storage.get("nonexistent", "default") == "default"


def test_set_complex_value(tmp_storage):
    val = {"findings": ["a", "b"], "count": 2}
    tmp_storage.set("research_state", val)
    assert tmp_storage.get("research_state") == val


def test_delete(tmp_storage):
    tmp_storage.set("k", "v")
    assert tmp_storage.delete("k") is True
    assert tmp_storage.get("k") is None
    assert tmp_storage.delete("k") is False  # already gone


def test_list_keys_empty(tmp_storage):
    assert tmp_storage.list_keys() == []


def test_list_keys_with_prefix(tmp_storage):
    tmp_storage.set("a", 1)
    tmp_storage.set("ab", 2)
    tmp_storage.set("xyz", 3)
    assert tmp_storage.list_keys("a") == ["a", "ab"]
    assert tmp_storage.list_keys("") == ["a", "ab", "xyz"]


def test_scope_isolation(tmp_path):
    s1 = FilesystemTablet(tmp_path, "Researcher", "r-1")
    s2 = FilesystemTablet(tmp_path, "Researcher", "r-2")
    s3 = FilesystemTablet(tmp_path, "Summarizer", "r-1")
    s1.set("k", "v1")
    s2.set("k", "v2")
    s3.set("k", "v3")
    assert s1.get("k") == "v1"
    assert s2.get("k") == "v2"
    assert s3.get("k") == "v3"


def test_prefix_mixin_rejects_empty(tmp_path):
    with pytest.raises(ValueError):
        FilesystemTablet(tmp_path, "", "r-1")
    with pytest.raises(ValueError):
        FilesystemTablet(tmp_path, "X", "")


def test_prefix_mixin_rejects_slashes(tmp_path):
    with pytest.raises(ValueError):
        FilesystemTablet(tmp_path, "X/Y", "r-1")
    with pytest.raises(ValueError):
        FilesystemTablet(tmp_path, "X", "r/1")


def test_scope_property(tmp_storage):
    assert tmp_storage.scope == "Researcher/r-1"
    assert tmp_storage.agent_type == "Researcher"
    assert tmp_storage.agent_id == "r-1"


def test_set_failure_leaves_prior_value(tmp_path, tmp_storage, monkeypatch):
    """A failed set must leave the prior value intact. Serialization happens
    in Python before the native core writes anything, so a serialization error
    never reaches disk; the Rust core's own write is atomic (tempfile + rename)."""
    tmp_storage.set("k", "original")

    import json as json_mod

    def boom(*args, **kw):
        raise IOError("serialize failed")

    monkeypatch.setattr(json_mod, "dumps", boom)

    with pytest.raises(IOError):
        tmp_storage.set("k", "updated")

    monkeypatch.undo()
    assert tmp_storage.get("k") == "original"


def test_corrupted_file_returns_default(tmp_storage):
    # Manually write garbage
    p = tmp_storage._path("k")
    p.write_text("not json {{{")
    assert tmp_storage.get("k", "default") == "default"


def test_snake_case():
    assert _snake_case("ResearchState") == "research_state"
    assert _snake_case("Summary") == "summary"
    assert _snake_case("MyHTTPClient") == "my_http_client"
    assert _snake_case("ABC") == "abc"


def test_safe_filename_special_chars(tmp_storage):
    """Keys with special chars should still work via safe-filename encoding."""
    tmp_storage.set("history/abc-123", {"x": 1})
    assert tmp_storage.get("history/abc-123") == {"x": 1}
    keys = tmp_storage.list_keys()
    assert "history/abc-123" in keys


def test_safe_filename_injective_slash_vs_double_underscore(tmp_storage):
    """Regression: "a/b" and "a__b" must NOT collide on disk.

    The old encoder treated `_` as a passthrough alnum char AND used `__` to
    mean "/" — so `_safe_filename("a/b") == _safe_filename("a__b") == "a__b"`.
    A caller writing both keys would have one silently overwrite the other,
    and list_keys() would report only the slash-form regardless of what was
    stored. The injective encoder escapes literal `_` as `%5f` so `__`
    stays reserved for the slash separator alone.
    """
    tmp_storage.set("a/b", {"slash": True})
    tmp_storage.set("a__b", {"underscores": True})

    # Both keys must coexist with their own values
    assert tmp_storage.get("a/b") == {"slash": True}
    assert tmp_storage.get("a__b") == {"underscores": True}

    # list_keys reports BOTH distinct keys
    keys = tmp_storage.list_keys()
    assert "a/b" in keys
    assert "a__b" in keys


def test_codec_matches_golden_vectors():
    """The native filename codec must stay byte-identical to the committed
    golden vectors (frozen from the original Python codec). This is the lasting
    regression guard now that the codec lives in Rust."""
    import rumi._core as core

    golden = json.loads(
        (Path(__file__).parent / "golden" / "tablet_codec.json").read_text(encoding="utf-8")
    )
    for v in golden:
        assert core.safe_filename(v["key"]) == v["file"], v["key"]
        assert core.decode_safe_filename(v["file"]) == v["decoded"], v["file"]
