"""Unit tests for photofinder.webui multi-user safety."""

from __future__ import annotations

import threading

import numpy as np
import pytest


def test_cancel_events_are_per_session():
    """Two sessions' cancel events must not interfere with each other."""
    ev_a = threading.Event()
    ev_b = threading.Event()

    ev_a.set()
    assert ev_a.is_set()
    assert not ev_b.is_set()

    ev_b.set()
    assert ev_b.is_set()

    ev_a.clear()
    assert not ev_a.is_set()
    assert ev_b.is_set()


def test_cancel_search_sets_event():
    from photofinder.webui import cancel_search

    ev = threading.Event()
    cancel_search(ev)
    assert ev.is_set()


def test_cancel_search_handles_none():
    from photofinder.webui import cancel_search

    cancel_search(None)  # should not raise


def test_auth_rejects_wrong_code(monkeypatch):
    monkeypatch.setattr("photofinder.webui.ACCESS_CODE", "secret123")

    # Re-import to pick up the patched ACCESS_CODE in the closure
    import photofinder.webui as webui_mod
    code = webui_mod.ACCESS_CODE

    def auth(username, password):
        return password == code

    assert auth("anyone", "secret123") is True
    assert auth("anyone", "wrong") is False
    assert auth("", "secret123") is True


# ── batch-download source fallback ────────────────────────────────
class _Resp:
    def __init__(self, content=b"", status=200):
        self.content = content
        self._status = status

    def raise_for_status(self):
        if self._status >= 400:
            raise RuntimeError(f"HTTP {self._status}")


class _Req:
    """Fake requests module: url → _Resp mapping."""

    def __init__(self, mapping):
        self.mapping = mapping
        self.called = []

    def get(self, url, timeout=None):
        self.called.append(url)
        return self.mapping.get(url, _Resp(status=404))


class _R:
    def __init__(self, **kw):
        self.photo_id = kw.get("photo_id", 1)
        self.full_url = kw.get("full_url", "http://x/full")
        self.detect_url = kw.get("detect_url", "http://x/detect")
        self.preview_url = kw.get("preview_url", "http://x/preview")
        self.thumb_path = kw.get("thumb_path", "")


def test_fetch_prefers_original():
    from photofinder.webui import _fetch_best_source

    req = _Req({"http://x/full": _Resp(b"ORIG")})
    _, content = _fetch_best_source(_R(), req)
    assert content == b"ORIG"


def test_fetch_falls_back_to_local_thumb(tmp_path):
    from photofinder.webui import _fetch_best_source

    thumb = tmp_path / "1.jpg"
    thumb.write_bytes(b"THUMB")
    # full 403s (not in mapping) → local thumb wins, no further network calls
    req = _Req({"http://x/detect": _Resp(b"DET")})
    _, content = _fetch_best_source(_R(thumb_path=str(thumb)), req)
    assert content == b"THUMB"
    # only the original was attempted; thumb came from disk, no fallback GETs
    assert req.called == ["http://x/full"]


def test_fetch_falls_back_to_detect_url():
    from photofinder.webui import _fetch_best_source

    req = _Req({"http://x/detect": _Resp(b"DET")})
    _, content = _fetch_best_source(_R(thumb_path=""), req)
    assert content == b"DET"


def test_fetch_falls_back_to_preview_url():
    from photofinder.webui import _fetch_best_source

    req = _Req({"http://x/preview": _Resp(b"PREV")})
    _, content = _fetch_best_source(_R(thumb_path="", detect_url=""), req)
    assert content == b"PREV"


def test_fetch_raises_when_no_source():
    from photofinder.webui import _fetch_best_source

    req = _Req({})  # everything 404s, no thumb
    with pytest.raises(RuntimeError):
        _fetch_best_source(_R(thumb_path=""), req)
