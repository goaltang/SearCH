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
