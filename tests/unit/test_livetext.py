"""Unit tests for spells.livetext: the reconciliation and the live insertion session.

Spec 6 (live partials), 11 (delivery) and decision B5-57. Every platform call is faked with
the same backends the delivery tests use, so nothing here types a key, reads the clipboard
or looks at a real window.
"""

from __future__ import annotations

import pytest

from spells.livetext import (
    TAG,
    LiveSession,
    common_prefix_len,
    live_typing_allowed,
    reconcile,
)
from spells.models import TargetContext

from .test_inject import OTHER_HWND, TARGET_HWND
from .test_inject import make as make_backends

CTX = TargetContext(window=TARGET_HWND, process="notepad.exe", title="Untitled", captured_at=1.0)


def make(**kwargs):
    log, backends = make_backends(**kwargs)
    return log, backends, LiveSession(CTX, backends)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("", "", 0),
        ("hello", "hello there", 5),
        ("hello", "help", 3),
    ],
)
def test_common_prefix_len(left, right, expected):
    assert common_prefix_len(left, right) == expected


def test_reconcile_rewrite_deletes_from_the_divergence():
    assert reconcile("i think we", "I think we") == (10, "I think we")


def test_reconcile_to_nothing_deletes_everything():
    assert reconcile("hello", "") == (5, "")


@pytest.mark.parametrize(
    ("profile", "enabled", "everywhere", "expected"),
    [
        ("Default", True, False, True),
        ("Code", True, False, False),
        ("Code", True, True, True),
        ("Chat", False, False, False),
    ],
)
def test_live_typing_allowed(profile, enabled, everywhere, expected):
    assert live_typing_allowed(profile, enabled=enabled, everywhere=everywhere) is expected


def test_first_partial_is_typed_and_counted():
    _log, backends, session = make()
    assert session.apply("hello there") is True
    assert session.inserted == "hello there"
    assert backends.keyboard.typed == [
        ("hello there", {"extra_info": TAG, "release_modifiers": True})
    ]
    assert backends.keyboard.backspaced == []


def test_a_growing_partial_types_only_the_tail():
    _log, backends, session = make()
    session.apply("hello")
    session.apply("hello there")
    assert [text for text, _ in backends.keyboard.typed] == ["hello", " there"]
    assert backends.keyboard.backspaced == []
    assert session.inserted == "hello there"


def test_a_rewritten_word_costs_exactly_its_characters():
    _log, backends, session = make()
    session.apply("meet at ten")
    session.apply("meet at two")
    assert backends.keyboard.backspaced == [(2, {"extra_info": TAG, "release_modifiers": True})]
    assert [text for text, _ in backends.keyboard.typed] == ["meet at ten", "wo"]
    assert session.inserted == "meet at two"


def test_an_unchanged_partial_sends_nothing():
    _log, backends, session = make()
    session.apply("hello")
    backends.keyboard.typed.clear()
    assert session.apply("hello") is True
    assert backends.keyboard.typed == []
    assert backends.keyboard.backspaced == []


def test_focus_moved_stops_the_session_and_removes_nothing():
    _log, backends, session = make()
    session.apply("hello")
    backends.focus.foreground_window = OTHER_HWND
    assert session.apply("hello there") is False
    assert session.stopped is True
    assert session.stop_reason == "focus moved"
    assert session.inserted == "hello"
    assert [text for text, _ in backends.keyboard.typed] == ["hello"]
    assert backends.keyboard.backspaced == []


def test_an_elevated_foreground_stops_the_session():
    _log, backends, session = make()
    backends.focus.elevated = True
    assert session.apply("hello") is False
    assert session.stop_reason == "foreground window is elevated"
    assert backends.keyboard.typed == []


def test_a_stopped_session_sends_nothing_more():
    _log, backends, session = make()
    session.stop("test")
    assert session.apply("hello") is False
    assert backends.keyboard.typed == []
    assert session.inserted == ""


def test_a_refused_backspace_stops_the_session_and_keeps_the_bookkeeping():
    _log, backends, session = make()
    session.apply("meet at ten")
    backends.keyboard.error_on.add("send_backspaces")
    assert session.apply("meet at two") is False
    assert session.stop_reason == "backspace refused"
    assert session.inserted == "meet at ten"
    assert [text for text, _ in backends.keyboard.typed] == ["meet at ten"]


def test_refused_typing_stops_the_session_without_counting_the_text():
    _log, _backends, session = make(input_error=("type_unicode",))
    assert session.apply("hello") is False
    assert session.stop_reason == "typing refused"
    assert session.inserted == ""


def test_the_clipboard_is_never_touched():
    log, _backends, session = make()
    session.apply("hello there")
    session.apply("hello two")
    session.apply("")
    assert not [entry for entry in log if entry[0] in ("set_text", "snapshot", "restore")]
