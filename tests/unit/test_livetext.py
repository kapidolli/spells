"""Unit tests for spells.livetext: the reconciliation and the live insertion session.

Spec 6 (live partials), 11 (delivery) and decision B5-57. Every Win32 call is faked with
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

CTX = TargetContext(hwnd=TARGET_HWND, process="notepad.exe", title="Untitled", captured_at=1.0)


def make(**kwargs):
    log, backends = make_backends(**kwargs)
    return log, backends, LiveSession(CTX, backends)


# The pure half -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("", "", 0),
        ("", "hello", 0),
        ("hello", "hello there", 5),
        ("hello there", "hello", 5),
        ("hello", "help", 3),
        ("abc", "xyz", 0),
    ],
)
def test_common_prefix_len(left, right, expected):
    assert common_prefix_len(left, right) == expected


def test_reconcile_growing_text_costs_no_backspace():
    assert reconcile("hello", "hello there") == (0, " there")


def test_reconcile_rewrite_deletes_from_the_divergence():
    assert reconcile("i think we", "I think we") == (10, "I think we")


def test_reconcile_a_rewritten_later_word_keeps_the_prefix():
    backspaces, tail = reconcile("meet at ten", "meet at two")
    assert (backspaces, tail) == (2, "wo")


def test_reconcile_shorter_target_only_deletes():
    assert reconcile("hello there", "hello") == (6, "")


def test_reconcile_to_nothing_deletes_everything():
    assert reconcile("hello", "") == (5, "")


@pytest.mark.parametrize(
    ("profile", "enabled", "everywhere", "expected"),
    [
        ("Default", True, False, True),
        ("Chat", True, False, True),
        ("Email and docs", True, False, True),
        ("Code", True, False, False),
        ("Terminal", True, False, False),
        ("Code", True, True, True),
        ("Terminal", True, True, True),
        ("Chat", False, False, False),
        ("Code", False, True, False),
    ],
)
def test_live_typing_allowed(profile, enabled, everywhere, expected):
    assert live_typing_allowed(profile, enabled=enabled, everywhere=everywhere) is expected


# The session -------------------------------------------------------------------------


def test_first_partial_is_typed_and_counted():
    _log, backends, session = make()
    assert session.apply("hello there") is True
    assert session.inserted == "hello there"
    assert backends.input.typed == [("hello there", {"extra_info": TAG, "release_modifiers": True})]
    assert backends.input.backspaced == []


def test_a_growing_partial_types_only_the_tail():
    _log, backends, session = make()
    session.apply("hello")
    session.apply("hello there")
    assert [text for text, _ in backends.input.typed] == ["hello", " there"]
    assert backends.input.backspaced == []
    assert session.inserted == "hello there"


def test_a_rewritten_word_costs_exactly_its_characters():
    _log, backends, session = make()
    session.apply("meet at ten")
    session.apply("meet at two")
    assert backends.input.backspaced == [(2, {"extra_info": TAG, "release_modifiers": True})]
    assert [text for text, _ in backends.input.typed] == ["meet at ten", "wo"]
    assert session.inserted == "meet at two"


def test_an_unchanged_partial_sends_nothing():
    _log, backends, session = make()
    session.apply("hello")
    backends.input.typed.clear()
    assert session.apply("hello") is True
    assert backends.input.typed == []
    assert backends.input.backspaced == []


def test_the_backspace_count_never_exceeds_what_was_typed():
    _log, backends, session = make()
    session.apply("hello")
    session.apply("")
    session.apply("")
    assert [count for count, _ in backends.input.backspaced] == [5]
    assert session.inserted == ""


def test_the_foreground_is_checked_before_the_backspaces_and_before_the_text():
    log, _backends, session = make()
    session.apply("meet at ten")
    log.clear()
    session.apply("meet at two")
    assert [entry[0] for entry in log].count("foreground_hwnd") == 2


def test_focus_moved_stops_the_session_and_removes_nothing():
    _log, backends, session = make()
    session.apply("hello")
    backends.window.foreground = OTHER_HWND
    assert session.apply("hello there") is False
    assert session.stopped is True
    assert session.stop_reason == "focus moved"
    assert session.inserted == "hello"
    assert [text for text, _ in backends.input.typed] == ["hello"]
    assert backends.input.backspaced == []


def test_an_elevated_foreground_stops_the_session():
    _log, backends, session = make()
    backends.window.elevated = True
    assert session.apply("hello") is False
    assert session.stop_reason == "foreground window is elevated"
    assert backends.input.typed == []


def test_no_target_window_stops_the_session():
    _log, backends, session = make()
    backends.window.foreground = 0
    assert session.apply("hello") is False
    assert session.stop_reason == "no target window"


def test_a_stopped_session_sends_nothing_more():
    _log, backends, session = make()
    session.stop("test")
    assert session.apply("hello") is False
    assert backends.input.typed == []
    assert session.inserted == ""


def test_a_refused_backspace_stops_the_session_and_keeps_the_bookkeeping():
    _log, backends, session = make()
    session.apply("meet at ten")
    backends.input.error_on.add("send_backspaces")
    assert session.apply("meet at two") is False
    assert session.stop_reason == "backspace refused"
    assert session.inserted == "meet at ten"
    assert [text for text, _ in backends.input.typed] == ["meet at ten"]


def test_refused_typing_stops_the_session_without_counting_the_text():
    _log, _backends, session = make(input_error=("type_unicode",))
    assert session.apply("hello") is False
    assert session.stop_reason == "typing refused"
    assert session.inserted == ""


def test_a_failing_foreground_check_stops_the_session():
    _log, _backends, session = make(window_error=("foreground_hwnd",))
    assert session.apply("hello") is False
    assert session.stop_reason == "foreground check failed"


def test_the_clipboard_is_never_touched():
    log, _backends, session = make()
    session.apply("hello there")
    session.apply("hello two")
    session.apply("")
    assert not [entry for entry in log if entry[0] in ("set_text", "snapshot", "restore")]


def test_typed_anything_and_active_report_the_session_state():
    _log, _backends, session = make()
    assert session.typed_anything is False
    assert session.active is True
    session.apply("hi")
    assert session.typed_anything is True
    session.stop("done")
    assert session.active is False
