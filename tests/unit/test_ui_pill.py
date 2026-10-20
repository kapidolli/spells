"""spells.ui.pill against the approved mockup.

Geometry, colours and motion numbers come from the mockup's numbers block; the states come
from spec 14.2 and the pipeline's PillState and Notice. Placement is checked against a fake
work area; nothing here touches Win32.
"""

from __future__ import annotations

import pytest
from PySide6 import QtCore

from spells.compose import rejection_text
from spells.pipeline import (
    COPIED_NOTICE,
    STARTING_ENGINES_TEXT,
)
from spells.ui.pill import (
    COPIED_NOTICE_MS,
    ERROR_NOTICE_MS,
    HIDE_MS,
    SHOW_MS,
    STATIC_POSE,
    Pill,
    PillMetrics,
    content_for,
    notice_for,
    place_pill,
)

from .test_ui_support import Notice, PillState, TrayState, event, flush, qt_app

WORK_AREA = QtCore.QRect(100, 50, 1920, 1000)


@pytest.fixture(scope="module")
def app():
    return qt_app()


class FakeScreen:
    def __init__(self, rect: QtCore.QRect = WORK_AREA, name: str = "fake") -> None:
        self._rect = rect
        self._name = name

    def availableGeometry(self) -> QtCore.QRect:
        return QtCore.QRect(self._rect)

    def name(self) -> str:
        return self._name


def make_pill(*, reduced_motion=True, rect=WORK_AREA):
    resolved: list[int | None] = []
    activated: list[int] = []

    def screen_for(hwnd):
        resolved.append(hwnd)
        return FakeScreen(rect)

    pill = Pill(screen_for=screen_for, reduced_motion=reduced_motion, apply_no_activate=activated.append)
    return pill, resolved, activated



def test_width_rule_minimum_growth_and_cap():
    m = PillMetrics.for_unit(1.0)
    assert m.width_for([87]) == 180
    assert m.width_for([14, 205]) == 14 + 205 + 8 + 28
    assert m.width_for([14, 600]) == 320
    assert m.width_for([]) == 180


def test_placement_is_bottom_centre_of_the_work_area_above_the_taskbar():
    m = PillMetrics.for_unit(1.0)
    x, y = place_pill((100, 50, 2020, 1050), 180, 36, m)
    assert x == 100 + (1920 - 180) / 2 - 24
    assert y == 1050 - 12 - 36 - 24


def test_idle_shows_nothing():
    assert content_for(event(PillState.IDLE)) is None


def test_recording_is_the_meter_alone():
    content = content_for(event(PillState.RECORDING, TrayState.RECORDING, level=0.4))
    assert content.meter and not content.lock and not content.badge and not content.spinner
    assert content.text == ""


def test_an_error_text_still_shows_while_typing_live():
    content = content_for(
        event(PillState.RECORDING, TrayState.RECORDING, live_typing=True, text="Mic is blocked")
    )
    assert content.text == "Mic is blocked" and content.error


def test_latched_shows_the_lock_and_the_meter():
    content = content_for(event(PillState.LATCHED, TrayState.RECORDING))
    assert content.lock and content.meter


def test_processing_is_the_ring_plus_the_word():
    content = content_for(event(PillState.PROCESSING, TrayState.PROCESSING))
    assert content.spinner and content.text == "Processing" and not content.meter


def test_starting_engines_is_the_ring_plus_the_text():
    content = content_for(
        event(PillState.STARTING_ENGINES, TrayState.PROCESSING, text=STARTING_ENGINES_TEXT)
    )
    assert content.spinner and content.text == STARTING_ENGINES_TEXT


def test_copied_notice_is_the_full_sentence_for_2600_ms():
    content, duration = notice_for(event(notice=Notice.COPIED, notice_text="Copied"))
    assert content.text == COPIED_NOTICE and not content.error
    assert duration == COPIED_NOTICE_MS == 2600


def test_window_flags_never_take_focus(app):
    pill, _resolved, activated = make_pill()
    flags = pill.windowFlags()
    for flag in (
        QtCore.Qt.WindowType.Tool,
        QtCore.Qt.WindowType.FramelessWindowHint,
        QtCore.Qt.WindowType.WindowStaysOnTopHint,
        QtCore.Qt.WindowType.WindowDoesNotAcceptFocus,
    ):
        assert flags & flag
    assert pill.testAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating)
    assert pill.testAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
    assert pill.focusPolicy() == QtCore.Qt.FocusPolicy.NoFocus
    pill.apply(event(PillState.RECORDING, TrayState.RECORDING))
    flush(app)
    assert activated == [int(pill.winId())]
    pill.close()


def test_text_states_grow_to_fit_and_cap_at_320(app):
    pill, _resolved, _activated = make_pill()
    m = pill.metrics
    pill.apply(event(PillState.PROCESSING, TrayState.PROCESSING))
    expected = m.width_for([m.spinner, pill.text_width("Processing")])
    assert pill.pill_size().width() == pytest.approx(expected)
    assert pill.pill_size().height() == 36

    pill.apply(event(notice=Notice.COPIED, notice_text="Copied"))
    expected = m.width_for([pill.text_width(COPIED_NOTICE)])
    assert pill.pill_size().width() == pytest.approx(expected)

    long_text = "This sentence is far too long to fit inside the widest pill the rule allows"
    pill.apply(event(notice=Notice.ERROR, notice_text=long_text))
    assert pill.pill_size().width() == 320
    assert pill.content.text != long_text
    assert pill.displayed_text().endswith("…")
    pill.close()


def test_pill_is_placed_on_the_target_monitor(app):
    pill, resolved, _activated = make_pill()
    pill.apply(event(PillState.RECORDING, TrayState.RECORDING, target_window=4242))
    flush(app)
    assert resolved == [4242]
    m = pill.metrics
    x, y = place_pill((100, 50, 2020, 1050), 180, 36, m)
    assert (pill.x(), pill.y()) == (x, y)
    assert pill.width() == 180 + 2 * m.margin
    assert pill.height() == 36 + 2 * m.margin
    pill.close()


def test_no_target_falls_back_to_the_primary_screen(app):
    pill, resolved, _activated = make_pill()
    pill.apply(event(PillState.PROCESSING, TrayState.PROCESSING))
    assert resolved == [None]
    pill.close()


def test_placement_is_recomputed_when_the_width_changes(app):
    pill, _resolved, _activated = make_pill()
    pill.apply(event(PillState.RECORDING, TrayState.RECORDING))
    x_meter = pill.x()
    pill.apply(event(notice=Notice.COPIED, notice_text="Copied"))
    width = pill.pill_size().width()
    assert width > 180
    assert pill.x() == pytest.approx(x_meter - (width - 180) / 2, abs=0.5)
    pill.close()


def test_notice_timer_returns_to_the_steady_state(app):
    pill, _resolved, _activated = make_pill()
    pill.apply(event(PillState.PROCESSING, TrayState.PROCESSING))
    pill.apply(
        event(PillState.PROCESSING, TrayState.PROCESSING, notice=Notice.ERROR, notice_text="Transcription failed")
    )
    assert pill.content.text == "Transcription failed" and pill.content.error
    assert pill.notice_timer.isActive()
    assert pill.notice_timer.interval() == ERROR_NOTICE_MS
    pill.notice_timer.stop()
    pill.notice_timer.timeout.emit()
    assert pill.content.text == "Processing" and pill.content.spinner
    pill.close()


def test_notice_over_idle_hides_after_the_timer(app):
    pill, _resolved, _activated = make_pill()
    pill.apply(event(notice=Notice.COPIED, notice_text="Copied"))
    flush(app)
    assert pill.isVisible()
    assert pill.notice_timer.interval() == COPIED_NOTICE_MS
    pill.notice_timer.stop()
    pill.notice_timer.timeout.emit()
    flush(app)
    assert pill.content is None
    assert not pill.isVisible()
    pill.close()


def test_a_new_steady_event_during_a_notice_keeps_the_notice(app):
    pill, _resolved, _activated = make_pill()
    pill.apply(event(notice=Notice.ERROR, notice_text="Mic is busy"))
    pill.apply(event(PillState.RECORDING, TrayState.RECORDING))
    assert pill.content.text == "Mic is busy"
    assert pill.steady is not None and pill.steady.meter
    pill.notice_timer.stop()
    pill.notice_timer.timeout.emit()
    assert pill.content.meter
    pill.close()


def test_reduced_motion_drops_the_animations_and_freezes_the_bars(app):
    pill, _resolved, _activated = make_pill(reduced_motion=True)
    assert pill.reduced_motion
    assert pill.show_duration_ms == 0 and pill.hide_duration_ms == 0
    pill.apply(event(PillState.RECORDING, TrayState.RECORDING, level=0.9))
    flush(app)
    assert pill.isVisible()
    first = list(pill.bar_heights)
    pill.tick()
    pill.tick()
    assert list(pill.bar_heights) == first == list(STATIC_POSE)
    pill.apply(event())
    flush(app)
    assert not pill.isVisible()
    pill.close()


def test_animated_show_and_hide_use_the_mockup_durations(app):
    pill, _resolved, _activated = make_pill(reduced_motion=False)
    assert (pill.show_duration_ms, pill.hide_duration_ms) == (SHOW_MS, HIDE_MS) == (120, 240)
    pill.apply(event(PillState.RECORDING, TrayState.RECORDING, level=0.9))
    flush(app)
    assert pill.isVisible()
    assert pill.animation.state() == QtCore.QAbstractAnimation.State.Running
    assert pill.animation.duration() == SHOW_MS
    pill.animation.setCurrentTime(SHOW_MS)
    flush(app)
    assert pill.windowOpacity() == pytest.approx(1.0)
    before = list(pill.bar_heights)
    pill.tick()
    assert list(pill.bar_heights) != before
    pill.apply(event())
    assert pill.isVisible()
    assert pill.animation.duration() == HIDE_MS
    pill.animation.setCurrentTime(HIDE_MS)
    flush(app)
    assert not pill.isVisible()
    pill.close()


def test_a_new_event_during_the_fade_out_brings_the_pill_back(app):
    pill, _resolved, _activated = make_pill(reduced_motion=False)
    pill.apply(event(PillState.RECORDING, TrayState.RECORDING))
    pill.animation.setCurrentTime(SHOW_MS)
    pill.apply(event())
    assert pill.animation.duration() == HIDE_MS
    pill.apply(event(PillState.PROCESSING, TrayState.PROCESSING))
    flush(app)
    assert pill.isVisible()
    assert pill.animation.duration() == SHOW_MS
    assert pill.content.spinner
    pill.close()


def test_default_screen_resolution_uses_the_monitor_corner(app):
    asked: list[QtCore.QPoint] = []
    target = object()

    def screen_at(point):
        asked.append(QtCore.QPoint(point))
        return target

    pill = Pill(
        reduced_motion=True,
        apply_no_activate=lambda hwnd: None,
        monitor_rect=lambda hwnd: (2560, -120, 5120, 1320),
        screen_at=screen_at,
    )
    assert pill.screen_for(4242) is target
    assert asked == [QtCore.QPoint(2560, -120)]
    assert pill.screen_for(None) is app.primaryScreen()
    assert asked == [QtCore.QPoint(2560, -120)]
    pill.close()


def test_default_screen_resolution_falls_back_to_the_primary_screen(app):
    no_screen = Pill(
        reduced_motion=True,
        apply_no_activate=lambda hwnd: None,
        monitor_rect=lambda hwnd: (9000, 9000, 9100, 9100),
        screen_at=lambda point: None,
    )
    assert no_screen.screen_for(1) is app.primaryScreen()
    no_screen.close()

    def broken(hwnd):
        raise OSError("no monitor")

    failing = Pill(reduced_motion=True, apply_no_activate=lambda hwnd: None, monitor_rect=broken)
    assert failing.screen_for(1) is app.primaryScreen()
    failing.close()


def test_meter_level_lifts_speech_off_the_floor():
    from spells.ui.pill import meter_level

    assert meter_level(0.0) == 0.0
    assert meter_level(0.001) == 0.0
    quiet = meter_level(0.01)
    speech = meter_level(0.07)
    loud = meter_level(0.2)
    assert 0.0 < quiet < 0.35
    assert 0.55 < speech < 0.95
    assert loud == pytest.approx(1.0)
    assert quiet < speech < loud


INSTRUCTION = "write an email to Marta asking for the September invoice"


def writing_event(instruction: str = INSTRUCTION):
    return event(PillState.WRITING, TrayState.PROCESSING, text=instruction)


def test_writing_is_the_ring_plus_the_state_and_the_instruction():
    content = content_for(writing_event())
    assert content.spinner is True
    assert content.writing is True
    assert content.meter is False
    assert content.text == f"Writing: {INSTRUCTION}"
    assert content.instruction == INSTRUCTION


def test_a_long_instruction_is_elided_inside_the_320_cap(app):
    pill, _resolved, _activated = make_pill()
    pill.apply(writing_event("write a very long letter " * 12))
    assert pill.pill_size().width() <= pill.metrics.max_width
    assert pill.displayed_text().startswith("Writing")
    assert pill.displayed_text() != pill.steady.text
    pill.close()


def test_the_seconds_appear_once_the_wait_passes_two_of_them(app):
    pill, _resolved, _activated = make_pill()
    pill.apply(writing_event())
    assert pill.displayed_text().startswith("Writing:")
    pill.tick(elapsed_ms=1000)
    assert pill.displayed_text().startswith("Writing:")
    pill.tick(elapsed_ms=1000)
    assert pill.displayed_text().startswith("Writing 2s:")
    pill.tick(elapsed_ms=8000)
    assert pill.displayed_text().startswith("Writing 10s:")
    pill.close()


def test_a_second_composition_starts_its_own_count(app):
    pill, _resolved, _activated = make_pill()
    pill.apply(writing_event())
    pill.tick(elapsed_ms=5000)
    assert pill.displayed_text().startswith("Writing 5s:")
    pill.apply(writing_event("fix this"))
    assert pill.displayed_text() == "Writing: fix this"
    pill.close()


def test_an_error_notice_covers_the_writing_state_while_it_runs(app):
    pill, _resolved, _activated = make_pill()
    pill.apply(writing_event())
    pill.apply(event(notice=Notice.ERROR, notice_text=rejection_text("refusal")))
    assert pill.content.error is True
    assert pill.displayed_text() == "The model refused"
    pill.close()

