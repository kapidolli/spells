"""spells.ui.pill against the approved mockup.

Geometry, colours and motion numbers come from the mockup's numbers block; the states come
from spec 14.2 and the pipeline's PillState and Notice. Placement is checked against a fake
work area; nothing here touches Win32.
"""

from __future__ import annotations

import pytest
from PySide6 import QtCore

from spells.compose import NOTHING_SELECTED_TEXT, REJECTION_TEXT, rejection_text
from spells.pipeline import (
    CAP_WARNING_TEXT,
    COPIED_NOTICE,
    MIC_BLOCKED_TEXT,
    MIC_PILL_TEXT,
    STARTING_ENGINES_TEXT,
    TRANSCRIPTION_FAILED,
)
from spells.ui.pill import (
    BAR_COUNT,
    COPIED_NOTICE_MS,
    ERROR_NOTICE_MS,
    HIDE_MS,
    PROCESSING_TEXT,
    SHOW_MS,
    STATIC_POSE,
    LevelSmoother,
    Pill,
    PillMetrics,
    content_for,
    notice_for,
    place_pill,
)

from .test_ui_support import Notice, PillState, TrayState, event, flush, qt_app

WORK_AREA = QtCore.QRect(100, 50, 1920, 1000)
MOCKUP_STRINGS = [
    "Processing",
    "Starting engines",
    "Mic is busy",
    "Transcription failed",
    "Copied. Press Ctrl+V to paste.",
    "One minute left",
]


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


# Metrics -------------------------------------------------------------------------------------


def test_metrics_at_100_percent_match_the_mockup():
    m = PillMetrics.for_unit(1.0)
    assert (m.height, m.radius, m.pad, m.gap) == (36, 18, 14, 8)
    assert (m.min_width, m.max_width) == (180, 320)
    assert (m.bar_width, m.bar_gap, m.bar_min, m.bar_max, m.meter_height) == (3, 3, 3, 20, 20)
    assert m.meter_width == 87
    assert (m.spinner, m.spinner_stroke, m.badge, m.badge_stroke) == (14, 1.5, 10, 1.25)
    assert (m.lock_width, m.lock_height, m.lock_stroke) == (11, 13, 1.5)
    assert (m.shadow_blur, m.shadow_offset, m.margin) == (24, 6, 24)
    assert m.taskbar_gap == 12
    assert (m.font_px, m.letter_spacing) == (12, 0.12)
    assert (m.show_rise, m.hide_fall) == (4, 2)
    assert BAR_COUNT == 15


def test_metrics_scale_from_one_unit():
    m = PillMetrics.for_unit(1.5)
    assert (m.height, m.radius, m.min_width, m.max_width) == (54, 27, 270, 480)
    assert (m.bar_width, m.bar_gap, m.font_px) == (4.5, 4.5, 18)
    assert m.meter_width == pytest.approx(130.5)
    assert m.margin == 36 and m.taskbar_gap == 18
    assert m.letter_spacing == pytest.approx(0.18)


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


def test_placement_at_150_percent_scales_the_gap_and_margin():
    m = PillMetrics.for_unit(1.5)
    x, y = place_pill((0, 0, 2880, 1560), 270, 54, m)
    assert x == (2880 - 270) / 2 - 36
    assert y == 1560 - 18 - 54 - 36


# Level smoothing ---------------------------------------------------------------------------


def test_envelope_uses_attack_and_decay_constants():
    smoother = LevelSmoother()
    smoother.step(1.0)
    assert smoother.envelope == pytest.approx(0.35)
    smoother.step(0.0)
    assert smoother.envelope == pytest.approx(0.35 - 0.35 * 0.10)


def test_bars_follow_the_envelope_with_per_bar_smoothing():
    smoother = LevelSmoother(noise=lambda index, frame: 1.0)
    bars = smoother.step(1.0)
    assert len(bars) == BAR_COUNT
    assert all(0.0 <= b <= 1.0 for b in bars)
    assert bars[7] == pytest.approx(0.35 * 0.45)
    assert bars[7] > bars[0]


def test_static_pose_has_fifteen_heights_in_range():
    assert len(STATIC_POSE) == BAR_COUNT
    assert all(3 <= h <= 20 for h in STATIC_POSE)


# Content per state ---------------------------------------------------------------------------


def test_idle_shows_nothing():
    assert content_for(event(PillState.IDLE)) is None


def test_recording_is_the_meter_alone():
    content = content_for(event(PillState.RECORDING, TrayState.RECORDING, level=0.4))
    assert content.meter and not content.lock and not content.badge and not content.spinner
    assert content.text == ""


def test_recording_with_a_dictation_processing_shows_the_badge():
    content = content_for(event(PillState.RECORDING, TrayState.RECORDING, busy=True))
    assert content.meter and content.badge and not content.spinner


def test_live_typing_shows_only_the_meter():
    content = content_for(event(PillState.RECORDING, TrayState.RECORDING, live_typing=True))
    assert content.meter and content.text == "" and not content.error


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


def test_the_pill_strings_are_the_six_the_mockup_approved():
    assert MOCKUP_STRINGS == [
        PROCESSING_TEXT,
        STARTING_ENGINES_TEXT,
        MIC_PILL_TEXT["busy"],
        TRANSCRIPTION_FAILED,
        COPIED_NOTICE,
        CAP_WARNING_TEXT,
    ]
    assert CAP_WARNING_TEXT == "One minute left"
    assert MIC_PILL_TEXT["busy"] == "Mic is busy"


def test_the_other_microphone_strings_share_the_mockup_short_form():
    assert MIC_PILL_TEXT == {
        "missing": "No mic found",
        "busy": "Mic is busy",
        "blocked": "Mic is blocked",
        "unknown": "Mic error",
    }
    assert MIC_BLOCKED_TEXT == MIC_PILL_TEXT["blocked"]
    for text in MIC_PILL_TEXT.values():
        assert "Microphone" not in text


def test_starting_engines_is_the_ring_plus_the_text():
    content = content_for(
        event(PillState.STARTING_ENGINES, TrayState.PROCESSING, text=STARTING_ENGINES_TEXT)
    )
    assert content.spinner and content.text == STARTING_ENGINES_TEXT


@pytest.mark.parametrize("text", [CAP_WARNING_TEXT, MIC_BLOCKED_TEXT])
def test_recording_warnings_sit_beside_the_meter_in_amber(text):
    content = content_for(event(PillState.RECORDING, TrayState.RECORDING, text=text))
    assert content.meter and content.text == text and content.error


def test_error_notice_is_amber_for_2200_ms():
    content, duration = notice_for(event(notice=Notice.ERROR, notice_text="Transcription failed"))
    assert content.text == "Transcription failed" and content.error
    assert not content.meter and not content.spinner
    assert duration == ERROR_NOTICE_MS == 2200


def test_copied_notice_is_the_full_sentence_for_2600_ms():
    content, duration = notice_for(event(notice=Notice.COPIED, notice_text="Copied"))
    assert content.text == COPIED_NOTICE and not content.error
    assert duration == COPIED_NOTICE_MS == 2600


def test_no_notice_gives_none():
    assert notice_for(event()) is None


# The widget ---------------------------------------------------------------------------------


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


def test_height_is_fixed_and_meter_width_is_the_minimum(app):
    pill, _resolved, _activated = make_pill()
    for pill_state in (PillState.RECORDING, PillState.LATCHED):
        pill.apply(event(pill_state, TrayState.RECORDING, busy=True))
        size = pill.pill_size()
        assert size.height() == 36
        assert size.width() == 180
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
    pill.apply(event(PillState.RECORDING, TrayState.RECORDING, target_hwnd=4242))
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
    assert pill.x() == pytest.approx(x_meter - (width - 180) / 2)
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


def test_spinner_advances_only_when_animated(app):
    animated, *_ = make_pill(reduced_motion=False)
    animated.apply(event(PillState.PROCESSING, TrayState.PROCESSING))
    angle = animated.spinner_angle()
    animated.tick(elapsed_ms=450)
    assert animated.spinner_angle() == pytest.approx((angle + 180) % 360)
    animated.close()
    frozen, *_ = make_pill(reduced_motion=True)
    frozen.apply(event(PillState.PROCESSING, TrayState.PROCESSING))
    angle = frozen.spinner_angle()
    frozen.tick(elapsed_ms=450)
    assert frozen.spinner_angle() == angle
    frozen.close()


def test_level_feeds_the_smoother(app):
    pill, *_ = make_pill(reduced_motion=False)
    pill.apply(event(PillState.RECORDING, TrayState.RECORDING, level=1.0))
    pill.tick()
    assert pill.smoother.envelope == pytest.approx(0.35)
    assert max(pill.bar_heights) > min(STATIC_POSE)
    pill.close()


def test_render_does_not_raise_in_any_state(app):
    pill, *_ = make_pill(reduced_motion=False)
    states = [
        event(PillState.RECORDING, TrayState.RECORDING, level=0.5),
        event(PillState.RECORDING, TrayState.RECORDING, busy=True),
        event(PillState.LATCHED, TrayState.RECORDING),
        event(PillState.PROCESSING, TrayState.PROCESSING),
        event(PillState.STARTING_ENGINES, TrayState.PROCESSING, text=STARTING_ENGINES_TEXT),
        event(PillState.RECORDING, TrayState.RECORDING, text=CAP_WARNING_TEXT),
        event(notice=Notice.ERROR, notice_text="Mic is busy"),
        event(notice=Notice.COPIED, notice_text="Copied"),
    ]
    for ev in states:
        pill.apply(ev)
        pill.tick()
        image = pill.grab().toImage()
        assert not image.isNull()
        assert image.width() == pill.width()
    pill.close()


def test_screen_for_default_uses_the_primary_screen(app):
    pill = Pill(reduced_motion=True, apply_no_activate=lambda hwnd: None)
    screen = pill.screen_for(None)
    assert screen is app.primaryScreen()
    pill.close()


def test_sample_reads_a_work_area_rect(app):
    pill, *_ = make_pill()
    assert pill.work_area(FakeScreen(QtCore.QRect(10, 20, 300, 400))) == (10, 20, 310, 420)
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


def test_default_screen_at_reaches_qt(app):
    pill = Pill(reduced_motion=True, apply_no_activate=lambda hwnd: None, monitor_rect=lambda hwnd: (0, 0, 10, 10))
    assert pill.screen_for(1) is app.primaryScreen()
    pill.close()


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


def test_meter_level_is_monotonic_and_bounded():
    from spells.ui.pill import meter_level

    values = [meter_level(rms / 100) for rms in range(101)]
    assert values == sorted(values)
    assert all(0.0 <= value <= 1.0 for value in values)


# --- the writing state (spec 14.2, 8.5) ---


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


def test_writing_is_a_state_of_its_own_and_not_processing():
    assert content_for(writing_event()).text != PROCESSING_TEXT
    assert content_for(event(PillState.PROCESSING, TrayState.PROCESSING)).writing is False


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


def test_the_seconds_count_even_with_motion_reduced(app):
    pill, _resolved, _activated = make_pill(reduced_motion=True)
    pill.apply(writing_event())
    pill.tick(elapsed_ms=3000)
    assert pill.displayed_text().startswith("Writing 3s:")
    pill.close()


def test_a_second_composition_starts_its_own_count(app):
    pill, _resolved, _activated = make_pill()
    pill.apply(writing_event())
    pill.tick(elapsed_ms=5000)
    assert pill.displayed_text().startswith("Writing 5s:")
    pill.apply(writing_event("fix this"))
    assert pill.displayed_text() == "Writing: fix this"
    pill.close()


def test_leaving_the_writing_state_stops_the_frame_timer(app):
    pill, _resolved, _activated = make_pill(reduced_motion=True)
    pill.apply(writing_event())
    assert pill._frame_timer.isActive() is True
    pill.apply(event())
    assert pill._frame_timer.isActive() is False
    pill.close()


def test_an_error_notice_covers_the_writing_state_while_it_runs(app):
    pill, _resolved, _activated = make_pill()
    pill.apply(writing_event())
    pill.apply(event(notice=Notice.ERROR, notice_text=rejection_text("refusal")))
    assert pill.content.error is True
    assert pill.displayed_text() == "The model refused"
    pill.close()


@pytest.mark.parametrize("reason", sorted(REJECTION_TEXT))
def test_every_writing_notice_fits_the_pill_whole(app, reason):
    pill, _resolved, _activated = make_pill()
    pill.apply(event(notice=Notice.ERROR, notice_text=rejection_text(reason)))
    assert pill.displayed_text() == rejection_text(reason)
    assert pill.pill_size().width() <= pill.metrics.max_width
    pill.close()


def test_the_nothing_selected_line_fits_the_pill_whole(app):
    pill, _resolved, _activated = make_pill()
    pill.apply(event(notice=Notice.ERROR, notice_text=NOTHING_SELECTED_TEXT))
    assert pill.displayed_text() == NOTHING_SELECTED_TEXT
    pill.close()
