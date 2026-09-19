"""The on-screen pill (spec 14.2), built to the approved pill mockup.

Geometry, colours, type and motion are the mockup's numbers block, all derived from one
unit. The widget draws in Qt's device-independent pixels, so the unit is one logical pixel
and Qt's per-screen scale factor turns the 180 x 36 drawing into 270 x 54 on a 150 percent
monitor; `PillMetrics.for_unit` exists so the numbers can be checked at any factor.

The window is a frameless, always-on-top tool window with Qt.WindowDoesNotAcceptFocus, and
WS_EX_NOACTIVATE is set on the native handle the first time it shows, so it never takes
focus from the window being dictated into. It sits at the bottom centre of the work area of
the monitor holding the target window (the QScreen at that monitor's top-left corner), 12
units above the taskbar, and falls back to the primary screen.
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from spells.compose import writing_text
from spells.pipeline import (
    COPIED_NOTICE,
    STARTING_ENGINES_TEXT,
    Notice,
    PillState,
    PipelineEvent,
)
from spells.ui import theme

log = logging.getLogger(__name__)

BAR_COUNT = 15
SHOW_MS = 120
HIDE_MS = 240
ERROR_NOTICE_MS = 2200
COPIED_NOTICE_MS = 2600
FRAME_MS = 16
SPIN_MS = 900
ATTACK = 0.35
DECAY = 0.10
BAR_SMOOTHING = 0.45
PROCESSING_TEXT = "Processing"
# The mockup's resting pose, used while motion is reduced.
STATIC_POSE = (4, 7, 11, 9, 14, 18, 13, 20, 15, 17, 10, 12, 6, 8, 4)


# Geometry -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PillMetrics:
    """Every number of the mockup, scaled from one unit (a logical pixel)."""

    unit: float
    height: float
    radius: float
    pad: float
    gap: float
    min_width: float
    max_width: float
    bar_width: float
    bar_gap: float
    bar_min: float
    bar_max: float
    meter_height: float
    spinner: float
    spinner_stroke: float
    badge: float
    badge_stroke: float
    lock_width: float
    lock_height: float
    lock_stroke: float
    shadow_blur: float
    shadow_offset: float
    margin: float
    taskbar_gap: float
    font_px: float
    letter_spacing: float
    show_rise: float
    hide_fall: float

    @classmethod
    def for_unit(cls, unit: float) -> PillMetrics:
        u = float(unit)
        return cls(
            unit=u,
            height=36 * u,
            radius=18 * u,
            pad=14 * u,
            gap=8 * u,
            min_width=180 * u,
            max_width=320 * u,
            bar_width=3 * u,
            bar_gap=3 * u,
            bar_min=3 * u,
            bar_max=20 * u,
            meter_height=20 * u,
            spinner=14 * u,
            spinner_stroke=1.5 * u,
            badge=10 * u,
            badge_stroke=1.25 * u,
            lock_width=11 * u,
            lock_height=13 * u,
            lock_stroke=1.5 * u,
            shadow_blur=24 * u,
            shadow_offset=6 * u,
            margin=24 * u,
            taskbar_gap=12 * u,
            font_px=12 * u,
            letter_spacing=0.12 * u,
            show_rise=4 * u,
            hide_fall=2 * u,
        )

    @property
    def bar_pitch(self) -> float:
        return self.bar_width + self.bar_gap

    @property
    def meter_width(self) -> float:
        return BAR_COUNT * self.bar_width + (BAR_COUNT - 1) * self.bar_gap

    def width_for(self, item_widths: Sequence[float]) -> float:
        """The width rule: minimum 180, grows to fit the items plus padding, capped at 320."""
        items = list(item_widths)
        total = 2 * self.pad + sum(items) + self.gap * max(0, len(items) - 1)
        return min(self.max_width, max(self.min_width, total))


def place_pill(
    work_area: tuple[float, float, float, float], pill_width: float, pill_height: float, m: PillMetrics
) -> tuple[float, float]:
    """Top-left of the widget (lozenge plus shadow margin) for the bottom centre placement."""
    left, _top, right, bottom = work_area
    x = left + (right - left - pill_width) / 2 - m.margin
    y = bottom - m.taskbar_gap - pill_height - m.margin
    return x, y


# Level smoothing ---------------------------------------------------------------------------------

METER_FLOOR_DB = -50.0
METER_CEILING_DB = -14.0


def meter_level(rms: float) -> float:
    """The meter's 0 to 1 reading for a raw RMS level (audio.rms_level).

    Speech sits around -30 to -15 dBFS, which is 3 to 18 percent of full scale, so feeding the
    raw value to the bars leaves them flat. The reading is the decibel position between a floor
    of room tone and a ceiling of loud speech.
    """
    value = float(rms)
    if value <= 0.0:
        return 0.0
    decibels = 20.0 * math.log10(min(1.0, value))
    if decibels <= METER_FLOOR_DB:
        return 0.0
    if decibels >= METER_CEILING_DB:
        return 1.0
    return (decibels - METER_FLOOR_DB) / (METER_CEILING_DB - METER_FLOOR_DB)


class LevelSmoother:
    """The mockup's meter motion: an envelope with attack and decay, then per-bar smoothing."""

    def __init__(
        self,
        count: int = BAR_COUNT,
        attack: float = ATTACK,
        decay: float = DECAY,
        bar_smoothing: float = BAR_SMOOTHING,
        noise: Callable[[int, int], float] | None = None,
    ) -> None:
        self.count = count
        self.attack = attack
        self.decay = decay
        self.bar_smoothing = bar_smoothing
        self.envelope = 0.0
        self.bars = [0.0] * count
        self.frame = 0
        self.shape = [0.42 + 0.58 * math.sin(math.pi * (i + 0.5) / count) for i in range(count)]
        rng = random.Random(7)
        self._seeds = [rng.random() * 7 for _ in range(count)]
        self._noise = noise or self._default_noise

    def _default_noise(self, index: int, frame: int) -> float:
        seed = self._seeds[index]
        ms = frame * FRAME_MS
        return 0.62 + 0.38 * math.sin(ms / (150 + seed * 40) + seed * 3.1)

    def step(self, target: float) -> list[float]:
        """Advance one frame toward target (0 to 1); returns each bar's value in 0 to 1."""
        target = min(1.0, max(0.0, float(target)))
        rate = self.attack if target > self.envelope else self.decay
        self.envelope += (target - self.envelope) * rate
        self.frame += 1
        for index in range(self.count):
            value = self.envelope * self.shape[index] * self._noise(index, self.frame)
            value = min(1.0, max(0.0, value))
            self.bars[index] += (value - self.bars[index]) * self.bar_smoothing
        return list(self.bars)


# Content -------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PillContent:
    """What the pill shows, left to right: lock, meter, badge; or spinner and text; or text."""

    meter: bool = False
    lock: bool = False
    badge: bool = False
    spinner: bool = False
    text: str = ""
    error: bool = False
    writing: bool = False
    instruction: str = ""


def content_for(event: PipelineEvent) -> PillContent | None:
    """The steady content for the event's pill state; None means hidden."""
    pill = event.pill
    if pill is PillState.IDLE:
        return None
    if pill in (PillState.RECORDING, PillState.LATCHED):
        text = event.text or ""
        error = bool(text)
        return PillContent(
            meter=True,
            lock=pill is PillState.LATCHED,
            badge=bool(event.busy),
            text=text,
            error=error,
        )
    if pill is PillState.PROCESSING:
        return PillContent(spinner=True, text=PROCESSING_TEXT)
    if pill is PillState.STARTING_ENGINES:
        return PillContent(spinner=True, text=event.text or STARTING_ENGINES_TEXT)
    if pill is PillState.WRITING:
        instruction = event.text or ""
        return PillContent(
            spinner=True,
            text=writing_text(instruction),
            writing=True,
            instruction=instruction,
        )
    return None


def notice_for(event: PipelineEvent) -> tuple[PillContent, int] | None:
    """The transient notice content and how long it stays, or None."""
    if event.notice is Notice.ERROR:
        return PillContent(text=event.notice_text or "Error", error=True), ERROR_NOTICE_MS
    if event.notice is Notice.COPIED:
        return PillContent(text=COPIED_NOTICE), COPIED_NOTICE_MS
    return None


# The widget -------------------------------------------------------------------------------------


def _default_monitor_rect(hwnd: int) -> tuple[int, int, int, int]:
    from spells.win32.window import monitor_rect_for_window

    return monitor_rect_for_window(hwnd)


def _default_screen_at(point: QtCore.QPoint) -> Any:
    return QtGui.QGuiApplication.screenAt(point)


def _default_no_activate(hwnd: int) -> None:
    try:
        from spells.win32.window import set_window_no_activate
    except Exception:
        log.debug("win32 window helpers unavailable", exc_info=True)
        return
    try:
        set_window_no_activate(hwnd)
    except Exception:
        log.debug("could not set WS_EX_NOACTIVATE on the pill", exc_info=True)


class Pill(QtWidgets.QWidget):
    def __init__(
        self,
        *,
        screen_for: Callable[[int | None], Any] | None = None,
        reduced_motion: bool | None = None,
        apply_no_activate: Callable[[int], Any] | None = None,
        monitor_rect: Callable[[int], tuple[int, int, int, int]] | None = None,
        screen_at: Callable[[QtCore.QPoint], Any] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        flags = (
            QtCore.Qt.WindowType.Tool
            | QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.WindowDoesNotAcceptFocus
        )
        super().__init__(parent, flags)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.setWindowTitle("Spells pill")

        self.metrics = PillMetrics.for_unit(1.0)
        self.reduced_motion = theme.reduced_motion() if reduced_motion is None else bool(reduced_motion)
        self._screen_for = screen_for
        self._monitor_rect = monitor_rect or _default_monitor_rect
        self._screen_at = screen_at or _default_screen_at
        self._apply_no_activate = apply_no_activate or _default_no_activate
        self._native_prepared = False

        self._steady: PillContent | None = None
        self._notice: PillContent | None = None
        self._content: PillContent | None = None
        self._displayed_text = ""
        self._pill_size = QtCore.QSizeF(self.metrics.min_width, self.metrics.height)
        self._target_hwnd: int | None = None
        self._level = 0.0
        self.smoother = LevelSmoother()
        self.bar_heights: list[float] = list(STATIC_POSE)
        self._spinner_angle = 0.0
        self._hiding = False
        self._writing_ms = 0.0
        self._writing_line = ""

        self._font = theme.pill_font(self.metrics.font_px, self.metrics.letter_spacing)
        self._fm = QtGui.QFontMetricsF(self._font)

        self.lozenge = _Lozenge(self)
        shadow = QtWidgets.QGraphicsDropShadowEffect(self.lozenge)
        shadow.setBlurRadius(self.metrics.shadow_blur)
        shadow.setOffset(0.0, self.metrics.shadow_offset)
        shadow.setColor(theme.PILL_SHADOW)
        self.lozenge.setGraphicsEffect(shadow)

        self.notice_timer = QtCore.QTimer(self)
        self.notice_timer.setSingleShot(True)
        self.notice_timer.timeout.connect(self._notice_expired)
        self._frame_timer = QtCore.QTimer(self)
        self._frame_timer.setInterval(FRAME_MS)
        self._frame_timer.timeout.connect(self.tick)
        self.animation = QtCore.QVariantAnimation(self)
        self.animation.valueChanged.connect(self._on_progress)
        self.animation.finished.connect(self._on_animation_finished)

        self._layout(None)

    # Public -----------------------------------------------------------------------------------

    @property
    def show_duration_ms(self) -> int:
        return 0 if self.reduced_motion else SHOW_MS

    @property
    def hide_duration_ms(self) -> int:
        return 0 if self.reduced_motion else HIDE_MS

    @property
    def content(self) -> PillContent | None:
        """What is on screen now (the notice when one runs, else the steady content)."""
        return self._content

    @property
    def steady(self) -> PillContent | None:
        return self._steady

    def displayed_text(self) -> str:
        return self._displayed_text

    def pill_size(self) -> QtCore.QSizeF:
        return QtCore.QSizeF(self._pill_size)

    def text_width(self, text: str) -> float:
        return float(self._fm.horizontalAdvance(text))

    def spinner_angle(self) -> float:
        return self._spinner_angle

    def apply(self, event: PipelineEvent) -> None:
        """Render the latest pipeline event (Qt thread)."""
        self._level = meter_level(event.level or 0.0)
        hwnd = getattr(event, "target_hwnd", None)
        if hwnd:
            self._target_hwnd = int(hwnd)
        steady = content_for(event)
        if steady is not None and steady.writing:
            previous = self._steady
            fresh = previous is None or not previous.writing
            if fresh or previous.instruction != steady.instruction:
                self._writing_ms = 0.0
            self._writing_line = steady.text
        self._steady = steady
        notice = notice_for(event)
        if notice is not None:
            self._notice, duration = notice
            self.notice_timer.start(duration)
        self._refresh()

    def tick(self, elapsed_ms: float | None = None) -> None:
        """One animation frame: advance the meter and the spinner, repaint."""
        content = self._content
        if content is None:
            return
        elapsed = FRAME_MS if elapsed_ms is None else float(elapsed_ms)
        if content.writing and self._tick_writing(elapsed):
            return
        if not self.reduced_motion:
            m = self.metrics
            if content.meter:
                values = self.smoother.step(self._level)
                self.bar_heights = [m.bar_min + (m.bar_max - m.bar_min) * v for v in values]
            if content.spinner or content.badge:
                self._spinner_angle = (self._spinner_angle + 360.0 * elapsed / SPIN_MS) % 360.0
        self.lozenge.update()

    def _tick_writing(self, elapsed: float) -> bool:
        """Count the seconds the writing model has been at it (spec 14.2, 8.5).

        A composition on a processor can take ten seconds, which looks frozen without a
        number moving. The count is driven by the animation frames rather than by a clock,
        so a test drives it exactly, and it runs under reduced motion too: the point of it
        is telling the user that something is still happening.
        """
        self._writing_ms += elapsed
        steady = self._steady
        if steady is None or not steady.writing:
            return False
        line = writing_text(steady.instruction, self._writing_ms / 1000.0)
        if line == self._writing_line:
            return False
        self._writing_line = line
        self._steady = replace(steady, text=line)
        self._refresh()
        return True

    def screen_for(self, hwnd: int | None) -> Any:
        """The QScreen at the top-left corner of the window's monitor, else the primary screen."""
        if self._screen_for is not None:
            return self._screen_for(hwnd)
        app = QtGui.QGuiApplication.instance()
        if hwnd:
            try:
                left, top, _right, _bottom = self._monitor_rect(hwnd)
                screen = self._screen_at(QtCore.QPoint(int(left), int(top)))
            except Exception:
                log.debug("could not resolve the target's monitor", exc_info=True)
                screen = None
            if screen is not None:
                return screen
        return app.primaryScreen()

    @staticmethod
    def work_area(screen: Any) -> tuple[int, int, int, int]:
        rect = screen.availableGeometry()
        return (
            rect.left(),
            rect.top(),
            rect.left() + rect.width(),
            rect.top() + rect.height(),
        )

    # Internals -----------------------------------------------------------------------------------

    def _notice_expired(self) -> None:
        self._notice = None
        self._refresh()

    def _refresh(self) -> None:
        content = self._notice or self._steady
        if content is None:
            self._content = None
            self._begin_hide()
            return
        self._layout(content)
        self._begin_show()
        self._update_frame_timer()
        self.lozenge.update()

    def _layout(self, content: PillContent | None) -> None:
        m = self.metrics
        items: list[float] = []
        text = ""
        overflow = False
        if content is not None:
            if content.lock:
                items.append(m.lock_width)
            if content.meter:
                items.append(m.meter_width)
            if content.badge:
                items.append(m.badge)
            if content.spinner:
                items.append(m.spinner)
            text = content.text
            if text:
                available = m.max_width - 2 * m.pad - sum(items) - m.gap * len(items)
                if self.text_width(text) > available:
                    text = self._fm.elidedText(text, QtCore.Qt.TextElideMode.ElideRight, available)
                    overflow = True
                items.append(self.text_width(text))
            self._content = replace(content, text=text)
        self._displayed_text = text
        width = m.max_width if overflow else m.width_for(items)
        self._pill_size = QtCore.QSizeF(width, m.height)
        self.setFixedSize(round(width + 2 * m.margin), round(m.height + 2 * m.margin))
        self.lozenge.setGeometry(
            round(m.margin), self.lozenge.y() if self._hiding else round(m.margin), round(width), round(m.height)
        )
        if content is None:
            return
        try:
            screen = self.screen_for(self._target_hwnd)
            x, y = place_pill(self.work_area(screen), width, m.height, m)
            self.move(round(x), round(y))
        except Exception:
            log.exception("could not place the pill")

    def _update_frame_timer(self) -> None:
        content = self._content
        animate = content is not None and (
            content.writing
            or (
                not self.reduced_motion
                and (content.meter or content.spinner or content.badge)
            )
        )
        if animate and not self._frame_timer.isActive():
            self._frame_timer.start()
        elif not animate:
            self._frame_timer.stop()

    def _prepare_native(self) -> None:
        if self._native_prepared:
            return
        self._native_prepared = True
        try:
            self._apply_no_activate(int(self.winId()))
        except Exception:
            log.debug("apply_no_activate failed", exc_info=True)

    def _begin_show(self) -> None:
        was_hiding = self._hiding
        if self.isVisible() and not was_hiding:
            return
        self._hiding = False
        if not self.isVisible():
            self.smoother = LevelSmoother()
            self.bar_heights = list(STATIC_POSE)
            self.show()
            self._prepare_native()
        if self.show_duration_ms == 0:
            self.animation.stop()
            self._set_progress(1.0)
            return
        start = float(self.windowOpacity()) if was_hiding else 0.0
        self.animation.stop()
        self.animation.setDuration(self.show_duration_ms)
        self.animation.setStartValue(start)
        self.animation.setEndValue(1.0)
        self.animation.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self.animation.start()

    def _begin_hide(self) -> None:
        if not self.isVisible() or self._hiding:
            return
        self._frame_timer.stop()
        if self.hide_duration_ms == 0:
            self.animation.stop()
            self.hide()
            self._set_progress(0.0)
            return
        self._hiding = True
        self.animation.stop()
        self.animation.setDuration(self.hide_duration_ms)
        self.animation.setStartValue(float(self.windowOpacity()))
        self.animation.setEndValue(0.0)
        self.animation.setEasingCurve(QtCore.QEasingCurve.Type.InCubic)
        self.animation.start()

    def _on_progress(self, value: Any) -> None:
        self._set_progress(float(value))

    def _set_progress(self, value: float) -> None:
        m = self.metrics
        self.setWindowOpacity(value)
        travel = m.hide_fall if self._hiding else m.show_rise
        offset = (1.0 - value) * travel
        self.lozenge.move(round(m.margin), round(m.margin + offset))

    def _on_animation_finished(self) -> None:
        if self._hiding:
            self._hiding = False
            self.hide()
            self._set_progress(0.0)

    def hideEvent(self, event: QtGui.QHideEvent) -> None:
        self._frame_timer.stop()
        super().hideEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._frame_timer.stop()
        self.notice_timer.stop()
        self.animation.stop()
        super().closeEvent(event)


class _Lozenge(QtWidgets.QWidget):
    """The black lozenge and its contents; the parent Pill holds the state it draws."""

    def __init__(self, pill: Pill) -> None:
        super().__init__(pill)
        self._pill = pill
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        pill = self._pill
        m = pill.metrics
        content = pill.content
        painter = QtGui.QPainter(self)
        try:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing)
            width = float(self.width())
            height = float(self.height())
            body = QtCore.QRectF(0.0, 0.0, width, height)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(theme.PILL_FILL)
            painter.drawRoundedRect(body, m.radius, m.radius)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.setPen(QtGui.QPen(theme.PILL_EDGE, 1.0))
            painter.drawRoundedRect(body.adjusted(0.5, 0.5, -0.5, -0.5), m.radius - 0.5, m.radius - 0.5)
            if content is None:
                return
            widths: list[float] = []
            if content.lock:
                widths.append(m.lock_width)
            if content.meter:
                widths.append(m.meter_width)
            if content.badge:
                widths.append(m.badge)
            if content.spinner:
                widths.append(m.spinner)
            text = content.text
            text_width = pill.text_width(text) if text else 0.0
            if text:
                widths.append(text_width)
            total = sum(widths) + m.gap * max(0, len(widths) - 1)
            ratio = max(1.0, float(self.devicePixelRatioF()))
            x = round((width - total) / 2.0 * ratio) / ratio
            cy = height / 2.0
            if content.lock:
                self._draw_lock(painter, x, cy, m)
                x += m.lock_width + m.gap
            if content.meter:
                self._draw_meter(painter, x, cy, m, pill.bar_heights)
                x += m.meter_width + m.gap
            if content.badge:
                self._draw_ring(painter, x, cy, m.badge, m.badge_stroke, pill.spinner_angle())
                x += m.badge + m.gap
            if content.spinner:
                self._draw_ring(painter, x, cy, m.spinner, m.spinner_stroke, pill.spinner_angle())
                x += m.spinner + m.gap
            if text:
                painter.setFont(pill._font)
                painter.setPen(theme.PILL_ERROR if content.error else theme.PILL_TEXT)
                rect = QtCore.QRectF(x, 0.0, text_width + 2.0, height)
                painter.drawText(
                    rect,
                    int(QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft),
                    text,
                )
        finally:
            painter.end()

    @staticmethod
    def _draw_meter(painter: QtGui.QPainter, x: float, cy: float, m: PillMetrics, heights: Sequence[float]) -> None:
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(theme.PILL_BAR)
        radius = m.bar_width / 2.0
        for index in range(BAR_COUNT):
            h = float(heights[index]) if index < len(heights) else m.bar_min
            h = min(m.bar_max, max(m.bar_min, h))
            bar = QtCore.QRectF(x + index * m.bar_pitch, cy - h / 2.0, m.bar_width, h)
            painter.drawRoundedRect(bar, radius, radius)

    @staticmethod
    def _draw_ring(
        painter: QtGui.QPainter, x: float, cy: float, size: float, stroke: float, angle: float
    ) -> None:
        radius = (size - stroke) / 2.0
        box = QtCore.QRectF(x + stroke / 2.0, cy - radius, 2 * radius, 2 * radius)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.setPen(QtGui.QPen(theme.PILL_SPINNER_TRACK, stroke, QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.RoundCap))
        painter.drawEllipse(box)
        painter.setPen(QtGui.QPen(theme.PILL_SPINNER, stroke, QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.RoundCap))
        start = int((90.0 - angle) * 16)
        painter.drawArc(box, start, -270 * 16)

    @staticmethod
    def _draw_lock(painter: QtGui.QPainter, x: float, cy: float, m: PillMetrics) -> None:
        u = m.lock_width / 11.0
        top = cy - m.lock_height / 2.0
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.setPen(QtGui.QPen(theme.PILL_LOCK, m.lock_stroke, QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.RoundCap, QtCore.Qt.PenJoinStyle.RoundJoin))
        body = QtCore.QRectF(x + 0.75 * u, top + 5.25 * u, 9.5 * u, 7.0 * u)
        painter.drawRoundedRect(body, 1.8 * u, 1.8 * u)
        shackle = QtGui.QPainterPath()
        shackle.moveTo(x + 3.0 * u, top + 5.25 * u)
        shackle.lineTo(x + 3.0 * u, top + 3.4 * u)
        shackle.arcTo(QtCore.QRectF(x + 3.0 * u, top + 0.9 * u, 5.0 * u, 5.0 * u), 180.0, -180.0)
        shackle.lineTo(x + 8.0 * u, top + 5.25 * u)
        painter.drawPath(shackle)


__all__ = [
    "BAR_COUNT",
    "COPIED_NOTICE_MS",
    "ERROR_NOTICE_MS",
    "HIDE_MS",
    "SHOW_MS",
    "STATIC_POSE",
    "LevelSmoother",
    "Pill",
    "PillContent",
    "PillMetrics",
    "content_for",
    "notice_for",
    "place_pill",
]
