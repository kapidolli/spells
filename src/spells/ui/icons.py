"""Tray icons drawn with QPainter (spec 14.1): one glyph per state, no binary assets.

The glyph is the mono Spells mark from data/brand (a drawn microphone while that file is
missing); the state is a badge in the bottom right whose shape differs per state, so the
icon reads on a monochrome taskbar and colour only adds to it. The foreground follows the
taskbar theme (white on dark, near-black on light).
"""

from __future__ import annotations

from enum import Enum

from PySide6 import QtCore, QtGui

from spells.ui import brand, theme

ICON_SIZES = (16, 20, 24, 32, 48)
_BOX = 32.0


class TrayIconState(str, Enum):
    """The seven tray states of spec 14.1."""

    STARTING = "starting"
    READY = "ready"
    RECORDING = "recording"
    PROCESSING = "processing"
    WARNING = "warning"
    ERROR = "error"
    IDLE = "idle"


def _draw_microphone(painter: QtGui.QPainter, colour: QtGui.QColor, *, filled: bool) -> None:
    pen = QtGui.QPen(colour, 2.4, QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    painter.setBrush(colour if filled else QtCore.Qt.BrushStyle.NoBrush)
    capsule = QtCore.QRectF(11.5, 3.0, 9.0, 15.0)
    painter.drawRoundedRect(capsule, 4.5, 4.5)
    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    arc = QtGui.QPainterPath()
    bowl = QtCore.QRectF(8.0, 6.0, 16.0, 16.0)
    arc.arcMoveTo(bowl, 180.0)
    arc.arcTo(bowl, 180.0, 180.0)
    painter.drawPath(arc)
    painter.drawLine(QtCore.QPointF(16.0, 22.0), QtCore.QPointF(16.0, 26.5))
    painter.drawLine(QtCore.QPointF(11.0, 27.5), QtCore.QPointF(21.0, 27.5))


def _draw_mark(painter: QtGui.QPainter, colour: QtGui.QColor, size: int, *, badged: bool) -> bool:
    """The mono brand mark from data/brand in the glyph colour; False when the file is missing."""
    pixels = max(8, size * 2)
    image = brand.tinted_mark(pixels, colour)
    if image is None:
        return False
    box = QtCore.QRectF(1.0, 1.0, 30.0, 30.0) if not badged else QtCore.QRectF(0.0, 0.0, 23.0, 23.0)
    painter.save()
    try:
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(box, image)
    finally:
        painter.restore()
    return True


def _clear_badge_area(painter: QtGui.QPainter, centre: QtCore.QPointF, radius: float) -> None:
    painter.save()
    painter.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_Clear)
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.setBrush(QtGui.QColor(0, 0, 0, 255))
    painter.drawEllipse(centre, radius + 2.0, radius + 2.0)
    painter.restore()


def _draw_badge(
    painter: QtGui.QPainter, state: TrayIconState, foreground: QtGui.QColor, mark: QtGui.QColor
) -> None:
    centre = QtCore.QPointF(24.0, 24.0)
    radius = 6.5
    _clear_badge_area(painter, centre, radius)
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    if state is TrayIconState.RECORDING:
        painter.setBrush(theme.TRAY_RED)
        painter.drawEllipse(centre, radius, radius)
        return
    if state is TrayIconState.WARNING:
        painter.setBrush(theme.TRAY_AMBER)
        painter.drawEllipse(centre, radius, radius)
        painter.setPen(QtGui.QPen(mark, 2.2, QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.RoundCap))
        painter.drawLine(QtCore.QPointF(24.0, 20.2), QtCore.QPointF(24.0, 25.0))
        painter.drawPoint(QtCore.QPointF(24.0, 28.0))
        return
    if state is TrayIconState.ERROR:
        painter.setBrush(theme.TRAY_RED)
        painter.drawEllipse(centre, radius, radius)
        painter.setPen(QtGui.QPen(mark, 2.2, QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.RoundCap))
        painter.drawLine(QtCore.QPointF(21.2, 21.2), QtCore.QPointF(26.8, 26.8))
        painter.drawLine(QtCore.QPointF(26.8, 21.2), QtCore.QPointF(21.2, 26.8))
        return
    ring = QtGui.QPen(foreground, 2.0, QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.RoundCap)
    painter.setPen(ring)
    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    box = QtCore.QRectF(centre.x() - radius + 1.0, centre.y() - radius + 1.0, 2 * radius - 2.0, 2 * radius - 2.0)
    if state is TrayIconState.PROCESSING:
        painter.drawArc(box, 90 * 16, -270 * 16)
        return
    if state is TrayIconState.STARTING:
        painter.drawEllipse(box)


def render_tray_pixmap(state: TrayIconState, size: int, *, light_taskbar: bool = False) -> QtGui.QPixmap:
    """One square pixmap of the state at the given pixel size."""
    foreground = theme.TRAY_DARK if light_taskbar else theme.TRAY_LIGHT
    mark = theme.TRAY_DARK
    pixmap = QtGui.QPixmap(size, size)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    try:
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.scale(size / _BOX, size / _BOX)
        glyph = QtGui.QColor(foreground)
        if state is TrayIconState.IDLE:
            glyph.setAlphaF(0.6)
        elif state is TrayIconState.STARTING:
            glyph.setAlphaF(0.55)
        if not _draw_mark(painter, glyph, size, badged=state not in (TrayIconState.READY, TrayIconState.IDLE)):
            _draw_microphone(painter, glyph, filled=state is not TrayIconState.IDLE)
        if state not in (TrayIconState.READY, TrayIconState.IDLE):
            _draw_badge(painter, state, foreground, mark)
    finally:
        painter.end()
    return pixmap


def tray_icon(state: TrayIconState, *, light_taskbar: bool = False) -> QtGui.QIcon:
    """The icon for a state at every tray size Windows may ask for."""
    icon = QtGui.QIcon()
    for size in ICON_SIZES:
        icon.addPixmap(render_tray_pixmap(state, size, light_taskbar=light_taskbar))
    return icon


__all__ = ["ICON_SIZES", "TrayIconState", "render_tray_pixmap", "tray_icon"]
