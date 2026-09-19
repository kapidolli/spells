"""The Spells marks from data/brand, drawn as placeholders while the files are missing.

spells-logo.svg is the full colour mark, spells-mark-mono.svg the one-colour glyph the tray
tints, and spells-wordmark.svg the name. A missing or unreadable file falls back to the
drawn mark: a rounded indigo to blue square holding three voice bars and a spark.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from PySide6 import QtCore, QtGui, QtSvg

from spells.datafiles import data_path

log = logging.getLogger(__name__)

LOGO = "spells-logo.svg"
MONO_MARK = "spells-mark-mono.svg"
WORDMARK = "spells-wordmark.svg"

_renderers: dict[str, tuple[float, QtSvg.QSvgRenderer | None]] = {}
_wordmarks: dict[str, tuple[float, QtSvg.QSvgRenderer | None]] = {}
_ROOT_COLOUR = re.compile(rb'(<svg\b[^>]*?\scolor=")[^"]*(")')


def brand_path(name: str) -> Path:
    return data_path("brand", name)


def renderer(name: str) -> QtSvg.QSvgRenderer | None:
    """A valid renderer for data/brand/<name>, reloaded when the file changes; None when absent."""
    path = brand_path(name)
    try:
        stamp = path.stat().st_mtime
    except OSError:
        _renderers.pop(name, None)
        return None
    cached = _renderers.get(name)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    loaded = QtSvg.QSvgRenderer(str(path))
    result = loaded if loaded.isValid() else None
    if result is None:
        log.warning("brand file %s is not a valid SVG", path)
    _renderers[name] = (stamp, result)
    return result


def fitted(bounds: QtCore.QRectF, size: QtCore.QSizeF) -> QtCore.QRectF:
    """The largest rectangle of the aspect of `size` centred in `bounds`."""
    if size.width() <= 0 or size.height() <= 0:
        return QtCore.QRectF(bounds)
    scale = min(bounds.width() / size.width(), bounds.height() / size.height())
    width = size.width() * scale
    height = size.height() * scale
    return QtCore.QRectF(
        bounds.x() + (bounds.width() - width) / 2, bounds.y() + (bounds.height() - height) / 2, width, height
    )


def _view_size(svg: QtSvg.QSvgRenderer) -> QtCore.QSizeF:
    box = svg.viewBoxF()
    if box.width() > 0 and box.height() > 0:
        return box.size()
    return QtCore.QSizeF(svg.defaultSize())


def paint_placeholder_mark(
    painter: QtGui.QPainter, rect: QtCore.QRectF, start: QtGui.QColor, end: QtGui.QColor, glyph: QtGui.QColor | None = None
) -> None:
    side = min(rect.width(), rect.height())
    box = QtCore.QRectF(rect.center().x() - side / 2, rect.center().y() - side / 2, side, side)
    unit = side / 28.0
    painter.save()
    try:
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        gradient = QtGui.QLinearGradient(box.topLeft(), box.bottomRight())
        gradient.setColorAt(0.0, start)
        gradient.setColorAt(1.0, end)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QBrush(gradient))
        painter.drawRoundedRect(box, 8 * unit, 8 * unit)
        ink = glyph if glyph is not None else QtGui.QColor(255, 255, 255)
        paint_voice_glyph(painter, box, ink)
    finally:
        painter.restore()


def paint_voice_glyph(painter: QtGui.QPainter, box: QtCore.QRectF, ink: QtGui.QColor) -> None:
    unit = min(box.width(), box.height()) / 28.0
    ox = box.x()
    oy = box.y()
    painter.save()
    try:
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(ink)
        for x, top, height in ((8.0, 11.0, 7.0), (12.7, 7.0, 15.0), (17.4, 10.0, 9.0)):
            painter.drawRoundedRect(QtCore.QRectF(ox + x * unit, oy + top * unit, 2.6 * unit, height * unit), 1.3 * unit, 1.3 * unit)
        cx = ox + 21.5 * unit
        cy = oy + 6.3 * unit
        outer = 2.4 * unit
        inner = 0.7 * unit
        star = QtGui.QPainterPath()
        star.moveTo(cx, cy - outer)
        star.lineTo(cx + inner, cy - inner)
        star.lineTo(cx + outer, cy)
        star.lineTo(cx + inner, cy + inner)
        star.lineTo(cx, cy + outer)
        star.lineTo(cx - inner, cy + inner)
        star.lineTo(cx - outer, cy)
        star.lineTo(cx - inner, cy - inner)
        star.closeSubpath()
        painter.drawPath(star)
    finally:
        painter.restore()


def wordmark_renderer(colour: QtGui.QColor) -> QtSvg.QSvgRenderer | None:
    """The wordmark with its letters (currentColor) in the given colour; None when absent."""
    path = brand_path(WORDMARK)
    try:
        stamp = path.stat().st_mtime
        raw = path.read_bytes()
    except OSError:
        return None
    key = colour.name(QtGui.QColor.NameFormat.HexRgb)
    cached = _wordmarks.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    tinted = _ROOT_COLOUR.sub(lambda m: m.group(1) + key.encode("ascii") + m.group(2), raw, count=1)
    loaded = QtSvg.QSvgRenderer(QtCore.QByteArray(tinted))
    result = loaded if loaded.isValid() else None
    _wordmarks[key] = (stamp, result)
    return result


def wordmark_aspect(colour: QtGui.QColor) -> float | None:
    svg = wordmark_renderer(colour)
    if svg is None:
        return None
    size = _view_size(svg)
    return size.width() / size.height() if size.height() > 0 else None


def paint_wordmark(painter: QtGui.QPainter, rect: QtCore.QRectF, colour: QtGui.QColor) -> bool:
    svg = wordmark_renderer(colour)
    if svg is None:
        return False
    svg.render(painter, fitted(rect, _view_size(svg)))
    return True


def paint_logo(painter: QtGui.QPainter, rect: QtCore.QRectF, start: QtGui.QColor, end: QtGui.QColor) -> bool:
    """Draw the logo into rect; True when the SVG was used, False for the placeholder."""
    svg = renderer(LOGO)
    if svg is None:
        paint_placeholder_mark(painter, rect, start, end)
        return False
    svg.render(painter, fitted(rect, _view_size(svg)))
    return True


def tinted_mark(size: int, colour: QtGui.QColor) -> QtGui.QImage | None:
    """The mono mark rendered in one colour at size by size pixels; None when the file is missing."""
    svg = renderer(MONO_MARK)
    if svg is None:
        return None
    image = QtGui.QImage(size, size, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(image)
    try:
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        svg.render(painter, fitted(QtCore.QRectF(0, 0, size, size), _view_size(svg)))
        painter.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_SourceIn)
        painter.fillRect(QtCore.QRect(0, 0, size, size), colour)
    finally:
        painter.end()
    return image


def logo_pixmap(size: int, start: QtGui.QColor, end: QtGui.QColor, device_ratio: float = 1.0) -> QtGui.QPixmap:
    pixels = max(1, round(size * device_ratio))
    pixmap = QtGui.QPixmap(pixels, pixels)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    try:
        paint_logo(painter, QtCore.QRectF(0, 0, pixels, pixels), start, end)
    finally:
        painter.end()
    pixmap.setDevicePixelRatio(device_ratio)
    return pixmap


def app_icon(start: QtGui.QColor, end: QtGui.QColor) -> QtGui.QIcon:
    icon = QtGui.QIcon()
    for size in (16, 20, 24, 32, 48, 64, 256):
        icon.addPixmap(logo_pixmap(size, start, end))
    return icon


__all__ = [
    "LOGO",
    "MONO_MARK",
    "WORDMARK",
    "app_icon",
    "brand_path",
    "fitted",
    "logo_pixmap",
    "paint_logo",
    "paint_placeholder_mark",
    "paint_voice_glyph",
    "paint_wordmark",
    "renderer",
    "tinted_mark",
    "wordmark_aspect",
    "wordmark_renderer",
]
