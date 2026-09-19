"""Draw every logo concept at tray, taskbar and store sizes into one comparison sheet.

Usage:
    .venv/Scripts/python.exe build/brand/render_sheet.py

Writes build/brand/concepts/comparison-sheet.png. Each row shows the full colour mark at 16, 24,
32, 48 and 256 px on a dark and a light Windows 11 taskbar colour, with the 16 px render also
enlarged five times, then the single colour glyph tinted white and near-black at 16, 24 and
32 px. The chosen mark uses its heavier small master up to 24 px, as the ICO does.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

BRAND_DIR = Path(__file__).resolve().parent
REPO_DIR = BRAND_DIR.parents[1]
CONCEPTS_DIR = BRAND_DIR / "concepts"
DATA_BRAND_DIR = REPO_DIR / "data" / "brand"
SHEET_PATH = CONCEPTS_DIR / "comparison-sheet.png"

DARK_TASKBAR = "#1F1F1F"
LIGHT_TASKBAR = "#F3F3F3"
GLYPH_ON_DARK = "#FFFFFF"
GLYPH_ON_LIGHT = "#15212A"
COLOUR_SIZES = (16, 24, 32, 48)
MONO_SIZES = (16, 24, 32)
BIG = 256
ZOOM = 5
SMALL_MASTER_UP_TO = 24


@dataclass(frozen=True)
class Concept:
    name: str
    role: str
    idea: str
    colour: Path
    mono: Path
    small: Path | None = None

    def colour_for(self, size: int) -> Path:
        return self.small if self.small is not None and size <= SMALL_MASTER_UP_TO else self.colour


CONCEPTS = (
    Concept("rising-spark", "chosen",
            "Two voice bars rising into a spark, with a small glint above it.",
            DATA_BRAND_DIR / "spells-logo.svg", DATA_BRAND_DIR / "spells-mark-mono.svg",
            BRAND_DIR / "spells-logo-small.svg"),
    Concept("glint-s", "runner-up",
            "One stroke, half sound wave and half S, that throws off a glint.",
            CONCEPTS_DIR / "glint-s.svg", CONCEPTS_DIR / "glint-s-mono.svg",
            CONCEPTS_DIR / "glint-s-small.svg"),
    Concept("echo-spark", "explored",
            "A spark that speaks: two sound arcs ripple out of it.",
            CONCEPTS_DIR / "echo-spark.svg", CONCEPTS_DIR / "echo-spark-mono.svg"),
    Concept("spark-meter", "explored",
            "The pill's level bars with the loudest bar turned into a spark.",
            CONCEPTS_DIR / "spark-meter.svg", CONCEPTS_DIR / "spark-meter-mono.svg"),
    Concept("ink-wave", "explored",
            "A waveform that turns into a written flourish ending in a spark.",
            CONCEPTS_DIR / "ink-wave.svg", CONCEPTS_DIR / "ink-wave-mono.svg"),
    Concept("caret-wand", "explored",
            "Sound arcs flowing into a text caret that is also a wand.",
            CONCEPTS_DIR / "caret-wand.svg", CONCEPTS_DIR / "caret-wand-mono.svg"),
)


def _qt():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    if fonts.is_dir():
        os.environ.setdefault("QT_QPA_FONTDIR", str(fonts))
    from PySide6 import QtCore, QtGui, QtSvg

    app = QtGui.QGuiApplication.instance() or QtGui.QGuiApplication([sys.argv[0]])
    return app, QtCore, QtGui, QtSvg


def render(svg_path: Path, size: int, tint: str | None = None):
    _app, QtCore, QtGui, QtSvg = _qt()
    renderer = QtSvg.QSvgRenderer(str(svg_path))
    if not renderer.isValid():
        raise ValueError(f"{svg_path} is not an SVG QtSvg can draw")
    image = QtGui.QImage(size, size, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(image)
    try:
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        renderer.render(painter, QtCore.QRectF(0, 0, size, size))
        if tint is not None:
            painter.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_SourceIn)
            painter.fillRect(image.rect(), QtGui.QColor(tint))
    finally:
        painter.end()
    return image


def enlarge(image, factor: int):
    _app, QtCore, _QtGui, _QtSvg = _qt()
    return image.scaled(image.width() * factor, image.height() * factor,
                        QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                        QtCore.Qt.TransformationMode.FastTransformation)


def _strip_widths() -> tuple[int, int]:
    gap = 12
    colour = gap + sum(s + gap for s in COLOUR_SIZES) + 16 * ZOOM + gap + BIG + gap
    mono = gap + sum(s + gap for s in MONO_SIZES) + 16 * ZOOM + gap
    return colour, mono


def draw_sheet(concepts=CONCEPTS, target: Path = SHEET_PATH) -> Path:
    _app, QtCore, QtGui, _QtSvg = _qt()
    label_w = 230
    colour_w, mono_w = _strip_widths()
    row_h = BIG + 24
    header_h = 36
    width = label_w + 2 * colour_w + 2 * mono_w
    height = header_h + row_h * len(concepts)
    sheet = QtGui.QImage(width, height, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    sheet.fill(QtGui.QColor("#D9E0E3"))
    painter = QtGui.QPainter(sheet)
    try:
        painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing)
        heading = QtGui.QFont("Segoe UI")
        heading.setPixelSize(14)
        heading.setWeight(QtGui.QFont.Weight.DemiBold)
        body = QtGui.QFont("Segoe UI")
        body.setPixelSize(12)
        painter.setPen(QtGui.QColor("#15212A"))
        painter.setFont(heading)
        titles = ((label_w, colour_w, "Full colour, dark taskbar: 16, 24, 32, 48, 16 at 5x, 256"),
                  (label_w + colour_w, colour_w, "Full colour, light taskbar"),
                  (label_w + 2 * colour_w, mono_w, "Tray glyph, white"),
                  (label_w + 2 * colour_w + mono_w, mono_w, "Tray glyph, near-black"))
        for x, w, text in titles:
            painter.drawText(QtCore.QRectF(x + 12, 0, w - 12, header_h),
                             QtCore.Qt.AlignmentFlag.AlignVCenter, text)
        for index, concept in enumerate(concepts):
            top = header_h + index * row_h
            painter.setPen(QtGui.QColor("#15212A"))
            painter.setFont(heading)
            painter.drawText(QtCore.QRectF(12, top + 20, label_w - 24, 24),
                             QtCore.Qt.AlignmentFlag.AlignLeft, concept.name)
            painter.setFont(body)
            painter.setPen(QtGui.QColor("#485A64"))
            painter.drawText(QtCore.QRectF(12, top + 44, label_w - 24, 20),
                             QtCore.Qt.AlignmentFlag.AlignLeft, concept.role)
            painter.drawText(QtCore.QRectF(12, top + 70, label_w - 24, 120),
                             QtCore.Qt.TextFlag.TextWordWrap, concept.idea)
            x = label_w
            for background in (DARK_TASKBAR, LIGHT_TASKBAR):
                painter.fillRect(QtCore.QRect(x, top + 6, colour_w - 6, row_h - 12),
                                 QtGui.QColor(background))
                cx = x + 12
                middle = top + row_h // 2
                for size in COLOUR_SIZES:
                    image = render(concept.colour_for(size), size)
                    painter.drawImage(cx, middle - size // 2, image)
                    cx += size + 12
                small = enlarge(render(concept.colour_for(16), 16), ZOOM)
                painter.drawImage(cx, middle - 8 * ZOOM, small)
                cx += 16 * ZOOM + 12
                painter.drawImage(cx, middle - BIG // 2, render(concept.colour, BIG))
                x += colour_w
            tones = ((DARK_TASKBAR, GLYPH_ON_DARK), (LIGHT_TASKBAR, GLYPH_ON_LIGHT))
            for background, glyph in tones:
                painter.fillRect(QtCore.QRect(x, top + 6, mono_w - 6, row_h - 12),
                                 QtGui.QColor(background))
                cx = x + 12
                middle = top + row_h // 2
                for size in MONO_SIZES:
                    painter.drawImage(cx, middle - size // 2, render(concept.mono, size, glyph))
                    cx += size + 12
                painter.drawImage(cx, middle - 8 * ZOOM,
                                  enlarge(render(concept.mono, 16, glyph), ZOOM))
                x += mono_w
    finally:
        painter.end()
    target.parent.mkdir(parents=True, exist_ok=True)
    if not sheet.save(str(target)):
        raise OSError(f"could not write {target}")
    return target


def main() -> int:
    target = draw_sheet()
    print(f"wrote {target.relative_to(REPO_DIR)} ({target.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
