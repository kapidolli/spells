"""Rasterise data/brand/spells-logo.svg into build/brand/spells.ico and the PNG exports.

Usage:
    .venv/Scripts/python.exe build/brand/make_icons.py

PySide6 draws every size straight from an SVG, offscreen: build/brand/spells-logo-small.svg (a
heavier stroke and a larger glint) up to 24 px, data/brand/spells-logo.svg above that and for
the PNG exports. The ICO container is written with the standard library: 32-bit DIB entries
below 256 px, a PNG entry at 256 px.
"""

from __future__ import annotations

import os
import struct
import sys
from collections.abc import Sequence
from pathlib import Path

BRAND_DIR = Path(__file__).resolve().parent
REPO_DIR = BRAND_DIR.parents[1]
LOGO_SVG = REPO_DIR / "data" / "brand" / "spells-logo.svg"
SMALL_LOGO_SVG = BRAND_DIR / "spells-logo-small.svg"
SMALL_MASTER_UP_TO = 24
ICO_PATH = BRAND_DIR / "spells.ico"
ICO_SIZES = (16, 20, 24, 32, 40, 48, 64, 256)
PNG_SIZES = (256, 1024)
PNG_ENTRY_FROM = 256

_ICONDIR = struct.Struct("<HHH")
_ICONDIRENTRY = struct.Struct("<BBBBHHII")
_BITMAPINFOHEADER = struct.Struct("<IiiHHIIiiII")


class BrandError(Exception):
    """The logo could not be rasterised or an icon file is malformed."""


def png_path(size: int) -> Path:
    return BRAND_DIR / f"spells-{size}.png"


def dib_payload(size: int, bgra: bytes) -> bytes:
    """A 32-bit bottom-up DIB with its AND mask, from top-down BGRA rows."""
    stride = size * 4
    if len(bgra) != stride * size:
        raise BrandError(f"expected {stride * size} bytes of BGRA for {size} px, got {len(bgra)}")
    rows = [bgra[y * stride:(y + 1) * stride] for y in range(size)]
    mask_stride = ((size + 31) // 32) * 4
    mask_rows = []
    for row in rows:
        bits = bytearray(mask_stride)
        for x in range(size):
            if row[x * 4 + 3] == 0:
                bits[x // 8] |= 0x80 >> (x % 8)
        mask_rows.append(bytes(bits))
    pixels = b"".join(reversed(rows)) + b"".join(reversed(mask_rows))
    header = _BITMAPINFOHEADER.pack(40, size, size * 2, 1, 32, 0, len(pixels), 0, 0, 0, 0)
    return header + pixels


def ico_bytes(images: Sequence[tuple[int, bytes]]) -> bytes:
    """An ICO file from (size, payload) pairs, in the order given."""
    offset = _ICONDIR.size + _ICONDIRENTRY.size * len(images)
    entries = []
    for size, payload in images:
        if not 1 <= size <= 256:
            raise BrandError(f"an icon entry must be 1 to 256 px, not {size}")
        dim = 0 if size == 256 else size
        entries.append(_ICONDIRENTRY.pack(dim, dim, 0, 0, 1, 32, len(payload), offset))
        offset += len(payload)
    return (_ICONDIR.pack(0, 1, len(images)) + b"".join(entries)
            + b"".join(payload for _size, payload in images))


def ico_entries(data: bytes) -> list[tuple[int, int, bool]]:
    """(width, bits per pixel, is PNG) for every entry an ICO file's directory lists."""
    if len(data) < _ICONDIR.size:
        raise BrandError("too short to be an ICO file")
    reserved, kind, count = _ICONDIR.unpack_from(data, 0)
    if reserved != 0 or kind != 1:
        raise BrandError(f"not an ICO header: reserved {reserved}, type {kind}")
    found = []
    for index in range(count):
        width, height, _colours, _zero, _planes, bits, length, offset = _ICONDIRENTRY.unpack_from(
            data, _ICONDIR.size + index * _ICONDIRENTRY.size)
        if width != height:
            raise BrandError(f"entry {index} is not square: {width} by {height}")
        if offset + length > len(data):
            raise BrandError(f"entry {index} runs past the end of the file")
        is_png = data[offset:offset + 8] == b"\x89PNG\r\n\x1a\n"
        found.append((width or 256, bits, is_png))
    return found


def _qt():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtCore, QtGui, QtSvg

    app = QtGui.QGuiApplication.instance() or QtGui.QGuiApplication([sys.argv[0]])
    return app, QtCore, QtGui, QtSvg


def render(svg_path: Path, size: int):
    """The SVG drawn into a square, straight (not premultiplied) ARGB32 image."""
    _app, QtCore, QtGui, QtSvg = _qt()
    renderer = QtSvg.QSvgRenderer(str(svg_path))
    if not renderer.isValid():
        raise BrandError(f"{svg_path} is not an SVG QtSvg can draw")
    image = QtGui.QImage(size, size, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(image)
    try:
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        renderer.render(painter, QtCore.QRectF(0, 0, size, size))
    finally:
        painter.end()
    return image.convertToFormat(QtGui.QImage.Format.Format_ARGB32)


def bgra_bytes(image) -> bytes:
    size = image.width()
    stride = image.bytesPerLine()
    raw = bytes(image.constBits())
    return b"".join(raw[y * stride:y * stride + size * 4] for y in range(image.height()))


def png_bytes(image) -> bytes:
    _app, QtCore, _QtGui, _QtSvg = _qt()
    buffer = QtCore.QBuffer()
    buffer.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
    if not image.save(buffer, "PNG"):
        raise BrandError("Qt could not encode a PNG")
    return bytes(buffer.data())


def master_for(size: int, svg_path: Path = LOGO_SVG, small_svg_path: Path = SMALL_LOGO_SVG) -> Path:
    return small_svg_path if size <= SMALL_MASTER_UP_TO else svg_path


def build(svg_path: Path = LOGO_SVG, small_svg_path: Path = SMALL_LOGO_SVG,
          ico_path: Path = ICO_PATH) -> list[Path]:
    images = []
    for size in ICO_SIZES:
        image = render(master_for(size, svg_path, small_svg_path), size)
        if size >= PNG_ENTRY_FROM:
            images.append((size, png_bytes(image)))
        else:
            images.append((size, dib_payload(size, bgra_bytes(image))))
    ico_path.write_bytes(ico_bytes(images))
    written = [ico_path]
    for size in PNG_SIZES:
        target = png_path(size)
        target.write_bytes(png_bytes(render(svg_path, size)))
        written.append(target)
    return written


def main() -> int:
    try:
        written = build()
    except BrandError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    for path in written:
        print(f"wrote {path.relative_to(REPO_DIR)} ({path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
