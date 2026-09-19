"""The Spells marks in data/brand, the concepts and icon in build/brand, and their build wiring.

No rendering here beyond asking QtSvg whether it can read each file: the SVGs must be plain XML
on a 256 unit canvas with no text, images or external references, the ICO directory must list
every size the taskbar, tray and Explorer ask for, and the executable and the installer must
both be pointed at that ICO.
"""

from __future__ import annotations

import os
import re
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO_DIR = Path(__file__).resolve().parents[2]
BUILD_DIR = REPO_DIR / "build"
BRAND_BUILD_DIR = BUILD_DIR / "brand"
BRAND_DATA_DIR = REPO_DIR / "data" / "brand"
CONCEPTS_DIR = BRAND_BUILD_DIR / "concepts"
sys.path.insert(0, str(BUILD_DIR))
sys.path.insert(0, str(BRAND_BUILD_DIR))

import make_icons
import package

SVG_NS = "{http://www.w3.org/2000/svg}"
LOGO = BRAND_DATA_DIR / "spells-logo.svg"
MONO = BRAND_DATA_DIR / "spells-mark-mono.svg"
WORDMARK = BRAND_DATA_DIR / "spells-wordmark.svg"
SMALL = BRAND_BUILD_DIR / "spells-logo-small.svg"
SQUARE_SVGS = [LOGO, MONO, SMALL]
CONCEPT_SVGS = sorted(CONCEPTS_DIR.glob("*.svg"))
ALL_SVGS = [LOGO, MONO, WORDMARK, SMALL, *CONCEPT_SVGS]
FORBIDDEN_ELEMENTS = ("text", "tspan", "font", "image", "foreignObject", "script", "style")


def _root(path: Path) -> ET.Element:
    return ET.parse(path).getroot()


def _view_box(path: Path) -> list[float]:
    return [float(part) for part in _root(path).attrib["viewBox"].split()]


def _ids(path: Path) -> str:
    return path.relative_to(REPO_DIR).as_posix()


def _star_box(d: str) -> tuple[float, float, float, float]:
    """(left, top, right, bottom) of a four point star written as M tip Q ctl tip four times."""
    numbers = [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", d)]
    top = (numbers[0], numbers[1])
    right = (numbers[4], numbers[5])
    bottom = (numbers[8], numbers[9])
    return (2 * top[0] - right[0], top[1], right[0], bottom[1])


def test_the_three_marks_the_app_loads_exist():
    for path in (LOGO, MONO, WORDMARK):
        assert path.is_file(), path


@pytest.mark.parametrize("path", SQUARE_SVGS, ids=_ids)
def test_the_square_marks_are_svg_on_a_256_unit_canvas(path):
    root = _root(path)
    assert root.tag == f"{SVG_NS}svg"
    assert _view_box(path) == [0, 0, 256, 256]


def test_the_wordmark_is_as_tall_as_the_mark_and_wider_than_it():
    root = _root(WORDMARK)
    assert root.tag == f"{SVG_NS}svg"
    x, y, width, height = _view_box(WORDMARK)
    assert (x, y, height) == (0, 0, 256)
    assert width > 256


def test_there_are_at_least_five_concepts_each_with_a_single_colour_glyph():
    colour = [p for p in CONCEPT_SVGS if not p.stem.endswith(("-mono", "-small"))]
    assert len(colour) >= 5
    for path in colour:
        assert (CONCEPTS_DIR / f"{path.stem}-mono.svg").is_file(), path


def test_the_runner_up_is_kept_beside_the_other_concepts():
    for name in ("glint-s.svg", "glint-s-mono.svg", "glint-s-small.svg"):
        assert (CONCEPTS_DIR / name).is_file(), name


@pytest.mark.parametrize("path", [LOGO, MONO, SMALL], ids=_ids)
def test_the_mark_is_two_rounded_bars_a_spark_and_a_glint(path):
    root = _root(path)
    bars = [element for element in root.iter(f"{SVG_NS}rect")
            if float(element.attrib["rx"]) * 2 == pytest.approx(float(element.attrib["width"]))]
    sparks = list(root.iter(f"{SVG_NS}path"))

    assert len(bars) == 2
    heights = sorted(float(bar.attrib["height"]) for bar in bars)
    assert heights[0] * 1.5 < heights[1]
    assert len(sparks) == 2
    assert all(spark.attrib["d"].count("Q") == 4 for spark in sparks)
    big, glint = sorted((_star_box(spark.attrib["d"]) for spark in sparks),
                        key=lambda box: box[3] - box[1], reverse=True)
    assert (big[3] - big[1]) > 2 * (glint[3] - glint[1])
    assert glint[0] > (big[0] + big[2]) / 2
    assert glint[3] < (big[1] + big[3]) / 2


@pytest.mark.parametrize("path", CONCEPT_SVGS, ids=_ids)
def test_every_concept_is_svg_on_a_256_unit_canvas(path):
    assert _root(path).tag == f"{SVG_NS}svg"
    assert _view_box(path) == [0, 0, 256, 256]


@pytest.mark.parametrize("path", ALL_SVGS, ids=_ids)
def test_no_svg_carries_text_images_scripts_or_external_references(path):
    root = _root(path)
    for element in root.iter():
        local = element.tag.split("}", 1)[-1]
        assert local not in FORBIDDEN_ELEMENTS, f"{path.name} has a <{local}>"
        for name, value in element.attrib.items():
            if name.endswith("href"):
                assert value.startswith("#"), f"{path.name} links out: {value}"
    text = path.read_text(encoding="utf-8")
    assert "http://" not in text.replace('xmlns="http://www.w3.org/2000/svg"', "")
    assert "https://" not in text
    assert "\u2014" not in text


@pytest.mark.parametrize("path", ALL_SVGS, ids=_ids)
def test_every_gradient_reference_resolves_inside_its_own_file(path):
    root = _root(path)
    ids = {element.attrib["id"] for element in root.iter() if "id" in element.attrib}
    for element in root.iter():
        for value in element.attrib.values():
            if value.startswith("url(#"):
                assert value[5:-1] in ids, f"{path.name} refers to a missing {value}"


def test_the_logo_uses_the_brand_gradient_and_a_white_glyph():
    text = LOGO.read_text(encoding="utf-8")
    assert 'stop-color="#5B3CF0"' in text
    assert 'stop-color="#16B4E8"' in text
    assert 'fill="#FFFFFF"' in text


def test_the_mono_glyph_is_one_colour_the_app_can_tint():
    root = _root(MONO)
    fills = {element.attrib["fill"] for element in root.iter() if "fill" in element.attrib}
    assert fills == {"currentColor"}
    assert "linearGradient" not in MONO.read_text(encoding="utf-8")
    assert "stroke" not in MONO.read_text(encoding="utf-8")


def test_the_wordmark_letters_follow_the_current_colour():
    root = _root(WORDMARK)
    assert root.attrib.get("color") == "#15212A"
    assert any(element.attrib.get("fill") == "currentColor" for element in root.iter())


@pytest.mark.parametrize("path", [LOGO, MONO, WORDMARK, SMALL], ids=_ids)
def test_qtsvg_can_draw_every_shipped_mark(path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    qtsvg = pytest.importorskip("PySide6.QtSvg")
    assert qtsvg.QSvgRenderer(str(path)).isValid()


def test_the_icon_lists_every_size_windows_asks_for():
    entries = make_icons.ico_entries(make_icons.ICO_PATH.read_bytes())
    assert [width for width, _bits, _png in entries] == [16, 20, 24, 32, 40, 48, 64, 256]
    assert list(make_icons.ICO_SIZES) == [16, 20, 24, 32, 40, 48, 64, 256]


def test_the_icon_is_32_bit_with_a_png_entry_only_at_256():
    entries = make_icons.ico_entries(make_icons.ICO_PATH.read_bytes())
    assert all(bits == 32 for _width, bits, _png in entries)
    assert [width for width, _bits, png in entries if png] == [256]


def test_the_icon_header_is_the_standard_icondir():
    data = make_icons.ICO_PATH.read_bytes()
    reserved, kind, count = struct.unpack_from("<HHH", data, 0)
    assert (reserved, kind, count) == (0, 1, 8)
    first_width, first_height = struct.unpack_from("<BB", data, 6)
    assert (first_width, first_height) == (16, 16)


def test_the_small_entries_are_bottom_up_dibs_with_a_doubled_height():
    data = make_icons.ICO_PATH.read_bytes()
    _dim, _dim2, _c, _z, _p, _bits, length, offset = struct.unpack_from("<BBBBHHII", data, 6)
    header = struct.unpack_from("<IiiHHI", data, offset)
    assert header == (40, 16, 32, 1, 32, 0)
    assert length == 40 + 16 * 16 * 4 + 16 * 4


@pytest.mark.parametrize("size", [256, 1024])
def test_the_png_exports_have_their_size(size):
    data = make_icons.png_path(size).read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    assert (width, height) == (size, size)


def test_an_icon_written_and_read_back_keeps_its_entries():
    transparent = bytes(4) * 4
    opaque = b"\x10\x20\x30\xff" * 4
    dib = make_icons.dib_payload(2, transparent[:8] + opaque[:8])
    png = b"\x89PNG\r\n\x1a\n" + bytes(16)
    data = make_icons.ico_bytes([(2, dib), (256, png)])

    assert make_icons.ico_entries(data) == [(2, 32, False), (256, 32, True)]


def test_the_and_mask_marks_only_fully_transparent_pixels():
    pixels = bytes(4) + b"\x00\x00\x00\x01" + b"\x01\x02\x03\xff" * 2
    dib = make_icons.dib_payload(2, pixels)
    mask = dib[40 + 2 * 2 * 4:]
    assert len(mask) == 2 * 4
    assert mask[:4] == bytes(4)
    assert mask[4:] == b"\x80\x00\x00\x00"


def test_a_malformed_icon_is_refused():
    with pytest.raises(make_icons.BrandError):
        make_icons.ico_entries(b"\x00\x00\x02\x00\x00\x00")
    with pytest.raises(make_icons.BrandError):
        make_icons.ico_bytes([(300, b"")])


def test_sizes_up_to_24_px_come_from_the_heavier_master():
    assert make_icons.master_for(16) == make_icons.SMALL_LOGO_SVG
    assert make_icons.master_for(24) == make_icons.SMALL_LOGO_SVG
    assert make_icons.master_for(32) == make_icons.LOGO_SVG
    assert make_icons.master_for(256) == make_icons.LOGO_SVG


def test_the_packaged_executable_carries_the_brand_icon():
    assert package.APP_ICON == make_icons.ICO_PATH
    args = package.pyinstaller_args(
        Path("python.exe"), Path("entry.py"),
        dist_dir=Path("dist"), work_dir=Path("work"), src_dir=Path("src"))

    assert args[args.index("--icon") + 1] == str(make_icons.ICO_PATH)
    assert args.index("--icon") < len(args) - 1
    assert args[-1] == "entry.py"


def test_the_installer_and_its_uninstall_entry_show_the_brand_icon():
    values = package.iss_values(
        version="0.1.0",
        output_dir=Path(r"C:\repo\dist"),
        dist_dir=Path(r"C:\repo\dist\Spells"),
        cleanup_model_name="qwen3-4b-q4_k_m.gguf",
        licenses_dir=Path(r"C:\repo\build\licenses"),
    )
    text = package.render_iss(package.ISS_TEMPLATE.read_text(encoding="utf-8"), values)

    assert f"SetupIconFile={make_icons.ICO_PATH}" in text
    assert "UninstallDisplayIcon={app}\\Spells.exe" in text


def test_a_missing_icon_stops_the_build_with_the_command_that_makes_it(tmp_path):
    with pytest.raises(package.PackageError, match="make_icons.py"):
        package.check_app_icon(tmp_path / "spells.ico")
    package.check_app_icon(make_icons.ICO_PATH)
