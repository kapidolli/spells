from __future__ import annotations

import dataclasses
import re

import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from spells.platform.stub import StubAppearance
from spells.ui import style, theme
from spells.ui.pill import Pill, place_pill
from spells.ui.style import AccentShades

from .fake_platform import fake_platform
from .test_ui_pill import FakeScreen
from .test_ui_support import PillState, TrayState, event, flush, qt_app

PALETTE = ("#F1E2D3", "#E1D2C3", "#D1C2B3", "#C1B2A3", "#B1A293", "#A19283", "#918273")
MAGENTA = bytes.fromhex("FF97FA00FF35EE00C800B3009A0089008C007C006B005E004C00420000CC6A00")
ACCENT_KEY = r"Software\Microsoft\Windows\CurrentVersion\Explorer\Accent"
DWM_KEY = r"Software\Microsoft\Windows\DWM"
PERSONALIZE_KEY = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
MONITOR = (2560, -120, 5120, 1320)


@pytest.fixture(scope="module")
def app():
    return qt_app()


class FakeAppearance(StubAppearance):
    def __init__(
        self,
        *,
        palette: tuple[str, ...] | None = None,
        colour: str | None = None,
        dark: bool | None = None,
        light_taskbar: bool = False,
        animations: bool = True,
        result: bool = True,
    ) -> None:
        self.palette = palette
        self.colour = colour
        self.dark = dark
        self.light_taskbar = light_taskbar
        self.animations = animations
        self.result = result
        self.calls: list[tuple] = []

    def apps_dark(self):
        return self.dark

    def taskbar_light(self):
        return self.light_taskbar

    def accent_palette(self):
        return self.palette

    def accent_colour(self):
        return self.colour

    def animations_enabled(self):
        return self.animations

    def style_title_bar(self, native_id, *, dark, caption, text):
        self.calls.append(("title", native_id, dark, caption, text))
        return self.result

    def round_corners(self, native_id):
        self.calls.append(("round", native_id))
        return self.result


class FakeOverlay:
    def __init__(self, rect: tuple[int, int, int, int] = MONITOR) -> None:
        self.rect = rect
        self.asked: list[int] = []
        self.guarded: list[int] = []

    def monitor_rect_for(self, window):
        self.asked.append(window)
        return self.rect

    def set_no_activate(self, native_id):
        self.guarded.append(native_id)


def test_the_system_accent_is_the_platform_palette(use_platform):
    use_platform(fake_platform(appearance=FakeAppearance(palette=PALETTE, colour="#0078D4")))

    assert style.system_accent() == AccentShades(*PALETTE)


def test_the_system_accent_spreads_a_single_platform_colour(app, use_platform):
    use_platform(fake_platform(appearance=FakeAppearance(colour="#0078D4")))

    assert style.system_accent() == style.shades_from_colour(QtGui.QColor("#0078D4"))


def test_there_is_no_system_accent_without_a_palette_or_a_colour(use_platform):
    use_platform(fake_platform(appearance=FakeAppearance()))

    assert style.system_accent() is None


@pytest.mark.parametrize("dark", [True, False])
def test_the_app_theme_follows_the_platform_setting(use_platform, dark):
    use_platform(fake_platform(appearance=FakeAppearance(dark=dark)))

    assert style.apps_use_dark_theme() is dark


def test_the_app_theme_falls_back_to_the_qt_colour_scheme(app, use_platform):
    use_platform(fake_platform(appearance=FakeAppearance(dark=None)))

    assert style.apps_use_dark_theme() is (app.styleHints().colorScheme() == QtCore.Qt.ColorScheme.Dark)


@pytest.mark.parametrize("light", [True, False])
def test_the_taskbar_theme_follows_the_platform(use_platform, light):
    use_platform(fake_platform(appearance=FakeAppearance(light_taskbar=light)))

    assert theme.taskbar_is_light() is light


@pytest.mark.parametrize("animations", [True, False])
def test_reduced_motion_follows_the_platform_animation_setting(use_platform, animations):
    use_platform(fake_platform(appearance=FakeAppearance(animations=animations)))

    assert theme.reduced_motion() is (not animations)


def test_motion_stays_on_when_the_animation_setting_cannot_be_read(use_platform):
    class Unreadable(FakeAppearance):
        def animations_enabled(self):
            raise OSError("no animation setting")

    use_platform(fake_platform(appearance=Unreadable()))

    assert theme.reduced_motion() is False


def test_window_chrome_styles_the_title_bar_through_the_platform(app, use_platform, monkeypatch):
    appearance = FakeAppearance()
    use_platform(fake_platform(appearance=appearance))
    monkeypatch.setattr(style, "_native_windows", lambda: True)
    widget = QtWidgets.QWidget()
    current = style.palette()

    assert style.apply_window_chrome(widget) is True
    assert appearance.calls == [("title", int(widget.winId()), current.dark, current.base, current.text)]
    widget.close()


def test_window_chrome_skips_frameless_windows_and_other_systems(app, use_platform, monkeypatch):
    appearance = FakeAppearance()
    use_platform(fake_platform(appearance=appearance))
    frameless = QtWidgets.QWidget(None, QtCore.Qt.WindowType.FramelessWindowHint)
    framed = QtWidgets.QWidget()

    monkeypatch.setattr(style, "_native_windows", lambda: True)
    assert style.apply_window_chrome(frameless) is False
    monkeypatch.setattr(style, "_native_windows", lambda: False)
    assert style.apply_window_chrome(framed) is False
    assert style.round_popup(framed) is False
    assert appearance.calls == []
    frameless.close()
    framed.close()


def test_window_chrome_reports_a_failing_platform_as_skipped(app, use_platform, monkeypatch):
    class Failing(FakeAppearance):
        def style_title_bar(self, native_id, *, dark, caption, text):
            raise OSError("no title bar")

        def round_corners(self, native_id):
            raise OSError("no corners")

    use_platform(fake_platform(appearance=Failing()))
    monkeypatch.setattr(style, "_native_windows", lambda: True)
    widget = QtWidgets.QWidget()

    assert style.apply_window_chrome(widget) is False
    assert style.round_popup(widget) is False
    widget.close()


@pytest.mark.parametrize("rounded", [True, False])
def test_popups_are_rounded_through_the_platform(app, use_platform, monkeypatch, rounded):
    appearance = FakeAppearance(result=rounded)
    use_platform(fake_platform(appearance=appearance))
    monkeypatch.setattr(style, "_native_windows", lambda: True)
    menu = QtWidgets.QMenu()

    assert style.round_popup(menu) is rounded
    assert appearance.calls == [("round", int(menu.winId()))]
    menu.close()


def test_the_pill_is_placed_and_kept_unfocused_through_the_platform_overlay(app, use_platform):
    overlay = FakeOverlay()
    use_platform(fake_platform(overlay=overlay))
    corners: list[QtCore.QPoint] = []
    area = QtCore.QRect(2560, -120, 2560, 1400)

    def screen_at(point):
        corners.append(QtCore.QPoint(point))
        return FakeScreen(area)

    pill = Pill(reduced_motion=True, screen_at=screen_at)
    pill.apply(event(PillState.RECORDING, TrayState.RECORDING, target_window=4242))
    flush(app)

    assert overlay.asked and set(overlay.asked) == {4242}
    assert corners[0] == QtCore.QPoint(2560, -120)
    assert overlay.guarded == [int(pill.winId())]
    x, y = place_pill((2560, -120, 5120, 1280), 180, 36, pill.metrics)
    assert (pill.x(), pill.y()) == (round(x), round(y))
    pill.close()


def test_the_pill_uses_the_primary_screen_where_the_platform_cannot_place_it(app, use_platform):
    use_platform(fake_platform())
    pill = Pill(reduced_motion=True)

    assert pill.screen_for(4242) is app.primaryScreen()
    pill.close()


def test_a_failing_overlay_does_not_stop_the_pill_showing(app, use_platform):
    class Refusing(FakeOverlay):
        def set_no_activate(self, native_id):
            raise OSError("no window")

    use_platform(fake_platform(overlay=Refusing()))
    pill = Pill(reduced_motion=True, screen_at=lambda point: FakeScreen())
    pill.apply(event(PillState.RECORDING, TrayState.RECORDING, target_window=4242))
    flush(app)

    assert pill.isVisible()
    pill.close()


def test_the_stub_leaves_the_theme_to_qt_and_styles_no_window():
    appearance = StubAppearance()

    assert appearance.apps_dark() is None
    assert appearance.taskbar_light() is False
    assert appearance.accent_palette() is None
    assert appearance.animations_enabled() is True
    assert appearance.style_title_bar(1, dark=True, caption="#000000", text="#FFFFFF") is False
    assert appearance.round_corners(1) is False


def test_the_stub_accent_colour_needs_a_qt_application(monkeypatch):
    monkeypatch.setattr(QtGui.QGuiApplication, "instance", staticmethod(lambda: None))

    assert StubAppearance().accent_colour() is None


def test_the_stub_accent_colour_is_the_qt_accent_as_hex(app):
    colour = StubAppearance().accent_colour()

    assert re.fullmatch(r"#[0-9A-F]{6}", colour)
    assert colour == QtGui.QGuiApplication.palette().color(QtGui.QPalette.ColorRole.Accent).name().upper()


@pytest.mark.windows
def test_windows_builds_its_overlay_and_appearance_from_win32():
    from spells.platform import windows

    record = windows.build()

    assert isinstance(record.overlay, windows.WindowsOverlay)
    assert isinstance(record.appearance, windows.WindowsAppearance)
    palette = record.appearance.accent_palette()
    assert palette is None or (len(palette) == 7 and all(colour.startswith("#") for colour in palette))
    assert isinstance(record.appearance.animations_enabled(), bool)


@pytest.mark.windows
def test_windows_reads_the_same_registry_values_as_before(monkeypatch):
    from spells.platform import windows
    from spells.win32 import appearance as win32_appearance

    values = {
        (ACCENT_KEY, "AccentPalette"): MAGENTA,
        (DWM_KEY, "AccentColor"): 0xFFD47800,
        (PERSONALIZE_KEY, "AppsUseLightTheme"): 0,
        (PERSONALIZE_KEY, "SystemUsesLightTheme"): 1,
    }
    monkeypatch.setattr(win32_appearance, "_read_registry", lambda path, name: values.get((path, name)))
    appearance = windows.build().appearance

    assert appearance.accent_palette() == dataclasses.astuple(style.shades_from_palette_bytes(MAGENTA))
    assert appearance.accent_colour() == "#0078D4"
    assert appearance.apps_dark() is True
    assert appearance.taskbar_light() is True

    values.update({
        (ACCENT_KEY, "AccentPalette"): b"\x00" * 8,
        (DWM_KEY, "AccentColor"): "blue",
        (PERSONALIZE_KEY, "AppsUseLightTheme"): 1,
        (PERSONALIZE_KEY, "SystemUsesLightTheme"): 0,
    })
    assert appearance.accent_palette() is None
    assert appearance.accent_colour() is None
    assert appearance.apps_dark() is False
    assert appearance.taskbar_light() is False

    values.clear()
    assert appearance.accent_palette() is None
    assert appearance.accent_colour() is None
    assert appearance.apps_dark() is None
    assert appearance.taskbar_light() is False


@pytest.mark.windows
def test_windows_sets_the_same_dwm_attributes_as_before(monkeypatch):
    from spells.platform import windows
    from spells.win32 import appearance as win32_appearance

    calls: list[tuple[int, int, int]] = []
    answer = {"ok": True}

    def set_attribute(hwnd, attribute, value):
        calls.append((hwnd, attribute, value))
        return answer["ok"]

    monkeypatch.setattr(win32_appearance, "_set_dwm_attribute", set_attribute)
    appearance = windows.build().appearance

    assert appearance.style_title_bar(99, dark=True, caption="#1C1C22", text="#F3F3F7") is True
    assert appearance.round_corners(99) is True
    answer["ok"] = False
    assert appearance.style_title_bar(98, dark=False, caption="#F3F3F7", text="#1A1A22") is True
    assert appearance.round_corners(98) is False
    assert calls == [
        (99, 20, 1),
        (99, 35, style.colorref("#1C1C22")),
        (99, 36, style.colorref("#F3F3F7")),
        (99, 33, 3),
        (98, 20, 0),
        (98, 35, style.colorref("#F3F3F7")),
        (98, 36, style.colorref("#1A1A22")),
        (98, 33, 3),
    ]


@pytest.mark.windows
def test_windows_overlay_and_motion_come_from_the_win32_window_helpers(monkeypatch):
    from spells.platform import windows
    from spells.win32 import window as win32_window

    guarded: list[int] = []
    monkeypatch.setattr(win32_window, "monitor_rect_for_window", lambda hwnd: (hwnd, 0, 10, 20))
    monkeypatch.setattr(win32_window, "set_window_no_activate", lambda hwnd: guarded.append(hwnd) or 0x08000080)
    monkeypatch.setattr(win32_window, "animations_enabled", lambda: False)
    record = windows.build()

    assert record.overlay.monitor_rect_for(5) == (5, 0, 10, 20)
    assert record.overlay.set_no_activate(6) is None
    assert guarded == [6]
    assert record.appearance.animations_enabled() is False
