import pytest

from spells import context, inject
from spells.models import TargetContext
from spells.platform.stub import StubFocus

from .fake_platform import fake_platform


class Focus(StubFocus):
    def foreground(self):
        return 42

    def app_name(self, window):
        return "notepad.exe" if window == 42 else ""

    def title(self, window):
        return "Untitled" if window == 42 else ""


def test_capture_asks_the_platform_focus(use_platform):
    use_platform(fake_platform(focus=Focus()))
    ctx = context.capture(clock=lambda: 5.0)
    assert ctx == TargetContext(window=42, process="notepad.exe", title="Untitled", captured_at=5.0)


def test_default_backends_come_from_the_platform(use_platform):
    record = use_platform(fake_platform(focus=Focus()))
    backends = inject.default_backends()
    assert backends.focus is record.focus
    assert backends.keyboard is record.keyboard
    assert backends.clipboard is record.clipboard


def test_default_backends_are_built_on_each_call(use_platform):
    use_platform(fake_platform())
    inject.default_backends()
    second = use_platform(fake_platform(focus=Focus()))
    assert inject.default_backends().focus is second.focus


@pytest.mark.windows
def test_windows_focus_and_keyboard_wrap_win32():
    from spells.platform import windows
    from spells.win32 import clipboard

    record = windows.build()
    assert record.clipboard is clipboard
    assert type(record.focus).__name__ == "WindowsFocus"
    assert type(record.keyboard).__name__ == "WindowsKeyboard"


@pytest.mark.windows
def test_windows_focus_asks_the_win32_window_module(monkeypatch):
    from spells.platform import windows
    from spells.win32 import window as win32_window

    monkeypatch.setattr(win32_window, "foreground_hwnd", lambda: 7)
    monkeypatch.setattr(win32_window, "window_process_name", {7: "a.exe"}.get)
    monkeypatch.setattr(win32_window, "window_title", {7: "A"}.get)
    monkeypatch.setattr(win32_window, "is_elevated_window", {7: True}.get)
    focus = windows.WindowsFocus()
    assert focus.foreground() == 7
    assert (focus.app_name(7), focus.title(7), focus.is_elevated(7)) == ("a.exe", "A", True)
    assert (focus.app_name(8), focus.title(8), focus.is_elevated(8)) == (None, None, None)


@pytest.mark.windows
def test_windows_keyboard_forwards_every_argument_to_win32_input(monkeypatch):
    from spells.platform import windows
    from spells.win32 import input as win32_input

    calls = []

    def recorder(name):
        return lambda *args, **kwargs: calls.append((name, args, kwargs))

    for name in ("type_unicode", "send_backspaces", "send_ctrl_v", "send_ctrl_c"):
        monkeypatch.setattr(win32_input, name, recorder(name))
    monkeypatch.setattr(win32_input, "release_held_modifiers", lambda: [0x5B])
    keyboard = windows.WindowsKeyboard()
    keyboard.type_unicode("hi", extra_info=9, release_modifiers=True)
    keyboard.type_unicode("hi", 3, extra_info=9)
    keyboard.send_backspaces(2, extra_info=9, release_modifiers=True)
    keyboard.send_backspaces(2, 1)
    keyboard.send_paste()
    keyboard.send_copy()
    assert keyboard.release_held_modifiers() == [0x5B]
    default = win32_input.DEFAULT_BATCH_SIZE
    assert calls == [
        ("type_unicode", ("hi", default), {"extra_info": 9, "release_modifiers": True}),
        ("type_unicode", ("hi", 3), {"extra_info": 9, "release_modifiers": False}),
        ("send_backspaces", (2, default), {"extra_info": 9, "release_modifiers": True}),
        ("send_backspaces", (2, 1), {"extra_info": 0, "release_modifiers": False}),
        ("send_ctrl_v", (), {}),
        ("send_ctrl_c", (), {}),
    ]
