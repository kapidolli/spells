import os
import sys
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("fcntl")
if not sys.platform.startswith("linux"):
    pytest.skip("exercises Linux itself", allow_module_level=True)

from spells import vk
from spells.linux import clipboard, desktop, evdev, instance, keycodes, kwin, process
from spells.models import Chord, CpuPlan
from spells.platform import linux
from spells.platform.base import MessageHandlers


def test_letters_digits_and_modifiers_map_to_their_vk():
    assert keycodes.vk_for_code(30) == ord("A")
    assert keycodes.vk_for_code(16) == ord("Q")
    assert keycodes.vk_for_code(50) == ord("M")
    assert keycodes.vk_for_code(2) == ord("1")
    assert keycodes.vk_for_code(11) == ord("0")
    assert keycodes.vk_for_code(59) == 0x70
    assert keycodes.vk_for_code(88) == 0x7B
    assert keycodes.vk_for_code(29) == vk.VK_LCONTROL
    assert keycodes.vk_for_code(97) == vk.VK_RCONTROL
    assert keycodes.vk_for_code(125) == vk.VK_LWIN
    assert keycodes.vk_for_code(1) == 0x1B


def test_unknown_codes_never_collide_with_a_vk():
    assert keycodes.vk_for_code(0x2A0) == keycodes.UNMAPPED_BASE + 0x2A0
    assert keycodes.codes_for_vk(keycodes.UNMAPPED_BASE + 0x2A0) == (0x2A0,)


def test_generic_modifiers_cover_both_sides():
    assert keycodes.codes_for_vk(vk.VK_CONTROL) == (29, 97)
    assert keycodes.codes_for_vk(vk.VK_SHIFT) == (42, 54)
    assert keycodes.codes_for_vk(vk.VK_MENU) == (56, 100)
    assert keycodes.codes_for_vk(vk.VK_LWIN) == (125,)
    assert keycodes.codes_for_vk(0xFFFF) == ()


def test_every_mapped_code_round_trips():
    for code, value in keycodes.CODE_TO_VK.items():
        assert code in keycodes.codes_for_vk(value)


def test_ioctl_numbers_match_the_kernel_headers():
    assert evdev.eviocgbit(evdev.EV_KEY, 96) == 0x80604521
    assert evdev.eviocgkey(96) == 0x80604518
    assert evdev.eviocgname(256) == 0x81004506


def _event(code, value, kind=evdev.EV_KEY, sec=12, usec=345678):
    return evdev.EVENT.pack(sec, usec, kind, code, value)


def test_parse_events_keeps_key_events_only():
    data = _event(30, 1) + _event(0, 0, kind=0) + _event(30, 0) + b"\x01\x02"
    assert evdev.parse_events(data) == [(12, 345678, 30, 1), (12, 345678, 30, 0)]


def test_event_for_builds_the_hook_event_shape():
    event = evdev.event_for(12, 345678, 125, 2)
    assert event.vk == vk.VK_LWIN
    assert event.keydown is True
    assert event.extra_info == 0
    assert event.time_ms == 12345
    assert evdev.event_for(1, 0, 125, 0).keydown is False


def test_a_keyboard_needs_letters_and_space():
    bits = bytearray(evdev.KEY_BYTES)
    for code in (keycodes.KEY_A, keycodes.KEY_Z):
        bits[code // 8] |= 1 << (code % 8)
    assert evdev.is_keyboard(bytes(bits)) is False
    bits[keycodes.KEY_SPACE // 8] |= 1 << (keycodes.KEY_SPACE % 8)
    assert evdev.is_keyboard(bytes(bits)) is True


class FakeKeyboard:
    def __init__(self):
        self.sent = []

    def key(self, code, down):
        self.sent.append(("key", code, down))

    def tap(self, *codes, repeat=1):
        self.sent.append(("tap", codes, repeat))


def test_the_hook_refuses_when_no_keyboard_is_readable(tmp_path):
    hook = evdev.EvdevHook(FakeKeyboard(), device_glob=str(tmp_path / "event*"))
    with pytest.raises(evdev.NoKeyboardAccess):
        hook.install_keyboard_hook(lambda event: False)


def test_the_message_loop_feeds_events_and_fires_timers(tmp_path):
    hook = evdev.EvdevHook(FakeKeyboard(), device_glob=str(tmp_path / "event*"))
    read_end, write_end = os.pipe()
    os.set_blocking(read_end, False)
    hook._devices["fake"] = read_end
    hook._next_rescan = float("inf")
    seen = []
    ticks = []
    stop = threading.Event()

    def callback(event):
        seen.append((event.vk, event.keydown))
        if len(seen) == 2:
            stop.set()
        return False

    hook._callback = callback
    os.write(write_end, _event(29, 1) + _event(29, 0))
    hook.run_message_loop(stop, {1: 1}, ticks.append)
    os.close(write_end)
    hook.uninstall_keyboard_hook(1)
    assert seen == [(vk.VK_LCONTROL, True), (vk.VK_LCONTROL, False)]


def test_send_key_types_the_first_code_of_a_vk():
    keyboard = FakeKeyboard()
    hook = evdev.EvdevHook(keyboard)
    hook.send_key(vk.VK_CONTROL, True)
    hook.send_key(ord("V"), False)
    assert keyboard.sent == [("key", 29, True), ("key", 47, False)]


def test_the_hook_drives_the_chord_state_machine():
    from spells.hotkey import HotkeyCallbacks, HotkeyThread

    pressed = []
    callbacks = HotkeyCallbacks(
        pressed=lambda chord, i: pressed.append(i),
        released=lambda chord, i: None,
        latched=lambda chord, i: None,
        discarded=lambda i: None,
        cancelled=lambda i: None,
    )
    hook = evdev.EvdevHook(FakeKeyboard())
    thread = HotkeyThread([Chord(keys=(vk.VK_CONTROL, vk.VK_LWIN))], callbacks, hook_backend=hook)
    thread.hook_callback(evdev.event_for(0, 0, 29, 1))
    thread.hook_callback(evdev.event_for(0, 0, 125, 1))
    assert pressed == [1]
    assert thread.recording


def test_clipboard_snapshot_prefers_images_then_files_then_text():
    assert clipboard.pick_type(["text/plain", "image/png"]) == "image/png"
    assert clipboard.pick_type(["text/plain", "text/uri-list"]) == "text/uri-list"
    assert clipboard.pick_type(["UTF8_STRING", "text/plain;charset=utf-8"]) == (
        "text/plain;charset=utf-8"
    )
    assert clipboard.pick_type(["application/x-thing"]) == "application/x-thing"
    assert clipboard.pick_type([]) is None


def test_clipboard_fingerprint_changes_with_types_or_text():
    base = clipboard.fingerprint(["text/plain"], b"hello")
    assert base == clipboard.fingerprint(["text/plain"], b"hello")
    assert base != clipboard.fingerprint(["text/plain"], b"hello!")
    assert base != clipboard.fingerprint(["text/plain", "image/png"], b"hello")
    assert 0 <= base <= 0x7FFFFFFF


def test_focus_tracker_maps_kwin_ids_to_small_integers():
    tracker = kwin.FocusTracker()
    assert tracker.foreground() == kwin.PSEUDO_WINDOW
    tracker._live = True
    tracker.activated("{a}", "kitty", "shell", "42", "0,106,1920,1080")
    first = tracker.foreground()
    tracker.activated("{b}", "firefox", "Gmail", "7.0", "")
    second = tracker.foreground()
    tracker.activated("{a}", "kitty", "vim", "42", "0,106,1920,1080")
    assert tracker.foreground() == first != second
    assert tracker.info(first) == kwin.WindowInfo("kitty", "vim", 42, (0, 106, 1920, 1186))
    assert tracker.info(second).pid == 7
    assert tracker.info(second).output is None
    tracker.activated("", "", "", "0", "")
    assert tracker.foreground() == 0


def test_the_kwin_script_calls_our_method():
    assert kwin.BUS_NAME in kwin.SCRIPT
    assert '"Activated"' in kwin.SCRIPT
    assert "workspace.windowActivated.connect" in kwin.SCRIPT


def test_cpu_lists_and_one_thread_per_core():
    assert process.parse_cpu_list("0-3,8,10-11\n") == {0, 1, 2, 3, 8, 10, 11}
    siblings = {0: {0, 1}, 1: {0, 1}, 2: {2, 3}, 3: {2, 3}}
    assert process.one_per_core({0, 1, 2, 3}, siblings) == [0, 2]
    assert process.plan_from({0, 1, 2, 3}, True, siblings) == CpuPlan(threads=2, affinity_mask=0b101)
    assert process.plan_from({0, 1}, False, siblings) == CpuPlan()
    assert process.mask_cpus(0b1010) == {1, 3}


def test_spawn_hidden_writes_the_engine_log(tmp_path):
    log = tmp_path / "engine.log"
    job = process.EngineJob()
    proc = process.spawn_hidden(
        ["sh", "-c", "echo out; echo err >&2"], stdout_path=log, stderr_path=log, job=job
    )
    assert proc.wait(5) == 0
    assert log.read_text().split() == ["out", "err"]
    job.close()


def test_single_instance_lock_and_command_socket(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    assert instance.acquire("Spells-test")
    assert not instance.acquire("Spells-test")
    received = []
    stop = threading.Event()
    window = instance.SocketWindow(
        "SpellsWindow", MessageHandlers(on_copydata=lambda text: (received.append(text), stop.set()))
    )
    thread = threading.Thread(target=window.run, args=(stop,))
    thread.start()
    assert instance.find_running("SpellsWindow") == 1
    assert instance.signal_running("SpellsWindow", "open-settings")
    thread.join(5)
    window.destroy()
    instance.release("Spells-test")
    assert received == ["open-settings"]
    assert instance.find_running("SpellsWindow") == 0
    assert instance.acquire("Spells-test")
    instance.release("Spells-test")


def test_autostart_writes_and_reads_a_desktop_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert desktop.autostart_exec() == ""
    assert desktop.apply_autostart(True, "/opt/spells/spells --autostart")
    assert desktop.autostart_exec() == "/opt/spells/spells --autostart"
    assert desktop.apply_autostart(False, "")
    assert desktop.autostart_exec() == ""


def test_autostart_command_starts_this_installation():
    assert desktop.autostart_command(frozen=True, executable="/opt/Spells/spells") == (
        "/opt/Spells/spells"
    )
    assert desktop.autostart_command(frozen=False, executable="/home/a b/.venv/bin/python") == (
        '"/home/a b/.venv/bin/python" -m spells'
    )


def test_qt_runs_through_xwayland_unless_told_otherwise():
    env = {"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"}
    linux.choose_qt_platform(env)
    assert env["QT_QPA_PLATFORM"] == "xcb"
    env = {"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0", "QT_QPA_PLATFORM": "wayland"}
    linux.choose_qt_platform(env)
    assert env["QT_QPA_PLATFORM"] == "wayland"
    env = {"WAYLAND_DISPLAY": "wayland-0", "SPELLS_QT_PLATFORM": "wayland"}
    linux.choose_qt_platform(env)
    assert env["QT_QPA_PLATFORM"] == "wayland"
    env = {"DISPLAY": ":0"}
    linux.choose_qt_platform(env)
    assert "QT_QPA_PLATFORM" not in env


def test_capabilities_follow_what_works():
    caps = linux.capabilities(keys_out=True, focus=True, clipboard=True)
    assert caps.hold_to_talk and caps.esc_cancel and caps.app_profiles and caps.edit_hotkey
    assert not caps.live_typing and not caps.self_update and not caps.elevation_check
    caps = linux.capabilities(keys_out=False, focus=False, clipboard=True)
    assert not caps.edit_hotkey and not caps.app_profiles and not caps.window_titles


class FakeFocus:
    def __init__(self, app):
        self.app = app

    def foreground(self):
        return 5

    def app_name(self, window):
        return self.app


class FakeClipboard:
    def __init__(self):
        self.calls = []

    def snapshot(self):
        self.calls.append("snapshot")
        return "snap"

    def set_text(self, text):
        self.calls.append(("set", text))
        return 9

    def restore(self, snap, sequence):
        self.calls.append(("restore", snap, sequence))
        return True


def test_paste_uses_ctrl_shift_v_in_terminals(monkeypatch):
    monkeypatch.setattr(linux, "PASTE_SETTLE_S", 0)
    keys = FakeKeyboard()
    linux.LinuxKeyboard(keys, FakeFocus("kitty"), FakeClipboard()).send_paste()
    linux.LinuxKeyboard(keys, FakeFocus("firefox"), FakeClipboard()).send_paste()
    linux.LinuxKeyboard(keys, FakeFocus("org.kde.konsole"), FakeClipboard()).send_copy()
    assert keys.sent == [
        ("tap", (keycodes.KEY_LEFTCTRL, keycodes.KEY_LEFTSHIFT, keycodes.KEY_V), 1),
        ("tap", (keycodes.KEY_LEFTCTRL, keycodes.KEY_V), 1),
        ("tap", (keycodes.KEY_LEFTCTRL, keycodes.KEY_LEFTSHIFT, keycodes.KEY_C), 1),
    ]


def test_typing_pastes_and_restores_the_clipboard(monkeypatch):
    monkeypatch.setattr(linux, "PASTE_SETTLE_S", 0)
    monkeypatch.setattr(linux, "TYPE_RESTORE_DELAY_S", 0)
    keys = FakeKeyboard()
    board = FakeClipboard()
    linux.LinuxKeyboard(keys, FakeFocus("kitty"), board).type_unicode("grüße")
    for thread in threading.enumerate():
        if isinstance(thread, threading.Timer):
            thread.join(2)
    assert board.calls == ["snapshot", ("set", "grüße"), ("restore", "snap", 9)]
    assert keys.sent[0][0] == "tap"


def test_backspaces_tap_backspace():
    keys = FakeKeyboard()
    keyboard = linux.LinuxKeyboard(keys, FakeFocus(""), FakeClipboard())
    keyboard.send_backspaces(3)
    keyboard.send_backspaces(0)
    assert keys.sent == [("tap", (keycodes.KEY_BACKSPACE,), 3)]


def test_overlay_uses_the_output_kwin_reported():
    tracker = kwin.FocusTracker()
    tracker._live = True
    tracker.activated("{a}", "kitty", "", "1", "1920,0,1707,960")
    overlay = linux.LinuxOverlay(tracker)
    assert overlay.monitor_rect_for(tracker.foreground()) == (1920, 0, 3627, 960)


def test_the_focus_reports_app_and_title():
    tracker = SimpleNamespace(
        foreground=lambda: 3, info=lambda window: kwin.WindowInfo("kitty", "vim", 1)
    )
    focus = linux.LinuxFocus(tracker)
    assert focus.foreground() == 3
    assert focus.app_name(3) == "kitty"
    assert focus.title(3) == "vim"
    assert focus.is_elevated(3) is False
