"""Live Win32 tests for spells.win32.

They run on the real desktop: they create small tkinter windows, steal focus briefly,
inject keystrokes, and use the clipboard (the user's clipboard is snapshotted before and
restored after every clipboard test). Run with:

    .venv/Scripts/python.exe -m pytest tests/integration/test_win32_live.py -m integration -q
"""

import ctypes
import os
import struct
import subprocess
import sys
import threading
import time
import tkinter as tk
import uuid
from ctypes import wintypes
from pathlib import Path

import pytest

from spells.win32 import (
    CF_DIB,
    CF_UNICODETEXT,
    ENDSESSION_CLOSEAPP,
    LLKHF_INJECTED,
    MOD_ALT,
    MOD_CONTROL,
    MOD_SHIFT,
    PBT_APMRESUMEAUTOMATIC,
    VK_PROBE,
    WM_ENDSESSION,
    WM_POWERBROADCAST,
    WM_QUERYENDSESSION,
    WM_WTSSESSION_CHANGE,
    WTS_SESSION_UNLOCK,
    JobObject,
    MessageHandlers,
    MessageWindow,
    acquire_single_instance,
    bring_to_foreground,
    current_integrity_level,
    exclusion_format,
    find_message_window,
    foreground_hwnd,
    get_text,
    install_keyboard_hook,
    is_elevated_window,
    is_key_down,
    kill_timer,
    monitor_work_area_for_window,
    process_image_path,
    process_integrity_level,
    register_hotkey_probe,
    release_held_modifiers,
    release_single_instance,
    restore,
    run_message_loop,
    send_ctrl_v,
    send_key,
    sequence_number,
    set_current_thread_priority_highest,
    set_text,
    set_timer,
    signal_running_instance,
    snapshot,
    spawn_hidden,
    type_unicode,
    uninstall_keyboard_hook,
    window_pid,
    window_process_name,
    window_title,
)
from spells.win32 import clipboard as clipboard_module
from spells.win32 import process as process_module

pytestmark = pytest.mark.integration

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
VK_A = 0x41
VK_B = 0x42
VK_F9 = 0x78
VK_F11 = 0x7A
VK_LSHIFT = 0xA0
VK_SHIFT = 0x10
SWALLOW_TAG = 0x5157A110
RAISE_TAG = 0x5157A111
PROBE_TAG = 12345

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_user32.SendMessageW.argtypes = (wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
_user32.SendMessageW.restype = ctypes.c_ssize_t
_user32.RegisterHotKey.argtypes = (wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT)
_user32.RegisterHotKey.restype = wintypes.BOOL
_user32.UnregisterHotKey.argtypes = (wintypes.HWND, ctypes.c_int)
_user32.UnregisterHotKey.restype = wintypes.BOOL


def wait_until(predicate, timeout_s=3.0, pump=None):
    deadline = time.monotonic() + timeout_s
    while True:
        if pump is not None:
            pump()
        if predicate():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)


class TkWindow:
    """A small always-on-top tkinter Toplevel with an Entry and a Text widget."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.top = tk.Toplevel(root)
        self.top.title(f"spells win32 live test {os.getpid()}")
        self.top.geometry("360x200+60+60")
        self.top.attributes("-topmost", True)
        self.entry = tk.Entry(self.top, width=44)
        self.entry.pack(padx=8, pady=8)
        self.text = tk.Text(self.top, width=44, height=5)
        self.text.pack(padx=8, pady=8)
        self.top.update()
        self.hwnd = int(self.top.wm_frame(), 16)

    def title(self) -> str:
        return self.top.title()

    def pump(self, seconds: float = 0.0) -> None:
        deadline = time.monotonic() + seconds
        while True:
            self.root.update()
            if time.monotonic() >= deadline:
                return
            time.sleep(0.01)

    def focus(self, widget: tk.Widget) -> None:
        assert bring_to_foreground(self.hwnd), "could not bring the tkinter window to front"
        widget.focus_force()
        self.pump(0.15)
        assert foreground_hwnd() == self.hwnd, "tkinter window lost the foreground"
        assert self.root.focus_get() is widget

    def close(self) -> None:
        self.top.destroy()
        self.root.update()


@pytest.fixture(scope="module")
def tk_root():
    # One Tk interpreter per module: tests use Toplevel windows on it. A second tk.Tk()
    # in the same process fails to re-source Tk's ttk scripts on this machine.
    root = tk.Tk()
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def tkwin(tk_root):
    win = TkWindow(tk_root)
    yield win
    win.close()


@pytest.fixture
def preserved_clipboard():
    saved = snapshot()
    yield saved
    restore(saved, sequence_number())


def _decode(payload: bytes) -> str:
    return payload.decode("utf-16-le").split("\x00", 1)[0]


# clipboard


def test_clipboard_snapshot_set_text_restore_round_trip(preserved_clipboard):
    set_text("before spells")
    snap = snapshot()
    assert _decode(snap.formats[CF_UNICODETEXT]) == "before spells"
    assert snap.sequence == sequence_number()

    seq = set_text("spells pasted text")
    assert seq != snap.sequence
    assert get_text() == "spells pasted text"
    tagged = snapshot()
    assert tagged.formats[exclusion_format()] == b"\x00\x00\x00\x00"

    assert restore(snap, seq) is True
    assert get_text() == "before spells"
    restored = snapshot()
    assert restored.formats[exclusion_format()] == b"\x00\x00\x00\x00"
    assert _decode(restored.formats[CF_UNICODETEXT]) == "before spells"


def test_clipboard_restore_refuses_after_an_external_change(preserved_clipboard):
    set_text("before")
    snap = snapshot()
    seq = set_text("ours")
    set_text("someone else")  # simulates another app changing the clipboard
    assert sequence_number() != seq
    assert restore(snap, seq) is False
    assert get_text() == "someone else"


def test_clipboard_get_text_handles_unicode(preserved_clipboard):
    text = "héllo \U0001f600 ✓"
    set_text(text)
    assert get_text() == text


def _tiny_dib() -> bytes:
    """A 2x2 32-bit BI_RGB device-independent bitmap, the CF_DIB payload."""
    header = struct.pack(
        "<IiiHHIIiiII",
        40,  # biSize
        2,  # biWidth
        2,  # biHeight
        1,  # biPlanes
        32,  # biBitCount
        0,  # biCompression, BI_RGB
        16,  # biSizeImage
        2835,  # biXPelsPerMeter
        2835,  # biYPelsPerMeter
        0,  # biClrUsed
        0,  # biClrImportant
    )
    return header + bytes([0x20, 0x40, 0x80, 0x00] * 4)


def test_clipboard_round_trip_restores_non_text_formats(preserved_clipboard):
    # An image or a file drop must survive a dictation as well as text does: the snapshot
    # keeps raw bytes per format and the restore puts every one of them back.
    dib = _tiny_dib()
    payload = b"spells non-text payload"
    custom = clipboard_module._user32.RegisterClipboardFormatW("SpellsLiveTestFormat")
    assert custom
    with clipboard_module._Session():
        assert clipboard_module._user32.EmptyClipboard()
        clipboard_module._put_global(CF_DIB, dib)
        clipboard_module._put_global(custom, payload)

    snap = snapshot()
    # GlobalAlloc may hand out a slightly larger block, so compare the payload prefix.
    assert snap.formats[CF_DIB][: len(dib)] == dib
    assert snap.formats[custom][: len(payload)] == payload

    seq = set_text("spells replaced the image")
    assert get_text() == "spells replaced the image"
    assert CF_DIB not in snapshot().formats

    assert restore(snap, seq) is True
    back = snapshot()
    assert back.formats[CF_DIB][: len(dib)] == dib
    assert back.formats[custom][: len(payload)] == payload
    assert get_text() is None
    assert back.formats[exclusion_format()] == b"\x00\x00\x00\x00"


def test_clipboard_restore_of_an_empty_snapshot_leaves_it_empty(preserved_clipboard):
    with clipboard_module._Session():
        assert clipboard_module._user32.EmptyClipboard()
    empty = snapshot()
    assert empty.formats == {}

    seq = set_text("spells filled the clipboard")
    assert get_text() == "spells filled the clipboard"
    assert restore(empty, seq) is True
    # An empty clipboard is restored to empty, without the exclusion tag standing in for it.
    assert get_text() is None
    assert snapshot().formats == {}


# window


def test_foreground_window_process_name_and_title(tkwin):
    tkwin.focus(tkwin.entry)
    hwnd = foreground_hwnd()
    assert hwnd == tkwin.hwnd
    assert window_pid(hwnd) == os.getpid()
    assert window_process_name(hwnd).lower() == Path(sys.executable).name.lower()
    # The venv python.exe is a launcher that runs the base interpreter as a child, so
    # only the basename is stable across layouts.
    image = process_image_path(os.getpid())
    assert Path(image).name.lower() == "python.exe"
    assert Path(image).is_file()
    assert window_title(hwnd) == tkwin.title()


def test_integrity_levels(tkwin):
    assert current_integrity_level() == 0x2000
    assert process_integrity_level(os.getpid()) == 0x2000
    assert is_elevated_window(tkwin.hwnd) is False


def test_unknown_process_and_window_are_handled():
    assert process_image_path(0) == ""
    assert process_integrity_level(0) is None
    assert window_process_name(0) == ""
    assert window_pid(0) == 0
    assert is_elevated_window(0) is True  # undeterminable counts as elevated


def test_monitor_work_area(tkwin):
    left, top, right, bottom = monitor_work_area_for_window(tkwin.hwnd)
    assert right > left
    assert bottom > top


# process


def _process_alive(pid: int) -> bool:
    return process_image_path(pid) != ""


def test_job_object_kills_child_on_close(tmp_path):
    pid_file = tmp_path / "child.pid"
    # sys.executable in a venv is a launcher that runs the base interpreter as its own
    # child; that grandchild must die with the job as well.
    code = f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(60)"
    job = JobObject()
    proc = spawn_hidden(
        [sys.executable, "-c", code],
        env=None,
        stdout_path=tmp_path / "out.log",
        stderr_path=tmp_path / "err.log",
        cwd=None,
        job=job,
    )
    try:
        assert wait_until(lambda: pid_file.is_file() and pid_file.read_text().strip() != "", 10.0)
        child_pid = int(pid_file.read_text())
        assert proc.poll() is None
        assert _process_alive(child_pid)
        job.close()
        assert job.handle is None
        assert wait_until(lambda: proc.poll() is not None, 2.0), "child outlived the job"
        assert wait_until(lambda: not _process_alive(child_pid), 2.0), "grandchild outlived the job"
    finally:
        if proc.poll() is None:
            proc.kill()


def test_spawn_hidden_appends_output_and_reports_exit_code(tmp_path):
    out = tmp_path / "engine.log"
    out.write_bytes(b"existing line\n")
    code = "import sys; print('hello from child'); print('to stderr', file=sys.stderr); sys.exit(7)"
    with JobObject() as job:
        proc = spawn_hidden(
            [sys.executable, "-c", code],
            env=None,
            stdout_path=out,
            stderr_path=out,
            cwd=tmp_path,
            job=job,
        )
        assert proc.wait(timeout=30) == 7
    data = out.read_bytes()
    assert data.startswith(b"existing line\n")
    assert b"hello from child" in data
    assert b"to stderr" in data


def test_spawn_hidden_kills_the_suspended_child_when_the_job_assignment_fails(
    tmp_path, monkeypatch
):
    created: list[subprocess.Popen] = []
    real_popen = subprocess.Popen

    def recording_popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        created.append(proc)
        return proc

    monkeypatch.setattr(process_module.subprocess, "Popen", recording_popen)
    job = JobObject()
    job.close()  # assign() now raises ValueError, which is not an OSError
    with pytest.raises(ValueError):
        spawn_hidden(
            [sys.executable, "-c", "import time; time.sleep(120)"],
            env=None,
            stdout_path=tmp_path / "out.log",
            stderr_path=tmp_path / "err.log",
            cwd=None,
            job=job,
        )
    assert len(created) == 1
    proc = created[0]
    try:
        # Without the kill this child would sit suspended forever: no window, no job, and
        # no Popen object left anywhere to stop it.
        assert proc.poll() is not None, "the suspended child outlived the failed assignment"
    finally:
        if proc.poll() is None:
            proc.kill()


# input


def test_type_unicode_into_entry_with_emoji(tkwin):
    tkwin.focus(tkwin.entry)
    text = "Héllo wörld \U0001f600 ✓ ünïcode"
    type_unicode(text)
    assert wait_until(lambda: tkwin.entry.get() == text, pump=tkwin.root.update), tkwin.entry.get()


def test_type_unicode_batches_and_newlines_into_text_widget(tkwin):
    tkwin.focus(tkwin.text)
    text = ("abcdefghij" * 7) + "\nsecond line\r\nthird \U0001f600 line"
    expected = text.replace("\r\n", "\n")
    type_unicode(text)
    assert wait_until(
        lambda: tkwin.text.get("1.0", "end-1c") == expected, pump=tkwin.root.update
    ), tkwin.text.get("1.0", "end-1c")


def test_send_ctrl_v_pastes_set_text_content(tkwin, preserved_clipboard):
    set_text("pasted by spells")
    tkwin.focus(tkwin.text)
    send_ctrl_v()
    assert wait_until(
        lambda: tkwin.text.get("1.0", "end-1c") == "pasted by spells", pump=tkwin.root.update
    ), tkwin.text.get("1.0", "end-1c")


def test_release_held_modifiers_releases_a_synthetic_shift(tkwin):
    tkwin.focus(tkwin.entry)
    send_key(VK_LSHIFT, True)
    try:
        assert wait_until(lambda: is_key_down(VK_LSHIFT), pump=tkwin.root.update)
        released = release_held_modifiers()
        assert VK_LSHIFT in released
        assert VK_SHIFT in released
        assert wait_until(lambda: not is_key_down(VK_LSHIFT), pump=tkwin.root.update)
        assert release_held_modifiers() == []
    finally:
        send_key(VK_LSHIFT, False)


# hook and message loop


class HookThread:
    """Installs the hook and runs the message loop on a worker thread."""

    def __init__(self, timers=None, on_timer=None) -> None:
        self.timers = timers or {}
        self.on_timer = on_timer
        self.events = []
        self.lock = threading.Lock()
        self.ready = threading.Event()
        self.stop = threading.Event()
        self.error: Exception | None = None
        self.thread_id = 0
        self.handle = 0
        self.thread = threading.Thread(target=self._run, name="spells-test-hook", daemon=True)

    def callback(self, event) -> bool:
        with self.lock:
            self.events.append((event, threading.get_ident()))
        if event.extra_info == RAISE_TAG:
            raise RuntimeError("callback failure is contained")
        if event.vk == VK_PROBE and event.extra_info == PROBE_TAG:
            return True
        return event.vk == VK_A and event.extra_info == SWALLOW_TAG

    def _run(self) -> None:
        try:
            set_current_thread_priority_highest()
            self.thread_id = threading.get_ident()
            self.handle = install_keyboard_hook(self.callback)
            self.ready.set()
            run_message_loop(self.stop, self.timers, self.on_timer, poll_ms=20)
        except Exception as exc:  # noqa: BLE001 - surfaced to the test thread
            self.error = exc
            self.ready.set()
        finally:
            if self.handle:
                uninstall_keyboard_hook(self.handle)

    def __enter__(self):
        self.thread.start()
        try:
            assert self.ready.wait(5), "hook thread did not start"
            assert self.error is None, self.error
        except BaseException:
            # The with body never runs when __enter__ raises, so __exit__ never runs
            # either: stop the thread here or the low-level hook stays installed for the
            # rest of the session and swallows the user's keys.
            self.stop.set()
            self.thread.join(5)
            raise
        return self

    def __exit__(self, *exc) -> None:
        self.stop.set()
        self.thread.join(5)
        assert not self.thread.is_alive(), "hook thread did not stop"

    def seen(self, vk: int, extra_info: int | None = None):
        with self.lock:
            return [
                event
                for event, _ in self.events
                if event.vk == vk and (extra_info is None or event.extra_info == extra_info)
            ]


def test_hook_sees_injected_probe_and_swallows_tagged_keys(tkwin):
    with HookThread() as hooked:
        tkwin.focus(tkwin.entry)
        send_key(VK_PROBE, True, PROBE_TAG)
        send_key(VK_PROBE, False, PROBE_TAG)
        assert wait_until(lambda: len(hooked.seen(VK_PROBE, PROBE_TAG)) >= 2, pump=tkwin.pump)
        down, up = hooked.seen(VK_PROBE, PROBE_TAG)[:2]
        assert down.vk == 0xE8 and down.keydown is True
        assert down.flags & LLKHF_INJECTED
        assert down.injected is True
        assert down.extra_info == 12345
        assert up.keydown is False and up.extra_info == 12345
        with hooked.lock:
            assert all(tid == hooked.thread_id for _, tid in hooked.events)
        tkwin.pump(0.2)
        assert tkwin.entry.get() == ""

        # A swallowed key never reaches the focused Entry; an untagged one does.
        send_key(VK_A, True, SWALLOW_TAG)
        send_key(VK_A, False, SWALLOW_TAG)
        assert wait_until(lambda: len(hooked.seen(VK_A, SWALLOW_TAG)) >= 2, pump=tkwin.pump)
        tkwin.pump(0.3)
        assert tkwin.entry.get() == ""
        send_key(VK_A, True, 0)
        send_key(VK_A, False, 0)
        assert wait_until(lambda: tkwin.entry.get() == "a", pump=tkwin.pump), tkwin.entry.get()

        # An exception inside the callback is logged and the key is passed on.
        send_key(VK_B, True, RAISE_TAG)
        send_key(VK_B, False, RAISE_TAG)
        assert wait_until(lambda: tkwin.entry.get() == "ab", pump=tkwin.pump), tkwin.entry.get()


def test_message_loop_timers_one_shot_and_prompt_stop():
    ticks = []
    done = threading.Event()

    def on_timer(timer_id: int) -> None:
        ticks.append((timer_id, threading.get_ident()))
        if timer_id == 1 and sum(1 for tid, _ in ticks if tid == 1) == 3:
            kill_timer(1)
            set_timer(2, 20)  # one-shot, armed from the loop thread
        elif timer_id == 2:
            kill_timer(2)
            done.set()

    stop = threading.Event()
    thread = threading.Thread(
        target=run_message_loop, args=(stop, {1: 30}, on_timer), kwargs={"poll_ms": 20}
    )
    thread.start()
    try:
        assert done.wait(5), ticks
        time.sleep(0.15)  # a leftover timer would tick again here
        started = time.monotonic()
    finally:
        stop.set()
        thread.join(5)
    assert not thread.is_alive()
    assert time.monotonic() - started < 1.0
    assert [tid for tid, _ in ticks] == [1, 1, 1, 2]
    assert all(ident == thread.ident for _, ident in ticks)


def test_probe_sent_from_the_loop_thread_reaches_the_callback():
    seen = threading.Event()
    hooked = HookThread()

    def on_timer(timer_id: int) -> None:
        kill_timer(timer_id)
        send_key(VK_PROBE, True, PROBE_TAG)
        send_key(VK_PROBE, False, PROBE_TAG)
        if len(hooked.seen(VK_PROBE, PROBE_TAG)) >= 2:
            seen.set()

    hooked.timers = {7: 20}
    hooked.on_timer = on_timer
    with hooked:
        assert wait_until(lambda: seen.is_set() or len(hooked.seen(VK_PROBE, PROBE_TAG)) >= 2)
    events = hooked.seen(VK_PROBE, PROBE_TAG)
    assert [event.keydown for event in events[:2]] == [True, False]
    assert all(event.injected for event in events)


# hotkey probe


def test_register_hotkey_probe_ctrl_alt_f9_is_free_and_not_left_registered():
    # F9, not F12: RegisterHotKey reserves F12 for the debugger, so a probe there answers
    # about the debugger rather than about the chord.
    modifiers = MOD_CONTROL | MOD_ALT
    first = register_hotkey_probe(modifiers, VK_F9)
    if not first and not _user32.RegisterHotKey(None, 0x0BEE, modifiers, VK_F9):
        pytest.skip("Ctrl+Alt+F9 is owned by another application on this machine")
    _user32.UnregisterHotKey(None, 0x0BEE)
    assert first is True
    assert register_hotkey_probe(modifiers, VK_F9) is True
    # Registering it ourselves succeeds only when the probe left nothing behind.
    assert _user32.RegisterHotKey(None, 0x0BEE, modifiers, VK_F9)
    _user32.UnregisterHotKey(None, 0x0BEE)


def test_register_hotkey_probe_reports_a_taken_chord():
    modifiers = MOD_CONTROL | MOD_ALT | MOD_SHIFT
    assert _user32.RegisterHotKey(None, 0x0BEF, modifiers, VK_F11), ctypes.get_last_error()
    try:
        assert register_hotkey_probe(modifiers, VK_F11) is False
    finally:
        _user32.UnregisterHotKey(None, 0x0BEF)
    assert register_hotkey_probe(modifiers, VK_F11) is True


# message window and single instance


class WindowThread:
    def __init__(self, class_name: str) -> None:
        self.class_name = class_name
        self.copydata: list[str] = []
        self.queries: list[int] = []
        self.query_result = True
        self.ended: list[int] = []
        self.resumed = 0
        self.unlocked = 0
        self.ready = threading.Event()
        self.stop = threading.Event()
        self.window: MessageWindow | None = None
        self.error: Exception | None = None
        self.thread = threading.Thread(target=self._run, name="spells-test-msgwnd", daemon=True)

    def _on_query(self, flags: int) -> bool:
        self.queries.append(flags)
        return self.query_result

    def _on_resume(self) -> None:
        self.resumed += 1

    def _on_unlock(self) -> None:
        self.unlocked += 1

    def _run(self) -> None:
        try:
            handlers = MessageHandlers(
                on_copydata=self.copydata.append,
                on_query_end_session=self._on_query,
                on_end_session=self.ended.append,
                on_resume=self._on_resume,
                on_session_unlock=self._on_unlock,
            )
            self.window = MessageWindow(self.class_name, handlers)
            self.ready.set()
            self.window.run(self.stop)
        except Exception as exc:  # noqa: BLE001 - surfaced to the test thread
            self.error = exc
            self.ready.set()
        finally:
            if self.window is not None:
                self.window.destroy()

    def __enter__(self):
        self.thread.start()
        assert self.ready.wait(5), "message window thread did not start"
        assert self.error is None, self.error
        return self

    def __exit__(self, *exc) -> None:
        self.stop.set()
        self.thread.join(5)
        assert not self.thread.is_alive()


def test_message_window_receives_copydata_from_signal_running_instance():
    class_name = f"SpellsTestMsgWnd_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    with WindowThread(class_name) as wnd:
        hwnd = wnd.window.hwnd
        assert hwnd
        assert find_message_window(class_name) == hwnd
        assert find_message_window(class_name + "_missing") == 0
        assert signal_running_instance(class_name + "_missing", "quit") is False

        assert signal_running_instance(class_name, "open-settings") is True
        assert signal_running_instance(class_name, "quit") is True
        assert wait_until(lambda: wnd.copydata == ["open-settings", "quit"]), wnd.copydata

        assert _user32.SendMessageW(hwnd, WM_QUERYENDSESSION, 0, ENDSESSION_CLOSEAPP) == 1
        _user32.SendMessageW(hwnd, WM_ENDSESSION, 1, ENDSESSION_CLOSEAPP)
        assert _user32.SendMessageW(hwnd, WM_POWERBROADCAST, PBT_APMRESUMEAUTOMATIC, 0) == 1
        _user32.SendMessageW(hwnd, WM_WTSSESSION_CHANGE, WTS_SESSION_UNLOCK, 0)
        assert wnd.queries == [ENDSESSION_CLOSEAPP]
        assert wnd.ended == [ENDSESSION_CLOSEAPP]
        assert wnd.resumed == 1
        assert wnd.unlocked == 1
    assert find_message_window(class_name) == 0


def test_message_window_passes_a_refused_query_end_session_through():
    class_name = f"SpellsTestMsgWnd_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    with WindowThread(class_name) as wnd:
        hwnd = wnd.window.hwnd
        wnd.query_result = False
        assert _user32.SendMessageW(hwnd, WM_QUERYENDSESSION, 0, ENDSESSION_CLOSEAPP) == 0
        assert wnd.queries == [ENDSESSION_CLOSEAPP]
        # Refusing the shutdown is an answer, not an exit: the window is still there and
        # still answering, and no WM_ENDSESSION was handled.
        assert find_message_window(class_name) == hwnd
        assert wnd.ended == []
        wnd.query_result = True
        assert _user32.SendMessageW(hwnd, WM_QUERYENDSESSION, 0, ENDSESSION_CLOSEAPP) == 1
        assert signal_running_instance(class_name, "still alive") is True
        assert wait_until(lambda: wnd.copydata == ["still alive"]), wnd.copydata


def test_message_window_destroy_from_another_thread():
    class_name = f"SpellsTestMsgWnd_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    with WindowThread(class_name) as wnd:
        hwnd = wnd.window.hwnd
        wnd.window.destroy()  # forwarded as WM_CLOSE to the owning thread
        assert wait_until(lambda: wnd.window.hwnd == 0)
        assert find_message_window(class_name) == 0
        assert hwnd


def test_single_instance_mutex_true_once_then_false_in_a_subprocess():
    name = f"SpellsTestMutex_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    code = (
        "import sys; from spells.win32 import acquire_single_instance; "
        "sys.exit(3 if acquire_single_instance(sys.argv[1]) else 4)"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC_DIR)

    def probe() -> int:
        result = subprocess.run(
            [sys.executable, "-c", code, name], env=env, timeout=60, check=False
        )
        return result.returncode

    assert acquire_single_instance(name) is True
    try:
        assert acquire_single_instance(name) is False
        assert probe() == 4
    finally:
        release_single_instance(name)
    assert probe() == 3
