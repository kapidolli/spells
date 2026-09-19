"""Live delivery tests for spells.context and spells.inject (spec 20.4).

They run on the real desktop: small tkinter windows are created and focused, real
keystrokes are injected and the real clipboard is used. The user's clipboard is
snapshotted before and restored after every test, and the window that had the foreground
when the module started gets it back at the end, including when a test fails. Run with:

    .venv/Scripts/python.exe -m pytest tests/integration/test_inject_live.py -m integration -q

The sleeper passed to deliver() pumps the tkinter event loop instead of blocking, because
here the target window lives in this same process: a plain time.sleep would hold the
Ctrl+V in the queue until after the clipboard had been restored. In the app the target is
another process with its own message loop, so the default time.sleep is right.
"""

import logging
import os
import sys
import threading
import time
import tkinter as tk
import uuid
from pathlib import Path

import pytest

from spells.context import capture
from spells.inject import deliver, ends_with_whitespace
from spells.models import DeliveryMethod, DeliveryOutcome
from spells.win32 import (
    bring_to_foreground,
    foreground_hwnd,
    get_text,
    is_key_down,
    restore,
    send_key,
    sequence_number,
    set_text,
    snapshot,
    window_process_name,
)

pytestmark = pytest.mark.integration

# Umlaut, sharp s and a non-BMP emoji in one string (decision V2-12, spec 20.4).
TEXT = "Grüße aus Spells \U0001f600 ✓"
# Spec 12: post-process plus inject are budgeted at 100 ms; inject alone stays well under.
BUDGET_MS = 100.0
VK_LSHIFT = 0xA0
VK_SHIFT = 0x10
# No window can be brought to the front while the workstation is locked or a screen
# saver runs, so every test here fails at the first focus call. Say so in the message.
NEEDS_UNLOCKED = (
    "could not bring the tkinter window to front; these tests need an unlocked, "
    "interactive desktop session"
)
LOCK_SCREEN_PROCESSES = {"lockapp.exe", "logonui.exe"}

logger = logging.getLogger(__name__)


def locked_desktop_reason():
    """Why this desktop cannot be driven, or None when these tests can run.

    A locked workstation keeps LockApp.exe in the foreground, refuses every attempt to
    bring a window to the front and fails OpenClipboard with error 5. Without this check
    that shows up as an error in every fixture here instead of one clear message.
    """
    process = window_process_name(foreground_hwnd()).lower()
    if process in LOCK_SCREEN_PROCESSES:
        return f"{process} holds the foreground, so the workstation is locked"
    try:
        snapshot()
    except OSError as exc:
        return f"the clipboard cannot be opened ({exc})"
    return None


_LOCKED = locked_desktop_reason()
if _LOCKED:
    pytest.skip(
        f"live delivery tests need an unlocked, interactive desktop session: {_LOCKED}",
        allow_module_level=True,
    )


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

    def __init__(self, root, offset=60):
        self.root = root
        self.top = tk.Toplevel(root)
        self.top.title(f"spells inject live test {os.getpid()} {offset}")
        self.top.geometry(f"360x200+{offset}+{offset}")
        self.top.attributes("-topmost", True)
        self.entry = tk.Entry(self.top, width=44)
        self.entry.pack(padx=8, pady=8)
        self.text = tk.Text(self.top, width=44, height=5)
        self.text.pack(padx=8, pady=8)
        self.top.update()
        self.hwnd = int(self.top.wm_frame(), 16)

    def title(self):
        return self.top.title()

    def pump(self, seconds=0.0):
        """Run the tkinter event loop for `seconds`; also used as deliver()'s sleeper."""
        deadline = time.monotonic() + seconds
        while True:
            self.root.update()
            if time.monotonic() >= deadline:
                return
            time.sleep(0.01)

    def focus(self, widget):
        assert bring_to_foreground(self.hwnd), NEEDS_UNLOCKED
        widget.focus_force()
        self.pump(0.15)
        assert foreground_hwnd() == self.hwnd, "tkinter window lost the foreground"
        assert self.root.focus_get() is widget

    def close(self):
        self.top.destroy()
        self.root.update()


@pytest.fixture(scope="module")
def tk_root():
    # One Tk interpreter per module: a second tk.Tk() in the same process fails to
    # re-source Tk's scripts on this machine.
    root = tk.Tk()
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture(scope="module", autouse=True)
def restored_focus():
    """Give the foreground back to whatever had it, even when a test fails."""
    before = foreground_hwnd()
    yield before
    if before:
        bring_to_foreground(before)


@pytest.fixture
def tkwin(tk_root):
    win = TkWindow(tk_root)
    yield win
    win.close()


@pytest.fixture
def preserved_clipboard():
    """Snapshot the user's clipboard before the test and force it back afterwards."""
    saved = snapshot()
    yield saved
    try:
        if not restore(saved, sequence_number()):
            logger.error("clipboard restore was refused; test text may still be on it")
    except OSError:
        logger.exception("clipboard could not be restored; test text may still be on it")


def report_ms(name, report):
    print(f"\n[delivery] {name}: {report.elapsed_ms:.1f} ms")


# context


def test_capture_describes_the_focused_window(tkwin):
    tkwin.focus(tkwin.entry)
    ctx = capture()
    assert ctx.hwnd == tkwin.hwnd
    assert ctx.process.lower() == Path(sys.executable).name.lower()
    assert ctx.title == tkwin.title()
    assert ctx.captured_at > 0


# paste


def test_paste_delivers_exact_text_into_a_text_widget(tkwin, preserved_clipboard):
    tkwin.focus(tkwin.text)
    ctx = capture()
    report = deliver(TEXT, ctx, DeliveryMethod.PASTE, sleeper=tkwin.pump)
    report_ms("paste", report)
    assert report.outcome is DeliveryOutcome.PASTED
    assert report.clipboard_restored is True
    assert report.elapsed_ms is not None and report.elapsed_ms < BUDGET_MS
    assert wait_until(
        lambda: tkwin.text.get("1.0", "end-1c") == TEXT, pump=tkwin.root.update
    ), tkwin.text.get("1.0", "end-1c")


def test_paste_restores_the_clipboard_marker(tkwin, preserved_clipboard):
    marker = f"spells marker {uuid.uuid4()}"
    set_text(marker)
    tkwin.focus(tkwin.text)
    ctx = capture()
    report = deliver(TEXT, ctx, DeliveryMethod.PASTE, sleeper=tkwin.pump)
    assert report.outcome is DeliveryOutcome.PASTED
    assert report.clipboard_restored is True
    assert report.result.detail == "clipboard restored"
    assert get_text() == marker


def test_paste_releases_a_held_modifier_first(tkwin, preserved_clipboard):
    tkwin.focus(tkwin.text)
    ctx = capture()
    send_key(VK_LSHIFT, True)
    try:
        assert wait_until(lambda: is_key_down(VK_LSHIFT), pump=tkwin.root.update)
        report = deliver(TEXT, ctx, DeliveryMethod.PASTE, sleeper=tkwin.pump)
        assert report.outcome is DeliveryOutcome.PASTED
        assert not is_key_down(VK_LSHIFT)
        assert not is_key_down(VK_SHIFT)
        assert wait_until(
            lambda: tkwin.text.get("1.0", "end-1c") == TEXT, pump=tkwin.root.update
        ), tkwin.text.get("1.0", "end-1c")
    finally:
        send_key(VK_LSHIFT, False)


def test_restore_is_refused_when_the_clipboard_changes_during_the_delay(
    tkwin, preserved_clipboard
):
    intruder = f"spells intruder {uuid.uuid4()}"

    def change_clipboard():
        time.sleep(0.12)
        set_text(intruder)

    tkwin.focus(tkwin.text)
    ctx = capture()
    thread = threading.Thread(target=change_clipboard, name="clipboard-intruder")
    thread.start()
    try:
        report = deliver(
            TEXT, ctx, DeliveryMethod.PASTE, restore_delay_s=0.5, sleeper=tkwin.pump
        )
    finally:
        thread.join(timeout=5)
    assert report.outcome is DeliveryOutcome.PASTED
    assert report.clipboard_restored is False
    assert "not restored" in report.result.detail
    assert get_text() == intruder


# type


def test_type_delivers_into_an_entry_including_an_emoji(tkwin, preserved_clipboard):
    tkwin.focus(tkwin.entry)
    ctx = capture()
    report = deliver(TEXT, ctx, DeliveryMethod.TYPE, sleeper=tkwin.pump)
    report_ms("type", report)
    assert report.outcome is DeliveryOutcome.TYPED
    assert report.clipboard_restored is None
    assert report.elapsed_ms is not None and report.elapsed_ms < BUDGET_MS
    assert wait_until(
        lambda: tkwin.entry.get() == TEXT, pump=tkwin.root.update
    ), tkwin.entry.get()


# fallbacks (spec 11 step 1)


def test_focus_moved_after_capture_copies_instead_of_pasting(tk_root, preserved_clipboard):
    first = TkWindow(tk_root, offset=60)
    second = None
    try:
        first.focus(first.text)
        ctx = capture()
        assert ctx.hwnd == first.hwnd
        # The second window is created only now, after the capture: two always-on-top
        # windows alive in one Tk interpreter fight over the foreground, and a
        # focus_force on the younger one then blocks the event loop for seconds.
        second = TkWindow(tk_root, offset=240)
        assert bring_to_foreground(second.hwnd), NEEDS_UNLOCKED
        assert foreground_hwnd() == second.hwnd
        report = deliver(TEXT, ctx, DeliveryMethod.PASTE, sleeper=second.pump)
        assert report.outcome is DeliveryOutcome.COPIED_FOCUS_CHANGED
        assert str(ctx.hwnd) in report.result.detail
        assert str(second.hwnd) in report.result.detail
        assert get_text() == TEXT
        # No pump here: with both windows alive, one tkinter update can block for ten
        # seconds or more. Nothing was injected, so there is nothing to wait for either.
        assert second.text.get("1.0", "end-1c") == ""
        assert first.text.get("1.0", "end-1c") == ""
    finally:
        if second is not None:
            second.close()
        first.close()


@pytest.mark.skip(
    reason="needs an elevated window; this account is a standard user and any elevation "
    "would need a UAC prompt. Manual check per spec 20.4: dictate into a normal window, "
    "focus an elevated terminal while it processes, expect outcome copied_elevated and "
    "the notification Copied. Press Ctrl+V to paste."
)
def test_focus_moved_to_an_elevated_window_copies_with_copied_elevated():
    raise AssertionError("manual test, see the skip reason")


# helper


def test_ends_with_whitespace_matches_what_was_delivered(tkwin, preserved_clipboard):
    tkwin.focus(tkwin.entry)
    ctx = capture()
    deliver("spells ", ctx, DeliveryMethod.TYPE, sleeper=tkwin.pump)
    assert wait_until(lambda: tkwin.entry.get() == "spells ", pump=tkwin.root.update)
    assert ends_with_whitespace("spells ") is True
