"""Pure-logic tests for spells.win32: no Win32 call is made here.

Importing the package binds ctypes prototypes but calls nothing; every assertion below
runs against plain Python data (batching, format filters, integrity comparison, struct to
KeyEvent conversion, struct layouts).
"""

import contextlib
import ctypes
import logging
from types import SimpleNamespace

import pytest

from spells import hotkey, vk
from spells.win32 import clipboard, hook, msgwindow, process, window
from spells.win32 import input as win_input

IS_64BIT = ctypes.sizeof(ctypes.c_void_p) == 8


# split_batches


def test_split_batches_keeps_surrogate_pair_together_at_the_boundary():
    text = "a" * 49 + "\U0001f600" + "b" * 10
    batches = win_input.split_batches(text, 50)
    assert batches[0] == "a" * 49 + "\U0001f600"
    assert len(batches[0]) == 50
    assert batches[1] == "b" * 10
    assert "".join(batches) == text


def test_split_batches_default_size_is_50_code_points():
    batches = win_input.split_batches("x" * 120)
    assert [len(batch) for batch in batches] == [50, 50, 20]


def test_split_batches_normalizes_crlf_and_lone_cr_to_one_newline():
    assert win_input.split_batches("a\r\nb\nc\rd") == ["a\nb\nc\nd"]


def test_split_batches_crlf_at_boundary_counts_as_one_code_point():
    text = "a" * 49 + "\r\n" + "b"
    assert win_input.split_batches(text, 50) == ["a" * 49 + "\n", "b"]


def test_split_batches_empty_text_gives_no_batches():
    assert win_input.split_batches("") == []


def test_split_batches_rejects_batch_size_below_one():
    with pytest.raises(ValueError):
        win_input.split_batches("x", 0)


# plan_batch


def test_plan_batch_newline_produces_enter_markers():
    keys = win_input.plan_batch("a\nb")
    assert [(k.keydown, k.unit, k.vk) for k in keys] == [
        (True, ord("a"), 0),
        (False, ord("a"), 0),
        (True, 0, win_input.VK_RETURN),
        (False, 0, win_input.VK_RETURN),
        (True, ord("b"), 0),
        (False, ord("b"), 0),
    ]
    assert [k.is_enter for k in keys] == [False, False, True, True, False, False]


def test_plan_batch_non_bmp_character_is_its_surrogate_pair_in_order():
    keys = win_input.plan_batch("\U0001f600")
    assert [(k.keydown, k.unit) for k in keys] == [
        (True, 0xD83D),
        (False, 0xD83D),
        (True, 0xDE00),
        (False, 0xDE00),
    ]
    assert all(k.vk == 0 for k in keys)


def test_plan_batch_bmp_characters_are_single_units():
    keys = win_input.plan_batch("é✓")
    assert [(k.keydown, k.unit) for k in keys] == [
        (True, 0x00E9),
        (False, 0x00E9),
        (True, 0x2713),
        (False, 0x2713),
    ]


def test_modifier_vks_cover_generic_and_side_specific_keys():
    assert win_input.MODIFIER_VKS == (
        0x10, 0x11, 0x12, 0x5B, 0x5C, 0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5,
    )  # fmt: skip


# should_snapshot_format


@pytest.mark.parametrize(
    "fmt",
    [
        clipboard.CF_BITMAP,
        clipboard.CF_PALETTE,
        clipboard.CF_ENHMETAFILE,
        clipboard.CF_METAFILEPICT,
        clipboard.CF_DSPBITMAP,
        clipboard.CF_DSPENHMETAFILE,
        clipboard.CF_DSPMETAFILEPICT,
        clipboard.CF_OWNERDISPLAY,
        clipboard.CF_DSPTEXT,
        clipboard.CF_PRIVATEFIRST,
        clipboard.CF_PRIVATELAST,
        clipboard.CF_GDIOBJFIRST,
        clipboard.CF_GDIOBJLAST,
        0,
    ],
)
def test_should_snapshot_format_rejects_gdi_display_private_and_invalid(fmt):
    assert clipboard.should_snapshot_format(fmt) is False


@pytest.mark.parametrize(
    "fmt",
    [
        clipboard.CF_TEXT,
        clipboard.CF_OEMTEXT,
        clipboard.CF_UNICODETEXT,
        clipboard.CF_LOCALE,
        clipboard.CF_DIB,
        clipboard.CF_DIBV5,
        clipboard.CF_HDROP,
        0xC000,  # first registered format id
        0xC0A5,  # a typical registered format such as HTML Format
        0xFFFF,
    ],
)
def test_should_snapshot_format_accepts_global_memory_formats(fmt):
    assert clipboard.should_snapshot_format(fmt) is True


def test_gdi_handle_formats_constant_matches_decision_v2_4():
    assert clipboard.GDI_HANDLE_FORMATS == frozenset({2, 9, 14, 3, 0x82, 0x8E, 0x83})
    assert clipboard.EXCLUDE_FORMAT_NAME == "ExcludeClipboardContentFromMonitorProcessing"


def test_clipboard_snapshot_defaults():
    snap = clipboard.ClipboardSnapshot()
    assert snap.formats == {}
    assert snap.sequence == 0


# integrity comparison


@pytest.mark.parametrize(
    ("level", "own", "expected"),
    [
        (None, 0x2000, True),  # access denied counts as higher
        (0x3000, 0x2000, True),  # high above medium
        (0x4000, 0x2000, True),  # system above medium
        (0x2000, 0x2000, False),  # same level
        (0x1000, 0x2000, False),  # low below medium
        (0x2000, 0x3000, False),  # medium seen from an elevated process
    ],
)
def test_is_higher_integrity(level, own, expected):
    assert window.is_higher_integrity(level, own) is expected


def test_integrity_rid_constants():
    assert window.SECURITY_MANDATORY_MEDIUM_RID == 0x2000
    assert window.SECURITY_MANDATORY_HIGH_RID == 0x3000


# KeyEvent from a fake KBDLLHOOKSTRUCT


def _fake_struct(**overrides):
    values = {
        "vkCode": hook.VK_PROBE,
        "scanCode": 0,
        "flags": hook.LLKHF_INJECTED,
        "time": 4321,
        "dwExtraInfo": 12345,
    }
    values.update(overrides)
    return hook.KBDLLHOOKSTRUCT(**values)


def test_key_event_from_struct_copies_every_field_for_keydown():
    event = hook.key_event_from_struct(_fake_struct(), hook.WM_KEYDOWN)
    assert event == hook.KeyEvent(
        vk=0xE8, scan=0, flags=0x10, extra_info=12345, keydown=True, time_ms=4321
    )
    assert event.injected is True


@pytest.mark.parametrize(
    ("wparam", "keydown"),
    [
        (hook.WM_KEYDOWN, True),
        (hook.WM_SYSKEYDOWN, True),
        (hook.WM_KEYUP, False),
        (hook.WM_SYSKEYUP, False),
    ],
)
def test_key_event_keydown_follows_the_message_id(wparam, keydown):
    event = hook.key_event_from_struct(_fake_struct(vkCode=0x41, scanCode=0x1E), wparam)
    assert event.keydown is keydown
    assert (event.vk, event.scan) == (0x41, 0x1E)


def test_key_event_not_injected_without_the_flag():
    event = hook.key_event_from_struct(_fake_struct(flags=0, dwExtraInfo=0), hook.WM_KEYDOWN)
    assert event.injected is False
    assert event.extra_info == 0


def test_key_event_is_frozen():
    event = hook.key_event_from_struct(_fake_struct(), hook.WM_KEYUP)
    with pytest.raises(AttributeError):
        event.vk = 1  # type: ignore[misc]


def test_hook_constants():
    assert hook.LLKHF_INJECTED == 0x10
    assert hook.VK_PROBE == 0xE8
    assert (hook.WM_KEYDOWN, hook.WM_KEYUP, hook.WM_SYSKEYDOWN, hook.WM_SYSKEYUP) == (
        0x0100,
        0x0101,
        0x0104,
        0x0105,
    )


def test_message_window_constants():
    assert msgwindow.WM_COPYDATA == 0x004A
    assert msgwindow.WM_QUERYENDSESSION == 0x0011
    assert msgwindow.WM_ENDSESSION == 0x0016
    assert msgwindow.ENDSESSION_CLOSEAPP == 0x00000001
    assert msgwindow.WM_POWERBROADCAST == 0x0218
    assert msgwindow.PBT_APMRESUMEAUTOMATIC == 0x0012
    assert msgwindow.WM_WTSSESSION_CHANGE == 0x02B1
    assert msgwindow.WTS_SESSION_UNLOCK == 0x8


def test_process_constants():
    assert process.STATUS_DLL_NOT_FOUND == 0xC0000135
    assert process.CREATE_NO_WINDOW == 0x08000000


# struct layouts (sizes as the Windows SDK defines them on x64)


@pytest.mark.skipif(not IS_64BIT, reason="layout sizes are asserted for 64-bit builds")
def test_struct_sizes_match_the_sdk():
    assert ctypes.sizeof(hook.INPUT) == 40
    assert ctypes.sizeof(hook.KEYBDINPUT) == 24
    assert ctypes.sizeof(hook.KBDLLHOOKSTRUCT) == 24
    assert ctypes.sizeof(process.JOBOBJECT_EXTENDED_LIMIT_INFORMATION) == 144
    assert ctypes.sizeof(msgwindow.COPYDATASTRUCT) == 24
    assert ctypes.sizeof(msgwindow.WNDCLASSEXW) == 80
    assert ctypes.sizeof(window.MONITORINFO) == 40


# spawn_hidden cleanup when the job assignment fails


class _FakeProc:
    """Stands in for subprocess.Popen: records kill() and wait()."""

    def __init__(self, pid: int = 4242) -> None:
        self.pid = pid
        self.killed = 0
        self.waited = 0

    def kill(self) -> None:
        self.killed += 1

    def wait(self, timeout: float | None = None) -> int:
        self.waited += 1
        return 1


class _FakeJob:
    """A job whose assign() raises whatever the test asks for."""

    def __init__(self, error: BaseException | None) -> None:
        self.error = error
        self.assigned: list[int] = []

    def assign(self, pid: int) -> None:
        self.assigned.append(pid)
        if self.error is not None:
            raise self.error


def _patched_spawn(monkeypatch, tmp_path, job, resume_error=None):
    proc = _FakeProc()
    calls: dict[str, object] = {}

    def fake_popen(args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return proc

    def fake_resume(pid: int) -> int:
        calls["resumed"] = pid
        if resume_error is not None:
            raise resume_error
        return 1

    monkeypatch.setattr(process.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(process, "resume_process", fake_resume)

    def spawn():
        return process.spawn_hidden(
            ["child.exe"],
            stdout_path=tmp_path / "out.log",
            stderr_path=tmp_path / "err.log",
            job=job,
        )

    return proc, calls, spawn


def test_spawn_hidden_kills_the_child_when_the_assignment_raises_value_error(
    monkeypatch, tmp_path
):
    # JobObject.assign raises ValueError on a closed job; only OSError used to be caught,
    # so the suspended child was left behind: invisible, running and owned by nobody.
    job = _FakeJob(ValueError("job object is closed"))
    proc, calls, spawn = _patched_spawn(monkeypatch, tmp_path, job)
    with pytest.raises(ValueError):
        spawn()
    assert proc.killed == 1
    assert proc.waited == 1
    assert "resumed" not in calls


def test_spawn_hidden_kills_the_child_when_the_assignment_raises_oserror(monkeypatch, tmp_path):
    job = _FakeJob(OSError(5, "access denied"))
    proc, _calls, spawn = _patched_spawn(monkeypatch, tmp_path, job)
    with pytest.raises(OSError):
        spawn()
    assert proc.killed == 1


def test_spawn_hidden_kills_the_child_when_the_resume_fails(monkeypatch, tmp_path):
    job = _FakeJob(None)
    proc, _calls, spawn = _patched_spawn(
        monkeypatch, tmp_path, job, resume_error=OSError("no thread could be resumed")
    )
    with pytest.raises(OSError):
        spawn()
    assert proc.killed == 1


def test_spawn_hidden_creates_the_child_suspended_only_with_a_job(monkeypatch, tmp_path):
    job = _FakeJob(None)
    proc, calls, spawn = _patched_spawn(monkeypatch, tmp_path, job)
    assert spawn() is proc
    flags = calls["kwargs"]["creationflags"]
    assert flags & process.CREATE_SUSPENDED
    assert flags & process.CREATE_NO_WINDOW
    assert proc.killed == 0
    assert calls["resumed"] == proc.pid

    without_job = _FakeProc()
    monkeypatch.setattr(process.subprocess, "Popen", lambda args, **kwargs: without_job)
    result = process.spawn_hidden(
        ["child.exe"], stdout_path=tmp_path / "o.log", stderr_path=tmp_path / "e.log"
    )
    assert result is without_job


def test_spawn_hidden_pins_the_child_after_the_resume(monkeypatch, tmp_path):
    job = _FakeJob(None)
    proc, _calls, _spawn = _patched_spawn(monkeypatch, tmp_path, job)
    order: list[str] = []
    real_resume = process.resume_process

    def resume(pid: int) -> int:
        order.append("resume")
        return real_resume(pid)

    def pin(pid: int, mask: int) -> None:
        order.append(f"pin {pid} {mask:x}")

    monkeypatch.setattr(process, "resume_process", resume)
    monkeypatch.setattr(process, "set_process_affinity", pin)
    result = process.spawn_hidden(
        ["child.exe"],
        stdout_path=tmp_path / "out.log",
        stderr_path=tmp_path / "err.log",
        job=job,
        affinity_mask=0x5555,
    )
    assert result is proc
    assert order == ["resume", f"pin {proc.pid} 5555"]


def test_spawn_hidden_keeps_the_child_when_pinning_fails(monkeypatch, tmp_path, caplog):
    job = _FakeJob(None)
    proc, _calls, _spawn = _patched_spawn(monkeypatch, tmp_path, job)

    def pin(pid: int, mask: int) -> None:
        raise OSError(87, "the parameter is incorrect")

    monkeypatch.setattr(process, "set_process_affinity", pin)
    with caplog.at_level(logging.WARNING):
        result = process.spawn_hidden(
            ["child.exe"],
            stdout_path=tmp_path / "out.log",
            stderr_path=tmp_path / "err.log",
            job=job,
            affinity_mask=0x3,
        )
    assert result is proc
    assert proc.killed == 0
    assert "could not pin" in caplog.text


def test_spawn_hidden_without_a_mask_never_pins(monkeypatch, tmp_path):
    job = _FakeJob(None)
    proc, _calls, spawn = _patched_spawn(monkeypatch, tmp_path, job)
    pinned: list[int] = []
    monkeypatch.setattr(process, "set_process_affinity", lambda pid, mask: pinned.append(mask))
    assert spawn() is proc
    assert pinned == []


def test_an_empty_affinity_mask_is_refused():
    with pytest.raises(ValueError):
        process.set_process_affinity(1, 0)


# session-notification registration retry


class _FakeSessionApi:
    """The four Win32 calls SessionNotificationRetry drives, recorded."""

    def __init__(self, results: list[int | None]) -> None:
        self.results = list(results)
        self.registered: list[int] = []
        self.unregistered: list[int] = []
        self.armed: list[tuple[int, int]] = []
        self.disarmed: list[int] = []

    def register(self, hwnd: int) -> int | None:
        result = self.results.pop(0) if self.results else None
        if result is None:
            self.registered.append(hwnd)
        return result

    def unregister(self, hwnd: int) -> None:
        self.unregistered.append(hwnd)

    def arm(self, timer_id: int, interval_ms: int) -> None:
        self.armed.append((timer_id, interval_ms))

    def disarm(self, timer_id: int) -> None:
        self.disarmed.append(timer_id)

    def retry(self, hwnd: int = 0x1234):
        return msgwindow.SessionNotificationRetry(
            hwnd, self.register, self.unregister, self.arm, self.disarm
        )


def test_session_notification_registers_on_the_first_try_without_a_timer():
    api = _FakeSessionApi([None])
    retry = api.retry()
    assert retry.start() is True
    assert retry.registered is True
    assert api.armed == []
    retry.stop()
    assert api.unregistered == [0x1234]
    assert retry.registered is False


def test_session_notification_retries_on_the_timer_until_it_succeeds():
    # RPC_S_INVALID_BINDING is what the call returns at logon autostart, before the
    # Remote Desktop Services dependencies have started.
    api = _FakeSessionApi([msgwindow.RPC_S_INVALID_BINDING, msgwindow.RPC_S_INVALID_BINDING, None])
    retry = api.retry()
    armed_once = [(msgwindow.SESSION_RETRY_TIMER_ID, msgwindow.SESSION_RETRY_INTERVAL_MS)]
    assert retry.start() is False
    assert api.armed == armed_once
    assert retry.on_timer() is False
    assert api.armed == armed_once  # armed once, not re-armed on every tick
    assert retry.on_timer() is True
    assert retry.registered is True
    assert retry.attempts == 3
    assert api.disarmed == [msgwindow.SESSION_RETRY_TIMER_ID]
    retry.stop()
    assert api.unregistered == [0x1234]


def test_session_notification_gives_up_after_five_minutes_with_one_log_line(caplog):
    api = _FakeSessionApi([msgwindow.RPC_S_INVALID_BINDING] * 200)
    retry = api.retry()
    with caplog.at_level(logging.WARNING, logger="spells.win32.msgwindow"):
        assert retry.start() is False
        for _ in range(msgwindow.SESSION_RETRY_MAX_ATTEMPTS + 10):
            assert retry.on_timer() is False
    assert retry.attempts == msgwindow.SESSION_RETRY_MAX_ATTEMPTS
    assert retry.gave_up is True
    assert retry.registered is False
    assert api.disarmed == [msgwindow.SESSION_RETRY_TIMER_ID]
    assert len([r for r in caplog.records if r.levelno >= logging.WARNING]) == 1
    retry.stop()
    assert api.unregistered == []  # nothing was ever registered, so nothing is undone


def test_session_notification_stop_while_retrying_kills_the_timer_only():
    api = _FakeSessionApi([msgwindow.RPC_S_INVALID_BINDING, None])
    retry = api.retry()
    assert retry.start() is False
    retry.stop()
    assert api.disarmed == [msgwindow.SESSION_RETRY_TIMER_ID]
    assert api.unregistered == []
    assert retry.on_timer() is False  # a tick after stop() does not register again
    assert retry.registered is False


def test_session_retry_constants_cover_five_minutes_at_five_seconds():
    assert msgwindow.RPC_S_INVALID_BINDING == 1702
    assert msgwindow.SESSION_RETRY_INTERVAL_MS == 5000
    assert msgwindow.SESSION_RETRY_LIMIT_S == 300
    assert msgwindow.SESSION_RETRY_MAX_ATTEMPTS == 60
    assert msgwindow.WM_TIMER == 0x0113


# release_held_modifiers and the Start menu mask


def _fake_input_backend(monkeypatch, down: tuple[int, ...]):
    """Records the (vk, keydown, extra_info) triples release_held_modifiers sends."""
    calls: list[list[tuple[int, bool, int]]] = []
    monkeypatch.setattr(win_input, "is_key_down", lambda vk: vk in down)
    monkeypatch.setattr(
        win_input, "keyboard_input", lambda vk, keydown, extra_info=0: (vk, keydown, extra_info)
    )
    monkeypatch.setattr(win_input, "send_inputs", lambda records: calls.append(list(records)))
    return calls


def test_release_held_modifiers_masks_a_held_left_win_key(monkeypatch):
    calls = _fake_input_backend(monkeypatch, (win_input.VK_LWIN,))
    assert win_input.release_held_modifiers() == [win_input.VK_LWIN]
    assert calls == [
        [
            (win_input.MASK_VK, True, win_input.MASK_TAG),
            (win_input.MASK_VK, False, win_input.MASK_TAG),
            (win_input.VK_LWIN, False, 0),
        ]
    ]


def test_release_held_modifiers_masks_each_win_key_in_release_order(monkeypatch):
    down = (win_input.VK_SHIFT, win_input.VK_LWIN, win_input.VK_RWIN, win_input.VK_LSHIFT)
    calls = _fake_input_backend(monkeypatch, down)
    assert win_input.release_held_modifiers() == [
        win_input.VK_SHIFT,
        win_input.VK_LWIN,
        win_input.VK_RWIN,
        win_input.VK_LSHIFT,
    ]
    assert calls[0] == [
        (win_input.MASK_VK, True, win_input.MASK_TAG),
        (win_input.MASK_VK, False, win_input.MASK_TAG),
        (win_input.VK_LWIN, False, 0),
        (win_input.MASK_VK, True, win_input.MASK_TAG),
        (win_input.MASK_VK, False, win_input.MASK_TAG),
        (win_input.VK_RWIN, False, 0),
        (win_input.VK_SHIFT, False, 0),
        (win_input.VK_LSHIFT, False, 0),
    ]
    assert len(calls) == 1  # one SendInput call, so nothing can interleave with the mask


def test_release_held_modifiers_sends_no_mask_without_a_win_key(monkeypatch):
    calls = _fake_input_backend(monkeypatch, (win_input.VK_CONTROL, win_input.VK_LCONTROL))
    assert win_input.release_held_modifiers() == [win_input.VK_CONTROL, win_input.VK_LCONTROL]
    assert calls == [[(win_input.VK_CONTROL, False, 0), (win_input.VK_LCONTROL, False, 0)]]


def test_release_held_modifiers_sends_nothing_when_no_modifier_is_down(monkeypatch):
    calls = _fake_input_backend(monkeypatch, ())
    assert win_input.release_held_modifiers() == []
    assert calls == []


def test_mask_key_and_tag_match_the_hook_and_hotkey_layers():
    # The masking key is the key the hook injects on a Win key-up, with the same private
    # dwExtraInfo tag, so the hotkey state machine passes it through instead of reading it
    # as a chord key or as the liveness probe.
    assert win_input.MASK_VK == hotkey.MASK_VK == hook.VK_PROBE == 0xE8
    assert win_input.MASK_TAG == hotkey.MASK_TAG
    assert win_input.MASK_TAG != hotkey.PROBE_TAG


def test_input_modifier_constants_come_from_the_shared_vk_module():
    for name in (
        "VK_SHIFT",
        "VK_CONTROL",
        "VK_MENU",
        "VK_LWIN",
        "VK_RWIN",
        "VK_LSHIFT",
        "VK_RSHIFT",
        "VK_LCONTROL",
        "VK_RCONTROL",
        "VK_LMENU",
        "VK_RMENU",
    ):
        assert getattr(win_input, name) == getattr(vk, name), name
    assert win_input.MODIFIER_VKS == (
        vk.VK_SHIFT,
        vk.VK_CONTROL,
        vk.VK_MENU,
        vk.VK_LWIN,
        vk.VK_RWIN,
        vk.VK_LSHIFT,
        vk.VK_RSHIFT,
        vk.VK_LCONTROL,
        vk.VK_RCONTROL,
        vk.VK_LMENU,
        vk.VK_RMENU,
    )


# set_text and restore sequence handling


class _FakeClipboard:
    """The Win32 side of spells.win32.clipboard, faked.

    It models what the real API does to the sequence number: EmptyClipboard and every
    SetClipboardData bump it by one while the clipboard is held, and CloseClipboard bumps
    it once more per format Windows synthesizes from CF_UNICODETEXT (CF_TEXT, CF_OEMTEXT
    and CF_LOCALE, measured on Windows 11).
    """

    SYNTHESIZED_FORMATS = 3

    def __init__(self, monkeypatch, *, text: str | None = None, sequence: int = 100) -> None:
        self.sequence = sequence
        self.open = False
        self.sessions = 0
        self.emptied = 0
        self.puts: list[tuple[int, bytes]] = []
        self.tags = 0
        self.text = text
        self.reads: list[tuple[int, bool]] = []
        self._pending_synthesis = 0
        self.open_error: OSError | None = None
        monkeypatch.setattr(clipboard, "_Session", self._session)
        monkeypatch.setattr(clipboard, "_put_global", self._put_global)
        monkeypatch.setattr(clipboard, "_put_exclusion_tag", self._put_tag)
        monkeypatch.setattr(clipboard, "sequence_number", self._sequence_number)
        monkeypatch.setattr(clipboard, "_current_text", lambda: self.text)
        monkeypatch.setattr(clipboard, "_user32", SimpleNamespace(EmptyClipboard=self._empty))

    @contextlib.contextmanager
    def _session(self):
        self.sessions += 1
        if self.open_error is not None:
            raise self.open_error
        self.open = True
        try:
            yield self
        finally:
            self.open = False
            self.sequence += self._pending_synthesis
            self._pending_synthesis = 0

    def _empty(self) -> int:
        self.emptied += 1
        self.sequence += 1
        self._pending_synthesis = 0
        return 1

    def _put_global(self, fmt: int, data: bytes) -> None:
        self.puts.append((fmt, data))
        self.sequence += 1
        if fmt == clipboard.CF_UNICODETEXT:
            self._pending_synthesis = self.SYNTHESIZED_FORMATS

    def _put_tag(self) -> None:
        self.tags += 1
        self.sequence += 1

    def _sequence_number(self) -> int:
        self.reads.append((self.sequence, self.open))
        return self.sequence


def test_set_text_returns_a_sequence_read_while_the_clipboard_is_held(monkeypatch):
    fake = _FakeClipboard(monkeypatch, text="dictated text")
    sequence = clipboard.set_text("dictated text")
    assert fake.puts == [(clipboard.CF_UNICODETEXT, "dictated text".encode("utf-16-le") + b"\x00\x00")]
    assert fake.tags == 1
    assert fake.reads == [(fake.sequence, True)]  # read inside a session, never in the gap
    assert sequence == fake.sequence
    # The number still matches after the close, so restore() can use it.
    assert clipboard.restore(clipboard.ClipboardSnapshot({1: b"x"}, sequence), sequence) is True


def test_set_text_refuses_the_sequence_when_another_app_replaced_the_text(monkeypatch):
    # Another application writing in the gap between CloseClipboard and the confirming
    # read must never end up with our number: restore() would overwrite its content.
    fake = _FakeClipboard(monkeypatch, text="someone else")
    sequence = clipboard.set_text("dictated text")
    assert sequence == clipboard.SEQUENCE_UNKNOWN
    assert clipboard.restore(clipboard.ClipboardSnapshot({1: b"x"}, 0), sequence) is False
    assert fake.tags == 1  # the text itself was still placed


def test_set_text_refuses_the_sequence_when_the_confirming_open_fails(monkeypatch):
    fake = _FakeClipboard(monkeypatch, text="dictated text")
    original_session = fake._session

    def failing_after_the_first(*args):
        if fake.sessions >= 1:
            fake.open_error = OSError(5, "OpenClipboard failed")
        return original_session()

    monkeypatch.setattr(clipboard, "_Session", failing_after_the_first)
    assert clipboard.set_text("dictated text") == clipboard.SEQUENCE_UNKNOWN


def test_restore_of_an_empty_snapshot_leaves_the_clipboard_empty(monkeypatch):
    fake = _FakeClipboard(monkeypatch)
    snap = clipboard.ClipboardSnapshot(formats={}, sequence=fake.sequence)
    assert clipboard.restore(snap, fake.sequence) is True
    assert fake.emptied == 1
    assert fake.puts == []
    assert fake.tags == 0  # an empty clipboard is restored to empty, tag included


def test_restore_of_a_non_empty_snapshot_writes_the_exclusion_tag_last(monkeypatch):
    fake = _FakeClipboard(monkeypatch)
    snap = clipboard.ClipboardSnapshot(formats={clipboard.CF_HDROP: b"drop"}, sequence=fake.sequence)
    assert clipboard.restore(snap, fake.sequence) is True
    assert fake.puts == [(clipboard.CF_HDROP, b"drop")]
    assert fake.tags == 1


# window: pill support (spec 14.2)


def test_pill_window_constants():
    assert window.GWL_EXSTYLE == -20
    assert window.WS_EX_TOOLWINDOW == 0x00000080
    assert window.WS_EX_NOACTIVATE == 0x08000000
    assert window.WS_EX_APPWINDOW == 0x00040000
    assert window.SPI_GETCLIENTAREAANIMATION == 0x1042
    assert window.WS_EX_TOOLWINDOW == msgwindow.WS_EX_TOOLWINDOW
    assert window.WS_EX_NOACTIVATE == msgwindow.WS_EX_NOACTIVATE


def test_monitor_helpers_share_the_monitorinfo_layout():
    assert ctypes.sizeof(window.MONITORINFO) == 40
    assert [name for name, _type in window.MONITORINFO._fields_] == [
        "cbSize",
        "rcMonitor",
        "rcWork",
        "dwFlags",
    ]
    assert callable(window.monitor_rect_for_window)
    assert callable(window.monitor_handle_for_window)


# Live insertion: backspaces and the tagged, atomic bursts (spec 6, B5-57)


def _fake_typing_backend(monkeypatch, down: tuple[int, ...] = ()):
    """Records every SendInput batch as (vk|unit, keydown, extra_info) triples."""
    calls: list[list[tuple[int, bool, int]]] = []
    monkeypatch.setattr(win_input, "is_key_down", lambda vk: vk in down)
    monkeypatch.setattr(
        win_input, "keyboard_input", lambda vk, keydown, extra_info=0: (vk, keydown, extra_info)
    )
    monkeypatch.setattr(
        win_input, "unicode_input", lambda unit, keydown, extra_info=0: (unit, keydown, extra_info)
    )
    monkeypatch.setattr(win_input, "send_inputs", lambda records: calls.append(list(records)))
    return calls


def test_plan_backspaces_is_one_down_and_up_per_tap():
    keys = win_input.plan_backspaces(3)
    assert len(keys) == 6
    assert all(key.vk == win_input.VK_BACK for key in keys)
    assert [key.keydown for key in keys] == [True, False, True, False, True, False]


def test_plan_backspaces_of_zero_or_less_is_empty():
    assert win_input.plan_backspaces(0) == []
    assert win_input.plan_backspaces(-4) == []


def test_send_backspaces_batches_like_typing(monkeypatch):
    calls = _fake_typing_backend(monkeypatch)
    win_input.send_backspaces(5, batch_size=2)
    assert [len(batch) for batch in calls] == [4, 4, 2]


def test_send_backspaces_carries_the_private_tag(monkeypatch):
    calls = _fake_typing_backend(monkeypatch)
    win_input.send_backspaces(1, extra_info=win_input.MASK_TAG)
    assert calls == [
        [
            (win_input.VK_BACK, True, win_input.MASK_TAG),
            (win_input.VK_BACK, False, win_input.MASK_TAG),
        ]
    ]


def test_live_typing_releases_the_held_chord_inside_the_same_send(monkeypatch):
    calls = _fake_typing_backend(monkeypatch, (win_input.VK_CONTROL, win_input.VK_LWIN))
    win_input.type_unicode("a", extra_info=win_input.MASK_TAG, release_modifiers=True)
    assert len(calls) == 1
    assert calls[0][:5] == [
        (win_input.MASK_VK, True, win_input.MASK_TAG),
        (win_input.MASK_VK, False, win_input.MASK_TAG),
        (win_input.VK_LWIN, False, win_input.MASK_TAG),
        (win_input.VK_CONTROL, False, win_input.MASK_TAG),
        (ord("a"), True, win_input.MASK_TAG),
    ]


def test_live_backspaces_release_the_held_chord_inside_the_same_send(monkeypatch):
    calls = _fake_typing_backend(monkeypatch, (win_input.VK_CONTROL,))
    win_input.send_backspaces(1, extra_info=win_input.MASK_TAG, release_modifiers=True)
    assert calls == [
        [
            (win_input.VK_CONTROL, False, win_input.MASK_TAG),
            (win_input.VK_BACK, True, win_input.MASK_TAG),
            (win_input.VK_BACK, False, win_input.MASK_TAG),
        ]
    ]


def test_type_unicode_without_the_live_options_is_unchanged(monkeypatch):
    calls = _fake_typing_backend(monkeypatch, (win_input.VK_CONTROL,))
    win_input.type_unicode("a")
    assert calls == [[(ord("a"), True, 0), (ord("a"), False, 0)]]


def test_the_live_tag_matches_the_hook_tag():
    from spells import livetext

    assert livetext.TAG == win_input.MASK_TAG == hotkey.MASK_TAG
