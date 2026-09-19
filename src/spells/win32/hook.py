"""Low-level keyboard hook, SendInput, thread timers and the hook thread's message loop.

Spec sections 5.1, 6, 13 and 14.4; batch 2 decisions V1-10 and V3-F3. Windows silently
removes a low-level hook whose procedure responds slowly, so the hook procedure here only
converts the raw KBDLLHOOKSTRUCT into a KeyEvent and hands it to the registered Python
callback, which is expected to enqueue and return quickly.

Threading notes:
- A WH_KEYBOARD_LL hook is called on the thread that installed it, and only while that
  thread retrieves messages. install_keyboard_hook must therefore run on the thread that
  then runs run_message_loop.
- Timers created by run_message_loop, set_timer and kill_timer are thread timers (no
  window). Windows ignores the requested id for such timers and assigns its own, so this
  module keeps a per-thread map between the caller's ids and the system ids.
- send_key called from the hook thread itself may invoke the hook callback re-entrantly,
  before send_key returns; from any other thread the callback runs asynchronously.
"""

import ctypes
import logging
import threading
from collections import deque
from collections.abc import Callable, Sequence
from ctypes import wintypes
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

LRESULT = ctypes.c_ssize_t
ULONG_PTR = ctypes.c_size_t
UINT_PTR = ctypes.c_size_t

WH_KEYBOARD_LL = 13
HC_ACTION = 0

LLKHF_EXTENDED = 0x01
LLKHF_LOWER_IL_INJECTED = 0x02
LLKHF_INJECTED = 0x10
LLKHF_ALTDOWN = 0x20
LLKHF_UP = 0x80

VK_PROBE = 0xE8  # unassigned virtual key used for the liveness probe and Start menu masking

WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_SYSKEYDOWN = 0x0104
WM_SYSKEYUP = 0x0105
WM_QUIT = 0x0012
WM_TIMER = 0x0113
WM_HOTKEY = 0x0312

PM_REMOVE = 0x0001
QS_ALLINPUT = 0x04FF
MWMO_INPUTAVAILABLE = 0x0004

INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_SCANCODE = 0x0008
MAPVK_VK_TO_VSC = 0

THREAD_PRIORITY_HIGHEST = 2

MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000
ERROR_HOTKEY_ALREADY_REGISTERED = 1409
_HOTKEY_PROBE_ID = 0xB0B0  # private id inside the application range 0x0000 to 0xBFFF

# Keys whose scan code carries the E0 prefix; SendInput needs KEYEVENTF_EXTENDEDKEY for them.
# fmt: off
_EXTENDED_VKS = frozenset({
    0x21, 0x22, 0x23, 0x24,  # VK_PRIOR, VK_NEXT, VK_END, VK_HOME
    0x25, 0x26, 0x27, 0x28,  # arrows
    0x2C, 0x2D, 0x2E,  # VK_SNAPSHOT, VK_INSERT, VK_DELETE
    0x5B, 0x5C, 0x5D,  # VK_LWIN, VK_RWIN, VK_APPS
    0x6F, 0x90,  # VK_DIVIDE, VK_NUMLOCK
    0xA3, 0xA5,  # VK_RCONTROL, VK_RMENU
})
# fmt: on


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode", wintypes.DWORD),
        ("scanCode", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUT_UNION)]


HOOKPROC = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)

_user32.SetWindowsHookExW.argtypes = (ctypes.c_int, HOOKPROC, wintypes.HINSTANCE, wintypes.DWORD)
_user32.SetWindowsHookExW.restype = wintypes.HHOOK
_user32.UnhookWindowsHookEx.argtypes = (wintypes.HHOOK,)
_user32.UnhookWindowsHookEx.restype = wintypes.BOOL
_user32.CallNextHookEx.argtypes = (wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
_user32.CallNextHookEx.restype = LRESULT
_user32.PeekMessageW.argtypes = (
    wintypes.LPMSG,
    wintypes.HWND,
    wintypes.UINT,
    wintypes.UINT,
    wintypes.UINT,
)
_user32.PeekMessageW.restype = wintypes.BOOL
_user32.TranslateMessage.argtypes = (wintypes.LPMSG,)
_user32.TranslateMessage.restype = wintypes.BOOL
_user32.DispatchMessageW.argtypes = (wintypes.LPMSG,)
_user32.DispatchMessageW.restype = LRESULT
_user32.MsgWaitForMultipleObjectsEx.argtypes = (
    wintypes.DWORD,
    wintypes.LPHANDLE,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.DWORD,
)
_user32.MsgWaitForMultipleObjectsEx.restype = wintypes.DWORD
_user32.SetTimer.argtypes = (wintypes.HWND, UINT_PTR, wintypes.UINT, ctypes.c_void_p)
_user32.SetTimer.restype = UINT_PTR
_user32.KillTimer.argtypes = (wintypes.HWND, UINT_PTR)
_user32.KillTimer.restype = wintypes.BOOL
_user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
_user32.SendInput.restype = wintypes.UINT
_user32.GetAsyncKeyState.argtypes = (ctypes.c_int,)
_user32.GetAsyncKeyState.restype = wintypes.SHORT
_user32.MapVirtualKeyW.argtypes = (wintypes.UINT, wintypes.UINT)
_user32.MapVirtualKeyW.restype = wintypes.UINT
_user32.RegisterHotKey.argtypes = (wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT)
_user32.RegisterHotKey.restype = wintypes.BOOL
_user32.UnregisterHotKey.argtypes = (wintypes.HWND, ctypes.c_int)
_user32.UnregisterHotKey.restype = wintypes.BOOL
_kernel32.GetModuleHandleW.argtypes = (wintypes.LPCWSTR,)
_kernel32.GetModuleHandleW.restype = wintypes.HMODULE
_kernel32.GetCurrentThread.argtypes = ()
_kernel32.GetCurrentThread.restype = wintypes.HANDLE
_kernel32.SetThreadPriority.argtypes = (wintypes.HANDLE, ctypes.c_int)
_kernel32.SetThreadPriority.restype = wintypes.BOOL


@dataclass(frozen=True)
class KeyEvent:
    """One keyboard event as seen by the low-level hook."""

    vk: int
    scan: int
    flags: int
    extra_info: int
    keydown: bool
    time_ms: int

    @property
    def injected(self) -> bool:
        """True for events produced by SendInput (ours or another app's)."""
        return bool(self.flags & LLKHF_INJECTED)


def key_event_from_struct(kb: KBDLLHOOKSTRUCT, wparam: int) -> KeyEvent:
    """Build a KeyEvent from the hook's KBDLLHOOKSTRUCT and the wParam message id."""
    return KeyEvent(
        vk=int(kb.vkCode),
        scan=int(kb.scanCode),
        flags=int(kb.flags),
        extra_info=int(kb.dwExtraInfo),
        keydown=wparam in (WM_KEYDOWN, WM_SYSKEYDOWN),
        time_ms=int(kb.time),
    )


# Hook handle -> (ctypes callback object, Python callback). The ctypes object must outlive the
# hook or Windows calls into freed memory.
_hooks: dict[int, tuple[object, Callable[[KeyEvent], bool]]] = {}
_hooks_lock = threading.Lock()
# Recently uninstalled procedures, kept a little longer in case a call is still in flight.
_retired_procs: deque[object] = deque(maxlen=8)


def install_keyboard_hook(callback: Callable[[KeyEvent], bool]) -> int:
    """Install a WH_KEYBOARD_LL hook on the calling thread and return its handle.

    The callback returns True to swallow the key (the hook procedure returns 1 without
    calling CallNextHookEx) and False to pass it on. Exceptions raised by the callback are
    logged and the key is passed on.
    """

    def proc(ncode: int, wparam: int, lparam: int) -> int:
        if ncode == HC_ACTION:
            try:
                kb = ctypes.cast(lparam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
                if callback(key_event_from_struct(kb, wparam)):
                    return 1
            except Exception:
                logger.exception("keyboard hook callback raised; passing the key on")
        return _user32.CallNextHookEx(None, ncode, wparam, lparam)

    cproc = HOOKPROC(proc)
    hmod = _kernel32.GetModuleHandleW(None)
    handle = _user32.SetWindowsHookExW(WH_KEYBOARD_LL, cproc, hmod, 0)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    with _hooks_lock:
        _hooks[handle] = (cproc, callback)
    return handle


def uninstall_keyboard_hook(handle: int) -> None:
    """Remove a hook installed by install_keyboard_hook.

    A failure (for example when Windows already removed the hook) is logged, not raised,
    because the usual reaction is to reinstall.
    """
    if not _user32.UnhookWindowsHookEx(handle):
        logger.warning(
            "UnhookWindowsHookEx(0x%x) failed with error %d", handle, ctypes.get_last_error()
        )
    with _hooks_lock:
        entry = _hooks.pop(handle, None)
        if entry is not None:
            _retired_procs.append(entry[0])


class _LoopState:
    __slots__ = ("caller_to_sys", "sys_to_caller")

    def __init__(self) -> None:
        self.caller_to_sys: dict[int, int] = {}
        self.sys_to_caller: dict[int, int] = {}


_tls = threading.local()


def _state() -> _LoopState:
    state = getattr(_tls, "state", None)
    if state is None:
        state = _LoopState()
        _tls.state = state
    return state


def set_timer(timer_id: int, interval_ms: int) -> None:
    """Create or re-arm a thread timer for the calling thread (call from the loop thread).

    WM_TIMER for it is delivered to on_timer(timer_id) by run_message_loop or pump_messages.
    """
    state = _state()
    existing = state.caller_to_sys.get(timer_id, 0)
    sys_id = _user32.SetTimer(None, existing, int(interval_ms), None)
    if not sys_id:
        raise ctypes.WinError(ctypes.get_last_error())
    if existing and existing != sys_id:
        state.sys_to_caller.pop(existing, None)
    state.caller_to_sys[timer_id] = sys_id
    state.sys_to_caller[sys_id] = timer_id


def kill_timer(timer_id: int) -> None:
    """Stop a thread timer created by set_timer (call from the loop thread). Idempotent."""
    state = _state()
    sys_id = state.caller_to_sys.pop(timer_id, None)
    if sys_id is None:
        return
    state.sys_to_caller.pop(sys_id, None)
    _user32.KillTimer(None, sys_id)


def pump_messages(on_timer: Callable[[int], None] | None = None) -> bool:
    """Drain the calling thread's message queue without blocking.

    Thread timer ticks go to on_timer(caller id); other messages are translated and
    dispatched, which is what delivers messages to windows owned by this thread. Returns
    False once WM_QUIT was retrieved.
    """
    state = _state()
    msg = wintypes.MSG()
    while _user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
        if msg.message == WM_QUIT:
            return False
        if msg.message == WM_TIMER and not msg.hWnd:
            timer_id = state.sys_to_caller.get(int(msg.wParam))
            if timer_id is not None and on_timer is not None:
                try:
                    on_timer(timer_id)
                except Exception:
                    logger.exception("timer callback raised for timer %d", timer_id)
            continue
        _user32.TranslateMessage(ctypes.byref(msg))
        _user32.DispatchMessageW(ctypes.byref(msg))
    return True


def run_message_loop(
    stop: threading.Event,
    timers: dict[int, int],
    on_timer: Callable[[int], None] | None,
    poll_ms: int = 50,
) -> None:
    """Pump messages on the calling thread until stop is set (checked at least every poll_ms).

    timers maps caller ids to intervals in milliseconds; each becomes a thread timer whose
    ticks call on_timer(id) on this thread. Hook procedures installed by this thread run
    from inside the message retrieval calls made here. All timers are killed on exit, and a
    WM_QUIT posted to the thread also ends the loop.
    """
    state = _state()
    for timer_id, interval_ms in timers.items():
        set_timer(timer_id, interval_ms)
    try:
        while not stop.is_set():
            if not pump_messages(on_timer):
                break
            if stop.is_set():
                break
            _user32.MsgWaitForMultipleObjectsEx(
                0, None, int(poll_ms), QS_ALLINPUT, MWMO_INPUTAVAILABLE
            )
    finally:
        for timer_id in list(state.caller_to_sys):
            kill_timer(timer_id)


def keyboard_input(vk: int, keydown: bool, extra_info: int = 0) -> INPUT:
    """An INPUT record for a virtual-key press or release, with its scan code filled in."""
    scan = _user32.MapVirtualKeyW(vk, MAPVK_VK_TO_VSC) if vk else 0
    flags = 0 if keydown else KEYEVENTF_KEYUP
    if vk in _EXTENDED_VKS:
        flags |= KEYEVENTF_EXTENDEDKEY
    record = INPUT(type=INPUT_KEYBOARD)
    record.ki = KEYBDINPUT(wVk=vk, wScan=scan, dwFlags=flags, time=0, dwExtraInfo=extra_info)
    return record


def unicode_input(unit: int, keydown: bool, extra_info: int = 0) -> INPUT:
    """An INPUT record for one UTF-16 code unit sent with KEYEVENTF_UNICODE."""
    flags = KEYEVENTF_UNICODE | (0 if keydown else KEYEVENTF_KEYUP)
    record = INPUT(type=INPUT_KEYBOARD)
    record.ki = KEYBDINPUT(wVk=0, wScan=unit, dwFlags=flags, time=0, dwExtraInfo=extra_info)
    return record


def send_inputs(records: Sequence[INPUT]) -> None:
    """Send the records with one SendInput call; raises OSError when any were blocked."""
    count = len(records)
    if count == 0:
        return
    array = (INPUT * count)(*records)
    sent = _user32.SendInput(count, array, ctypes.sizeof(INPUT))
    if sent != count:
        error = ctypes.get_last_error()
        raise OSError(error, f"SendInput inserted {sent} of {count} events (error {error})")


def send_key(vk: int, keydown: bool, extra_info: int = 0) -> None:
    """Inject one key press or release with SendInput, tagging it with dwExtraInfo."""
    send_inputs([keyboard_input(vk, keydown, extra_info)])


def is_key_down(vk: int) -> bool:
    """Whether the key is physically or synthetically down right now (GetAsyncKeyState)."""
    return bool(_user32.GetAsyncKeyState(vk) & 0x8000)


def set_current_thread_priority_highest() -> None:
    """Raise the calling thread to THREAD_PRIORITY_HIGHEST (spec 5.1)."""
    if not _kernel32.SetThreadPriority(_kernel32.GetCurrentThread(), THREAD_PRIORITY_HIGHEST):
        raise ctypes.WinError(ctypes.get_last_error())


def register_hotkey_probe(modifiers: int, vk: int) -> bool:
    """Check whether a chord is free by registering and immediately unregistering it.

    Returns False when RegisterHotKey fails with ERROR_HOTKEY_ALREADY_REGISTERED (another
    app or Windows owns the chord). Any other failure raises OSError.
    """
    if _user32.RegisterHotKey(None, _HOTKEY_PROBE_ID, modifiers, vk):
        _user32.UnregisterHotKey(None, _HOTKEY_PROBE_ID)
        return True
    error = ctypes.get_last_error()
    if error == ERROR_HOTKEY_ALREADY_REGISTERED:
        return False
    raise ctypes.WinError(error)
