"""Hidden window for WM_COPYDATA, session end, power resume and session unlock.

Spec sections 13 (hook reinstall on resume and unlock), 15 (single-instance channel) and
19.5 (Restart Manager); batch 2 decisions V1-10, V3-19 (answer WM_QUERYENDSESSION with
TRUE, exit on WM_ENDSESSION) and V4-8.

The window is a hidden top-level window rather than a message-only (HWND_MESSAGE) one:
message-only windows never receive broadcast messages, and WM_QUERYENDSESSION,
WM_ENDSESSION and WM_POWERBROADCAST are broadcasts. It is never shown, carries
WS_EX_TOOLWINDOW so it stays out of Alt+Tab, and instance.find_message_window locates
it by class name. The window belongs to the thread that creates it; run, pump_once and
DestroyWindow must happen on that thread (destroy() forwards WM_CLOSE from other threads).

Session notifications are registered through SessionNotificationRetry: at logon autostart
the registration fails until Remote Desktop Services is up, so a window timer retries it
for five minutes before giving up.
"""

import ctypes
import logging
import threading
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import dataclass

from . import hook

logger = logging.getLogger(__name__)

RegisterCall = Callable[[int], int | None]
UnregisterCall = Callable[[int], None]
ArmTimerCall = Callable[[int, int], None]
DisarmTimerCall = Callable[[int], None]

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_wtsapi32 = ctypes.WinDLL("wtsapi32", use_last_error=True)

LRESULT = ctypes.c_ssize_t
ULONG_PTR = ctypes.c_size_t
UINT_PTR = ctypes.c_size_t

WM_DESTROY = 0x0002
WM_CLOSE = 0x0010
WM_QUERYENDSESSION = 0x0011
WM_ENDSESSION = 0x0016
WM_COPYDATA = 0x004A
WM_TIMER = 0x0113
WM_POWERBROADCAST = 0x0218
WM_WTSSESSION_CHANGE = 0x02B1

ENDSESSION_CLOSEAPP = 0x00000001
ENDSESSION_CRITICAL = 0x40000000
ENDSESSION_LOGOFF = 0x80000000
PBT_APMRESUMEAUTOMATIC = 0x0012
WTS_SESSION_LOCK = 0x7
WTS_SESSION_UNLOCK = 0x8
NOTIFY_FOR_THIS_SESSION = 0

# WTSRegisterSessionNotification answers this while the Remote Desktop Services dependencies
# are still starting, which is the normal state at logon autostart.
RPC_S_INVALID_BINDING = 1702
SESSION_RETRY_TIMER_ID = 1
SESSION_RETRY_INTERVAL_MS = 5000
SESSION_RETRY_LIMIT_S = 300
SESSION_RETRY_MAX_ATTEMPTS = SESSION_RETRY_LIMIT_S * 1000 // SESSION_RETRY_INTERVAL_MS

WS_POPUP = 0x80000000
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000
HWND_MESSAGE = ctypes.c_void_p(-3)
ERROR_CLASS_ALREADY_EXISTS = 1410
SMTO_NORMAL = 0x0000
SMTO_ABORTIFHUNG = 0x0002
COPYDATA_TIMEOUT_MS = 5000

# Private dwData tag so stray WM_COPYDATA from other programs is ignored.
COPYDATA_TAG = 0x55545452  # "UTTR"

WNDPROC = ctypes.WINFUNCTYPE(
    LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
)


class WNDCLASSEXW(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.UINT),
        ("style", wintypes.UINT),
        ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HICON),
        ("hCursor", wintypes.HANDLE),
        ("hbrBackground", wintypes.HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
        ("hIconSm", wintypes.HICON),
    ]


class COPYDATASTRUCT(ctypes.Structure):
    _fields_ = [
        ("dwData", ULONG_PTR),
        ("cbData", wintypes.DWORD),
        ("lpData", ctypes.c_void_p),
    ]


_user32.RegisterClassExW.argtypes = (ctypes.POINTER(WNDCLASSEXW),)
_user32.RegisterClassExW.restype = wintypes.ATOM
_user32.CreateWindowExW.argtypes = (
    wintypes.DWORD,
    wintypes.LPCWSTR,
    wintypes.LPCWSTR,
    wintypes.DWORD,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    wintypes.HWND,
    wintypes.HMENU,
    wintypes.HINSTANCE,
    wintypes.LPVOID,
)
_user32.CreateWindowExW.restype = wintypes.HWND
_user32.DestroyWindow.argtypes = (wintypes.HWND,)
_user32.DestroyWindow.restype = wintypes.BOOL
_user32.DefWindowProcW.argtypes = (wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
_user32.DefWindowProcW.restype = LRESULT
_user32.SendMessageTimeoutW.argtypes = (
    wintypes.HWND,
    wintypes.UINT,
    wintypes.WPARAM,
    wintypes.LPARAM,
    wintypes.UINT,
    wintypes.UINT,
    ctypes.POINTER(ctypes.c_size_t),
)
_user32.SendMessageTimeoutW.restype = LRESULT
_user32.IsWindow.argtypes = (wintypes.HWND,)
_user32.IsWindow.restype = wintypes.BOOL
_kernel32.GetModuleHandleW.argtypes = (wintypes.LPCWSTR,)
_kernel32.GetModuleHandleW.restype = wintypes.HMODULE
_kernel32.GetCurrentThreadId.argtypes = ()
_kernel32.GetCurrentThreadId.restype = wintypes.DWORD
_user32.SetTimer.argtypes = (wintypes.HWND, UINT_PTR, wintypes.UINT, ctypes.c_void_p)
_user32.SetTimer.restype = UINT_PTR
_user32.KillTimer.argtypes = (wintypes.HWND, UINT_PTR)
_user32.KillTimer.restype = wintypes.BOOL
_wtsapi32.WTSRegisterSessionNotification.argtypes = (wintypes.HWND, wintypes.DWORD)
_wtsapi32.WTSRegisterSessionNotification.restype = wintypes.BOOL
_wtsapi32.WTSUnRegisterSessionNotification.argtypes = (wintypes.HWND,)
_wtsapi32.WTSUnRegisterSessionNotification.restype = wintypes.BOOL


@dataclass
class MessageHandlers:
    """Callbacks invoked on the window's thread. Any of them may be None."""

    on_copydata: Callable[[str], None] | None = None
    on_query_end_session: Callable[[int], bool] | None = None
    on_end_session: Callable[[int], None] | None = None
    on_resume: Callable[[], None] | None = None
    on_session_unlock: Callable[[], None] | None = None


class SessionNotificationRetry:
    """Keeps WTSRegisterSessionNotification trying until it works or the budget runs out.

    At logon autostart the call fails with RPC_S_INVALID_BINDING because Remote Desktop
    Services and its dependents are not up yet. A single attempt in the constructor would
    then lose WTS_SESSION_UNLOCK for the whole session, and with it the hook reinstall of
    spec 13. Instead a window timer retries every SESSION_RETRY_INTERVAL_MS, and after
    SESSION_RETRY_LIMIT_S the retry stops with one log line. The documented alternative is
    to wait for the Global\\TermSrvReadyEvent event, which would need a thread of its own;
    the timer the window already pumps costs nothing. The Win32 calls are passed in so the
    policy can be unit tested without touching Win32.
    """

    def __init__(
        self,
        hwnd: int,
        register: RegisterCall,
        unregister: UnregisterCall,
        arm_timer: ArmTimerCall,
        disarm_timer: DisarmTimerCall,
    ) -> None:
        self._hwnd = hwnd
        self._register = register
        self._unregister = unregister
        self._arm_timer = arm_timer
        self._disarm_timer = disarm_timer
        self.registered = False
        self.attempts = 0
        self.gave_up = False
        self._armed = False
        self._stopped = False

    def start(self) -> bool:
        """First attempt; arms the retry timer when it fails. True once registered."""
        return self._attempt()

    def on_timer(self) -> bool:
        """One retry tick, driven by the window's WM_TIMER. True once registered."""
        if self.registered or self.gave_up or self._stopped:
            self._disarm()
            return self.registered
        return self._attempt()

    def stop(self) -> None:
        """Stop retrying and undo a registration that succeeded. Idempotent.

        A tick that was already queued when the window went away must not register a dead
        hwnd, so this is final: on_timer() does nothing afterwards.
        """
        self._stopped = True
        self._disarm()
        if self.registered:
            self.registered = False
            self._unregister(self._hwnd)

    def _attempt(self) -> bool:
        if self._stopped:
            return False
        self.attempts += 1
        error = self._register(self._hwnd)
        if error is None:
            self.registered = True
            self._disarm()
            return True
        if self.attempts >= SESSION_RETRY_MAX_ATTEMPTS:
            self.gave_up = True
            self._disarm()
            logger.warning(
                "WTSRegisterSessionNotification still failing with error %d after %d tries "
                "over %d s; giving up, session lock and unlock will not be reported",
                error,
                self.attempts,
                SESSION_RETRY_LIMIT_S,
            )
            return False
        if self.attempts == 1:
            logger.debug(
                "WTSRegisterSessionNotification failed with error %d; retrying every %d ms",
                error,
                SESSION_RETRY_INTERVAL_MS,
            )
        self._arm()
        return False

    def _arm(self) -> None:
        if not self._armed:
            self._arm_timer(SESSION_RETRY_TIMER_ID, SESSION_RETRY_INTERVAL_MS)
            self._armed = True

    def _disarm(self) -> None:
        if self._armed:
            self._armed = False
            self._disarm_timer(SESSION_RETRY_TIMER_ID)


def _register_session_notification(hwnd: int) -> int | None:
    """Register hwnd for session change notifications; None on success, else the error."""
    if _wtsapi32.WTSRegisterSessionNotification(hwnd, NOTIFY_FOR_THIS_SESSION):
        return None
    return ctypes.get_last_error()


def _unregister_session_notification(hwnd: int) -> None:
    _wtsapi32.WTSUnRegisterSessionNotification(hwnd)


# hwnd -> MessageWindow; one shared window procedure dispatches to the instance.
_windows: dict[int, "MessageWindow"] = {}
_registered_classes: set[str] = set()
_class_lock = threading.Lock()


def _window_proc(hwnd: int | None, msg: int, wparam: int, lparam: int) -> int:
    hwnd = int(hwnd or 0)
    window = _windows.get(hwnd)
    if window is not None:
        try:
            result = window._handle(msg, wparam, lparam)
        except Exception:
            logger.exception("message window handler raised for message 0x%x", msg)
            result = None
        if result is not None:
            return result
    return _user32.DefWindowProcW(hwnd, msg, wparam, lparam)


_WNDPROC = WNDPROC(_window_proc)


def _ensure_class(class_name: str) -> None:
    with _class_lock:
        if class_name in _registered_classes:
            return
        wc = WNDCLASSEXW()
        wc.cbSize = ctypes.sizeof(WNDCLASSEXW)
        wc.lpfnWndProc = _WNDPROC
        wc.hInstance = _kernel32.GetModuleHandleW(None)
        wc.lpszClassName = class_name
        if not _user32.RegisterClassExW(ctypes.byref(wc)):
            error = ctypes.get_last_error()
            if error != ERROR_CLASS_ALREADY_EXISTS:
                raise ctypes.WinError(error)
        _registered_classes.add(class_name)


class MessageWindow:
    """A hidden window whose class name doubles as the single-instance channel name."""

    def __init__(self, class_name: str, handlers: MessageHandlers) -> None:
        self.class_name = class_name
        self.handlers = handlers
        self._thread_id = int(_kernel32.GetCurrentThreadId())
        _ensure_class(class_name)
        hwnd = _user32.CreateWindowExW(
            WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
            class_name,
            class_name,
            WS_POPUP,
            0,
            0,
            0,
            0,
            None,
            None,
            _kernel32.GetModuleHandleW(None),
            None,
        )
        if not hwnd:
            raise ctypes.WinError(ctypes.get_last_error())
        self._hwnd = int(hwnd)
        _windows[self._hwnd] = self
        self._session = SessionNotificationRetry(
            self._hwnd,
            _register_session_notification,
            _unregister_session_notification,
            self._set_window_timer,
            self._kill_window_timer,
        )
        self._session.start()

    @property
    def hwnd(self) -> int:
        return self._hwnd

    def _set_window_timer(self, timer_id: int, interval_ms: int) -> None:
        if self._hwnd:
            _user32.SetTimer(self._hwnd, timer_id, interval_ms, None)

    def _kill_window_timer(self, timer_id: int) -> None:
        if self._hwnd:
            _user32.KillTimer(self._hwnd, timer_id)

    def _handle(self, msg: int, wparam: int, lparam: int) -> int | None:
        handlers = self.handlers
        if msg == WM_COPYDATA:
            cds = ctypes.cast(lparam, ctypes.POINTER(COPYDATASTRUCT)).contents
            if cds.dwData != COPYDATA_TAG:
                return 0
            text = ""
            if cds.cbData and cds.lpData:
                raw = ctypes.string_at(cds.lpData, cds.cbData)
                text = raw.decode("utf-16-le", errors="replace").rstrip("\x00")
            if handlers.on_copydata is not None:
                handlers.on_copydata(text)
            return 1
        if msg == WM_QUERYENDSESSION:
            allow = True
            if handlers.on_query_end_session is not None:
                allow = bool(handlers.on_query_end_session(lparam))
            return 1 if allow else 0
        if msg == WM_ENDSESSION:
            if wparam and handlers.on_end_session is not None:
                handlers.on_end_session(lparam)
            return 0
        if msg == WM_POWERBROADCAST:
            if wparam == PBT_APMRESUMEAUTOMATIC and handlers.on_resume is not None:
                handlers.on_resume()
            return 1
        if msg == WM_WTSSESSION_CHANGE:
            if wparam == WTS_SESSION_UNLOCK and handlers.on_session_unlock is not None:
                handlers.on_session_unlock()
            return 0
        if msg == WM_TIMER and wparam == SESSION_RETRY_TIMER_ID:
            self._session.on_timer()
            return 0
        if msg == WM_CLOSE:
            _user32.DestroyWindow(self._hwnd)
            return 0
        if msg == WM_DESTROY:
            self._session.stop()
            _windows.pop(self._hwnd, None)
            self._hwnd = 0
            return 0
        return None

    def pump_once(self) -> None:
        """Dispatch pending messages for the window's thread (call on that thread)."""
        hook.pump_messages()

    def run(self, stop: threading.Event) -> None:
        """Pump the window's thread until stop is set (call on that thread)."""
        hook.run_message_loop(stop, {}, None)

    def destroy(self) -> None:
        """Destroy the window; from another thread this sends WM_CLOSE to its thread."""
        hwnd = self._hwnd
        if not hwnd:
            return
        if int(_kernel32.GetCurrentThreadId()) == self._thread_id:
            _user32.DestroyWindow(hwnd)
            _windows.pop(hwnd, None)
            self._hwnd = 0
        else:
            _user32.SendMessageTimeoutW(hwnd, WM_CLOSE, 0, 0, SMTO_ABORTIFHUNG, 2000, None)


def send_copydata(hwnd: int, text: str) -> bool:
    """Send text to a window as WM_COPYDATA (UTF-16 payload); True when it was accepted."""
    payload = text.encode("utf-16-le") + b"\x00\x00"
    buffer = ctypes.create_string_buffer(payload, len(payload))
    cds = COPYDATASTRUCT()
    cds.dwData = COPYDATA_TAG
    cds.cbData = len(payload)
    cds.lpData = ctypes.cast(buffer, ctypes.c_void_p)
    result = ctypes.c_size_t(0)
    ok = _user32.SendMessageTimeoutW(
        hwnd,
        WM_COPYDATA,
        0,
        ctypes.addressof(cds),
        SMTO_ABORTIFHUNG,
        COPYDATA_TIMEOUT_MS,
        ctypes.byref(result),
    )
    return bool(ok) and result.value != 0
