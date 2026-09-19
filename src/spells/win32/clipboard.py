"""Clipboard snapshot, set and guarded restore.

Spec section 11.2; batch 2 decisions V2-4 (snapshot only global-memory formats) and
V2-16 (delayed-render changes do not bump the sequence number, accepted). The text we
place is tagged with the ExcludeClipboardContentFromMonitorProcessing format so it stays
out of clipboard history and cloud sync, and the restore sets the tag again (decision E11).
"""

import ctypes
import logging
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass, field
from typing import Self

logger = logging.getLogger(__name__)

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

CF_TEXT = 1
CF_BITMAP = 2
CF_METAFILEPICT = 3
CF_SYLK = 4
CF_DIF = 5
CF_TIFF = 6
CF_OEMTEXT = 7
CF_DIB = 8
CF_PALETTE = 9
CF_PENDATA = 10
CF_RIFF = 11
CF_WAVE = 12
CF_UNICODETEXT = 13
CF_ENHMETAFILE = 14
CF_HDROP = 15
CF_LOCALE = 16
CF_DIBV5 = 17
CF_OWNERDISPLAY = 0x0080
CF_DSPTEXT = 0x0081
CF_DSPBITMAP = 0x0082
CF_DSPMETAFILEPICT = 0x0083
CF_DSPENHMETAFILE = 0x008E
CF_PRIVATEFIRST = 0x0200
CF_PRIVATELAST = 0x02FF
CF_GDIOBJFIRST = 0x0300
CF_GDIOBJLAST = 0x03FF

EXCLUDE_FORMAT_NAME = "ExcludeClipboardContentFromMonitorProcessing"

# Formats whose clipboard handle is a GDI object, not global memory.
GDI_HANDLE_FORMATS = frozenset(
    {
        CF_BITMAP,
        CF_PALETTE,
        CF_ENHMETAFILE,
        CF_METAFILEPICT,
        CF_DSPBITMAP,
        CF_DSPENHMETAFILE,
        CF_DSPMETAFILEPICT,
    }
)
# CF_OWNERDISPLAY and the CF_DSP* block: rendered by the owner, nothing to copy.
_DISPLAY_FORMAT_RANGE = range(CF_OWNERDISPLAY, 0x0100)

GMEM_MOVEABLE = 0x0002
# GetClipboardSequenceNumber returns a DWORD, so -1 is a value it never produces: set_text
# returns it when it could not confirm that its own text is on the clipboard, and restore()
# then refuses rather than overwriting whatever another application put there.
SEQUENCE_UNKNOWN = -1
OPEN_TRIES = 10
OPEN_RETRY_DELAY_S = 0.02
MAX_FORMAT_BYTES = 256 * 1024 * 1024

_user32.OpenClipboard.argtypes = (wintypes.HWND,)
_user32.OpenClipboard.restype = wintypes.BOOL
_user32.CloseClipboard.argtypes = ()
_user32.CloseClipboard.restype = wintypes.BOOL
_user32.EmptyClipboard.argtypes = ()
_user32.EmptyClipboard.restype = wintypes.BOOL
_user32.EnumClipboardFormats.argtypes = (wintypes.UINT,)
_user32.EnumClipboardFormats.restype = wintypes.UINT
_user32.GetClipboardData.argtypes = (wintypes.UINT,)
_user32.GetClipboardData.restype = wintypes.HANDLE
_user32.SetClipboardData.argtypes = (wintypes.UINT, wintypes.HANDLE)
_user32.SetClipboardData.restype = wintypes.HANDLE
_user32.RegisterClipboardFormatW.argtypes = (wintypes.LPCWSTR,)
_user32.RegisterClipboardFormatW.restype = wintypes.UINT
_user32.GetClipboardSequenceNumber.argtypes = ()
_user32.GetClipboardSequenceNumber.restype = wintypes.DWORD
_kernel32.GlobalAlloc.argtypes = (wintypes.UINT, ctypes.c_size_t)
_kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
_kernel32.GlobalFree.argtypes = (wintypes.HGLOBAL,)
_kernel32.GlobalFree.restype = wintypes.HGLOBAL
_kernel32.GlobalLock.argtypes = (wintypes.HGLOBAL,)
_kernel32.GlobalLock.restype = wintypes.LPVOID
_kernel32.GlobalUnlock.argtypes = (wintypes.HGLOBAL,)
_kernel32.GlobalUnlock.restype = wintypes.BOOL
_kernel32.GlobalSize.argtypes = (wintypes.HGLOBAL,)
_kernel32.GlobalSize.restype = ctypes.c_size_t

# Serializes this process's clipboard sessions; OpenClipboard is a system-wide lock and two
# of our own threads would otherwise just burn the retry budget against each other.
_lock = threading.RLock()
_exclusion_format: int | None = None


@dataclass
class ClipboardSnapshot:
    """Raw bytes per clipboard format plus the sequence number at snapshot time."""

    formats: dict[int, bytes] = field(default_factory=dict)
    sequence: int = 0


def should_snapshot_format(fmt: int) -> bool:
    """Whether a format's data is global memory we can copy and later put back."""
    if fmt in GDI_HANDLE_FORMATS:
        return False
    if fmt in _DISPLAY_FORMAT_RANGE:
        return False
    if CF_PRIVATEFIRST <= fmt <= CF_GDIOBJLAST:
        # Both ranges carry application-defined handles rather than global memory, so
        # GlobalLock on them is not safe. Private formats are never freed by the system
        # either: only the owning application knows what the handle means (decision B3-32).
        return False
    return fmt > 0


def exclusion_format() -> int:
    """The registered id of the clipboard-history exclusion format (cached)."""
    global _exclusion_format
    if _exclusion_format is None:
        fmt = _user32.RegisterClipboardFormatW(EXCLUDE_FORMAT_NAME)
        if not fmt:
            raise ctypes.WinError(ctypes.get_last_error())
        _exclusion_format = fmt
    return _exclusion_format


def sequence_number() -> int:
    return int(_user32.GetClipboardSequenceNumber())


class _Session:
    """Context manager holding the clipboard open, with retries while another app has it."""

    def __enter__(self) -> Self:
        _lock.acquire()
        try:
            for attempt in range(OPEN_TRIES):
                if _user32.OpenClipboard(None):
                    return self
                error = ctypes.get_last_error()
                if attempt + 1 < OPEN_TRIES:
                    time.sleep(OPEN_RETRY_DELAY_S)
            raise OSError(error, f"OpenClipboard failed after {OPEN_TRIES} tries (error {error})")
        except BaseException:
            _lock.release()
            raise

    def __exit__(self, *exc: object) -> None:
        try:
            _user32.CloseClipboard()
        finally:
            _lock.release()


def _read_global(handle: int) -> bytes | None:
    pointer = _kernel32.GlobalLock(handle)
    if not pointer:
        return None
    try:
        size = _kernel32.GlobalSize(handle)
        if size == 0 or size > MAX_FORMAT_BYTES:
            return None
        return ctypes.string_at(pointer, size)
    finally:
        _kernel32.GlobalUnlock(handle)


def _put_global(fmt: int, data: bytes) -> None:
    handle = _kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data))
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    pointer = _kernel32.GlobalLock(handle)
    if not pointer:
        error = ctypes.get_last_error()
        _kernel32.GlobalFree(handle)
        raise ctypes.WinError(error)
    try:
        ctypes.memmove(pointer, data, len(data))
    finally:
        _kernel32.GlobalUnlock(handle)
    if not _user32.SetClipboardData(fmt, handle):
        error = ctypes.get_last_error()
        _kernel32.GlobalFree(handle)
        raise ctypes.WinError(error)


def _put_exclusion_tag() -> None:
    _put_global(exclusion_format(), b"\x00\x00\x00\x00")


def snapshot() -> ClipboardSnapshot:
    """Copy every snapshot-able format currently on the clipboard.

    Delayed-render formats whose GetClipboardData returns NULL are skipped, as are formats
    rejected by should_snapshot_format.
    """
    formats: dict[int, bytes] = {}
    with _Session():
        sequence = sequence_number()
        fmt = _user32.EnumClipboardFormats(0)
        while fmt:
            if should_snapshot_format(fmt):
                handle = _user32.GetClipboardData(fmt)
                data = _read_global(handle) if handle else None
                if data is None:
                    logger.debug("clipboard format 0x%x skipped (no readable data)", fmt)
                else:
                    formats[fmt] = data
            fmt = _user32.EnumClipboardFormats(fmt)
    return ClipboardSnapshot(formats=formats, sequence=sequence)


def set_text(text: str) -> int:
    """Replace the clipboard with Unicode text plus the exclusion tag.

    Returns the clipboard sequence number restore() later compares against, or
    SEQUENCE_UNKNOWN when it cannot be attributed to this write. The number is read while
    the clipboard is held open again and only together with a check that our own text is
    still there, so a write by another application in the gap after CloseClipboard is
    never mistaken for ours and never overwritten by a later restore().

    The number cannot simply be read before CloseClipboard: closing is what makes Windows
    synthesize CF_TEXT, CF_OEMTEXT and CF_LOCALE from CF_UNICODETEXT, and each of those
    bumps the sequence number after the read (measured on Windows 11), so restore() would
    always find a mismatch and never put the user's clipboard back.
    """
    payload = text.encode("utf-16-le") + b"\x00\x00"
    with _Session():
        if not _user32.EmptyClipboard():
            raise ctypes.WinError(ctypes.get_last_error())
        _put_global(CF_UNICODETEXT, payload)
        _put_exclusion_tag()
    try:
        with _Session():
            if _current_text() != text:
                logger.debug("clipboard changed right after set_text; restore will be skipped")
                return SEQUENCE_UNKNOWN
            return sequence_number()
    except OSError:
        logger.debug("clipboard sequence could not be confirmed after set_text", exc_info=True)
        return SEQUENCE_UNKNOWN


def restore(snap: ClipboardSnapshot, expected_sequence: int) -> bool:
    """Put the snapshot back if nobody changed the clipboard since expected_sequence.

    Returns True when the restore happened. The clipboard is emptied and every
    snapshotted format is set again, followed by the exclusion tag so the restore itself
    creates no clipboard-history entry. A snapshot with no formats restores an empty
    clipboard: the tag is not written either, because writing it would leave a clipboard
    that reports one format where the user had none.

    expected_sequence is the value set_text() returned; SEQUENCE_UNKNOWN never matches a
    real sequence number, so a set_text() that could not confirm its own write refuses.
    """
    if sequence_number() != expected_sequence:
        return False
    with _Session():
        # Re-check while we hold the clipboard: nothing else can change it now.
        if sequence_number() != expected_sequence:
            return False
        if not _user32.EmptyClipboard():
            raise ctypes.WinError(ctypes.get_last_error())
        for fmt, data in snap.formats.items():
            try:
                _put_global(fmt, data)
            except OSError:
                logger.debug("clipboard format 0x%x could not be restored", fmt, exc_info=True)
        if snap.formats:
            _put_exclusion_tag()
    return True


def _current_text() -> str | None:
    """The clipboard's Unicode text; call with the clipboard already open."""
    handle = _user32.GetClipboardData(CF_UNICODETEXT)
    if not handle:
        return None
    pointer = _kernel32.GlobalLock(handle)
    if not pointer:
        return None
    try:
        size = _kernel32.GlobalSize(handle)
        raw = ctypes.string_at(pointer, size)
    finally:
        _kernel32.GlobalUnlock(handle)
    text = raw.decode("utf-16-le", errors="replace")
    end = text.find("\x00")
    return text if end < 0 else text[:end]


def get_text() -> str | None:
    """The clipboard's Unicode text, or None when there is none."""
    with _Session():
        return _current_text()
