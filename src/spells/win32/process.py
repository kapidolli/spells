"""Job objects and hidden child process launch for the engine supervisor.

Spec section 13 (launch, job object); batch 2 decisions V3-JOB (extended limit
information with KILL_ON_JOB_CLOSE, the job handle is never duplicated, so process death
closes the last handle and the kernel kills the engines) and V3-F1 (STATUS_DLL_NOT_FOUND).
"""

import contextlib
import ctypes
import logging
import subprocess
from ctypes import wintypes
from pathlib import Path
from typing import Self

logger = logging.getLogger(__name__)

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

STATUS_DLL_NOT_FOUND = 0xC0000135
CREATE_NO_WINDOW = 0x08000000
CREATE_SUSPENDED = 0x00000004
TH32CS_SNAPTHREAD = 0x00000004
THREAD_SUSPEND_RESUME = 0x0002
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
JOBOBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9  # JobObjectExtendedLimitInformation
PROCESS_SET_QUOTA = 0x0100
PROCESS_TERMINATE = 0x0001
PROCESS_SET_INFORMATION = 0x0200
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

# How long spawn_hidden waits for a child it had to kill, so the handle is reaped.
KILL_WAIT_S = 5.0


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class THREADENTRY32(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ThreadID", wintypes.DWORD),
        ("th32OwnerProcessID", wintypes.DWORD),
        ("tpBasePri", wintypes.LONG),
        ("tpDeltaPri", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
    ]


_kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
_kernel32.CreateJobObjectW.restype = wintypes.HANDLE
_kernel32.SetInformationJobObject.argtypes = (
    wintypes.HANDLE,
    ctypes.c_int,
    wintypes.LPVOID,
    wintypes.DWORD,
)
_kernel32.SetInformationJobObject.restype = wintypes.BOOL
_kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
_kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
_kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
_kernel32.OpenProcess.restype = wintypes.HANDLE
_kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
_kernel32.CloseHandle.restype = wintypes.BOOL
_kernel32.CreateToolhelp32Snapshot.argtypes = (wintypes.DWORD, wintypes.DWORD)
_kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
_kernel32.Thread32First.argtypes = (wintypes.HANDLE, ctypes.POINTER(THREADENTRY32))
_kernel32.Thread32First.restype = wintypes.BOOL
_kernel32.Thread32Next.argtypes = (wintypes.HANDLE, ctypes.POINTER(THREADENTRY32))
_kernel32.Thread32Next.restype = wintypes.BOOL
_kernel32.OpenThread.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
_kernel32.OpenThread.restype = wintypes.HANDLE
_kernel32.ResumeThread.argtypes = (wintypes.HANDLE,)
_kernel32.ResumeThread.restype = wintypes.DWORD
_kernel32.SetProcessAffinityMask.argtypes = (wintypes.HANDLE, ctypes.c_size_t)
_kernel32.SetProcessAffinityMask.restype = wintypes.BOOL
_INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


class JobObject:
    """An anonymous job whose processes die when its last handle closes."""

    def __init__(self) -> None:
        handle = _kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = _kernel32.SetInformationJobObject(
            handle,
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            error = ctypes.get_last_error()
            _kernel32.CloseHandle(handle)
            raise ctypes.WinError(error)
        self._handle: int | None = int(handle)

    @property
    def handle(self) -> int | None:
        return self._handle

    def assign(self, pid: int) -> None:
        """Put a running process into the job."""
        if self._handle is None:
            raise ValueError("job object is closed")
        process = _kernel32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid)
        if not process:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not _kernel32.AssignProcessToJobObject(self._handle, process):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            _kernel32.CloseHandle(process)

    def close(self) -> None:
        """Close the job handle; with KILL_ON_JOB_CLOSE this terminates its processes."""
        handle, self._handle = self._handle, None
        if handle is not None:
            _kernel32.CloseHandle(handle)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        if getattr(self, "_handle", None) is not None:
            self.close()


def resume_process(pid: int) -> int:
    """Resume every thread of a process started with CREATE_SUSPENDED; returns the count."""
    snap = _kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
    if not snap or snap == _INVALID_HANDLE_VALUE:
        raise ctypes.WinError(ctypes.get_last_error())
    resumed = 0
    try:
        entry = THREADENTRY32()
        entry.dwSize = ctypes.sizeof(THREADENTRY32)
        found = _kernel32.Thread32First(snap, ctypes.byref(entry))
        while found:
            if entry.th32OwnerProcessID == pid:
                thread = _kernel32.OpenThread(THREAD_SUSPEND_RESUME, False, entry.th32ThreadID)
                if thread:
                    try:
                        _kernel32.ResumeThread(thread)
                    finally:
                        _kernel32.CloseHandle(thread)
                    resumed += 1
            entry.dwSize = ctypes.sizeof(THREADENTRY32)
            found = _kernel32.Thread32Next(snap, ctypes.byref(entry))
    finally:
        _kernel32.CloseHandle(snap)
    if resumed == 0:
        raise OSError(f"no thread of process {pid} could be resumed")
    return resumed


def set_process_affinity(pid: int, mask: int) -> None:
    if mask <= 0:
        raise ValueError("an affinity mask needs at least one processor")
    handle = _kernel32.OpenProcess(
        PROCESS_SET_INFORMATION | PROCESS_QUERY_LIMITED_INFORMATION, False, pid
    )
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not _kernel32.SetProcessAffinityMask(handle, mask):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        _kernel32.CloseHandle(handle)


def spawn_hidden(
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    stdout_path: Path,
    stderr_path: Path,
    cwd: Path | None = None,
    job: JobObject | None = None,
    affinity_mask: int | None = None,
) -> subprocess.Popen:
    """Start a console-less child with stdout and stderr appended to files.

    With a job the child is created suspended, assigned to the job and only then resumed,
    so nothing it spawns can escape the job. If the assignment or the resume fails, for
    any reason, the child is killed before the error is raised.
    """
    flags = CREATE_NO_WINDOW | (CREATE_SUSPENDED if job is not None else 0)
    with contextlib.ExitStack() as stack:
        stdout_file = stack.enter_context(open(stdout_path, "ab"))
        if Path(stderr_path) == Path(stdout_path):
            stderr_target: object = subprocess.STDOUT
        else:
            stderr_target = stack.enter_context(open(stderr_path, "ab"))
        proc = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=stdout_file,
            stderr=stderr_target,
            env=env,
            cwd=str(cwd) if cwd is not None else None,
            creationflags=flags,
        )
    if job is not None:
        try:
            job.assign(proc.pid)
            resume_process(proc.pid)
        except BaseException:
            # Every failure counts, not just OSError: assign() raises ValueError on a closed
            # job, and anything left uncaught here would leave the child suspended, without a
            # window and owned by no job, so nothing would ever clean it up.
            proc.kill()
            with contextlib.suppress(OSError, subprocess.SubprocessError):
                proc.wait(timeout=KILL_WAIT_S)
            raise
    if affinity_mask:
        try:
            set_process_affinity(proc.pid, affinity_mask)
        except (OSError, ValueError) as exc:
            logger.warning("could not pin process %d to mask %x: %s", proc.pid, affinity_mask, exc)
    return proc
