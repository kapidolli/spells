"""Engine supervisor: launch, readiness, warm-up, health, restart, classification, idle unload.

Spec 13 and 16 with decisions V1-3, V2-9, V2-13, V3-F1, V3-F2, V3-F4, V3-F5, V3-F10, V3-SM,
V3-17, V3-JOB, V4-6, V4-11, B3-30 and B5-13 to B5-18.

Threads: start() runs one thread per engine instance; each loops over step(), which performs
one action and returns how long the thread may sleep. step() blocks only on the engine's own
HTTP requests, so tests call it directly with a fake clock. Commands from other threads
(restart, ensure_ready, set_gpu) are flags the engine thread picks up in its next step, so
status callbacks always run on a supervisor thread and never under a lock.

Identity: one speech engine instance runs per distinct speech choice (EngineId(WHISPER, slot))
plus the cleanup engine (EngineId(LLAMA)). An Engine member names a role: its status is the
worst state among the role's instances, which is what the tray and Diagnostics show.

State semantics: a state names what is serving. READY means the Vulkan build answers and
CPU_FALLBACK means the CPU build answers (the URL is exposed in exactly those two states).
STARTING and RESTARTING mean a process is being launched and the reason says why. PAUSED
(llama only) and UNLOADED mean nothing runs. FAILED means the CPU build failed as well, or,
with the reason "no_model", that no cleanup model is installed: llama is then never launched.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import socket
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from spells.datafiles import data_path, read_lines
from spells.gpu import GpuSelection, engine_env
from spells.modelcatalog import (
    LLAMA_ASR,
    LLAMA_SERVER,
    WHISPER_SERVER,
    language_name,
    resolve_extra_files,
)
from spells.models import CpuPlan, Engine, EngineId, EngineState

logger = logging.getLogger(__name__)

Variant = Literal["vulkan", "cpu"]
EngineRef = Engine | EngineId
StatusCallback = Callable[[Engine, EngineState, str], None]
InstanceStatusCallback = Callable[[EngineId, EngineState, str], None]

REASONS = (
    "ok",
    "no_vulkan_gpu",
    "oom",
    "crash",
    "dll_not_found",
    "idle",
    "no_model",
    "cpu_selected",
)
STATUS_DLL_NOT_FOUND = 0xC0000135
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_KEEP = 5
HEALTH_POLL_S = 0.2
LLAMA_CONTEXT = 4096
GPU_FOUND_MARKER = "ggml_vulkan: Found"
BIND_FAILURE_MARKERS = ("couldn't bind", "could not bind", "bind failed", "address already in use")
MAX_BIND_RETRIES = 5
IDLE_WAIT_S = 3600.0
SERVING_STATES = frozenset({EngineState.READY, EngineState.CPU_FALLBACK})
SPEECH = Engine.WHISPER
CLEANUP_ID = EngineId(Engine.LLAMA)
WRITER_ID = EngineId(Engine.LLAMA, 1)
STATE_RANK = {
    EngineState.READY: 0,
    EngineState.CPU_FALLBACK: 1,
    EngineState.UNLOADED: 2,
    EngineState.STARTING: 3,
    EngineState.PAUSED: 4,
    EngineState.RESTARTING: 5,
    EngineState.FAILED: 6,
}
BENIGN_REASONS = frozenset({"ok", "cpu_selected", "idle"})
ROLE_NAMES = {Engine.WHISPER: "Speech engine", Engine.LLAMA: "Cleanup engine"}

_FOUND_LINE = re.compile(r"ggml_vulkan: Found (\d+) Vulkan device")


def engine_id(engine: EngineRef) -> EngineId:
    return engine if isinstance(engine, EngineId) else EngineId(Engine(engine))


def engine_label(engine: EngineRef, languages: Sequence[str] = ()) -> str:
    key = engine_id(engine)
    base = "Writing engine" if key == WRITER_ID else ROLE_NAMES[key.role]
    names = [language_name(code) for code in dict.fromkeys(languages)]
    if not names:
        return base
    joined = names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"
    return f"{base} for {joined}"


@dataclass(frozen=True)
class SpeechEngine:
    runtime: str
    model: Path
    args: tuple[str, ...] = ()
    extra_files: tuple[Path, ...] = ()
    languages: tuple[str, ...] = ()


@dataclass(frozen=True)
class SpeechSlot:
    engine: EngineId
    runtime: str
    languages: tuple[str, ...]


@dataclass(frozen=True)
class EnginePaths:
    vulkan_dir: Path
    cpu_dir: Path
    whisper_model: Path
    vad_model: Path
    llama_model: Path | None
    log_dir: Path
    whisper_args: tuple[str, ...] = ()
    llama_args: tuple[str, ...] = ()
    whisper_runtime: str = WHISPER_SERVER
    whisper_extra_files: tuple[Path, ...] = ()
    whisper_languages: tuple[str, ...] = ()
    llama_extra_files: tuple[Path, ...] = ()
    llama_languages: tuple[str, ...] = ()
    extra_speech: tuple[SpeechEngine, ...] = ()
    writer_model: Path | None = None
    writer_args: tuple[str, ...] = ()
    writer_extra_files: tuple[Path, ...] = ()

    @property
    def speech(self) -> tuple[SpeechEngine, ...]:
        first = SpeechEngine(
            runtime=self.whisper_runtime,
            model=self.whisper_model,
            args=self.whisper_args,
            extra_files=self.whisper_extra_files,
            languages=self.whisper_languages,
        )
        return (first, *self.extra_speech)

    @property
    def engine_ids(self) -> tuple[EngineId, ...]:
        speech = tuple(EngineId(SPEECH, slot) for slot in range(len(self.speech)))
        writer = (WRITER_ID,) if self.writer_model is not None else ()
        return (*speech, CLEANUP_ID, *writer)

    def speech_engine(self, engine: EngineRef) -> SpeechEngine | None:
        key = engine_id(engine)
        if key.role is not SPEECH or key.slot >= len(self.speech):
            return None
        return self.speech[key.slot]

    def runtime(self, engine: EngineRef) -> str:
        spec = self.speech_engine(engine)
        return spec.runtime if spec is not None else LLAMA_SERVER

    def variant_dir(self, variant: Variant) -> Path:
        return self.vulkan_dir if variant == "vulkan" else self.cpu_dir

    def whisper_exe(self, variant: Variant) -> Path:
        return self.variant_dir(variant) / "whisper-server.exe"

    def llama_exe(self, variant: Variant) -> Path:
        return self.variant_dir(variant) / "llama-server.exe"

    def exe(self, engine: EngineRef, variant: Variant) -> Path:
        if self.runtime(engine) == WHISPER_SERVER:
            return self.whisper_exe(variant)
        return self.llama_exe(variant)

    def log_path(self, engine: EngineRef) -> Path:
        return self.log_dir / f"{engine_id(engine).key}.log"

    def with_speech(self, models_dir: Path, choices: Sequence[Any]) -> EnginePaths:
        if not choices:
            return self
        specs = [
            SpeechEngine(
                runtime=choice.runtime or WHISPER_SERVER,
                model=Path(models_dir) / choice.file,
                args=tuple(choice.engine_args),
                extra_files=tuple(Path(models_dir) / name for name in choice.extra_files),
                languages=tuple(sorted(choice.languages)),
            )
            for choice in choices
        ]
        first = specs[0]
        return replace(
            self,
            whisper_model=first.model,
            whisper_args=first.args,
            whisper_runtime=first.runtime,
            whisper_extra_files=first.extra_files,
            whisper_languages=first.languages,
            extra_speech=tuple(specs[1:]),
        )

    def with_writer(self, models_dir: Path, choice: Any | None) -> EnginePaths:
        if choice is None:
            return replace(self, writer_model=None, writer_args=(), writer_extra_files=())
        return replace(
            self,
            writer_model=Path(models_dir) / choice.file,
            writer_args=tuple(choice.engine_args),
            writer_extra_files=tuple(Path(models_dir) / name for name in choice.extra_files),
        )

    def text_model(self, engine: EngineRef) -> tuple[Path | None, tuple[str, ...], tuple[Path, ...]]:
        if engine_id(engine) == WRITER_ID:
            return self.writer_model, self.writer_args, self.writer_extra_files
        return self.llama_model, self.llama_args, self.llama_extra_files

    def with_cleanup(self, models_dir: Path, choice: Any | None) -> EnginePaths:
        if choice is None:
            return replace(
                self, llama_model=None, llama_args=(), llama_extra_files=(), llama_languages=()
            )
        return replace(
            self,
            llama_model=Path(models_dir) / choice.file,
            llama_args=tuple(choice.engine_args),
            llama_extra_files=tuple(Path(models_dir) / name for name in choice.extra_files),
            llama_languages=tuple(sorted(choice.languages)),
        )


def _launch_key(paths: EnginePaths, cpu_only: bool, engine: EngineRef) -> tuple:
    dirs = (paths.vulkan_dir, paths.cpu_dir, cpu_only)
    key = engine_id(engine)
    if key.role is SPEECH:
        spec = paths.speech_engine(key)
        if spec is None:
            return (*dirs, None)
        vad = paths.vad_model if spec.runtime == WHISPER_SERVER else None
        return (*dirs, spec.runtime, spec.model, spec.args, spec.extra_files, vad)
    return (*dirs, *paths.text_model(key))


def launch_args(
    paths: EnginePaths,
    engine: EngineRef,
    variant: Variant,
    port: int,
    *,
    whisper_threads: int,
    cpu_plan: CpuPlan,
) -> list[str]:
    key = engine_id(engine)
    exe = str(paths.exe(key, variant))
    spec = paths.speech_engine(key)
    if spec is not None and spec.runtime == WHISPER_SERVER:
        return [
            exe,
            "-m",
            str(spec.model),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--vad",
            "--vad-model",
            str(paths.vad_model),
            "-t",
            str(whisper_threads),
            *resolve_extra_files(spec.args, spec.extra_files),
        ]
    if spec is not None:
        model, extra = spec.model, resolve_extra_files(spec.args, spec.extra_files)
    else:
        model, args, extra_files = paths.text_model(key)
        extra = resolve_extra_files(args, extra_files)
    return [
        exe,
        "-m",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-c",
        str(LLAMA_CONTEXT),
        "-ngl",
        "99" if variant == "vulkan" else "0",
        *llama_cpu_args(cpu_plan),
        *extra,
    ]


def llama_cpu_args(plan: CpuPlan) -> list[str]:
    args: list[str] = []
    if plan.threads:
        args += ["-t", str(plan.threads)]
    if plan.affinity_mask:
        args += ["-C", plan.cpu_mask_hex or "", "--cpu-strict", "1"]
    return args


@lru_cache(maxsize=1)
def no_device_patterns() -> tuple[str, ...]:
    return tuple(read_lines("engine_patterns/no_device.txt"))


@lru_cache(maxsize=1)
def oom_patterns() -> tuple[str, ...]:
    return tuple(read_lines("engine_patterns/oom.txt"))


def classify_failure(exit_code: int | None, log: str, startup_log: str | None = None) -> str:
    """One of "dll_not_found", "no_vulkan_gpu" (class a), "oom" (class b), "crash" (class c).

    The no-device patterns are matched against startup_log only (what the process wrote before
    it first answered /health; log itself when None): whisper-server prints "no GPU found" for
    its CPU-side VAD context on the first request of a perfectly healthy Vulkan run. The OOM
    patterns apply to the whole log because an OOM can happen at any time.
    """
    if exit_code is not None and exit_code & 0xFFFFFFFF == STATUS_DLL_NOT_FOUND:
        return "dll_not_found"
    if _matches(log if startup_log is None else startup_log, no_device_patterns()):
        return "no_vulkan_gpu"
    if _matches(log, oom_patterns()):
        return "oom"
    return "crash"


def gpu_verified_from_log(log: str, gpu_name: str) -> bool | None:
    """Whether the device block after "ggml_vulkan: Found N Vulkan devices:" names gpu_name.

    None when the log has no such line yet (CPU build, or the backend has not printed it).
    """
    lines = log.splitlines()
    for index, line in enumerate(lines):
        if GPU_FOUND_MARKER not in line:
            continue
        match = _FOUND_LINE.search(line)
        count = int(match.group(1)) if match else 8
        block = "\n".join(lines[index : index + 1 + count])
        return bool(gpu_name) and gpu_name in block
    return None


def _matches(text: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in text for pattern in patterns)


@lru_cache(maxsize=1)
def _warmup_wav() -> bytes:
    return data_path("warmup.wav").read_bytes()


def _default_whisper_client(base_url: str) -> Any:
    from spells.asr import WhisperClient

    return WhisperClient(base_url, timeout_s=60.0)


def _default_llama_client(base_url: str) -> Any:
    from spells.cleanup import LlamaClient

    return LlamaClient(base_url, timeout_s=60.0)


def _default_llama_asr_client(base_url: str) -> Any:
    from spells.asr import LlamaAsrClient

    return LlamaAsrClient(base_url, timeout_s=60.0)


class _Win32ProcessBackend:
    def __init__(self) -> None:
        from spells.win32 import process as win32_process

        self._process = win32_process

    def create_job(self) -> Any:
        return self._process.JobObject()

    def spawn(self, args, *, env, stdout_path, stderr_path, cwd, job, affinity_mask=None) -> Any:
        return self._process.spawn_hidden(
            args,
            env=env,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            cwd=cwd,
            job=job,
            affinity_mask=affinity_mask,
        )


@dataclass
class _Proc:
    proc: Any
    variant: Variant
    port: int
    url: str
    client: Any
    log_offset: int
    launched_at: float
    next_poll_at: float
    ready: bool = False
    gpu_verified: bool | None = None
    startup_end: int | None = None

    @property
    def alive(self) -> bool:
        return self.proc.poll() is None


class _Runtime:
    def __init__(self, engine: EngineId) -> None:
        self.engine = engine
        self.state = EngineState.STARTING
        self.reason = "ok"
        self.phase = "launch"
        self.want_variant: Variant = "vulkan"
        self.fallback_reason = "ok"
        self.current: _Proc | None = None
        self.trial: _Proc | None = None
        self.consecutive_failures = 0
        self.bind_retries = 0
        self.retry_interval: float | None = None
        self.retry_at: float | None = None
        self.resume_at = 0.0
        self.next_health_at = 0.0
        self.gpu_verified: bool | None = None
        self.command: str | None = None
        self.reconfigure = False
        self.retired = False
        self.wake = threading.Event()
        self.pending: list[tuple[EngineState, str]] = []
        self.thread: threading.Thread | None = None

    @property
    def role(self) -> Engine:
        return self.engine.role

    @property
    def key(self) -> str:
        return self.engine.key


class EngineSupervisor:
    """Supervises the speech engines and the cleanup engine (spec 13)."""

    def __init__(
        self,
        paths: EnginePaths,
        gpu: GpuSelection,
        on_status: StatusCallback,
        *,
        idle_unload_minutes: float = 0,
        process_backend: Any = None,
        whisper_client_factory: Callable[[str], Any] | None = None,
        llama_client_factory: Callable[[str], Any] | None = None,
        llama_asr_client_factory: Callable[[str], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        health_interval_s: float = 10.0,
        backoff_s: tuple[float, ...] = (1, 2, 5, 10, 30),
        oom_retry_s: float = 60.0,
        cpu_retry_s: float = 600.0,
        max_consecutive_failures: int = 3,
        startup_timeout_s: float = 180.0,
        whisper_threads: int | None = None,
        cpu_only: bool = False,
        cpu_cleanup_allowed: bool = False,
        cpu_plan: CpuPlan | None = None,
        on_instance_status: InstanceStatusCallback | None = None,
    ) -> None:
        self._paths = paths
        self._cpu_only = bool(cpu_only)
        self._cpu_cleanup_allowed = bool(cpu_cleanup_allowed)
        self._gpu = gpu
        self._on_status = on_status
        self._on_instance_status = on_instance_status
        self._idle_s = float(idle_unload_minutes) * 60.0
        self._backend = process_backend
        self._whisper_factory = whisper_client_factory or _default_whisper_client
        self._llama_factory = llama_client_factory or _default_llama_client
        self._llama_asr_factory = llama_asr_client_factory or _default_llama_asr_client
        self._clock = clock
        self._sleeper = sleeper
        self._health_interval_s = health_interval_s
        self._backoff_s = tuple(backoff_s) or (1,)
        self._oom_retry_s = oom_retry_s
        self._cpu_retry_s = cpu_retry_s
        self._max_failures = max(1, int(max_consecutive_failures))
        self._startup_timeout_s = startup_timeout_s
        self._cpu_plan = cpu_plan or CpuPlan()
        self._whisper_threads = (
            whisper_threads
            or self._cpu_plan.threads
            or max(1, min(8, (os.cpu_count() or 4) // 2))
        )
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._stopping = threading.Event()
        self._started = False
        self._job: Any = None
        self._runtimes: dict[EngineId, _Runtime] = {
            engine: _Runtime(engine) for engine in paths.engine_ids
        }
        self._retired_threads: list[threading.Thread] = []
        self._role_state: dict[Engine, tuple[EngineState, str]] = {
            role: (EngineState.STARTING, "ok") for role in Engine
        }
        self._role_events: list[tuple[Engine, EngineState, str]] = []
        self._delivering = False
        self._last_activity = clock()
        for runtime in self._runtimes.values():
            if self._has_no_model(runtime):
                self._mark_no_model(runtime)

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._stopping.clear()
        self.note_activity()
        for runtime in self._runtime_list():
            self._start_thread(runtime)

    def stop(self) -> None:
        self._stopping.set()
        runtimes = self._runtime_list()
        for runtime in runtimes:
            runtime.wake.set()
        self._kill_all()
        me = threading.current_thread()
        with self._lock:
            threads = [runtime.thread for runtime in runtimes] + list(self._retired_threads)
        for thread in threads:
            if thread is not None and thread is not me and thread.is_alive():
                thread.join(timeout=30.0)
        self._kill_all()
        with self._lock:
            job, self._job = self._job, None
            for runtime in self._runtimes.values():
                runtime.current = None
                runtime.trial = None
            self._cond.notify_all()
        if job is not None:
            job.close()

    @property
    def engine_ids(self) -> tuple[EngineId, ...]:
        with self._lock:
            return tuple(self._runtimes)

    def speech_engines(self) -> tuple[SpeechSlot, ...]:
        with self._lock:
            paths = self._paths
            present = [engine for engine in self._runtimes if engine.role is SPEECH]
        slots = []
        for engine in present:
            spec = paths.speech_engine(engine)
            if spec is not None:
                slots.append(SpeechSlot(engine, spec.runtime, spec.languages))
        return tuple(slots)

    def status(self, engine: EngineRef) -> EngineState:
        with self._lock:
            return self._view_locked(engine)[0]

    def reason(self, engine: EngineRef) -> str:
        with self._lock:
            return self._view_locked(engine)[1]

    def url(self, engine: EngineRef) -> str | None:
        return self._url(engine_id(engine))

    @property
    def whisper_url(self) -> str | None:
        return self._url(EngineId(SPEECH))

    @property
    def llama_url(self) -> str | None:
        return self._url(CLEANUP_ID)

    @property
    def writer_engine(self) -> EngineId | None:
        with self._lock:
            return WRITER_ID if WRITER_ID in self._runtimes else None

    @property
    def cleanup_available(self) -> bool:
        with self._lock:
            state = self._runtimes[CLEANUP_ID].state
            planned = self._cpu_cleanup_allowed
        return state is EngineState.READY or (state is EngineState.CPU_FALLBACK and planned)

    @property
    def cpu_cleanup_allowed(self) -> bool:
        with self._lock:
            return self._cpu_cleanup_allowed

    @property
    def cpu_only(self) -> bool:
        with self._lock:
            return self._cpu_only

    @property
    def cleanup_languages(self) -> frozenset[str] | None:
        with self._lock:
            languages = self._paths.llama_languages
        return frozenset(languages) if languages else None

    @property
    def cpu_plan(self) -> CpuPlan:
        return self._cpu_plan

    def set_models(
        self, paths: EnginePaths, *, cpu_only: bool, cpu_cleanup_allowed: bool,
        gpu: GpuSelection | None = None,
    ) -> tuple[EngineId, ...]:
        wanted = paths.engine_ids
        with self._lock:
            old_paths, old_cpu_only = self._paths, self._cpu_only
            gpu_changed = gpu is not None and (
                gpu.raw_index, gpu.name
            ) != (self._gpu.raw_index, self._gpu.name)
            if gpu is not None:
                self._gpu = gpu
            self._paths = paths
            self._cpu_only = bool(cpu_only)
            self._cpu_cleanup_allowed = bool(cpu_cleanup_allowed)
            retired = [runtime for engine, runtime in self._runtimes.items() if engine not in wanted]
            for runtime in retired:
                runtime.retired = True
                del self._runtimes[runtime.engine]
                if runtime.thread is not None:
                    self._retired_threads.append(runtime.thread)
            added = [_Runtime(engine) for engine in wanted if engine not in self._runtimes]
            for runtime in added:
                self._runtimes[runtime.engine] = runtime
            changed = [
                runtime
                for engine, runtime in self._runtimes.items()
                if runtime not in added
                and (gpu_changed or _launch_key(old_paths, old_cpu_only, engine)
                     != _launch_key(paths, self._cpu_only, engine))
            ]
            for runtime in changed:
                runtime.reconfigure = True
            self._runtimes = {engine: self._runtimes[engine] for engine in wanted}
            for role in Engine:
                self._queue_role_locked(role)
            started = self._started and not self._stopping.is_set()
        for runtime in retired:
            logger.info("engine %s: no longer needed, stopping", runtime.key)
            for proc in (runtime.current, runtime.trial):
                if proc is not None:
                    self._kill(proc)
            runtime.current = runtime.trial = None
            runtime.wake.set()
        for runtime in added:
            logger.info("engine %s: added for the new models", runtime.key)
            if started:
                self._start_thread(runtime)
        for runtime in changed:
            logger.info("engine %s: models changed, relaunching", runtime.key)
        for runtime in self._runtime_list():
            runtime.wake.set()
        return (
            *(runtime.engine for runtime in changed),
            *(runtime.engine for runtime in added),
            *(runtime.engine for runtime in retired),
        )

    def variant(self, engine: EngineRef) -> Variant:
        with self._lock:
            members = self._members_locked(engine)
            variants = [
                runtime.current.variant if runtime.current else runtime.want_variant
                for runtime in members
            ]
        if not variants:
            return "cpu"
        return "cpu" if "cpu" in variants else "vulkan"

    def pid(self, engine: EngineRef) -> int | None:
        with self._lock:
            runtime = self._runtimes.get(engine_id(engine))
            current = runtime.current if runtime is not None else None
        if current is None or not current.alive:
            return None
        return current.proc.pid

    def gpu_verified(self, engine: EngineRef) -> bool | None:
        with self._lock:
            values = [runtime.gpu_verified for runtime in self._members_locked(engine)]
        if not values or any(value is False for value in values):
            return False if values else None
        if any(value is None for value in values):
            return None
        return True

    def ensure_ready(self) -> None:
        self.note_activity()
        for runtime in self._runtime_list():
            with self._lock:
                if runtime.state is EngineState.UNLOADED:
                    runtime.command = "ensure_ready"
            runtime.wake.set()

    def restart(self, engine: EngineRef) -> None:
        with self._lock:
            members = self._members_locked(engine)
        for runtime in members:
            self._request(runtime, "restart")

    def set_gpu(self, selection: GpuSelection) -> None:
        with self._lock:
            self._gpu = selection
        for runtime in self._runtime_list():
            self._request(runtime, "restart")

    def note_activity(self) -> None:
        with self._lock:
            self._last_activity = self._clock()

    def set_idle_unload_minutes(self, minutes: float) -> None:
        seconds = max(0.0, float(minutes)) * 60.0
        with self._lock:
            if seconds == self._idle_s:
                return
            self._idle_s = seconds
        for runtime in self._runtime_list():
            runtime.wake.set()

    def wait_ready(self, engine: EngineRef, timeout_s: float) -> bool:
        def serving() -> bool:
            members = self._members_locked(engine)
            return bool(members) and all(runtime.state in SERVING_STATES for runtime in members)

        with self._cond:
            self._cond.wait_for(lambda: serving() or self._stopping.is_set(), timeout_s)
            return serving()

    def step(self, engine: EngineRef) -> float:
        with self._lock:
            runtime = self._runtimes.get(engine_id(engine))
        if runtime is None:
            return IDLE_WAIT_S
        return self._step_runtime(runtime)

    def _step_runtime(self, runtime: _Runtime) -> float:
        try:
            if self._stopping.is_set() or runtime.retired:
                return IDLE_WAIT_S
            with self._lock:
                command, runtime.command = runtime.command, None
                reconfigure, runtime.reconfigure = runtime.reconfigure, False
            unloaded = runtime.phase == "unloaded"
            if (
                command == "restart"
                or (command == "ensure_ready" and unloaded)
                or (reconfigure and not unloaded)
            ):
                self._begin_fresh(runtime)
            return self._dispatch(runtime)
        finally:
            self._flush_callbacks(runtime)

    def _start_thread(self, runtime: _Runtime) -> None:
        runtime.thread = threading.Thread(
            target=self._run, args=(runtime,), name=f"engine-{runtime.key}", daemon=True
        )
        runtime.thread.start()

    def _run(self, runtime: _Runtime) -> None:
        while not self._stopping.is_set() and not runtime.retired:
            try:
                delay = self._step_runtime(runtime)
            except Exception:
                logger.exception("engine %s: supervision step failed", runtime.key)
                delay = 1.0
            if delay > 0:
                self._wait(runtime, delay)

    def _wait(self, runtime: _Runtime, seconds: float) -> None:
        deadline = self._clock() + seconds
        while not self._stopping.is_set() and not runtime.retired:
            if runtime.wake.is_set():
                runtime.wake.clear()
                return
            remaining = deadline - self._clock()
            if remaining <= 0:
                return
            self._sleeper(min(remaining, 0.1))

    def _dispatch(self, runtime: _Runtime) -> float:
        phase = runtime.phase
        now = self._clock()
        if phase == "launch":
            self._launch_current(runtime)
            return 0.0
        if phase == "starting":
            return self._advance_starting(runtime)
        if phase == "serving":
            return self._advance_serving(runtime)
        if phase == "backoff":
            if now >= runtime.resume_at:
                runtime.phase = "launch"
                return 0.0
            return runtime.resume_at - now
        if phase == "paused":
            if runtime.retry_at is not None and now >= runtime.retry_at:
                runtime.retry_at = None
                runtime.want_variant = "vulkan"
                runtime.phase = "launch"
                return 0.0
            return IDLE_WAIT_S if runtime.retry_at is None else runtime.retry_at - now
        return IDLE_WAIT_S

    def _launch_current(self, runtime: _Runtime) -> None:
        if self._stopping.is_set() or runtime.retired:
            return
        if self._has_no_model(runtime):
            self._mark_no_model(runtime)
            return
        variant = runtime.want_variant
        if variant == "vulkan" and (self._gpu.raw_index is None or self._cpu_only):
            variant = "cpu"
            reason = "no_vulkan_gpu" if self._gpu.raw_index is None else "cpu_selected"
            runtime.fallback_reason = reason
            runtime.retry_interval = None
            state = runtime.state
            if state not in (EngineState.STARTING, EngineState.RESTARTING):
                state = EngineState.RESTARTING
            self._set_state(runtime, state, reason)
        try:
            proc = self._spawn(runtime, variant)
        except (OSError, ValueError) as exc:
            logger.warning("engine %s: launch failed: %s", runtime.key, exc)
            self._on_classified(runtime, variant, "crash", is_trial=False)
            return
        if runtime.retired:
            self._kill(proc)
            return
        runtime.current = proc
        runtime.phase = "starting"

    def _launch_trial(self, runtime: _Runtime) -> None:
        runtime.retry_at = None
        try:
            proc = self._spawn(runtime, "vulkan")
        except (OSError, ValueError) as exc:
            logger.warning("engine %s: Vulkan retry launch failed: %s", runtime.key, exc)
            self._on_trial_failure(runtime, "crash")
            return
        if runtime.retired:
            self._kill(proc)
            return
        runtime.trial = proc

    def _spawn(self, runtime: _Runtime, variant: Variant) -> _Proc:
        engine = runtime.engine
        port = self._free_port()
        with self._lock:
            paths = self._paths
            env = engine_env(self._gpu)
        exe = paths.exe(engine, variant)
        args = self._args(engine, variant, port, paths)
        log = paths.log_path(engine)
        log.parent.mkdir(parents=True, exist_ok=True)
        self._rotate(log)
        offset = log.stat().st_size if log.exists() else 0
        job = self._ensure_job()
        runtime_name = paths.runtime(engine)
        logger.info(
            "engine %s: launching %s %s build on port %d", engine.key, runtime_name, variant, port
        )
        proc = self._backend_instance().spawn(
            args,
            env=env,
            stdout_path=log,
            stderr_path=log,
            cwd=exe.parent,
            job=job,
            affinity_mask=self._cpu_plan.affinity_mask,
        )
        url = f"http://127.0.0.1:{port}"
        now = self._clock()
        return _Proc(proc, variant, port, url, self._client(runtime_name, url), offset, now, now)

    def _client(self, runtime_name: str, url: str) -> Any:
        if runtime_name == WHISPER_SERVER:
            return self._whisper_factory(url)
        if runtime_name == LLAMA_ASR:
            return self._llama_asr_factory(url)
        return self._llama_factory(url)

    def _args(
        self, engine: EngineRef, variant: Variant, port: int, paths: EnginePaths
    ) -> list[str]:
        return launch_args(
            paths,
            engine,
            variant,
            port,
            whisper_threads=self._whisper_threads,
            cpu_plan=self._cpu_plan,
        )

    def _backend_instance(self) -> Any:
        if self._backend is None:
            self._backend = _Win32ProcessBackend()
        return self._backend

    def _ensure_job(self) -> Any:
        with self._lock:
            if self._job is None:
                self._job = self._backend_instance().create_job()
            return self._job

    def _free_port(self) -> int:
        with self._lock:
            used = {
                proc.port
                for runtime in self._runtimes.values()
                for proc in (runtime.current, runtime.trial)
                if proc is not None
            }
        port = 0
        for _ in range(20):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            if port not in used:
                break
        return port

    def _rotate(self, log: Path) -> None:
        try:
            if not log.exists() or log.stat().st_size < LOG_MAX_BYTES:
                return
            oldest = log.with_name(f"{log.name}.{LOG_KEEP - 1}")
            if oldest.exists():
                oldest.unlink()
            for index in range(LOG_KEEP - 2, 0, -1):
                source = log.with_name(f"{log.name}.{index}")
                if source.exists():
                    source.rename(log.with_name(f"{log.name}.{index + 1}"))
            log.rename(log.with_name(f"{log.name}.1"))
        except OSError as exc:
            logger.warning("could not rotate %s: %s", log, exc)

    def _advance_starting(self, runtime: _Runtime) -> float:
        proc = runtime.current
        if proc is None:
            runtime.phase = "launch"
            return 0.0
        outcome = self._poll_startup(runtime, proc, is_trial=False)
        if outcome == "ready":
            self._became_ready(runtime, proc)
            return 0.0
        if outcome != "waiting":
            return 0.0
        return max(0.0, proc.next_poll_at - self._clock())

    def _poll_startup(self, runtime: _Runtime, proc: _Proc, *, is_trial: bool) -> str:
        engine = runtime.engine
        now = self._clock()
        exit_code = proc.proc.poll()
        log = self._read_log(engine, proc.log_offset)
        if exit_code is not None:
            if self._is_bind_failure(log) and runtime.bind_retries < MAX_BIND_RETRIES:
                runtime.bind_retries += 1
                logger.warning("engine %s: port %d refused, relaunching", engine.key, proc.port)
                self._relaunch_same(runtime, proc, is_trial)
                return "failed"
            self._on_process_failure(runtime, proc, exit_code, log, is_trial)
            return "failed"
        if proc.variant == "vulkan" and _matches(log, no_device_patterns()):
            self._kill(proc)
            self._on_classified(runtime, proc.variant, "no_vulkan_gpu", is_trial=is_trial)
            return "failed"
        if now >= proc.next_poll_at:
            proc.next_poll_at = now + HEALTH_POLL_S
            if proc.client.health():
                proc.startup_end = self._log_size(engine)
                startup_log = self._startup_log(runtime, proc)
                if proc.variant == "vulkan":
                    if _matches(startup_log, no_device_patterns()):
                        self._kill(proc)
                        self._on_classified(
                            runtime, proc.variant, "no_vulkan_gpu", is_trial=is_trial
                        )
                        return "failed"
                    proc.gpu_verified = gpu_verified_from_log(startup_log, self._gpu.name)
                try:
                    self._warm_up(runtime, proc)
                except Exception:
                    logger.exception("engine %s: warm-up failed", engine.key)
                    self._kill(proc)
                    log = self._read_log(engine, proc.log_offset)
                    self._on_process_failure(runtime, proc, None, log, is_trial)
                    return "failed"
                proc.ready = True
                return "ready"
        if now - proc.launched_at >= self._startup_timeout_s:
            logger.warning(
                "engine %s: not healthy after %.0f s", engine.key, now - proc.launched_at
            )
            self._kill(proc)
            self._on_classified(runtime, proc.variant, "crash", is_trial=is_trial)
            return "failed"
        return "waiting"

    def _relaunch_same(self, runtime: _Runtime, proc: _Proc, is_trial: bool) -> None:
        if is_trial:
            runtime.trial = None
            self._launch_trial(runtime)
            return
        runtime.current = None
        runtime.want_variant = proc.variant
        runtime.phase = "launch"

    def _warm_up(self, runtime: _Runtime, proc: _Proc) -> None:
        with self._lock:
            runtime_name = self._paths.runtime(runtime.engine)
        if runtime_name == WHISPER_SERVER:
            proc.client.inference(_warmup_wav(), language=None, prompt="")
            return
        if runtime_name == LLAMA_ASR:
            proc.client.transcribe(_warmup_wav())
            return
        from spells.cleanup import warmup_messages

        system, user = warmup_messages()
        proc.client.chat(system, user, 1)

    def _became_ready(self, runtime: _Runtime, proc: _Proc) -> None:
        now = self._clock()
        runtime.consecutive_failures = 0
        runtime.bind_retries = 0
        runtime.phase = "serving"
        runtime.next_health_at = now + self._health_interval_s
        with self._lock:
            runtime.gpu_verified = proc.gpu_verified
        if proc.variant == "vulkan":
            runtime.retry_at = None
            runtime.retry_interval = None
            runtime.fallback_reason = "ok"
            self._set_state(runtime, EngineState.READY, "ok")
        else:
            runtime.retry_at = (
                now + runtime.retry_interval if runtime.retry_interval is not None else None
            )
            self._set_state(runtime, EngineState.CPU_FALLBACK, runtime.fallback_reason)
        logger.info("engine %s: ready on %s build at %s", runtime.key, proc.variant, proc.url)

    def _advance_serving(self, runtime: _Runtime) -> float:
        proc = runtime.current
        if proc is None:
            runtime.phase = "launch"
            return 0.0
        engine = runtime.engine
        now = self._clock()
        exit_code = proc.proc.poll()
        if exit_code is not None:
            self._on_process_failure(
                runtime, proc, exit_code, self._read_log(engine, proc.log_offset), False
            )
            return 0.0
        with self._lock:
            idle_for = now - self._last_activity
        if self._idle_s > 0 and idle_for >= self._idle_s:
            self._unload(runtime)
            return 0.0
        if now >= runtime.next_health_at:
            runtime.next_health_at = now + self._health_interval_s
            if not proc.client.health():
                logger.warning("engine %s: health check failed", engine.key)
                self._kill(proc)
                log = self._read_log(engine, proc.log_offset)
                self._on_process_failure(runtime, proc, None, log, False)
                return 0.0
            if proc.variant == "vulkan" and proc.gpu_verified is None:
                log = self._read_log(engine, proc.log_offset)
                proc.gpu_verified = gpu_verified_from_log(log, self._gpu.name)
                with self._lock:
                    runtime.gpu_verified = proc.gpu_verified
        if runtime.trial is not None:
            outcome = self._poll_startup(runtime, runtime.trial, is_trial=True)
            if outcome == "ready":
                self._switch_to_trial(runtime)
                return 0.0
            if outcome != "waiting":
                return 0.0
        elif runtime.retry_at is not None and now >= runtime.retry_at:
            self._launch_trial(runtime)
            return 0.0
        delays = [runtime.next_health_at - now]
        if runtime.trial is not None:
            delays.append(runtime.trial.next_poll_at - now)
        if runtime.retry_at is not None:
            delays.append(runtime.retry_at - now)
        if self._idle_s > 0:
            delays.append(self._idle_s - idle_for)
        return max(0.0, min(delays))

    def _switch_to_trial(self, runtime: _Runtime) -> None:
        old = runtime.current
        with self._lock:
            runtime.current, runtime.trial = runtime.trial, None
        self._became_ready(runtime, runtime.current)
        if old is not None:
            self._kill(old)

    def _unload(self, runtime: _Runtime) -> None:
        with self._lock:
            procs = [runtime.current, runtime.trial]
            runtime.current = runtime.trial = None
        for proc in procs:
            if proc is not None:
                self._kill(proc)
        runtime.retry_at = None
        runtime.phase = "unloaded"
        self._set_state(runtime, EngineState.UNLOADED, "idle")

    def _on_process_failure(
        self, runtime: _Runtime, proc: _Proc, exit_code: int | None, log: str, is_trial: bool
    ) -> None:
        classification = classify_failure(exit_code, log, self._startup_log(runtime, proc))
        logger.warning(
            "engine %s: %s build failed (exit %s): %s",
            runtime.key,
            proc.variant,
            exit_code,
            classification,
        )
        self._kill(proc)
        self._on_classified(runtime, proc.variant, classification, is_trial=is_trial)

    def _on_classified(
        self, runtime: _Runtime, variant: Variant, classification: str, *, is_trial: bool
    ) -> None:
        if is_trial:
            with self._lock:
                runtime.trial = None
            self._on_trial_failure(runtime, classification)
            return
        with self._lock:
            runtime.current = None
        if self._stopping.is_set() or runtime.retired:
            return
        if variant == "cpu":
            reason = "dll_not_found" if classification == "dll_not_found" else "crash"
            runtime.phase = "failed"
            self._set_state(runtime, EngineState.FAILED, reason)
            return
        now = self._clock()
        if classification in ("no_vulkan_gpu", "dll_not_found"):
            self._fall_back(runtime, classification, retry_interval=None)
        elif classification == "oom":
            if runtime.role is SPEECH:
                self._fall_back(runtime, "oom", retry_interval=self._oom_retry_s)
            else:
                runtime.retry_at = now + self._oom_retry_s
                runtime.want_variant = "vulkan"
                runtime.phase = "paused"
                self._set_state(runtime, EngineState.PAUSED, "oom")
        else:
            runtime.consecutive_failures += 1
            if runtime.consecutive_failures >= self._max_failures:
                self._fall_back(runtime, "crash", retry_interval=self._cpu_retry_s)
                return
            index = min(runtime.consecutive_failures - 1, len(self._backoff_s) - 1)
            runtime.resume_at = now + self._backoff_s[index]
            runtime.want_variant = "vulkan"
            runtime.phase = "backoff"
            self._set_state(runtime, EngineState.RESTARTING, "crash")

    def _fall_back(self, runtime: _Runtime, reason: str, *, retry_interval: float | None) -> None:
        runtime.fallback_reason = reason
        runtime.retry_interval = retry_interval
        runtime.retry_at = None
        runtime.want_variant = "cpu"
        runtime.phase = "launch"
        self._set_state(runtime, EngineState.RESTARTING, reason)

    def _on_trial_failure(self, runtime: _Runtime, classification: str) -> None:
        now = self._clock()
        if classification in ("no_vulkan_gpu", "dll_not_found"):
            runtime.retry_interval = None
            runtime.retry_at = None
        elif classification == "oom":
            runtime.retry_interval = self._oom_retry_s
            runtime.retry_at = now + self._oom_retry_s
        else:
            runtime.retry_interval = self._cpu_retry_s
            runtime.retry_at = now + self._cpu_retry_s
        runtime.fallback_reason = classification
        self._set_state(runtime, EngineState.CPU_FALLBACK, classification)

    def _begin_fresh(self, runtime: _Runtime) -> None:
        with self._lock:
            procs = [runtime.current, runtime.trial]
            runtime.current = runtime.trial = None
            runtime.gpu_verified = None
        for proc in procs:
            if proc is not None:
                self._kill(proc)
        if self._has_no_model(runtime):
            self._mark_no_model(runtime)
            return
        runtime.consecutive_failures = 0
        runtime.bind_retries = 0
        runtime.retry_interval = None
        runtime.retry_at = None
        runtime.fallback_reason = "ok"
        runtime.want_variant = "vulkan"
        runtime.phase = "launch"
        self._set_state(runtime, EngineState.STARTING, "ok")

    def _has_no_model(self, runtime: _Runtime) -> bool:
        if runtime.role is not Engine.LLAMA:
            return False
        return self._paths.text_model(runtime.engine)[0] is None

    def _mark_no_model(self, runtime: _Runtime) -> None:
        runtime.phase = "failed"
        runtime.retry_at = None
        self._set_state(runtime, EngineState.FAILED, "no_model")

    def _runtime_list(self) -> list[_Runtime]:
        with self._lock:
            return list(self._runtimes.values())

    def _members_locked(self, engine: EngineRef) -> list[_Runtime]:
        if isinstance(engine, EngineId):
            runtime = self._runtimes.get(engine)
            return [runtime] if runtime is not None else []
        role = Engine(engine)
        return [runtime for runtime in self._runtimes.values() if runtime.role is role]

    def _view_locked(self, engine: EngineRef) -> tuple[EngineState, str]:
        members = self._members_locked(engine)
        if not members:
            return EngineState.FAILED, "no_model"
        worst = max(
            members,
            key=lambda runtime: (
                STATE_RANK[runtime.state],
                runtime.reason not in BENIGN_REASONS,
                -runtime.engine.slot,
            ),
        )
        return worst.state, worst.reason

    def _queue_role_locked(self, role: Engine) -> None:
        view = self._view_locked(role)
        if self._role_state.get(role) != view:
            self._role_state[role] = view
            self._role_events.append((role, *view))

    def _url(self, engine: EngineId) -> str | None:
        with self._lock:
            runtime = self._runtimes.get(engine)
            if runtime is None:
                return None
            current = runtime.current
            if runtime.state in SERVING_STATES and current is not None and current.ready:
                return current.url
            return None

    def _request(self, runtime: _Runtime, command: str) -> None:
        with self._lock:
            runtime.command = command
        runtime.wake.set()

    def _set_state(self, runtime: _Runtime, state: EngineState, reason: str) -> None:
        with self._lock:
            if (runtime.state, runtime.reason) == (state, reason):
                return
            runtime.state, runtime.reason = state, reason
            runtime.pending.append((state, reason))
            if not runtime.retired:
                self._queue_role_locked(runtime.role)
            self._cond.notify_all()

    def _flush_callbacks(self, runtime: _Runtime) -> None:
        with self._lock:
            pending, runtime.pending = runtime.pending, []
        if self._on_instance_status is not None:
            for state, reason in pending:
                try:
                    self._on_instance_status(runtime.engine, state, reason)
                except Exception:
                    logger.exception("engine %s: instance status callback failed", runtime.key)
        self._deliver_role_events()

    def _deliver_role_events(self) -> None:
        with self._lock:
            if self._delivering:
                return
            self._delivering = True
        try:
            while True:
                with self._lock:
                    if not self._role_events:
                        self._delivering = False
                        return
                    role, state, reason = self._role_events.pop(0)
                try:
                    self._on_status(role, state, reason)
                except Exception:
                    logger.exception("engine %s: status callback failed", role.value)
        except BaseException:
            with self._lock:
                self._delivering = False
            raise

    def _read_log(self, engine: EngineRef, offset: int, end: int | None = None) -> str:
        try:
            with open(self._paths.log_path(engine), "rb") as fh:
                fh.seek(offset)
                data = fh.read() if end is None else fh.read(max(0, end - offset))
                return data.decode("utf-8", errors="replace")
        except OSError:
            return ""

    def _log_size(self, engine: EngineRef) -> int:
        try:
            return self._paths.log_path(engine).stat().st_size
        except OSError:
            return 0

    def _startup_log(self, runtime: _Runtime, proc: _Proc) -> str:
        return self._read_log(runtime.engine, proc.log_offset, proc.startup_end)

    @staticmethod
    def _is_bind_failure(log: str) -> bool:
        lowered = log.lower()
        return any(marker in lowered for marker in BIND_FAILURE_MARKERS)

    @staticmethod
    def _kill(proc: _Proc) -> None:
        if proc.proc.poll() is not None:
            return
        with contextlib.suppress(OSError):
            proc.proc.kill()
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            proc.proc.wait(timeout=5.0)

    def _kill_all(self) -> None:
        with self._lock:
            procs = [
                proc
                for runtime in self._runtimes.values()
                for proc in (runtime.current, runtime.trial)
                if proc is not None
            ]
        for proc in procs:
            self._kill(proc)
