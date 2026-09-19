from __future__ import annotations

import contextlib
import difflib
import json
import logging
import os
import re
import socket
import statistics
import subprocess
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from spells.datafiles import data_path
from spells.engines import EnginePaths, Variant, launch_args
from spells.gpu import GpuSelection, engine_env
from spells.modelcatalog import (
    LLAMA_ASR,
    WHISPER_SERVER,
    CatalogModel,
    Hardware,
    ModelChoice,
    ModelKind,
    hardware_from_gpu,
    select_models,
    with_latencies,
)
from spells.models import CpuPlan, Engine, EngineId

logger = logging.getLogger(__name__)

CALIBRATION_VERSION = 1
CALIBRATION_TEXT = (
    "Thanks for the quick reply. I checked the numbers again this morning, and the report "
    "looks right to me. Could you send the final version to the whole team before Friday, "
    "so we can review it together next week?"
)
CALIBRATION_LANGUAGE = "en"
GPU_WIN_RATIO = 0.8
MAX_SLOWDOWN = 1.5
MIN_SIMILARITY = 0.8
STARTUP_TIMEOUT_S = 240.0
REQUEST_TIMEOUT_S = 180.0
TIMED_RUNS = 2
SLOW_RUN_MS = 4000
HEALTH_POLL_S = 0.25
KILL_WAIT_S = 5.0
STORE_NAME = "calibration.json"
LOG_NAME = "calibration.log"
SPEECH_ENGINE = EngineId(Engine.WHISPER, 0)

_WORD = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True)
class Run:
    variant: str
    latency_ms: int | None = None
    similarity: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.latency_ms is not None and self.similarity >= MIN_SIMILARITY

    def describe(self) -> str:
        if self.latency_ms is None:
            return f"{self.variant} failed ({self.error or 'no answer'})"
        verdict = "" if self.ok else f", transcript wrong ({self.similarity:.2f})"
        return f"{self.variant} {self.latency_ms} ms{verdict}"


@dataclass(frozen=True)
class SpeechCalibration:
    model_id: str
    device: str
    gpu: Run
    cpu: Run
    version: int = CALIBRATION_VERSION
    measured_at: str = ""

    @property
    def prefers_gpu(self) -> bool:
        if not self.gpu.ok or self.gpu.latency_ms is None:
            return False
        if not self.cpu.ok or self.cpu.latency_ms is None:
            return True
        return self.gpu.latency_ms <= self.cpu.latency_ms * GPU_WIN_RATIO

    def latency_ms(self, on_gpu: bool) -> int | None:
        run = self.gpu if on_gpu else self.cpu
        return run.latency_ms if run.ok else None

    def describe(self) -> str:
        winner = "graphics" if self.prefers_gpu else "processor"
        return (
            f"{self.model_id} on {self.device}: {self.gpu.describe()}, {self.cpu.describe()}; "
            f"the {winner} wins"
        )


def similarity(text: str, reference: str = CALIBRATION_TEXT) -> float:
    return difflib.SequenceMatcher(
        None, _WORD.findall(text.lower()), _WORD.findall(reference.lower())
    ).ratio()


def store_path(settings_path: Path) -> Path:
    return Path(settings_path).parent / STORE_NAME


def _run_from(raw: object, variant: str) -> Run:
    if not isinstance(raw, dict):
        return Run(variant, error="missing")
    latency = raw.get("latency_ms")
    score = raw.get("similarity", 0.0)
    return Run(
        variant=variant,
        latency_ms=latency if isinstance(latency, int) and latency >= 0 else None,
        similarity=float(score) if isinstance(score, int | float) else 0.0,
        error=str(raw.get("error", ""))[:200],
    )


def _run_to(run: Run) -> dict[str, Any]:
    return {"latency_ms": run.latency_ms, "similarity": run.similarity, "error": run.error}


def _read_entries(path: Path) -> list[dict[str, Any]]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as exc:
        logger.warning("calibration results in %s could not be read: %s", path, exc)
        return []
    entries = raw.get("entries") if isinstance(raw, dict) else None
    return [entry for entry in entries or [] if isinstance(entry, dict)]


def load(path: Path, device: str) -> dict[str, SpeechCalibration]:
    results: dict[str, SpeechCalibration] = {}
    if not device:
        return results
    for entry in _read_entries(path):
        if entry.get("device") != device or entry.get("version") != CALIBRATION_VERSION:
            continue
        model_id = entry.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            continue
        results[model_id] = SpeechCalibration(
            model_id=model_id,
            device=device,
            gpu=_run_from(entry.get("gpu"), "vulkan"),
            cpu=_run_from(entry.get("cpu"), "cpu"),
            measured_at=str(entry.get("measured_at", "")),
        )
    return results


def save(path: Path, calibrations: Iterable[SpeechCalibration]) -> None:
    fresh = list(calibrations)
    replaced = {(item.device, item.model_id) for item in fresh}
    kept = [
        entry
        for entry in _read_entries(path)
        if (entry.get("device"), entry.get("model_id")) not in replaced
    ]
    entries = kept + [
        {
            "model_id": item.model_id,
            "device": item.device,
            "version": item.version,
            "measured_at": item.measured_at,
            "gpu": _run_to(item.gpu),
            "cpu": _run_to(item.cpu),
        }
        for item in fresh
    ]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps({"entries": entries}, indent=2), encoding="utf-8")
    os.replace(temporary, target)


def measured_latencies(
    calibrations: Mapping[str, SpeechCalibration], on_gpu: bool
) -> dict[str, int]:
    latencies = {}
    for model_id, calibration in calibrations.items():
        latency = calibration.latency_ms(on_gpu)
        if latency is not None:
            latencies[model_id] = latency
    return latencies


def needs_measuring(gpu: GpuSelection) -> bool:
    return gpu.raw_index is not None and hardware_from_gpu(gpu) is Hardware.CPU


def choose_hardware(
    gpu: GpuSelection,
    languages: Sequence[str],
    installed: frozenset[str],
    catalog: Sequence[CatalogModel],
    calibrations: Mapping[str, SpeechCalibration],
) -> tuple[Hardware, bool, tuple[CatalogModel, ...]]:
    if not needs_measuring(gpu):
        return hardware_from_gpu(gpu), False, tuple(catalog)
    gpu_catalog = with_latencies(catalog, Hardware.GPU, measured_latencies(calibrations, True))
    cpu_catalog = with_latencies(catalog, Hardware.CPU, measured_latencies(calibrations, False))
    gpu_plan = select_models(languages, Hardware.GPU, installed, gpu_catalog)
    cpu_plan = select_models(languages, Hardware.CPU, installed, cpu_catalog)
    if gpu_plan_wins(gpu_plan, cpu_plan, languages, calibrations):
        return Hardware.GPU, True, gpu_catalog
    return Hardware.CPU, False, cpu_catalog


def gpu_plan_wins(
    gpu_plan: Any,
    cpu_plan: Any,
    languages: Sequence[str],
    calibrations: Mapping[str, SpeechCalibration],
) -> bool:
    gpu_total = cpu_total = 0
    for code in dict.fromkeys(languages):
        on_gpu, on_cpu = gpu_plan.asr_for(code), cpu_plan.asr_for(code)
        if on_gpu is None:
            if on_cpu is not None:
                return False
            continue
        measured = calibrations.get(on_gpu.model_id)
        if measured is None or not measured.gpu.ok or measured.gpu.latency_ms is None:
            return False
        gpu_ms = measured.gpu.latency_ms
        cpu_ms = None if on_cpu is None else on_cpu.latency_ms
        if cpu_ms is None:
            continue
        if gpu_ms > cpu_ms * MAX_SLOWDOWN:
            return False
        gpu_total += gpu_ms
        cpu_total += cpu_ms
    return gpu_total > 0 and gpu_total <= cpu_total * GPU_WIN_RATIO


def pending(
    gpu: GpuSelection,
    languages: Sequence[str],
    installed: frozenset[str],
    catalog: Sequence[CatalogModel],
    calibrations: Mapping[str, SpeechCalibration],
) -> list[ModelChoice]:
    if not needs_measuring(gpu):
        return []
    todo = []
    for model in catalog:
        if model.kind is not ModelKind.ASR or model.id not in installed:
            continue
        if model.id in calibrations or model.runtime not in (WHISPER_SERVER, LLAMA_ASR):
            continue
        served = tuple(code for code in dict.fromkeys(languages) if model.score(code) is not None)
        if not served:
            continue
        todo.append(
            ModelChoice(
                kind=model.kind,
                model_id=model.id,
                display_name=model.display_name,
                languages=served,
                reason="",
                installed=True,
                file=model.file,
                extra_files=model.extra_files,
                engine_args=model.engine_args,
                runtime=model.runtime,
                latency_ms=model.latency_ms(Hardware.CPU),
            )
        )
    return todo


def _default_client(runtime: str, url: str) -> Any:
    if runtime == LLAMA_ASR:
        from spells.asr import LlamaAsrClient

        return LlamaAsrClient(url, timeout_s=REQUEST_TIMEOUT_S)
    from spells.asr import WhisperClient

    return WhisperClient(url, timeout_s=REQUEST_TIMEOUT_S)


def _default_spawn(args, *, env, log_path, cwd, job, affinity_mask) -> Any:
    from spells.win32 import process as win32_process

    return win32_process.spawn_hidden(
        args,
        env=env,
        stdout_path=log_path,
        stderr_path=log_path,
        cwd=cwd,
        job=job,
        affinity_mask=affinity_mask,
    )


def _default_job() -> Any:
    from spells.win32 import process as win32_process

    return win32_process.JobObject()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class Cancelled(Exception):
    pass


@dataclass
class Calibrator:
    paths: EnginePaths
    gpu: GpuSelection
    cpu_plan: CpuPlan
    whisper_threads: int
    models_dir: Path
    clip: bytes = b""
    warmup: bytes = b""
    spawn: Callable[..., Any] = _default_spawn
    job_factory: Callable[[], Any] = _default_job
    client_factory: Callable[[str, str], Any] = _default_client
    clock: Callable[[], float] = time.monotonic
    sleeper: Callable[[float], None] = time.sleep
    now: Callable[[], datetime] = lambda: datetime.now(UTC)
    _stop: threading.Event = field(default_factory=threading.Event)
    _job: Any = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def cancel(self) -> None:
        self._stop.set()
        with self._lock:
            job, self._job = self._job, None
        if job is not None:
            with contextlib.suppress(Exception):
                job.close()

    @property
    def cancelled(self) -> bool:
        return self._stop.is_set()

    def calibrate(self, choice: ModelChoice) -> SpeechCalibration:
        paths = self.paths.with_speech(self.models_dir, [choice])
        runtime = choice.runtime or WHISPER_SERVER
        gpu = self._measure(paths, runtime, "vulkan")
        cpu = self._measure(paths, runtime, "cpu")
        return SpeechCalibration(
            model_id=choice.model_id,
            device=self.gpu.name,
            gpu=gpu,
            cpu=cpu,
            measured_at=self.now().isoformat(timespec="seconds"),
        )

    def _ensure_job(self) -> Any:
        with self._lock:
            if self._job is None:
                self._job = self.job_factory()
            return self._job

    def _clip(self) -> bytes:
        return self.clip or data_path("calibration.wav").read_bytes()

    def _warmup(self) -> bytes:
        return self.warmup or data_path("warmup.wav").read_bytes()

    def _measure(self, paths: EnginePaths, runtime: str, variant: Variant) -> Run:
        if self._stop.is_set():
            return Run(variant, error="cancelled")
        if variant == "vulkan" and self.gpu.raw_index is None:
            return Run(variant, error="no graphics device")
        exe = paths.exe(SPEECH_ENGINE, variant)
        if not exe.is_file():
            return Run(variant, error=f"{exe.name} is missing")
        port = _free_port()
        args = launch_args(
            paths,
            SPEECH_ENGINE,
            variant,
            port,
            whisper_threads=self.whisper_threads,
            cpu_plan=self.cpu_plan,
        )
        log_path = paths.log_dir / LOG_NAME
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            proc = self.spawn(
                args,
                env=engine_env(self.gpu),
                log_path=log_path,
                cwd=exe.parent,
                job=self._ensure_job(),
                affinity_mask=self.cpu_plan.affinity_mask,
            )
        except Exception as exc:
            logger.exception("the %s build of %s could not start", variant, exe.name)
            return Run(variant, error=f"could not start: {exc}"[:200])
        try:
            client = self.client_factory(runtime, f"http://127.0.0.1:{port}")
            if not self._wait_healthy(proc, client):
                return Run(variant, error="the engine did not start")
            self._transcribe(runtime, client, self._warmup())
            clip = self._clip()
            times: list[int] = []
            text = ""
            for _ in range(TIMED_RUNS):
                started = self.clock()
                text = self._transcribe(runtime, client, clip)
                times.append(round((self.clock() - started) * 1000))
                if times[-1] > SLOW_RUN_MS:
                    break
            return Run(
                variant,
                latency_ms=int(statistics.median(times)),
                similarity=round(similarity(text), 3),
            )
        except Cancelled:
            return Run(variant, error="cancelled")
        except Exception as exc:
            logger.exception("measuring the %s build of %s failed", variant, exe.name)
            return Run(variant, error=str(exc)[:200] or type(exc).__name__)
        finally:
            self._kill(proc)

    def _wait_healthy(self, proc: Any, client: Any) -> bool:
        deadline = self.clock() + STARTUP_TIMEOUT_S
        while self.clock() < deadline:
            if self._stop.is_set():
                raise Cancelled
            if proc.poll() is not None:
                return False
            with contextlib.suppress(Exception):
                if client.health():
                    return True
            self.sleeper(HEALTH_POLL_S)
        return False

    def _transcribe(self, runtime: str, client: Any, wav: bytes) -> str:
        if self._stop.is_set():
            raise Cancelled
        if runtime == LLAMA_ASR:
            reply = client.transcribe(
                wav, language=CALIBRATION_LANGUAGE, timeout_s=REQUEST_TIMEOUT_S
            )
            return str(getattr(reply, "text", "") or "")
        result = client.inference(
            wav, language=CALIBRATION_LANGUAGE, prompt="", timeout_s=REQUEST_TIMEOUT_S
        )
        return str(result.get("text", "") if isinstance(result, dict) else "")

    @staticmethod
    def _kill(proc: Any) -> None:
        with contextlib.suppress(Exception):
            if proc.poll() is None:
                proc.kill()
        with contextlib.suppress(Exception, subprocess.TimeoutExpired):
            proc.wait(timeout=KILL_WAIT_S)
