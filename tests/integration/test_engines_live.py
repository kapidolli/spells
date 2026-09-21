"""Live engine checks (slices of spec 20.2; brief 02).

Part 1 drives the engine supervisor against the real binaries in build/out/engines and the
tiny test models in build/cache/models. It skips with a reason when a binary or model is
missing. The Silero VAD model and a tiny instruct GGUF are downloaded into
build/cache/models on first use and verified against the sha256 pinned below.

Part 2 talks to servers that are already running, named by SPELLS_WHISPER_URL and
SPELLS_LLAMA_URL (for example http://127.0.0.1:8080 and http://127.0.0.1:8081).

Run with:

    .venv/Scripts/python.exe -m pytest -m integration tests/integration/test_engines_live.py -s
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import struct
import subprocess
import sys
import textwrap
import threading
import time
import urllib.request
from dataclasses import replace
from pathlib import Path

import pytest

from spells.asr import WhisperClient, pcm16_to_wav
from spells.cleanup import GateInput, LlamaClient, clean, default_corrections, default_fillers
from spells.engines import EnginePaths, EngineSupervisor
from spells.gpu import select_device
from spells.models import (
    CleanResult,
    DeliveryMethod,
    Engine,
    EngineState,
    Profile,
    Transcript,
)

pytestmark = pytest.mark.integration

REPO_DIR = Path(__file__).resolve().parents[2]
ENGINES_DIR = REPO_DIR / "build" / "out" / "engines"
MODELS_DIR = REPO_DIR / "build" / "cache" / "models"
WHISPER_TINY = MODELS_DIR / "ggml-tiny.bin"
VAD_MODEL = MODELS_DIR / "ggml-silero-v5.1.2.bin"
TINY_LLAMA = MODELS_DIR / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
DOWNLOADS = {
    VAD_MODEL: (
        "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin",
        "29940d98d42b91fbd05ce489f3ecf7c72f0a42f027e4875919a28fb4c04ea2cf",
    ),
    TINY_LLAMA: (
        (
            "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/"
            "qwen2.5-0.5b-instruct-q4_k_m.gguf"
        ),
        "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db",
    ),
}
READY_TIMEOUT_S = 90.0
RESTART_TIMEOUT_S = 60.0
JOB_KILL_TIMEOUT_S = 3.0
CREATE_NO_WINDOW = 0x08000000

WHISPER = Engine.WHISPER
LLAMA = Engine.LLAMA


# Helpers -----------------------------------------------------------------------------


def _missing_binaries() -> list[str]:
    missing = []
    for variant in ("vulkan", "cpu"):
        for exe in ("whisper-server.exe", "llama-server.exe"):
            path = ENGINES_DIR / variant / exe
            if not path.exists():
                missing.append(str(path.relative_to(REPO_DIR)))
    return missing


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_download(path: Path, url: str, sha256: str) -> str | None:
    """Download url to path unless present; returns a skip reason on any problem."""
    if path.exists():
        return None
    part = path.with_suffix(path.suffix + ".part")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={"User-Agent": "spells-tests/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response, open(part, "wb") as out:
            out.writelines(iter(lambda: response.read(1 << 20), b""))
        actual = _sha256(part)
        if actual != sha256:
            part.unlink(missing_ok=True)
            return f"{path.name} downloaded from {url} has sha256 {actual}, expected {sha256}"
        part.replace(path)
    except OSError as exc:
        part.unlink(missing_ok=True)
        return f"could not download {path.name} from {url}: {exc}"
    return None


def _alive(pid: int) -> bool:
    from spells.win32.window import process_image_path

    return process_image_path(pid) != ""


def _wait(predicate, timeout_s: float, interval_s: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout_s
    while True:
        if predicate():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(interval_s)


def test_switch_cpu_to_gpu_without_restarting_app(live_paths):
    live_paths = replace(live_paths, llama_args=("-lv", "4"))
    selection = select_device(
        live_paths.llama_exe("vulkan"), whisper_server=live_paths.whisper_exe("vulkan"))
    if selection.raw_index is None:
        pytest.skip("no Vulkan GPU available")
    supervisor = EngineSupervisor(
        live_paths, selection, lambda *args: None, cpu_only=True, cpu_cleanup_allowed=True)
    try:
        supervisor.start()
        for engine in (WHISPER, LLAMA):
            assert supervisor.wait_ready(engine, READY_TIMEOUT_S)
            assert supervisor.variant(engine) == "cpu"
        old_pids = {engine: supervisor.pid(engine) for engine in (WHISPER, LLAMA)}
        supervisor.set_models(
            live_paths, cpu_only=False, cpu_cleanup_allowed=False, gpu=selection)
        for engine in (WHISPER, LLAMA):
            assert _wait(lambda engine=engine: supervisor.pid(engine) not in (None, old_pids[engine]),
                         RESTART_TIMEOUT_S)
            assert supervisor.wait_ready(engine, READY_TIMEOUT_S)
            assert supervisor.variant(engine) == "vulkan"
            if engine is WHISPER:
                assert supervisor.gpu_verified(engine) is True
            else:
                # Recent llama.cpp logs model-device allocation rather than whisper's
                # ggml_vulkan device banner. Inspect the actual offload evidence.
                text = live_paths.log_path(engine).read_text(encoding="utf-8")
                assert f"using device Vulkan0 ({selection.name})" in text
                assert re.search(r"offloaded [1-9]\d*/\d+ layers to GPU", text)
        print(f"CPU to GPU switch verified on {selection.name}")
    finally:
        supervisor.stop()


# Part 1: the supervisor against the real engines ------------------------------------------


@pytest.fixture(scope="module")
def live_paths(tmp_path_factory) -> EnginePaths:
    missing = _missing_binaries()
    if missing:
        pytest.skip("engine binaries missing: " + ", ".join(missing))
    if not WHISPER_TINY.exists():
        pytest.skip(f"{WHISPER_TINY} missing; run: py build/fetch.py models.whisper_tiny_test")
    for path, (url, sha256) in DOWNLOADS.items():
        reason = _ensure_download(path, url, sha256)
        if reason is not None:
            pytest.skip(reason)
    log_dir = Path(tmp_path_factory.mktemp("engine-logs"))
    print(f"\nengine logs: {log_dir}")
    return EnginePaths(
        vulkan_dir=ENGINES_DIR / "vulkan",
        cpu_dir=ENGINES_DIR / "cpu",
        whisper_model=WHISPER_TINY,
        vad_model=VAD_MODEL,
        llama_model=TINY_LLAMA,
        log_dir=log_dir,
    )


class Live:
    """One started supervisor plus every status change it reported, with timestamps."""

    def __init__(self, paths: EnginePaths) -> None:
        self.paths = paths
        probe_started = time.monotonic()
        self.selection = select_device(
            paths.llama_exe("vulkan"), whisper_server=paths.whisper_exe("vulkan")
        )
        self.probe_s = time.monotonic() - probe_started
        self.events: list[tuple[float, Engine, EngineState, str]] = []
        self.lock = threading.Lock()
        self.started_at = time.monotonic()
        self.sup = EngineSupervisor(paths, self.selection, self.on_status)
        self.sup.start()

    def on_status(self, engine: Engine, state: EngineState, reason: str) -> None:
        with self.lock:
            self.events.append((time.monotonic() - self.started_at, engine, state, reason))

    def wait_for(self, engine: Engine, state: EngineState, timeout_s: float, since: int) -> bool:
        def seen() -> bool:
            with self.lock:
                return any(e is engine and s is state for _, e, s, _ in self.events[since:])

        return _wait(seen, timeout_s)

    def log_text(self, engine: Engine) -> str:
        path = self.paths.log_path(engine)
        return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""

    def describe(self) -> str:
        with self.lock:
            lines = [f"{t:7.1f} s  {e.value:8} {s.value:13} {r}" for t, e, s, r in self.events]
        return "\n".join(lines)


@pytest.fixture(scope="module")
def live(live_paths) -> Live:
    live = Live(live_paths)
    yield live
    live.sup.stop()


def test_both_engines_reach_ready_on_the_gpu(live):
    for engine in (WHISPER, LLAMA):
        ok = live.sup.wait_ready(engine, READY_TIMEOUT_S)
        assert ok, (
            f"{engine.value} not serving within {READY_TIMEOUT_S:.0f} s\n{live.describe()}\n"
            f"{live.log_text(engine)[-3000:]}"
        )
        print(
            f"[{engine.value}] serving after {time.monotonic() - live.started_at:.1f} s: "
            f"{live.sup.status(engine).value} ({live.sup.reason(engine)}), "
            f"{live.sup.variant(engine)} build, {live.sup.whisper_url if engine is WHISPER else live.sup.llama_url}"
        )
    print(live.describe())
    for line in live.log_text(LLAMA).splitlines():
        if "prompt eval time" in line or "total time" in line:
            print(f"[llama warm-up] {line.split('|')[-1].strip()}")
    assert live.sup.status(WHISPER) is EngineState.READY, live.describe()
    assert live.sup.status(LLAMA) is EngineState.READY, live.describe()
    assert live.sup.whisper_url and live.sup.llama_url
    assert live.sup.cleanup_available


def test_engine_logs_name_the_selected_gpu(live):
    print(
        f"selected GPU: raw index {live.selection.raw_index}, {live.selection.name!r}, "
        f"{live.selection.memory_mb} MiB (probe took {live.probe_s:.1f} s); probed:"
    )
    for device in live.selection.devices:
        print(f"  {device}")
    assert live.selection.raw_index is not None, "select_device found no Vulkan GPU"
    for engine in (WHISPER, LLAMA):
        print(f"[{engine.value}] gpu_verified = {live.sup.gpu_verified(engine)}")
    # whisper-server 1.9.4 prints the ggml_vulkan device block at startup.
    assert live.sup.gpu_verified(WHISPER) is True, live.log_text(WHISPER)[-3000:]
    assert live.selection.name in live.log_text(WHISPER)
    # llama-server b10997 prints no device line at its default verbosity, and the verbose
    # levels would log request text (spec 17), so the answer there is "unknown", never False.
    assert live.sup.gpu_verified(LLAMA) is not False, live.log_text(LLAMA)[-3000:]


def test_selected_gpu_is_the_rtx_5060(live):
    """Decision B3-59: the discrete RTX 5060 wins over the Radeon 610M iGPU that reports more
    (shared) memory. The engine logs must name it as well."""
    assert "RTX 5060" in live.selection.name, (
        f"select_device picked {live.selection.name!r} ({live.selection.memory_mb} MiB) "
        f"instead of the RTX 5060; probed devices: {live.selection.devices}"
    )
    chosen = next(d for d in live.selection.devices if d.raw_index == live.selection.raw_index)
    assert chosen.uma is False, chosen
    assert "RTX 5060" in live.log_text(WHISPER)
    # llama-server b10997 logs the device name only at -lv 3 in its device_info block.
    assert "RTX 5060" in live.log_text(LLAMA) or live.sup.gpu_verified(LLAMA) is None


def test_killed_llama_server_restarts(live):
    assert live.sup.wait_ready(LLAMA, READY_TIMEOUT_S), live.describe()
    pid = live.sup.pid(LLAMA)
    assert pid, "no llama-server pid"
    with live.lock:
        since = len(live.events)
    killed_at = time.monotonic()
    subprocess.run(
        ["taskkill", "/F", "/PID", str(pid)],
        check=True,
        capture_output=True,
        creationflags=CREATE_NO_WINDOW,
    )
    assert live.wait_for(LLAMA, EngineState.RESTARTING, 15.0, since), live.describe()
    assert live.wait_for(LLAMA, EngineState.READY, RESTART_TIMEOUT_S, since), live.describe()
    print(f"[llama] READY again {time.monotonic() - killed_at:.1f} s after the external kill")
    new_pid = live.sup.pid(LLAMA)
    assert new_pid and new_pid != pid
    assert not _alive(pid)
    assert live.sup.status(WHISPER) is EngineState.READY


def test_stop_leaves_no_engine_processes(live):
    pids = [live.sup.pid(engine) for engine in Engine]
    assert all(pids), pids
    live.sup.stop()
    assert _wait(lambda: not any(_alive(pid) for pid in pids), 10.0), pids
    assert live.sup.whisper_url is None and live.sup.llama_url is None


def test_engines_die_with_the_app_process(live_paths, tmp_path):
    """Decision V4-11: the job object kills the engines when the app process dies."""
    pid_file = tmp_path / "pids.txt"
    code = textwrap.dedent(
        f"""
        import os, sys, time
        from pathlib import Path
        sys.path.insert(0, {str(REPO_DIR / "src")!r})
        from spells.engines import EnginePaths, EngineSupervisor
        from spells.gpu import NO_GPU
        from spells.models import Engine
        paths = EnginePaths(
            vulkan_dir=Path({str(live_paths.vulkan_dir)!r}),
            cpu_dir=Path({str(live_paths.cpu_dir)!r}),
            whisper_model=Path({str(live_paths.whisper_model)!r}),
            vad_model=Path({str(live_paths.vad_model)!r}),
            llama_model=Path({str(live_paths.llama_model)!r}),
            log_dir=Path({str(tmp_path / "logs")!r}),
        )
        sup = EngineSupervisor(paths, NO_GPU, lambda *args: None)
        sup.start()
        deadline = time.monotonic() + 60
        pids = []
        while time.monotonic() < deadline:
            pids = [sup.pid(Engine.WHISPER), sup.pid(Engine.LLAMA)]
            if all(pids):
                break
            time.sleep(0.05)
        Path({str(pid_file)!r}).write_text(" ".join(str(p) for p in pids))
        os._exit(0)
        """
    )
    child = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=CREATE_NO_WINDOW,
    )
    out, err = child.communicate(timeout=180)
    exited_at = time.monotonic()
    assert child.returncode == 0, err.decode("utf-8", errors="replace")
    pids = [int(p) for p in pid_file.read_text().split() if p != "None"]
    assert len(pids) == 2, (pid_file.read_text(), out, err)
    assert _wait(lambda: not any(_alive(pid) for pid in pids), JOB_KILL_TIMEOUT_S), (
        f"engine processes {pids} outlived the app process"
    )
    print(f"[job] engines gone {time.monotonic() - exited_at:.2f} s after the app process exited")


# Part 2: already running servers named by environment variables ------------------------------

WHISPER_URL = os.environ.get("SPELLS_WHISPER_URL", "")
LLAMA_URL = os.environ.get("SPELLS_LLAMA_URL", "")
needs_engines = pytest.mark.skipif(
    not (WHISPER_URL and LLAMA_URL),
    reason="set SPELLS_WHISPER_URL and SPELLS_LLAMA_URL to run against live engines",
)

SAMPLE_RATE = 16000
FIXED_TRANSCRIPT = (
    "um so I think we should, uh, meet tomorrow at ten, you know, to go through the release "
    "plan and, um, decide who owns the installer work"
)


def _tone(seconds: float = 3.0, hz: float = 440.0) -> bytes:
    n = int(seconds * SAMPLE_RATE)
    samples = (int(8000 * math.sin(2 * math.pi * hz * i / SAMPLE_RATE)) for i in range(n))
    return struct.pack(f"<{n}h", *samples)


@needs_engines
def test_whisper_health():
    assert WhisperClient(WHISPER_URL).health()


@needs_engines
def test_llama_health():
    assert LlamaClient(LLAMA_URL).health()


@needs_engines
def test_whisper_inference_on_a_three_second_tone_returns_a_dict():
    client = WhisperClient(WHISPER_URL, timeout_s=60.0)
    result = client.inference(pcm16_to_wav(_tone(), SAMPLE_RATE), language="en", prompt="")
    assert isinstance(result, dict)
    assert "text" in result


@needs_engines
def test_clean_fixed_transcript_against_live_llama():
    client = LlamaClient(LLAMA_URL, timeout_s=30.0)
    transcript = Transcript(text=FIXED_TRANSCRIPT, language="en", duration_s=12.0)
    profile = Profile(name="default", cleanup=True, tone="Neutral.", delivery=DeliveryMethod.PASTE)
    gate = GateInput(
        cleanup_enabled=True, profile_cleanup=True, engine_available=True, cpu_fallback=False
    )
    result = clean(
        transcript,
        profile,
        ["Spells"],
        default_fillers(),
        default_corrections(),
        ["en", "de", "sq"],
        gate,
        client,
    )
    assert isinstance(result, CleanResult)
    assert result.text.strip()
    assert result.reason in {
        "ok",
        "empty",
        "length_ratio",
        "finish_length",
        "preamble",
        "language_switch",
    }
