"""Engine supervisor with fake processes and a fake clock (spec 13, 16, 20.1).

The supervisor's engine threads call step() in a loop; here the Harness calls step() directly
and advances the fake clock by the delay step() asks for, so every scenario is deterministic.
"""

from __future__ import annotations

import itertools
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from spells.datafiles import data_path
from spells.engines import (
    LOG_KEEP,
    LOG_MAX_BYTES,
    REASONS,
    STATUS_DLL_NOT_FOUND,
    EnginePaths,
    EngineSupervisor,
    SpeechEngine,
    SpeechSlot,
    classify_failure,
    engine_label,
    gpu_verified_from_log,
)
from spells.gpu import GpuDevice, GpuSelection
from spells.modelcatalog import ModelChoice, ModelKind
from spells.models import CpuPlan, Engine, EngineId, EngineState

from .fake_process import FakeClock, FakePopen, FakeWorld

WHISPER = Engine.WHISPER
LLAMA = Engine.LLAMA
SPEECH_1 = EngineId(WHISPER, 0)
SPEECH_2 = EngineId(WHISPER, 1)
CLEANUP = EngineId(LLAMA)
RTX = "NVIDIA GeForce RTX 5060 Laptop GPU"
GPU = GpuSelection(
    raw_index=0,
    name=RTX,
    memory_mb=7810,
    devices=(GpuDevice(0, "Vulkan0", RTX, 7810, 7042),),
)
NO_GPU = GpuSelection(raw_index=None, name="", memory_mb=0, devices=())
FOUND_RTX = (
    "ggml_vulkan: Found 1 Vulkan devices:\n"
    f"ggml_vulkan: 0 = {RTX} (NVIDIA) | uma: 0 | fp16: 1 | bf16: 0 | warp size: 32\n"
)
FOUND_OTHER = (
    "ggml_vulkan: Found 1 Vulkan devices:\n"
    "ggml_vulkan: 0 = AMD Radeon(TM) 610M (AMD proprietary driver) | uma: 1 | fp16: 1\n"
)
NO_DEVICES = "ggml_vulkan: No devices found.\n"
OOM = "ggml_vulkan: Device memory allocation of size 1234567 failed.\nErrorOutOfDeviceMemory\n"
CRASH = "llama_model_load: error loading model: unexpectedly reached end of file\n"


def make_paths(tmp_path: Path) -> EnginePaths:
    vulkan = tmp_path / "engines" / "vulkan"
    cpu = tmp_path / "engines" / "cpu"
    for folder in (vulkan, cpu):
        folder.mkdir(parents=True)
        (folder / "whisper-server.exe").write_bytes(b"")
        (folder / "llama-server.exe").write_bytes(b"")
    return EnginePaths(
        vulkan_dir=vulkan,
        cpu_dir=cpu,
        whisper_model=tmp_path / "models" / "whisper.bin",
        vad_model=tmp_path / "models" / "vad.bin",
        llama_model=tmp_path / "models" / "cleanup.gguf",
        log_dir=tmp_path / "logs",
    )


class Harness:
    def __init__(
        self,
        tmp_path: Path,
        world: FakeWorld,
        gpu: GpuSelection = GPU,
        paths: EnginePaths | None = None,
        **kwargs,
    ):
        self.world = world
        self.clock = FakeClock()
        self.events: list[tuple[Engine, EngineState, str]] = []
        self.paths = paths if paths is not None else make_paths(tmp_path)
        self.sup = EngineSupervisor(
            self.paths,
            gpu,
            self.on_status,
            process_backend=world,
            whisper_client_factory=world.whisper_client,
            llama_client_factory=world.llama_client,
            llama_asr_client_factory=world.llama_asr_client,
            clock=self.clock.now,
            sleeper=self.clock.sleep,
            **kwargs,
        )

    def on_status(self, engine: Engine, state: EngineState, reason: str) -> None:
        self.events.append((engine, state, reason))

    def run(self, engine: Engine, seconds: float = 0.0, max_steps: int = 10000) -> None:
        """Drive one engine's loop as its thread would, for `seconds` of fake time."""
        end = self.clock.now() + seconds
        for _ in range(max_steps):
            delay = self.sup.step(engine)
            if delay <= 0:
                continue
            remaining = end - self.clock.now()
            if remaining <= 0:
                return
            self.clock.advance(min(delay, remaining))
        raise AssertionError("the supervision loop never settled")

    def run_both(self, seconds: float = 0.0, max_steps: int = 10000) -> None:
        end = self.clock.now() + seconds
        for _ in range(max_steps):
            delays = [self.sup.step(WHISPER), self.sup.step(LLAMA)]
            if min(delays) <= 0:
                continue
            remaining = end - self.clock.now()
            if remaining <= 0:
                return
            self.clock.advance(min(min(delays), remaining))
        raise AssertionError("the supervision loops never settled")

    def events_for(self, engine: Engine) -> list[tuple[EngineState, str]]:
        return [(state, reason) for e, state, reason in self.events if e is engine]


@pytest.fixture
def world() -> FakeWorld:
    return FakeWorld()


@pytest.fixture
def harness(tmp_path, world) -> Harness:
    return Harness(tmp_path, world)


# Startup ---------------------------------------------------------------------------


def test_starting_to_ready_with_warm_up(harness, world):
    assert harness.sup.status(WHISPER) is EngineState.STARTING
    assert harness.sup.whisper_url is None
    harness.run_both()
    whisper = world.latest("whisper")
    llama = world.latest("llama")
    assert whisper.variant == "vulkan" and llama.variant == "vulkan"
    assert harness.sup.status(WHISPER) is EngineState.READY
    assert harness.sup.status(LLAMA) is EngineState.READY
    assert harness.sup.reason(WHISPER) == "ok"
    assert harness.sup.whisper_url == f"http://127.0.0.1:{whisper.port}"
    assert harness.sup.llama_url == f"http://127.0.0.1:{llama.port}"
    assert whisper.inference_calls == 1
    assert whisper.inference_wavs[0] == data_path("warmup.wav").read_bytes()
    assert llama.chat_calls == 1
    system, user, max_tokens = llama.chats[0]
    assert max_tokens == 1
    assert system and "transcript" in user.lower()
    assert harness.events_for(WHISPER) == [(EngineState.READY, "ok")]
    assert harness.events_for(LLAMA) == [(EngineState.READY, "ok")]


def test_url_is_hidden_until_warm_up_finished(harness, world):
    world.on_spawn = lambda proc: setattr(proc, "healthy", False)
    harness.run(WHISPER, 5.0)
    assert harness.sup.status(WHISPER) is EngineState.STARTING
    assert harness.sup.whisper_url is None
    proc = world.latest("whisper")
    assert proc.health_calls > 1
    proc.healthy = True
    harness.run(WHISPER, 1.0)
    assert harness.sup.status(WHISPER) is EngineState.READY
    assert harness.sup.whisper_url is not None


def test_no_gpu_launches_cpu_variants_directly(tmp_path, world):
    harness = Harness(tmp_path, world, gpu=NO_GPU)
    harness.run_both()
    assert [p.variant for p in world.processes] == ["cpu", "cpu"]
    assert harness.sup.status(WHISPER) is EngineState.CPU_FALLBACK
    assert harness.sup.status(LLAMA) is EngineState.CPU_FALLBACK
    assert harness.sup.reason(WHISPER) == "no_vulkan_gpu"
    assert harness.sup.reason(LLAMA) == "no_vulkan_gpu"
    assert harness.sup.whisper_url is not None
    assert harness.sup.llama_url is not None
    assert harness.sup.cleanup_available is False
    assert harness.sup.gpu_verified(WHISPER) is None
    assert "GGML_VK_VISIBLE_DEVICES" not in world.processes[0].env
    assert harness.events_for(WHISPER) == [
        (EngineState.STARTING, "no_vulkan_gpu"),
        (EngineState.CPU_FALLBACK, "no_vulkan_gpu"),
    ]
    # No Vulkan retry ever: there is nothing to retry with.
    harness.run_both(3600.0)
    assert len(world.processes) == 2


def test_log_scan_switches_a_vulkan_build_that_found_no_device_to_cpu(harness, world):
    def on_spawn(proc: FakePopen) -> None:
        if proc.engine == "whisper" and proc.variant == "vulkan":
            proc.write_log(NO_DEVICES)

    world.on_spawn = on_spawn
    harness.run(WHISPER)
    vulkan = world.latest("whisper", "vulkan")
    cpu = world.latest("whisper", "cpu")
    assert not vulkan.alive and vulkan.killed
    assert cpu.alive
    assert harness.sup.status(WHISPER) is EngineState.CPU_FALLBACK
    assert harness.sup.reason(WHISPER) == "no_vulkan_gpu"
    assert harness.sup.variant(WHISPER) == "cpu"
    assert harness.sup.whisper_url == f"http://127.0.0.1:{cpu.port}"
    assert harness.events_for(WHISPER) == [
        (EngineState.RESTARTING, "no_vulkan_gpu"),
        (EngineState.CPU_FALLBACK, "no_vulkan_gpu"),
    ]
    harness.run(WHISPER, 7200.0)
    assert len(world.spawned("whisper", "vulkan")) == 1


def test_no_device_line_flushed_just_before_health_still_counts(harness, world):
    def on_spawn(proc: FakePopen) -> None:
        if proc.variant == "vulkan":
            proc.on_health = lambda: proc.write_log(NO_DEVICES)

    world.on_spawn = on_spawn
    harness.run(WHISPER)
    assert harness.sup.status(WHISPER) is EngineState.CPU_FALLBACK
    assert harness.sup.reason(WHISPER) == "no_vulkan_gpu"
    assert world.latest("whisper").variant == "cpu"


def test_no_device_line_after_ready_is_ignored(harness, world):
    # whisper-server 1.9.4 prints "whisper_backend_init_gpu: no GPU found" for its CPU-side VAD
    # context on the first request of a healthy Vulkan run; only the startup window counts.
    harness.run(WHISPER)
    vulkan = world.latest("whisper")
    vulkan.write_log("whisper_backend_init_gpu: no GPU found\n")
    harness.run(WHISPER, 30.0)
    assert vulkan.alive
    assert harness.sup.status(WHISPER) is EngineState.READY
    assert len(world.processes) == 1


def test_crash_after_ready_with_a_late_no_device_line_is_a_crash(harness, world):
    harness.run(WHISPER)
    vulkan = world.latest("whisper")
    vulkan.write_log("whisper_backend_init_gpu: no GPU found\n")
    vulkan.exit(1, CRASH)
    harness.run(WHISPER)
    assert harness.sup.status(WHISPER) is EngineState.RESTARTING
    assert harness.sup.reason(WHISPER) == "crash"


def test_classify_failure_reads_no_device_from_the_startup_window_only():
    assert classify_failure(1, NO_DEVICES, startup_log="") == "crash"
    assert classify_failure(1, "", startup_log=NO_DEVICES) == "no_vulkan_gpu"
    assert classify_failure(1, OOM, startup_log="") == "oom"


def test_gpu_verified_true_when_the_log_names_the_chosen_device(harness, world):
    world.on_spawn = lambda proc: proc.write_log(FOUND_RTX)
    harness.run_both()
    assert harness.sup.gpu_verified(WHISPER) is True
    assert harness.sup.gpu_verified(LLAMA) is True


def test_gpu_verified_false_when_the_log_names_another_device(harness, world):
    world.on_spawn = lambda proc: proc.write_log(FOUND_OTHER)
    harness.run(WHISPER)
    assert harness.sup.status(WHISPER) is EngineState.READY
    assert harness.sup.gpu_verified(WHISPER) is False


def test_gpu_verified_none_without_a_found_line(harness):
    harness.run(WHISPER)
    assert harness.sup.gpu_verified(WHISPER) is None


def test_gpu_verified_is_re_read_at_health_ticks_until_known(harness, world):
    harness.run(WHISPER)
    assert harness.sup.gpu_verified(WHISPER) is None
    world.latest("whisper").write_log(FOUND_RTX)
    harness.run(WHISPER, 5.0)
    assert harness.sup.gpu_verified(WHISPER) is None
    harness.run(WHISPER, 6.0)
    assert harness.sup.gpu_verified(WHISPER) is True


def test_gpu_verified_from_log_reads_only_the_device_block():
    log = "ggml_vulkan: Found 1 Vulkan devices:\nggml_vulkan: 0 = Other (X) | uma: 0\n" + RTX
    assert gpu_verified_from_log(log, RTX) is False
    assert gpu_verified_from_log(FOUND_RTX, RTX) is True
    assert gpu_verified_from_log("nothing here", RTX) is None
    assert gpu_verified_from_log(FOUND_RTX, "") is False


# OOM ----------------------------------------------------------------------------


def test_whisper_oom_switches_to_cpu_and_retries_vulkan_with_a_second_process(harness, world):
    harness.run(WHISPER)
    first = world.latest("whisper")
    first.exit(1, OOM)
    harness.run(WHISPER)
    cpu = world.latest("whisper", "cpu")
    assert cpu.alive
    assert harness.sup.status(WHISPER) is EngineState.CPU_FALLBACK
    assert harness.sup.reason(WHISPER) == "oom"
    assert harness.sup.whisper_url == f"http://127.0.0.1:{cpu.port}"
    assert harness.events_for(WHISPER) == [
        (EngineState.READY, "ok"),
        (EngineState.RESTARTING, "oom"),
        (EngineState.CPU_FALLBACK, "oom"),
    ]
    # The retry launches beside the CPU process: it does not become healthy right away.
    world.on_spawn = lambda proc: setattr(proc, "healthy", False)
    harness.run(WHISPER, 59.0)
    assert len(world.spawned("whisper", "vulkan")) == 1
    harness.run(WHISPER, 1.5)
    trial = world.latest("whisper", "vulkan")
    assert trial is not first and trial.alive and cpu.alive
    assert harness.sup.whisper_url == f"http://127.0.0.1:{cpu.port}"
    assert harness.sup.status(WHISPER) is EngineState.CPU_FALLBACK
    trial.healthy = True
    harness.run(WHISPER, 1.0)
    assert harness.sup.status(WHISPER) is EngineState.READY
    assert harness.sup.reason(WHISPER) == "ok"
    assert harness.sup.whisper_url == f"http://127.0.0.1:{trial.port}"
    assert trial.inference_calls == 1
    assert not cpu.alive and cpu.killed
    assert harness.events_for(WHISPER)[-1] == (EngineState.READY, "ok")


def test_whisper_retry_that_ooms_again_keeps_cpu_and_reschedules(harness, world):
    harness.run(WHISPER)
    world.latest("whisper").exit(1, OOM)
    harness.run(WHISPER)
    cpu = world.latest("whisper", "cpu")
    world.on_spawn = lambda proc: proc.exit(1, OOM) if proc.variant == "vulkan" else None
    harness.run(WHISPER, 60.5)
    assert len(world.spawned("whisper", "vulkan")) == 2
    assert harness.sup.status(WHISPER) is EngineState.CPU_FALLBACK
    assert harness.sup.reason(WHISPER) == "oom"
    assert cpu.alive and harness.sup.whisper_url == f"http://127.0.0.1:{cpu.port}"
    harness.run(WHISPER, 60.5)
    assert len(world.spawned("whisper", "vulkan")) == 3


def test_whisper_retry_that_finds_no_device_stops_retrying(harness, world):
    harness.run(WHISPER)
    world.latest("whisper").exit(1, OOM)
    harness.run(WHISPER)
    world.on_spawn = lambda proc: proc.write_log(NO_DEVICES) if proc.variant == "vulkan" else None
    harness.run(WHISPER, 61.0)
    assert len(world.spawned("whisper", "vulkan")) == 2
    assert harness.sup.status(WHISPER) is EngineState.CPU_FALLBACK
    assert harness.sup.reason(WHISPER) == "no_vulkan_gpu"
    harness.run(WHISPER, 7200.0)
    assert len(world.spawned("whisper", "vulkan")) == 2


def test_llama_oom_pauses_and_retries_every_minute(harness, world):
    harness.run(LLAMA)
    first = world.latest("llama")
    first.exit(1, OOM)
    harness.run(LLAMA)
    assert harness.sup.status(LLAMA) is EngineState.PAUSED
    assert harness.sup.reason(LLAMA) == "oom"
    assert harness.sup.llama_url is None
    assert harness.sup.cleanup_available is False
    assert world.live("llama") == []
    harness.run(LLAMA, 59.0)
    assert len(world.spawned("llama")) == 1
    world.on_spawn = lambda proc: proc.exit(1, OOM)
    harness.run(LLAMA, 1.5)
    assert len(world.spawned("llama")) == 2
    assert harness.sup.status(LLAMA) is EngineState.PAUSED
    world.on_spawn = None
    harness.run(LLAMA, 60.5)
    assert len(world.spawned("llama")) == 3
    assert harness.sup.status(LLAMA) is EngineState.READY
    assert harness.sup.reason(LLAMA) == "ok"
    assert harness.sup.cleanup_available is True
    assert harness.sup.llama_url == f"http://127.0.0.1:{world.latest('llama').port}"
    assert harness.events_for(LLAMA) == [
        (EngineState.READY, "ok"),
        (EngineState.PAUSED, "oom"),
        (EngineState.READY, "ok"),
    ]


def test_oom_at_runtime_is_classified_like_a_startup_failure(harness, world):
    harness.run(LLAMA)
    proc = world.latest("llama")
    proc.write_log("unable to allocate Vulkan buffer\n")
    proc.exit(3)
    harness.run(LLAMA)
    assert harness.sup.status(LLAMA) is EngineState.PAUSED
    assert harness.sup.reason(LLAMA) == "oom"


# Crashes and backoff ----------------------------------------------------------------


def test_backoff_schedule_is_exactly_1_2_5_10_30_30(tmp_path, world):
    harness = Harness(tmp_path, world, max_consecutive_failures=100)
    spawn_times: list[float] = []

    def on_spawn(proc: FakePopen) -> None:
        spawn_times.append(harness.clock.now())
        proc.exit(1, CRASH)

    world.on_spawn = on_spawn
    harness.run(LLAMA, 100.0)
    gaps = [round(b - a, 3) for a, b in itertools.pairwise(spawn_times)]
    assert gaps[:6] == [1, 2, 5, 10, 30, 30]
    assert harness.sup.status(LLAMA) is EngineState.RESTARTING
    assert harness.sup.reason(LLAMA) == "crash"
    assert harness.sup.llama_url is None
    assert all(p.variant == "vulkan" for p in world.processes)


def test_three_failures_then_cpu_fallback_with_ten_minute_vulkan_retry(harness, world):
    spawn_times: dict[str, list[float]] = {"vulkan": [], "cpu": []}

    def on_spawn(proc: FakePopen) -> None:
        spawn_times[proc.variant].append(harness.clock.now())
        if proc.variant == "vulkan":
            proc.exit(1, CRASH)

    world.on_spawn = on_spawn
    harness.run(LLAMA, 30.0)
    assert len(world.spawned("llama", "vulkan")) == 3
    assert spawn_times["vulkan"] == [1000.0, 1001.0, 1003.0]
    cpu = world.latest("llama", "cpu")
    assert cpu.alive
    assert spawn_times["cpu"] == [1003.0]
    assert harness.sup.status(LLAMA) is EngineState.CPU_FALLBACK
    assert harness.sup.reason(LLAMA) == "crash"
    assert harness.sup.llama_url == f"http://127.0.0.1:{cpu.port}"
    assert harness.sup.cleanup_available is False
    assert harness.events_for(LLAMA) == [
        (EngineState.RESTARTING, "crash"),
        (EngineState.CPU_FALLBACK, "crash"),
    ]
    # The Vulkan retry is due 600 s after the CPU build became ready (at 1003.0).
    harness.run(LLAMA, 1003.0 + 599.0 - harness.clock.now())
    assert len(world.spawned("llama", "vulkan")) == 3
    harness.run(LLAMA, 2.0)
    assert len(world.spawned("llama", "vulkan")) == 4
    assert spawn_times["vulkan"][-1] == 1603.0
    assert harness.sup.status(LLAMA) is EngineState.CPU_FALLBACK
    assert cpu.alive
    # The retry crashed again: the next one is another ten minutes out.
    harness.run(LLAMA, 1603.0 + 599.0 - harness.clock.now())
    assert len(world.spawned("llama", "vulkan")) == 4
    harness.run(LLAMA, 2.0)
    assert len(world.spawned("llama", "vulkan")) == 5


def test_successful_vulkan_retry_from_crash_fallback_switches_back(harness, world):
    world.on_spawn = lambda proc: proc.exit(1, CRASH) if proc.variant == "vulkan" else None
    harness.run(LLAMA, 30.0)
    cpu = world.latest("llama", "cpu")
    world.on_spawn = None
    harness.run(LLAMA, 601.0)
    trial = world.latest("llama", "vulkan")
    assert harness.sup.status(LLAMA) is EngineState.READY
    assert harness.sup.llama_url == f"http://127.0.0.1:{trial.port}"
    assert harness.sup.cleanup_available is True
    assert not cpu.alive


def test_consecutive_failure_counter_resets_on_ready(harness, world):
    for _ in range(5):
        harness.run(LLAMA, 40.0)
        assert harness.sup.status(LLAMA) is EngineState.READY
        world.latest("llama").exit(1, CRASH)
    harness.run(LLAMA, 40.0)
    assert harness.sup.status(LLAMA) is EngineState.READY
    assert all(p.variant == "vulkan" for p in world.processes)


def test_crash_after_ready_goes_restarting_then_ready(harness, world):
    harness.run(WHISPER)
    world.latest("whisper").exit(139, "Access violation\n")
    harness.run(WHISPER)
    assert harness.sup.status(WHISPER) is EngineState.RESTARTING
    assert harness.sup.reason(WHISPER) == "crash"
    assert harness.sup.whisper_url is None
    harness.run(WHISPER, 0.5)
    assert len(world.spawned("whisper")) == 1
    harness.run(WHISPER, 0.6)
    assert len(world.spawned("whisper")) == 2
    assert harness.sup.status(WHISPER) is EngineState.READY
    assert harness.events_for(WHISPER) == [
        (EngineState.READY, "ok"),
        (EngineState.RESTARTING, "crash"),
        (EngineState.READY, "ok"),
    ]


def test_dll_not_found_exit_code_switches_to_cpu_without_retry(harness, world):
    world.on_spawn = lambda p: p.exit(STATUS_DLL_NOT_FOUND) if p.variant == "vulkan" else None
    harness.run(WHISPER)
    assert harness.sup.status(WHISPER) is EngineState.CPU_FALLBACK
    assert harness.sup.reason(WHISPER) == "dll_not_found"
    assert harness.sup.variant(WHISPER) == "cpu"
    harness.run(WHISPER, 7200.0)
    assert len(world.spawned("whisper", "vulkan")) == 1


def test_negative_exit_code_form_of_dll_not_found_is_recognized():
    assert classify_failure(STATUS_DLL_NOT_FOUND - (1 << 32), "") == "dll_not_found"
    assert classify_failure(STATUS_DLL_NOT_FOUND, "") == "dll_not_found"
    assert classify_failure(1, NO_DEVICES) == "no_vulkan_gpu"
    assert classify_failure(None, OOM) == "oom"
    assert classify_failure(1, CRASH) == "crash"
    assert classify_failure(None, "") == "crash"


def test_cpu_variant_failure_is_failed(tmp_path, world):
    harness = Harness(tmp_path, world, gpu=NO_GPU)
    harness.run(LLAMA)
    world.latest("llama").exit(1, CRASH)
    harness.run(LLAMA, 3600.0)
    assert harness.sup.status(LLAMA) is EngineState.FAILED
    assert harness.sup.reason(LLAMA) == "crash"
    assert harness.sup.llama_url is None
    assert len(world.processes) == 1


def test_cpu_fallback_whose_cpu_build_dies_is_failed(harness, world):
    world.on_spawn = lambda proc: proc.exit(1, NO_DEVICES) if proc.variant == "vulkan" else None
    harness.run(WHISPER)
    assert harness.sup.status(WHISPER) is EngineState.CPU_FALLBACK
    world.latest("whisper", "cpu").exit(1, CRASH)
    harness.run(WHISPER, 3600.0)
    assert harness.sup.status(WHISPER) is EngineState.FAILED
    assert harness.events_for(WHISPER)[-1] == (EngineState.FAILED, "crash")


def test_failed_engine_is_relaunched_by_restart(tmp_path, world):
    harness = Harness(tmp_path, world, gpu=NO_GPU)
    harness.run(LLAMA)
    world.latest("llama").exit(1, CRASH)
    harness.run(LLAMA)
    assert harness.sup.status(LLAMA) is EngineState.FAILED
    harness.sup.restart(LLAMA)
    harness.run(LLAMA)
    assert harness.sup.status(LLAMA) is EngineState.CPU_FALLBACK
    assert len(world.processes) == 2


def test_spawn_error_counts_as_a_crash(harness, world):
    world.spawn_error = OSError("exe missing")
    harness.run(LLAMA, 0.5)
    assert harness.sup.status(LLAMA) is EngineState.RESTARTING
    assert harness.sup.reason(LLAMA) == "crash"
    assert world.processes == []
    world.spawn_error = None
    harness.run(LLAMA, 2.0)
    assert harness.sup.status(LLAMA) is EngineState.READY


def test_persistent_spawn_error_ends_in_failed(harness, world):
    world.spawn_error = OSError("exe missing")
    harness.run(LLAMA, 10.0)
    # Three Vulkan launch failures, then the CPU launch fails as well.
    assert harness.sup.status(LLAMA) is EngineState.FAILED
    assert harness.events_for(LLAMA)[-2:] == [
        (EngineState.RESTARTING, "crash"),
        (EngineState.FAILED, "crash"),
    ]


def test_startup_timeout_is_a_crash(tmp_path, world):
    harness = Harness(tmp_path, world, startup_timeout_s=30.0)
    world.on_spawn = lambda proc: setattr(proc, "healthy", False)
    harness.run(LLAMA, 29.0)
    assert len(world.spawned("llama")) == 1
    harness.run(LLAMA, 2.0)
    assert not world.processes[0].alive
    assert harness.sup.status(LLAMA) is EngineState.RESTARTING
    assert harness.sup.reason(LLAMA) == "crash"


def test_warm_up_failure_is_a_failure(harness, world):
    world.on_spawn = lambda proc: setattr(proc, "chat_error", RuntimeError("boom"))
    harness.run(LLAMA, 0.0)
    assert not world.processes[0].alive
    assert harness.sup.status(LLAMA) is EngineState.RESTARTING
    assert harness.sup.llama_url is None


def test_bind_failure_relaunches_on_a_new_port_without_counting(harness, world):
    ports: list[int] = []

    def on_spawn(proc: FakePopen) -> None:
        ports.append(proc.port)
        if len(ports) == 1:
            proc.exit(1, "couldn't bind HTTP server socket, hostname: 127.0.0.1, port: 1\n")

    world.on_spawn = on_spawn
    harness.run(LLAMA)
    assert len(ports) == 2 and ports[0] != ports[1]
    assert harness.sup.status(LLAMA) is EngineState.READY
    assert harness.events_for(LLAMA) == [(EngineState.READY, "ok")]


# Health, idle unload, commands -----------------------------------------------------


def test_health_failure_while_ready_restarts(harness, world):
    harness.run(WHISPER)
    proc = world.latest("whisper")
    calls = proc.health_calls
    harness.run(WHISPER, 9.5)
    assert proc.health_calls == calls
    proc.healthy = False
    harness.run(WHISPER, 1.0)
    assert proc.health_calls == calls + 1
    assert not proc.alive and proc.killed
    assert harness.sup.status(WHISPER) is EngineState.RESTARTING
    assert harness.sup.reason(WHISPER) == "crash"
    harness.run(WHISPER, 2.0)
    assert harness.sup.status(WHISPER) is EngineState.READY
    assert world.latest("whisper") is not proc


def test_health_is_checked_every_interval(harness, world):
    harness.run(LLAMA)
    proc = world.latest("llama")
    base = proc.health_calls
    harness.run(LLAMA, 35.0)
    assert proc.health_calls == base + 3


def test_idle_unload_and_ensure_ready(tmp_path, world):
    harness = Harness(tmp_path, world, idle_unload_minutes=2)
    harness.run_both()
    harness.run_both(60.0)
    harness.sup.note_activity()
    harness.run_both(100.0)
    assert harness.sup.status(WHISPER) is EngineState.READY
    harness.run_both(25.0)
    assert harness.sup.status(WHISPER) is EngineState.UNLOADED
    assert harness.sup.status(LLAMA) is EngineState.UNLOADED
    assert harness.sup.reason(WHISPER) == "idle"
    assert harness.sup.whisper_url is None and harness.sup.llama_url is None
    assert world.live() == []
    harness.run_both(3600.0)
    assert len(world.processes) == 2
    harness.sup.ensure_ready()
    harness.run_both()
    assert len(world.processes) == 4
    assert all(p.variant == "vulkan" for p in world.processes[2:])
    assert harness.sup.status(WHISPER) is EngineState.READY
    assert harness.sup.status(LLAMA) is EngineState.READY
    assert harness.events_for(LLAMA) == [
        (EngineState.READY, "ok"),
        (EngineState.UNLOADED, "idle"),
        (EngineState.STARTING, "ok"),
        (EngineState.READY, "ok"),
    ]


def test_ensure_ready_reloads_with_fresh_classification(tmp_path, world):
    harness = Harness(tmp_path, world, idle_unload_minutes=1)
    world.on_spawn = lambda proc: proc.exit(1, NO_DEVICES) if proc.variant == "vulkan" else None
    harness.run(WHISPER)
    assert harness.sup.status(WHISPER) is EngineState.CPU_FALLBACK
    harness.run(WHISPER, 61.0)
    assert harness.sup.status(WHISPER) is EngineState.UNLOADED
    world.on_spawn = None
    harness.sup.ensure_ready()
    harness.run(WHISPER)
    assert harness.sup.status(WHISPER) is EngineState.READY
    assert world.latest("whisper").variant == "vulkan"


def test_ensure_ready_is_a_no_op_while_loaded(harness, world):
    harness.run_both()
    harness.sup.ensure_ready()
    harness.run_both()
    assert len(world.processes) == 2


def test_idle_unload_off_by_default(harness, world):
    harness.run_both()
    harness.run_both(24 * 3600.0)
    assert harness.sup.status(WHISPER) is EngineState.READY
    assert len(world.processes) == 2


def test_idle_unload_minutes_can_change_at_runtime(harness, world):
    """The setting is editable while the app runs, so the loop reads it on every step."""
    harness.run_both()
    harness.run_both(3600.0)
    assert harness.sup.status(WHISPER) is EngineState.READY
    harness.sup.set_idle_unload_minutes(1)
    harness.sup.note_activity()
    harness.run_both(30.0)
    assert harness.sup.status(WHISPER) is EngineState.READY
    harness.run_both(35.0)
    assert harness.sup.status(WHISPER) is EngineState.UNLOADED
    assert harness.sup.status(LLAMA) is EngineState.UNLOADED
    harness.sup.set_idle_unload_minutes(0)
    harness.sup.ensure_ready()
    harness.run_both()
    assert harness.sup.status(WHISPER) is EngineState.READY
    harness.run_both(24 * 3600.0)
    assert harness.sup.status(WHISPER) is EngineState.READY


def test_restart_replaces_the_process(harness, world):
    harness.run(WHISPER)
    old = world.latest("whisper")
    harness.sup.restart(WHISPER)
    harness.run(WHISPER)
    new = world.latest("whisper")
    assert new is not old and new.alive and not old.alive
    assert harness.sup.status(WHISPER) is EngineState.READY
    assert harness.sup.whisper_url == f"http://127.0.0.1:{new.port}"
    assert harness.events_for(WHISPER) == [
        (EngineState.READY, "ok"),
        (EngineState.STARTING, "ok"),
        (EngineState.READY, "ok"),
    ]


def test_restart_from_cpu_fallback_tries_vulkan_again(harness, world):
    world.on_spawn = lambda proc: proc.exit(1, NO_DEVICES) if proc.variant == "vulkan" else None
    harness.run(LLAMA)
    assert harness.sup.status(LLAMA) is EngineState.CPU_FALLBACK
    world.on_spawn = None
    harness.sup.restart(LLAMA)
    harness.run(LLAMA)
    assert harness.sup.status(LLAMA) is EngineState.READY
    assert world.latest("llama").variant == "vulkan"


def test_set_gpu_restarts_both_with_the_new_index(harness, world):
    harness.run_both()
    assert world.processes[0].env["GGML_VK_VISIBLE_DEVICES"] == "0"
    other = GpuSelection(raw_index=1, name="AMD Radeon(TM) 610M", memory_mb=48688, devices=())
    harness.sup.set_gpu(other)
    harness.run_both()
    assert len(world.processes) == 4
    assert world.live() == world.processes[2:]
    assert all(p.env["GGML_VK_VISIBLE_DEVICES"] == "1" for p in world.processes[2:])
    assert harness.sup.status(WHISPER) is EngineState.READY


def test_set_gpu_to_none_moves_both_to_cpu(harness, world):
    harness.run_both()
    harness.sup.set_gpu(NO_GPU)
    harness.run_both()
    assert [p.variant for p in world.live()] == ["cpu", "cpu"]
    assert harness.sup.status(LLAMA) is EngineState.CPU_FALLBACK


# Launch details -------------------------------------------------------------------


def test_whisper_and_llama_arguments(harness, world):
    harness.run_both()
    whisper = world.latest("whisper")
    llama = world.latest("llama")
    assert whisper.exe == harness.paths.whisper_exe("vulkan")
    assert whisper.arg_after("-m") == str(harness.paths.whisper_model)
    assert whisper.arg_after("--host") == "127.0.0.1"
    assert "--vad" in whisper.args
    assert whisper.arg_after("--vad-model") == str(harness.paths.vad_model)
    assert int(whisper.arg_after("-t")) >= 1
    assert "--convert" not in whisper.args
    assert "--print-realtime" not in whisper.args
    assert "-pr" not in whisper.args
    assert llama.exe == harness.paths.llama_exe("vulkan")
    assert llama.arg_after("-m") == str(harness.paths.llama_model)
    assert llama.arg_after("--host") == "127.0.0.1"
    assert llama.arg_after("-c") == "4096"
    assert llama.arg_after("-ngl") == "99"
    assert whisper.cwd == harness.paths.vulkan_dir
    assert whisper.stdout_path == harness.paths.log_dir / "whisper.log"
    assert whisper.stderr_path == whisper.stdout_path
    assert llama.stdout_path == harness.paths.log_dir / "llama.log"


def test_cpu_llama_uses_zero_gpu_layers(tmp_path, world):
    harness = Harness(tmp_path, world, gpu=NO_GPU)
    harness.run(LLAMA)
    llama = world.latest("llama")
    assert llama.exe == harness.paths.llama_exe("cpu")
    assert llama.arg_after("-ngl") == "0"


def test_ports_are_unique_and_taken_from_the_loopback_range(harness, world):
    harness.run_both()
    whisper, llama = world.latest("whisper"), world.latest("llama")
    assert whisper.port != llama.port
    assert 1024 < whisper.port < 65536


def test_both_pids_share_one_job(harness, world):
    harness.run_both()
    assert len(world.jobs) == 1
    assert sorted(world.jobs[0].pids) == sorted(p.pid for p in world.processes)
    assert harness.sup.pid(WHISPER) == world.latest("whisper").pid


def test_log_rotation_at_five_megabytes_keeps_five_files(harness, world):
    log = harness.paths.log_dir / "llama.log"
    log.parent.mkdir(parents=True)
    log.write_bytes(b"x" * LOG_MAX_BYTES)
    for i in range(1, LOG_KEEP):
        (harness.paths.log_dir / f"llama.log.{i}").write_bytes(b"%d" % i)
    harness.run(LLAMA)
    names = sorted(p.name for p in harness.paths.log_dir.iterdir())
    assert names == ["llama.log"] + [f"llama.log.{i}" for i in range(1, LOG_KEEP)]
    assert (harness.paths.log_dir / "llama.log.1").stat().st_size == LOG_MAX_BYTES
    assert (harness.paths.log_dir / f"llama.log.{LOG_KEEP - 1}").read_bytes() == b"%d" % (
        LOG_KEEP - 2
    )
    assert log.stat().st_size < LOG_MAX_BYTES


def test_small_logs_are_not_rotated(harness, world):
    log = harness.paths.log_dir / "whisper.log"
    log.parent.mkdir(parents=True)
    log.write_bytes(b"old line\n")
    harness.run(WHISPER)
    assert not (harness.paths.log_dir / "whisper.log.1").exists()
    assert log.read_bytes().startswith(b"old line\n")


def test_classification_only_reads_the_lines_of_the_failing_process(harness, world):
    harness.run(LLAMA)
    first = world.latest("llama")
    first.exit(1, OOM)
    harness.run(LLAMA)
    assert harness.sup.status(LLAMA) is EngineState.PAUSED
    harness.run(LLAMA, 61.0)
    assert harness.sup.status(LLAMA) is EngineState.READY
    world.latest("llama").exit(1, CRASH)
    harness.run(LLAMA)
    # The old OOM lines are still in the file but belong to the earlier process.
    assert harness.sup.status(LLAMA) is EngineState.RESTARTING
    assert harness.sup.reason(LLAMA) == "crash"


# Reasons, callbacks, threads -------------------------------------------------------


def test_every_reason_string():
    assert REASONS == (
        "ok",
        "no_vulkan_gpu",
        "oom",
        "crash",
        "dll_not_found",
        "idle",
        "no_model",
        "cpu_selected",
    )


# Models for the dictation languages (batch 5) -----------------------------------------------

NGRAM = ("--spec-type", "ngram-simple", "--spec-ngram-simple-size-n", "2")


def test_engine_args_from_the_chosen_models_follow_the_fixed_arguments(tmp_path, world):
    paths = replace(make_paths(tmp_path), whisper_args=("--beam-size", "5"), llama_args=NGRAM)
    harness = Harness(tmp_path, world, paths=paths)
    harness.run_both()
    whisper = world.latest("whisper")
    llama = world.latest("llama")
    assert whisper.args[-2:] == ["--beam-size", "5"]
    assert whisper.arg_after("-t") is not None
    assert llama.args[-len(NGRAM):] == list(NGRAM)
    assert llama.arg_after("-ngl") == "99"


def test_cpu_hardware_launches_both_cpu_builds_on_a_machine_with_a_gpu(tmp_path, world):
    harness = Harness(tmp_path, world, cpu_only=True, cpu_cleanup_allowed=True)
    harness.run_both()
    assert [p.variant for p in world.processes] == ["cpu", "cpu"]
    assert world.latest("llama").arg_after("-ngl") == "0"
    for engine in (WHISPER, LLAMA):
        assert harness.sup.status(engine) is EngineState.CPU_FALLBACK
        assert harness.sup.reason(engine) == "cpu_selected"
    assert harness.events_for(LLAMA) == [
        (EngineState.STARTING, "cpu_selected"),
        (EngineState.CPU_FALLBACK, "cpu_selected"),
    ]
    assert harness.sup.cpu_cleanup_allowed is True
    assert harness.sup.cleanup_available is True
    assert harness.sup.llama_url is not None
    harness.run_both(3600.0)
    assert len(world.processes) == 2


def test_cpu_hardware_without_any_gpu_keeps_the_no_vulkan_gpu_reason(tmp_path, world):
    harness = Harness(tmp_path, world, gpu=NO_GPU, cpu_only=True)
    harness.run_both()
    assert harness.sup.reason(LLAMA) == "no_vulkan_gpu"


def test_cpu_cleanup_is_not_allowed_by_default(harness):
    assert harness.sup.cpu_cleanup_allowed is False


def test_a_new_cleanup_model_relaunches_llama_only(harness, world):
    harness.run_both()
    whisper = world.latest("whisper")
    old_llama = world.latest("llama")
    new_model = harness.paths.whisper_model.parent / "gemma.gguf"
    new_paths = replace(harness.paths, llama_model=new_model, llama_args=NGRAM)
    changed = harness.sup.set_models(new_paths, cpu_only=False, cpu_cleanup_allowed=False)
    assert changed == (CLEANUP,)
    harness.run_both()
    assert old_llama.killed
    assert whisper.alive and not whisper.killed
    assert world.spawned("whisper") == [whisper]
    llama = world.latest("llama")
    assert llama is not old_llama
    assert llama.arg_after("-m") == str(new_model)
    assert llama.args[-len(NGRAM):] == list(NGRAM)
    assert harness.sup.status(LLAMA) is EngineState.READY


def test_the_same_models_relaunch_nothing(harness, world):
    harness.run_both()
    before = list(world.processes)
    changed = harness.sup.set_models(
        replace(harness.paths), cpu_only=False, cpu_cleanup_allowed=True
    )
    harness.run_both(30.0)
    assert changed == ()
    assert world.processes == before
    assert all(proc.alive for proc in before)
    assert harness.sup.cpu_cleanup_allowed is True


def test_a_new_speech_model_relaunches_whisper_only(harness, world):
    harness.run_both()
    llama = world.latest("llama")
    new_paths = replace(harness.paths, whisper_model=harness.paths.whisper_model.with_name("sq.bin"))
    assert harness.sup.set_models(new_paths, cpu_only=False, cpu_cleanup_allowed=False) == (
        SPEECH_1,
    )
    harness.run_both()
    assert world.latest("whisper").arg_after("-m").endswith("sq.bin")
    assert world.spawned("llama") == [llama] and llama.alive


def test_switching_to_cpu_hardware_relaunches_both_engines_on_cpu(harness, world):
    harness.run_both()
    changed = harness.sup.set_models(harness.paths, cpu_only=True, cpu_cleanup_allowed=True)
    assert changed == (SPEECH_1, CLEANUP)
    harness.run_both()
    assert [p.variant for p in world.live()] == ["cpu", "cpu"]
    assert harness.sup.reason(LLAMA) == "cpu_selected"


def test_switching_gpu_and_models_clears_cpu_plan_in_one_reconfiguration(harness, world):
    harness.sup.set_models(harness.paths, cpu_only=True, cpu_cleanup_allowed=True)
    harness.run_both()
    other = GpuSelection(raw_index=1, name="Other GPU", memory_mb=16000, devices=())
    changed = harness.sup.set_models(
        harness.paths, cpu_only=False, cpu_cleanup_allowed=False, gpu=other)
    assert changed == (SPEECH_1, CLEANUP)
    harness.run_both()
    assert len(world.live()) == 2
    assert all(p.variant == "vulkan" for p in world.live())
    assert all(p.env["GGML_VK_VISIBLE_DEVICES"] == "1" for p in world.live())


def test_switching_gpu_relaunches_even_when_models_and_hardware_class_match(harness, world):
    harness.run_both()
    other = GpuSelection(raw_index=1, name="Other GPU", memory_mb=16000, devices=())
    changed = harness.sup.set_models(
        harness.paths, cpu_only=False, cpu_cleanup_allowed=False, gpu=other)
    assert changed == (SPEECH_1, CLEANUP)
    harness.run_both()
    assert len(world.processes) == 4
    assert all(p.env["GGML_VK_VISIBLE_DEVICES"] == "1" for p in world.live())


def test_dropping_the_cleanup_model_stops_llama_with_no_model(harness, world):
    harness.run_both()
    llama = world.latest("llama")
    harness.sup.set_models(
        replace(harness.paths, llama_model=None), cpu_only=False, cpu_cleanup_allowed=False
    )
    harness.run_both()
    assert llama.killed
    assert harness.sup.status(LLAMA) is EngineState.FAILED
    assert harness.sup.reason(LLAMA) == "no_model"
    assert harness.sup.llama_url is None
    assert world.spawned("llama") == [llama]


def make_paths_for_llama(harness: Harness):
    return replace(harness.paths, llama_model=harness.paths.whisper_model.with_name("new.gguf"))


def test_a_cleanup_model_after_no_model_launches_llama(tmp_path, world):
    harness = no_model_harness(tmp_path, world)
    harness.run_both()
    assert world.spawned("llama") == []
    harness.sup.set_models(make_paths_for_llama(harness), cpu_only=False, cpu_cleanup_allowed=False)
    harness.run_both()
    assert harness.sup.status(LLAMA) is EngineState.READY
    assert len(world.spawned("llama")) == 1


def test_a_model_change_while_unloaded_waits_for_the_next_press(tmp_path, world):
    harness = Harness(tmp_path, world, idle_unload_minutes=1)
    harness.run_both()
    harness.run_both(120.0)
    assert harness.sup.status(LLAMA) is EngineState.UNLOADED
    count = len(world.processes)
    new_paths = make_paths_for_llama(harness)
    harness.sup.set_models(new_paths, cpu_only=False, cpu_cleanup_allowed=False)
    harness.run_both(5.0)
    assert len(world.processes) == count
    assert harness.sup.status(LLAMA) is EngineState.UNLOADED
    harness.sup.ensure_ready()
    harness.run_both()
    assert harness.sup.status(LLAMA) is EngineState.READY
    assert world.latest("llama").arg_after("-m") == str(new_paths.llama_model)


# No cleanup model (spec 14.1, 16) -----------------------------------------------------------


def no_model_harness(tmp_path: Path, world: FakeWorld, **kwargs) -> Harness:
    paths = replace(make_paths(tmp_path), llama_model=None)
    return Harness(tmp_path, world, paths=paths, **kwargs)


def test_without_a_cleanup_model_llama_is_failed_with_no_model_and_never_launched(tmp_path, world):
    harness = no_model_harness(tmp_path, world)
    assert harness.sup.status(LLAMA) is EngineState.FAILED
    assert harness.sup.reason(LLAMA) == "no_model"
    harness.run_both()
    assert [p.engine for p in world.processes] == ["whisper"]
    assert harness.sup.status(WHISPER) is EngineState.READY
    assert harness.sup.status(LLAMA) is EngineState.FAILED
    assert harness.sup.reason(LLAMA) == "no_model"
    assert harness.sup.llama_url is None
    assert harness.sup.cleanup_available is False
    assert harness.sup.pid(LLAMA) is None
    # Announced once, on the supervisor thread's first step, like every other change.
    assert harness.events_for(LLAMA) == [(EngineState.FAILED, "no_model")]


def test_without_a_cleanup_model_restart_gpu_change_and_reload_never_launch_llama(tmp_path, world):
    harness = no_model_harness(tmp_path, world, idle_unload_minutes=1)
    harness.run_both()
    harness.sup.restart(LLAMA)
    harness.run_both()
    other = GpuSelection(raw_index=1, name="Other", memory_mb=4096, devices=())
    harness.sup.set_gpu(other)
    harness.run_both()
    harness.run_both(120.0)
    assert harness.sup.status(WHISPER) is EngineState.UNLOADED
    harness.sup.ensure_ready()
    harness.run_both()
    assert harness.sup.status(WHISPER) is EngineState.READY
    assert world.spawned("llama") == []
    assert harness.sup.status(LLAMA) is EngineState.FAILED
    assert harness.sup.reason(LLAMA) == "no_model"
    assert harness.events_for(LLAMA) == [(EngineState.FAILED, "no_model")]


def test_without_a_cleanup_model_wait_ready_for_llama_returns_false(tmp_path, world):
    harness = no_model_harness(tmp_path, world)
    harness.run_both()
    assert harness.sup.wait_ready(LLAMA, 0.0) is False


def test_cleanup_available_only_when_llama_is_ready_on_vulkan(harness, world):
    assert harness.sup.cleanup_available is False
    harness.run(LLAMA)
    assert harness.sup.cleanup_available is True
    world.on_spawn = lambda proc: proc.write_log(NO_DEVICES) if proc.variant == "vulkan" else None
    harness.sup.restart(LLAMA)
    harness.run(LLAMA)
    assert harness.sup.status(LLAMA) is EngineState.CPU_FALLBACK
    assert harness.sup.llama_url is not None
    assert harness.sup.cleanup_available is False


def test_status_callback_runs_outside_the_lock(tmp_path, world):
    seen = []

    class Reentrant(Harness):
        def on_status(self, engine, state, reason):
            seen.append((engine, state, self.sup.status(engine), self.sup.whisper_url))

    harness = Reentrant(tmp_path, world)
    harness.run(WHISPER)
    assert seen[-1][1] is EngineState.READY
    assert seen[-1][2] is EngineState.READY
    assert seen[-1][3] == harness.sup.whisper_url


def test_callback_exception_does_not_break_supervision(tmp_path, world):
    class Broken(Harness):
        def on_status(self, engine, state, reason):
            raise RuntimeError("ui gone")

    harness = Broken(tmp_path, world)
    harness.run_both()
    assert harness.sup.status(WHISPER) is EngineState.READY


def test_start_runs_threads_and_stop_kills_everything(tmp_path, world):
    events: list[tuple[Engine, EngineState, str]] = []
    paths = make_paths(tmp_path)
    sup = EngineSupervisor(
        paths,
        GPU,
        lambda engine, state, reason: events.append((engine, state, reason)),
        process_backend=world,
        whisper_client_factory=world.whisper_client,
        llama_client_factory=world.llama_client,
        clock=time.monotonic,
        sleeper=time.sleep,
    )
    assert sup.wait_ready(WHISPER, 0.05) is False
    sup.start()
    try:
        assert sup.wait_ready(WHISPER, 10.0)
        assert sup.wait_ready(LLAMA, 10.0)
        assert sup.status(WHISPER) is EngineState.READY
        assert sup.whisper_url and sup.llama_url
        assert threading.active_count() >= 3
    finally:
        sup.stop()
    assert world.live() == []
    assert world.jobs[0].closed
    assert (WHISPER, EngineState.READY, "ok") in events
    sup.stop()  # idempotent


def test_stop_without_start_closes_nothing_and_is_safe(harness, world):
    harness.sup.stop()
    assert world.jobs == []


def test_stop_after_manual_steps_kills_processes_and_closes_the_job(harness, world):
    harness.run_both()
    harness.sup.stop()
    assert world.live() == []
    assert world.jobs[0].closed
    assert harness.sup.whisper_url is None


# Several speech engines (B5-13 to B5-15) ---------------------------------------------------------


def two_speech_paths(tmp_path: Path) -> EnginePaths:
    base = make_paths(tmp_path)
    models = tmp_path / "models"
    return replace(
        base,
        whisper_model=models / "qwen-asr.gguf",
        whisper_runtime="llama-asr",
        whisper_args=("--mmproj", "{extra:0}", "--no-webui"),
        whisper_extra_files=(models / "mmproj.gguf",),
        whisper_languages=("en", "de"),
        extra_speech=(
            SpeechEngine("whisper-server", models / "flutra.bin", (), (), ("sq",)),
        ),
    )


class MultiHarness(Harness):
    def run_all(self, seconds: float = 0.0, max_steps: int = 10000) -> None:
        end = self.clock.now() + seconds
        for _ in range(max_steps):
            delays = [self.sup.step(engine) for engine in self.sup.engine_ids]
            if min(delays) <= 0:
                continue
            remaining = end - self.clock.now()
            if remaining <= 0:
                return
            self.clock.advance(min(min(delays), remaining))
        raise AssertionError("the supervision loops never settled")


def multi(tmp_path, world, **kwargs) -> MultiHarness:
    instance_events: list = []
    harness = MultiHarness(
        tmp_path,
        world,
        paths=two_speech_paths(tmp_path),
        on_instance_status=lambda *event: instance_events.append(event),
        **kwargs,
    )
    harness.instance_events = instance_events
    return harness


def by_model(world: FakeWorld, name: str) -> FakePopen:
    procs = [proc for proc in world.processes if proc.model == name]
    assert procs, f"no process for {name}"
    return procs[-1]


def test_one_process_per_speech_engine_plus_cleanup_each_with_its_own_log_and_port(
    tmp_path, world
):
    harness = multi(tmp_path, world)
    assert harness.sup.engine_ids == (SPEECH_1, SPEECH_2, CLEANUP)
    harness.run_all()
    qwen = by_model(world, "qwen-asr.gguf")
    flutra = by_model(world, "flutra.bin")
    cleanup = by_model(world, "cleanup.gguf")
    assert qwen.exe == harness.paths.llama_exe("vulkan")
    assert flutra.exe == harness.paths.whisper_exe("vulkan")
    assert qwen.arg_after("--mmproj") == str((tmp_path / "models" / "mmproj.gguf").absolute())
    assert "--no-webui" in qwen.args
    assert qwen.arg_after("-c") == "4096"
    assert "--vad" not in qwen.args
    assert flutra.arg_after("--vad-model") == str(harness.paths.vad_model)
    assert len({qwen.port, flutra.port, cleanup.port}) == 3
    assert qwen.stdout_path.name == "whisper.log"
    assert flutra.stdout_path.name == "whisper-2.log"
    assert cleanup.stdout_path.name == "llama.log"
    assert qwen.transcribe_calls == 1 and qwen.inference_calls == 0
    assert flutra.inference_calls == 1
    assert cleanup.chat_calls == 1
    assert len(world.jobs) == 1 and len(world.jobs[0].pids) == 3
    for engine in (SPEECH_1, SPEECH_2, CLEANUP):
        assert harness.sup.status(engine) is EngineState.READY
    assert harness.sup.url(SPEECH_1) == f"http://127.0.0.1:{qwen.port}"
    assert harness.sup.url(SPEECH_2) == f"http://127.0.0.1:{flutra.port}"
    assert harness.sup.whisper_url == harness.sup.url(SPEECH_1)
    assert harness.sup.pid(SPEECH_2) == flutra.pid


def test_the_speech_engines_describe_their_runtime_and_languages(tmp_path, world):
    harness = multi(tmp_path, world)
    assert harness.sup.speech_engines() == (
        SpeechSlot(SPEECH_1, "llama-asr", ("en", "de")),
        SpeechSlot(SPEECH_2, "whisper-server", ("sq",)),
    )


def test_the_speech_role_reports_the_worst_instance_and_announces_only_changes(tmp_path, world):
    harness = multi(tmp_path, world)
    harness.run_all()
    assert harness.events_for(WHISPER) == [(EngineState.READY, "ok")]
    assert (SPEECH_2, EngineState.READY, "ok") in harness.instance_events
    flutra = by_model(world, "flutra.bin")
    flutra.exit(1, CRASH)
    harness.sup.step(SPEECH_2)
    assert harness.sup.status(SPEECH_1) is EngineState.READY
    assert harness.sup.status(SPEECH_2) is EngineState.RESTARTING
    assert harness.sup.status(WHISPER) is EngineState.RESTARTING
    assert harness.sup.reason(WHISPER) == "crash"
    assert harness.events_for(WHISPER)[-1] == (EngineState.RESTARTING, "crash")
    assert harness.sup.wait_ready(WHISPER, 0) is False
    assert harness.sup.wait_ready(SPEECH_1, 0) is True
    harness.run_all(5.0)
    assert harness.sup.status(WHISPER) is EngineState.READY
    assert harness.events_for(WHISPER)[-1] == (EngineState.READY, "ok")
    assert by_model(world, "qwen-asr.gguf").alive


def test_restarting_the_speech_role_restarts_every_instance_and_one_id_only_itself(
    tmp_path, world
):
    harness = multi(tmp_path, world)
    harness.run_all()
    qwen, flutra = by_model(world, "qwen-asr.gguf"), by_model(world, "flutra.bin")
    harness.sup.restart(SPEECH_2)
    harness.run_all()
    assert flutra.killed and qwen.alive
    harness.sup.restart(WHISPER)
    harness.run_all()
    assert qwen.killed
    assert by_model(world, "flutra.bin").alive and by_model(world, "qwen-asr.gguf").alive
    assert by_model(world, "cleanup.gguf").chat_calls == 1


def test_a_second_speech_engine_is_added_and_removed_with_the_models(tmp_path, world):
    harness = Harness(tmp_path, world)
    harness.run_both()
    whisper = world.latest("whisper")
    added = harness.sup.set_models(
        replace(
            harness.paths,
            extra_speech=(SpeechEngine("whisper-server", tmp_path / "models" / "sq.bin"),),
        ),
        cpu_only=False,
        cpu_cleanup_allowed=False,
    )
    assert added == (SPEECH_2,)
    assert harness.sup.status(WHISPER) is EngineState.STARTING
    for _ in range(50):
        harness.sup.step(SPEECH_2)
    sq = by_model(world, "sq.bin")
    assert harness.sup.status(SPEECH_2) is EngineState.READY
    assert harness.sup.status(WHISPER) is EngineState.READY
    assert whisper.alive
    removed = harness.sup.set_models(harness.paths, cpu_only=False, cpu_cleanup_allowed=False)
    assert removed == (SPEECH_2,)
    assert sq.killed
    assert harness.sup.engine_ids == (SPEECH_1, CLEANUP)
    assert harness.sup.status(SPEECH_2) is EngineState.FAILED
    assert harness.sup.url(SPEECH_2) is None
    assert harness.sup.step(SPEECH_2) > 0
    assert whisper.alive


def test_a_started_supervisor_runs_an_added_engine_on_its_own_thread(tmp_path, world):
    harness = Harness(tmp_path, world)
    harness.sup = EngineSupervisor(
        harness.paths,
        GPU,
        harness.on_status,
        process_backend=world,
        whisper_client_factory=world.whisper_client,
        llama_client_factory=world.llama_client,
        llama_asr_client_factory=world.llama_asr_client,
    )
    harness.sup.start()
    try:
        assert harness.sup.wait_ready(WHISPER, 5.0)
        harness.sup.set_models(
            replace(
                harness.paths,
                extra_speech=(SpeechEngine("llama-asr", tmp_path / "models" / "fr.gguf"),),
            ),
            cpu_only=False,
            cpu_cleanup_allowed=False,
        )
        assert harness.sup.wait_ready(SPEECH_2, 5.0)
        assert harness.sup.wait_ready(WHISPER, 5.0)
        names = {thread.name for thread in threading.enumerate()}
        assert "engine-whisper-2" in names
    finally:
        harness.sup.stop()
    assert not any(proc.alive for proc in world.processes)


def test_an_unplanned_cpu_instance_makes_the_role_variant_cpu(tmp_path, world):
    harness = multi(tmp_path, world)
    world.on_spawn = lambda proc: (
        proc.write_log(NO_DEVICES) if proc.model == "flutra.bin" and proc.variant == "vulkan"
        else None
    )
    harness.run_all()
    assert harness.sup.variant(SPEECH_1) == "vulkan"
    assert harness.sup.variant(SPEECH_2) == "cpu"
    assert harness.sup.variant(WHISPER) == "cpu"
    assert harness.sup.status(WHISPER) is EngineState.CPU_FALLBACK
    assert harness.sup.reason(WHISPER) == "no_vulkan_gpu"


def test_gpu_verification_of_the_speech_role_needs_every_instance(tmp_path, world):
    harness = Harness(tmp_path, world)
    world.on_spawn = lambda proc: proc.write_log(FOUND_RTX) if proc.engine == "whisper" else None
    harness.run_both()
    assert harness.sup.gpu_verified(WHISPER) is True
    harness.sup.set_models(
        replace(
            harness.paths,
            extra_speech=(SpeechEngine("llama-asr", tmp_path / "models" / "fr.gguf"),),
        ),
        cpu_only=False,
        cpu_cleanup_allowed=False,
    )
    for _ in range(50):
        harness.sup.step(SPEECH_2)
    assert harness.sup.gpu_verified(SPEECH_1) is True
    assert harness.sup.gpu_verified(SPEECH_2) is None
    assert harness.sup.gpu_verified(WHISPER) is None


def test_a_llama_asr_warm_up_failure_is_a_launch_failure(tmp_path, world):
    world.on_spawn = lambda proc: setattr(
        proc, "transcribe_error", RuntimeError("bad audio") if proc.model == "qwen-asr.gguf" else None
    )
    harness = multi(tmp_path, world)
    harness.sup.step(SPEECH_1)
    harness.sup.step(SPEECH_1)
    assert harness.sup.status(SPEECH_1) is EngineState.RESTARTING
    assert by_model(world, "qwen-asr.gguf").killed


# Placeholders, performance cores and the gate inputs ---------------------------------------------


def test_a_cleanup_extra_file_placeholder_becomes_the_absolute_path(tmp_path, world):
    draft = tmp_path / "models" / "mtp.gguf"
    paths = replace(
        make_paths(tmp_path),
        llama_args=("--spec-type", "draft-mtp", "-md", "{extra:0}"),
        llama_extra_files=(draft,),
    )
    harness = Harness(tmp_path, world, paths=paths)
    harness.run(LLAMA)
    llama = world.latest("llama")
    assert llama.args[-4:] == ["--spec-type", "draft-mtp", "-md", str(draft.absolute())]


def test_a_placeholder_without_its_extra_file_fails_the_launch(tmp_path, world):
    paths = replace(make_paths(tmp_path), llama_args=("-md", "{extra:0}"))
    harness = Harness(tmp_path, world, paths=paths)
    harness.sup.step(LLAMA)
    assert world.spawned("llama") == []
    assert harness.sup.status(LLAMA) is EngineState.RESTARTING
    assert harness.sup.reason(LLAMA) == "crash"


def test_a_hybrid_processor_pins_every_engine_to_the_performance_cores(tmp_path, world):
    plan = CpuPlan(threads=8, affinity_mask=0x5555)
    harness = multi(tmp_path, world, gpu=NO_GPU, cpu_only=True, cpu_plan=plan)
    harness.run_all()
    assert len(world.processes) == 3
    for proc in world.processes:
        assert proc.affinity_mask == 0x5555
        assert proc.arg_after("-t") == "8"
    for name in ("qwen-asr.gguf", "cleanup.gguf"):
        proc = by_model(world, name)
        assert proc.arg_after("-C") == "5555"
        assert proc.arg_after("--cpu-strict") == "1"
        assert proc.arg_after("-ngl") == "0"
    assert "-C" not in by_model(world, "flutra.bin").args
    assert harness.sup.cpu_plan == plan


def test_without_a_hybrid_processor_nothing_is_pinned(harness, world):
    harness.run_both()
    for proc in world.processes:
        assert proc.affinity_mask is None
        assert "-C" not in proc.args
    assert "-t" not in world.latest("llama").args


def test_explicit_whisper_threads_win_over_the_plan(tmp_path, world):
    harness = Harness(tmp_path, world, whisper_threads=3, cpu_plan=CpuPlan(8, 0xFF))
    harness.run(WHISPER)
    assert world.latest("whisper").arg_after("-t") == "3"


def test_the_gate_inputs_follow_the_models(tmp_path, world):
    harness = Harness(tmp_path, world, cpu_only=True)
    assert harness.sup.cpu_only is True
    assert harness.sup.cleanup_languages is None
    harness.sup.set_models(
        replace(harness.paths, llama_languages=("en", "de")), cpu_only=False,
        cpu_cleanup_allowed=False,
    )
    assert harness.sup.cpu_only is False
    assert harness.sup.cleanup_languages == frozenset({"en", "de"})


def test_engine_paths_take_the_selection_choices(tmp_path):
    models = tmp_path / "models"

    def choice(kind, model_id, file, runtime, languages, extra=(), args=()):
        return ModelChoice(kind=kind, model_id=model_id, display_name=model_id,
                           languages=languages, reason="", installed=True, file=file,
                           extra_files=extra, engine_args=args, runtime=runtime)

    speech = (
        choice(ModelKind.ASR, "q", "q.gguf", "llama-asr", ("en", "de"), ("mm.gguf",),
               ("--mmproj", "{extra:0}")),
        choice(ModelKind.ASR, "f", "f.bin", "whisper-server", ("sq",)),
    )
    cleanup = choice(ModelKind.CLEANUP, "g", "g.gguf", "llama-server", ("en", "de", "sq"),
                     ("mtp.gguf",), ("-md", "{extra:0}"))
    paths = make_paths(tmp_path).with_speech(models, speech).with_cleanup(models, cleanup)
    assert paths.whisper_model == models / "q.gguf"
    assert paths.whisper_runtime == "llama-asr"
    assert paths.whisper_extra_files == (models / "mm.gguf",)
    assert paths.whisper_languages == ("de", "en")
    assert paths.extra_speech == (SpeechEngine("whisper-server", models / "f.bin", (), (), ("sq",)),)
    assert paths.llama_model == models / "g.gguf"
    assert paths.llama_extra_files == (models / "mtp.gguf",)
    assert paths.llama_languages == ("de", "en", "sq")
    assert paths.engine_ids == (SPEECH_1, SPEECH_2, CLEANUP)
    assert paths.log_path(SPEECH_2).name == "whisper-2.log"
    assert paths.exe(SPEECH_1, "cpu").name == "llama-server.exe"
    assert paths.exe(SPEECH_2, "cpu").name == "whisper-server.exe"
    assert paths.exe(LLAMA, "cpu").name == "llama-server.exe"
    unchanged = make_paths(tmp_path / "second")
    assert unchanged.with_speech(models, ()) == unchanged
    none = unchanged.with_cleanup(models, None)
    assert (none.llama_model, none.llama_args, none.llama_languages) == (None, (), ())


def test_engine_labels_name_the_languages():
    assert engine_label(WHISPER) == "Speech engine"
    assert engine_label(SPEECH_2, ["sq"]) == "Speech engine for Albanian"
    assert engine_label(SPEECH_1, ["en", "de"]) == "Speech engine for English and German"
    assert engine_label(CLEANUP) == "Cleanup engine"
    assert EngineId(WHISPER, 1).key == "whisper-2" and str(CLEANUP) == "llama"


# The writing engine ----------------------------------------------------------------------


def test_a_writing_model_of_its_own_gets_an_engine_of_its_own(tmp_path, world):
    from spells.engines import WRITER_ID, engine_label

    paths = replace(
        make_paths(tmp_path),
        llama_args=NGRAM,
        writer_model=tmp_path / "models" / "writer.gguf",
        writer_args=("--jinja",),
    )
    assert paths.engine_ids[-1] == WRITER_ID
    harness = Harness(tmp_path, world, paths=paths)
    assert harness.sup.writer_engine == WRITER_ID
    harness.run_both()
    harness.run(WRITER_ID)
    models = sorted(proc.model for proc in world.spawned("llama"))
    assert models == ["cleanup.gguf", "writer.gguf"]
    writer = next(proc for proc in world.spawned("llama") if proc.model == "writer.gguf")
    assert writer.args[-1] == "--jinja"
    assert "--spec-type" not in writer.args
    assert harness.sup.url(WRITER_ID) is not None
    assert harness.sup.url(WRITER_ID) != harness.sup.llama_url
    assert engine_label(WRITER_ID) == "Writing engine"


def test_without_a_writing_model_there_is_no_writing_engine(tmp_path, world):
    harness = Harness(tmp_path, world)
    assert harness.sup.writer_engine is None
    assert len(harness.paths.engine_ids) == 2


def test_dropping_the_writing_model_retires_its_engine(tmp_path, world):
    from spells.engines import WRITER_ID

    paths = replace(make_paths(tmp_path), writer_model=tmp_path / "models" / "writer.gguf")
    harness = Harness(tmp_path, world, paths=paths)
    harness.run_both()
    harness.run(WRITER_ID)
    changed = harness.sup.set_models(
        replace(paths, writer_model=None), cpu_only=False, cpu_cleanup_allowed=False
    )
    assert WRITER_ID in changed
    assert harness.sup.writer_engine is None
