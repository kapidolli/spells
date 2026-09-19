from __future__ import annotations

import json
from pathlib import Path

import pytest

from spells import calibrate
from spells.calibrate import (
    CALIBRATION_TEXT,
    GPU_WIN_RATIO,
    Calibrator,
    Run,
    SpeechCalibration,
)
from spells.engines import EnginePaths
from spells.gpu import NO_GPU, GpuDevice, GpuSelection
from spells.modelcatalog import Hardware, parse_catalog, select_models
from spells.models import CpuPlan

ARC = "Intel(R) Arc(TM) Pro Graphics"
WHISPER = "whisper-turbo"
QWEN = "qwen-asr"


def device(uma: bool | None, memory_mb: int = 37032, name: str = ARC) -> GpuSelection:
    found = GpuDevice(
        raw_index=0, list_name="Vulkan0", name=name, memory_mb=memory_mb, free_mb=memory_mb,
        uma=uma,
    )
    return GpuSelection(raw_index=0, name=name, memory_mb=memory_mb, devices=(found,))


def catalog():
    def entry(model_id, runtime, languages, gpu, cpu, file):
        extra = ["mmproj.gguf"] if runtime == "llama-asr" else []
        return {
            "id": model_id,
            "kind": "asr",
            "display_name": model_id,
            "file": file,
            "extra_files": extra,
            "size_bytes": 1000,
            "license": "MIT",
            "runtime": runtime,
            "languages": languages,
            "hardware": {"gpu": gpu, "cpu": cpu},
            "engine_args": ["--mmproj", "{extra:0}"] if extra else [],
        }

    return parse_catalog(
        {
            "schema_version": 1,
            "models": [
                entry(WHISPER, "whisper-server", {"en": 87, "de": 85, "sq": 48}, 300, 4500,
                      "whisper.bin"),
                entry(QWEN, "llama-asr", {"en": 92, "de": 88}, 400, 1300, "qwen.gguf"),
            ],
        }
    )


INSTALLED = frozenset({WHISPER, QWEN})


def measured(model_id, gpu_ms, cpu_ms, *, gpu_similarity=1.0, cpu_similarity=1.0):
    return SpeechCalibration(
        model_id=model_id,
        device=ARC,
        gpu=Run("vulkan", gpu_ms, gpu_similarity),
        cpu=Run("cpu", cpu_ms, cpu_similarity),
    )


# The verdict ---------------------------------------------------------------------------


def test_a_clearly_faster_gpu_wins():
    assert measured(QWEN, 300, 1300).prefers_gpu is True


def test_a_gpu_that_is_only_a_little_faster_loses():
    cpu_ms = 1000
    assert measured(QWEN, int(cpu_ms * GPU_WIN_RATIO) + 1, cpu_ms).prefers_gpu is False
    assert measured(QWEN, int(cpu_ms * GPU_WIN_RATIO), cpu_ms).prefers_gpu is True


def test_a_gpu_that_transcribes_wrongly_never_wins():
    assert measured(QWEN, 100, 1300, gpu_similarity=0.3).prefers_gpu is False


def test_a_gpu_wins_when_the_processor_run_failed():
    result = SpeechCalibration(QWEN, ARC, Run("vulkan", 500, 1.0), Run("cpu", error="timed out"))
    assert result.prefers_gpu is True
    assert result.latency_ms(on_gpu=False) is None


def test_a_failed_gpu_run_loses():
    result = SpeechCalibration(QWEN, ARC, Run("vulkan", error="crash"), Run("cpu", 1300, 1.0))
    assert result.prefers_gpu is False
    assert result.latency_ms(on_gpu=False) == 1300


def test_similarity_compares_words_not_punctuation():
    assert calibrate.similarity(CALIBRATION_TEXT.upper().replace(",", "")) == 1.0
    assert calibrate.similarity("something else entirely") < 0.2


def test_describe_names_the_winner():
    assert "the graphics wins" in measured(QWEN, 300, 1300).describe()
    assert "the processor wins" in measured(QWEN, 3000, 1300).describe()


# The store -----------------------------------------------------------------------------


def test_results_round_trip_per_device(tmp_path):
    path = tmp_path / "calibration.json"
    other = SpeechCalibration(QWEN, "Other GPU", Run("vulkan", 1, 1.0), Run("cpu", 2, 1.0))
    calibrate.save(path, [measured(QWEN, 300, 1300), other])
    loaded = calibrate.load(path, ARC)
    assert set(loaded) == {QWEN}
    assert loaded[QWEN].gpu.latency_ms == 300
    assert loaded[QWEN].cpu.latency_ms == 1300
    assert calibrate.load(path, "Other GPU")[QWEN].gpu.latency_ms == 1


def test_saving_again_replaces_only_the_same_device_and_model(tmp_path):
    path = tmp_path / "calibration.json"
    calibrate.save(path, [measured(QWEN, 300, 1300), measured(WHISPER, 900, 5000)])
    calibrate.save(path, [measured(QWEN, 350, 1250)])
    loaded = calibrate.load(path, ARC)
    assert loaded[QWEN].gpu.latency_ms == 350
    assert loaded[WHISPER].gpu.latency_ms == 900


def test_an_older_version_or_a_broken_file_is_ignored(tmp_path):
    path = tmp_path / "calibration.json"
    calibrate.save(path, [measured(QWEN, 300, 1300)])
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["entries"][0]["version"] = 0
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert calibrate.load(path, ARC) == {}
    path.write_text("{not json", encoding="utf-8")
    assert calibrate.load(path, ARC) == {}
    assert calibrate.load(tmp_path / "missing.json", ARC) == {}


def test_the_store_sits_beside_the_settings(tmp_path):
    assert calibrate.store_path(tmp_path / "settings.json") == tmp_path / "calibration.json"


# Which hardware ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        (NO_GPU, False),
        (device(uma=False, memory_mb=8192, name="RTX"), False),
        (device(uma=True), True),
        (device(uma=None, memory_mb=2048, name="Small"), True),
        (device(uma=None, memory_mb=8192, name="Unknown big"), False),
    ],
)
def test_only_a_gpu_the_catalog_cannot_judge_is_measured(selection, expected):
    assert calibrate.needs_measuring(selection) is expected


def test_without_measurements_an_integrated_gpu_runs_on_the_processor():
    hardware, integrated, models = calibrate.choose_hardware(
        device(uma=True), ["en", "de"], INSTALLED, catalog(), {}
    )
    assert hardware is Hardware.CPU
    assert integrated is False
    assert {m.id: m.latency_ms(Hardware.CPU) for m in models} == {WHISPER: 4500, QWEN: 1300}


def test_a_measured_win_moves_the_models_to_the_integrated_gpu():
    calibrations = {QWEN: measured(QWEN, 450, 2300), WHISPER: measured(WHISPER, 900, 10000)}
    hardware, integrated, models = calibrate.choose_hardware(
        device(uma=True), ["en", "de"], INSTALLED, catalog(), calibrations
    )
    assert hardware is Hardware.GPU
    assert integrated is True
    latencies = {m.id: m.latency_ms(Hardware.GPU) for m in models}
    wanted = {c.model_id for c in select_models(["en", "de"], Hardware.GPU, INSTALLED, models).asr}
    assert all(latencies[model_id] in (450, 900) for model_id in wanted)


def test_a_measured_loss_keeps_the_processor_with_its_real_speed():
    calibrations = {QWEN: measured(QWEN, 3000, 2300), WHISPER: measured(WHISPER, 12000, 10000)}
    hardware, integrated, models = calibrate.choose_hardware(
        device(uma=True), ["en", "de", "sq"], INSTALLED, catalog(), calibrations
    )
    assert hardware is Hardware.CPU
    assert integrated is False
    assert {m.id: m.latency_ms(Hardware.CPU) for m in models} == {WHISPER: 10000, QWEN: 2300}


def test_one_losing_model_keeps_everything_on_the_processor():
    selected = select_models(["en", "de", "sq"], Hardware.GPU, INSTALLED, catalog()).asr
    calibrations = {choice.model_id: measured(choice.model_id, 300, 1300) for choice in selected}
    loser = selected[0].model_id
    calibrations[loser] = measured(loser, 3000, 1300)
    hardware, _, _ = calibrate.choose_hardware(
        device(uma=True), ["en", "de", "sq"], INSTALLED, catalog(), calibrations
    )
    assert hardware is Hardware.CPU


def test_a_big_win_in_one_language_outweighs_a_small_loss_in_another():
    calibrations = {WHISPER: measured(WHISPER, 1500, 10000), QWEN: measured(QWEN, 1400, 1100)}
    hardware, integrated, models = calibrate.choose_hardware(
        device(uma=True), ["en", "de", "sq"], INSTALLED, catalog(), calibrations
    )
    assert hardware is Hardware.GPU
    assert integrated is True
    assert {m.id: m.latency_ms(Hardware.GPU) for m in models} == {WHISPER: 1500, QWEN: 1400}


def test_a_language_that_would_get_much_slower_keeps_the_processor():
    calibrations = {WHISPER: measured(WHISPER, 2400, 10000), QWEN: measured(QWEN, 2400, 1100)}
    hardware, _, _ = calibrate.choose_hardware(
        device(uma=True), ["en", "de", "sq"], INSTALLED, catalog(), calibrations
    )
    assert hardware is Hardware.CPU


def test_an_unmeasured_model_in_the_graphics_plan_keeps_the_processor():
    calibrations = {WHISPER: measured(WHISPER, 3000, 10000)}
    hardware, _, _ = calibrate.choose_hardware(
        device(uma=True), ["en", "de", "sq"], INSTALLED, catalog(), calibrations
    )
    assert hardware is Hardware.CPU


def test_a_discrete_gpu_is_never_second_guessed():
    calibrations = {QWEN: measured(QWEN, 3000, 1300)}
    hardware, integrated, models = calibrate.choose_hardware(
        device(uma=False, name="RTX"), ["en"], INSTALLED, catalog(), calibrations
    )
    assert hardware is Hardware.GPU
    assert integrated is False
    assert {m.id: m.latency_ms(Hardware.GPU) for m in models} == {WHISPER: 300, QWEN: 400}
    assert calibrate.pending(device(uma=False, name="RTX"), ["en"], INSTALLED, catalog(), {}) == []


def test_pending_lists_every_installed_speech_model_for_the_languages_once():
    todo = calibrate.pending(device(uma=True), ["en", "de", "sq"], INSTALLED, catalog(), {})
    ids = [choice.model_id for choice in todo]
    assert sorted(ids) == sorted(set(ids))
    assert set(ids) == {WHISPER, QWEN}
    by_id = {choice.model_id: choice for choice in todo}
    assert by_id[QWEN].languages == ("en", "de")
    assert by_id[QWEN].extra_files == ("mmproj.gguf",)
    assert by_id[WHISPER].runtime == "whisper-server"
    only_sq = calibrate.pending(device(uma=True), ["sq"], INSTALLED, catalog(), {})
    assert [choice.model_id for choice in only_sq] == [WHISPER]
    assert calibrate.pending(device(uma=True), ["en"], frozenset({WHISPER}), catalog(), {})[
        0
    ].model_id == WHISPER
    done = {WHISPER: measured(WHISPER, 900, 10000)}
    rest = calibrate.pending(device(uma=True), ["en", "de", "sq"], INSTALLED, catalog(), done)
    assert [choice.model_id for choice in rest] == [QWEN]


# The measurement -----------------------------------------------------------------------


class FakeProc:
    def __init__(self) -> None:
        self.killed = False
        self.exit_code = None

    def poll(self):
        return self.exit_code

    def kill(self):
        self.killed = True
        self.exit_code = 1

    def wait(self, timeout=None):
        return self.exit_code


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeWhisper:
    def __init__(self, clock: FakeClock, seconds: float, text: str = CALIBRATION_TEXT) -> None:
        self.clock = clock
        self.seconds = seconds
        self.text = text
        self.requests: list[dict] = []

    def health(self) -> bool:
        return True

    def inference(self, wav, **kwargs):
        self.requests.append(kwargs)
        self.clock.now += self.seconds
        return {"text": self.text}


class FakeLlamaAsr(FakeWhisper):
    def transcribe(self, wav, **kwargs):
        self.requests.append(kwargs)
        self.clock.now += self.seconds

        class Reply:
            text = self.text

        return Reply()


class Rig:
    def __init__(self, tmp_path: Path, seconds: dict[str, float], text=CALIBRATION_TEXT) -> None:
        for variant in ("vulkan", "cpu"):
            folder = tmp_path / "engines" / variant
            folder.mkdir(parents=True)
            (folder / "whisper-server.exe").write_bytes(b"")
            (folder / "llama-server.exe").write_bytes(b"")
        self.clock = FakeClock()
        self.spawned: list[dict] = []
        self.procs: list[FakeProc] = []
        self.clients: list[FakeWhisper] = []
        self.jobs: list[object] = []
        self.seconds = seconds
        self.text = text
        self.paths = EnginePaths(
            vulkan_dir=tmp_path / "engines" / "vulkan",
            cpu_dir=tmp_path / "engines" / "cpu",
            whisper_model=tmp_path / "models" / "whisper.bin",
            vad_model=tmp_path / "models" / "vad.bin",
            llama_model=None,
            log_dir=tmp_path / "logs",
        )
        self.calibrator = Calibrator(
            paths=self.paths,
            gpu=device(uma=True),
            cpu_plan=CpuPlan(6, 0x55401),
            whisper_threads=6,
            models_dir=tmp_path / "models",
            clip=b"clip",
            warmup=b"warm",
            spawn=self.spawn,
            job_factory=self.job,
            client_factory=self.client,
            clock=self.clock,
            sleeper=self.sleep,
        )

    def sleep(self, seconds):
        self.clock.now += seconds

    def spawn(self, args, **kwargs):
        self.spawned.append({"args": args, **kwargs})
        proc = FakeProc()
        self.procs.append(proc)
        return proc

    def job(self):
        job = object()
        self.jobs.append(job)
        return job

    def client(self, runtime, url):
        variant = Path(self.spawned[-1]["args"][0]).parent.name
        kind = FakeLlamaAsr if runtime == "llama-asr" else FakeWhisper
        client = kind(self.clock, self.seconds[variant], self.text)
        self.clients.append(client)
        return client


def choice_for(model_id, languages=("en",)):
    installed = frozenset({model_id})
    return select_models(list(languages), Hardware.CPU, installed, catalog()).asr[0]


def test_whisper_is_measured_on_both_builds_and_the_processes_end(tmp_path):
    rig = Rig(tmp_path, {"vulkan": 0.9, "cpu": 5.2})
    result = rig.calibrator.calibrate(choice_for(WHISPER, ("sq",)))
    assert result.gpu.latency_ms == 900
    assert result.cpu.latency_ms == 5200
    assert result.gpu.similarity == 1.0
    assert result.prefers_gpu is True
    assert result.device == ARC
    assert [Path(item["args"][0]).parent.name for item in rig.spawned] == ["vulkan", "cpu"]
    assert all(proc.killed for proc in rig.procs)
    assert len(rig.jobs) == 1
    assert rig.spawned[0]["env"]["GGML_VK_VISIBLE_DEVICES"] == "0"
    assert rig.spawned[1]["affinity_mask"] == 0x55401
    assert all(request["language"] == "en" for request in rig.clients[0].requests)


def test_a_fast_run_is_repeated_and_a_slow_one_is_not(tmp_path):
    rig = Rig(tmp_path, {"vulkan": 0.4, "cpu": 6.0})
    rig.calibrator.calibrate(choice_for(WHISPER, ("sq",)))
    fast, slow = rig.clients
    assert len(fast.requests) == 1 + calibrate.TIMED_RUNS
    assert len(slow.requests) == 2


def test_llama_asr_uses_the_llama_build_with_all_layers_on_the_gpu(tmp_path):
    rig = Rig(tmp_path, {"vulkan": 0.5, "cpu": 2.3})
    result = rig.calibrator.calibrate(choice_for(QWEN))
    vulkan_args, cpu_args = (item["args"] for item in rig.spawned)
    assert Path(vulkan_args[0]).name == "llama-server.exe"
    assert vulkan_args[vulkan_args.index("-ngl") + 1] == "99"
    assert cpu_args[cpu_args.index("-ngl") + 1] == "0"
    assert "--mmproj" in vulkan_args
    assert result.gpu.latency_ms == 500
    assert result.cpu.latency_ms == 2300


def test_a_wrong_transcript_is_recorded_as_such(tmp_path):
    rig = Rig(tmp_path, {"vulkan": 0.5, "cpu": 2.3}, text="garbled words here")
    result = rig.calibrator.calibrate(choice_for(QWEN))
    assert result.gpu.ok is False
    assert result.prefers_gpu is False


def test_a_missing_build_is_an_error_not_a_crash(tmp_path):
    rig = Rig(tmp_path, {"vulkan": 0.5, "cpu": 2.3})
    (tmp_path / "engines" / "vulkan" / "llama-server.exe").unlink()
    result = rig.calibrator.calibrate(choice_for(QWEN))
    assert result.gpu.latency_ms is None
    assert "missing" in result.gpu.error
    assert result.cpu.latency_ms == 2300


def test_an_engine_that_exits_early_counts_as_not_started(tmp_path):
    rig = Rig(tmp_path, {"vulkan": 0.5, "cpu": 2.3})
    original = rig.spawn

    def dying(args, **kwargs):
        proc = original(args, **kwargs)
        if Path(args[0]).parent.name == "vulkan":
            proc.exit_code = 3
        return proc

    rig.calibrator.spawn = dying

    def unhealthy(runtime, url):
        client = Rig.client(rig, runtime, url)
        client.health = lambda: False
        return client

    rig.calibrator.client_factory = unhealthy
    result = rig.calibrator.calibrate(choice_for(QWEN))
    assert result.gpu.error == "the engine did not start"


def test_cancel_stops_the_measurement(tmp_path):
    rig = Rig(tmp_path, {"vulkan": 0.5, "cpu": 2.3})
    rig.calibrator.cancel()
    result = rig.calibrator.calibrate(choice_for(QWEN))
    assert result.gpu.error == "cancelled"
    assert result.cpu.error == "cancelled"
    assert rig.spawned == []
