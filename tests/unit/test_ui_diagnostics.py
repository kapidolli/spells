"""spells.ui.diagnostics: engine status, the GPU override, the medians, the bundle, licences."""

from __future__ import annotations

import sys
import wave
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from spells.audio import CaptureInfo
from spells.gpu import GpuSelection
from spells.history import AudioPolicy, HistoryEntry, HistoryStore
from spells.models import EngineState, StageTimings
from spells.ui.diagnostics import (
    BUNDLE_CLEAN_NOTE,
    BUNDLE_DEBUG_NOTE,
    CAPTURE_MOVED,
    CAPTURE_UNKNOWN,
    LICENSE_COMPONENTS,
    MIC_OPEN_TARGET_MS,
    NO_CAPTURES,
    DiagnosticsTab,
    capture_device,
    capture_path,
    dropped_summary,
    license_text,
    licenses_dir,
    manifest_entries,
    median,
    recommend_keep_mic_warm,
    stage_medians,
    write_bundle,
)

from .test_ui_support import (
    LLAMA,
    SELECTION,
    WHISPER,
    FakeEngines,
    FakeHotkey,
    FakePipeline,
    Messages,
    make_config,
    qt_app,
)


@pytest.fixture(scope="module")
def app():
    return qt_app()


def timings(*mic_open: float) -> list[StageTimings]:
    return [StageTimings(press_to_pill_ms=40.0, mic_open_ms=value, delivery_ms=6.0) for value in mic_open]



def test_median_handles_odd_even_and_empty():
    assert median([3.0, 1.0, 2.0]) == 2.0
    assert median([4.0, 1.0, 2.0, 3.0]) == 2.5
    assert median([]) is None


def test_recommendation_threshold_is_the_spec_12_target():
    assert MIC_OPEN_TARGET_MS == 250
    assert not recommend_keep_mic_warm(timings(200.0, 240.0, 250.0))
    assert recommend_keep_mic_warm(timings(200.0, 300.0, 260.0))
    assert not recommend_keep_mic_warm([])
    assert not recommend_keep_mic_warm([StageTimings(mic_open_ms=None)])


def test_stage_medians_skip_missing_values():
    values = [
        StageTimings(press_to_pill_ms=50.0, mic_open_ms=None, release_to_transcript_ms=600.0),
        StageTimings(press_to_pill_ms=70.0, mic_open_ms=30.0, release_to_transcript_ms=None),
    ]
    medians = stage_medians(values)
    assert medians["press_to_pill_ms"] == 60.0
    assert medians["mic_open_ms"] == 30.0
    assert medians["release_to_transcript_ms"] == 600.0
    assert medians["delivery_ms"] is None


def test_bundle_holds_logs_and_settings_but_never_history(tmp_path: Path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "spells.log").write_text("app log line\n", encoding="utf-8")
    (log_dir / "spells.log.1").write_text("older\n", encoding="utf-8")
    (log_dir / "whisper.log").write_text("ggml_vulkan: Found 1 Vulkan devices\n", encoding="utf-8")
    (log_dir / "history.db").write_bytes(b"SQLite format 3\x00secret transcript text")
    (tmp_path / "history.db").write_bytes(b"SQLite format 3\x00another transcript")
    settings_path = tmp_path / "settings.json"
    settings_path.write_text('{"schema_version": 1}', encoding="utf-8")
    target = tmp_path / "bundle.zip"
    names = write_bundle(target, log_dir=log_dir, settings_path=settings_path, summary="engines ok\n")
    with zipfile.ZipFile(target) as archive:
        listed = set(archive.namelist())
        assert listed == set(names)
        assert "logs/spells.log" in listed and "logs/spells.log.1" in listed
        assert "logs/whisper.log" in listed
        assert "settings.json" in listed
        assert "diagnostics.txt" in listed
        assert not any("history" in name for name in listed)
        blob = b"".join(archive.read(name) for name in listed)
        assert b"transcript" not in blob


def test_the_bundle_never_carries_a_recording_or_a_transcript(tmp_path: Path):
    """Spec 17: recordings and transcripts stay on the machine, whatever the user saves."""
    data_dir = tmp_path / "Spells"
    log_dir = data_dir / "logs"
    log_dir.mkdir(parents=True)
    (log_dir / "spells.log").write_text("app log line\n", encoding="utf-8")
    recordings = data_dir / "recordings"
    recordings.mkdir()
    with wave.open(str(recordings / "000001.wav"), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(bytes(3200))
    store = HistoryStore(data_dir / "history.db")
    store.add(
        HistoryEntry(
            id=None,
            created_at=1_700_000_000.0,
            raw_text="the secret transcript text",
            cleaned_text="The secret transcript text.",
            delivered_text="The secret transcript text.",
            app_process="notepad.exe",
            app_title="Untitled",
            language="en",
            used_llm=True,
            cleanup_reason="ok",
            outcome="pasted",
            timings=StageTimings(),
        ),
        pcm16=bytes(3200),
        audio=AudioPolicy(keep=True),
    )
    store.close()
    settings_path = data_dir / "settings.json"
    settings_path.write_text('{"schema_version": 1}', encoding="utf-8")

    target = tmp_path / "bundle.zip"
    names = write_bundle(target, log_dir=log_dir, settings_path=settings_path, summary="engines ok\n")

    with zipfile.ZipFile(target) as archive:
        listed = set(archive.namelist())
        assert listed == set(names)
        assert not any(name.lower().endswith(".wav") for name in listed)
        assert not any("recording" in name.lower() for name in listed)
        assert not any("history" in name.lower() for name in listed)
        blob = b"".join(archive.read(name) for name in listed)
    assert b"RIFF" not in blob
    assert b"secret transcript" not in blob


def make_tab(tmp_path: Path, *, selection: GpuSelection = SELECTION):
    config = make_config(tmp_path)
    engines = FakeEngines()
    engines.states = {WHISPER: EngineState.READY, LLAMA: EngineState.CPU_FALLBACK}
    engines.reasons = {WHISPER: "ok", LLAMA: "oom"}
    engines.variants = {WHISPER: "vulkan", LLAMA: "cpu"}
    engines.verified = {WHISPER: False, LLAMA: None}
    pipeline = FakePipeline()
    pipeline.timings = timings(300.0, 280.0, 100.0)
    pipeline.guards = {"length_ratio": 2, "preamble": 1}
    messages = Messages()
    launched: list[str] = []
    saved: list[Path] = []
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "spells.log").write_text("line\n", encoding="utf-8")

    def save_dialog(default: Path) -> Path | None:
        saved.append(default)
        return tmp_path / "out.zip"

    tab = DiagnosticsTab(
        config=config,
        engines=engines,
        pipeline=pipeline,
        hotkey=FakeHotkey(),
        gpu_selection=selection,
        log_dir=log_dir,
        notify=messages,
        launcher=launched.append,
        save_dialog=save_dialog,
    )
    tab.refresh()
    return tab, config, engines, pipeline, messages, launched, saved


def test_gpu_override_saves_without_restarting_engines_with_a_stale_model_plan(app, tmp_path):
    tab, config, engines, *_ = make_tab(tmp_path)
    combo = tab.gpu_override
    assert combo.count() == 3
    assert combo.itemText(0).startswith("Automatic")
    assert "RTX 5060" in combo.itemText(2)
    combo.setCurrentIndex(1)
    assert config.settings.diagnostics.gpu_device_override == 0
    assert not any(call[0] == "set_gpu" for call in engines.calls)
    combo.setCurrentIndex(0)
    assert config.settings.diagnostics.gpu_device_override is None
    assert not any(call[0] == "set_gpu" for call in engines.calls)
    tab.close()


def test_guard_counts_are_listed(app, tmp_path):
    tab, *_ = make_tab(tmp_path)
    text = tab.guards_text()
    assert "length_ratio" in text and "2" in text
    assert "preamble" in text
    assert {reason: label.text() for reason, label in tab.guard_rows.items()} == {"length_ratio": "2", "preamble": "1"}
    tab.close()


def test_engine_rows_show_a_level_a_badge_and_sentences(app, tmp_path):
    tab, _config, engines, *_ = make_tab(tmp_path)
    assert tab.engine_view(WHISPER)[:2] == ("ok", "Ready")
    level, badge, detail = tab.engine_view(LLAMA)
    assert (level, badge) == ("caution", "Running on CPU")
    assert detail.startswith("CPU build. Out of GPU memory.")
    engines.states[WHISPER] = EngineState.FAILED
    engines.reasons[WHISPER] = "crash"
    assert tab.engine_view(WHISPER)[0] == "critical"
    engines.states[LLAMA] = EngineState.CPU_FALLBACK
    engines.reasons[LLAMA] = "cpu_selected"
    level, _badge, detail = tab.engine_view(LLAMA)
    assert level == "ok"
    assert "Chosen for this computer." in detail
    tab.refresh()
    assert tab._engine_badges[WHISPER].text() == "Failed"
    tab.close()


def test_restart_debug_logging_and_log_folder(app, tmp_path):
    tab, config, engines, _pipeline, _messages, launched, _saved = make_tab(tmp_path)
    tab.restart_button.click()
    assert ("restart", WHISPER) in engines.calls and ("restart", LLAMA) in engines.calls
    tab.debug_logging.setChecked(True)
    assert config.settings.diagnostics.debug_logging is True
    tab.open_log_button.click()
    assert launched == [str(tab.log_dir)]
    tab.close()


def test_bundle_button_saves_to_the_chosen_path_without_history(app, tmp_path):
    tab, config, _engines, _pipeline, messages, _launched, saved = make_tab(tmp_path)
    (tab.log_dir / "history.db").write_bytes(b"transcript")
    tab.bundle_button.click()
    assert saved and saved[0].name.endswith(".zip")
    assert "Desktop" in str(saved[0]) or saved[0].parent == Path.home()
    with zipfile.ZipFile(tmp_path / "out.zip") as archive:
        names = archive.namelist()
        assert "logs/spells.log" in names
        assert not any("history" in n for n in names)
        summary = archive.read("diagnostics.txt").decode("utf-8")
        assert "whisper" in summary.lower() and "RTX 5060" in summary
    assert messages.shown[-1][0] == "info"
    assert BUNDLE_CLEAN_NOTE in messages.texts[-1]
    assert "dictated text" not in messages.texts[-1]
    config.update(lambda s: replace(s, diagnostics=replace(s.diagnostics, debug_logging=True)))
    tab.apply_settings(config.settings)
    tab.bundle_button.click()
    assert BUNDLE_DEBUG_NOTE in messages.texts[-1]
    assert "no transcript text" not in messages.texts[-1]
    tab.close()


REPO_LICENSES = Path(__file__).resolve().parents[2] / "build" / "licenses"
POINTER = "The full text ships in the licenses folder"


def test_licenses_dir_sits_next_to_the_frozen_executable(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "Spells" / "Spells.exe"))
    assert licenses_dir() == (tmp_path / "Spells").resolve() / "licenses"


def test_every_row_names_manifest_entries_and_every_shipping_entry_has_a_row():
    entries = manifest_entries(REPO_LICENSES)
    assert entries, "build/licenses/MANIFEST.json must list components"
    covered: set[str] = set()
    for _component, _license, names in LICENSE_COMPONENTS:
        assert names, "every row keys at least one manifest name"
        for name in names:
            assert name in entries, f"{name} is not in MANIFEST.json"
            covered.add(name)
    shipping = {name for name, entry in entries.items() if entry.get("ships") and entry.get("file")}
    assert shipping <= covered, f"shipping components without a row: {sorted(shipping - covered)}"


def test_missing_folder_or_manifest_gives_the_pointer(tmp_path):
    assert manifest_entries(tmp_path) == {}
    text = license_text("winpthreads", "MIT", ("winpthreads",), folder=tmp_path)
    assert POINTER in text and str(tmp_path) in text
    (tmp_path / "MANIFEST.json").write_text("not json", encoding="utf-8")
    assert manifest_entries(tmp_path) == {}
    (tmp_path / "MANIFEST.json").write_text(
        '{"components": [{"name": "winpthreads", "file": "missing.txt", "ships": true}]}', encoding="utf-8"
    )
    assert POINTER in license_text("winpthreads", "MIT", ("winpthreads",), folder=tmp_path)




def capture(**changes) -> CaptureInfo:
    fields = {
        "device": "Microphone (HyperX Cloud III Wireless)",
        "host_api": "Windows WASAPI",
        "sample_rate": 48000,
        "channels": 1,
    }
    fields.update(changes)
    return CaptureInfo(**fields)


def test_capture_path_names_the_host_api_rate_and_channels():
    assert capture_path([capture()]) == "Windows WASAPI, 48 kHz, mono"
    assert capture_path([capture(sample_rate=44100, host_api="MME")]) == "MME, 44.1 kHz, mono"
    assert "2 channels mixed to mono" in capture_path([capture(channels=2, downmixed=True)])
    assert capture_path([]) == CAPTURE_UNKNOWN


def test_capture_device_names_the_device_and_a_fallback():
    assert capture_device([capture()]) == "Microphone (HyperX Cloud III Wireless)"
    assert CAPTURE_MOVED in capture_device([capture(match="default")])
    assert capture_device([]) == NO_CAPTURES


def test_dropped_summary_counts_the_blocks_and_the_dictations():
    assert dropped_summary([]) == NO_CAPTURES
    assert dropped_summary([capture(), capture()]) == "None over the last 2 dictations."
    text = dropped_summary([capture(dropped_blocks=3), capture(), capture(dropped_blocks=1)])
    assert text.startswith("4 blocks over the last 3 dictations, in 2 dictations")
    assert "words may be missing" in text
    assert dropped_summary([capture(dropped_blocks=1)]).startswith("1 block over the last 1 dictation")

