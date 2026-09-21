"""The Diagnostics page (spec 14.4, 13, 12, 17) and the licence helpers of spec 21.

Engine state, backend and reason per engine with the GPU verification result, the chosen
GPU with an override dropdown (the one change that restarts both engines), how the last
dictations were captured (host API, device, sample rate and dropped input blocks), the
medians of the pipeline's last 20 stage timings with the keep-mic-warm recommendation, the
guard rejection counts, the hotkey stats, restart, debug logging, the log folder, a
diagnostics bundle that never contains transcript text, and the elevated window note. The licence list
itself is shown on the About page. The pure helpers at the top carry the rules so they are
testable without Qt.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtWidgets

from spells.audio import CaptureInfo
from spells.config import ConfigStore, Settings, SettingsError
from spells.datafiles import data_dir
from spells.gpu import GpuDevice, GpuSelection, choose_device
from spells.models import Engine, EngineState, StageTimings
from spells.ui import style, theme
from spells.ui.tray import ENGINE_NAMES, REASON_TEXT, STATE_WORDS, reason_text
from spells.ui.welcome import Notify, default_notify
from spells.ui.widgets import (
    Badge,
    Card,
    ComboBox,
    Divider,
    InfoBar,
    ScrollPage,
    SettingRow,
    StatusDot,
    ToggleSwitch,
    make_button,
    make_label,
    set_prop,
)

log = logging.getLogger(__name__)

MIC_OPEN_TARGET_MS = 250
STAGES: tuple[tuple[str, str], ...] = (
    ("press_to_pill_ms", "Press to pill"),
    ("mic_open_ms", "Mic open (cold)"),
    ("release_to_transcript_ms", "Release to transcript"),
    ("release_to_cleaned_ms", "Release to cleaned text"),
    ("delivery_ms", "Delivery"),
)
NO_CAPTURES = "No dictation yet."
CAPTURE_UNKNOWN = "Unknown"
DEFAULT_HOST_API = "the default host API"
CAPTURE_MOVED = "The microphone that was picked was gone, so this is the system default."
RECOMMENDATION = (
    "The microphone takes longer than {target} ms to open (median {median:.0f} ms over the last "
    "{count} dictations). Enable keep mic warm on the General page."
)
ELEVATED_NOTE = (
    "The hotkey cannot fire while an elevated (administrator) window is in the foreground: "
    "Windows lets no keyboard hook of a normal app see input there (UIPI)."
)
# Spec 21, in table order: (row label, licence, manifest names). The texts and their
# MANIFEST.json live in build/licenses at development time and in the licenses folder next
# to the executable when frozen (spec 19.3); the manifest names key the files.
LICENSE_COMPONENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("Spells", "GPL-3.0-or-later", ("spells",)),
    ("whisper.cpp, llama.cpp, ggml", "MIT", ("whisper_cpp", "llama_cpp", "ggml")),
    (
        "cpp-httplib, nlohmann json (compiled into whisper-server)",
        "MIT",
        ("cpp_httplib", "nlohmann_json"),
    ),
    (
        "libgcc, libstdc++, libgomp (GCC runtime, statically linked)",
        "GPLv3 with GCC Runtime Library Exception 3.1",
        ("gcc_runtime_gplv3", "gcc_runtime_library_exception"),
    ),
    ("winpthreads", "MIT and BSD-3-Clause", ("winpthreads",)),
    ("LLVM OpenMP runtime (libomp.dll)", "Apache-2.0 with LLVM exceptions", ("llvm_openmp",)),
    ("Whisper model weights", "MIT", ("whisper_model_weights",)),
    ("Silero VAD", "MIT", ("silero_vad",)),
    ("Gemma 4 models", "Apache-2.0", ("gemma4",)),
    ("Qwen3 models", "Apache-2.0", ("qwen3",)),
    ("Qwen3-ASR 0.6B model weights", "Apache-2.0", ("qwen3_asr",)),
    ("Gemma models", "Gemma Terms of Use", ("gemma3",)),
    ("PySide6 / Qt", "LGPLv3", ("pyside6_qt", "qt_licensing_notice")),
    ("Python runtime", "PSF License", ("python",)),
    ("PyInstaller bootloader", "GPL with the bootloader exception", ("pyinstaller_bootloader",)),
    ("numpy", "BSD-3-Clause", ("numpy",)),
    ("sounddevice, PortAudio", "MIT", ("sounddevice", "portaudio")),
)
MANIFEST_NAME = "MANIFEST.json"
BUNDLE_CLEAN_NOTE = "It holds no transcript text."
BUNDLE_DEBUG_NOTE = "It includes debug logs, which may contain dictated text."
BUNDLE_EXCLUDED_SUFFIXES = (".db", ".db-wal", ".db-shm", ".sqlite")


# Pure helpers ---------------------------------------------------------------------------------------


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def capture_path(captures: Sequence[CaptureInfo]) -> str:
    """The host API, rate and channel handling of the newest recording."""
    if not captures:
        return CAPTURE_UNKNOWN
    info = captures[-1]
    host_api = info.host_api or DEFAULT_HOST_API
    rate = f"{info.sample_rate / 1000:g} kHz" if info.sample_rate else CAPTURE_UNKNOWN
    channels = (
        f"{info.channels} channels mixed to mono" if info.downmixed else "mono"
    )
    return f"{host_api}, {rate}, {channels}"


def capture_device(captures: Sequence[CaptureInfo]) -> str:
    """The device the newest recording came from, with a note when it was a fallback."""
    if not captures:
        return NO_CAPTURES
    info = captures[-1]
    name = info.device or CAPTURE_UNKNOWN
    if info.match == "default":
        return f"{name}. {CAPTURE_MOVED}"
    return name


def dropped_summary(captures: Sequence[CaptureInfo]) -> str:
    """How many input blocks the sound card dropped over the dictations on record."""
    if not captures:
        return NO_CAPTURES
    total = sum(int(info.dropped_blocks) for info in captures)
    over = f"over the last {_plural(len(captures), 'dictation')}"
    if not total:
        return f"None {over}."
    affected = sum(1 for info in captures if info.dropped_blocks)
    return (
        f"{_plural(total, 'block')} {over}, in {_plural(affected, 'dictation')}; "
        "words may be missing."
    )


def median(values: Sequence[float]) -> float | None:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def stage_medians(timings: Sequence[StageTimings]) -> dict[str, float | None]:
    """The median per stage over the timings, ignoring stages a dictation did not reach."""
    result: dict[str, float | None] = {}
    for key, _label in STAGES:
        values = [getattr(t, key) for t in timings if getattr(t, key) is not None]
        result[key] = median(values)
    return result


def recommend_keep_mic_warm(timings: Sequence[StageTimings], target_ms: float = MIC_OPEN_TARGET_MS) -> bool:
    """Spec 12: recommend keep mic warm when the rolling median of the cold mic open exceeds 250 ms."""
    value = stage_medians(timings)["mic_open_ms"]
    return value is not None and value > target_ms


def automatic_choice(devices: Sequence[GpuDevice]) -> GpuDevice | None:
    """The gpu module's rule without a probe: discrete first, then most memory, then lower index."""
    selection = choose_device(devices)
    return next((d for d in devices if d.raw_index == selection.raw_index), None)


def _bundle_wanted(path: Path) -> bool:
    name = path.name.lower()
    if "history" in name:
        return False
    return not any(name.endswith(suffix) for suffix in BUNDLE_EXCLUDED_SUFFIXES)


def write_bundle(target: Path, *, log_dir: Path, settings_path: Path | None, summary: str) -> list[str]:
    """Zip the logs and the settings file plus a summary; never the history database.

    Returns the archive member names. Log files are copied as they are (they carry no
    transcript text unless debug logging was on, spec 17); nothing else is redacted.
    """
    names: list[str] = []
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if log_dir.is_dir():
            for path in sorted(log_dir.iterdir()):
                if path.is_file() and _bundle_wanted(path):
                    name = f"logs/{path.name}"
                    try:
                        archive.write(path, name)
                    except OSError:
                        log.warning("could not add %s to the bundle", path, exc_info=True)
                        continue
                    names.append(name)
        if settings_path is not None and settings_path.is_file():
            archive.write(settings_path, "settings.json")
            names.append("settings.json")
        archive.writestr("diagnostics.txt", summary)
        names.append("diagnostics.txt")
    return names


def licenses_dir() -> Path:
    """build/licenses in the repository, or the licenses folder next to the frozen executable."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "licenses"
    return data_dir().parent / "build" / "licenses"


def manifest_entries(folder: Path | None = None) -> dict[str, dict]:
    """The MANIFEST.json components of the licences folder keyed by name; {} when absent."""
    folder = licenses_dir() if folder is None else folder
    try:
        raw = json.loads((folder / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    entries = raw.get("components", []) if isinstance(raw, dict) else []
    return {str(e["name"]): e for e in entries if isinstance(e, dict) and e.get("name")}


def license_text(
    component: str, license_name: str, names: tuple[str, ...] = (), folder: Path | None = None
) -> str:
    """The licence texts of the row's manifest names, one header per file, else a pointer."""
    folder = licenses_dir() if folder is None else folder
    entries = manifest_entries(folder)
    parts: list[str] = []
    for name in names:
        entry = entries.get(name)
        file_name = entry.get("file") if entry else None
        if not file_name:
            continue
        path = folder / str(file_name)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            log.warning("could not read %s", path, exc_info=True)
            continue
        version = str(entry.get("version", "")).strip()
        label = f"{name} {version}".strip()
        parts.append(f"=== {label} ({entry.get('license', license_name)}) ===\n\n{text.strip()}\n")
    if parts:
        return f"{component}\nLicense: {license_name}\n\n" + "\n".join(parts)
    return (
        f"{component}\nLicense: {license_name}\n\n"
        f"The full text ships in the licenses folder of the installed app ({folder})."
    )


# The page ------------------------------------------------------------------------------------------

ENGINE_LEVELS = {
    EngineState.READY: "ok",
    EngineState.STARTING: "busy",
    EngineState.RESTARTING: "caution",
    EngineState.PAUSED: "caution",
    EngineState.CPU_FALLBACK: "caution",
    EngineState.UNLOADED: "idle",
    EngineState.FAILED: "critical",
}
BADGE_KINDS = {"ok": "ok", "busy": "neutral", "caution": "caution", "critical": "critical", "idle": "muted"}
HOTKEY_STATS: tuple[tuple[str, str], ...] = (
    ("probes_sent", "Probes sent"),
    ("probes_missed", "Probes missed"),
    ("reinstalls", "Hook reinstalls"),
    ("hook_exceptions", "Hook exceptions"),
    ("install_failures", "Install failures"),
)


def _default_save_dialog(default: Path) -> Path | None:
    chosen, _filter = QtWidgets.QFileDialog.getSaveFileName(
        None, "Save diagnostics bundle", str(default), "Zip archives (*.zip)"
    )
    return Path(chosen) if chosen else None


def _capital(text: str) -> str:
    return text[:1].upper() + text[1:]


def _sentence(part: str) -> str:
    part = part.strip()
    if not part:
        return ""
    text = part[0].upper() + part[1:]
    return text if text.endswith((".", ")")) else text + "."


class StatTile(QtWidgets.QWidget):
    """A number with its unit over a caption."""

    def __init__(self, caption: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 12, 14)
        layout.setSpacing(2)
        line = QtWidgets.QHBoxLayout()
        line.setSpacing(3)
        self.value = make_label("", "stat", parent=self)
        line.addWidget(self.value, 0, QtCore.Qt.AlignmentFlag.AlignBaseline)
        self.unit = make_label("", "caption", "secondary", parent=self)
        line.addWidget(self.unit, 0, QtCore.Qt.AlignmentFlag.AlignBaseline)
        line.addStretch(1)
        layout.addLayout(line)
        self.caption = make_label(caption, "caption", "secondary", wrap=True, parent=self)
        layout.addWidget(self.caption)
        layout.addStretch(1)

    def set_value(self, value: str, unit: str = "", tone: str | None = None, *, numeric: bool = True) -> None:
        self.value.setText(value)
        self.unit.setText(unit)
        self.unit.setVisible(bool(unit))
        self.value.setFont(style.font("stat" if numeric else "body"))
        set_prop(self.value, "tone", tone if numeric else "tertiary")


class TileStrip(QtWidgets.QWidget):
    def __init__(self, captions: Sequence[str], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.tiles: list[StatTile] = []
        for index, caption in enumerate(captions):
            if index:
                layout.addWidget(Divider(self, vertical=True))
            tile = StatTile(caption, self)
            layout.addWidget(tile, 1)
            self.tiles.append(tile)


class DiagnosticsTab(ScrollPage):
    gpu_changed = QtCore.Signal()

    def __init__(
        self,
        *,
        config: ConfigStore,
        engines: Any,
        pipeline: Any,
        hotkey: Any,
        gpu_selection: GpuSelection,
        log_dir: Path,
        notify: Notify | None = None,
        launcher: Callable[[str], Any] | None = None,
        save_dialog: Callable[[Path], Path | None] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(
            "Diagnostics",
            "Engine health, speed and logs. Nothing here leaves this computer unless you save a bundle.",
            parent,
        )
        self._config = config
        self._engines = engines
        self._pipeline = pipeline
        self._hotkey = hotkey
        self._selection = gpu_selection
        self.log_dir = Path(log_dir)
        self._notify = notify or default_notify
        self._launcher = launcher or os.startfile
        self._save_dialog = save_dialog or _default_save_dialog
        self._loading = False

        engines_card = Card(self.body)
        self._engine_rows: dict[Engine, SettingRow] = {}
        self._engine_dots: dict[Engine, StatusDot] = {}
        self._engine_badges: dict[Engine, Badge] = {}
        for engine in Engine:
            dot = StatusDot("idle")
            badge = Badge("", "muted")
            row = SettingRow(ENGINE_NAMES[engine], " ", badge, leading=dot)
            engines_card.add_row(row)
            self._engine_rows[engine] = row
            self._engine_dots[engine] = dot
            self._engine_badges[engine] = badge
        self.gpu_override = ComboBox(engines_card)
        self.gpu_override.setMinimumWidth(260)
        self.gpu_override.setMaximumWidth(320)
        self._fill_override()
        self.gpu_override.currentIndexChanged.connect(self._on_override)
        self.gpu_row = SettingRow("GPU", " ", self.gpu_override, glyph=style.Glyph.CHIP)
        self.gpu_label = self.gpu_row.description_label
        engines_card.add_row(self.gpu_row)
        header = self.add_section("Engines", engines_card)
        self.restart_button = make_button("Restart engines", glyph=style.Glyph.REFRESH, parent=header)
        self.restart_button.clicked.connect(self._restart)
        header.add_trailing(self.restart_button)

        capture_card = Card(self.body)
        self.capture_row = SettingRow("Capture", " ", glyph=style.Glyph.MICROPHONE)
        capture_card.add_row(self.capture_row)
        self.dropped_row = SettingRow("Dropped input blocks", " ")
        capture_card.add_row(self.dropped_row)
        self.add_section(
            "Microphone",
            capture_card,
            description="How the last dictations reached Spells.",
        )

        latency_card = Card(self.body)
        self.latency_tiles = TileStrip([label for _key, label in STAGES], latency_card)
        latency_card.add_widget(self.latency_tiles)
        self._median_tiles: dict[str, StatTile] = {
            key: tile for (key, _label), tile in zip(STAGES, self.latency_tiles.tiles, strict=True)
        }
        self.recommendation = InfoBar("", "caution", flush=True, parent=latency_card)
        latency_card.add_widget(self.recommendation)
        self.add_section("Latency", latency_card, description="Median of the last 20 dictations.")

        self.guards_card = Card(self.body)
        self.guard_rows: dict[str, QtWidgets.QLabel] = {}
        self._guard_key: tuple = ()
        self.add_section("Cleanup guards", self.guards_card, description="Cleanup output rejected since Spells started, by guard.")

        hotkey_card = Card(self.body)
        self.hotkey_tiles = TileStrip([label for _key, label in HOTKEY_STATS], hotkey_card)
        hotkey_card.add_widget(self.hotkey_tiles)
        self.hotkey_stats = make_label("", "caption", "secondary", wrap=True, parent=hotkey_card)
        self.hotkey_stats.setContentsMargins(16, 0, 16, 12)
        hotkey_card.add_widget(self.hotkey_stats)
        self.add_section("Hotkey", hotkey_card, description="Health of the keyboard hook.")
        self.note = InfoBar(ELEVATED_NOTE, "info", parent=self.body)
        self.add_widget(self.note, spacing_before=8)

        logs_card = Card(self.body)
        self.debug_logging = ToggleSwitch(logs_card)
        self.debug_logging.setAccessibleName("Debug logging")
        self.debug_logging.toggled.connect(self._on_debug)
        logs_card.add_row(SettingRow("Debug logging", "Takes effect at once. Logs then include transcript text.", self.debug_logging))
        self.open_log_button = make_button("Open folder", glyph=style.Glyph.FOLDER, parent=logs_card)
        self.open_log_button.clicked.connect(self._open_logs)
        logs_card.add_row(SettingRow("Log folder", str(self.log_dir), self.open_log_button))
        self.bundle_button = make_button("Save bundle", glyph=style.Glyph.SAVE, parent=logs_card)
        self.bundle_button.clicked.connect(self._save_bundle)
        logs_card.add_row(
            SettingRow("Diagnostics bundle", "A zip of the logs and the settings file, never the history database.", self.bundle_button)
        )
        self.add_section("Logs and support", logs_card)
        self.finish()

        self.apply_settings(config.settings)

    @property
    def selection(self) -> GpuSelection:
        return self._selection

    # Refresh -------------------------------------------------------------------------------------------

    def refresh(self) -> None:
        for engine in Engine:
            level, badge, detail = self.engine_view(engine)
            self._engine_dots[engine].set_level(level)
            self._engine_badges[engine].set_badge(BADGE_KINDS.get(level, "muted"), badge)
            self._engine_rows[engine].set_description(detail)
        self.gpu_row.title_label.setText(self._selection.name if self._selection.raw_index is not None else "No Vulkan GPU")
        self.gpu_label.setText(self._gpu_detail())
        captures = self._captures()
        self.capture_row.title_label.setText(capture_path(captures))
        self.capture_row.set_description(capture_device(captures))
        self.dropped_row.set_description(dropped_summary(captures))
        timings = self._timings()
        medians = stage_medians(timings)
        recommend = recommend_keep_mic_warm(timings)
        for key, _label in STAGES:
            value = medians[key]
            tile = self._median_tiles[key]
            if value is None:
                tile.set_value("No data", numeric=False)
            else:
                tone = "caution" if key == "mic_open_ms" and recommend else None
                tile.set_value(f"{value:.0f}", "ms", tone)
        if recommend:
            self.recommendation.set_text(
                RECOMMENDATION.format(
                    target=MIC_OPEN_TARGET_MS, median=medians["mic_open_ms"] or 0.0, count=len(timings)
                )
            )
        else:
            self.recommendation.set_text("")
        self._refresh_guards()
        self._refresh_hotkey()

    def apply_settings(self, settings: Settings) -> None:
        self._loading = True
        try:
            self.debug_logging.setChecked(settings.diagnostics.debug_logging)
            override = settings.diagnostics.gpu_device_override
            index = 0
            for i in range(self.gpu_override.count()):
                if self.gpu_override.itemData(i) == override:
                    index = i
                    break
            self.gpu_override.setCurrentIndex(index)
        finally:
            self._loading = False
        self.refresh()

    def engine_summary(self, engine: Engine) -> str:
        try:
            state = EngineState(self._engines.status(engine))
            reason = str(self._engines.reason(engine))
            variant = str(self._engines.variant(engine))
            verified = self._engines.gpu_verified(engine)
        except Exception as exc:
            log.exception("engine status query failed")
            return f"unknown ({exc})"
        if reason == "no_model":
            return f"{_capital(STATE_WORDS[state])}; {reason_text(engine, 'no_model')}"
        parts = [_capital(STATE_WORDS[state])]
        parts.append("Vulkan build" if variant == "vulkan" else "CPU build")
        detail = reason_text(engine, reason)
        if detail:
            parts.append(detail)
        if engine is Engine.LLAMA:
            parts.append("GPU device not verified (llama-server prints no device line)")
        elif verified is True:
            parts.append("GPU device verified from the engine log")
        elif verified is False:
            parts.append("warning: the engine log did not name the chosen device")
        else:
            parts.append("GPU device not verified yet")
        return "; ".join(parts)

    def engine_view(self, engine: Engine) -> tuple[str, str, str]:
        """(status level, badge text, detail sentences) for the engine's row."""
        try:
            state = EngineState(self._engines.status(engine))
            reason = str(self._engines.reason(engine))
        except Exception as exc:
            log.exception("engine status query failed")
            return "critical", "Unknown", f"The status could not be read ({exc})."
        summary = self.engine_summary(engine)
        parts = summary.split("; ")[1:]
        if engine is Engine.LLAMA and state is EngineState.FAILED and reason == "no_model":
            return "caution", "Cleanup off", f"{_sentence(REASON_TEXT['no_model'])} Dictation runs without cleanup."
        detail = " ".join(_sentence(part) for part in parts)
        level = ENGINE_LEVELS.get(state, "idle")
        if state is EngineState.CPU_FALLBACK and reason == "cpu_selected":
            level = "ok"
        return level, _capital(STATE_WORDS[state]), detail

    def summary_text(self) -> str:
        lines = [f"{theme.APP_NAME} diagnostics, {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
        for engine in Engine:
            lines.append(f"{engine.value}: {self.engine_summary(engine)}")
        lines.append(f"GPU: {self._gpu_summary()}")
        lines.append("")
        medians = stage_medians(self._timings())
        for key, label in STAGES:
            value = medians[key]
            lines.append(f"{label}: {'no data' if value is None else f'{value:.0f} ms'}")
        lines.append("")
        lines.append("Guard rejections:")
        lines.append(self.guards_text())
        lines.append("")
        lines.append(self._hotkey_text())
        settings = self._config.settings
        lines.append("")
        lines.append(
            "Settings: keep_mic_warm="
            f"{settings.general.keep_mic_warm}, cleanup={settings.cleanup.enabled}, "
            f"idle_unload_minutes={settings.general.idle_unload_minutes}, "
            f"debug_logging={settings.diagnostics.debug_logging}"
        )
        return "\n".join(lines) + "\n"

    def median_label(self, key: str) -> QtWidgets.QLabel:
        return self._median_tiles[key].value

    def guard_counts(self) -> dict[str, int]:
        try:
            return {str(k): int(v) for k, v in dict(self._pipeline.guard_counts()).items()}
        except Exception:
            log.exception("guard_counts failed")
            return {}

    def guards_text(self) -> str:
        counts = self.guard_counts()
        if not counts:
            return "none since start"
        return "\n".join(f"{reason}: {count}" for reason, count in sorted(counts.items()))

    # Internals ------------------------------------------------------------------------------------

    def _timings(self) -> list[StageTimings]:
        try:
            return list(self._pipeline.recent_timings())
        except Exception:
            log.exception("recent_timings failed")
            return []

    def _captures(self) -> list[CaptureInfo]:
        try:
            return [
                info
                for info in self._pipeline.recent_captures()
                if isinstance(info, CaptureInfo)
            ]
        except Exception:
            log.exception("recent_captures failed")
            return []

    def _refresh_guards(self) -> None:
        counts = self.guard_counts()
        key = tuple(sorted(counts))
        if key != self._guard_key or not self.guards_card.rows():
            self._guard_key = key
            self.guards_card.clear()
            self.guard_rows = {}
            if not counts:
                row = SettingRow("No rejections since start", "Every cleanup result passed the guards.", leading=StatusDot("ok"))
                self.guards_card.add_row(row)
            for reason in sorted(counts):
                count = make_label("", "caption_strong")
                count.setProperty("badge", "muted")
                count.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                count.setMinimumWidth(36)
                count.setFixedHeight(24)
                row = SettingRow(reason.replace("_", " ").capitalize(), "", count)
                row.setMinimumHeight(48)
                row.setToolTip(reason)
                self.guards_card.add_row(row)
                self.guard_rows[reason] = count
        for reason, label in self.guard_rows.items():
            label.setText(str(counts.get(reason, 0)))

    def _refresh_hotkey(self) -> None:
        stats = getattr(self._hotkey, "stats", None)
        for (key, _label), tile in zip(HOTKEY_STATS, self.hotkey_tiles.tiles, strict=True):
            if stats is None:
                tile.set_value("No data", numeric=False)
                continue
            value = int(getattr(stats, key, 0) or 0)
            alarming = key in ("probes_missed", "hook_exceptions", "install_failures") and value > 0
            tile.set_value(str(value), "", "caution" if alarming else None)
        last = getattr(stats, "last_error", "") if stats is not None else ""
        if stats is None:
            self.hotkey_stats.setText("Hotkey stats unavailable.")
        elif last:
            self.hotkey_stats.setText(f"Last error: {last}")
        else:
            self.hotkey_stats.setText("No hook errors reported.")

    def _hotkey_text(self) -> str:
        stats = getattr(self._hotkey, "stats", None)
        if stats is None:
            return "Hotkey stats unavailable"
        text = (
            f"Liveness probes sent {getattr(stats, 'probes_sent', 0)}, missed "
            f"{getattr(stats, 'probes_missed', 0)}; hook reinstalls {getattr(stats, 'reinstalls', 0)}; "
            f"hook exceptions {getattr(stats, 'hook_exceptions', 0)}; install failures "
            f"{getattr(stats, 'install_failures', 0)}"
        )
        last = getattr(stats, "last_error", "")
        if last:
            text += f"; last error: {last}"
        return text

    def _gpu_detail(self) -> str:
        selection = self._selection
        if selection.raw_index is None:
            return "Both engines run their CPU builds."
        text = f"Device {selection.raw_index}, {selection.memory_mb} MB"
        for device in selection.devices:
            if device.raw_index == selection.raw_index and device.free_mb:
                text += f", {device.free_mb} MB free at the probe"
        return text + ". Changing the device restarts both engines."

    def _gpu_summary(self) -> str:
        selection = self._selection
        if selection.raw_index is None:
            return "No Vulkan GPU: both engines run their CPU builds"
        text = f"{selection.name} (device {selection.raw_index}, {selection.memory_mb} MB)"
        for device in selection.devices:
            if device.raw_index == selection.raw_index and device.free_mb:
                text += f", {device.free_mb} MB free at the probe"
        return text

    def _fill_override(self) -> None:
        self.gpu_override.blockSignals(True)
        try:
            self.gpu_override.clear()
            automatic = automatic_choice(self._selection.devices)
            label = "Automatic" + (f" ({automatic.name})" if automatic else "")
            self.gpu_override.addItem(label, None)
            for device in self._selection.devices:
                kind = "" if device.uma is None else (" integrated" if device.uma else " discrete")
                self.gpu_override.addItem(f"{device.name} ({device.memory_mb} MB{kind})", device.raw_index)
        finally:
            self.gpu_override.blockSignals(False)

    def _on_override(self, index: int) -> None:
        if self._loading:
            return
        raw_index = self.gpu_override.itemData(index)
        selection = choose_device(self._selection.devices, raw_index)

        def mutate(settings: Settings) -> Settings:
            return replace(
                settings,
                diagnostics=replace(settings.diagnostics, gpu_device_override=raw_index),
            )

        try:
            self._config.update(mutate)
        except SettingsError as exc:
            self._notify("warning", "GPU override", str(exc))
            return
        self._selection = selection
        # The app's settings subscriber rebuilds the device and model plan together.
        # Restarting here would retain the previous hardware's CPU-only policy.
        self.refresh()
        self.gpu_changed.emit()

    def _on_debug(self, checked: bool) -> None:
        if self._loading:
            return

        def mutate(settings: Settings) -> Settings:
            return replace(settings, diagnostics=replace(settings.diagnostics, debug_logging=bool(checked)))

        try:
            self._config.update(mutate)
        except SettingsError as exc:
            self._notify("warning", "Debug logging", str(exc))

    def _restart(self) -> None:
        for engine in Engine:
            try:
                self._engines.restart(engine)
            except Exception:
                log.exception("restart(%s) failed", engine.value)
        self.refresh()

    def _open_logs(self) -> None:
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._launcher(str(self.log_dir))
        except Exception as exc:
            log.exception("could not open the log folder")
            self._notify("warning", "Log folder", f"Could not open {self.log_dir}: {exc}")

    def _save_bundle(self) -> None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        default = Path.home() / "Desktop" / f"spells-diagnostics-{stamp}.zip"
        target = self._save_dialog(default)
        if target is None:
            return
        try:
            names = write_bundle(
                Path(target),
                log_dir=self.log_dir,
                settings_path=self._config.path,
                summary=self.summary_text(),
            )
        except OSError as exc:
            log.exception("could not write the diagnostics bundle")
            self._notify("warning", "Diagnostics bundle", f"Could not save the bundle: {exc}")
            return
        debug = self._config.settings.diagnostics.debug_logging
        note = BUNDLE_DEBUG_NOTE if debug else BUNDLE_CLEAN_NOTE
        self._notify("info", "Diagnostics bundle", f"Saved {len(names)} files to {target}. {note}")


__all__ = [
    "LICENSE_COMPONENTS",
    "MIC_OPEN_TARGET_MS",
    "STAGES",
    "DiagnosticsTab",
    "automatic_choice",
    "license_text",
    "licenses_dir",
    "manifest_entries",
    "median",
    "recommend_keep_mic_warm",
    "stage_medians",
    "write_bundle",
]
