"""Where the engines, the models, the data files and the user's own files live (spec 15).

One `Layout`, resolved once at startup, answers every question about the install tree. A
frozen PyInstaller build keeps everything beside the executable, as the installer lays it
out (spec 19.3 step 6); a development checkout uses the build folder the pipeline writes
into. The user's settings, history, recordings and logs follow spec 15, with two
environment overrides (`SPELLS_SETTINGS_DIR` and `SPELLS_DATA_DIR`) so the live test can
point a real process at a temporary directory instead of the real profile. The recordings
folder sits beside the history database and exists only once the user turns recordings on.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from spells.config import settings_path
from spells.engines import EnginePaths, Variant
from spells.history import RECORDINGS_DIR_NAME, default_history_path, recordings_dir_for

log = logging.getLogger(__name__)

SETTINGS_DIR_ENV = "SPELLS_SETTINGS_DIR"
DATA_DIR_ENV = "SPELLS_DATA_DIR"

WHISPER_MODEL_NAME = "ggml-large-v3-turbo-q8_0.bin"
WHISPER_DEV_MODEL_NAME = "ggml-tiny.bin"
VAD_MODEL_NAME = "ggml-silero-v5.1.2.bin"

SETTINGS_FILE = "settings.json"
HISTORY_FILE = "history.db"
LOGS_DIR = "logs"
RECORDINGS_DIR = RECORDINGS_DIR_NAME
PINS_FILE = "pins.json"


@dataclass(frozen=True)
class Layout:
    """Every path the app needs, resolved once (spec 15)."""

    frozen: bool
    root: Path
    vulkan_dir: Path
    cpu_dir: Path
    models_dir: Path
    data_dir: Path
    licenses_dir: Path
    settings_path: Path
    history_path: Path
    recordings_dir: Path
    log_dir: Path
    whisper_model: Path
    vad_model: Path
    llama_model: Path | None
    # Empty unless the cleanup model is missing; the app shows it as a tray warning.
    cleanup_notice: str = ""
    # Empty unless the speech model is missing; a packaged build never falls back to the
    # development model, so this is a startup failure the app reports (spec 16).
    whisper_notice: str = ""

    def variant_dir(self, variant: Variant) -> Path:
        return self.vulkan_dir if variant == "vulkan" else self.cpu_dir

    def whisper_exe(self, variant: Variant) -> Path:
        return self.variant_dir(variant) / "whisper-server.exe"

    def llama_exe(self, variant: Variant) -> Path:
        return self.variant_dir(variant) / "llama-server.exe"

    def engine_paths(self) -> EnginePaths:
        """The supervisor's view of this layout (spec 13).

        Without a cleanup model llama_model stays None: the supervisor then never launches
        llama and reports it FAILED with reason no_model, a tray warning (spec 14.1, 16).
        """
        return EnginePaths(
            vulkan_dir=self.vulkan_dir,
            cpu_dir=self.cpu_dir,
            whisper_model=self.whisper_model,
            vad_model=self.vad_model,
            llama_model=self.llama_model,
            log_dir=self.log_dir,
        )


def resolve(
    *,
    frozen: bool | None = None,
    executable: Path | None = None,
    repo_root: Path | None = None,
    env: Mapping[str, str] | None = None,
    pins_path: Path | None = None,
) -> Layout:
    """Work out the layout. Only the defaults touch the real process and environment."""
    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else bool(frozen)
    environ = os.environ if env is None else env
    if is_frozen:
        exe = Path(executable) if executable is not None else Path(sys.executable)
        root = exe.resolve().parent
        engines = root / "engines"
        models_dir = root / "models"
        data_dir = root / "data"
        licenses_dir = root / "licenses"
        pins = None
    else:
        root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
        engines = root / "build" / "out" / "engines"
        models_dir = root / "build" / "cache" / "models"
        data_dir = root / "data"
        licenses_dir = root / "build" / "licenses"
        pins = Path(pins_path) if pins_path is not None else root / "build" / PINS_FILE

    history_path = _history_path(environ)
    whisper_model, whisper_notice = _whisper_model(models_dir, frozen=is_frozen)
    llama_model, cleanup_notice = _llama_model(models_dir, pins)
    return Layout(
        frozen=is_frozen,
        root=root,
        vulkan_dir=engines / "vulkan",
        cpu_dir=engines / "cpu",
        models_dir=models_dir,
        data_dir=data_dir,
        licenses_dir=licenses_dir,
        settings_path=_settings_path(environ),
        history_path=history_path,
        recordings_dir=recordings_dir_for(history_path),
        log_dir=_log_dir(environ),
        whisper_model=whisper_model,
        vad_model=models_dir / VAD_MODEL_NAME,
        llama_model=llama_model,
        cleanup_notice=cleanup_notice,
        whisper_notice=whisper_notice,
    )


# User data (spec 15) --------------------------------------------------------------------


def _settings_path(env: Mapping[str, str]) -> Path:
    override = env.get(SETTINGS_DIR_ENV)
    if override:
        return Path(override) / SETTINGS_FILE
    return settings_path()


def _history_path(env: Mapping[str, str]) -> Path:
    override = env.get(DATA_DIR_ENV)
    if override:
        return Path(override) / HISTORY_FILE
    return default_history_path()


def _log_dir(env: Mapping[str, str]) -> Path:
    override = env.get(DATA_DIR_ENV)
    if override:
        return Path(override) / LOGS_DIR
    return default_history_path().parent / LOGS_DIR


# Models ---------------------------------------------------------------------------------


def _whisper_model(models_dir: Path, *, frozen: bool) -> tuple[Path, str]:
    """The speech model and, when it is missing, the reason (spec 16).

    The tiny development model is a fallback for a checkout only: shipping it would quietly
    turn a released build into one that transcribes badly, so a frozen build that misses its
    model reports the failure instead and lets the engine fail visibly.
    """
    shipped = models_dir / WHISPER_MODEL_NAME
    if _exists(shipped):
        return shipped, ""
    if not frozen and _exists(models_dir / WHISPER_DEV_MODEL_NAME):
        log.warning(
            "%s is missing; falling back to %s, which is a development model and transcribes badly",
            WHISPER_MODEL_NAME,
            WHISPER_DEV_MODEL_NAME,
        )
        return models_dir / WHISPER_DEV_MODEL_NAME, ""
    return shipped, f"The speech model {WHISPER_MODEL_NAME} is missing from {models_dir}"


def _llama_model(models_dir: Path, pins: Path | None) -> tuple[Path | None, str]:
    """The cleanup model: the pinned name, else a single stray GGUF, else none (spec 19.1)."""
    pinned = _pinned_cleanup_name(pins)
    if pinned:
        candidate = models_dir / pinned
        if _exists(candidate):
            return candidate, ""
    found = sorted(p for p in _gguf_files(models_dir) if p.is_file())
    if len(found) == 1:
        return found[0], ""
    if not found:
        return None, f"Cleanup is unavailable: no cleanup model in {models_dir}"
    names = ", ".join(p.name for p in found)
    return None, f"Cleanup is unavailable: {models_dir} holds several models ({names})"


def _gguf_files(models_dir: Path) -> list[Path]:
    try:
        return list(models_dir.glob("*.gguf"))
    except OSError:
        log.warning("Could not list %s", models_dir, exc_info=True)
        return []


def _pinned_cleanup_name(pins: Path | None) -> str:
    """The file name of models.cleanup_model in build/pins.json, empty when unavailable."""
    if pins is None or not _exists(pins):
        return ""
    try:
        data = json.loads(pins.read_text(encoding="utf-8"))
        entry = data["models"]["cleanup_model"]
        name = entry.get("filename") or str(entry["url"]).rsplit("/", 1)[-1]
    except Exception:
        log.warning("Could not read the cleanup model pin from %s", pins, exc_info=True)
        return ""
    return name if name and "/" not in name and "\\" not in name else ""


def _exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


__all__ = [
    "DATA_DIR_ENV",
    "RECORDINGS_DIR",
    "SETTINGS_DIR_ENV",
    "VAD_MODEL_NAME",
    "WHISPER_DEV_MODEL_NAME",
    "WHISPER_MODEL_NAME",
    "Layout",
    "resolve",
]
