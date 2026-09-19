"""Install layout and user data locations (spec 15, 19.3 step 6, 20.1).

Every case resolves against a temporary tree, so nothing here depends on where the repo
or an installation actually sits.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from spells import paths
from spells.config import settings_path
from spells.history import default_history_path


def make_dev_tree(root: Path, *, models: tuple[str, ...] = (), pins: str | None = None) -> Path:
    (root / "build" / "out" / "engines" / "vulkan").mkdir(parents=True)
    (root / "build" / "out" / "engines" / "cpu").mkdir(parents=True)
    (root / "build" / "cache" / "models").mkdir(parents=True)
    (root / "build" / "licenses").mkdir(parents=True)
    (root / "data").mkdir(parents=True)
    for name in models:
        (root / "build" / "cache" / "models" / name).write_bytes(b"model")
    if pins is not None:
        entry = {"models": {"cleanup_model": {"url": f"https://example.invalid/{pins}"}}}
        (root / "build" / "pins.json").write_text(json.dumps(entry), encoding="utf-8")
    return root


def make_frozen_tree(root: Path, *, models: tuple[str, ...] = ()) -> Path:
    for part in ("engines/vulkan", "engines/cpu", "models", "data", "licenses"):
        (root / part).mkdir(parents=True)
    for name in models:
        (root / "models" / name).write_bytes(b"model")
    return root


def dev(root: Path, **kwargs) -> paths.Layout:
    return paths.resolve(frozen=False, repo_root=root, env={}, **kwargs)


def frozen(root: Path, **kwargs) -> paths.Layout:
    return paths.resolve(frozen=True, executable=root / "Spells.exe", env={}, **kwargs)


# Layout ---------------------------------------------------------------------------------


def test_dev_layout_points_into_the_build_folder(tmp_path):
    make_dev_tree(tmp_path)
    layout = dev(tmp_path)
    assert layout.frozen is False
    assert layout.root == tmp_path
    assert layout.vulkan_dir == tmp_path / "build" / "out" / "engines" / "vulkan"
    assert layout.cpu_dir == tmp_path / "build" / "out" / "engines" / "cpu"
    assert layout.models_dir == tmp_path / "build" / "cache" / "models"
    assert layout.data_dir == tmp_path / "data"
    assert layout.licenses_dir == tmp_path / "build" / "licenses"


def test_frozen_layout_sits_beside_the_executable(tmp_path):
    make_frozen_tree(tmp_path)
    layout = frozen(tmp_path)
    assert layout.frozen is True
    assert layout.root == tmp_path
    assert layout.vulkan_dir == tmp_path / "engines" / "vulkan"
    assert layout.cpu_dir == tmp_path / "engines" / "cpu"
    assert layout.models_dir == tmp_path / "models"
    assert layout.data_dir == tmp_path / "data"
    assert layout.licenses_dir == tmp_path / "licenses"


def test_engine_executables_and_engine_paths(tmp_path):
    make_frozen_tree(tmp_path, models=(paths.WHISPER_MODEL_NAME, paths.VAD_MODEL_NAME, "m.gguf"))
    layout = frozen(tmp_path)
    assert layout.whisper_exe("vulkan") == tmp_path / "engines" / "vulkan" / "whisper-server.exe"
    assert layout.llama_exe("cpu") == tmp_path / "engines" / "cpu" / "llama-server.exe"
    engine_paths = layout.engine_paths()
    assert engine_paths.vulkan_dir == layout.vulkan_dir
    assert engine_paths.cpu_dir == layout.cpu_dir
    assert engine_paths.whisper_model == tmp_path / "models" / paths.WHISPER_MODEL_NAME
    assert engine_paths.vad_model == tmp_path / "models" / paths.VAD_MODEL_NAME
    assert engine_paths.llama_model == tmp_path / "models" / "m.gguf"
    assert engine_paths.log_dir == layout.log_dir


# Model resolution -----------------------------------------------------------------------


def test_whisper_prefers_the_shipped_model(tmp_path):
    make_dev_tree(tmp_path, models=(paths.WHISPER_MODEL_NAME, paths.WHISPER_DEV_MODEL_NAME))
    layout = dev(tmp_path)
    assert layout.whisper_model == layout.models_dir / paths.WHISPER_MODEL_NAME
    assert layout.vad_model == layout.models_dir / paths.VAD_MODEL_NAME


def test_whisper_falls_back_to_tiny_with_a_warning(tmp_path, caplog):
    make_dev_tree(tmp_path, models=(paths.WHISPER_DEV_MODEL_NAME,))
    with caplog.at_level(logging.WARNING, logger="spells.paths"):
        layout = dev(tmp_path)
    assert layout.whisper_model == layout.models_dir / paths.WHISPER_DEV_MODEL_NAME
    assert paths.WHISPER_DEV_MODEL_NAME in caplog.text
    assert layout.whisper_notice == ""


def test_whisper_keeps_the_shipped_name_when_nothing_is_there(tmp_path):
    make_dev_tree(tmp_path)
    layout = dev(tmp_path)
    assert layout.whisper_model == layout.models_dir / paths.WHISPER_MODEL_NAME
    assert paths.WHISPER_MODEL_NAME in layout.whisper_notice


def test_a_frozen_build_never_falls_back_to_the_development_model(tmp_path):
    make_frozen_tree(tmp_path, models=(paths.WHISPER_DEV_MODEL_NAME,))
    layout = frozen(tmp_path)
    assert layout.whisper_model == layout.models_dir / paths.WHISPER_MODEL_NAME
    assert layout.whisper_model.name != paths.WHISPER_DEV_MODEL_NAME
    assert paths.WHISPER_MODEL_NAME in layout.whisper_notice
    assert str(layout.models_dir) in layout.whisper_notice


def test_a_frozen_build_with_its_model_has_no_notice(tmp_path):
    make_frozen_tree(tmp_path, models=(paths.WHISPER_MODEL_NAME,))
    layout = frozen(tmp_path)
    assert layout.whisper_model == layout.models_dir / paths.WHISPER_MODEL_NAME
    assert layout.whisper_notice == ""


def test_llama_model_comes_from_the_pins_file(tmp_path):
    make_dev_tree(tmp_path, models=("Qwen3-4B-Q4_K_M.gguf", "other.gguf"), pins="Qwen3-4B-Q4_K_M.gguf")
    layout = dev(tmp_path)
    assert layout.llama_model == layout.models_dir / "Qwen3-4B-Q4_K_M.gguf"
    assert layout.cleanup_notice == ""


def test_llama_model_falls_back_to_a_single_gguf(tmp_path):
    make_dev_tree(tmp_path, models=("qwen2.5-0.5b-instruct-q4_k_m.gguf",))
    layout = dev(tmp_path)
    assert layout.llama_model == layout.models_dir / "qwen2.5-0.5b-instruct-q4_k_m.gguf"


def test_llama_model_is_none_without_any_gguf(tmp_path):
    make_dev_tree(tmp_path)
    layout = dev(tmp_path)
    assert layout.llama_model is None
    assert "cleanup" in layout.cleanup_notice.lower()
    # No placeholder path: the supervisor never launches llama and reports no_model (spec 16).
    assert layout.engine_paths().llama_model is None


def test_llama_model_is_none_when_several_gguf_files_are_ambiguous(tmp_path):
    make_dev_tree(tmp_path, models=("a.gguf", "b.gguf"))
    layout = dev(tmp_path)
    assert layout.llama_model is None
    assert layout.cleanup_notice


def test_a_pins_file_that_cannot_be_read_is_survivable(tmp_path):
    make_dev_tree(tmp_path, models=("only.gguf",))
    (tmp_path / "build" / "pins.json").write_text("{ not json", encoding="utf-8")
    layout = dev(tmp_path)
    assert layout.llama_model == layout.models_dir / "only.gguf"


# User data locations --------------------------------------------------------------------


def test_user_data_defaults_to_the_windows_locations(tmp_path):
    make_dev_tree(tmp_path)
    layout = paths.resolve(frozen=False, repo_root=tmp_path, env=None)
    assert layout.settings_path == settings_path()
    assert layout.history_path == default_history_path()
    assert layout.log_dir == default_history_path().parent / "logs"


def test_environment_overrides_move_settings_history_and_logs(tmp_path):
    make_dev_tree(tmp_path)
    settings_dir = tmp_path / "cfg"
    data_dir = tmp_path / "userdata"
    env = {paths.SETTINGS_DIR_ENV: str(settings_dir), paths.DATA_DIR_ENV: str(data_dir)}
    layout = paths.resolve(frozen=False, repo_root=tmp_path, env=env)
    assert layout.settings_path == settings_dir / "settings.json"
    assert layout.history_path == data_dir / "history.db"
    assert layout.log_dir == data_dir / "logs"


@pytest.mark.parametrize("variant", ["vulkan", "cpu"])
def test_variant_dir_covers_both_engines(tmp_path, variant):
    make_frozen_tree(tmp_path)
    layout = frozen(tmp_path)
    assert layout.variant_dir(variant).name == variant


def test_recordings_dir_sits_beside_the_history_database(tmp_path):
    layout = paths.resolve(frozen=False, repo_root=tmp_path, env={paths.DATA_DIR_ENV: str(tmp_path / "data")})
    assert layout.history_path == tmp_path / "data" / "history.db"
    assert layout.recordings_dir == tmp_path / "data" / "recordings"


def test_recordings_dir_follows_the_real_profile(tmp_path):
    env = {"LOCALAPPDATA": str(tmp_path), "APPDATA": str(tmp_path)}
    layout = paths.resolve(frozen=False, repo_root=tmp_path, env=env)
    assert layout.recordings_dir == layout.history_path.parent / "recordings"


def test_resolve_creates_no_recordings_folder(tmp_path):
    layout = paths.resolve(frozen=False, repo_root=tmp_path, env={paths.DATA_DIR_ENV: str(tmp_path / "data")})
    assert not layout.recordings_dir.exists()
