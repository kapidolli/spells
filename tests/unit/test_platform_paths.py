from pathlib import Path

import pytest

from spells.platform.paths import (
    engine_suffix,
    linux_paths,
    macos_paths,
    paths_for,
    windows_paths,
)

HOME = Path("/home/ada")


def test_windows_paths_match_the_existing_locations():
    env = {"APPDATA": r"C:\Users\ada\AppData\Roaming", "LOCALAPPDATA": r"C:\Users\ada\AppData\Local"}
    paths = windows_paths(env, Path(r"C:\Users\ada"))
    assert paths.settings_path == Path(env["APPDATA"]) / "Spells" / "settings.json"
    assert paths.history_path == Path(env["LOCALAPPDATA"]) / "Spells" / "history.db"
    assert paths.log_dir == Path(env["LOCALAPPDATA"]) / "Spells" / "logs"


def test_windows_history_falls_back_to_the_home_folder():
    paths = windows_paths({"APPDATA": r"C:\R"}, Path(r"C:\Users\ada"))
    assert paths.history_path == Path(r"C:\Users\ada") / "AppData" / "Local" / "Spells" / "history.db"


def test_windows_paths_need_appdata_as_before():
    with pytest.raises(KeyError):
        windows_paths({}, Path(r"C:\Users\ada"))


def test_linux_paths_use_the_xdg_defaults():
    paths = linux_paths({}, HOME)
    assert paths.settings_path == HOME / ".config" / "spells" / "settings.json"
    assert paths.history_path == HOME / ".local" / "share" / "spells" / "history.db"
    assert paths.log_dir == HOME / ".local" / "state" / "spells" / "logs"


def test_linux_paths_follow_absolute_xdg_variables():
    env = {"XDG_CONFIG_HOME": "/cfg", "XDG_DATA_HOME": "/data", "XDG_STATE_HOME": "/state"}
    paths = linux_paths(env, HOME)
    assert paths.settings_path == Path("/cfg") / "spells" / "settings.json"
    assert paths.history_path == Path("/data") / "spells" / "history.db"
    assert paths.log_dir == Path("/state") / "spells" / "logs"


def test_linux_paths_ignore_relative_xdg_variables():
    paths = linux_paths({"XDG_CONFIG_HOME": "relative/cfg"}, HOME)
    assert paths.settings_path == HOME / ".config" / "spells" / "settings.json"


def test_macos_paths_live_in_the_library():
    paths = macos_paths({}, HOME)
    support = HOME / "Library" / "Application Support" / "Spells"
    assert paths.settings_path == support / "settings.json"
    assert paths.history_path == support / "history.db"
    assert paths.log_dir == HOME / "Library" / "Logs" / "Spells"


def test_paths_for_dispatches_by_system():
    assert paths_for("linux", {}, HOME) == linux_paths({}, HOME)
    assert paths_for("macos", {}, HOME) == macos_paths({}, HOME)


def test_engine_suffix():
    assert engine_suffix("windows") == ".exe"
    assert engine_suffix("linux") == ""
    assert engine_suffix("macos") == ""
