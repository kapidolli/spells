"""Autostart through HKCU Run (spec 15, 20.1).

Every test writes under HKCU\\Software\\SpellsTest\\Run, never the real Run key, by passing
the key path in. The fixture deletes the test tree afterwards.
"""

from __future__ import annotations

import sys
import winreg
from pathlib import Path

import pytest

from spells import autostart

TEST_ROOT = r"Software\SpellsTest"
TEST_KEY = TEST_ROOT + r"\Run"
COMMAND = r'"C:\Users\someone\AppData\Local\Programs\Spells\Spells.exe"'


def delete_tree(path: str) -> None:
    try:
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, path)
    except OSError:
        pass


@pytest.fixture
def key_path() -> str:
    delete_tree(TEST_KEY)
    delete_tree(TEST_ROOT)
    yield TEST_KEY
    delete_tree(TEST_KEY)
    delete_tree(TEST_ROOT)


def value(key_path: str) -> str | None:
    return autostart.current_value(key_path=key_path)


# Writing and deleting -------------------------------------------------------------------


def test_enabling_writes_the_run_value(key_path):
    assert autostart.apply(True, COMMAND, key_path=key_path) is True
    assert value(key_path) == COMMAND
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
        stored, kind = winreg.QueryValueEx(key, autostart.VALUE_NAME)
    assert stored == COMMAND
    assert kind == winreg.REG_SZ


def test_disabling_removes_the_run_value(key_path):
    autostart.apply(True, COMMAND, key_path=key_path)
    assert autostart.apply(False, COMMAND, key_path=key_path) is True
    assert value(key_path) is None


def test_disabling_twice_is_harmless(key_path):
    assert autostart.apply(False, COMMAND, key_path=key_path) is True
    assert autostart.apply(False, COMMAND, key_path=key_path) is True
    assert value(key_path) is None


def test_enabling_twice_writes_once(key_path, monkeypatch):
    autostart.apply(True, COMMAND, key_path=key_path)
    writes: list[str] = []
    real = winreg.SetValueEx

    def counted(key, name, reserved, kind, data):
        writes.append(data)
        return real(key, name, reserved, kind, data)

    monkeypatch.setattr(autostart.winreg, "SetValueEx", counted)
    assert autostart.apply(True, COMMAND, key_path=key_path) is True
    assert writes == []
    assert autostart.apply(True, COMMAND + " --settings", key_path=key_path) is True
    assert writes == [COMMAND + " --settings"]


def test_a_registry_error_is_logged_and_never_raised(key_path, monkeypatch, caplog):
    def boom(*args, **kwargs):
        raise OSError(5, "access denied")

    monkeypatch.setattr(autostart.winreg, "CreateKeyEx", boom)
    monkeypatch.setattr(autostart.winreg, "OpenKey", boom)
    with caplog.at_level("WARNING", logger="spells.autostart"):
        assert autostart.apply(True, COMMAND, key_path=key_path) is False
        assert autostart.apply(False, COMMAND, key_path=key_path) is False
        assert autostart.current_value(key_path=key_path) is None
    assert "autostart" in caplog.text.lower()


def test_current_value_is_none_without_the_key(key_path):
    assert value(key_path) is None


# The command ----------------------------------------------------------------------------


def test_command_for_a_frozen_build():
    command = autostart.current_command(frozen=True, executable=Path(r"C:\Programs\Spells\Spells.exe"))
    assert command == r'"C:\Programs\Spells\Spells.exe"'


def test_command_in_development_uses_pythonw_and_the_module(tmp_path):
    scripts = tmp_path / "Scripts"
    scripts.mkdir()
    pythonw = scripts / "pythonw.exe"
    pythonw.write_bytes(b"")
    command = autostart.current_command(frozen=False, executable=scripts / "python.exe")
    assert command == f'"{pythonw}" -m spells'


def test_command_falls_back_to_the_running_interpreter(tmp_path):
    scripts = tmp_path / "Scripts"
    scripts.mkdir()
    command = autostart.current_command(frozen=False, executable=scripts / "python.exe")
    assert command == f'"{scripts / "python.exe"}" -m spells'


def test_command_defaults_to_this_process():
    command = autostart.current_command()
    assert command.startswith('"')
    assert command.endswith('" -m spells')
    assert "python" in command.lower()
    assert str(Path(sys.executable).parent) in command
