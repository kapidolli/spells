"""The real process: start, second launch, --quit (spec 15, 19.5, 20.2).

Launches `.venv/Scripts/python.exe -m spells` with SPELLS_SETTINGS_DIR and SPELLS_DATA_DIR
pointing at a temporary directory, so the developer's own settings, history and logs are
never touched. The temporary settings have autostart off, so the real HKCU Run key stays
untouched as well, and both tests assert that it did not change. Skips with a reason when an
engine binary or a model is missing.

Run with:

    .venv/Scripts/python.exe -m pytest -m integration tests/integration/test_app_live.py -s
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from spells import autostart, paths
from spells.config import default_settings, save
from spells.win32.instance import find_message_window

pytestmark = pytest.mark.integration

REPO_DIR = Path(__file__).resolve().parents[2]
PYTHON = REPO_DIR / ".venv" / "Scripts" / "python.exe"
WINDOW_CLASS = "SpellsMessageWindow"
CREATE_NO_WINDOW = 0x08000000
START_TIMEOUT_S = 60.0
QUIT_TIMEOUT_S = 10.0
SECOND_LAUNCH_TIMEOUT_S = 30.0
ENGINE_IMAGES = ("whisper-server.exe", "llama-server.exe")


def missing_pieces() -> list[str]:
    layout = paths.resolve(frozen=False, repo_root=REPO_DIR, env={})
    missing = []
    if not PYTHON.exists():
        missing.append(str(PYTHON))
    for variant in ("vulkan",):
        for name in ENGINE_IMAGES:
            exe = layout.variant_dir(variant) / name
            if not exe.exists():
                missing.append(str(exe))
    if not layout.whisper_model.exists():
        missing.append(str(layout.whisper_model))
    if not layout.vad_model.exists():
        missing.append(str(layout.vad_model))
    if layout.llama_model is None or not layout.llama_model.exists():
        missing.append(f"a cleanup model in {layout.models_dir}")
    return missing


_MISSING = missing_pieces()
if _MISSING:
    pytest.skip(f"missing for the live app test: {', '.join(_MISSING)}", allow_module_level=True)


def engine_pids() -> dict[str, set[int]]:
    """The pids of every running engine image, by image name."""
    found: dict[str, set[int]] = {name: set() for name in ENGINE_IMAGES}
    for name in ENGINE_IMAGES:
        result = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {name}", "/NH", "/FO", "CSV"],
            capture_output=True,
            text=True,
            check=False,
        )
        for line in result.stdout.splitlines():
            fields = [field.strip('"') for field in line.strip().split('","')]
            if len(fields) > 1 and fields[0].lower() == name.lower():
                try:
                    found[name].add(int(fields[1]))
                except ValueError:
                    continue
    return found


def wait_for_window(timeout_s: float) -> int:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        hwnd = find_message_window(WINDOW_CLASS)
        if hwnd:
            return hwnd
        time.sleep(0.2)
    return 0


def wait_for_no_window(timeout_s: float) -> float:
    started = time.monotonic()
    deadline = started + timeout_s
    while time.monotonic() < deadline:
        if not find_message_window(WINDOW_CLASS):
            return time.monotonic() - started
        time.sleep(0.1)
    return -1.0


def environment(tmp_path: Path) -> dict[str, str]:
    env = dict(os.environ)
    env[paths.SETTINGS_DIR_ENV] = str(tmp_path / "cfg")
    env[paths.DATA_DIR_ENV] = str(tmp_path / "data")
    env["PYTHONPATH"] = str(REPO_DIR / "src")
    return env


def run_cli(tmp_path: Path, *args: str, timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(PYTHON), "-m", "spells", *args],
        env=environment(tmp_path),
        cwd=str(REPO_DIR),
        creationflags=CREATE_NO_WINDOW,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def spawn(tmp_path: Path, *args: str) -> subprocess.Popen:
    env = environment(tmp_path)
    return subprocess.Popen(
        [str(PYTHON), "-m", "spells", *args],
        env=env,
        cwd=str(REPO_DIR),
        creationflags=CREATE_NO_WINDOW,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


@pytest.fixture
def workspace(tmp_path) -> Path:
    """A throwaway user profile with autostart off, so nothing reaches the real Run key."""
    settings = default_settings()
    settings = replace(settings, general=replace(settings.general, autostart=False))
    (tmp_path / "cfg").mkdir(parents=True, exist_ok=True)
    save(settings, tmp_path / "cfg" / "settings.json")
    before = autostart.current_value()
    yield tmp_path
    assert autostart.current_value() == before, "the live test changed the real autostart entry"


def app_log(tmp_path: Path) -> str:
    log_file = tmp_path / "data" / "logs" / "app.log"
    if not log_file.exists():
        return ""
    return log_file.read_text(encoding="utf-8", errors="replace")


def test_quit_without_a_running_instance_returns_at_once(workspace):
    if find_message_window(WINDOW_CLASS):
        pytest.skip("a Spells instance is already running on this desktop")
    started = time.monotonic()
    proc = spawn(workspace, "--quit")
    out, _ = proc.communicate(timeout=60)
    assert proc.returncode == 0, out
    assert time.monotonic() - started < 30.0


def test_start_second_launch_and_quit(workspace):
    if find_message_window(WINDOW_CLASS):
        pytest.skip("a Spells instance is already running on this desktop")
    before = engine_pids()
    proc = spawn(workspace, *[])
    try:
        hwnd = wait_for_window(START_TIMEOUT_S)
        if not hwnd:
            out = proc.communicate(timeout=10)[0] if proc.poll() is not None else ""
            pytest.fail(
                f"the hidden window never appeared within {START_TIMEOUT_S} s.\n"
                f"process output:\n{out}\napp log:\n{app_log(workspace)}"
            )
        assert proc.poll() is None, "the app exited while its window was up"

        started = time.monotonic()
        second = run_cli(workspace, timeout=SECOND_LAUNCH_TIMEOUT_S)
        assert second.returncode == 0, second.stdout + second.stderr
        assert time.monotonic() - started < SECOND_LAUNCH_TIMEOUT_S
        assert proc.poll() is None, "the second launch killed the running instance"
        assert find_message_window(WINDOW_CLASS) == hwnd

        quit_run = run_cli(workspace, "--quit", timeout=60)
        assert quit_run.returncode == 0, quit_run.stdout + quit_run.stderr
        elapsed = wait_for_no_window(QUIT_TIMEOUT_S)
        assert elapsed >= 0.0, f"the window outlived --quit.\napp log:\n{app_log(workspace)}"
        proc.wait(timeout=20)
        assert proc.returncode == 0, proc.stdout.read() if proc.stdout else proc.returncode
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)

    after = engine_pids()
    for name in ENGINE_IMAGES:
        leaked = after[name] - before[name]
        assert not leaked, f"{name} survived the quit: {sorted(leaked)}"

    text = app_log(workspace)
    assert "Shutdown took" in text, text[-4000:]
    assert (workspace / "cfg" / "settings.json").exists()
    print("\n--- app log tail ---\n" + "\n".join(text.splitlines()[-25:]), file=sys.stderr)
