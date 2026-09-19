"""Smoke test for the built whisper-server variants (spec 19.2 and 19.3 step 3).

Each exe is copied into an empty temp directory and started with a PATH that holds only
C:\\Windows\\System32 and C:\\Windows, which proves it needs no MSYS2 DLLs. It must answer
/health, transcribe a synthetic 2 s tone posted to /inference as JSON, and exit when told to.
The Vulkan variant must additionally report a Vulkan device in its startup log.

Run with: .venv\\Scripts\\python.exe -m pytest tests/integration -m integration
"""

import http.client
import json
import math
import os
import shutil
import socket
import struct
import subprocess
import threading
import time
import uuid
import wave
from pathlib import Path

import pytest

REPO_DIR = Path(__file__).resolve().parents[2]
ENGINES_DIR = REPO_DIR / "build" / "out" / "engines"
MODEL = REPO_DIR / "build" / "cache" / "models" / "ggml-tiny.bin"
VARIANTS = ("vulkan", "cpu")
HEALTH_TIMEOUT_S = 30.0
INFERENCE_TIMEOUT_S = 120.0
ERROR_APPCONTROL_BLOCKED = 4551  # WinError raised by CreateProcess when code integrity rejects the exe

pytestmark = pytest.mark.integration


def _write_tone(path: Path, seconds: float = 2.0, rate: int = 16000, freq: float = 440.0) -> None:
    frames = int(seconds * rate)
    samples = (int(0.3 * 32767 * math.sin(2 * math.pi * freq * i / rate)) for i in range(frames))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(struct.pack(f"<{frames}h", *samples))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _clean_env(tmp_dir: Path) -> dict:
    """Only Windows system directories on PATH; nothing from the build toolchain."""
    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    env = {
        "PATH": os.pathsep.join([os.path.join(system_root, "System32"), system_root]),
        "SystemRoot": system_root,
        "windir": system_root,
        "SystemDrive": os.environ.get("SystemDrive", "C:"),
        "TEMP": str(tmp_dir),
        "TMP": str(tmp_dir),
    }
    for name in ("ProgramData", "ALLUSERSPROFILE", "LOCALAPPDATA", "USERPROFILE", "USERNAME"):
        if name in os.environ:
            env[name] = os.environ[name]
    return env


def _launch(cmd: list[str], *, cwd: Path, env: dict) -> subprocess.Popen:
    """Start the server. Windows code integrity (Smart App Control) may block an unsigned exe
    the first time it is seen and cache the verdict, so one block is retried before failing."""
    last_error = None
    for attempt in range(3):
        try:
            return subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
        except OSError as exc:
            if getattr(exc, "winerror", None) != ERROR_APPCONTROL_BLOCKED:
                raise
            last_error = exc
            time.sleep(5)
    pytest.fail(f"Windows code integrity policy (Smart App Control / WDAC) keeps blocking {cmd[0]}: "
                f"{last_error}. Check the Microsoft-Windows-CodeIntegrity/Operational event log.")


def _pump(stream, sink: list) -> threading.Thread:
    def reader():
        for line in iter(stream.readline, b""):
            sink.append(line.decode("utf-8", errors="replace").rstrip("\r\n"))
        stream.close()

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    return thread


def _wait_for_health(port: int, proc: subprocess.Popen, log) -> None:
    """Poll /health until 200; ``log`` is a callable returning the lines captured so far."""
    deadline = time.monotonic() + HEALTH_TIMEOUT_S
    last = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            pytest.fail(f"whisper-server exited early with {proc.returncode}:\n" + "\n".join(log()))
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            conn.request("GET", "/health")
            response = conn.getresponse()
            last = (response.status, response.read().decode("utf-8", errors="replace"))
            conn.close()
            if last[0] == 200:
                return
        except OSError as exc:
            last = ("no connection", str(exc))
        time.sleep(0.25)
    pytest.fail(f"/health did not return 200 within {HEALTH_TIMEOUT_S} s, last: {last}\n" + "\n".join(log()))


def _post_inference(port: int, wav_path: Path) -> tuple[int, bytes]:
    boundary = f"----spells{uuid.uuid4().hex}"
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="response_format"\r\n\r\njson\r\n',
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{wav_path.name}"\r\n'.encode(),
        b"Content-Type: audio/wav\r\n\r\n",
        wav_path.read_bytes(),
        b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ])
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=INFERENCE_TIMEOUT_S)
    conn.request("POST", "/inference", body=body,
                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                          "Content-Length": str(len(body))})
    response = conn.getresponse()
    payload = response.read()
    conn.close()
    return response.status, payload


@pytest.mark.parametrize("variant", VARIANTS)
def test_whisper_server_smoke(variant, tmp_path):
    exe = ENGINES_DIR / variant / "whisper-server.exe"
    if not exe.exists():
        pytest.skip(f"{exe} not built; run 'py build/build_whisper.py --variant {variant}'")
    if not MODEL.exists():
        pytest.skip(f"{MODEL} missing; run 'py build/fetch.py models.whisper_tiny_test'")

    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    exe_copy = clean_dir / exe.name
    shutil.copy2(exe, exe_copy)
    assert [p.name for p in clean_dir.iterdir()] == [exe.name], "clean directory must hold only the exe"

    wav_path = tmp_path / "tone.wav"
    _write_tone(wav_path)
    port = _free_port()

    proc = _launch(
        [str(exe_copy), "-m", str(MODEL), "--host", "127.0.0.1", "--port", str(port)],
        cwd=clean_dir,
        env=_clean_env(tmp_path),
    )
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    pumps: list[threading.Thread] = []
    try:
        # inside the try: a failure starting the readers must still terminate the server
        pumps = [_pump(proc.stdout, stdout_lines), _pump(proc.stderr, stderr_lines)]
        _wait_for_health(port, proc, lambda: stderr_lines + stdout_lines)

        status, payload = _post_inference(port, wav_path)
        assert status == 200, f"/inference returned {status}: {payload[:500]!r}"
        body = json.loads(payload.decode("utf-8"))
        assert "text" in body, f"no 'text' key in {body!r}"
        print(f"[{variant}] /inference text={body['text']!r}")
    finally:
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)
        for pump in pumps:
            pump.join(timeout=5)

    assert proc.returncode is not None, "whisper-server did not exit after terminate()"

    if variant == "vulkan":
        log = stderr_lines + stdout_lines
        vulkan_lines = [line for line in log if "ggml_vulkan" in line or "Vulkan" in line]
        assert vulkan_lines, "vulkan variant did not mention a Vulkan device at startup:\n" + "\n".join(log)
        device_lines = [line for line in vulkan_lines if "ggml_vulkan:" in line and "=" in line] or vulkan_lines
        print(f"[vulkan] {device_lines[0]}")
