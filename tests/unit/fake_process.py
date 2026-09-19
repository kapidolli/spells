"""In-process fakes for the engine supervisor tests.

One FakeWorld stands in for the process backend (the spells.win32.process contract the
supervisor uses: create_job and spawn) and for both HTTP client factories. Every spawn
produces a FakePopen that a test scripts: whether it answers health, what it writes to its
log file, whether warm-up requests fail, and when it exits. The fake clients look up the
process behind their base URL by port, so a process and its client always agree.

Log lines are written to the real stdout_path the supervisor passed, because the supervisor
reads its classification patterns from the log file exactly as it does in production.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path


class FakeClock:
    """A manually advanced monotonic clock in seconds. sleep() advances it."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds

    def sleep(self, seconds: float) -> None:
        self.t += seconds


class FakeJobObject:
    """Records the pids assigned to it; close() mirrors KILL_ON_JOB_CLOSE by marking closed."""

    def __init__(self) -> None:
        self.pids: list[int] = []
        self.handle: int | None = 0x1234
        self.closed = False

    def assign(self, pid: int) -> None:
        if self.handle is None:
            raise ValueError("job object is closed")
        self.pids.append(pid)

    def close(self) -> None:
        self.handle = None
        self.closed = True


class FakePopen:
    """One launched engine process, controlled by the test."""

    def __init__(
        self,
        pid: int,
        args: list[str],
        env: dict[str, str] | None,
        stdout_path: Path,
        stderr_path: Path,
        cwd: Path | None,
        affinity_mask: int | None = None,
    ) -> None:
        self.affinity_mask = affinity_mask
        self.pid = pid
        self.args = list(args)
        self.env = dict(env) if env is not None else None
        self.stdout_path = Path(stdout_path)
        self.stderr_path = Path(stderr_path)
        self.cwd = cwd
        self.returncode: int | None = None
        # While True (and the process is alive) health() answers True.
        self.healthy = True
        # Raised by the warm-up request when set.
        self.inference_error: Exception | None = None
        self.chat_error: Exception | None = None
        self.health_calls = 0
        self.inference_calls = 0
        self.chat_calls = 0
        self.transcribe_calls = 0
        self.transcribe_error: Exception | None = None
        self.inference_wavs: list[bytes] = []
        self.chats: list[tuple[str, str, int]] = []
        self.killed = False
        # Called on every health request, before the answer (to write log lines "late").
        self.on_health: Callable[[], None] | None = None

    # Introspection ---------------------------------------------------------

    @property
    def exe(self) -> Path:
        return Path(self.args[0])

    @property
    def engine(self) -> str:
        return "whisper" if "whisper" in self.exe.name else "llama"

    @property
    def variant(self) -> str:
        return self.exe.parent.name

    @property
    def model(self) -> str | None:
        value = self.arg_after("-m")
        return Path(value).name if value is not None else None

    @property
    def port(self) -> int:
        return int(self.args[self.args.index("--port") + 1])

    @property
    def alive(self) -> bool:
        return self.returncode is None

    def arg_after(self, flag: str) -> str | None:
        if flag not in self.args:
            return None
        return self.args[self.args.index(flag) + 1]

    # subprocess.Popen contract ---------------------------------------------

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            raise subprocess.TimeoutExpired(self.args, timeout or 0)
        return self.returncode

    def kill(self) -> None:
        if self.returncode is None:
            self.killed = True
            self.returncode = 1

    terminate = kill

    # Test controls -----------------------------------------------------------

    def write_log(self, text: str) -> None:
        with open(self.stdout_path, "a", encoding="utf-8") as fh:
            fh.write(text if text.endswith("\n") else text + "\n")

    def exit(self, code: int = 1, log: str = "") -> None:
        if log:
            self.write_log(log)
        self.returncode = code


def _health(proc: FakePopen | None) -> bool:
    if proc is None:
        return False
    proc.health_calls += 1
    if proc.on_health is not None:
        proc.on_health()
    return proc.alive and proc.healthy


class FakeWhisperClient:
    def __init__(self, world: FakeWorld, base_url: str) -> None:
        self.world = world
        self.base_url = base_url

    def health(self) -> bool:
        return _health(self.world.process_for(self.base_url))

    def inference(self, wav_bytes: bytes, *, language, prompt, **kwargs) -> dict:
        proc = self.world.process_for(self.base_url)
        if proc is None or not proc.alive:
            raise ConnectionError("no whisper process on " + self.base_url)
        proc.inference_calls += 1
        proc.inference_wavs.append(wav_bytes)
        if proc.inference_error is not None:
            raise proc.inference_error
        return {"text": ""}


class FakeLlamaAsrClient:
    def __init__(self, world: FakeWorld, base_url: str) -> None:
        self.world = world
        self.base_url = base_url

    def health(self) -> bool:
        return _health(self.world.process_for(self.base_url))

    def transcribe(self, wav_bytes: bytes, **kwargs) -> object:
        proc = self.world.process_for(self.base_url)
        if proc is None or not proc.alive:
            raise ConnectionError("no llama-asr process on " + self.base_url)
        proc.transcribe_calls += 1
        proc.inference_wavs.append(wav_bytes)
        if proc.transcribe_error is not None:
            raise proc.transcribe_error
        return None


class FakeLlamaClient:
    def __init__(self, world: FakeWorld, base_url: str) -> None:
        self.world = world
        self.base_url = base_url

    def health(self) -> bool:
        return _health(self.world.process_for(self.base_url))

    def chat(self, system: str, user: str, max_tokens: int) -> tuple[str, str]:
        proc = self.world.process_for(self.base_url)
        if proc is None or not proc.alive:
            raise ConnectionError("no llama process on " + self.base_url)
        proc.chat_calls += 1
        proc.chats.append((system, user, max_tokens))
        if proc.chat_error is not None:
            raise proc.chat_error
        return ("", "stop")


class FakeWorld:
    """Process backend plus client factories for one supervisor under test."""

    def __init__(self) -> None:
        self.processes: list[FakePopen] = []
        self.jobs: list[FakeJobObject] = []
        # Called with each new FakePopen right after spawn, before the supervisor sees it.
        self.on_spawn: Callable[[FakePopen], None] | None = None
        # When set, spawn() raises it instead of creating a process.
        self.spawn_error: Exception | None = None
        self._next_pid = 4000

    # Backend contract ----------------------------------------------------------

    def create_job(self) -> FakeJobObject:
        job = FakeJobObject()
        self.jobs.append(job)
        return job

    def spawn(
        self,
        args: list[str],
        *,
        env: dict[str, str] | None,
        stdout_path: Path,
        stderr_path: Path,
        cwd: Path | None,
        job: FakeJobObject | None,
        affinity_mask: int | None = None,
    ) -> FakePopen:
        if self.spawn_error is not None:
            raise self.spawn_error
        Path(stdout_path).parent.mkdir(parents=True, exist_ok=True)
        Path(stdout_path).touch()
        proc = FakePopen(
            self._next_pid, args, env, stdout_path, stderr_path, cwd, affinity_mask=affinity_mask
        )
        self._next_pid += 1
        if job is not None:
            job.assign(proc.pid)
        self.processes.append(proc)
        if self.on_spawn is not None:
            self.on_spawn(proc)
        return proc

    # Client factories --------------------------------------------------------

    def whisper_client(self, base_url: str) -> FakeWhisperClient:
        return FakeWhisperClient(self, base_url)

    def llama_client(self, base_url: str) -> FakeLlamaClient:
        return FakeLlamaClient(self, base_url)

    def llama_asr_client(self, base_url: str) -> FakeLlamaAsrClient:
        return FakeLlamaAsrClient(self, base_url)

    # Test helpers ------------------------------------------------------------

    def process_for(self, base_url: str) -> FakePopen | None:
        port = int(base_url.rstrip("/").rsplit(":", 1)[1])
        for proc in reversed(self.processes):
            if proc.port == port:
                return proc
        return None

    def spawned(self, engine: str | None = None, variant: str | None = None) -> list[FakePopen]:
        return [
            proc
            for proc in self.processes
            if (engine is None or proc.engine == engine)
            and (variant is None or proc.variant == variant)
        ]

    def live(self, engine: str | None = None, variant: str | None = None) -> list[FakePopen]:
        return [proc for proc in self.spawned(engine, variant) if proc.alive]

    def latest(self, engine: str, variant: str | None = None) -> FakePopen:
        procs = self.spawned(engine, variant)
        if not procs:
            raise AssertionError(f"no {variant or ''} {engine} process was spawned")
        return procs[-1]
