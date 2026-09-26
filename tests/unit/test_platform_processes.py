import pytest

from spells import engines, gpu
from spells.platform.stub import StubProcesses

from .fake_platform import fake_platform


class Processes(StubProcesses):
    def __init__(self):
        super().__init__("linux")

    def hidden_process_kwargs(self):
        return {"marker": 1}


def test_probes_run_with_the_platform_process_flags(tmp_path, use_platform):
    use_platform(fake_platform(processes=Processes()))
    seen = {}

    def runner(args, **kwargs):
        seen.update(kwargs)

        class Result:
            stdout = "out"
            stderr = ""

        return Result()

    assert gpu._run_hidden(["probe"], None, tmp_path, runner).startswith("out")
    assert seen["marker"] == 1
    assert "creationflags" not in seen


@pytest.mark.windows
def test_windows_hides_console_windows_and_plans_cores():
    from spells.platform import windows

    processes = windows.build().processes
    assert processes.hidden_process_kwargs() == {"creationflags": 0x08000000}
    assert processes.cpu_plan() is not None


class RecordingProcesses(StubProcesses):
    def __init__(self):
        super().__init__("linux")
        self.job = object()
        self.spawn_calls = []

    def create_job(self):
        return self.job

    def spawn_hidden(self, args, **kwargs):
        self.spawn_calls.append((args, kwargs))
        return "spawned"


def test_platform_process_backend_forwards_create_job_and_spawn(tmp_path, use_platform):
    processes = RecordingProcesses()
    use_platform(fake_platform(processes=processes))
    backend = engines._PlatformProcessBackend()

    assert backend.create_job() is processes.job

    stdout_path = tmp_path / "out.log"
    stderr_path = tmp_path / "err.log"
    result = backend.spawn(
        ["engine"],
        env={"A": "1"},
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        cwd=tmp_path,
        job=processes.job,
        affinity_mask=0x3,
    )

    assert result == "spawned"
    args, kwargs = processes.spawn_calls[0]
    assert args == ["engine"]
    assert kwargs == {
        "env": {"A": "1"},
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "cwd": tmp_path,
        "job": processes.job,
        "affinity_mask": 0x3,
    }
