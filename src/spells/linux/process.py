"""Engine processes and the processor plan on Linux.

Windows ties the engines to a job object so they die with the app. Here each engine starts
under `setpriv --pdeathsig TERM`, so the kernel sends it SIGTERM when the thread that started
it goes away, a crash included, and the job is a list the app can close to stop whatever is
still running.

On a hybrid Intel processor the kernel lists the performance cores in
/sys/devices/cpu_core/cpus. The plan pins the engines to one logical processor per
performance core, as the Windows plan does.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from collections.abc import Iterable
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from spells.models import CpuPlan

log = logging.getLogger(__name__)

MAX_ENGINE_THREADS = 8
SYS_CPU = Path("/sys/devices/system/cpu")
P_CORES = Path("/sys/devices/cpu_core/cpus")
E_CORES = Path("/sys/devices/cpu_atom/cpus")



def with_deathsig(args: list[str], setpriv: str | None) -> list[str]:
    return [setpriv, "--pdeathsig", "TERM", "--", *args] if setpriv else list(args)


class EngineJob:
    def __init__(self) -> None:
        self._procs: list[subprocess.Popen] = []

    def add(self, proc: subprocess.Popen) -> None:
        self._procs = [item for item in self._procs if item.poll() is None]
        self._procs.append(proc)

    def close(self) -> None:
        for proc in self._procs:
            if proc.poll() is None:
                proc.terminate()
        self._procs = []


def mask_cpus(mask: int) -> set[int]:
    return {index for index in range(mask.bit_length()) if mask >> index & 1}


def spawn_hidden(
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    stdout_path: Path,
    stderr_path: Path,
    cwd: Path | None = None,
    job: Any = None,
    affinity_mask: int | None = None,
) -> subprocess.Popen:
    with ExitStack() as stack:
        stdout_file = stack.enter_context(open(stdout_path, "ab"))
        if Path(stderr_path) == Path(stdout_path):
            stderr_target: Any = subprocess.STDOUT
        else:
            stderr_target = stack.enter_context(open(stderr_path, "ab"))
        proc = subprocess.Popen(
            with_deathsig(args, shutil.which("setpriv")),
            env=env,
            cwd=str(cwd) if cwd is not None else None,
            stdin=subprocess.DEVNULL,
            stdout=stdout_file,
            stderr=stderr_target,
        )
    if affinity_mask:
        try:
            os.sched_setaffinity(proc.pid, mask_cpus(affinity_mask))
        except OSError:
            log.warning("could not pin process %d to mask %x", proc.pid, affinity_mask)
    if isinstance(job, EngineJob):
        job.add(proc)
    return proc


def parse_cpu_list(text: str) -> set[int]:
    cpus: set[int] = set()
    for part in text.strip().split(","):
        if not part:
            continue
        first, _, last = part.partition("-")
        cpus.update(range(int(first), int(last or first) + 1))
    return cpus


def one_per_core(cpus: Iterable[int], siblings: dict[int, set[int]]) -> list[int]:
    chosen: list[int] = []
    covered: set[int] = set()
    for cpu in sorted(cpus):
        if cpu in covered:
            continue
        chosen.append(cpu)
        covered |= siblings.get(cpu, {cpu})
    return chosen


def plan_from(p_cores: set[int], hybrid: bool, siblings: dict[int, set[int]]) -> CpuPlan:
    if not hybrid or not p_cores:
        return CpuPlan()
    chosen = one_per_core(p_cores, siblings)
    mask = 0
    for cpu in chosen:
        mask |= 1 << cpu
    return CpuPlan(threads=min(len(chosen), MAX_ENGINE_THREADS), affinity_mask=mask)


def detect_cpu_plan() -> CpuPlan:
    try:
        if not (P_CORES.exists() and E_CORES.exists()):
            return CpuPlan()
        p_cores = parse_cpu_list(P_CORES.read_text())
        siblings = {
            cpu: parse_cpu_list(
                (SYS_CPU / f"cpu{cpu}" / "topology" / "thread_siblings_list").read_text()
            )
            for cpu in p_cores
        }
    except (OSError, ValueError) as exc:
        log.warning("could not read the processor topology: %s", exc)
        return CpuPlan()
    plan = plan_from(p_cores, True, siblings)
    if plan.affinity_mask is not None:
        log.info(
            "hybrid processor: engines pinned to %d performance cores (mask %s)",
            plan.threads,
            plan.cpu_mask_hex,
        )
    return plan
