"""Synthesise a few English clips so bench/run.py --smoke can be exercised without a microphone.

Windows ships System.Speech, so the clips are spoken by SAPI through PowerShell straight into a
16 kHz mono 16-bit WAV. When SAPI is unavailable the script falls back to copying
data/warmup.wav once per clip and says so: the run then still exercises every code path, but
the transcripts and the word error rates mean nothing.

Usage:
    py bench/smoke_clips.py
    py bench/smoke_clips.py --out bench/clips/smoke

The clips land in bench/clips/smoke/ with a manifest bench/run.py can read. bench/clips/ is
gitignored, so nothing here is ever committed.
"""

from __future__ import annotations

import argparse
import datetime
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
BENCH_DIR = REPO_DIR / "bench"
for _extra in (REPO_DIR / "src", BENCH_DIR):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import prompts as prompt_set

CREATE_NO_WINDOW = 0x08000000
SAPI_TIMEOUT_S = 120

# Two short lines and one long enough to land in the 10 to 30 s bucket of spec 12, so a smoke
# run also exercises the pass or fail table rather than only the short rows.
SMOKE_TEXTS: tuple[tuple[str, str, str], ...] = (
    (
        "smoke-short-1",
        "short",
        "Push it to main.",
    ),
    (
        "smoke-filler-1",
        "filler",
        (
            "So I was thinking we could move the standup to nine fifteen, because half the team "
            "is still on the train at nine."
        ),
    ),
    (
        "smoke-long-1",
        "long",
        (
            "Hi Marta, thanks for sending the draft over. I read it on the train this morning "
            "and I only have two small comments, both on page four. Could we go through them on "
            "Thursday at half past two? I will book the small meeting room, and I will bring "
            "the printout."
        ),
    ),
)

_SAPI_SCRIPT = """Add-Type -AssemblyName System.Speech
$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(
    16000,
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
    [System.Speech.AudioFormat.AudioChannel]::Mono)
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $synth.SetOutputToWaveFile(__PATH__, $format)
    $synth.Speak(__TEXT__)
} finally {
    $synth.Dispose()
}
"""


def _now() -> str:
    """Local wall clock with its offset, for the manifest's recorded_at."""
    return datetime.datetime.now(datetime.UTC).astimezone().isoformat(timespec="seconds")


def _ps_literal(value: str) -> str:
    """A PowerShell single-quoted literal: only the quote itself needs escaping."""
    return "'" + value.replace("'", "''") + "'"


def synthesise(text: str, destination: Path) -> bool:
    """Speak text into destination with SAPI. False when PowerShell or System.Speech refused.

    The script goes into a temporary .ps1 run with -File rather than -Command: the path and the
    text are embedded as PowerShell literals, so no quoting has to survive the Windows command
    line twice.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    script = _SAPI_SCRIPT.replace("__PATH__", _ps_literal(str(destination))).replace(
        "__TEXT__", _ps_literal(text)
    )
    handle, script_path = tempfile.mkstemp(suffix=".ps1", prefix="spells-sapi-")
    os.close(handle)
    script_file = Path(script_path)
    try:
        script_file.write_text(script, encoding="utf-8")
        completed = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script_file),
            ],
            capture_output=True,
            text=True,
            timeout=SAPI_TIMEOUT_S,
            creationflags=CREATE_NO_WINDOW,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"  SAPI unavailable: {exc}", file=sys.stderr)
        return False
    finally:
        script_file.unlink(missing_ok=True)
    if completed.returncode != 0 or not destination.is_file():
        print(f"  SAPI failed: {(completed.stderr or '').strip()[:300]}", file=sys.stderr)
        return False
    return True


def wav_info(path: Path) -> tuple[float, int]:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate(), handle.getframerate()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=BENCH_DIR / "clips" / "smoke")
    args = parser.parse_args(argv)

    warmup = REPO_DIR / "data" / "warmup.wav"
    entries: list[prompt_set.ManifestEntry] = []
    synthetic = True
    for clip_id, category, text in SMOKE_TEXTS:
        destination = args.out / f"{clip_id}.wav"
        print(f"{clip_id}: {text[:60]}...")
        if not synthesise(text, destination):
            synthetic = False
            if not warmup.is_file():
                print(f"ERROR: SAPI failed and {warmup} is missing.", file=sys.stderr)
                return 1
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(warmup, destination)
            print(f"  Fell back to a copy of {warmup.name}; the transcripts will not match.")
        duration_s, sample_rate = wav_info(destination)
        if sample_rate != 16000:
            print(
                f"  WARNING: {destination.name} is {sample_rate} Hz, not 16 kHz; whisper-server "
                "expects 16 kHz mono.",
                file=sys.stderr,
            )
        entries.append(
            prompt_set.ManifestEntry(
                id=clip_id,
                language="en",
                category=category,
                reference=text,
                text=text,
                duration_s=round(duration_s, 3),
                recorded_at=_now(),
                file=destination.name,
            )
        )
        print(f"  {duration_s:.1f} s at {sample_rate} Hz -> {destination}")

    prompt_set.write_manifest(args.out / "manifest.json", entries)
    print(f"\nwrote {args.out / 'manifest.json'} with {len(entries)} clip(s)")
    if not synthetic:
        print("At least one clip is a warm-up copy, so word error rates from this run are noise.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
