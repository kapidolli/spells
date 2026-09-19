"""Convert a Hugging Face Whisper fine-tune to quantized ggml for whisper.cpp.

Orchestrates three steps that already exist elsewhere: a Hugging Face snapshot
download of the model's config and weight files, whisper.cpp's own
models/convert-h5-to-ggml.py (which needs the openai/whisper repo for the mel
filters and tokenizer assets), and whisper-quantize. This script only builds
the commands and runs them; it carries no torch or transformers import of its
own, so it can be exercised from the app's .venv with the actual subprocess
calls faked out.

Run it from the CPU-only conversion venv at build/cache/asr-sq/venv, since the
convert script needs torch and transformers, which never go into .venv:

    build/cache/asr-sq/venv/Scripts/python.exe bench/convert_whisper_hf.py \
        --python-exe build/cache/asr-sq/venv/Scripts/python.exe \
        --convert-script build/cache/whisper.cpp-1.9.4/models/convert-h5-to-ggml.py \
        --hf-dir build/cache/asr-sq/hf/<model> \
        --whisper-repo build/cache/asr-sq/whisper \
        --out-dir build/cache/asr-sq/ggml/<model> \
        --quantize-exe build/cache/whisper-build-cpu/bin/whisper-quantize.exe
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

Runner = Callable[..., "subprocess.CompletedProcess[bytes]"]

DOWNLOAD_ALLOW_PATTERNS = ["*.json", "*.txt", "*.safetensors", "vocab.json", "merges.txt"]


def convert_command(
    python_exe: Path, convert_script: Path, hf_dir: Path, whisper_repo: Path, out_dir: Path
) -> list[str]:
    return [str(python_exe), str(convert_script), str(hf_dir), str(whisper_repo), str(out_dir)]


def quantize_command(quantize_exe: Path, f16_bin: Path, q_bin: Path, qtype: str) -> list[str]:
    return [str(quantize_exe), str(f16_bin), str(q_bin), qtype]


def run_conversion(
    python_exe: Path,
    convert_script: Path,
    hf_dir: Path,
    whisper_repo: Path,
    out_dir: Path,
    runner: Runner = subprocess.run,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = convert_command(python_exe, convert_script, hf_dir, whisper_repo, out_dir)
    runner(cmd, check=True)
    return out_dir / "ggml-model.bin"


def run_quantize(
    quantize_exe: Path,
    f16_bin: Path,
    q_bin: Path,
    qtype: str = "q8_0",
    runner: Runner = subprocess.run,
) -> Path:
    q_bin.parent.mkdir(parents=True, exist_ok=True)
    cmd = quantize_command(quantize_exe, f16_bin, q_bin, qtype)
    runner(cmd, check=True)
    return q_bin


def convert_and_quantize(
    python_exe: Path,
    convert_script: Path,
    hf_dir: Path,
    whisper_repo: Path,
    out_dir: Path,
    quantize_exe: Path,
    qtype: str = "q8_0",
    runner: Runner = subprocess.run,
) -> Path:
    f16_bin = run_conversion(python_exe, convert_script, hf_dir, whisper_repo, out_dir, runner)
    q_bin = out_dir / f"ggml-model-{qtype}.bin"
    return run_quantize(quantize_exe, f16_bin, q_bin, qtype, runner)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--python-exe", required=True, type=Path)
    ap.add_argument("--convert-script", required=True, type=Path)
    ap.add_argument("--hf-dir", required=True, type=Path)
    ap.add_argument("--whisper-repo", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--quantize-exe", required=True, type=Path)
    ap.add_argument("--qtype", default="q8_0")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    a = parse_args(argv)
    q_bin = convert_and_quantize(
        a.python_exe,
        a.convert_script,
        a.hf_dir,
        a.whisper_repo,
        a.out_dir,
        a.quantize_exe,
        a.qtype,
    )
    print(q_bin)
    return 0


if __name__ == "__main__":
    sys.exit(main())
