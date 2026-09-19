"""Convert a Hugging Face Qwen3-ASR model folder into the two GGUF files Spells needs.

Orchestrates four steps: fetching and caching llama.cpp's convert_hf_to_gguf.py plus its
gguf-py and conversion packages at a pinned tag, running that script twice (once for the
text decoder, once with --mmproj for the audio encoder), quantizing the text model with
llama-quantize, and starting llama-server briefly to prove both files load together. This
script carries no torch or transformers import of its own: the actual conversion runs in a
separate CPU-only Python interpreter (build/cache/convert/venv by default), so this file can
be exercised from any interpreter, including .venv, with the subprocess calls faked out.

Run it against a downloaded Hugging Face model directory:

    build/cache/convert/venv/Scripts/python.exe bench/convert_qwen3_asr.py \\
        --input-dir build/cache/convert/hf/Qwen3-ASR-0.6B \\
        --output-dir build/cache/convert/out/qwen3-asr-0.6b

--quant defaults to Q4_K_M for the text model; the mmproj is always written at Q8_0.
--name-prefix defaults to the input folder's own name. See
the Qwen3-ASR model card on Hugging Face for the conversion background, including
what the Albanian model owner needs to do on Hugging Face before this script has anything to
convert.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON_EXE = REPO_DIR / "build" / "cache" / "convert" / "venv" / "Scripts" / "python.exe"
DEFAULT_CACHE_DIR = REPO_DIR / "build" / "cache" / "convert"
DEFAULT_PINS_PATH = REPO_DIR / "build" / "pins.json"
DEFAULT_QUANTIZE_EXE = REPO_DIR / "build" / "cache" / "llama-tools" / "llama-quantize.exe"
DEFAULT_SERVER_EXE = REPO_DIR / "build" / "cache" / "llama-tools" / "llama-server.exe"
DEFAULT_QUANT = "Q4_K_M"
MMPROJ_QUANT = "Q8_0"
DEFAULT_THREADS = 6
DEFAULT_PORT = 8831
GITHUB_ARCHIVE_URL = "https://github.com/ggml-org/llama.cpp/archive/refs/tags/{tag}.tar.gz"
USER_AGENT = "spells-bench-convert-qwen3-asr/0.1 (+https://github.com/ggml-org/llama.cpp)"
CHUNK_SIZE = 1 << 20
HEALTH_TIMEOUT_S = 90
HEALTH_POLL_S = 0.5

REQUIRED_INPUT_FILES = ("config.json", "tokenizer_config.json")
VOCAB_ALTERNATIVES = (("tokenizer.json",), ("vocab.json", "merges.txt"))
WEIGHT_GLOBS = ("*.safetensors", "pytorch_model.bin", "pytorch_model.bin.index.json")

Runner = Callable[..., "subprocess.CompletedProcess[bytes]"]


class ConvertError(Exception):
    """A polite, user-facing refusal: bad input, missing tool, or a failed step."""


# --------------------------------------------------------------------------- input checks


def missing_input_files(input_dir: Path) -> list[str]:
    missing = [name for name in REQUIRED_INPUT_FILES if not (input_dir / name).is_file()]
    if not any(
        all((input_dir / name).is_file() for name in group) for group in VOCAB_ALTERNATIVES
    ):
        missing.append("tokenizer.json (or vocab.json and merges.txt)")
    if not any(any(input_dir.glob(pattern)) for pattern in WEIGHT_GLOBS):
        missing.append("model weights (*.safetensors or pytorch_model.bin)")
    return missing


def require_input_dir(input_dir: Path) -> None:
    if not input_dir.is_dir():
        raise ConvertError(f"{input_dir} is not a folder. Point --input-dir at a downloaded "
                            "Hugging Face model directory.")
    missing = missing_input_files(input_dir)
    if missing:
        names = ", ".join(missing)
        raise ConvertError(
            f"{input_dir} does not look like a Hugging Face model folder yet: it is missing "
            f"{names}. Download the repository's files into this folder and try again."
        )


# --------------------------------------------------------------------------- naming and sizes


def default_prefix_from_input_dir(input_dir: Path) -> str:
    name = input_dir.name.strip()
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name)
    name = name.strip("-")
    return name or "qwen3-asr"


def slugify_id(prefix: str, quant: str) -> str:
    combined = f"{prefix}-{quant}".lower()
    combined = re.sub(r"[^a-z0-9._-]+", "-", combined)
    combined = re.sub(r"-+", "-", combined).strip("-")
    return combined


def human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- fetching llama.cpp


def read_llama_cpp_tag(pins_path: Path) -> str:
    try:
        pins = json.loads(pins_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConvertError(f"could not read {pins_path}: {exc}") from exc
    try:
        return pins["binaries"]["llama_cpp_cpu"]["version"]
    except (KeyError, TypeError) as exc:
        raise ConvertError(
            f"{pins_path} has no binaries.llama_cpp_cpu.version entry to read the llama.cpp "
            "tag from"
        ) from exc


def llama_cpp_archive_url(tag: str) -> str:
    return GITHUB_ARCHIVE_URL.format(tag=tag)


def conversion_paths(cache_dir: Path, tag: str) -> tuple[Path, Path, Path]:
    root = cache_dir / f"llama.cpp-{tag}"
    return root / "convert_hf_to_gguf.py", root / "gguf-py", root / "conversion"


def conversion_ready(cache_dir: Path, tag: str) -> bool:
    convert_script, gguf_py, conversion_pkg = conversion_paths(cache_dir, tag)
    return (
        convert_script.is_file()
        and (gguf_py / "gguf" / "__init__.py").is_file()
        and (conversion_pkg / "__init__.py").is_file()
    )


def default_downloader(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, open(dest, "wb") as out:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                out.write(chunk)
    except urllib.error.URLError as exc:
        raise ConvertError(f"could not download {url}: {exc}") from exc


def extract_conversion_sources(archive_path: Path, tag: str, cache_dir: Path) -> None:
    root_prefix = f"llama.cpp-{tag}/"
    wanted_prefixes = (
        f"{root_prefix}convert_hf_to_gguf.py",
        f"{root_prefix}gguf-py/",
        f"{root_prefix}conversion/",
    )
    dest_root = cache_dir / f"llama.cpp-{tag}"
    dest_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.name.startswith(wanted_prefixes):
                continue
            relative = member.name[len(root_prefix):]
            if not relative:
                continue
            target = (dest_root / relative).resolve()
            if not str(target).startswith(str(dest_root.resolve())):
                raise ConvertError(f"refusing to extract {member.name} outside {dest_root}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, open(target, "wb") as out:
                out.write(source.read())


def fetch_llama_cpp(
    cache_dir: Path,
    tag: str,
    downloader: Callable[[str, Path], None] = default_downloader,
    extractor: Callable[[Path, str, Path], None] = extract_conversion_sources,
) -> Path:
    convert_script, _gguf_py, _conversion_pkg = conversion_paths(cache_dir, tag)
    if conversion_ready(cache_dir, tag):
        return convert_script
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / f"llama.cpp-{tag}.tar.gz"
    downloader(llama_cpp_archive_url(tag), archive_path)
    extractor(archive_path, tag, cache_dir)
    if not conversion_ready(cache_dir, tag):
        raise ConvertError(
            f"downloaded llama.cpp {tag} but convert_hf_to_gguf.py, gguf-py or conversion "
            "did not end up where expected"
        )
    return convert_script


# --------------------------------------------------------------------------- commands


def text_convert_command(
    python_exe: Path, convert_script: Path, input_dir: Path, outfile: Path
) -> list[str]:
    return [
        str(python_exe), str(convert_script), str(input_dir),
        "--outfile", str(outfile), "--outtype", "bf16",
    ]


def mmproj_convert_command(
    python_exe: Path, convert_script: Path, input_dir: Path, outfile: Path
) -> list[str]:
    return [
        str(python_exe), str(convert_script), str(input_dir), "--mmproj",
        "--outfile", str(outfile), "--outtype", "q8_0",
    ]


def quantize_command(
    quantize_exe: Path, src: Path, dst: Path, quant: str, threads: int
) -> list[str]:
    return [str(quantize_exe), str(src), str(dst), quant, str(threads)]


def server_command(
    server_exe: Path, text_gguf: Path, mmproj_gguf: Path, port: int, threads: int
) -> list[str]:
    return [
        str(server_exe),
        "-m", str(text_gguf),
        "--mmproj", str(mmproj_gguf),
        "--host", "127.0.0.1",
        "--port", str(port),
        "-t", str(threads),
        "-ngl", "0",
        "-c", "4096",
        "--no-webui",
    ]


def run_text_convert(
    python_exe: Path, convert_script: Path, input_dir: Path, outfile: Path,
    runner: Runner = subprocess.run,
) -> Path:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    cmd = text_convert_command(python_exe, convert_script, input_dir, outfile)
    runner(cmd, check=True)
    return outfile


def run_mmproj_convert(
    python_exe: Path, convert_script: Path, input_dir: Path, outfile: Path,
    runner: Runner = subprocess.run,
) -> Path:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    cmd = mmproj_convert_command(python_exe, convert_script, input_dir, outfile)
    runner(cmd, check=True)
    return outfile


def run_quantize(
    quantize_exe: Path, src: Path, dst: Path, quant: str, threads: int,
    runner: Runner = subprocess.run,
) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = quantize_command(quantize_exe, src, dst, quant, threads)
    runner(cmd, check=True)
    return dst


# --------------------------------------------------------------------------- health check


def wait_for_health(port: int, timeout_s: float = HEALTH_TIMEOUT_S) -> bool:
    deadline = time.perf_counter() + timeout_s
    url = f"http://127.0.0.1:{port}/health"
    while time.perf_counter() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(HEALTH_POLL_S)
    return False


def verify_loads(
    server_exe: Path,
    text_gguf: Path,
    mmproj_gguf: Path,
    port: int = DEFAULT_PORT,
    threads: int = DEFAULT_THREADS,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
    health_check: Callable[[int], bool] = wait_for_health,
) -> bool:
    cmd = server_command(server_exe, text_gguf, mmproj_gguf, port, threads)
    process = popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        return health_check(port)
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)


# --------------------------------------------------------------------------- catalog entry


def catalog_entry_fields(
    prefix: str, quant: str, text_file: str, mmproj_file: str, text_size: int, mmproj_size: int
) -> dict:
    return {
        "id": slugify_id(prefix, quant),
        "file": text_file,
        "extra_files": [mmproj_file],
        "size_bytes": text_size + mmproj_size,
        "runtime": "llama-asr",
        "engine_args": ["--mmproj", "{extra:0}", "--no-webui"],
    }


def render_catalog_entry(fields: dict) -> str:
    return json.dumps(fields, indent=2)


# --------------------------------------------------------------------------- CLI


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", required=True, type=Path,
                     help="a downloaded Hugging Face model directory")
    ap.add_argument("--output-dir", required=True, type=Path,
                     help="where to write the two GGUF files")
    ap.add_argument("--quant", default=DEFAULT_QUANT,
                     help=f"text model quantization, default {DEFAULT_QUANT}")
    ap.add_argument("--name-prefix", default=None,
                     help="file name prefix, default the input folder's own name")
    ap.add_argument("--python-exe", type=Path, default=DEFAULT_PYTHON_EXE,
                     help="the CPU-only venv with torch and transformers")
    ap.add_argument("--pins-path", type=Path, default=DEFAULT_PINS_PATH)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument("--llama-cpp-tag", default=None,
                     help="default: read from pins.json binaries.llama_cpp_cpu.version")
    ap.add_argument("--quantize-exe", type=Path, default=DEFAULT_QUANTIZE_EXE)
    ap.add_argument("--server-exe", type=Path, default=DEFAULT_SERVER_EXE)
    ap.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--skip-verify", action="store_true",
                     help="skip the llama-server health check")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        require_input_dir(args.input_dir)
    except ConvertError as exc:
        print(f"convert_qwen3_asr: {exc}", file=sys.stderr)
        return 2

    prefix = args.name_prefix or default_prefix_from_input_dir(args.input_dir)
    tag = args.llama_cpp_tag or read_llama_cpp_tag(args.pins_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"fetching llama.cpp {tag} conversion sources into {args.cache_dir}")
    convert_script = fetch_llama_cpp(args.cache_dir, tag)

    print(f"converting {prefix} text model")
    text_bf16 = run_text_convert(
        args.python_exe, convert_script, args.input_dir,
        args.output_dir / f"{prefix}-bf16.gguf",
    )
    text_gguf = run_quantize(
        args.quantize_exe, text_bf16, args.output_dir / f"{prefix}-{args.quant}.gguf",
        args.quant, args.threads,
    )

    print(f"converting {prefix} mmproj")
    mmproj_gguf = run_mmproj_convert(
        args.python_exe, convert_script, args.input_dir,
        args.output_dir / f"mmproj-{prefix}-{MMPROJ_QUANT}.gguf",
    )

    if not args.skip_verify:
        print("starting llama-server for a health check")
        healthy = verify_loads(
            args.server_exe, text_gguf, mmproj_gguf, args.port, args.threads
        )
        if not healthy:
            print(
                "convert_qwen3_asr: llama-server did not report healthy; the GGUF files were "
                "written but did not load cleanly",
                file=sys.stderr,
            )
            return 3
        print("llama-server loaded both files and reported healthy")

    text_size = text_gguf.stat().st_size
    mmproj_size = mmproj_gguf.stat().st_size
    text_sha = sha256_file(text_gguf)
    mmproj_sha = sha256_file(mmproj_gguf)

    print()
    print(f"{text_gguf.name}: {human_size(text_size)} ({text_size} bytes)")
    print(f"  sha256 {text_sha}")
    print(f"{mmproj_gguf.name}: {human_size(mmproj_size)} ({mmproj_size} bytes)")
    print(f"  sha256 {mmproj_sha}")

    fields = catalog_entry_fields(
        prefix, args.quant, text_gguf.name, mmproj_gguf.name, text_size, mmproj_size
    )
    print()
    print("catalog entry fields:")
    print(render_catalog_entry(fields))

    return 0


if __name__ == "__main__":
    sys.exit(main())
