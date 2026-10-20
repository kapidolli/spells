"""Make a development checkout runnable on Linux: engines and models, stdlib only.

Usage:
    python3 build/linux_engines.py [--engines] [--models] [--jobs N]

No flags means both. Every step is idempotent and skips what already exists and verifies.

Engines, into build/out/engines/<variant>/ for variant vulkan and cpu:

1. llama-server from the official llama.cpp Ubuntu x64 release of the same tag the Windows
   build pins (binaries.llama_cpp_linux_vulkan and binaries.llama_cpp_linux_cpu in
   build/pins.json; the entries are added with the hash of the first download). Only
   llama-server, the shared libraries it loads and the licence are kept, as in
   build_whisper.is_shipped_llama_file. The release binaries resolve their libraries through
   RPATH $ORIGIN, so they sit next to llama-server.
2. whisper-server built from the pinned whisper.cpp source (sources.whisper_cpp) with cmake and
   ninja into build/cache/whisper-linux-<variant>/. Linked dynamically: glibc and the Vulkan
   loader come from the system.

Models, into build/cache/models/: the files of the default English, German and Albanian
install that the project's models-v1 GitHub release publishes, verified against that
release's SHA256SUMS.txt, plus models.cleanup_model from build/pins.json.

Finally a smoke test runs llama-server --list-devices and whisper-server --help.
"""

from __future__ import annotations

import argparse
import glob
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch

REPO_DIR = Path(__file__).resolve().parents[1]
BUILD_DIR = REPO_DIR / "build"
CACHE_DIR = BUILD_DIR / "cache"
MODELS_DIR = CACHE_DIR / "models"
OUT_DIR = BUILD_DIR / "out"

VARIANTS = ("vulkan", "cpu")
LLAMA_VERSION = "b10997"
LLAMA_BASE_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_VERSION}"
LLAMA_ARCHIVES = {
    "vulkan": ("binaries.llama_cpp_linux_vulkan", f"llama-{LLAMA_VERSION}-bin-ubuntu-vulkan-x64.tar.gz"),
    "cpu": ("binaries.llama_cpp_linux_cpu", f"llama-{LLAMA_VERSION}-bin-ubuntu-x64.tar.gz"),
}
LLAMA_NOTES = {
    "vulkan": "llama.cpp Linux x64 Vulkan release archive providing llama-server for development "
              "checkouts (build/linux_engines.py). Same layout as the Windows zip: a llama-server "
              "stub next to libllama-server-impl.so, libllama*.so, libggml*.so (RPATH $ORIGIN) in "
              "one wrapping folder. Hash computed from the first download.",
    "cpu": "llama.cpp Linux x64 CPU release archive, the fallback llama-server for development "
           "checkouts (build/linux_engines.py). Same layout as the Vulkan archive minus "
           "libggml-vulkan.so. Hash computed from the first download.",
}
LLAMA_KEEP_BIN = "llama-server"
LLAMA_KEEP_IMPL = "libllama-server-impl.so"
LLAMA_KEEP_LIBS = ("libllama.so", "libllama-common.so", "libmtmd.so")

WHISPER_KEY = "sources.whisper_cpp"
SPIRV_KEY = "sources.spirv_headers"
SPIRV_CONFIG_GLOBS = ("/usr/share/cmake*/SPIRV-Headers*/*Config.cmake", "/usr/lib*/cmake/SPIRV-Headers*/*Config.cmake",
                      "/usr/share/cmake*/spirv-headers*/*onfig.cmake")
WHISPER_TARGET = "whisper-server"
COMMON_CMAKE_ARGS = [
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_SHARED_LIBS=OFF",
    "-DWHISPER_BUILD_EXAMPLES=ON",
    "-DWHISPER_BUILD_SERVER=ON",
    "-DWHISPER_BUILD_TESTS=OFF",
    "-DWHISPER_SDL2=OFF",
    "-DWHISPER_CURL=OFF",
    "-DGGML_NATIVE=OFF",
]
VARIANT_CMAKE_ARGS = {
    "vulkan": ["-DGGML_VULKAN=ON"],
    "cpu": ["-DGGML_VULKAN=OFF"],
}
VULKAN_LOADER_DIR = Path("/usr/lib64")
SYSTEM_LIB_PREFIXES = (
    "linux-vdso", "ld-linux", "libc.so", "libm.so", "libdl.so", "libpthread.so", "librt.so",
    "libstdc++.so", "libgcc_s.so", "libgomp.so", "libvulkan.so", "libutil.so",
)

RELEASE_URL = "https://github.com/kapidolli/spells/releases/download/models-v1"
RELEASE_MODELS = (
    "Qwen3-ASR-0.6B-Q4_K_M.gguf",
    "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf",
    "ggml-large-v3-turbo-q8_0.bin",
    "ggml-silero-v5.1.2.bin",
    "mtp-gemma-4-E2B-it-Q4_0.gguf",
)
PINNED_MODELS = ("models.cleanup_model",)


class SetupError(Exception):
    pass


def is_shipped_llama_file(name: str) -> bool:
    """True for a llama.cpp archive member the app needs at runtime, by its name in the folder."""
    if not name or Path(name).name != name:
        return False
    if name.startswith("LICENSE"):
        return True
    if name == LLAMA_KEEP_BIN or name == LLAMA_KEEP_IMPL:
        return True
    if name.endswith("-impl.so"):
        return False
    if not name.startswith("lib") or ".so" not in name:
        return False
    return name.startswith("libggml") or name.split(".so")[0] + ".so" in LLAMA_KEEP_LIBS


def strip_prefix(names: list[str]) -> str:
    """The single wrapping folder to remove from member names, or "" when the archive is flat."""
    roots = {name.split("/", 1)[0] for name in names}
    if len(roots) == 1 and all("/" in name for name in names):
        return next(iter(roots)) + "/"
    return ""


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    if subprocess.run(cmd, cwd=cwd, check=False).returncode != 0:
        raise SetupError(f"command failed: {cmd[0]}")


def capture(cmd: list[str], *, timeout: float = 60) -> str:
    done = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, errors="replace", check=False
    )
    return done.stdout + done.stderr


def require_tools(*names: str) -> None:
    missing = [name for name in names if shutil.which(name) is None]
    if missing:
        raise SetupError(f"missing tools: {', '.join(missing)}")


def ensure_llama_pin(variant: str) -> str:
    """Add the pin entry on first use (hash from the download), return its key."""
    key, filename = LLAMA_ARCHIVES[variant]
    pins = fetch.load_pins()
    if key.partition(".")[2] in pins["binaries"]:
        return key
    entry = {"url": f"{LLAMA_BASE_URL}/{filename}", "version": LLAMA_VERSION, "sha256": "",
             "notes": LLAMA_NOTES[variant]}
    dest = CACHE_DIR / filename
    if not dest.exists():
        print(f"{key}: downloading {entry['url']}")
        fetch.download(entry["url"], dest)
    entry["sha256"] = fetch.sha256_file(dest)
    pins["binaries"][key.partition(".")[2]] = entry
    fetch.save_pins(pins)
    print(f"{key}: pinned sha256 {entry['sha256']}")
    return key


def stage_llama_cpp(variant: str) -> None:
    key = ensure_llama_pin(variant)
    archive = fetch.fetch(key)
    dest_dir = OUT_DIR / "engines" / variant
    if (dest_dir / LLAMA_KEEP_BIN).exists() and (dest_dir / LLAMA_KEEP_IMPL).exists() \
            and (dest_dir / ".llama-sha256").exists() \
            and (dest_dir / ".llama-sha256").read_text().strip() == fetch.sha256_file(archive):
        print(f"llama.cpp ({variant}): already staged in {dest_dir}")
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    kept = 0
    with tarfile.open(archive, "r:gz") as tar:
        members = [m for m in tar.getmembers() if not m.isdir()]
        strip = strip_prefix([m.name for m in members])
        for member in members:
            member.name = member.name[len(strip):]
            if not is_shipped_llama_file(member.name):
                continue
            (dest_dir / member.name).unlink(missing_ok=True)
            tar.extract(member, dest_dir, filter="fully_trusted")
            kept += 1
    if not (dest_dir / LLAMA_KEEP_BIN).exists():
        raise SetupError(f"{archive.name} did not contain {LLAMA_KEEP_BIN}")
    (dest_dir / ".llama-sha256").write_text(fetch.sha256_file(archive) + "\n")
    print(f"llama.cpp ({variant}): staged {kept} files into {dest_dir}")


def ensure_whisper_source() -> Path:
    archive = fetch.fetch(WHISPER_KEY)
    version = fetch.resolve(fetch.load_pins(), WHISPER_KEY)["version"].lstrip("v")
    source_dir = CACHE_DIR / f"whisper.cpp-{version}"
    if (source_dir / "CMakeLists.txt").exists():
        return source_dir
    print(f"source: extracting {archive.name} into {CACHE_DIR}")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(CACHE_DIR, filter="data")
    if not (source_dir / "CMakeLists.txt").exists():
        raise SetupError(f"extraction did not produce {source_dir / 'CMakeLists.txt'}")
    return source_dir


def library_names(exe: Path) -> list[str]:
    names = []
    for line in capture(["ldd", str(exe)]).splitlines():
        line = line.strip()
        if line:
            names.append(line.split()[0].rsplit("/", 1)[-1])
    return names


def verify_dynamic_libs(exe: Path) -> None:
    names = library_names(exe)
    print(f"libraries of {exe.parent.name}/{exe.name}: {', '.join(names)}")
    bad = [name for name in names if not name.startswith(SYSTEM_LIB_PREFIXES)]
    if bad:
        raise SetupError(f"{exe} links libraries it may not: {', '.join(bad)}")


def spirv_headers_args() -> list[str]:
    """Install the pinned SPIRV-Headers into build/cache when the system has no CMake package."""
    if any(glob.glob(pattern) for pattern in SPIRV_CONFIG_GLOBS):
        return []
    prefix = CACHE_DIR / "spirv-headers-prefix"
    if not any(prefix.glob("**/SPIRV-Headers*Config.cmake")):
        archive = fetch.fetch(SPIRV_KEY)
        source_dir = CACHE_DIR / archive.name.removesuffix(".tar.gz")
        with tarfile.open(archive, "r:gz") as tar:
            top = tar.getnames()[0].split("/", 1)[0]
            tar.extractall(CACHE_DIR, filter="data")
        shutil.rmtree(source_dir, ignore_errors=True)
        (CACHE_DIR / top).rename(source_dir)
        build_dir = CACHE_DIR / "spirv-headers-build"
        run(["cmake", "-S", str(source_dir), "-B", str(build_dir), "-G", "Ninja",
             f"-DCMAKE_INSTALL_PREFIX={prefix}", "-DSPIRV_HEADERS_ENABLE_TESTS=OFF"])
        run(["cmake", "--install", str(build_dir)])
    return [f"-DCMAKE_PREFIX_PATH={prefix}", f"-DCMAKE_CXX_FLAGS=-I{prefix / 'include'}"]


def vulkan_loader_args() -> list[str]:
    """Point CMake at libvulkan.so.1 when the vulkan-loader-devel symlink libvulkan.so is absent."""
    for directory in (VULKAN_LOADER_DIR, Path("/usr/lib/x86_64-linux-gnu")):
        if (directory / "libvulkan.so").exists():
            return []
        if (directory / "libvulkan.so.1").exists():
            return [f"-DVulkan_LIBRARY={directory / 'libvulkan.so.1'}"]
    return []


def build_whisper(variant: str, source_dir: Path, jobs: int | None) -> None:
    dest = OUT_DIR / "engines" / variant / WHISPER_TARGET
    build_dir = CACHE_DIR / f"whisper-linux-{variant}"
    built = build_dir / "bin" / WHISPER_TARGET
    if not built.exists():
        run(["cmake", "-S", str(source_dir), "-B", str(build_dir),
             *COMMON_CMAKE_ARGS, *VARIANT_CMAKE_ARGS[variant],
             *(vulkan_loader_args() + spirv_headers_args() if variant == "vulkan" else [])])
        cmd = ["ninja", "-C", str(build_dir)]
        if jobs:
            cmd += ["-j", str(jobs)]
        run(cmd + [WHISPER_TARGET])
    if not built.exists():
        raise SetupError(f"{built} was not produced")
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = dest.with_name(dest.name + ".new")
    shutil.copy2(built, staging)
    staging.replace(dest)
    verify_dynamic_libs(dest)
    print(f"whisper-server ({variant}): staged {dest} ({dest.stat().st_size:,} bytes)")


def setup_engines(jobs: int | None) -> None:
    require_tools("cmake", "ninja", "g++", "ldd")
    source_dir = ensure_whisper_source()
    for variant in VARIANTS:
        print(f"\n===== {variant} =====")
        stage_llama_cpp(variant)
        build_whisper(variant, source_dir, jobs)


def release_checksums() -> dict[str, str]:
    request = urllib.request.Request(f"{RELEASE_URL}/SHA256SUMS.txt", headers={"User-Agent": fetch.USER_AGENT})
    with urllib.request.urlopen(request) as response:
        text = response.read().decode("utf-8")
    sums = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            sums[parts[1].lstrip("*")] = parts[0].lower()
    return sums


def fetch_release_model(name: str, sums: dict[str, str]) -> Path:
    expected = sums.get(name)
    if not expected:
        raise SetupError(f"{name} is not listed in the release SHA256SUMS.txt")
    dest = MODELS_DIR / name
    if dest.exists():
        if fetch.sha256_file(dest) == expected:
            print(f"{name}: present (sha256 ok)")
            return dest
        print(f"{name}: hash differs, downloading again")
        dest.unlink()
    print(f"{name}: downloading")
    actual = fetch.download(f"{RELEASE_URL}/{name}", dest)
    if actual != expected:
        dest.unlink(missing_ok=True)
        raise SetupError(f"{name}: sha256 mismatch (expected {expected}, got {actual})")
    print(f"{name}: sha256 ok")
    return dest


def setup_models() -> None:
    sums = release_checksums()
    for name in RELEASE_MODELS:
        fetch_release_model(name, sums)
    for key in PINNED_MODELS:
        fetch.fetch(key)


def smoke_test() -> None:
    print("\n===== smoke test =====")
    for variant in VARIANTS:
        engines = OUT_DIR / "engines" / variant
        print(f"-- {variant}/llama-server --list-devices")
        print(capture([str(engines / LLAMA_KEEP_BIN), "--list-devices"]).strip())
        print(f"-- {variant}/whisper-server --help (first lines)")
        print("\n".join(capture([str(engines / WHISPER_TARGET), "--help"]).splitlines()[:6]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--engines", action="store_true", help="only the llama-server and whisper-server engines")
    parser.add_argument("--models", action="store_true", help="only the models")
    parser.add_argument("--jobs", type=int, default=None, help="ninja parallelism (default: ninja's)")
    args = parser.parse_args(argv)
    both = not (args.engines or args.models)

    try:
        if args.engines or both:
            setup_engines(args.jobs)
        if args.models or both:
            setup_models()
        if args.engines or both:
            smoke_test()
    except (SetupError, fetch.PinError, OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
