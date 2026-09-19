"""Build whisper-server from the pinned whisper.cpp tag with the MSYS2 UCRT64 toolchain.

Usage:
    py build/build_whisper.py [--variant vulkan|cpu|all] [--no-llama] [--jobs N]

Spec 19.2 and 19.3 step 3. For each variant the script:

1. extracts build/cache/whisper.cpp-<ver>.tar.gz (fetched and hash-checked via fetch.py)
   into build/cache/whisper.cpp-<ver>/,
2. configures with cmake + ninja (Release, BUILD_SHARED_LIBS=OFF, static runtime link) into
   build/cache/whisper-build-<variant>/,
3. for the vulkan variant parses the configure output and fails unless glslc reports the
   cooperative matrix, cooperative matrix 2, and integer dot product extensions as supported,
4. builds only the whisper-server target,
5. checks with objdump that the exe imports nothing but Windows system DLLs,
6. copies it to build/out/engines/<variant>/whisper-server.exe,
7. unless --no-llama: extracts the pinned llama.cpp Windows release zip for the same variant
   next to it (llama-server.exe plus its DLLs, as shipped upstream).

Steps that launch toolchain programs are retried when Windows code integrity (Smart App
Control) blocks a file at first sight; see bootstrap_toolchain.run_tool for the details.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch
from bootstrap_toolchain import (
    POLICY_ATTEMPTS,
    POLICY_RETRY_DELAY_S,
    clean_windows_env,
    cmake_exe,
    is_policy_block,
    policy_blocks_since,
)

REPO_DIR = Path(__file__).resolve().parents[1]
BUILD_DIR = REPO_DIR / "build"
CACHE_DIR = BUILD_DIR / "cache"
OUT_DIR = BUILD_DIR / "out"
TOOLCHAIN_DIR = BUILD_DIR / "toolchain" / "msys64"
UCRT_BIN = TOOLCHAIN_DIR / "ucrt64" / "bin"

SOURCE_KEY = "sources.whisper_cpp"
VARIANTS = ("vulkan", "cpu")
LLAMA_KEYS = {"vulkan": "binaries.llama_cpp_vulkan", "cpu": "binaries.llama_cpp_cpu"}
TARGET = "whisper-server"

# Option names verified in whisper.cpp v1.9.4: CMakeLists.txt (WHISPER_*, BUILD_SHARED_LIBS),
# ggml/CMakeLists.txt (GGML_VULKAN, GGML_NATIVE). GGML_NATIVE=OFF keeps the CPU code at the
# AVX2/FMA/F16C baseline that ggml enables by default for non-native x86 builds, so the exe
# runs on machines other than the build host. -static makes gcc link libgcc, libstdc++,
# libwinpthread and libgomp statically; the result imports only Windows DLLs.
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
    "-DCMAKE_EXE_LINKER_FLAGS=-static",
]
VARIANT_CMAKE_ARGS = {
    "vulkan": ["-DGGML_VULKAN=ON"],
    "cpu": ["-DGGML_VULKAN=OFF"],
}

# Exactly the STATUS lines printed by ggml/src/ggml-vulkan/CMakeLists.txt
# (test_shader_extension_support): "<ext> supported by glslc" / "<ext> not supported by glslc".
VULKAN_FEATURES = {
    "coopmat": "GL_KHR_cooperative_matrix",
    "coopmat2": "GL_NV_cooperative_matrix2",
    "integer_dot": "GL_EXT_integer_dot_product",
}
FEATURE_LINE_RE = re.compile(
    r"^-- (?P<ext>GL_[A-Za-z0-9_]+) (?P<verdict>not supported|supported) by glslc\s*$",
    re.MULTILINE,
)

# DLLs that every Windows 10/11 installation provides. Anything else (libstdc++-6.dll,
# libgcc_s_seh-1.dll, libwinpthread-1.dll, libgomp-1.dll, ...) would tie the exe to MSYS2.
# vulkan-1.dll is deliberately not in this set: the Vulkan loader is installed by the GPU
# driver, so it is a system DLL for the vulkan variant and a build mistake for the cpu one.
SYSTEM_DLLS = {
    "advapi32.dll", "bcrypt.dll", "crypt32.dll", "dbghelp.dll", "gdi32.dll", "kernel32.dll",
    "msvcrt.dll", "ntdll.dll", "ole32.dll", "oleaut32.dll", "rpcrt4.dll", "shell32.dll",
    "shlwapi.dll", "ucrtbase.dll", "user32.dll", "userenv.dll", "version.dll",
    "winmm.dll", "ws2_32.dll", "wsock32.dll", "iphlpapi.dll", "psapi.dll", "secur32.dll",
    "setupapi.dll", "cfgmgr32.dll", "imm32.dll", "comdlg32.dll", "powrprof.dll",
}
SYSTEM_DLL_PREFIXES = ("api-ms-win-", "ext-ms-win-")
VULKAN_LOADER_DLL = "vulkan-1.dll"

# Files from the llama.cpp release zip that the app ships: llama-server.exe, the libraries it
# loads (its own -impl DLL, the ggml backends, llama, llama-common, mtmd, libomp) and the
# licence texts. The zip also carries a dozen other tools with their own -impl DLLs, which are
# never launched and cost about 5 MB per variant against the spec 19.4 payload budget.
LLAMA_KEEP_EXE = "llama-server.exe"
LLAMA_KEEP_IMPL = "llama-server-impl.dll"
LLAMA_KEEP_DLLS = ("llama.dll", "llama-common.dll", "mtmd.dll", "libomp.dll")
DLL_NAME_RE = re.compile(r"^\s*DLL Name:\s*(\S+)\s*$", re.MULTILINE)


class BuildError(Exception):
    pass


# ----------------------------------------------------------------------------- pure helpers


def parse_vulkan_feature_checks(configure_output: str) -> dict[str, bool | None]:
    """Map coopmat / coopmat2 / integer_dot to True, False, or None (line absent)."""
    verdicts = {}
    for match in FEATURE_LINE_RE.finditer(configure_output):
        verdicts[match.group("ext")] = match.group("verdict") == "supported"
    return {feature: verdicts.get(ext) for feature, ext in VULKAN_FEATURES.items()}


def missing_vulkan_features(results: dict[str, bool | None]) -> list[str]:
    return [feature for feature in VULKAN_FEATURES if results.get(feature) is not True]


def parse_dll_names(objdump_output: str) -> list[str]:
    return DLL_NAME_RE.findall(objdump_output)


def is_system_dll(name: str, *, allow_vulkan: bool = False) -> bool:
    lowered = name.lower()
    if lowered == VULKAN_LOADER_DLL:
        return allow_vulkan
    return lowered in SYSTEM_DLLS or lowered.startswith(SYSTEM_DLL_PREFIXES)


def foreign_dlls(names: list[str], *, allow_vulkan: bool = False) -> list[str]:
    return [name for name in names if not is_system_dll(name, allow_vulkan=allow_vulkan)]


def is_shipped_llama_file(name: str) -> bool:
    """True for a llama.cpp zip member the app needs at runtime, by its name in the zip root."""
    if not name or Path(name).name != name:
        return False
    lowered = name.lower()
    if lowered.startswith("license"):
        return True
    if lowered.endswith(".exe"):
        return lowered == LLAMA_KEEP_EXE
    if not lowered.endswith(".dll"):
        return False
    if lowered.endswith("-impl.dll"):
        return lowered == LLAMA_KEEP_IMPL
    return lowered.startswith("ggml") or lowered in LLAMA_KEEP_DLLS


# ----------------------------------------------------------------------------- toolchain


def toolchain_env() -> dict[str, str]:
    """Environment for running the UCRT64 tools directly: only the toolchain and Windows on PATH."""
    env = clean_windows_env([UCRT_BIN, cmake_exe().parent])
    env["MSYSTEM"] = "UCRT64"
    return env


def require_tool(name: str) -> Path:
    exe = cmake_exe() if name == "cmake" else UCRT_BIN / f"{name}.exe"
    if not exe.exists():
        raise BuildError(f"{exe} is missing; run 'py build/bootstrap_toolchain.py' first")
    return exe


def _run_once(cmd: list[str], cwd: Path, env: dict, log: Path | None, capture: bool) -> tuple[int, str]:
    collected = []
    log_target = log if log is not None else Path(os.devnull)
    with open(log_target, "a", encoding="utf-8") as log_fh, \
            subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                             errors="replace", bufsize=1) as proc:
        log_fh.write("+ " + " ".join(cmd) + "\n")
        for line in proc.stdout:
            collected.append(line)
            log_fh.write(line)
            if not capture:
                sys.stdout.write(line)
                sys.stdout.flush()
        rc = proc.wait()
    return rc, "".join(collected)


def run(cmd: list[str], *, cwd: Path, env: dict, log: Path | None = None, capture: bool = False) -> str:
    """Run a command, streaming output to the console (and a log file); return the output.

    A failure caused by a code integrity first-sight block is retried (see bootstrap_toolchain).
    """
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    blocked: list[str] = []
    for attempt in range(1, POLICY_ATTEMPTS + 1):
        started = time.time()
        try:
            rc, output = _run_once(cmd, cwd, env, log, capture)
        except OSError as exc:
            if not is_policy_block(exc):
                raise
            rc, output = -1, str(exc)
        if rc == 0:
            return output
        blocked = policy_blocks_since(started) or ([cmd[0]] if rc == -1 else [])
        if not blocked:
            if capture:
                sys.stdout.write(output)
            raise BuildError(f"command failed with exit code {rc}: {cmd[0]}")
        print(f"code integrity blocked {', '.join(blocked)} during attempt {attempt}/{POLICY_ATTEMPTS}; "
              f"retrying once the verdict is cached", flush=True)
        if attempt < POLICY_ATTEMPTS:
            time.sleep(POLICY_RETRY_DELAY_S)
    raise BuildError(f"code integrity keeps blocking {', '.join(blocked)}; the build cannot continue "
                     f"on this machine without a signed tool or a policy change")


# ----------------------------------------------------------------------------- steps


def ensure_source() -> Path:
    archive = fetch.fetch(SOURCE_KEY)
    pins = fetch.load_pins()
    version = fetch.resolve(pins, SOURCE_KEY)["version"].lstrip("v")
    source_dir = CACHE_DIR / f"whisper.cpp-{version}"
    if (source_dir / "CMakeLists.txt").exists():
        print(f"source: {source_dir} already extracted")
        return source_dir
    print(f"source: extracting {archive.name} into {CACHE_DIR}")
    with tarfile.open(archive, "r:gz") as tar:
        top_levels = {member.name.split("/", 1)[0] for member in tar.getmembers()}
        if top_levels != {source_dir.name}:
            raise BuildError(f"unexpected archive layout {sorted(top_levels)}; expected {source_dir.name}/")
        tar.extractall(CACHE_DIR, filter="data")
    if not (source_dir / "CMakeLists.txt").exists():
        raise BuildError(f"extraction did not produce {source_dir / 'CMakeLists.txt'}")
    return source_dir


def prefix_path_args() -> list[str]:
    """Point CMake at the MSYS2 UCRT64 prefix.

    The Kitware cmake runs outside the MSYS2 shell, so it has no idea that ucrt64 is a
    sysroot. Without this, FindVulkan reports "Could NOT find Vulkan (missing:
    Vulkan_LIBRARY Vulkan_INCLUDE_DIR)" and find_package(SPIRV-Headers CONFIG) fails,
    even though the vulkan-devel and spirv-headers packages are installed. CMAKE_PREFIX_PATH
    adds <prefix>/include, /lib, /bin and /share/cmake to every find_ call.
    """
    return [f"-DCMAKE_PREFIX_PATH={UCRT_BIN.parent.as_posix()}"]


def openmp_args() -> list[str]:
    """Link OpenMP statically: FindOpenMP picks libgomp.dll.a (an import library) by full path,
    which -static cannot override, so the exe would import libgomp-1.dll. Point CMake at the
    static archive instead; without one, drop OpenMP and use ggml's own thread pool."""
    static_gomp = UCRT_BIN.parent / "lib" / "libgomp.a"
    if static_gomp.exists():
        return [f"-DOpenMP_gomp_LIBRARY={static_gomp.as_posix()}"]
    print("warning: no static libgomp.a in the toolchain; building with GGML_OPENMP=OFF")
    return ["-DGGML_OPENMP=OFF"]


def discard_stale_cache(source_dir: Path, build_dir: Path) -> None:
    """Delete a build directory whose CMakeCache points at another source tree.

    CMake refuses to reuse such a cache, and it happens whenever the repository is moved or
    renamed, so the build directory is removed rather than reported.
    """
    cache = build_dir / "CMakeCache.txt"
    if not cache.exists():
        return
    match = re.search(r"^CMAKE_HOME_DIRECTORY:INTERNAL=(.*)$", cache.read_text(
        encoding="utf-8", errors="replace"), re.MULTILINE)
    recorded = match.group(1).strip() if match else ""
    if recorded and Path(recorded) != source_dir:
        print(f"stale cache in {build_dir} (configured for {recorded}); removing it")
        shutil.rmtree(build_dir)


def configure(variant: str, source_dir: Path, build_dir: Path, env: dict) -> str:
    discard_stale_cache(source_dir, build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    cmake = require_tool("cmake")
    ninja = require_tool("ninja")
    gcc = require_tool("gcc")
    gxx = require_tool("g++")
    cmd = [str(cmake), "-S", str(source_dir), "-B", str(build_dir),
           f"-DCMAKE_MAKE_PROGRAM={ninja}",
           f"-DCMAKE_C_COMPILER={gcc}",
           f"-DCMAKE_CXX_COMPILER={gxx}",
           *COMMON_CMAKE_ARGS, *prefix_path_args(), *openmp_args(),
           *VARIANT_CMAKE_ARGS[variant]]
    output = run(cmd, cwd=build_dir, env=env, log=build_dir / "configure.log", capture=True)
    if variant == "vulkan":
        results = parse_vulkan_feature_checks(output)
        print("vulkan feature checks (glslc):")
        for feature, ext in VULKAN_FEATURES.items():
            state = {True: "supported", False: "NOT supported", None: "line not found"}[results[feature]]
            print(f"  {feature:12s} {ext:40s} {state}")
        missing = missing_vulkan_features(results)
        if missing:
            raise BuildError(
                f"vulkan configure did not report these features as supported: {', '.join(missing)}. "
                f"Full output is in {build_dir / 'configure.log'}"
            )
    elif "Vulkan found" in output:
        raise BuildError("cpu variant configured with Vulkan; check GGML_VULKAN=OFF")
    return output


def build(build_dir: Path, env: dict, jobs: int | None) -> Path:
    ninja = require_tool("ninja")
    cmd = [str(ninja), "-C", str(build_dir)]
    if jobs:
        cmd += ["-j", str(jobs)]
    cmd.append(TARGET)
    started = time.monotonic()
    run(cmd, cwd=build_dir, env=env, log=build_dir / "build.log")
    print(f"build finished in {time.monotonic() - started:.0f} s")
    exe = build_dir / "bin" / f"{TARGET}.exe"
    if not exe.exists():
        raise BuildError(f"{exe} was not produced")
    return exe


def verify_imports(exe: Path, env: dict, variant: str) -> list[str]:
    objdump = require_tool("objdump")
    output = run([str(objdump), "-p", str(exe)], cwd=exe.parent, env=env, capture=True)
    names = parse_dll_names(output)
    print(f"imports of {exe.name}: {', '.join(names)}")
    bad = foreign_dlls(names, allow_vulkan=variant == "vulkan")
    if bad:
        raise BuildError(f"{exe} ({variant}) imports DLLs it may not: {', '.join(bad)}")
    return names


def stage(variant: str, exe: Path) -> Path:
    dest_dir = OUT_DIR / "engines" / variant
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / exe.name
    shutil.copy2(exe, dest)
    size = dest.stat().st_size
    print(f"staged {dest} ({size:,} bytes, {size / (1 << 20):.1f} MiB)")
    return dest


def strip_prefix(names: list[str]) -> str:
    """The single wrapping folder to remove from zip member names, or "" when the zip is flat."""
    roots = {name.split("/", 1)[0] for name in names}
    if len(roots) == 1 and all("/" in name for name in names):
        return next(iter(roots)) + "/"
    return ""


def llama_member_names(variant: str) -> list[str]:
    """Every file in the pinned llama.cpp zip for this variant, relative to its root."""
    archive = fetch.fetch(LLAMA_KEYS[variant])
    with zipfile.ZipFile(archive) as zf:
        names = [m.filename for m in zf.infolist() if not m.is_dir()]
    strip = strip_prefix(names)
    return [name[len(strip):] for name in names]


def stage_llama_cpp(variant: str) -> list[str]:
    """Extract the shipped part of the pinned llama.cpp zip into build/out/engines/<variant>/."""
    archive = fetch.fetch(LLAMA_KEYS[variant])
    dest_dir = OUT_DIR / "engines" / variant
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    skipped = 0
    with zipfile.ZipFile(archive) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        strip = strip_prefix([m.filename for m in members])
        for member in members:
            relative = member.filename[len(strip):]
            if not is_shipped_llama_file(relative):
                skipped += 1
                continue
            target = dest_dir / relative
            with zf.open(member) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            extracted.append(relative)
    if not (dest_dir / LLAMA_KEEP_EXE).exists():
        raise BuildError(f"{archive.name} did not contain {LLAMA_KEEP_EXE} at its root")
    print(f"staged llama.cpp ({variant}): {len(extracted)} files into {dest_dir}, "
          f"{skipped} tool files left in the archive")
    for name in sorted(extracted):
        print(f"  {name} ({(dest_dir / name).stat().st_size:,} bytes)")
    return extracted


def prune_staged_llama(variant: str) -> list[str]:
    """Delete llama.cpp tool files from an existing build/out/engines/<variant>/.

    Only files the pinned zip carries and is_shipped_llama_file rejects are removed, so
    whisper-server.exe, llama-server.exe and every library DLL are left exactly as they are.
    A llama-server.exe another process is running is never touched.
    """
    dest_dir = OUT_DIR / "engines" / variant
    if not dest_dir.is_dir():
        print(f"prune ({variant}): {dest_dir} does not exist")
        return []
    removed = []
    for name in sorted(llama_member_names(variant)):
        if is_shipped_llama_file(name):
            continue
        path = dest_dir / name
        if path.is_file():
            path.unlink()
            removed.append(name)
    total = sum(p.stat().st_size for p in dest_dir.rglob("*") if p.is_file())
    print(f"prune ({variant}): removed {len(removed)} tool files; {dest_dir} is now "
          f"{total:,} bytes ({total / (1 << 20):.1f} MiB)")
    for name in removed:
        print(f"  removed {name}")
    return removed


def build_variant(variant: str, source_dir: Path, jobs: int | None, with_llama: bool) -> Path:
    env = toolchain_env()
    build_dir = CACHE_DIR / f"whisper-build-{variant}"
    print(f"\n===== {variant} =====")
    configure(variant, source_dir, build_dir, env)
    exe = build(build_dir, env, jobs)
    verify_imports(exe, env, variant)
    staged = stage(variant, exe)
    if with_llama:
        stage_llama_cpp(variant)
    return staged


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--variant", choices=(*VARIANTS, "all"), default="all")
    parser.add_argument("--jobs", type=int, default=None, help="ninja parallelism (default: ninja's)")
    parser.add_argument("--no-llama", action="store_true",
                        help="do not extract the llama.cpp release zips next to whisper-server")
    parser.add_argument("--prune-only", action="store_true",
                        help="build nothing; only delete the llama.cpp tool files already "
                             "staged in build/out/engines/<variant>/")
    args = parser.parse_args(argv)
    variants = VARIANTS if args.variant == "all" else (args.variant,)

    if args.prune_only:
        try:
            for variant in variants:
                prune_staged_llama(variant)
        except (BuildError, fetch.PinError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        return 0

    try:
        require_tool("cmake")
        source_dir = ensure_source()
        staged = [build_variant(v, source_dir, args.jobs, not args.no_llama) for v in variants]
    except (BuildError, fetch.PinError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("\nresults:")
    for path in staged:
        print(f"  {path}  {path.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
