"""Bootstrap the per-user MSYS2 UCRT64 toolchain into build/toolchain (spec 19.2).

Usage:
    py build/bootstrap_toolchain.py [--record] [--offline]

No admin, no registry, no PATH changes: everything lands under build/toolchain and
build/cache. The sequence follows https://www.msys2.org/docs/ci/ and
https://www.msys2.org/docs/installer/:

1. fetch and hash-check the pinned msys2 base sfx, run it as ``<sfx> -y -o<build/toolchain>``
   (a 7-Zip self extractor; it creates build/toolchain/msys64)
2. ``usr\\bin\\bash.exe -lc ' '`` once, which runs the first-start initialization
3. online: ``pacman --noconfirm -Syuu`` twice (the first run may stop after updating the
   core packages and ask for a restart; the second run finishes the update)
   offline (--offline): ``pacman --noconfirm -U --needed`` over the package files cached in
   build/cache/msys2-pkgs/, each verified against pins.json first
4. online: ``pacman --noconfirm -S --needed`` for the packages listed below
5. copy every file pacman downloaded (msys64/var/cache/pacman/pkg/) into
   build/cache/msys2-pkgs/ and record name-to-version plus file-to-sha256 in pins.json
   (``--record`` writes, otherwise differences are only reported)
6. extract the pinned, Authenticode-signed Kitware CMake into build/toolchain/cmake
7. run gcc, cmake, ninja, and glslc and print their versions, then compile, archive, link
   and run a small C/C++ program plus a compute shader to prove the whole chain works

Rerunning skips whatever is already in place.

Why a second CMake: the reference machine runs Smart App Control (Windows code integrity,
user-mode enforcement). It allows signed binaries and unsigned binaries with cloud
reputation, blocks an unsigned file the first time it is seen (the verdict is cached, so a
retry usually passes), and keeps blocking unsigned files without reputation. MSYS2's
cmake.exe stays blocked, Kitware's signed one runs. ``run_tool`` below retries around the
first-sight blocks and names any file that stays blocked.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch

REPO_DIR = Path(__file__).resolve().parents[1]
BUILD_DIR = REPO_DIR / "build"
CACHE_DIR = BUILD_DIR / "cache"
TOOLCHAIN_ROOT = BUILD_DIR / "toolchain"
MSYS_DIR = TOOLCHAIN_ROOT / "msys64"
BASH = MSYS_DIR / "usr" / "bin" / "bash.exe"
UCRT_BIN = MSYS_DIR / "ucrt64" / "bin"
PACMAN_PKG_CACHE = MSYS_DIR / "var" / "cache" / "pacman" / "pkg"
PKG_CACHE = CACHE_DIR / "msys2-pkgs"
CMAKE_DIR = TOOLCHAIN_ROOT / "cmake"
CMAKE_EXE = CMAKE_DIR / "bin" / "cmake.exe"
SELFTEST_DIR = TOOLCHAIN_ROOT / "selftest"

MSYS2_KEY = "toolchain.msys2"
CMAKE_KEY = "toolchain.cmake_kitware"
PACKAGES = [
    "mingw-w64-ucrt-x86_64-gcc",
    "mingw-w64-ucrt-x86_64-cmake",
    "mingw-w64-ucrt-x86_64-ninja",
    "mingw-w64-ucrt-x86_64-vulkan-devel",
    "mingw-w64-ucrt-x86_64-shaderc",
    "mingw-w64-ucrt-x86_64-spirv-headers",
]
TOOLS = ("gcc", "cmake", "ninja", "glslc")
PACMAN_ATTEMPTS = 3
POLICY_ATTEMPTS = 4
POLICY_RETRY_DELAY_S = 5
ERROR_APPCONTROL_BLOCKED = 4551  # WinError from CreateProcess when code integrity rejects the exe
CODE_INTEGRITY_LOG = "Microsoft-Windows-CodeIntegrity/Operational"

# Windows variables that native tools and the MSYS2 runtime need. Nothing from the calling
# shell (a Git Bash or another MSYS2 instance) leaks in, so MSYSTEM, HOME, PATH and friends are
# always the values set here.
PASSTHROUGH_VARS = (
    "SystemRoot", "SystemDrive", "windir", "TEMP", "TMP", "USERPROFILE", "USERNAME",
    "COMPUTERNAME", "ProgramData", "ALLUSERSPROFILE", "APPDATA", "LOCALAPPDATA", "HOMEDRIVE",
    "HOMEPATH", "PATHEXT", "COMSPEC", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE",
    "PROCESSOR_IDENTIFIER", "PROCESSOR_LEVEL", "PROCESSOR_REVISION", "OS", "PUBLIC",
)

HELLO_C = "int hello_c(void) { return 42; }\n"
HELLO_CPP = (
    "#include <cstdio>\n#include <string>\n"
    'extern "C" int hello_c(void);\n'
    'int main() { std::string s = "hello from a fresh exe"; '
    'std::printf("%s %d\\n", s.c_str(), hello_c()); return 0; }\n'
)
SHADER_COMP = (
    "#version 450\nlayout(local_size_x = 64) in;\n"
    "layout(binding = 0) buffer Data { float v[]; };\n"
    "void main() { v[gl_GlobalInvocationID.x] *= 2.0; }\n"
)


class BootstrapError(Exception):
    pass


# ----------------------------------------------------------------------------- environment


def clean_windows_env(extra_path: list[Path] | None = None) -> dict[str, str]:
    """A minimal Windows environment: system variables plus PATH = extra_path + System32."""
    lookup = {k.upper(): v for k, v in os.environ.items()}
    env = {name: lookup[name.upper()] for name in PASSTHROUGH_VARS if name.upper() in lookup}
    system_root = env.get("SystemRoot", r"C:\Windows")
    path = [str(p) for p in (extra_path or [])]
    path += [os.path.join(system_root, "System32"), system_root]
    env["PATH"] = os.pathsep.join(path)
    return env


def msys_env() -> dict[str, str]:
    env = clean_windows_env()
    env["MSYSTEM"] = "UCRT64"
    env["CHERE_INVOKING"] = "yes"
    env["MSYS2_PATH_TYPE"] = "minimal"
    # keep the login shell's home inside the toolchain directory, not in the user profile
    env["HOME"] = str(MSYS_DIR / "home" / env.get("USERNAME", "user"))
    return env


def to_posix(path: Path) -> str:
    """C:\\a\\b -> /c/a/b for arguments passed to MSYS2 programs."""
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    rest = str(resolved)[len(resolved.drive):].replace("\\", "/")
    return f"/{drive}{rest}"


def cmake_exe() -> Path:
    """The signed Kitware cmake when extracted, otherwise MSYS2's."""
    return CMAKE_EXE if CMAKE_EXE.exists() else UCRT_BIN / "cmake.exe"


# ----------------------------------------------------------------------------- code integrity


def is_policy_block(exc: OSError) -> bool:
    return getattr(exc, "winerror", None) == ERROR_APPCONTROL_BLOCKED


def policy_blocks_since(started: float) -> list[str]:
    """Files that Windows code integrity blocked since ``started`` (a time.time() value).

    Reads events 3077 and 3033 from the CodeIntegrity operational log with wevtutil, which a
    standard user may query. Paths come back without the \\Device\\HarddiskVolumeN prefix.
    """
    window_ms = int((time.time() - started) * 1000) + 3000
    query = (f"*[System[(EventID=3077 or EventID=3033) and "
             f"TimeCreated[timediff(@SystemTime) <= {window_ms}]]]")
    try:
        result = subprocess.run(
            ["wevtutil", "qe", CODE_INTEGRITY_LOG, f"/q:{query}", "/f:text", "/c:200"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
            check=False)
    except (OSError, subprocess.TimeoutExpired):
        return []
    blocked: list[str] = []
    for match in re.finditer(r"attempted to load (\S+) that", result.stdout):
        path = re.sub(r"^\\Device\\HarddiskVolume\d+", "", match.group(1))
        if path not in blocked:
            blocked.append(path)
    return blocked


def run_tool(cmd: list[str], *, cwd: Path, env: dict[str, str],
             attempts: int = POLICY_ATTEMPTS) -> subprocess.CompletedProcess:
    """Run a native tool; retry when code integrity blocked a file it needed.

    Smart App Control blocks an unsigned file the first time it is seen and caches the cloud
    verdict, so the next attempt usually succeeds. A file that stays blocked is named.
    """
    blocked: list[str] = []
    for attempt in range(1, attempts + 1):
        started = time.time()
        result = None
        try:
            result = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", check=False)
        except OSError as exc:
            if not is_policy_block(exc):
                raise
        if result is not None and result.returncode == 0:
            return result
        blocked = policy_blocks_since(started) or ([cmd[0]] if result is None else [])
        if not blocked:
            output = (result.stdout + result.stderr).strip()
            raise BootstrapError(f"{cmd[0]} failed with exit code {result.returncode}:\n{output}")
        print(f"code integrity blocked {', '.join(blocked)} (attempt {attempt}/{attempts}); "
              f"retrying once the verdict is cached", flush=True)
        if attempt < attempts:
            time.sleep(POLICY_RETRY_DELAY_S)
    raise BootstrapError(
        f"code integrity keeps blocking {', '.join(blocked)} after {attempts} attempts. "
        f"Smart App Control / WDAC does not accept this unsigned file; a signed build of the tool "
        f"or a policy change is needed (see Windows Security > App & browser control).")


# ----------------------------------------------------------------------------- msys2 shell


def run_bash(command: str, *, check: bool = True) -> subprocess.CompletedProcess:
    print(f"+ bash -lc {command!r}", flush=True)
    result = subprocess.run([str(BASH), "-lc", command], env=msys_env(), cwd=str(TOOLCHAIN_ROOT),
                            capture_output=True, text=True, encoding="utf-8", errors="replace",
                            check=False)
    output = (result.stdout + result.stderr).strip()
    if output:
        print(output, flush=True)
    if check and result.returncode != 0:
        raise BootstrapError(f"bash -lc {command!r} failed with exit code {result.returncode}")
    return result


def run_pacman(args: str) -> subprocess.CompletedProcess:
    """pacman with retries: the MSYS2 mirrors drop connections often enough to matter."""
    command = f"pacman --noconfirm --disable-download-timeout {args}"
    last = None
    for attempt in range(1, PACMAN_ATTEMPTS + 1):
        last = run_bash(command, check=False)
        if last.returncode == 0:
            return last
        print(f"pacman attempt {attempt}/{PACMAN_ATTEMPTS} failed with exit code {last.returncode}",
              flush=True)
        if attempt < PACMAN_ATTEMPTS:
            time.sleep(5)
    raise BootstrapError(f"{command!r} failed {PACMAN_ATTEMPTS} times (last exit code {last.returncode})")


# ----------------------------------------------------------------------------- steps


def extract_sfx() -> None:
    if BASH.exists():
        print(f"msys2: {MSYS_DIR} already extracted")
        return
    sfx = fetch.fetch(MSYS2_KEY)
    TOOLCHAIN_ROOT.mkdir(parents=True, exist_ok=True)
    cmd = [str(sfx), "-y", f"-o{TOOLCHAIN_ROOT}"]
    print("+ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, env=clean_windows_env(), cwd=str(TOOLCHAIN_ROOT),
                            capture_output=True, text=True, encoding="utf-8", errors="replace",
                            check=False)
    print((result.stdout + result.stderr).strip())
    if result.returncode != 0 or not BASH.exists():
        raise BootstrapError(f"sfx extraction failed (exit {result.returncode}); {BASH} missing")


def first_start() -> None:
    run_bash(" ")


def _query(command: str) -> dict[str, str]:
    """Run a pacman query and return {first column: second column} of its stdout lines."""
    result = run_bash(command, check=False)
    pairs = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            pairs[parts[0]] = parts[1]
    return pairs


def installed_versions() -> dict[str, str | None]:
    """Version per requested name; None when absent.

    ``mingw-w64-ucrt-x86_64-vulkan-devel`` is a package group, so ``pacman -Q`` does not know
    it. Groups are resolved with ``pacman -Qg`` and recorded as ``group: member version, ...``.
    """
    versions: dict[str, str | None] = dict.fromkeys(PACKAGES)
    versions.update((k, v) for k, v in _query("pacman -Q " + " ".join(PACKAGES)).items() if k in versions)
    unresolved = [name for name, version in versions.items() if version is None]
    if not unresolved:
        return versions
    members: dict[str, list[str]] = {}
    result = run_bash("pacman -Qg " + " ".join(unresolved), check=False)
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in versions:
            members.setdefault(parts[0], []).append(parts[1])
    if members:
        all_members = sorted({m for group_members in members.values() for m in group_members})
        member_versions = _query("pacman -Q " + " ".join(all_members))
        for group, group_members in members.items():
            versions[group] = "group: " + ", ".join(
                f"{m} {member_versions.get(m, '?')}" for m in sorted(group_members))
    return versions


def system_update_online() -> None:
    # The first run may stop right after replacing msys2-runtime and pacman ("all MSYS2
    # processes will be closed"); every bash below is a fresh process, so the second run
    # simply continues with the rest of the update.
    run_pacman("-Syuu")
    run_pacman("-Syuu")


def install_online(missing: list[str]) -> None:
    run_pacman("-S --needed " + " ".join(missing))


def install_offline(pins: dict) -> None:
    recorded = pins["toolchain"].get("msys2_package_files") or {}
    if not recorded:
        raise BootstrapError("pins.json has no toolchain.msys2_package_files; run online with --record first")
    if not PKG_CACHE.is_dir():
        raise BootstrapError(f"{PKG_CACHE} is missing; run online with --record first")
    packages = []
    for name, expected in sorted(recorded.items()):
        path = PKG_CACHE / name
        if not path.exists():
            raise BootstrapError(f"offline: {path} is missing")
        actual = fetch.sha256_file(path)
        if actual != expected:
            raise BootstrapError(f"offline: {path} sha256 {actual} does not match pinned {expected}")
        if name.endswith((".pkg.tar.zst", ".pkg.tar.xz")):
            packages.append(to_posix(path))
    print(f"offline: {len(packages)} package files verified in {PKG_CACHE}")
    # twice for the same reason as the online -Syuu pair: a runtime update may stop the first run
    run_pacman("-U --needed " + " ".join(packages))
    run_pacman("-U --needed " + " ".join(packages))


def snapshot_package_files() -> dict[str, str]:
    """Copy pacman's download cache into build/cache/msys2-pkgs and hash every file."""
    PKG_CACHE.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for src in sorted(PACMAN_PKG_CACHE.glob("*")) if PACMAN_PKG_CACHE.is_dir() else []:
        if not src.is_file():
            continue
        dest = PKG_CACHE / src.name
        if not dest.exists() or dest.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dest)
    for path in sorted(PKG_CACHE.glob("*")):
        if path.is_file():
            hashes[path.name] = fetch.sha256_file(path)
    return hashes


def record_pins(versions: dict[str, str], files: dict[str, str], record: bool) -> None:
    pins = fetch.load_pins()
    toolchain = pins["toolchain"]
    old_versions = toolchain.get("msys2_packages") or {}
    old_files = toolchain.get("msys2_package_files") or {}
    changed = old_versions != versions or old_files != files
    print("installed package versions:")
    for name, version in versions.items():
        marker = "" if old_versions.get(name) == version else "   (pins.json: " + repr(old_versions.get(name, "")) + ")"
        print(f"  {name} {version}{marker}")
    print(f"{len(files)} package files in {PKG_CACHE}")
    if not changed:
        print("pins.json already matches")
        return
    if record:
        toolchain["msys2_packages"] = versions
        toolchain["msys2_package_files"] = files
        fetch.save_pins(pins)
        print(f"recorded package versions and {len(files)} file hashes into {fetch.PINS_PATH}")
    else:
        print("WARNING: installed packages differ from pins.json; rerun with --record to update it",
              file=sys.stderr)


def extract_cmake() -> None:
    if CMAKE_EXE.exists():
        print(f"cmake: {CMAKE_DIR} already extracted")
        return
    archive = fetch.fetch(CMAKE_KEY)
    print(f"cmake: extracting {archive.name} into {CMAKE_DIR}")
    with zipfile.ZipFile(archive) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        roots = {m.filename.split("/", 1)[0] for m in members}
        if len(roots) != 1:
            raise BootstrapError(f"unexpected cmake zip layout: {sorted(roots)}")
        root = roots.pop() + "/"
        for member in members:
            target = CMAKE_DIR / member.filename[len(root):]
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
    if not CMAKE_EXE.exists():
        raise BootstrapError(f"{CMAKE_EXE} missing after extraction")


def verify_tools() -> dict[str, str]:
    env = clean_windows_env([UCRT_BIN])
    versions = {}
    for tool in TOOLS:
        exe = cmake_exe() if tool == "cmake" else UCRT_BIN / f"{tool}.exe"
        if not exe.exists():
            raise BootstrapError(f"{exe} is missing")
        result = run_tool([str(exe), "--version"], cwd=TOOLCHAIN_ROOT, env=env)
        first_line = (result.stdout or result.stderr).strip().splitlines()[0]
        versions[tool] = first_line
        print(f"  {tool:6s} {first_line}   [{exe}]")
    return versions


def toolchain_selftest() -> None:
    """Exercise every tool the whisper build needs, including the stages gcc spawns itself."""
    SELFTEST_DIR.mkdir(parents=True, exist_ok=True)
    (SELFTEST_DIR / "hello.c").write_text(HELLO_C, encoding="utf-8")
    (SELFTEST_DIR / "hello.cpp").write_text(HELLO_CPP, encoding="utf-8")
    (SELFTEST_DIR / "shader.comp").write_text(SHADER_COMP, encoding="utf-8")
    env = clean_windows_env([UCRT_BIN])
    b = UCRT_BIN
    steps = [
        ("gcc -c", [b / "gcc.exe", "-O2", "-c", "hello.c", "-o", "hello_c.o"]),
        ("g++ -c", [b / "g++.exe", "-O2", "-c", "hello.cpp", "-o", "hello_cpp.o"]),
        ("ar", [b / "ar.exe", "rcs", "libhello.a", "hello_c.o"]),
        ("ranlib", [b / "ranlib.exe", "libhello.a"]),
        ("g++ link", [b / "g++.exe", "hello_cpp.o", "-L.", "-lhello", "-static", "-o", "hello.exe"]),
        ("objdump", [b / "objdump.exe", "-p", "hello.exe"]),
        ("hello.exe", [SELFTEST_DIR / "hello.exe"]),
        ("glslc", [b / "glslc.exe", "-fshader-stage=compute", "--target-env=vulkan1.3",
                   "-o", "shader.spv", "shader.comp"]),
        ("ninja", [b / "ninja.exe", "--version"]),
        ("cmake", [cmake_exe(), "--version"]),
    ]
    for label, cmd in steps:
        result = run_tool([str(c) for c in cmd], cwd=SELFTEST_DIR, env=env)
        if label == "hello.exe" and "hello from a fresh exe 42" not in result.stdout:
            raise BootstrapError(f"selftest hello.exe printed {result.stdout!r}")
        print(f"  selftest {label:10s} ok")


# ----------------------------------------------------------------------------- main


def report_elevation() -> bool | None:
    """Print whether this process runs elevated; return True, False, or None when unknown.

    Spec 19.2 requires the whole bootstrap to work as a standard user. A UAC prompt cannot be
    observed from inside the process, so the check that is actually available is the token:
    running unelevated to the end means nothing asked for elevation and was granted it.
    """
    try:
        elevated = bool(ctypes.windll.shell32.IsUserAnAdmin())
    except (AttributeError, OSError) as exc:
        print(f"elevation: could not be determined ({exc})")
        return None
    if elevated:
        print("elevation: WARNING, this process runs elevated, so the no-admin claim is "
              "unverified. Rerun it from a standard, unelevated shell.")
    else:
        print("elevation: not elevated (IsUserAnAdmin is false), and every step above "
              "succeeded, so nothing here needed admin rights. A UAC prompt cannot be seen "
              "from inside the process; an unelevated run that finishes is the evidence.")
    return elevated


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--record", action="store_true",
                        help="write installed package versions and file hashes into pins.json")
    parser.add_argument("--offline", action="store_true",
                        help="install from build/cache/msys2-pkgs instead of the MSYS2 mirrors")
    args = parser.parse_args(argv)

    try:
        extract_sfx()
        first_start()
        versions = installed_versions()
        missing = [name for name, version in versions.items() if version is None]
        if missing:
            print(f"packages to install: {', '.join(missing)}")
            if args.offline:
                install_offline(fetch.load_pins())
            else:
                system_update_online()
                install_online(missing)
            versions = installed_versions()
            still_missing = [name for name, version in versions.items() if version is None]
            if still_missing:
                raise BootstrapError(f"packages still missing after install: {', '.join(still_missing)}")
        else:
            print("all toolchain packages are already installed")
        files = snapshot_package_files()
        record_pins({name: version for name, version in versions.items()}, files, args.record)
        extract_cmake()
        print("tool versions:")
        verify_tools()
        print("toolchain self-test:")
        toolchain_selftest()
    except (BootstrapError, fetch.PinError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    report_elevation()
    print(f"toolchain ready in {TOOLCHAIN_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
