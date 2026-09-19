"""Build the Spells app folder and the installer (spec 19.3 steps 4 to 8, 19.4, 19.5, 20.5).

Usage:
    py build/package.py --all [--cleanup-model <path>] [--span]
    py build/package.py --online [--model-mirror <base url>]
    py build/package.py --step tests --step app --step dist --step installer --step size
    py build/package.py --test-install [--installer <path>]
    py build/package.py --online --update-source https://<server>/spells/

The six steps are spec 19.3 steps 4 to 8 plus the version file of 19.7, named rather than
numbered:

    tests      step 4: .venv/Scripts/python.exe -m pytest tests/unit -q, and stop on failure
    app        step 5: the PyInstaller one-folder, windowed, no-UPX build of build/entry.py,
               with every Qt module src/spells/ui does not import excluded and the Qt qml and
               translations folders pruned afterwards
    dist       step 6: assemble dist/Spells from that build, build/out/engines, the three
               models, the repository's data/ and build/licenses.py assemble
    installer  step 7: fill build/spells.iss.template and compile it with the Inno Setup
               compiler build/bootstrap_packaging.py unpacked into build/toolchain; spans
               across Setup.exe plus .bin slices and zips them when the build needs it
    size       step 8: without spanning, fail above 4,000,000,000 bytes; with it, report the
               size across every slice instead, since Inno Setup's own ceiling does not apply
    latest     spec 19.7: write dist/latest.json, the version file the app checks, from
               CHANGELOG.md and the installer this run produced, and print its path

``--update-source <base url>`` is where the owner serves the installers and the version file.
It is fixed into the build twice over and nowhere else: as ``data/update-source.json`` inside
the app folder, which is the only address the installed app will ever open, and as the
installer address inside ``dist/latest.json``. The installed program offers no way to point
either of them somewhere else, and an update downloads that installer and nothing else, never
a model file. Without the flag the build runs exactly as before and simply cannot check for
updates: the About page says so.

The models that ship are the ones the default selection for English, German and Albanian needs
(spells.modelcatalog, data/models/catalog.json): the speech model, plus one cleanup model with
its extra files. When the selection for a graphics card and the one for the processor differ,
the processor's choice ships, because it serves both kinds of computer and two cleanup models
would not fit the 4,000,000,000-byte gate. A release build refuses to run unless
models.cleanup_model pins exactly that file. ``--cleanup-model <path>`` ships that GGUF (and,
when it is a catalog model, its extra files beside it) instead and marks the build with a
``-dev`` version suffix, which the installer's AppVersion and file name carry. Neither flag
changes what the default bundle contains.

A build whose models exceed the 4,000,000,000-byte single-file ceiling needs Inno Setup's disk
spanning, which the compiler itself requires above 4,200,000,000 compressed bytes. The installer
step estimates the compressed size ahead of compiling (the stored models plus a fixed estimate
for the compressed app, engines and data, spec 19.4) and spans automatically once that estimate
passes the same 4,000,000,000-byte line a non-spanning build would fail against; ``--span``
forces it regardless of the estimate. A spanning build sets ``DiskSliceSize`` to Inno's own
2,100,000,000-byte maximum, so the result is Setup.exe plus one or two ``-N.bin`` slices rather
than many small ones, and every slice is then zipped into ``dist/Spells-Setup-<version>.zip``,
the deliverable to share. Step 8 reports both arithmetics either way: the estimate computed
before compiling and the actual size measured after.

``--online`` builds the second flavour of the installer (B5-29 to B5-33, B5-48): the same app
folder, both engine folders, the data files and the licences, with no model weights at all, so
the download the owner hands out is a few hundred megabytes rather than the offline installer.
The models are downloaded during the installation, verified against the pinned sha256 hashes,
and moved into the models folder beside the app. Which files a machine downloads comes from ``install_sets``:
one row per hardware class and language set, computed here from the catalog and written into
the generated Inno Setup script, so the installer looks its own machine up rather than
reasoning about models in Pascal. It detects a graphics card from the display adapters in the
registry, shows one plain line saying which writing model that earns it, and offers a checkbox
for the smaller one. The wizard never shows an address: it names the models and their sizes and
logs the rest. Where the files come from is fixed at build time, the pinned public addresses or
the folder ``--model-mirror <base url>`` names, with the pinned addresses as a silent fallback.
The two artifacts are siblings and a release makes both: ``--all`` for the offline installer,
which is the one for an air-gapped machine, and ``--online`` for everybody with a network.

``--test-install`` runs the installer tests of spec 20.5 that need no second machine and no
microphone: a silent install, the Add/Remove Programs entry, the app coming up, its TCP
endpoints, an upgrade over a running instance, and both uninstall paths. It puts the machine
back exactly as it found it. Airplane-mode dictation and autostart after a reboot stay manual.

A check this machine cannot answer is reported as SKIP with its reason, never as a pass, and a
run with any skip exits 2: an installer nobody could verify must not read as verified. The
skip that hits today is Smart App Control. It is enforcing on the reference laptop and refuses
the unsigned PyInstaller executable and Inno Setup's unins000.exe with WinError 4551, so the
checks that need a running Spells or a working uninstaller are skipped and the cleanup removes
the install folder, its Start menu shortcut and its registry entries by hand instead.

Smart App Control only accepts an Authenticode signature from a certificate that chains to a
trusted CA, plus the file's own reputation; a self-signed certificate does not satisfy it and
neither does anything this build can do to an unsigned exe. Until a real certificate exists,
every Smart App Control machine reports the app-launch checks as SKIP. ``--signtool
"<command with $f>"`` (or the SPELLS_SIGNTOOL environment variable) is the hook for that day:
given a command, the build signs every unsigned .exe, .dll and .pyd in dist with it before ISCC
and hands the same command to Inno Setup as SignTool, which then signs the installer and the
uninstaller too. build/sign.ps1 is a ready command for a certificate in the user's store. It
is off by default and nothing is signed without it.

Stdlib only, plus the build's own fetch, licenses and bootstrap_packaging modules.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import tomllib
import winreg
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import bootstrap_packaging as packaging
import bootstrap_toolchain as toolchain
import fetch
import licenses as licenses_mod

from spells import updates
from spells.modelcatalog import (
    INSTALL_BUDGET_BYTES,
    VAD_MODEL_BYTES,
    CatalogModel,
    Hardware,
    InstallPlan,
    ModelKind,
    language_name,
    load_catalog,
    model_bytes,
    plan_install,
)

REPO_DIR = Path(__file__).resolve().parents[1]
BUILD_DIR = REPO_DIR / "build"
OUT_DIR = BUILD_DIR / "out"
DIST_DIR = REPO_DIR / "dist"
SRC_DIR = REPO_DIR / "src"
DATA_DIR = REPO_DIR / "data"
ENGINES_DIR = OUT_DIR / "engines"
MODELS_CACHE = BUILD_DIR / "cache" / "models"
LICENSES_DIR = BUILD_DIR / "licenses"
PYPROJECT_PATH = REPO_DIR / "pyproject.toml"
CHANGELOG_PATH = REPO_DIR / "CHANGELOG.md"
VENV_PYTHON = REPO_DIR / ".venv" / "Scripts" / "python.exe"

APP_NAME = "Spells"
ENTRY_SCRIPT = BUILD_DIR / "entry.py"
APP_ICON = BUILD_DIR / "brand" / "spells.ico"
ISS_TEMPLATE = BUILD_DIR / "spells.iss.template"
ONLINE_ISS_TEMPLATE = BUILD_DIR / "spells-online.iss.template"
APP_BUILD_DIR = OUT_DIR / "app"           # PyInstaller writes <APP_BUILD_DIR>/Spells here
PYINSTALLER_WORK = OUT_DIR / "pyinstaller-work"
GENERATED_ISS = OUT_DIR / "spells.iss"
GENERATED_ONLINE_ISS = OUT_DIR / "spells-online.iss"
DIST_APP_DIR = DIST_DIR / APP_NAME
DIST_ONLINE_DIR = DIST_DIR / f"{APP_NAME}-Online"

WHISPER_PIN = "models.whisper_large_v3_turbo_q8_0"
VAD_PIN = "models.silero_vad"
CLEANUP_PIN = "models.cleanup_model"

# src/spells/ui imports QtCore, QtGui, QtWidgets and QtSvg (the brand SVGs in data/brand) and
# nothing else: no QtNetwork (the engines are reached with http.client).
QT_KEEP = ("QtCore", "QtGui", "QtWidgets", "QtSvg")
QT_DROP = (
    "QtConcurrent", "QtDBus", "QtDesigner", "QtHelp", "QtNetwork", "QtOpenGL",
    "QtOpenGLWidgets", "QtPrintSupport", "QtQml", "QtQuick", "QtQuickControls2",
    "QtQuickTest", "QtQuickWidgets", "QtSql", "QtSvgWidgets", "QtTest",
    "QtUiTools", "QtXml",
)
# Collected by the PySide6 hook whatever is excluded, and useless without QtQuick or a
# translated UI. Removing them is worth about 12 MB of the app budget of 19.4.
PYSIDE_PRUNE_DIRS = ("qml", "translations")

# Spec 19.4 budgets the app part at about 120 MB; 160 MB is where the build says so loudly.
APP_SIZE_TARGET_BYTES = 120_000_000
APP_SIZE_WARN_BYTES = 160_000_000
# Spec 19.3 step 8, B5-49: ISCC refuses a single-file installation above 2,100,000,000 bytes
# ("Disk spanning must be enabled in order to create an installation larger than 2100000000
# bytes in size"), measured from the compiler itself, not the 4,200,000,000 bytes B5-24
# assumed. A single-file Setup.exe is refused here above the lower INSTALLER_MAX_BYTES, the
# same number a build uses to decide ahead of time that it has to span rather than wait for
# Inno to refuse it.
INSTALLER_MAX_BYTES = 2_000_000_000
INSTALLER_SPAN_REQUIRED_BYTES = 2_100_000_000
DISK_SLICE_SIZE_BYTES = 2_100_000_000
NONMODEL_COMPRESSED_ESTIMATE_BYTES = 84_000_000

DIST_TOP_LEVEL = ("Spells.exe", "_internal", "data", "engines", "licenses", "models")
# The online build ships no models\ at all: the installer downloads them (B5-29).
ONLINE_DIST_TOP_LEVEL = tuple(name for name in DIST_TOP_LEVEL if name != "models")
GEMMA_LICENSE_FILE = "gemma3.txt"

# The languages the online installer's first page offers, and the one ticked to begin with.
ONLINE_LANGUAGES = ("en", "de", "sq")
ONLINE_DEFAULT_LANGUAGES = ("en",)
# The VAD model is no catalog entry: it belongs to whisper-server rather than to a language.
VAD_MODEL_ID = "silero-vad"
VAD_PURPOSE = "voice detection, which every language needs"
CLEANUP_PURPOSE = "cleanup in every language"
VAD_TITLE = "Voice detection model"
CLEANUP_TITLE = "Writing and cleanup model"

# What --test-install touches, and puts back (spec 15).
INSTALL_DIR = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / APP_NAME
SETTINGS_DIR = Path(os.environ.get("APPDATA", "")) / APP_NAME
USER_DATA_DIR = Path(os.environ.get("LOCALAPPDATA", "")) / APP_NAME
START_MENU_SHORTCUT = (Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows"
                       / "Start Menu" / "Programs" / f"{APP_NAME}.lnk")
UNINSTALL_KEY = r"Software\Microsoft\Windows\CurrentVersion\Uninstall"
# Inno Setup names the uninstall key after the AppId of build/spells.iss.template.
APP_ID = "{9D5B0F4E-3E2A-4C7A-9C3E-2A1D6B8F41C7}"
UNINSTALL_SUBKEY = f"{APP_ID}_is1"
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
RUN_VALUE = "Spells"
WINDOW_CLASS = "SpellsMessageWindow"
ENGINE_PROCESSES = ("whisper-server.exe", "llama-server.exe")
SILENT_INSTALL_ARGS = ("/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART")
LOOPBACK = ("127.0.0.1", "::1")

BUNDLE_LANGUAGES = ("en", "de", "sq")

STEPS = ("tests", "app", "dist", "installer", "size", "latest")

# The version file of spec 19.7 and the address the app reads out of the app folder. Both are
# written only when --update-source says where the owner serves this release.
VERSION_FILE_NAME = updates.MANIFEST_NAME
VERSION_FILE_SCHEMA = updates.MANIFEST_SCHEMA
UPDATE_SOURCE_NAME = updates.SOURCE_FILE
UPDATE_SOURCE_ENV = "SPELLS_UPDATE_SOURCE"
CHANGE_MAX_CHARS = updates.MAX_CHANGE_CHARS
SIGNTOOL_ENV = "SPELLS_SIGNTOOL"

# What main() returns. A skipped 20.5 check is not a pass: an installer nobody could verify
# must not read as verified, so it gets an exit code of its own rather than 0.
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_SKIPPED = 2


class PackageError(Exception):
    """A packaging problem: a failed step, a layout that does not match, a size over budget."""


# ----------------------------------------------------------------------------- pure helpers


def qt_excludes() -> list[str]:
    """``--exclude-module`` arguments for every Qt module the ui never imports."""
    return sorted(f"PySide6.{name}" for name in QT_DROP)


def pyinstaller_args(python_exe: Path, script: Path, *, dist_dir: Path, work_dir: Path,
                     src_dir: Path, icon: Path = APP_ICON) -> list[str]:
    """One folder, windowed, no UPX, unused Qt modules excluded (spec 19.3 step 5).

    No ``--add-data``: the app finds its models, engines, data files and licences beside the
    executable through spells.paths, exactly as the installer lays them out (19.3 step 6), so
    PyInstaller never learns about them. The icon is build/brand/spells.ico, which
    build/brand/make_icons.py draws from data/brand/spells-logo.svg.
    """
    args = [
        str(python_exe), "-m", "PyInstaller", "--noconfirm", "--clean",
        "--onedir", "--windowed", "--noupx", "--log-level", "WARN",
        "--name", APP_NAME,
        "--icon", str(icon),
        "--paths", str(src_dir),
        "--distpath", str(dist_dir),
        "--workpath", str(work_dir),
        "--specpath", str(work_dir),
    ]
    for module in qt_excludes():
        args += ["--exclude-module", module]
    args.append(str(script))
    return args


def project_version(pyproject_text: str) -> str:
    """The version in pyproject.toml's [project] table."""
    try:
        version = tomllib.loads(pyproject_text)["project"]["version"]
    except (tomllib.TOMLDecodeError, KeyError, TypeError) as exc:
        raise PackageError(f"pyproject.toml has no [project] version: {exc}") from exc
    if not isinstance(version, str) or not version.strip():
        raise PackageError(f"pyproject.toml's [project] version is not a version: {version!r}")
    return version.strip()


def build_version(base: str, *, dev: bool) -> str:
    """The version this build carries; a dev build says so in its name (spec 18 step 4)."""
    return f"{base}-dev" if dev else base


def installer_basename(version: str) -> str:
    return f"{APP_NAME}-Setup-{version}"


def app_size_verdict(size: int) -> str:
    """"ok", "near" or "over" against the app budget of spec 19.4."""
    if size <= APP_SIZE_TARGET_BYTES:
        return "ok"
    if size <= APP_SIZE_WARN_BYTES:
        return "near"
    return "over"


def installer_size_ok(size: int) -> bool:
    return size <= INSTALLER_MAX_BYTES


def estimate_compressed_bytes(models_bytes: int) -> int:
    """The expected installer size: the stored models plus the measured app+engines+data cost.

    Spec 19.4's dev build (a 491 MB test model alongside the fixed Whisper and VAD models)
    compiled to 1.45 GB, which puts the compressed app, engines and data near 84 MB; that part
    barely moves with the cleanup model chosen, so it stands in as a constant here.
    """
    return models_bytes + NONMODEL_COMPRESSED_ESTIMATE_BYTES


def needs_spanning(estimated_bytes: int) -> bool:
    """Whether to span proactively: the same ceiling a non-spanning build would fail against."""
    return estimated_bytes > INSTALLER_MAX_BYTES


def slice_count(total_bytes: int, slice_size: int = DISK_SLICE_SIZE_BYTES) -> int:
    """How many Setup.exe/.bin parts Inno Setup would emit for this compressed size."""
    return max(1, -(-total_bytes // slice_size))


def disk_spanning_lines(span: bool) -> str:
    """The [Setup] DiskSpanning block, or a comment saying this build fits one file."""
    if not span:
        return (f"; This build fits one Setup.exe, so DiskSpanning stays off. Inno Setup "
                f"requires it above {INSTALLER_SPAN_REQUIRED_BYTES:,} compressed bytes.")
    return f"DiskSpanning=yes\nDiskSliceSize={DISK_SLICE_SIZE_BYTES}"


def ships_gemma(cleanup_model_name: str, catalog: Sequence[CatalogModel] | None = None) -> bool:
    """Whether the shipped cleanup model needs the Gemma acceptance page (spec 21)."""
    model = catalog_model_for(cleanup_model_name, catalog)
    if model is not None:
        return model.license.lower().startswith("gemma")
    return cleanup_model_name.lower().startswith("gemma")


def catalog_model_for(file_name: str, catalog: Sequence[CatalogModel] | None = None
                      ) -> CatalogModel | None:
    models = load_catalog() if catalog is None else catalog
    wanted = file_name.lower()
    return next((model for model in models if model.file.lower() == wanted), None)


def catalog_extra_names(catalog: Sequence[CatalogModel] | None = None) -> frozenset[str]:
    models = load_catalog() if catalog is None else catalog
    return frozenset(name.lower() for model in models for name in model.extra_files)


@dataclass(frozen=True)
class Bundle:
    models: tuple[CatalogModel, ...]
    left_out: tuple[CatalogModel, ...]
    personal: tuple[CatalogModel, ...] = ()
    plan: InstallPlan | None = None

    @property
    def cleanup(self) -> CatalogModel | None:
        return next((m for m in self.models if m.kind is ModelKind.CLEANUP), None)

    @property
    def speech(self) -> tuple[CatalogModel, ...]:
        return tuple(m for m in self.models if m.kind is ModelKind.ASR)

    @property
    def footprint_bytes(self) -> int:
        return self.plan.footprint_bytes if self.plan is not None else 0


def default_bundle(catalog: Sequence[CatalogModel] | None = None,
                   languages: Sequence[str] = BUNDLE_LANGUAGES) -> Bundle:
    """What the offline installer ships: the processor set, which serves any machine (B5-46).

    A machine with a graphics card would install a different and slightly smaller set, but one
    offline artifact has to work on both, and only the processor set does: its cleanup model is
    the one that answers inside the 1.8 s target without a graphics card.
    """
    everything = tuple(load_catalog() if catalog is None else catalog)
    personal = tuple(model for model in everything if not model.redistributable)
    processor = plan_install(languages, Hardware.CPU, everything)
    graphics = plan_install(languages, Hardware.GPU, everything)
    plan = processor if processor.writing is not None else graphics
    other = graphics if plan is processor else processor
    shipped = tuple(model for model in plan.models if model.kind is not ModelKind.COMPOSE)
    left_out = tuple(model for model in (*plan.dropped, *other.models)
                     if model.kind is not ModelKind.COMPOSE
                     and model.id not in {m.id for m in shipped})
    return Bundle(shipped, left_out, personal, plan)


def bundle_lines(bundle: Bundle, languages: Sequence[str] = BUNDLE_LANGUAGES) -> list[str]:
    names = "/".join(language_name(code) for code in languages)
    lines = [f"  the processor set for {names}, which serves any machine:"]
    for model in bundle.models:
        lines.append(f"    ships {model.id}: {', '.join(model.files)} ({model.size_bytes:,} bytes)")
    dropped = {model.id for model in (bundle.plan.dropped if bundle.plan else ())}
    for model in bundle.left_out:
        why = ("over the install budget with the rest of this set"
               if model.id in dropped
               else "a machine with a graphics card installs it instead, and one artifact "
                    "cannot carry both")
        lines.append(f"    left out {model.id} ({model.size_bytes:,} bytes): {why}")
    for model in bundle.personal:
        lines.append(f"    never ships {model.id}: not redistributable ({model.license})")
    if bundle.plan is not None:
        lines.append(f"    installed footprint {bundle.plan.footprint_bytes:,} bytes against the "
                     f"{INSTALL_BUDGET_BYTES:,}-byte budget")
    return lines


@dataclass(frozen=True)
class ModelDownload:
    """One file the online installer may fetch during the installation (19.3 step 7, B5-29).

    Which of them a machine actually fetches is an install set, not a property of the file
    (B5-46). ``purpose`` is what the installer tells the user goes missing when a download is
    skipped.
    """

    url: str
    name: str
    sha256: str
    size_bytes: int
    model_id: str
    title: str
    model_bytes: int
    purpose: str


def pinned_downloads(pins: Mapping | None = None) -> dict[str, dict]:
    """Every pinned model file by lower-cased file name, with its url, hash and size."""
    data = fetch.load_pins() if pins is None else pins
    models = data.get("models") or {}
    found: dict[str, dict] = {}
    for name, entry in models.items():
        if not isinstance(entry, dict) or not entry.get("url"):
            continue
        found[fetch.filename_for(entry).lower()] = {
            "key": f"models.{name}",
            "url": str(entry["url"]),
            "sha256": str(entry.get("sha256") or "").strip().lower(),
            "size_bytes": entry.get("size_bytes"),
        }
    return found


def pinned_download(file_name: str, pins: Mapping | None = None) -> dict:
    """The pin for one model file, refusing one that cannot be downloaded and verified."""
    entry = pinned_downloads(pins).get(file_name.lower())
    if entry is None:
        raise PackageError(
            f"the online installer needs a pinned url for {file_name}, which build/pins.json "
            f"does not have; add the pin first")
    if not entry["sha256"]:
        raise PackageError(
            f"{entry['key']} has no sha256, so the installer could not verify {file_name}. "
            f"Record the hash before building an online installer.")
    size = entry["size_bytes"]
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise PackageError(
            f"{entry['key']} has no size_bytes, so the language page could not total the "
            f"download. Add \"size_bytes\": <bytes> to that pin in build/pins.json.")
    return entry


def language_subsets(languages: Sequence[str]) -> list[tuple[str, ...]]:
    """Every non-empty combination of the offered languages, shortest first."""
    found: list[tuple[str, ...]] = []
    for mask in range(1, 1 << len(languages)):
        found.append(tuple(code for index, code in enumerate(languages) if mask & (1 << index)))
    return sorted(found, key=len)


def language_mask(ticked: Sequence[str], languages: Sequence[str] = ONLINE_LANGUAGES) -> int:
    """The ticked boxes as a bit per offered language, which is the installer lookup key."""
    chosen = set(ticked)
    return sum(1 << index for index, code in enumerate(languages) if code in chosen)


@dataclass(frozen=True)
class InstallSet:
    """One row of the installer table: what this machine downloads for these languages.

    The rows are computed here and written into the generated script, so the installer never
    reasons about models in Pascal. It reads its own hardware class and the ticked boxes, looks
    the row up, and downloads exactly the files the row names.
    """

    hardware: Hardware
    languages: tuple[str, ...]
    plan: InstallPlan
    files: tuple[str, ...]
    download_bytes: int

    @property
    def mask(self) -> int:
        return language_mask(self.languages)

    @property
    def reason(self) -> str:
        return self.plan.reason

    @property
    def footprint_bytes(self) -> int:
        return self.plan.footprint_bytes

    @property
    def fits(self) -> bool:
        return self.plan.fits


def install_sets(languages: Sequence[str] = ONLINE_LANGUAGES,
                 catalog: Sequence[CatalogModel] | None = None,
                 pins: Mapping | None = None) -> tuple[InstallSet, ...]:
    """Every machine the installer can meet: both hardware classes by every language set.

    A table rather than one mask per language, because the models a language needs depend on
    the other languages once the size budget of spec 19.4 has a say: a processor that dictates
    in Albanian as well cannot also carry the faster English and German model.
    """
    rows: list[InstallSet] = []
    for hardware in (Hardware.CPU, Hardware.GPU):
        for ticked in language_subsets(languages):
            plan = plan_install(ticked, hardware, catalog)
            files = (pinned_file_name(VAD_PIN), *plan.files)
            total = sum(pinned_download(name, pins)["size_bytes"] for name in files)
            rows.append(InstallSet(hardware=hardware, languages=ticked, plan=plan, files=files,
                                   download_bytes=total))
    return tuple(rows)


def install_set_for(rows: Sequence[InstallSet], hardware: Hardware,
                    ticked: Sequence[str]) -> InstallSet:
    wanted = set(ticked)
    for row in rows:
        if row.hardware is hardware and set(row.languages) == wanted:
            return row
    raise PackageError(f"no install set for {hardware.value} and {sorted(wanted)}")


def check_install_budget(rows: Sequence[InstallSet]) -> None:
    """No machine may end up with an installation over the budget of spec 19.4 (B5-45)."""
    over = [row for row in rows if not row.fits]
    if not over:
        return
    row = over[0]
    excess = row.footprint_bytes - INSTALL_BUDGET_BYTES
    names = "/".join(language_name(code) for code in row.languages)
    where = "a graphics card" if row.hardware is Hardware.GPU else "a processor"
    raise PackageError(
        f"{names} on {where} would install {row.footprint_bytes:,} bytes, which is "
        f"{excess:,} over the {INSTALL_BUDGET_BYTES:,}-byte budget of spec 19.4, and no speech "
        f"model can be left out without losing a language: "
        f"{', '.join(model.id for model in row.plan.models)}. A catalog change has to make one "
        f"of those models smaller, or the budget has to be raised on purpose.")


def online_manifest(languages: Sequence[str] = ONLINE_LANGUAGES,
                    catalog: Sequence[CatalogModel] | None = None,
                    pins: Mapping | None = None,
                    rows: Sequence[InstallSet] | None = None) -> tuple[ModelDownload, ...]:
    """Every file the online installer may download, generated from the pins and the catalog.

    The union of the install sets, in catalog order with the VAD model first. An entry counts
    when an install set names its own file, so an entry that only shares an extra file with a
    chosen one, like the audio projector both Qwen3-ASR quantisations use, adds nothing, and a
    shared file is listed once. The pinned per-file sizes are cross-checked against the catalog
    size_bytes for the model, so a pin and a catalog entry that disagree stop the build rather
    than mislead the language page total.
    """
    models = tuple(load_catalog() if catalog is None else catalog)
    table = install_sets(languages, models, pins) if rows is None else tuple(rows)
    wanted: list[str] = []
    for row in table:
        for name in row.files:
            if name not in wanted:
                wanted.append(name)

    vad_name = pinned_file_name(VAD_PIN)
    vad = pinned_download(vad_name, pins)
    found = [ModelDownload(url=vad["url"], name=vad_name, sha256=vad["sha256"],
                           size_bytes=vad["size_bytes"], model_id=VAD_MODEL_ID,
                           title=VAD_TITLE, model_bytes=vad["size_bytes"],
                           purpose=VAD_PURPOSE)]
    for model in models:
        if model.file not in wanted:
            continue
        total = 0
        for name in model.files:
            entry = pinned_download(name, pins)
            total += entry["size_bytes"]
            if any(found_entry.name == name for found_entry in found):
                continue
            found.append(ModelDownload(
                url=entry["url"], name=name, sha256=entry["sha256"],
                size_bytes=entry["size_bytes"], model_id=model.id,
                title=model_title(model, table, languages), model_bytes=model.size_bytes,
                purpose=model_purpose(model, table, languages)))
        if total != model.size_bytes:
            raise PackageError(
                f"{model.id}: the pinned sizes of {', '.join(model.files)} add up to {total:,} "
                f"bytes, but data/models/catalog.json says {model.size_bytes:,}. One of the two "
                f"is wrong, and the installer's download total would be too.")
    return tuple(found)


def model_purpose(model: CatalogModel, rows: Sequence[InstallSet],
                  languages: Sequence[str] = ONLINE_LANGUAGES) -> str:
    """What the installer says goes missing when this file is skipped."""
    if model.kind is not ModelKind.ASR:
        return CLEANUP_PURPOSE
    return f"speech recognition for {language_phrase(model_languages(model, rows, languages))}"


def model_title(model: CatalogModel, rows: Sequence[InstallSet],
                languages: Sequence[str] = ONLINE_LANGUAGES) -> str:
    """What the download page calls this model. It never names a file or an address (B5-48)."""
    if model.kind is not ModelKind.ASR:
        return CLEANUP_TITLE
    return f"Speech model for {language_phrase(model_languages(model, rows, languages))}"


def model_languages(model: CatalogModel, rows: Sequence[InstallSet],
                    languages: Sequence[str] = ONLINE_LANGUAGES) -> tuple[str, ...]:
    """The offered languages this model recognises and is installed for on some machine.

    A row that ticks Albanian as well can carry Qwen3-ASR for English and German beside
    Whisper, so a language only counts when the model has a score for it.
    """
    return tuple(code for code in languages
                 if model.score(code) is not None
                 and any(code in row.languages and model.file in row.files for row in rows))


def language_phrase(codes: Sequence[str]) -> str:
    """"English", "English and German", "English, German and Albanian"."""
    names = [language_name(code) for code in codes]
    if len(names) <= 1:
        return names[0] if names else ""
    return f"{', '.join(names[:-1])} and {names[-1]}"


def manifest_lines(manifest: Sequence[ModelDownload], rows: Sequence[InstallSet]) -> list[str]:
    """What the build prints about the downloads the installer will do."""
    lines = ["  the installer downloads, per pinned url and sha256:"]
    for item in manifest:
        lines.append(f"    {item.name:<40} {item.size_bytes:>15,} bytes  ({item.purpose})")
    for row in rows:
        names = "/".join(language_name(code) for code in row.languages)
        where = "graphics card" if row.hardware is Hardware.GPU else "processor"
        lines.append(
            f"    {where:<14} {names:<32} {row.download_bytes:>15,} bytes downloaded, "
            f"{row.footprint_bytes:>15,} bytes installed")
    return lines


def model_url(base_url: str, file_name: str, pinned_url: str) -> str:
    """Where one file comes from: a mirror's folder, or the pinned url when there is no mirror.

    A mirror is a plain folder serving the same file names, so the address is joined with the
    file name and nothing else. Trailing slashes are dropped, a path in the address is kept.
    """
    base = (base_url or "").strip().rstrip("/")
    if not base:
        return pinned_url
    return f"{base}/{file_name}"


def download_sources(base_url: str, file_name: str, pinned_url: str) -> tuple[str, ...]:
    """The addresses the installer tries for one file, in order: the mirror, then the pin."""
    mirrored = model_url(base_url, file_name, pinned_url)
    if mirrored == pinned_url:
        return (pinned_url,)
    return (mirrored, pinned_url)


def payload_lines(parts: Sequence[tuple[str, int]],
                  limit: int = INSTALLER_MAX_BYTES) -> tuple[list[str], int]:
    total = sum(size for _name, size in parts)
    lines = [f"  {name:<44} {size:>15,} bytes" for name, size in parts]
    lines.append(f"  {'total before compression':<44} {total:>15,} bytes")
    if total > limit:
        lines.append(f"  WARNING: {total - limit:,} bytes over the {limit:,}-byte gate before "
                     f"compression; only the app, the engines and the data compress (the models "
                     f"are stored), so step 8 decides")
    else:
        lines.append(f"  {limit - total:,} bytes under the {limit:,}-byte gate before compression")
    return lines, total


def check_release_cleanup(pinned_name: str, bundle: Bundle) -> None:
    cleanup = bundle.cleanup
    if cleanup is None:
        raise PackageError("no catalog cleanup model suits the default languages; a release "
                           "cannot ship without one")
    if pinned_name.lower() != cleanup.file.lower():
        raise PackageError(
            f"models.cleanup_model pins {pinned_name or 'nothing'}, but the default selection "
            f"for English, German and Albanian needs {cleanup.file} ({cleanup.id}). Pin that "
            f"file in build/pins.json, or pass --cleanup-model <path> for a dev build.")


def check_bundle_budget(bundle: Bundle) -> None:
    """The offline bundle is an installation too, so it answers to the same budget (B5-45)."""
    plan = bundle.plan
    if plan is None or plan.fits:
        return
    excess = plan.footprint_bytes - INSTALL_BUDGET_BYTES
    raise PackageError(
        f"the offline bundle would install {plan.footprint_bytes:,} bytes, {excess:,} over the "
        f"{INSTALL_BUDGET_BYTES:,}-byte budget of spec 19.4, with "
        f"{', '.join(model.id for model in plan.models)}, and no model in it can be left out "
        f"without losing a language. A catalog change has to make one of them smaller.")


def check_bundle_speech(pinned_names: Iterable[str], bundle: Bundle) -> None:
    pinned = {name.lower() for name in pinned_names}
    wanted = [name for model in bundle.speech for name in (model.file, *model.extra_files)]
    missing = sorted(name for name in wanted if name.lower() not in pinned)
    if not wanted:
        raise PackageError("the default selection needs a speech model but names none")
    if missing:
        raise PackageError(
            f"the default selection needs {', '.join(missing)}, which build/pins.json does not "
            f"pin; add the pin first")


def model_pins() -> dict[str, str]:
    """Every pinned model file name, lower-cased, to its pin key."""
    pins = fetch.load_pins()
    models = pins.get("models") or {}
    index = {}
    for name, entry in models.items():
        if isinstance(entry, dict) and entry.get("url"):
            index[fetch.filename_for(entry).lower()] = f"models.{name}"
    return index


def catalog_speech_names(catalog: Sequence[CatalogModel] | None = None) -> frozenset[str]:
    models = load_catalog() if catalog is None else catalog
    return frozenset(
        name.lower()
        for model in models
        if model.kind is ModelKind.ASR
        for name in (model.file, *model.extra_files)
    )


def extra_model_files(cleanup_model: Path, catalog: Sequence[CatalogModel] | None = None
                      ) -> tuple[Path, ...]:
    model = catalog_model_for(cleanup_model.name, catalog)
    if model is None:
        return ()
    extras = tuple(cleanup_model.parent / name for name in model.extra_files)
    missing = [str(path) for path in extras if not path.is_file()]
    if missing:
        raise PackageError(f"{model.id} needs {', '.join(missing)} beside {cleanup_model.name}")
    return extras


@dataclass(frozen=True)
class DistItem:
    """One entry of the dist/Spells layout of spec 19.3 step 6.

    ``kind`` is "app" for the PyInstaller folder whose contents land at the root, "tree" for a
    directory copied under ``dest``, "file" for a single file placed at ``dest``, and
    "licenses" for the folder build/licenses.py assembles (it has no single source).
    """

    kind: str
    source: Path | None
    dest: str


def dist_plan(*, app_dir: Path, engines_dir: Path, speech_models: Sequence[Path],
              vad_model: Path, cleanup_model: Path, data_dir: Path,
              extra_model_files: Sequence[Path] = ()) -> list[DistItem]:
    """Where every part of the release comes from and where it goes (spec 19.3 step 6)."""
    return [
        DistItem("app", app_dir, "."),
        DistItem("tree", engines_dir / "vulkan", "engines/vulkan"),
        DistItem("tree", engines_dir / "cpu", "engines/cpu"),
        *(DistItem("file", model, f"models/{model.name}") for model in speech_models),
        DistItem("file", vad_model, f"models/{vad_model.name}"),
        DistItem("file", cleanup_model, f"models/{cleanup_model.name}"),
        *(DistItem("file", extra, f"models/{extra.name}") for extra in extra_model_files),
        DistItem("tree", data_dir, "data"),
        DistItem("licenses", None, "licenses"),
    ]


def online_dist_plan(*, app_dir: Path, engines_dir: Path, data_dir: Path) -> list[DistItem]:
    """The same layout as dist_plan without a single model file (B5-29)."""
    return [
        DistItem("app", app_dir, "."),
        DistItem("tree", engines_dir / "vulkan", "engines/vulkan"),
        DistItem("tree", engines_dir / "cpu", "engines/cpu"),
        DistItem("tree", data_dir, "data"),
        DistItem("licenses", None, "licenses"),
    ]


def gguf_names(plan: Sequence[DistItem]) -> list[str]:
    """The GGUF file names the plan puts in models\\, in plan order."""
    return [
        item.dest.rsplit("/", 1)[-1]
        for item in plan
        if item.dest.startswith("models/") and item.dest.lower().endswith(".gguf")
    ]


def check_one_gguf(plan: Sequence[DistItem], extra_names: Iterable[str] = ()) -> None:
    """Exactly one cleanup model ships, beside the catalog's extra files for it.

    The frozen app picks its cleanup model through the catalog selection, and without a catalog
    model falls back to spells.paths._llama_model's "single stray GGUF in models\\". Zero or two
    cleanup models means the app starts with cleanup disabled and a tray warning, which a
    release must never do.
    """
    extras = {name.lower() for name in extra_names}
    names = [name for name in gguf_names(plan) if name.lower() not in extras]
    if len(names) != 1:
        raise PackageError(
            f"models\\ must hold exactly one .gguf, the cleanup model; this plan has "
            f"{len(names)}: {names or 'none'}")


def check_dist_top_level(names: Iterable[str], expected_names: Sequence[str] = DIST_TOP_LEVEL,
                         dist_dir: Path = DIST_APP_DIR,
                         template: Path = ISS_TEMPLATE) -> None:
    """The dist folder must hold exactly what the [Files] section of the installer copies."""
    found = set(names)
    expected = set(expected_names)
    missing = sorted(expected - found)
    extra = sorted(found - expected)
    if missing or extra:
        raise PackageError(
            f"{dist_dir} does not match the installer's [Files] section: "
            f"missing {missing or 'nothing'}, unexpected {extra or 'nothing'}. Add the new name "
            f"to the expected list and to {template.name} together.")


def placeholders_in(template: str) -> set[str]:
    """Every @@NAME@@ the template asks to be filled in."""
    found = set()
    rest = template
    while True:
        start = rest.find("@@")
        if start < 0:
            return found
        end = rest.find("@@", start + 2)
        if end < 0:
            raise PackageError("build/spells.iss.template has an unterminated @@ placeholder")
        found.add(rest[start + 2:end])
        rest = rest[end + 2:]


def license_file_line(licenses_dir: Path, cleanup_model_name: str) -> str:
    """The [Setup] LicenseFile line, or a comment saying why there is none (spec 21)."""
    if not ships_gemma(cleanup_model_name):
        return (f"; No licence acceptance page: the shipped cleanup model "
                f"{cleanup_model_name or '(none)'} is not a Gemma model.")
    return f"LicenseFile={licenses_dir / GEMMA_LICENSE_FILE}"


def sign_setup_lines(signtool: str | None) -> str:
    """The [Setup] SignTool block, or a comment saying this build signs nothing (19.5).

    Inno Setup calls the named SignTool on the files flagged ``sign`` and, with
    SignedUninstaller=yes, on unins000.exe as well. The command carries $f, which Inno
    replaces with the file to sign.
    """
    if not signtool:
        return ("; This build is not signed. Smart App Control needs an Authenticode signature\n"
                "; from a certificate that chains to a trusted CA; pass --signtool to add one.")
    return "SignTool=spells\nSignedUninstaller=yes"


def sign_flag(signtool: str | None) -> str:
    """The extra [Files] flag on Spells.exe, empty when nothing is signed."""
    return " sign" if signtool else ""


def iss_values(*, version: str, output_dir: Path, dist_dir: Path, cleanup_model_name: str,
               licenses_dir: Path, signtool: str | None = None, span: bool = False,
               setup_icon: Path = APP_ICON) -> dict[str, str]:
    """Everything build/spells.iss.template asks for."""
    return {
        "APP_VERSION": version,
        "VERSION_INFO": version.split("-", 1)[0],
        "OUTPUT_DIR": str(output_dir),
        "OUTPUT_BASENAME": installer_basename(version),
        "DIST_DIR": str(dist_dir),
        "SETUP_ICON_FILE": str(setup_icon),
        "LICENSE_FILE_LINE": license_file_line(licenses_dir, cleanup_model_name),
        "SIGN_SETUP_LINES": sign_setup_lines(signtool),
        "SIGN_FLAG": sign_flag(signtool),
        "DISK_SPANNING_LINES": disk_spanning_lines(span),
    }


def online_installer_basename(version: str) -> str:
    return f"{APP_NAME}-Online-Setup-{version}"


def offline_artifact_name(version: str, bundle: Bundle | None = None) -> str:
    """What to hand somebody with no network: the zip when the build spans, else the exe.

    The online installer names it in every download failure, so it has to be the file that
    actually exists in dist.
    """
    models = model_bytes((bundle or default_bundle()).models)
    if needs_spanning(estimate_compressed_bytes(models + VAD_MODEL_BYTES)):
        return f"{installer_basename(version)}.zip"
    return f"{installer_basename(version)}.exe"


# ------------------------------------------------------------- spec 19.7: the version file

def changelog_release(text: str, version: str) -> tuple[str, tuple[str, ...]]:
    """The date and the bullet lines of one version's section of CHANGELOG.md.

    The section heading is ``## [<version>] - <date>``; the lines are every ``- `` bullet
    under it, in order, with the group headings (``### Added`` and the rest) left out and a
    wrapped continuation line folded back into its bullet. The app shows exactly these lines,
    so the changelog and the version file cannot say different things.
    """
    wanted = f"## [{version}]"
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.startswith(wanted):
            start = index
            break
    if start is None:
        raise PackageError(
            f"CHANGELOG.md has no section for {version}; add one before building a release")
    heading = lines[start]
    released = ""
    if " - " in heading:
        released = heading.split(" - ", 1)[1].strip()
    if released and not re.match(r"^\d{4}-\d{2}-\d{2}$", released):
        raise PackageError(
            f"CHANGELOG.md: the date of {version} is {released!r}, not a YYYY-MM-DD date")
    entries: list[str] = []
    for line in lines[start + 1:]:
        if line.startswith("## "):
            break
        if line.startswith("- "):
            entries.append(line[2:].strip())
        elif entries and line.startswith(("  ", "\t")) and line.strip():
            entries[-1] = f"{entries[-1]} {line.strip()}"
    if not entries:
        raise PackageError(f"CHANGELOG.md: the section for {version} lists no changes")
    for entry in entries:
        if len(entry) > CHANGE_MAX_CHARS:
            raise PackageError(
                f"CHANGELOG.md: a line of {version} is {len(entry)} characters, over the "
                f"{CHANGE_MAX_CHARS} the app will read")
    return released, tuple(entries)


def join_url(base: str, name: str) -> str:
    """``<base>/<name>``, whether or not the base was given with a trailing slash."""
    return f"{base.rstrip('/')}/{name}"


def check_update_source(base: str) -> str:
    """The update base address, checked the way the app checks it, or PackageError."""
    parts = urlsplit(base)
    if parts.scheme not in ("https", "http") or not parts.hostname:
        raise PackageError(
            f"--update-source {base!r} is not an http or https address with a server name")
    if parts.query or parts.fragment:
        raise PackageError("--update-source is a folder address, without a query or a fragment")
    return base


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def version_manifest(*, version: str, released: str, changes: Sequence[str], installer: Path,
                     base_url: str, minimum_version: str = "") -> dict:
    """The version file the owner serves beside the installers (spec 19.7).

    Everything in it is measured from the file this build produced: the size and the hash are
    the installer's own and the change lines come out of CHANGELOG.md, so the file cannot
    drift from the release it describes.
    """
    return {
        "schema": VERSION_FILE_SCHEMA,
        "product": APP_NAME,
        "version": version,
        "released": released,
        "minimum_version": minimum_version,
        "installer": {
            "url": join_url(base_url, installer.name),
            "size_bytes": installer.stat().st_size,
            "sha256": sha256_of(installer),
        },
        "changes": list(changes),
    }


def write_update_source(dist_dir: Path, base_url: str) -> Path | None:
    """Fix the update address into the app folder, or remove a stale one (spec 19.7)."""
    target = dist_dir / "data" / UPDATE_SOURCE_NAME
    if not base_url:
        target.unlink(missing_ok=True)
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    body = {"manifest_url": join_url(base_url, VERSION_FILE_NAME)}
    target.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8", newline="\n")
    return target


def pascal_string(value: str) -> str:
    """``value`` as a Pascal Script string literal, apostrophes doubled."""
    return "'" + value.replace("'", "''") + "'"


def pascal_bool(value: bool) -> str:
    return "True" if value else "False"


def model_manifest_lines(manifest: Sequence[ModelDownload]) -> str:
    """The generated body of InitModelManifest in build/spells-online.iss.template.

    The sizes go through StrToInt64: a model file is past what a Pascal Script integer literal
    holds, and the language page adds them up.
    """
    lines = [f"  SetArrayLength(ModelFiles, {len(manifest)});"]
    for index, item in enumerate(manifest):
        lines += [
            f"  ModelFiles[{index}].Url := {pascal_string(item.url)};",
            f"  ModelFiles[{index}].Name := {pascal_string(item.name)};",
            f"  ModelFiles[{index}].Hash := {pascal_string(item.sha256)};",
            f"  ModelFiles[{index}].Size := StrToInt64({pascal_string(str(item.size_bytes))});",
            (f"  ModelFiles[{index}].ModelSize := "
             f"StrToInt64({pascal_string(str(item.model_bytes))});"),
            f"  ModelFiles[{index}].ModelId := {pascal_string(item.model_id)};",
            f"  ModelFiles[{index}].Title := {pascal_string(item.title)};",
            f"  ModelFiles[{index}].Purpose := {pascal_string(item.purpose)};",
        ]
    return "\n".join(lines)


def install_set_lines(rows: Sequence[InstallSet], manifest: Sequence[ModelDownload]) -> str:
    """The generated body of InitInstallSets: one row per hardware class and language set.

    ``Files`` is a comma-separated list of ModelFiles indexes, so the installer downloads a set
    without knowing anything about models. ``Hardware`` is 0 for a processor and 1 for a graphics
    card, and ``Mask`` is the ticked boxes as one bit per offered language.
    """
    index_of = {item.name: index for index, item in enumerate(manifest)}
    lines = [f"  SetArrayLength(InstallSets, {len(rows)});"]
    for index, row in enumerate(rows):
        files = ",".join(str(index_of[name]) for name in row.files)
        lines += [
            f"  InstallSets[{index}].Hardware := {1 if row.hardware is Hardware.GPU else 0};",
            f"  InstallSets[{index}].Mask := {row.mask};",
            f"  InstallSets[{index}].Files := {pascal_string(files)};",
            (f"  InstallSets[{index}].Bytes := "
             f"StrToInt64({pascal_string(str(row.download_bytes))});"),
            f"  InstallSets[{index}].Reason := {pascal_string(row.reason)};",
        ]
    return "\n".join(lines)


def language_page_lines(languages: Sequence[str] = ONLINE_LANGUAGES,
                        ticked: Sequence[str] = ONLINE_DEFAULT_LANGUAGES) -> str:
    """The generated body of InitLanguageList: the checkboxes and the codes behind them."""
    lines = [f"  SetArrayLength(LanguageCodes, {len(languages)});",
             f"  SetArrayLength(LanguageNames, {len(languages)});"]
    for index, code in enumerate(languages):
        name = language_name(code)
        lines += [
            f"  LanguageCodes[{index}] := {pascal_string(code)};",
            f"  LanguageNames[{index}] := {pascal_string(name)};",
            f"  LanguagePage.Add({pascal_string(name)});",
            f"  LanguagePage.Values[{index}] := {pascal_bool(code in ticked)};",
        ]
    return "\n".join(lines)


def uninstall_model_lines(manifest: Sequence[ModelDownload]) -> str:
    """[UninstallDelete] entries for the downloaded files, which Setup keeps no log of."""
    lines = [f'Type: files; Name: "{{app}}\\models\\{item.name}"' for item in manifest]
    lines.append('Type: files; Name: "{app}\\first-run.json"')
    lines.append('Type: dirifempty; Name: "{app}\\models"')
    return "\n".join(lines)


def online_iss_values(*, version: str, output_dir: Path, dist_dir: Path,
                      manifest: Sequence[ModelDownload], rows: Sequence[InstallSet],
                      cleanup_model_name: str,
                      licenses_dir: Path, signtool: str | None = None, mirror: str = "",
                      languages: Sequence[str] = ONLINE_LANGUAGES,
                      setup_icon: Path = APP_ICON) -> dict[str, str]:
    """Everything build/spells-online.iss.template asks for."""
    return {
        "APP_VERSION": version,
        "VERSION_INFO": version.split("-", 1)[0],
        "OUTPUT_DIR": str(output_dir),
        "OUTPUT_BASENAME": online_installer_basename(version),
        "DIST_DIR": str(dist_dir),
        "SETUP_ICON_FILE": str(setup_icon),
        "LICENSE_FILE_LINE": license_file_line(licenses_dir, cleanup_model_name),
        "SIGN_SETUP_LINES": sign_setup_lines(signtool),
        "SIGN_FLAG": sign_flag(signtool),
        "MODEL_MANIFEST_LINES": model_manifest_lines(manifest),
        "INSTALL_SET_LINES": install_set_lines(rows, manifest),
        "LANGUAGE_PAGE_LINES": language_page_lines(languages),
        "UNINSTALL_MODEL_LINES": uninstall_model_lines(manifest),
        "DEFAULT_MIRROR_URL": pascal_string((mirror or "").strip()),
        "OFFLINE_ZIP_NAME": pascal_string(offline_artifact_name(version)),
    }


def signtool_command(explicit: str | None, env: Mapping[str, str] | None = None) -> str | None:
    """The signing command for this build: the flag, else SPELLS_SIGNTOOL, else none."""
    if explicit:
        return explicit.strip() or None
    environ = os.environ if env is None else env
    return (environ.get(SIGNTOOL_ENV) or "").strip() or None


def signtool_args(command: str, target: Path) -> list[str]:
    """``command`` with Inno Setup's $f replaced by the file to sign, split into arguments."""
    if "$f" not in command:
        raise PackageError(
            f"--signtool {command!r} has no $f. Inno Setup replaces $f with the file to sign, "
            f"so the command must name it that way, for example "
            f'\'signtool.exe sign /fd sha256 /tr <url> $f\'.')
    return [part.replace("$f", str(target)) for part in shlex.split(command, posix=False)]


SIGNABLE_SUFFIXES = (".exe", ".dll", ".pyd")
SIGN_LIST = OUT_DIR / "sign-list.txt"


def signable_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*")
                  if path.is_file() and path.suffix.lower() in SIGNABLE_SUFFIXES
                  and path.relative_to(root).parts[0] != "models")


def unsigned_files(files: Sequence[Path]) -> list[Path]:
    if not files:
        return []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SIGN_LIST.write_text("\n".join(str(path) for path in files), encoding="utf-8")
    script = ("[Console]::OutputEncoding = [Text.Encoding]::UTF8; "
              f"Get-Content -LiteralPath '{SIGN_LIST}' -Encoding UTF8 | ForEach-Object {{ "
              "if ((Get-AuthenticodeSignature -LiteralPath $_).Status -eq 'NotSigned') { $_ } }")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                            capture_output=True, text=True, encoding="utf-8", check=False)
    if result.returncode != 0:
        raise PackageError(f"reading the signatures failed: {result.stderr.strip()}")
    unsigned = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    return [path for path in files if str(path) in unsigned]


def sign_tree(signtool: str, root: Path) -> None:
    targets = unsigned_files(signable_files(root))
    print(f"  signing {len(targets)} unsigned program file(s) in {root}", flush=True)
    for target in targets:
        run(signtool_args(signtool, target), cwd=DIST_DIR,
            title=f"signing {target.relative_to(root)}")


def render_iss(template: str, values: Mapping[str, str]) -> str:
    """Fill the template, refusing a missing value and a value nothing asked for."""
    wanted = placeholders_in(template)
    given = set(values)
    if wanted - given:
        raise PackageError(f"the Inno Setup template needs {sorted(wanted - given)}")
    if given - wanted:
        raise PackageError(f"the Inno Setup template has no place for {sorted(given - wanted)}")
    text = template
    for name, value in values.items():
        text = text.replace(f"@@{name}@@", value)
    if "@@" in text:
        raise PackageError("a placeholder survived the rendering of the Inno Setup script")
    return text


def check_app_icon(icon: Path = APP_ICON) -> None:
    """The executable and the installer both carry build/brand/spells.ico."""
    if not icon.is_file():
        raise PackageError(f"{icon} is missing; run py build/brand/make_icons.py first")


def format_size(size: int) -> str:
    return f"{size:,} bytes ({size / 1_000_000:.1f} MB)"


# ----------------------------------------------------------------------------- filesystem helpers


def dir_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file() and not item.is_symlink():
            total += item.stat().st_size
    return total


def wipe(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
    if path.exists():
        raise PackageError(f"{path} could not be removed; is something holding a file open?")


def run(cmd: Sequence[str], *, cwd: Path, title: str) -> None:
    """Run a build step with its output going straight to this console."""
    print(f"+ {' '.join(str(part) for part in cmd)}", flush=True)
    started = time.perf_counter()
    result = subprocess.run([str(part) for part in cmd], cwd=str(cwd), check=False)
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        raise PackageError(f"{title} failed with exit code {result.returncode} after {elapsed:.0f} s")
    print(f"  {title} took {elapsed:.0f} s", flush=True)


def start_process(args: Sequence[str], *, cwd: Path,
                  attempts: int = toolchain.POLICY_ATTEMPTS) -> subprocess.Popen:
    """Start a process, surviving a Smart App Control block the way run_tool does.

    Windows refuses an unsigned executable the first time it sees one (WinError 4551) and
    caches the cloud verdict, so a freshly installed Spells.exe starts on a later attempt. The
    frozen exe is unsigned by design: spec 19 signs nothing.
    """
    last: OSError | None = None
    for attempt in range(1, attempts + 1):
        try:
            return subprocess.Popen([str(part) for part in args], cwd=str(cwd))
        except OSError as exc:
            if not toolchain.is_policy_block(exc):
                raise
            last = exc
            print(f"  code integrity blocked {args[0]} (attempt {attempt}/{attempts}); "
                  f"retrying once the verdict is cached", flush=True)
            time.sleep(toolchain.POLICY_RETRY_DELAY_S)
    raise PackageError(
        f"code integrity keeps blocking {args[0]} after {attempts} attempts ({last}). Smart App "
        f"Control does not accept this unsigned file; see Windows Security > App & browser "
        f"control.")


def run_process(args: Sequence[str], *, cwd: Path) -> int:
    """start_process plus a wait; returns the exit code."""
    return start_process(args, cwd=cwd).wait()


# ----------------------------------------------------------------------------- model resolution


def resolve_cleanup_model(override: Path | None) -> tuple[Path, bool]:
    """The cleanup model to ship and whether this is therefore a dev build.

    Without an override the pinned models.cleanup_model is fetched; while the benchmark has
    not picked one its sha256 is empty and build/fetch.py refuses it, which is the release
    build refusing to ship an unchosen model (spec 18 step 4).
    """
    if override is not None:
        path = Path(override).resolve()
        if not path.is_file():
            raise PackageError(f"--cleanup-model {path} does not exist")
        if path.suffix.lower() != ".gguf":
            raise PackageError(f"--cleanup-model {path} is not a .gguf file")
        return path, True
    try:
        return fetch.fetch(CLEANUP_PIN), False
    except fetch.UnpinnedHash as exc:
        raise PackageError(
            f"{exc}\nThe benchmark has not picked a cleanup model yet (spec 18 step 4). Pass "
            f"--cleanup-model <path> for a dev build.") from exc


def pinned_file_name(key: str) -> str:
    return fetch.filename_for(fetch.resolve(fetch.load_pins(), key))


def resolve_models(cleanup_override: Path | None, bundle: Bundle
                   ) -> tuple[tuple[Path, ...], Path, Path, tuple[Path, ...], bool]:
    pins = model_pins()
    check_bundle_speech(pins, bundle)
    if cleanup_override is None:
        check_release_cleanup(pinned_file_name(CLEANUP_PIN), bundle)
    speech = tuple(
        fetch.fetch(pins[name.lower()])
        for model in bundle.speech
        for name in (model.file, *model.extra_files)
    )
    vad = fetch.fetch(VAD_PIN)
    cleanup, dev = resolve_cleanup_model(cleanup_override)
    extras = extra_model_files(cleanup)
    if dev and bundle.cleanup is not None and cleanup.name.lower() != bundle.cleanup.file.lower():
        print(f"  WARNING: this dev build ships {cleanup.name}; the default selection for "
              f"English, German and Albanian needs {bundle.cleanup.file}", file=sys.stderr,
              flush=True)
    return speech, vad, cleanup, extras, dev


# ----------------------------------------------------------------------------- step 4: unit tests


def step_tests() -> None:
    print("\n=== step 4: unit tests ===", flush=True)
    if not VENV_PYTHON.is_file():
        raise PackageError(f"{VENV_PYTHON} is missing; create the project virtual environment first")
    run([VENV_PYTHON, "-m", "pytest", "tests/unit", "-q"], cwd=REPO_DIR, title="the unit tests")


# ----------------------------------------------------------------------------- step 5: PyInstaller


def prune_pyside6(app_dir: Path) -> int:
    """Remove the Qt folders no excluded module can reach; returns the bytes reclaimed."""
    reclaimed = 0
    for parent in (app_dir / "_internal" / "PySide6", app_dir / "PySide6"):
        for name in PYSIDE_PRUNE_DIRS:
            folder = parent / name
            if not folder.is_dir():
                continue
            size = dir_size(folder)
            wipe(folder)
            reclaimed += size
            print(f"  pruned {folder.relative_to(app_dir)} ({format_size(size)})", flush=True)
    return reclaimed


def step_app() -> Path:
    print("\n=== step 5: PyInstaller one-folder build ===", flush=True)
    if not ENTRY_SCRIPT.is_file():
        raise PackageError(f"{ENTRY_SCRIPT} is missing")
    check_app_icon()
    wipe(APP_BUILD_DIR)
    wipe(PYINSTALLER_WORK)
    PYINSTALLER_WORK.mkdir(parents=True, exist_ok=True)
    args = pyinstaller_args(VENV_PYTHON, ENTRY_SCRIPT, dist_dir=APP_BUILD_DIR,
                            work_dir=PYINSTALLER_WORK, src_dir=SRC_DIR)
    run(args, cwd=REPO_DIR, title="PyInstaller")

    app_dir = APP_BUILD_DIR / APP_NAME
    exe = app_dir / f"{APP_NAME}.exe"
    if not exe.is_file():
        raise PackageError(f"{exe} is missing after the PyInstaller build")
    prune_pyside6(app_dir)

    size = dir_size(app_dir)
    verdict = app_size_verdict(size)
    print(f"  app folder: {format_size(size)} ({verdict} against the ~{APP_SIZE_TARGET_BYTES // 1_000_000}"
          f" MB line of spec 19.4)", flush=True)
    if verdict == "over":
        print(f"  WARNING: the app folder is over {format_size(APP_SIZE_WARN_BYTES)}; spec 19.4 "
              f"budgets about {format_size(APP_SIZE_TARGET_BYTES)}", file=sys.stderr, flush=True)
    return app_dir


# ----------------------------------------------------------------------------- step 6: dist/Spells


def step_dist(cleanup_override: Path | None, update_source: str = "") -> tuple[Path, Path]:
    """Assemble dist/Spells; returns (dist dir, the cleanup model that went into it)."""
    print("\n=== step 6: assemble dist/Spells ===", flush=True)
    app_dir = APP_BUILD_DIR / APP_NAME
    if not (app_dir / f"{APP_NAME}.exe").is_file():
        raise PackageError(f"{app_dir} holds no PyInstaller build; run the app step first")
    bundle = default_bundle()
    for line in bundle_lines(bundle):
        print(line, flush=True)
    check_bundle_budget(bundle)
    speech, vad, cleanup, extras, _dev = resolve_models(cleanup_override, bundle)
    for variant in ("vulkan", "cpu"):
        exe = ENGINES_DIR / variant / "whisper-server.exe"
        if not exe.is_file():
            raise PackageError(f"{exe} is missing; run build/build_whisper.py first (19.3 step 3)")

    plan = dist_plan(app_dir=app_dir, engines_dir=ENGINES_DIR, speech_models=speech,
                     vad_model=vad, cleanup_model=cleanup, data_dir=DATA_DIR,
                     extra_model_files=extras)
    check_one_gguf(plan, catalog_extra_names() | catalog_speech_names())

    wipe(DIST_APP_DIR)
    DIST_APP_DIR.mkdir(parents=True, exist_ok=True)
    for item in plan:
        target = DIST_APP_DIR if item.dest == "." else DIST_APP_DIR / item.dest
        if item.kind == "app":
            shutil.copytree(item.source, target, dirs_exist_ok=True)
        elif item.kind == "tree":
            shutil.copytree(item.source, target)
        elif item.kind == "file":
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item.source, target)
        elif item.kind == "licenses":
            licenses_mod.assemble(target)
        else:
            raise PackageError(f"unknown dist item kind {item.kind!r}")
        print(f"  {item.dest:<16} <- {item.source if item.source else LICENSES_DIR}", flush=True)

    source_file = write_update_source(DIST_APP_DIR, update_source)
    if source_file is not None:
        print(f"  data/{UPDATE_SOURCE_NAME} <- {update_source}", flush=True)

    check_dist_top_level(p.name for p in DIST_APP_DIR.iterdir())
    warn_unlisted_cleanup_model(cleanup.name)

    parts = [("app (Spells.exe and _internal)",
              dir_size(DIST_APP_DIR / "_internal") + (DIST_APP_DIR / f"{APP_NAME}.exe").stat().st_size)]
    parts += [(name, dir_size(DIST_APP_DIR / name)) for name in ("engines", "data", "licenses")]
    parts += [(f"models/{path.name}", path.stat().st_size)
              for path in sorted((DIST_APP_DIR / "models").iterdir()) if path.is_file()]
    lines, _total = payload_lines(parts)
    for line in lines:
        print(line, flush=True)
    return DIST_APP_DIR, cleanup


def step_dist_online(update_source: str = "") -> tuple[
        Path, tuple[ModelDownload, ...], tuple[InstallSet, ...]]:
    """Assemble dist/Spells-Online; returns (dist dir, the manifest, the install sets)."""
    print("\n=== step 6: assemble dist/Spells-Online (no model files) ===", flush=True)
    app_dir = APP_BUILD_DIR / APP_NAME
    if not (app_dir / f"{APP_NAME}.exe").is_file():
        raise PackageError(f"{app_dir} holds no PyInstaller build; run the app step first")
    bundle = default_bundle()
    for line in bundle_lines(bundle):
        print(line, flush=True)
    check_bundle_speech(model_pins(), bundle)
    check_release_cleanup(pinned_file_name(CLEANUP_PIN), bundle)
    check_bundle_budget(bundle)
    rows = install_sets()
    check_install_budget(rows)
    manifest = online_manifest(rows=rows)
    for variant in ("vulkan", "cpu"):
        exe = ENGINES_DIR / variant / "whisper-server.exe"
        if not exe.is_file():
            raise PackageError(f"{exe} is missing; run build/build_whisper.py first (19.3 step 3)")

    plan = online_dist_plan(app_dir=app_dir, engines_dir=ENGINES_DIR, data_dir=DATA_DIR)
    wipe(DIST_ONLINE_DIR)
    DIST_ONLINE_DIR.mkdir(parents=True, exist_ok=True)
    for item in plan:
        target = DIST_ONLINE_DIR if item.dest == "." else DIST_ONLINE_DIR / item.dest
        if item.kind == "app":
            shutil.copytree(item.source, target, dirs_exist_ok=True)
        elif item.kind == "tree":
            shutil.copytree(item.source, target)
        elif item.kind == "licenses":
            licenses_mod.assemble(target)
        else:
            raise PackageError(f"unknown dist item kind {item.kind!r}")
        print(f"  {item.dest:<16} <- {item.source if item.source else LICENSES_DIR}", flush=True)

    source_file = write_update_source(DIST_ONLINE_DIR, update_source)
    if source_file is not None:
        print(f"  data/{UPDATE_SOURCE_NAME} <- {update_source}", flush=True)

    check_dist_top_level(
        (p.name for p in DIST_ONLINE_DIR.iterdir()), ONLINE_DIST_TOP_LEVEL,
        DIST_ONLINE_DIR, ONLINE_ISS_TEMPLATE)
    warn_unlisted_cleanup_model(bundle.cleanup.file)

    parts = [("app (Spells.exe and _internal)",
              dir_size(DIST_ONLINE_DIR / "_internal")
              + (DIST_ONLINE_DIR / f"{APP_NAME}.exe").stat().st_size)]
    parts += [(name, dir_size(DIST_ONLINE_DIR / name)) for name in ("engines", "data", "licenses")]
    lines, _total = payload_lines(parts)
    for line in lines:
        print(line, flush=True)
    for line in manifest_lines(manifest, rows):
        print(line, flush=True)
    return DIST_ONLINE_DIR, manifest, rows


def step_installer_online(version: str, manifest: Sequence[ModelDownload],
                          rows: Sequence[InstallSet], signtool: str | None = None,
                          mirror: str = "") -> Path:
    print("\n=== step 7: compile the online installer ===", flush=True)
    if not packaging.ISCC_EXE.is_file():
        raise PackageError(
            f"{packaging.ISCC_EXE} is missing; run py build/bootstrap_packaging.py first")
    if not (DIST_ONLINE_DIR / f"{APP_NAME}.exe").is_file():
        raise PackageError(f"{DIST_ONLINE_DIR} is not assembled; run the dist step first")
    check_app_icon()
    cleanup_name = default_bundle().cleanup.file
    values = online_iss_values(version=version, output_dir=DIST_DIR, dist_dir=DIST_ONLINE_DIR,
                               manifest=manifest, rows=rows, cleanup_model_name=cleanup_name,
                               licenses_dir=LICENSES_DIR, signtool=signtool, mirror=mirror)
    text = render_iss(ONLINE_ISS_TEMPLATE.read_text(encoding="utf-8"), values)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_ONLINE_ISS.write_text(text, encoding="utf-8", newline="\r\n")
    print(f"  wrote {GENERATED_ONLINE_ISS} ({len(manifest)} downloadable files, licence page: "
          f"{'Gemma terms' if ships_gemma(cleanup_name) else 'none'}, "
          f"mirror prefilled: {mirror or 'none'}, signing: {'on' if signtool else 'off'})",
          flush=True)

    if signtool:
        sign_tree(signtool, DIST_ONLINE_DIR)

    installer = DIST_DIR / f"{online_installer_basename(version)}.exe"
    installer.unlink(missing_ok=True)
    command = [packaging.ISCC_EXE]
    if signtool:
        command.append(f"/Sspells={signtool}")
    command.append(GENERATED_ONLINE_ISS)
    run(command, cwd=OUT_DIR, title="ISCC")
    if not installer.is_file():
        raise PackageError(f"ISCC reported success but {installer} is missing")
    return installer


def step_size_online(installer: Path, rows: Sequence[InstallSet]) -> None:
    print("\n=== step 8: online installer size ===", flush=True)
    size = installer.stat().st_size
    print(f"  {installer.name}: {format_size(size)}", flush=True)
    for row in rows:
        names = "/".join(language_name(code) for code in row.languages)
        where = "graphics card" if row.hardware is Hardware.GPU else "processor"
        print(f"  {where:<14} {names:<32} downloads {format_size(row.download_bytes)}, "
              f"installs {format_size(row.footprint_bytes)} of the "
              f"{format_size(INSTALL_BUDGET_BYTES)} budget", flush=True)
    check_install_budget(rows)
    if not installer_size_ok(size):
        raise PackageError(
            f"{installer.name} is {size} bytes, over the {INSTALLER_MAX_BYTES}-byte ceiling of "
            f"spec 19.3 step 8, which an online installer should never approach.")
    print(f"  inside the {INSTALLER_MAX_BYTES:,}-byte ceiling of spec 19.3 step 8", flush=True)


def shipping_license_names() -> set[str]:
    """The component names build/licenses/MANIFEST.json marks as shipping (B4-25)."""
    return {str(entry.get("name", "")).lower()
            for entry in licenses_mod.components() if entry.get("ships")}


def _normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def warn_unlisted_cleanup_model(cleanup_model_name: str) -> str:
    """Warn when the shipped cleanup model has no shipping licence entry (spec 21, B4-25).

    A dev build can ship any GGUF, and a catalog model such as gemma-4-E2B-it-Q4_0.gguf may
    have no shipping entry yet even in a release, so this always checks rather than trusting
    the caller. The comparison strips hyphens and underscores from both sides first: a file
    stem like "gemma-4-e2b-it-q4_0" would otherwise miss a component literally named "gemma4".
    Returns the warning, or "" when the model is listed.
    """
    stem = _normalized(cleanup_model_name.rsplit(".", 1)[0])
    listed = shipping_license_names()
    if any(name and _normalized(name) in stem for name in listed):
        return ""
    message = (f"WARNING: {cleanup_model_name} is not a shipping entry of "
               f"{LICENSES_DIR / licenses_mod.MANIFEST_NAME}, so licenses/index.txt in this "
               f"build lists no licence for the cleanup model. Expected for a dev build; a "
               f"release must mark the chosen model's component ships: true first (spec 21).")
    print(f"  {message}", file=sys.stderr, flush=True)
    return message


# ----------------------------------------------------------------------------- step 7: the installer


def cleanup_gguf_name(names: Iterable[str], extra_names: Iterable[str] = ()) -> str:
    extras = {name.lower() for name in extra_names}
    found = sorted(name for name in names
                   if name.lower().endswith(".gguf") and name.lower() not in extras)
    if len(found) != 1:
        raise PackageError(
            f"{DIST_APP_DIR / 'models'} holds {len(found)} cleanup .gguf files ({found}); it "
            f"must hold exactly one")
    return found[0]


def cleanup_model_in_dist() -> str:
    """The name of the cleanup GGUF sitting in dist/Spells/models, for the Gemma page decision."""
    names = [path.name for path in (DIST_APP_DIR / "models").iterdir()]
    return cleanup_gguf_name(names, catalog_extra_names() | catalog_speech_names())


def installer_parts(installer: Path) -> list[Path]:
    """installer plus its ``-N.bin`` slices, in slice order; just [installer] when not spanned."""
    slices = sorted(installer.parent.glob(f"{installer.stem}-*.bin"),
                    key=lambda path: int(path.stem.rsplit("-", 1)[-1]))
    return [installer, *slices]


def zip_installer(installer: Path, version: str) -> Path:
    """Zip a spanned installer's Setup.exe and its .bin slices into one deliverable file."""
    dest = installer.parent / f"{installer_basename(version)}.zip"
    dest.unlink(missing_ok=True)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_STORED) as archive:
        for part in installer_parts(installer):
            archive.write(part, part.name)
    return dest


def step_installer(version: str, signtool: str | None = None, span: bool = False) -> Path:
    print("\n=== step 7: compile the installer ===", flush=True)
    if not packaging.ISCC_EXE.is_file():
        raise PackageError(
            f"{packaging.ISCC_EXE} is missing; run py build/bootstrap_packaging.py first")
    if not (DIST_APP_DIR / f"{APP_NAME}.exe").is_file():
        raise PackageError(f"{DIST_APP_DIR} is not assembled; run the dist step first")
    check_app_icon()
    cleanup_name = cleanup_model_in_dist()
    estimate = estimate_compressed_bytes(dir_size(DIST_APP_DIR / "models"))
    span = span or needs_spanning(estimate)
    print(f"  estimated compressed size {format_size(estimate)}, spanning "
          f"{'on' if span else 'off'}", flush=True)

    values = iss_values(version=version, output_dir=DIST_DIR, dist_dir=DIST_APP_DIR,
                        cleanup_model_name=cleanup_name, licenses_dir=LICENSES_DIR,
                        signtool=signtool, span=span)
    text = render_iss(ISS_TEMPLATE.read_text(encoding="utf-8"), values)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_ISS.write_text(text, encoding="utf-8", newline="\r\n")
    print(f"  wrote {GENERATED_ISS} (licence page: "
          f"{'Gemma terms' if ships_gemma(cleanup_name) else 'none'}, "
          f"signing: {'on' if signtool else 'off'})", flush=True)

    if signtool:
        sign_tree(signtool, DIST_APP_DIR)

    installer = DIST_DIR / f"{installer_basename(version)}.exe"
    for stale in installer_parts(installer):
        stale.unlink(missing_ok=True)
    print("  compiling; the model files are copied uncompressed, so this takes a few minutes",
          flush=True)
    command = [packaging.ISCC_EXE]
    if signtool:
        # Inno replaces $p with what /Sspells= gives it, and $f with the file it is signing.
        command.append(f"/Sspells={signtool}")
    command.append(GENERATED_ISS)
    run(command, cwd=OUT_DIR, title="ISCC")
    if not installer.is_file():
        raise PackageError(f"ISCC reported success but {installer} is missing")
    if span:
        zip_path = zip_installer(installer, version)
        print(f"  zipped {len(installer_parts(installer))} installer file(s) into {zip_path}",
              flush=True)
    return installer


# ----------------------------------------------------------------------------- step 8: the size gate


def step_size(installer: Path) -> None:
    print("\n=== step 8: installer size ===", flush=True)
    parts = installer_parts(installer)
    sizes = [(part, part.stat().st_size) for part in parts]
    for part, size in sizes:
        print(f"  {part.name}: {format_size(size)}", flush=True)
    total = sum(size for _part, size in sizes)
    models_dir = DIST_APP_DIR / "models"
    if models_dir.is_dir():
        estimate = estimate_compressed_bytes(dir_size(models_dir))
        print(f"  estimated {format_size(estimate)} compressed, actual {format_size(total)} "
              f"across {len(parts)} file(s)", flush=True)
    if len(parts) > 1:
        print(f"  spanned: Inno Setup required this above {INSTALLER_SPAN_REQUIRED_BYTES:,} "
              f"compressed bytes, so the {INSTALLER_MAX_BYTES:,}-byte single-file ceiling of "
              f"spec 19.3 step 8 does not apply", flush=True)
        return
    if not installer_size_ok(total):
        raise PackageError(
            f"{installer.name} is {total} bytes, over the {INSTALLER_MAX_BYTES}-byte ceiling of "
            f"spec 19.3 step 8. Pass --span, or let it trigger automatically, for a build this "
            f"size.")
    print(f"  inside the {INSTALLER_MAX_BYTES:,}-byte ceiling of spec 19.3 step 8", flush=True)


def step_latest(version: str, installer: Path | None, base_url: str,
                minimum_version: str = "") -> Path | None:
    """Write dist/latest.json from CHANGELOG.md and the installer this run built (19.7)."""
    print("\n=== version file: dist/latest.json ===", flush=True)
    if not base_url:
        print("  skipped: pass --update-source <base url> to write one", flush=True)
        return None
    if installer is None or not installer.is_file():
        print(f"  skipped: {installer} was not built in this run", flush=True)
        return None
    parts = installer_parts(installer)
    if len(parts) > 1:
        raise PackageError(
            f"{installer.name} is spanned across {len(parts)} files, and an update installs "
            f"from one file. Publish the online installer (--online) as the update artifact.")
    released, changes = changelog_release(CHANGELOG_PATH.read_text(encoding="utf-8"), version)
    body = version_manifest(version=version, released=released, changes=changes,
                            installer=installer, base_url=base_url,
                            minimum_version=minimum_version)
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    target = DIST_DIR / VERSION_FILE_NAME
    target.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8", newline="\n")
    print(f"  wrote {target}", flush=True)
    print(f"  {version} of {released or 'no date'}, {len(changes)} change lines, installer "
          f"{format_size(body['installer']['size_bytes'])}", flush=True)
    print(f"  serve it as {join_url(base_url, VERSION_FILE_NAME)} beside {installer.name}",
          flush=True)
    return target


# ----------------------------------------------------------------------------- 20.5 installer tests

user32 = ctypes.WinDLL("user32", use_last_error=True)


def window_exists(class_name: str = WINDOW_CLASS) -> bool:
    return bool(user32.FindWindowW(ctypes.c_wchar_p(class_name), None))


def wait_for(predicate, timeout_s: float, poll_s: float = 0.25) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_s)
    return predicate()


def powershell(script: str) -> str:
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if result.returncode != 0:
        raise PackageError(f"powershell failed: {result.stderr.strip()}")
    return result.stdout


def pids_of(names: Sequence[str]) -> dict[str, list[int]]:
    """Process ids by executable name, from tasklist (no elevation needed)."""
    out = subprocess.run(["tasklist.exe", "/FO", "CSV", "/NH"], capture_output=True, text=True,
                         encoding="utf-8", errors="replace", check=False).stdout
    wanted = {name.lower() for name in names}
    found: dict[str, list[int]] = {name: [] for name in names}
    for line in out.splitlines():
        parts = [part.strip('"') for part in line.split('","')]
        if len(parts) < 2:
            continue
        image = parts[0].strip('"').lower()
        if image in wanted:
            try:
                found[next(n for n in names if n.lower() == image)].append(int(parts[1]))
            except (ValueError, StopIteration):
                pass
    return found


def tcp_endpoints(pids: Sequence[int]) -> list[dict]:
    """The TCP endpoints those processes own, State as its name rather than its number.

    Get-NetTCPConnection's State is an enum, and ConvertTo-Json writes an enum as its integer
    (Listen 2, Established 5, Bound 100), so the name is asked for explicitly here. The state
    decides how loopback_only reads the row and a number there would read as an unknown state.
    """
    if not pids:
        return []
    ids = ",".join(str(pid) for pid in pids)
    script = (
        f"$ids = @({ids}); "
        "Get-NetTCPConnection -ErrorAction SilentlyContinue | "
        "Where-Object { $ids -contains $_.OwningProcess } | "
        "Select-Object @{Name='State';Expression={$_.State.ToString()}},"
        "LocalAddress,LocalPort,RemoteAddress,RemotePort,OwningProcess | "
        "ConvertTo-Json -Compress -Depth 3"
    )
    text = powershell(script).strip()
    if not text:
        return []
    data = json.loads(text)
    return data if isinstance(data, list) else [data]


# A socket in one of these states has no peer yet, so only where it is bound can be judged.
NO_PEER_STATES = ("Listen", "Bound", "Closed")


def loopback_only(endpoint: Mapping) -> bool:
    """Whether this endpoint stays on the machine (spec 17, 20.5).

    A listening or bound socket has no peer, so what matters is that it is bound to loopback
    rather than to 0.0.0.0, where anything on the network could reach it. A connected socket
    has to be loopback on both sides.
    """
    state = str(endpoint.get("State", ""))
    local = str(endpoint.get("LocalAddress", ""))
    remote = str(endpoint.get("RemoteAddress", ""))
    if state in NO_PEER_STATES:
        return local in LOOPBACK
    return local in LOOPBACK and remote in LOOPBACK


def registry_value(root, key: str, name: str):
    try:
        with winreg.OpenKey(root, key) as handle:
            return winreg.QueryValueEx(handle, name)[0]
    except OSError:
        return None


def uninstall_entries() -> dict[str, str]:
    """This build's own Add/Remove Programs entry, by its exact key name.

    Only the key Inno Setup derives from the AppId of build/spells.iss.template is looked at.
    A substring match on the display name would find, and the cleanup would then delete,
    somebody else's entry that happens to mention Spells.
    """
    display = registry_value(winreg.HKEY_CURRENT_USER, f"{UNINSTALL_KEY}\\{UNINSTALL_SUBKEY}",
                             "DisplayName")
    return {UNINSTALL_SUBKEY: str(display)} if display is not None else {}


PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


@dataclass
class Check:
    """One spec 20.5 check. SKIP is for a check this machine cannot answer, never a pass."""

    name: str
    status: str
    detail: str = ""


class InstallerTests:
    """Spec 20.5, the parts that run on this account now, and the cleanup that follows them."""

    def __init__(self, installer: Path) -> None:
        self.installer = installer
        self.checks: list[Check] = []
        self.backup_root = OUT_DIR / "test-install-backup"
        self.saved: dict[str, Path] = {}

    # reporting ---------------------------------------------------------------------------

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self._record(name, PASS if ok else FAIL, detail)
        return ok

    def skip(self, name: str, reason: str) -> None:
        """Record a check this machine cannot answer. It is not a pass and never becomes one."""
        self._record(name, SKIP, reason)

    def _record(self, name: str, status: str, detail: str) -> None:
        self.checks.append(Check(name, status, detail))
        print(f"  [{status}] {name}{': ' + detail if detail else ''}", flush=True)

    def counts(self) -> dict[str, int]:
        return {state: sum(1 for c in self.checks if c.status == state)
                for state in (PASS, FAIL, SKIP)}

    # the machine's state before and after ------------------------------------------------

    def save_existing_data(self) -> None:
        """Move the user's own settings and history aside before the tests touch them.

        Moved, not copied: these are the owner's real dictation history and settings, and the
        tests then run against folders the installer and the app create from nothing. Nothing
        is ever deleted here, so a crash between here and the restore leaves the originals
        sitting in build/out/test-install-backup rather than gone.
        """
        wipe(self.backup_root)
        self.backup_root.mkdir(parents=True, exist_ok=True)
        for label, folder in (("settings", SETTINGS_DIR), ("data", USER_DATA_DIR)):
            if folder.exists():
                target = self.backup_root / label
                shutil.move(str(folder), str(target))
                self.saved[label] = folder
                print(f"  moved the existing {folder} aside into {target}", flush=True)
        if not self.saved:
            print("  no existing %APPDATA%\\Spells or %LOCALAPPDATA%\\Spells to move", flush=True)

    def restore_existing_data(self) -> None:
        """Put them back. What the tests created is removed first, and only then."""
        for label, folder in (("settings", SETTINGS_DIR), ("data", USER_DATA_DIR)):
            source = self.backup_root / label
            if label not in self.saved:
                if folder.exists():
                    wipe(folder)
                    print(f"  removed {folder}, which did not exist before this run", flush=True)
                continue
            if not source.is_dir():
                print(f"  WARNING: {source} is gone; {folder} cannot be put back", flush=True)
                continue
            wipe(folder)
            shutil.move(str(source), str(folder))
            print(f"  moved {folder} back", flush=True)
        if not any((self.backup_root / label).exists() for label in self.saved):
            wipe(self.backup_root)

    # the steps ----------------------------------------------------------------------------

    def install(self, *extra: str) -> int:
        args = [str(self.installer), *SILENT_INSTALL_ARGS, *extra]
        print(f"+ {' '.join(args)}", flush=True)
        code = run_process(args, cwd=DIST_DIR)
        # Inno's setup loader usually waits for the setup it extracted, but the uninstaller
        # copy it leaves behind does not, so never trust the return alone: wait for the files.
        wait_for(lambda: (INSTALL_DIR / f"{APP_NAME}.exe").is_file(), timeout_s=600.0, poll_s=0.5)
        wait_for(lambda: not self._setup_running(), timeout_s=600.0, poll_s=0.5)
        return code

    def _setup_running(self) -> bool:
        """Whether Setup is still going.

        The loader extracts the real setup into %TEMP% and runs it as <basename>.tmp, so that
        is the process to watch: the .exe we started may already have returned.
        """
        return any(pids_of(setup_process_names(self.installer)).values())

    def uninstall(self, *extra: str) -> str:
        """Uninstall silently; returns "" on success or the reason it could not be run."""
        uninstaller = INSTALL_DIR / "unins000.exe"
        if not uninstaller.is_file():
            return f"{uninstaller} is missing"
        args = [str(uninstaller), *SILENT_INSTALL_ARGS, *extra]
        print(f"+ {' '.join(args)}", flush=True)
        try:
            run_process(args, cwd=Path(os.environ.get("TEMP", ".")))
        except PackageError as exc:
            return str(exc)
        # The uninstaller copies itself into %TEMP% and the process we started returns at once.
        wait_for(lambda: not INSTALL_DIR.exists(), timeout_s=180.0, poll_s=0.5)
        return ""

    def quit_app(self) -> None:
        exe = INSTALL_DIR / f"{APP_NAME}.exe"
        if exe.is_file():
            try:
                run_process([exe, "--quit"], cwd=INSTALL_DIR)
            except PackageError as exc:
                print(f"  could not run --quit: {exc}", flush=True)
        wait_for(lambda: not window_exists(), timeout_s=15.0)

    def run(self) -> bool:
        print("\n=== spec 20.5: installer tests ===", flush=True)
        if INSTALL_DIR.exists():
            raise PackageError(
                f"{INSTALL_DIR} already exists. These tests install and uninstall Spells, so they "
                f"refuse to run over an installation they did not make. Uninstall it first.")
        self.save_existing_data()
        try:
            self._body()
        finally:
            self._cleanup()
        counts = self.counts()
        print(f"\n  {counts[PASS]} passed, {counts[FAIL]} failed, {counts[SKIP]} skipped of "
              f"{len(self.checks)} installer checks", flush=True)
        if counts[SKIP]:
            print("  a skipped check is not a pass: it is one this machine could not answer.",
                  flush=True)
        return counts[FAIL] == 0

    def _body(self) -> None:
        # 1: a silent per-user install. A UAC prompt cannot be answered by a silent install, so
        # a clean exit is the proof that spec 19.5's "no UAC prompt" holds.
        code = self.install()
        self.check("install exits cleanly, so Setup never asked for elevation", code == 0,
                   f"exit code {code}")
        self.check("the install folder is there", (INSTALL_DIR / f"{APP_NAME}.exe").is_file(),
                   str(INSTALL_DIR))

        # 2: the per-user Add/Remove Programs entry.
        entries = uninstall_entries()
        self.check("a per-user Add/Remove Programs entry exists", bool(entries),
                   ", ".join(f"{k} = {v}" for k, v in entries.items()) or "none found")

        # 3: the app comes up and its hidden window appears.
        running = self._launch_and_check_endpoints()

        # 4: reinstall over the running app; PrepareToInstall closes it with no prompt.
        if running:
            self._upgrade_over_running_app()
        else:
            for name in self.UPGRADE_CHECKS:
                self.skip(name, "the app could not be started on this machine")

        # 5: uninstall keeps the data folders and removes the Run value.
        self._uninstall_keeping_data(app_ran=running)

        # 6: a fresh install, then uninstall with the data removal parameter.
        self._uninstall_removing_data()

    RUNNING_APP_CHECKS = (
        "both engine processes are running",
        "every TCP endpoint of the process tree is on 127.0.0.1",
        "autostart wrote the Run value the uninstaller has to remove",
    )
    UPGRADE_CHECKS = (
        "the app is running before the upgrade",
        "installing over the running app exits cleanly, with no prompt",
        "PrepareToInstall closed the running app before replacing its files",
    )

    def _launch_and_check_endpoints(self) -> bool:
        exe = INSTALL_DIR / f"{APP_NAME}.exe"
        print(f"+ {exe}", flush=True)
        try:
            start_process([exe], cwd=INSTALL_DIR)
        except PackageError as exc:
            self.skip(f"the {WINDOW_CLASS} window appears, so the app started", str(exc))
            for name in self.RUNNING_APP_CHECKS:
                self.skip(name, "the app could not be started on this machine")
            return False
        up = wait_for(window_exists, timeout_s=180.0)
        self.check(f"the {WINDOW_CLASS} window appears, so the app started", up)
        if not up:
            for name in self.RUNNING_APP_CHECKS:
                self.skip(name, "the hidden window never appeared")
            return False
        engines_up = wait_for(
            lambda: all(pids_of(ENGINE_PROCESSES)[name] for name in ENGINE_PROCESSES),
            timeout_s=240.0, poll_s=1.0)
        self.check("both engine processes are running", engines_up,
                   ", ".join(f"{k}={v}" for k, v in pids_of(ENGINE_PROCESSES).items()))
        # Give the engines a moment to bind their loopback ports before listing the endpoints.
        wait_for(lambda: bool(self._all_endpoints()), timeout_s=60.0, poll_s=1.0)
        endpoints = self._all_endpoints()
        offenders = [e for e in endpoints if not loopback_only(e)]
        self.check("every TCP endpoint of the process tree is on 127.0.0.1",
                   bool(endpoints) and not offenders,
                   f"{len(endpoints)} endpoints, {len(offenders)} off loopback"
                   + (f": {offenders}" if offenders else ""))
        self.check("autostart wrote the Run value the uninstaller has to remove",
                   registry_value(winreg.HKEY_CURRENT_USER, RUN_KEY, RUN_VALUE) is not None,
                   str(registry_value(winreg.HKEY_CURRENT_USER, RUN_KEY, RUN_VALUE)))
        return True

    def _all_endpoints(self) -> list[dict]:
        names = (f"{APP_NAME}.exe", *ENGINE_PROCESSES)
        pids = [pid for group in pids_of(names).values() for pid in group]
        return tcp_endpoints(pids)

    def _upgrade_over_running_app(self) -> None:
        before = pids_of([f"{APP_NAME}.exe"])[f"{APP_NAME}.exe"]
        self.check("the app is running before the upgrade", bool(before), f"pids {before}")
        code = self.install()
        self.check("installing over the running app exits cleanly, with no prompt", code == 0,
                   f"exit code {code}")
        gone = wait_for(
            lambda: not set(before) & set(pids_of([f"{APP_NAME}.exe"])[f"{APP_NAME}.exe"]),
            timeout_s=30.0)
        self.check("PrepareToInstall closed the running app before replacing its files", gone,
                   f"pids before {before}, now "
                   f"{pids_of([f'{APP_NAME}.exe'])[f'{APP_NAME}.exe']}")
        self.quit_app()

    KEEP_DATA_CHECKS = (
        "uninstall removed the install folder",
        "uninstall removed the autostart Run value",
        "uninstall removed the Add/Remove Programs entry",
        "uninstall kept the settings and data folders",
    )

    def _uninstall_keeping_data(self, *, app_ran: bool) -> None:
        had_settings = SETTINGS_DIR.exists()
        had_data = USER_DATA_DIR.exists()
        name = "the app wrote its user data, so the uninstall options can be told apart"
        if app_ran:
            self.check(name, had_settings and had_data,
                       f"{SETTINGS_DIR} {had_settings}, {USER_DATA_DIR} {had_data}")
        else:
            self.skip(name, "the app never ran, so there is no user data to keep or remove")
        blocked = self.uninstall()
        if blocked:
            for check_name in self.KEEP_DATA_CHECKS:
                self.skip(check_name, blocked)
            return
        self.check("uninstall removed the install folder", not INSTALL_DIR.exists(),
                   str(INSTALL_DIR))
        self.check("uninstall removed the autostart Run value",
                   registry_value(winreg.HKEY_CURRENT_USER, RUN_KEY, RUN_VALUE) is None)
        self.check("uninstall removed the Add/Remove Programs entry", not uninstall_entries())
        self.check("uninstall kept the settings and data folders",
                   SETTINGS_DIR.exists() == had_settings and USER_DATA_DIR.exists() == had_data,
                   f"{SETTINGS_DIR} {SETTINGS_DIR.exists()}, "
                   f"{USER_DATA_DIR} {USER_DATA_DIR.exists()}")

    REMOVE_DATA_CHECKS = (
        "uninstall with /REMOVEDATA=1 removed the install folder",
        "uninstall with /REMOVEDATA=1 removed the settings and data folders",
    )

    def _uninstall_removing_data(self) -> None:
        if INSTALL_DIR.exists():
            self.skip("a second silent install exits cleanly",
                      "the first install is still on disk")
            for name in self.REMOVE_DATA_CHECKS:
                self.skip(name, "the first install could not be removed")
            return
        code = self.install()
        self.check("a second silent install exits cleanly", code == 0, f"exit code {code}")
        blocked = self.uninstall("/REMOVEDATA=1")
        if blocked:
            for name in self.REMOVE_DATA_CHECKS:
                self.skip(name, blocked)
            return
        self.check("uninstall with /REMOVEDATA=1 removed the install folder",
                   not INSTALL_DIR.exists())
        self.check("uninstall with /REMOVEDATA=1 removed the settings and data folders",
                   not SETTINGS_DIR.exists() and not USER_DATA_DIR.exists(),
                   f"{SETTINGS_DIR} {SETTINGS_DIR.exists()}, "
                   f"{USER_DATA_DIR} {USER_DATA_DIR.exists()}")

    def _cleanup(self) -> None:
        print("\n  putting the machine back", flush=True)
        self.quit_app()
        if INSTALL_DIR.exists():
            blocked = self.uninstall()
            if blocked:
                print(f"  the uninstaller could not run ({blocked}); removing the folder by hand",
                      flush=True)
        if INSTALL_DIR.exists():
            wipe(INSTALL_DIR)
        if START_MENU_SHORTCUT.exists():
            START_MENU_SHORTCUT.unlink(missing_ok=True)
            print(f"  removed {START_MENU_SHORTCUT}", flush=True)
        for name in list(uninstall_entries()):
            try:
                winreg.DeleteKey(winreg.HKEY_CURRENT_USER, f"{UNINSTALL_KEY}\\{name}")
                print(f"  removed the leftover uninstall entry {name}", flush=True)
            except OSError:
                pass
        if registry_value(winreg.HKEY_CURRENT_USER, RUN_KEY, RUN_VALUE) is not None:
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as k:
                    winreg.DeleteValue(k, RUN_VALUE)
                print("  removed the leftover Run value", flush=True)
            except OSError:
                pass
        self.restore_existing_data()
        self.check("the machine is as it was: no install folder", not INSTALL_DIR.exists())
        self.check("the machine is as it was: no Start menu shortcut",
                   not START_MENU_SHORTCUT.exists())
        self.check("the machine is as it was: no uninstall entry", not uninstall_entries())
        self.check("the machine is as it was: no Run value",
                   registry_value(winreg.HKEY_CURRENT_USER, RUN_KEY, RUN_VALUE) is None)
        self.check("the machine is as it was: the user data folders",
                   SETTINGS_DIR.exists() == ("settings" in self.saved)
                   and USER_DATA_DIR.exists() == ("data" in self.saved),
                   f"{SETTINGS_DIR} {SETTINGS_DIR.exists()}, "
                   f"{USER_DATA_DIR} {USER_DATA_DIR.exists()}")


def setup_process_names(installer: Path) -> tuple[str, ...]:
    """What an Inno Setup install shows in the task list: the loader and its %TEMP% child."""
    return (installer.name, f"{installer.stem}.tmp", "setup.tmp")


def test_install_exit_code(counts: Mapping[str, int]) -> int:
    """1 when a spec 20.5 check failed, 2 when one was skipped, 0 only for a clean sweep.

    A failure outranks a skip: a build that broke something is worse news than one nobody
    could finish checking. Neither reads as verified.
    """
    if counts.get(FAIL):
        return EXIT_FAILED
    if counts.get(SKIP):
        return EXIT_SKIPPED
    return EXIT_OK


def newest_installer() -> Path:
    found = sorted(DIST_DIR.glob(f"{APP_NAME}-Setup-*.exe"), key=lambda p: p.stat().st_mtime)
    if not found:
        raise PackageError(f"no {APP_NAME}-Setup-*.exe in {DIST_DIR}; build one first")
    return found[-1]


# ----------------------------------------------------------------------------- cli


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--all", action="store_true", help="run every step, 4 to 8")
    parser.add_argument("--step", action="append", choices=STEPS, default=None,
                        help="run one step; repeatable, in the order given")
    parser.add_argument("--cleanup-model", type=Path, default=None,
                        help="ship this GGUF instead of the pinned one and mark the build -dev")
    parser.add_argument("--test-install", action="store_true",
                        help="run the spec 20.5 installer tests against the built installer")
    parser.add_argument("--installer", type=Path, default=None,
                        help="the installer --test-install should use (default: the newest)")
    parser.add_argument("--signtool", default=None,
                        help=f"sign Spells.exe, the installer and the uninstaller with this "
                             f"command, whose $f is the file to sign (default: none, or the "
                             f"{SIGNTOOL_ENV} environment variable)")
    parser.add_argument("--span", action="store_true",
                        help="force a spanning installer build (Setup.exe plus .bin slices, "
                             "zipped into dist/Spells-Setup-<version>.zip); spanning also "
                             "switches on by itself once the estimated payload needs it")
    parser.add_argument("--online", action="store_true",
                        help="build dist/Spells-Online-Setup-<version>.exe instead: the app and "
                             "the engines, with the models downloaded during the installation")
    parser.add_argument("--update-source", default=None,
                        help="the base address where this release is served, for example "
                             "https://spells.example.com/. It is fixed into the build: the "
                             "app checks <base>/latest.json and nothing else, and the build "
                             f"writes that version file into dist/ (default: none, or the "
                             f"{UPDATE_SOURCE_ENV} environment variable)")
    parser.add_argument("--minimum-version", default="",
                        help="the oldest version that can install this one directly; an older "
                             "installation is told which version to install first "
                             "(default: empty, meaning any version can update to this one)")
    parser.add_argument("--model-mirror", default="",
                        help="build the online installer to download the models from this base "
                             "url, a folder serving the same file names, with the pinned public "
                             "addresses as the silent fallback; the wizard never shows either "
                             "and the installed program cannot change them (default: empty, "
                             "which means the pinned addresses alone)")
    args = parser.parse_args(argv)

    if args.online and args.all:
        parser.error("--all and --online build different artifacts; run them one after the other")
    if args.online and args.cleanup_model:
        parser.error("--online downloads the pinned models by url and hash, so it has no place "
                     "for --cleanup-model; use --all for a dev build")
    if args.model_mirror and not args.online:
        parser.error("--model-mirror only means something for --online")

    update_source = args.update_source
    if update_source is None:
        update_source = os.environ.get(UPDATE_SOURCE_ENV, "")
    if update_source:
        try:
            update_source = check_update_source(update_source)
        except PackageError as exc:
            parser.error(str(exc))
    if args.minimum_version and not update_source:
        parser.error("--minimum-version belongs to the version file, so it needs "
                     "--update-source")

    steps = list(STEPS) if (args.all or (args.online and not args.step)) else list(args.step or ())
    if not steps and not args.test_install:
        parser.error("give --all, --online, at least one --step, or --test-install")

    started = time.perf_counter()
    signtool = signtool_command(args.signtool)
    try:
        version = None
        installer = None
        if steps:
            base = project_version(PYPROJECT_PATH.read_text(encoding="utf-8"))
            version = build_version(base, dev=args.cleanup_model is not None)
            print(f"Spells {version} (pyproject {base})", flush=True)
            manifest: tuple[ModelDownload, ...] = ()
            rows: tuple[InstallSet, ...] = ()
            if "tests" in steps:
                step_tests()
            if "app" in steps:
                step_app()
            if "dist" in steps:
                if args.online:
                    _dist, manifest, rows = step_dist_online(update_source)
                else:
                    step_dist(args.cleanup_model, update_source)
            if "installer" in steps:
                if args.online:
                    rows = rows or install_sets()
                    manifest = manifest or online_manifest(rows=rows)
                    installer = step_installer_online(version, manifest, rows, signtool,
                                                      mirror=args.model_mirror)
                else:
                    installer = step_installer(version, signtool, span=args.span)
            if "size" in steps:
                if args.online:
                    step_size_online(
                        installer or DIST_DIR / f"{online_installer_basename(version)}.exe",
                        rows or install_sets())
                else:
                    step_size(installer or DIST_DIR / f"{installer_basename(version)}.exe")
            if "latest" in steps:
                basename = (online_installer_basename(version) if args.online
                            else installer_basename(version))
                step_latest(version, installer or DIST_DIR / f"{basename}.exe", update_source,
                            args.minimum_version)
            print(f"\nbuild took {time.perf_counter() - started:.0f} s", flush=True)
        if args.test_install:
            target = args.installer or installer or newest_installer()
            print(f"\ntesting {target} ({format_size(target.stat().st_size)})", flush=True)
            tests = InstallerTests(target)
            tests.run()
            return test_install_exit_code(tests.counts())
    except (PackageError, fetch.PinError, licenses_mod.LicenseError,
            toolchain.BootstrapError, packaging.PackagingError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_FAILED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
