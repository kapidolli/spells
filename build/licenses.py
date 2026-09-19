"""Assemble and verify the third-party licence texts the Spells installer ships (spec 21).

Usage:
    py build/licenses.py verify
    py build/licenses.py assemble <dest>

build/licenses/MANIFEST.json lists every component spec 21 requires (plus a few the build
actually ships that spec 21 groups under a wider name): its name, version, licence name, a
source (a URL or a path inside this repo, recorded so a text can be re-fetched later), the
text file name (or null for a component that ships no file, such as the Vulkan loader,
which is a system component installed by the GPU driver), that file's sha256, a `ships`
flag, and a free-form note explaining anything not obvious from the fields above (why a
version is approximate, why two components share one text, and so on).

build/licenses/<component>.txt holds the text itself, fetched from the component's
authoritative source at the pinned version where build/pins.json pins one, or copied from a
local copy already in the repository (the whisper.cpp source tree, the llama.cpp release
zips, or the MSYS2 toolchain) when one exists.

Stdlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
LICENSES_DIR = REPO_DIR / "build" / "licenses"
MANIFEST_NAME = "MANIFEST.json"
INDEX_NAME = "index.txt"


class LicenseError(Exception):
    """A licences folder problem: missing MANIFEST.json, malformed JSON, or a bad text file."""


def load_manifest(folder: Path | None = None) -> dict:
    folder = LICENSES_DIR if folder is None else folder
    with open(folder / MANIFEST_NAME, encoding="utf-8") as fh:
        return json.load(fh)


def components(folder: Path | None = None) -> list[dict]:
    """Every component recorded in MANIFEST.json, in file order."""
    return load_manifest(folder)["components"]


def models_covered(folder: Path | None = None) -> dict[str, str]:
    """Map each data/models/catalog.json model id to the component name that licenses it.

    A component names the catalog ids it covers in an optional "models" list. Only a shipped
    model needs to appear here for a release to be sound; a benchmark candidate's entry may
    list its models too so the same check can run against the whole catalog.
    """
    covered: dict[str, str] = {}
    for entry in components(folder):
        for model_id in entry.get("models", []):
            covered[model_id] = entry["name"]
    return covered


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(folder: Path | None = None) -> list[str]:
    """Check every component's text file exists and matches its recorded sha256.

    Returns a list of human-readable problems; an empty list means the folder is sound. A
    component whose "file" is null (a system component that ships no text of its own) is
    skipped. A missing or unparsable MANIFEST.json is itself reported as a single problem
    rather than raising, so callers can treat verify() as the one place that inspects the
    folder's health.
    """
    folder = LICENSES_DIR if folder is None else folder
    try:
        entries = components(folder)
    except FileNotFoundError:
        return [f"{folder / MANIFEST_NAME} does not exist"]
    except json.JSONDecodeError as exc:
        return [f"{folder / MANIFEST_NAME} is not valid JSON: {exc}"]

    problems: list[str] = []
    for entry in entries:
        name = entry.get("name", "<unnamed component>")
        filename = entry.get("file")
        if filename is None:
            continue
        path = folder / filename
        if not path.exists():
            problems.append(f"{name}: {path} is missing")
            continue
        actual = _sha256(path)
        expected = entry.get("sha256")
        if actual != expected:
            problems.append(f"{name}: {path} sha256 is {actual}, manifest says {expected}")
    return problems


def assemble(dest: Path, folder: Path | None = None) -> Path:
    """Copy the licences folder into a dist tree and write dest/index.txt.

    Every component's text file is copied (spec 21: "all texts live in the repo's
    licenses/, ship in licenses\\"), including a benchmark candidate not yet marked as
    shipping, so a licence stays available the moment the benchmark picks it. index.txt
    lists only the components that actually ship in this build, one line per component,
    tab-separated: name, version, licence name, source.

    Raises LicenseError if verify() finds a problem first, so a corrupt or tampered folder
    is never copied into a release.
    """
    folder = LICENSES_DIR if folder is None else folder
    problems = verify(folder)
    if problems:
        raise LicenseError("; ".join(problems))

    dest.mkdir(parents=True, exist_ok=True)
    lines = []
    for entry in components(folder):
        filename = entry.get("file")
        if filename is not None:
            shutil.copy2(folder / filename, dest / filename)
        if entry.get("ships"):
            lines.append(
                "\t".join(
                    [
                        entry["name"],
                        str(entry["version"]),
                        entry["license"],
                        entry.get("source") or "",
                    ]
                )
            )
    (dest / INDEX_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify", help="check every text file against MANIFEST.json")
    assemble_cmd = sub.add_parser("assemble", help="copy the licences folder into dest")
    assemble_cmd.add_argument("dest", type=Path)
    args = parser.parse_args(argv)

    if args.command == "verify":
        problems = verify()
        if problems:
            for problem in problems:
                print(f"ERROR: {problem}", file=sys.stderr)
            return 1
        print(f"{len(components())} components OK")
        return 0

    try:
        dest = assemble(args.dest)
    except LicenseError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    shipped = sum(1 for entry in components() if entry.get("ships"))
    print(f"assembled {shipped} shipping licence texts (of {len(components())} recorded) into {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
