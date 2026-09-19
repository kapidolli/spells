"""Download pinned inputs into build/cache and verify their SHA256 (spec 19.1).

Usage:
    py build/fetch.py <group.name> [<group.name> ...]
    py build/fetch.py --all
    py build/fetch.py <group.name> --record

Keys look like ``sources.whisper_cpp`` or ``models.whisper_tiny_test``. Entries in the
``models`` group (and ``bench_candidates``) land in build/cache/models/, everything else in
build/cache/. An entry whose pinned sha256 is empty is refused unless ``--record`` is passed,
in which case the file is downloaded, hashed, and the hash is written back into pins.json.

Stdlib only. Streams to disk, prints progress, never resumes a partial download.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
PINS_PATH = REPO_DIR / "build" / "pins.json"
CACHE_DIR = REPO_DIR / "build" / "cache"

DOWNLOAD_GROUPS = ("toolchain", "sources", "models", "binaries")
MODEL_GROUPS = ("models", "bench_candidates")
CHUNK_SIZE = 1 << 20
UNSAFE_NAME_CHARS = ("/", "\\", ":")
USER_AGENT = "spells-build/0.1 (+https://github.com/ggml-org/whisper.cpp)"


class PinError(Exception):
    """A pins.json problem: unknown key, malformed entry, or a hash that does not match."""


class UnpinnedHash(PinError):
    """The entry has no sha256 yet and --record was not given."""


class HashMismatch(PinError):
    """The downloaded or cached file does not match the pinned sha256."""


# ----------------------------------------------------------------------------- pins access


def load_pins(pins_path: Path = PINS_PATH) -> dict:
    with open(pins_path, encoding="utf-8") as fh:
        return json.load(fh)


def save_pins(pins: dict, pins_path: Path = PINS_PATH) -> None:
    tmp = pins_path.with_suffix(pins_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(pins, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, pins_path)


def resolve(pins: dict, key: str) -> dict:
    """Return the entry for ``group.name``; bench_candidates are addressed by their ``name``."""
    group, _, name = key.partition(".")
    if not name:
        raise PinError(f"pin key must look like group.name, got {key!r}")
    members = pins.get(group)
    if isinstance(members, dict):
        entry = members.get(name)
    elif isinstance(members, list):
        entry = next((item for item in members if item.get("name") == name), None)
    else:
        entry = None
    if not isinstance(entry, dict) or "url" not in entry:
        raise PinError(f"no download entry {key!r} in pins.json")
    return entry


def iter_download_keys(pins: dict):
    for group in DOWNLOAD_GROUPS:
        for name, entry in pins.get(group, {}).items():
            if isinstance(entry, dict) and "url" in entry:
                yield f"{group}.{name}"


def filename_for(entry: dict) -> str:
    explicit = entry.get("filename")
    if explicit:
        return explicit
    return urllib.parse.urlparse(entry["url"]).path.rsplit("/", 1)[-1]


def is_plain_file_name(name: str) -> bool:
    """True for a name that can only ever land directly inside the cache directory.

    The name comes from pins.json or from the pinned URL, so a separator, a "..", or a drive
    letter in it would write outside build/cache. The rule is spelled out rather than left to
    Path(name).name, whose handling of ".." differs between Python versions.
    """
    return bool(name) and name not in (".", "..") and not any(c in name for c in UNSAFE_NAME_CHARS)


def cache_path(key: str, entry: dict, cache_dir: Path = CACHE_DIR) -> Path:
    """Where this entry is cached; refuses a name that could escape build/cache."""
    name = filename_for(entry)
    if not is_plain_file_name(name):
        raise PinError(
            f"{key}: refusing the file name {name!r}. A pinned entry must name a plain file "
            f"inside build/cache: no path separators, no '..', no drive letter."
        )
    group = key.partition(".")[0]
    base = cache_dir / "models" if group in MODEL_GROUPS else cache_dir
    return base / name


# ----------------------------------------------------------------------------- hashing and download


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _format_size(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024 or unit == "GB":
            return f"{num:.1f} {unit}" if unit != "B" else f"{int(num)} B"
        num /= 1024
    return f"{num:.1f} GB"


def download(url: str, dest: Path, *, progress=None) -> str:
    """Stream ``url`` into ``dest`` and return its sha256. Writes to ``dest.part`` first."""
    progress = sys.stderr if progress is None else progress
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request) as response, open(part, "wb") as out:
            total = response.headers.get("Content-Length")
            total = int(total) if total and total.isdigit() else None
            done = 0
            last_report = 0.0
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                now = time.monotonic()
                if progress and (now - last_report >= 0.5 or done == total):
                    last_report = now
                    if total:
                        pct = 100.0 * done / total
                        progress.write(f"\r  {_format_size(done)} / {_format_size(total)} ({pct:5.1f}%)")
                    else:
                        progress.write(f"\r  {_format_size(done)}")
                    progress.flush()
        if progress:
            progress.write("\n")
            progress.flush()
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    os.replace(part, dest)
    return digest.hexdigest()


# ----------------------------------------------------------------------------- fetch


def fetch(key: str, *, pins_path: Path = PINS_PATH, cache_dir: Path = CACHE_DIR,
          record: bool = False) -> Path:
    """Ensure the pinned file for ``key`` is in the cache with a matching hash; return its path."""
    pins = load_pins(pins_path)
    entry = resolve(pins, key)
    pinned = (entry.get("sha256") or "").strip().lower()
    dest = cache_path(key, entry, cache_dir)

    if not pinned and not record:
        raise UnpinnedHash(
            f"{key}: pinned sha256 is empty. Rerun with --record to download it, compute the "
            f"hash, and write it into {pins_path.name}. Never do that for a release build."
        )

    if pinned and dest.exists():
        actual = sha256_file(dest)
        if actual == pinned:
            print(f"{key}: cached {dest} (sha256 ok)")
            return dest
        print(f"{key}: cached file has a different hash, downloading again", file=sys.stderr)
        dest.unlink()

    print(f"{key}: downloading {entry['url']}")
    print(f"  -> {dest}")
    actual = download(entry["url"], dest)

    if pinned:
        if actual != pinned:
            dest.unlink(missing_ok=True)
            raise HashMismatch(
                f"{key}: sha256 mismatch\n  pinned:   {pinned}\n  computed: {actual}\n"
                f"  The file was deleted. Check the URL or update pins.json deliberately."
            )
        print(f"{key}: sha256 ok")
        return dest

    # record mode for an unpinned entry: trust this download and write the hash back
    entry["sha256"] = actual
    save_pins(pins, pins_path)
    print(
        f"WARNING: {key} had no pinned sha256; recorded {actual} into {pins_path} from this "
        f"download. Verify it against an upstream checksum before relying on it.",
        file=sys.stderr,
    )
    return dest


# ----------------------------------------------------------------------------- cli


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("keys", nargs="*", help="pin keys such as sources.whisper_cpp")
    parser.add_argument("--all", action="store_true", help="fetch every download entry")
    parser.add_argument("--record", action="store_true",
                        help="for entries with an empty sha256: download and write the hash back")
    parser.add_argument("--pins", type=Path, default=PINS_PATH, help=argparse.SUPPRESS)
    parser.add_argument("--cache", type=Path, default=CACHE_DIR, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    keys = list(args.keys)
    if args.all:
        keys.extend(k for k in iter_download_keys(load_pins(args.pins)) if k not in keys)
    if not keys:
        parser.error("give at least one key or --all")

    failures = []
    for key in keys:
        try:
            fetch(key, pins_path=args.pins, cache_dir=args.cache, record=args.record)
        except PinError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            failures.append(key)
        except OSError as exc:
            print(f"ERROR: {key}: {exc}", file=sys.stderr)
            failures.append(key)
    if failures:
        print(f"{len(failures)} of {len(keys)} entries failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
