"""Version checks and the one-click installer download (spec 2 G3, 3, 17, 19.7).

Spells is an offline app and stays one. Nothing in this module runs by itself: every
function here has to be called, by the About page's button or by the weekly check the user
switched on, and a build that was not given an update address checks nothing at all
(``source_url`` returns an empty string and the About page says so).

The split is deliberate. Everything above ``fetch_release`` is pure: version comparison,
parsing and validating the version file, deciding whether a release applies, and the two
timers. Only the three workers at the bottom touch the network or the disk, and each takes
its opener, its clock and its runner as arguments so the tests never reach either.

The version file is JSON the owner serves beside the installers, ``latest.json``, written
by ``build/package.py`` from CHANGELOG.md and the built installer so the two cannot drift.
Its address is fixed when the build is made (``data/update-source.json`` inside the install
tree, written by the build and read here): the installed program offers no setting, no
field and no environment variable that redirects it, and the installer named in the file
has to come from the same host as the file itself, so a version file that was tampered with
cannot send the download anywhere else.

An update downloads exactly one file, the installer. Model weights are never fetched here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from spells import __version__

log = logging.getLogger(__name__)

SOURCE_FILE = "update-source.json"
MANIFEST_NAME = "latest.json"
MANIFEST_SCHEMA = 1
PRODUCT = "Spells"

FETCH_TIMEOUT_S = 6.0
DOWNLOAD_TIMEOUT_S = 30.0
MAX_MANIFEST_BYTES = 64 * 1024
MAX_CHANGES = 60
MAX_CHANGE_CHARS = 300
MAX_INSTALLER_BYTES = 4_000_000_000
MIN_INSTALLER_BYTES = 1024
CHUNK_BYTES = 256 * 1024

WEEK_S = 7 * 24 * 60 * 60
DAY_S = 24 * 60 * 60

ALLOWED_SCHEMES = ("https", "http")

INSTALL_ARGS = ("/SILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/RELAUNCH")

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200

USER_AGENT = f"Spells/{__version__}"

_VERSION_RE = re.compile(r"^v?(\d{1,6}(?:\.\d{1,6}){0,3})(?:[-+]([0-9A-Za-z.\-+]{1,32}))?$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_FILE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}\.exe$")


class UpdateError(Exception):
    """Anything that stops a check, a download or a launch. Always carries a plain reason."""


class UpdateCancelled(UpdateError):
    """The user pressed Cancel during the download."""


@dataclass(frozen=True, order=True)
class Version:
    """A comparable version. ``pre`` is a suffix such as ``dev``; a release beats its own pre."""

    parts: tuple[int, ...]
    release: bool = True
    pre: str = ""


@dataclass(frozen=True)
class Release:
    """One row of the version file: what it is, where it is, and what changed."""

    version: str
    released: str = ""
    url: str = ""
    size_bytes: int = 0
    sha256: str = ""
    changes: tuple[str, ...] = ()
    minimum_version: str = ""


@dataclass(frozen=True)
class Decision:
    """Whether a release applies to the version running here, and why."""

    release: Release | None = None
    available: bool = False
    blocked: bool = False
    reason: str = ""


def parse_version(text: object) -> Version | None:
    """A comparable Version, or None for anything this app cannot reason about.

    Accepts ``0.2.0``, ``v0.2.0``, ``1.2``, ``1.2.3.4`` and a suffix such as ``0.2.0-dev``.
    Refuses an empty string, a non-string, letters where numbers belong, a negative number,
    more than four parts and a part of more than six digits. Nothing here raises.
    """
    if not isinstance(text, str):
        return None
    match = _VERSION_RE.match(text.strip())
    if match is None:
        return None
    parts = tuple(int(piece) for piece in match.group(1).split("."))
    parts = parts + (0,) * (4 - len(parts))
    pre = (match.group(2) or "").lower()
    return Version(parts=parts, release=not pre, pre=pre)


def compare_versions(left: object, right: object) -> int:
    """-1, 0 or 1. An unreadable version raises, so a caller has to say what it wants."""
    first = parse_version(left)
    second = parse_version(right)
    if first is None or second is None:
        raise UpdateError(f"{left!r} and {right!r} are not both versions this app can read")
    if first == second:
        return 0
    return -1 if first < second else 1


def is_newer(candidate: object, current: object) -> bool:
    """True only when both versions read cleanly and the candidate is the higher one.

    An unreadable version on either side answers False: an update this app cannot reason
    about is never offered.
    """
    try:
        return compare_versions(candidate, current) > 0
    except UpdateError:
        return False


def _clean_line(raw: object, where: str) -> str:
    if not isinstance(raw, str):
        raise UpdateError(f"{where}: expected a line of text")
    text = _CONTROL_RE.sub(" ", raw).strip()
    if not text:
        raise UpdateError(f"{where}: an empty line")
    return text[:MAX_CHANGE_CHARS].strip()


def _same_host(url: str, source: str) -> bool:
    first = urlsplit(url)
    second = urlsplit(source)
    return (first.scheme, first.hostname, first.port) == (second.scheme, second.hostname, second.port)


def check_url(url: object, where: str = "url") -> str:
    """An http or https address with a host, or UpdateError. No other scheme is ever used."""
    if not isinstance(url, str) or not url.strip():
        raise UpdateError(f"{where}: expected an address")
    text = url.strip()
    parts = urlsplit(text)
    if parts.scheme not in ALLOWED_SCHEMES:
        raise UpdateError(f"{where}: {parts.scheme or 'no'} is not an address Spells will open")
    if not parts.hostname:
        raise UpdateError(f"{where}: the address names no server")
    return text


def installer_name(url: str) -> str:
    """The file name to save the installer under, taken from the address and checked.

    Anything with a path separator, a parent reference, a name that is not a plain ``.exe``
    or a name longer than the pattern allows falls back to a fixed name, so a version file
    can never choose where on the disk the download lands.
    """
    tail = urlsplit(url).path.rsplit("/", 1)[-1]
    if _FILE_NAME_RE.match(tail):
        return tail
    return "Spells-Setup.exe"


def parse_manifest(raw: object, *, source_url: str = "") -> Release:
    """Read and validate the version file. Every failure is an UpdateError with a reason.

    The file is treated as hostile: the schema has to be one this build knows, the product
    has to be Spells, the version and the minimum version have to parse, the size has to be
    a plausible installer, the hash has to be 64 hex characters, and the installer's address
    has to be an http or https address on the same host as the version file itself. Change
    lines are stripped of control characters, trimmed and capped in number and in length.
    """
    if isinstance(raw, (bytes, bytearray)):
        if len(raw) > MAX_MANIFEST_BYTES:
            raise UpdateError("the version file is far larger than a version file should be")
        try:
            raw = bytes(raw).decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise UpdateError("the version file is not readable text") from exc
    if not isinstance(raw, str):
        raise UpdateError("the version file is not readable text")
    if len(raw) > MAX_MANIFEST_BYTES:
        raise UpdateError("the version file is far larger than a version file should be")
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise UpdateError(f"the version file is not valid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise UpdateError("the version file is not an object")

    schema = data.get("schema", MANIFEST_SCHEMA)
    if not isinstance(schema, int) or isinstance(schema, bool) or schema < 1:
        raise UpdateError("schema: expected a whole number")
    if schema > MANIFEST_SCHEMA:
        raise UpdateError(
            f"the version file is written for a newer Spells (schema {schema}); "
            "install the newest version by hand once"
        )

    product = data.get("product", PRODUCT)
    if product != PRODUCT:
        raise UpdateError(f"the version file is for {product!r}, not for Spells")

    version = data.get("version")
    if parse_version(version) is None:
        raise UpdateError(f"version: {version!r} is not a version this app can read")
    assert isinstance(version, str)

    released = data.get("released", "")
    if released in (None, ""):
        released = ""
    elif not isinstance(released, str) or not _DATE_RE.match(released.strip()):
        raise UpdateError("released: expected a date written as YYYY-MM-DD")
    else:
        released = released.strip()

    minimum = data.get("minimum_version", "")
    if minimum in (None, ""):
        minimum = ""
    elif parse_version(minimum) is None:
        raise UpdateError(f"minimum_version: {minimum!r} is not a version this app can read")
    else:
        assert isinstance(minimum, str)
        minimum = minimum.strip()

    installer = data.get("installer")
    if not isinstance(installer, dict):
        raise UpdateError("installer: expected an object with the url, the size and the hash")
    url = check_url(installer.get("url"), "installer.url")
    if source_url and not _same_host(url, source_url):
        raise UpdateError(
            "installer.url: the installer is on another server than the version file, "
            "so Spells will not download it"
        )
    size = installer.get("size_bytes")
    if not isinstance(size, int) or isinstance(size, bool):
        raise UpdateError("installer.size_bytes: expected a whole number of bytes")
    if not MIN_INSTALLER_BYTES <= size <= MAX_INSTALLER_BYTES:
        raise UpdateError(f"installer.size_bytes: {size} is not the size of an installer")
    digest = installer.get("sha256")
    if not isinstance(digest, str) or not _SHA256_RE.match(digest.strip().lower()):
        raise UpdateError("installer.sha256: expected 64 hexadecimal characters")

    raw_changes = data.get("changes", [])
    if raw_changes in (None, ""):
        raw_changes = []
    if not isinstance(raw_changes, list):
        raise UpdateError("changes: expected a list of lines")
    if len(raw_changes) > MAX_CHANGES:
        raise UpdateError(f"changes: more than {MAX_CHANGES} lines")
    changes = tuple(
        _clean_line(line, f"changes[{index}]") for index, line in enumerate(raw_changes)
    )

    return Release(
        version=version.strip(),
        released=released,
        url=url,
        size_bytes=size,
        sha256=digest.strip().lower(),
        changes=changes,
        minimum_version=minimum,
    )


def decide(release: Release | None, current: str = __version__) -> Decision:
    """Whether this release applies to the version running here.

    An older or equal version is not an update. A release whose ``minimum_version`` is newer
    than the installed one is an update the user cannot take in one step, so it comes back
    blocked, with the version to install first named in the reason.
    """
    if release is None:
        return Decision(reason="No version file was read.")
    if not is_newer(release.version, current):
        return Decision(release=release, reason=f"Spells {current} is the newest version.")
    if release.minimum_version and is_newer(release.minimum_version, current):
        return Decision(
            release=release,
            available=True,
            blocked=True,
            reason=(
                f"Spells {release.version} can only be installed over "
                f"{release.minimum_version} or newer. Install {release.minimum_version} "
                "first, then check again."
            ),
        )
    return Decision(release=release, available=True)


def check_due(*, enabled: bool, now: float, last_check: float) -> bool:
    """The weekly timer. Off means never, and a clock that moved backwards checks once."""
    if not enabled:
        return False
    if last_check <= 0:
        return True
    if last_check > now:
        return True
    return now - last_check >= WEEK_S


def balloon_due(*, now: float, last_at: float, version: str, last_version: str) -> bool:
    """At most one balloon a day, and one straight away for a version not shown before."""
    if not version:
        return False
    if version != last_version:
        return True
    if last_at <= 0 or last_at > now:
        return True
    return now - last_at >= DAY_S


def source_url(data_dir: Path | str) -> str:
    """The address this build checks, fixed when it was built. Empty when there is none.

    ``build/package.py`` writes ``data/update-source.json`` into the install tree from its
    ``--update-source`` argument. A development checkout has no such file, so it checks
    nothing. Nothing here raises: an unreadable, malformed or non-http entry reads as
    "this build has no update address".
    """
    path = Path(data_dir) / SOURCE_FILE
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return ""
    except Exception:
        log.warning("The update source file %s could not be read", path, exc_info=True)
        return ""
    if not isinstance(raw, dict):
        return ""
    try:
        return check_url(raw.get("manifest_url"), "manifest_url")
    except UpdateError:
        log.warning("The update source file %s names no usable address", path)
        return ""


def _default_opener(url: str, timeout: float):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(request, timeout=timeout)


def fetch_release(
    url: str,
    *,
    opener: Callable[[str, float], object] | None = None,
    timeout_s: float = FETCH_TIMEOUT_S,
) -> Release:
    """Fetch and validate the version file. Only called when the user asked for a check.

    Times out quickly and reads at most ``MAX_MANIFEST_BYTES``, so a server that hangs or
    answers with a huge body costs a few seconds and nothing else. Every failure, network or
    content, comes back as an UpdateError with a sentence the About page can show.
    """
    address = check_url(url, "update address")
    read = opener or _default_opener
    try:
        response = read(address, timeout_s)
    except UpdateError:
        raise
    except Exception as exc:
        raise UpdateError(f"Spells could not reach the update server ({exc}).") from exc
    try:
        body = response.read(MAX_MANIFEST_BYTES + 1)
    except Exception as exc:
        raise UpdateError(f"The version file could not be read ({exc}).") from exc
    finally:
        _close(response)
    return parse_manifest(body, source_url=address)


def _close(response: object) -> None:
    closer = getattr(response, "close", None)
    if closer is None:
        return
    try:
        closer()
    except Exception:
        log.debug("closing the response failed", exc_info=True)


def download_installer(
    release: Release,
    directory: Path | str,
    *,
    opener: Callable[[str, float], object] | None = None,
    timeout_s: float = DOWNLOAD_TIMEOUT_S,
    progress: Callable[[int, int], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> Path:
    """Download the installer into a folder of the caller's choosing and verify it.

    The file is hashed while it streams. A size that grows past the one the version file
    promised, a size that falls short of it, a hash that does not match, or a cancel from
    the user all delete the partial file before raising, so nothing unverified is ever left
    on the disk for anybody to run.
    """
    check_url(release.url, "installer.url")
    if not _SHA256_RE.match(release.sha256):
        raise UpdateError("The update has no usable checksum, so Spells will not download it.")
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    path = target / installer_name(release.url)
    read = opener or _default_opener
    digest = hashlib.sha256()
    done = 0
    try:
        response = read(release.url, timeout_s)
    except Exception as exc:
        raise UpdateError(f"Spells could not reach the update server ({exc}).") from exc
    try:
        with open(path, "wb") as handle:
            while True:
                if cancel is not None and cancel():
                    raise UpdateCancelled("The download was cancelled.")
                try:
                    chunk = response.read(CHUNK_BYTES)
                except Exception as exc:
                    raise UpdateError(f"The download stopped ({exc}).") from exc
                if not chunk:
                    break
                done += len(chunk)
                if done > release.size_bytes:
                    raise UpdateError("The download is larger than the version file promised.")
                handle.write(chunk)
                digest.update(chunk)
                if progress is not None:
                    progress(done, release.size_bytes)
        if done != release.size_bytes:
            raise UpdateError("The download ended early, so Spells will not run it.")
        if digest.hexdigest() != release.sha256:
            raise UpdateError(
                "The downloaded installer does not match its checksum, so Spells deleted it "
                "and will not run it."
            )
    except BaseException:
        _unlink(path)
        raise
    finally:
        _close(response)
    return path


def _unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        log.warning("The unverified download %s could not be removed", path, exc_info=True)


def verify_file(path: Path | str, release: Release) -> bool:
    """True when the file on disk is exactly the installer the version file describes."""
    file = Path(path)
    try:
        if file.stat().st_size != release.size_bytes:
            return False
        digest = hashlib.sha256()
        with open(file, "rb") as handle:
            while True:
                chunk = handle.read(CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError:
        return False
    return digest.hexdigest() == release.sha256


def launch_installer(
    path: Path | str,
    *,
    args: Sequence[str] = INSTALL_ARGS,
    runner: Callable[[Sequence[str]], object] | None = None,
) -> None:
    """Start the verified installer detached and return, so the app can quit behind it.

    Setup closes the running Spells itself through the ``--quit`` step of spec 19.5, but the
    app quits anyway: the files it holds open are the ones Setup replaces.
    """
    file = Path(path)
    if file.suffix.lower() != ".exe" or not file.is_file():
        raise UpdateError("The installer is not where Spells left it.")
    command = [str(file), *args]
    start = runner or _default_runner
    try:
        start(command)
    except Exception as exc:
        raise UpdateError(f"The installer could not be started ({exc}).") from exc


def _default_runner(command: Sequence[str]) -> object:
    flags = 0
    if sys.platform == "win32":
        flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen(list(command), close_fds=True, creationflags=flags)


def size_text(size: int) -> str:
    """A size the way the rest of the app writes one."""
    if size >= 1_000_000_000:
        return f"{size / 1_000_000_000:.2f} GB"
    if size >= 1_000_000:
        return f"{size / 1_000_000:.1f} MB"
    if size >= 1_000:
        return f"{size / 1_000:.0f} kB"
    return f"{size} bytes"


def date_text(released: str) -> str:
    """``2026-09-18`` as ``18 September 2026``; anything else comes back unchanged."""
    if not _DATE_RE.match(released or ""):
        return released or ""
    try:
        stamp = time.strptime(released, "%Y-%m-%d")
    except ValueError:
        return released
    return f"{stamp.tm_mday} {_MONTHS[stamp.tm_mon - 1]} {stamp.tm_year}"


_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


__all__ = [
    "DAY_S",
    "INSTALL_ARGS",
    "MANIFEST_NAME",
    "MANIFEST_SCHEMA",
    "SOURCE_FILE",
    "WEEK_S",
    "Decision",
    "Release",
    "UpdateCancelled",
    "UpdateError",
    "Version",
    "balloon_due",
    "check_due",
    "check_url",
    "compare_versions",
    "date_text",
    "decide",
    "download_installer",
    "fetch_release",
    "installer_name",
    "is_newer",
    "launch_installer",
    "parse_manifest",
    "parse_version",
    "size_text",
    "source_url",
    "verify_file",
]
