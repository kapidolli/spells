"""build/fetch.py: hash verification and --record behaviour, offline (file:// URLs only)."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

BUILD_DIR = Path(__file__).resolve().parents[2] / "build"
sys.path.insert(0, str(BUILD_DIR))

import fetch


@pytest.fixture
def payload(tmp_path):
    src = tmp_path / "source.bin"
    src.write_bytes(bytes(range(256)) * 4096)  # 1 MiB, spans several stream chunks
    return src, hashlib.sha256(src.read_bytes()).hexdigest()


def _write_pins(tmp_path, url, sha256, group="sources"):
    pins = {
        group: {
            "thing": {
                "url": url,
                "version": "1",
                "sha256": sha256,
                "notes": "test entry",
                "filename": "thing.bin",
            }
        }
    }
    pins_path = tmp_path / "pins.json"
    pins_path.write_text(json.dumps(pins), encoding="utf-8")
    return pins_path


def test_sha256_file_streams_and_matches_hashlib(payload):
    src, digest = payload
    assert fetch.sha256_file(src) == digest


def test_fetch_accepts_matching_hash(tmp_path, payload):
    src, digest = payload
    pins_path = _write_pins(tmp_path, src.as_uri(), digest)
    cache = tmp_path / "cache"

    out = fetch.fetch("sources.thing", pins_path=pins_path, cache_dir=cache)

    assert out == cache / "thing.bin"
    assert out.read_bytes() == src.read_bytes()
    assert not list(cache.glob("*.part")), "temporary download file must be renamed away"


def test_fetch_rejects_mismatching_hash_and_removes_file(tmp_path, payload):
    src, _ = payload
    pins_path = _write_pins(tmp_path, src.as_uri(), "0" * 64)
    cache = tmp_path / "cache"

    with pytest.raises(fetch.HashMismatch) as excinfo:
        fetch.fetch("sources.thing", pins_path=pins_path, cache_dir=cache)

    assert "sources.thing" in str(excinfo.value)
    assert not (cache / "thing.bin").exists()
    assert not list(cache.glob("*.part"))


def test_models_group_lands_in_models_subdirectory(tmp_path, payload):
    src, digest = payload
    pins_path = _write_pins(tmp_path, src.as_uri(), digest, group="models")
    cache = tmp_path / "cache"

    out = fetch.fetch("models.thing", pins_path=pins_path, cache_dir=cache)

    assert out == cache / "models" / "thing.bin"


def test_empty_hash_is_refused_without_record(tmp_path, payload):
    src, _ = payload
    pins_path = _write_pins(tmp_path, src.as_uri(), "")
    cache = tmp_path / "cache"

    with pytest.raises(fetch.UnpinnedHash) as excinfo:
        fetch.fetch("sources.thing", pins_path=pins_path, cache_dir=cache)

    assert "--record" in str(excinfo.value)
    assert not (cache / "thing.bin").exists(), "nothing may be downloaded for an unpinned entry"


def test_record_downloads_and_writes_hash_back(tmp_path, payload, capsys):
    src, digest = payload
    pins_path = _write_pins(tmp_path, src.as_uri(), "")
    cache = tmp_path / "cache"

    out = fetch.fetch("sources.thing", pins_path=pins_path, cache_dir=cache, record=True)

    assert out.read_bytes() == src.read_bytes()
    pins = json.loads(pins_path.read_text(encoding="utf-8"))
    assert pins["sources"]["thing"]["sha256"] == digest
    captured = capsys.readouterr()
    assert "WARNING" in captured.out + captured.err


def test_record_does_not_override_an_existing_pin(tmp_path, payload):
    src, _ = payload
    pins_path = _write_pins(tmp_path, src.as_uri(), "0" * 64)
    cache = tmp_path / "cache"

    with pytest.raises(fetch.HashMismatch):
        fetch.fetch("sources.thing", pins_path=pins_path, cache_dir=cache, record=True)

    pins = json.loads(pins_path.read_text(encoding="utf-8"))
    assert pins["sources"]["thing"]["sha256"] == "0" * 64


def test_cached_file_with_matching_hash_is_not_downloaded_again(tmp_path, payload):
    src, digest = payload
    pins_path = _write_pins(tmp_path, src.as_uri(), digest)
    cache = tmp_path / "cache"
    fetch.fetch("sources.thing", pins_path=pins_path, cache_dir=cache)
    src.unlink()  # the source is gone, so a second fetch can only succeed from the cache

    out = fetch.fetch("sources.thing", pins_path=pins_path, cache_dir=cache)

    assert out.exists()


def test_unknown_key_is_a_clear_error(tmp_path, payload):
    src, digest = payload
    pins_path = _write_pins(tmp_path, src.as_uri(), digest)

    with pytest.raises(fetch.PinError) as excinfo:
        fetch.fetch("sources.nope", pins_path=pins_path, cache_dir=tmp_path / "cache")

    assert "sources.nope" in str(excinfo.value)


def test_main_returns_nonzero_on_unpinned_entry(tmp_path, payload, capsys):
    src, _ = payload
    pins_path = _write_pins(tmp_path, src.as_uri(), "")

    rc = fetch.main(["sources.thing", "--pins", str(pins_path), "--cache", str(tmp_path / "cache")])

    assert rc != 0
    assert "--record" in capsys.readouterr().err


@pytest.mark.parametrize("filename", [
    "../escape.bin",
    r"..\escape.bin",
    "sub/dir.bin",
    r"sub\dir.bin",
    "/absolute.bin",
    "C:escape.bin",
    "..",
    ".",
])
def test_a_file_name_that_could_escape_the_cache_is_refused(tmp_path, payload, filename):
    src, digest = payload
    pins = {"sources": {"thing": {"url": src.as_uri(), "version": "1", "sha256": digest,
                                  "notes": "test entry", "filename": filename}}}
    pins_path = tmp_path / "pins.json"
    pins_path.write_text(json.dumps(pins), encoding="utf-8")
    cache = tmp_path / "cache"

    with pytest.raises(fetch.PinError) as excinfo:
        fetch.fetch("sources.thing", pins_path=pins_path, cache_dir=cache)

    assert "sources.thing" in str(excinfo.value)
    assert not list(tmp_path.rglob("escape.bin")), "nothing may be written outside the cache"


def test_a_plain_file_name_is_accepted(tmp_path, payload):
    src, digest = payload
    pins_path = _write_pins(tmp_path, src.as_uri(), digest)
    entry = fetch.resolve(fetch.load_pins(pins_path), "sources.thing")

    assert fetch.cache_path("sources.thing", entry, tmp_path / "cache") == \
        tmp_path / "cache" / "thing.bin"


def test_a_url_with_no_file_name_is_refused(tmp_path):
    pins = {"sources": {"thing": {"url": "https://example.invalid/downloads/", "version": "1",
                                  "sha256": "0" * 64, "notes": "test entry"}}}
    pins_path = tmp_path / "pins.json"
    pins_path.write_text(json.dumps(pins), encoding="utf-8")

    with pytest.raises(fetch.PinError):
        fetch.fetch("sources.thing", pins_path=pins_path, cache_dir=tmp_path / "cache")


def test_an_empty_filename_field_falls_back_to_the_url_name(tmp_path, payload):
    src, digest = payload
    pins = {"sources": {"thing": {"url": src.as_uri(), "version": "1", "sha256": digest,
                                  "notes": "test entry", "filename": ""}}}
    pins_path = tmp_path / "pins.json"
    pins_path.write_text(json.dumps(pins), encoding="utf-8")

    out = fetch.fetch("sources.thing", pins_path=pins_path, cache_dir=tmp_path / "cache")

    assert out.name == src.name
