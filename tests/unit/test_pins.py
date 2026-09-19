"""build/pins.json shape checks: every pinned input is well formed."""

import json
import re
from pathlib import Path

import pytest

PINS_PATH = Path(__file__).resolve().parents[2] / "build" / "pins.json"
REQUIRED_FIELDS = {"url", "version", "sha256", "notes"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DOWNLOAD_GROUPS = ("toolchain", "sources", "models", "binaries")
METADATA_KEYS = {"msys2_packages", "msys2_package_files"}


def _load():
    return json.loads(PINS_PATH.read_text(encoding="utf-8"))


def _entries():
    """Yield (key, entry) for every pinned download entry.

    toolchain.msys2_packages and toolchain.msys2_package_files are name-to-version and
    file-to-hash maps, not downloads, so they are skipped here and checked separately.
    bench_candidates is a list addressed by each item's name.
    """
    pins = _load()
    for group in DOWNLOAD_GROUPS:
        for name, entry in pins[group].items():
            if group == "toolchain" and name in METADATA_KEYS:
                continue
            yield f"{group}.{name}", entry
    for item in pins["bench_candidates"]:
        yield f"bench_candidates.{item.get('name', '?')}", item


def test_pins_parse_with_expected_groups():
    pins = _load()
    assert set(pins) == {*DOWNLOAD_GROUPS, "bench_candidates"}
    assert all(isinstance(pins[group], dict) and pins[group] for group in DOWNLOAD_GROUPS)
    assert isinstance(pins["bench_candidates"], list) and pins["bench_candidates"]


@pytest.mark.parametrize("key,entry", list(_entries()))
def test_entry_has_required_fields(key, entry):
    missing = REQUIRED_FIELDS - set(entry)
    assert not missing, f"{key} is missing {sorted(missing)}"
    for field in REQUIRED_FIELDS:
        assert isinstance(entry[field], str), f"{key}.{field} must be a string"
    assert entry["url"] and entry["version"] and entry["notes"], f"{key} has an empty field"


@pytest.mark.parametrize("key,entry", list(_entries()))
def test_non_empty_sha256_is_64_hex(key, entry):
    sha = entry["sha256"]
    if sha:
        assert SHA256_RE.match(sha), f"{key}.sha256 is not 64 lowercase hex chars: {sha!r}"


@pytest.mark.parametrize("key,entry", list(_entries()))
def test_url_scheme_is_https(key, entry):
    assert entry["url"].startswith("https://"), f"{key}.url must use https: {entry['url']}"


def test_bench_candidates_have_unique_names_and_no_downloads_yet():
    items = _load()["bench_candidates"]
    names = [item["name"] for item in items]
    assert len(names) == len(set(names)) == 4
    assert all(item["url"].endswith(".gguf") for item in items)


def test_msys2_packages_is_name_to_version_map():
    packages = _load()["toolchain"]["msys2_packages"]
    expected = {
        "mingw-w64-ucrt-x86_64-gcc",
        "mingw-w64-ucrt-x86_64-cmake",
        "mingw-w64-ucrt-x86_64-ninja",
        "mingw-w64-ucrt-x86_64-vulkan-devel",
        "mingw-w64-ucrt-x86_64-shaderc",
        "mingw-w64-ucrt-x86_64-spirv-headers",
    }
    assert set(packages) == expected
    assert all(isinstance(version, str) for version in packages.values())


def test_msys2_package_files_map_file_names_to_sha256():
    files = _load()["toolchain"]["msys2_package_files"]
    assert isinstance(files, dict)
    for name, sha in files.items():
        assert name.endswith((".pkg.tar.zst", ".pkg.tar.zst.sig", ".pkg.tar.xz", ".pkg.tar.xz.sig")), name
        assert SHA256_RE.match(sha), f"{name}: {sha!r}"


def test_inputs_needed_now_are_fully_pinned():
    pins = _load()
    for key in ("toolchain.msys2", "toolchain.cmake_kitware", "toolchain.inno_setup",
                "toolchain.innoextract", "toolchain.pyinstaller", "sources.whisper_cpp",
                "models.whisper_tiny_test", "models.whisper_large_v3_turbo_q8_0",
                "models.silero_vad", "binaries.llama_cpp_vulkan", "binaries.llama_cpp_cpu"):
        group, name = key.split(".")
        assert SHA256_RE.match(pins[group][name]["sha256"]), f"{key} must carry a real sha256"


def test_packaging_toolchain_is_pinned_the_way_bootstrap_packaging_reads_it():
    """The three packaging inputs of spec 19.3 steps 5 and 7, as build/bootstrap_packaging.py
    expects them: an Inno Setup 6 installer exe, an innoextract Windows zip to unpack it with,
    and the PyInstaller wheel that goes into .venv."""
    toolchain = _load()["toolchain"]

    assert toolchain["inno_setup"]["version"].startswith("6.")
    assert toolchain["inno_setup"]["url"].endswith(".exe")
    assert toolchain["innoextract"]["url"].endswith(".zip")
    assert toolchain["pyinstaller"]["version"].startswith("6.")
    assert toolchain["pyinstaller"]["url"].endswith(".whl")


def test_only_the_later_downloads_may_have_an_empty_sha256():
    """Everything the build needs now is hashed; only the benchmark inputs may still be blank.

    The packaging toolchain (Inno Setup, innoextract, PyInstaller) is fetched and hashed by
    build/bootstrap_packaging.py, so no toolchain entry is blank any more. The shipped Whisper
    model was fetched and hashed by the packaging task (19.3 step 6), so it is no longer blank
    either. The benchmark inputs are blank before and filled after a run, so both states are
    accepted: `bench/run.py --fetch` records the candidate hashes and `--select` fills
    models.cleanup_model (18 steps 2 and 4), and neither may break this suite.
    """
    blank = {key for key, entry in _entries() if not entry["sha256"]}
    may_be_blank = {
        "models.cleanup_model",
    } | {f"bench_candidates.{item['name']}" for item in _load()["bench_candidates"]}
    unexpected = blank - may_be_blank
    assert not unexpected, f"{sorted(unexpected)} must carry a real sha256"


def test_a_chosen_cleanup_model_carries_a_real_version_and_url():
    """Once bench/run.py --select has run, the placeholder version must be gone (18 step 4)."""
    entry = _load()["models"]["cleanup_model"]
    if entry["sha256"]:
        assert entry["version"] != "not selected yet"
        assert entry["url"].endswith(".gguf")
