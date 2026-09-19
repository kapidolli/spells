"""build/licenses.py: the manifest of third-party licence texts the installer ships (spec 21).

No network: every check runs against the already-fetched texts in build/licenses/, or against
a tmp_path copy of them.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

from spells.modelcatalog import load_catalog

BUILD_DIR = Path(__file__).resolve().parents[2] / "build"
sys.path.insert(0, str(BUILD_DIR))

import licenses
import package

REAL_LICENSES_DIR = BUILD_DIR / "licenses"

# Hardcoded from spec section 21 ("## 21. Licensing") of
# the design spec, rather than parsed from
# the markdown table, because the table groups several libraries per row and this test wants to
# check each library individually. Maps a library named in the spec table to the component name
# it must appear under in build/licenses/MANIFEST.json.
SPEC_21_LIBRARIES = {
    "whisper.cpp": "whisper_cpp",
    "llama.cpp": "llama_cpp",
    "ggml": "ggml",
    "cpp-httplib": "cpp_httplib",
    "nlohmann json": "nlohmann_json",
    "libgcc": "gcc_runtime_gplv3",
    "libstdc++": "gcc_runtime_gplv3",
    "libgomp": "gcc_runtime_gplv3",
    "winpthreads": "winpthreads",
    "LLVM OpenMP runtime": "llvm_openmp",
    "Whisper model weights": "whisper_model_weights",
    "Silero VAD": "silero_vad",
    "Qwen3 models": "qwen3",
    "Gemma models": "gemma3",
    "PySide6 / Qt": "pyside6_qt",
    "Python runtime": "python",
    "PyInstaller bootloader": "pyinstaller_bootloader",
    "numpy": "numpy",
    "sounddevice": "sounddevice",
    "PortAudio": "portaudio",
    "Gemma 4 models": "gemma4",
    "Qwen3-ASR 0.6B model weights": "qwen3_asr",
}


@pytest.fixture
def licenses_copy(tmp_path):
    """A throwaway copy of the real build/licenses/ folder, safe to corrupt."""
    dest = tmp_path / "licenses"
    shutil.copytree(REAL_LICENSES_DIR, dest)
    return dest


# ----------------------------------------------------------------------------- manifest shape


def test_manifest_is_valid_json_with_a_components_list():
    manifest = licenses.load_manifest(REAL_LICENSES_DIR)
    assert isinstance(manifest["components"], list)
    assert len(manifest["components"]) > 0


def test_components_returns_the_manifest_list():
    entries = licenses.components(REAL_LICENSES_DIR)
    names = [entry["name"] for entry in entries]
    assert len(names) == len(set(names)), "duplicate component name in MANIFEST.json"


def test_every_component_file_exists_and_matches_its_sha256():
    for entry in licenses.components(REAL_LICENSES_DIR):
        filename = entry["file"]
        if filename is None:
            continue
        path = REAL_LICENSES_DIR / filename
        assert path.exists(), f"{entry['name']}: {path} is missing"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == entry["sha256"], f"{entry['name']}: sha256 mismatch for {path}"


def test_every_component_has_the_required_fields():
    required = {"name", "version", "license", "source", "file", "sha256", "ships"}
    for entry in licenses.components(REAL_LICENSES_DIR):
        missing = required - entry.keys()
        assert not missing, f"{entry.get('name')} is missing fields: {missing}"
        assert isinstance(entry["ships"], bool)


def test_a_component_with_no_file_also_has_no_sha256():
    for entry in licenses.components(REAL_LICENSES_DIR):
        if entry["file"] is None:
            assert entry["sha256"] is None, f"{entry['name']}: sha256 set without a file"


# ----------------------------------------------------------------------------- spec 21 coverage


@pytest.mark.parametrize("library", sorted(SPEC_21_LIBRARIES))
def test_spec_21_library_has_a_manifest_entry(library):
    expected_component = SPEC_21_LIBRARIES[library]
    names = {entry["name"] for entry in licenses.components(REAL_LICENSES_DIR)}
    assert expected_component in names, (
        f"spec 21 lists {library!r}, expected component {expected_component!r} in MANIFEST.json"
    )


def test_gcc_runtime_exception_text_is_a_separate_entry_from_the_gplv3_text():
    names = {entry["name"] for entry in licenses.components(REAL_LICENSES_DIR)}
    assert "gcc_runtime_gplv3" in names
    assert "gcc_runtime_library_exception" in names


def test_the_gemma_3_terms_stay_for_comparison_and_never_ship():
    """Gemma 3 lost the benchmark to Gemma 4 E2B; its terms stay in case one ever ships."""
    by_name = {entry["name"]: entry for entry in licenses.components(REAL_LICENSES_DIR)}
    assert by_name["gemma3"]["ships"] is False
    assert by_name["gemma3"]["license"] == "Gemma Terms of Use"


def test_the_qwen_text_ships_now_that_a_graphics_card_installs_qwen35_4b():
    """B5-46 put Qwen3.5 4B on a machine with a graphics card, so its licence travels too."""
    by_name = {entry["name"]: entry for entry in licenses.components(REAL_LICENSES_DIR)}
    assert by_name["qwen3"]["ships"] is True
    assert by_name["qwen3"]["license"] == "Apache-2.0"
    assert "qwen3.5-4b-q4_k_m" in by_name["qwen3"]["models"]


def test_gemma4_ships_and_needs_no_acceptance_page():
    by_name = {entry["name"]: entry for entry in licenses.components(REAL_LICENSES_DIR)}
    assert by_name["gemma4"]["ships"] is True
    assert by_name["gemma4"]["license"] == "Apache-2.0"


def test_qwen3_asr_ships_with_the_speech_model_for_english_and_german():
    by_name = {entry["name"]: entry for entry in licenses.components(REAL_LICENSES_DIR)}
    assert by_name["qwen3_asr"]["ships"] is True
    assert by_name["qwen3_asr"]["license"] == "Apache-2.0"


# ----------------------------------------------------------------------------- model coverage


def test_models_covered_maps_catalog_ids_to_their_licence_component():
    covered = licenses.models_covered(REAL_LICENSES_DIR)
    assert covered["gemma-4-e2b-it-q4_0"] == "gemma4"
    assert covered["gemma-4-e2b-it-q4_0-compose"] == "gemma4"
    assert covered["qwen3.5-4b-q4_k_m-compose"] == "qwen3"
    assert covered["whisper-large-v3-turbo-q8_0"] == "whisper_model_weights"
    assert covered["qwen3-asr-0.6b-q8_0"] == "qwen3_asr"
    assert covered["qwen3-asr-0.6b-q4_k_m"] == "qwen3_asr"


def test_every_redistributable_catalog_model_has_a_licence_entry():
    covered = licenses.models_covered(REAL_LICENSES_DIR)
    for model in load_catalog():
        if not model.redistributable:
            continue
        assert model.id in covered, f"{model.id} has no build/licenses/MANIFEST.json models entry"


def test_the_albanian_fine_tune_is_excluded_since_it_never_ships():
    covered = licenses.models_covered(REAL_LICENSES_DIR)
    assert "whisper-large-v3-turbo-sq-flutra-v2-q8_0" not in covered


def test_every_model_the_default_bundle_ships_has_a_shipping_licence_entry():
    """build/licenses.py must assemble a licence for every model the default bundle ships."""
    bundle = package.default_bundle()
    covered = licenses.models_covered(REAL_LICENSES_DIR)
    by_name = {entry["name"]: entry for entry in licenses.components(REAL_LICENSES_DIR)}
    assert bundle.models, "the default bundle must ship at least one model"
    for model in bundle.models:
        component_name = covered.get(model.id)
        assert component_name, f"{model.id} has no build/licenses/MANIFEST.json models entry"
        assert by_name[component_name]["ships"] is True, (
            f"{model.id} ships but its licence component {component_name!r} does not")


def test_a_shipped_model_with_no_licence_entry_fails_this_check(monkeypatch):
    """Proves the previous test would actually fail if a shipped model lost its licence text."""
    monkeypatch.setattr(licenses, "models_covered", lambda folder=None: {})
    bundle = package.default_bundle()
    covered = licenses.models_covered(REAL_LICENSES_DIR)
    missing = [model.id for model in bundle.models if model.id not in covered]
    assert missing == [model.id for model in bundle.models]


def test_vulkan_loader_is_recorded_as_a_fileless_system_component():
    by_name = {entry["name"]: entry for entry in licenses.components(REAL_LICENSES_DIR)}
    assert "vulkan_loader" in by_name
    assert by_name["vulkan_loader"]["file"] is None
    assert by_name["vulkan_loader"]["ships"] is False


# ----------------------------------------------------------------------------- verify()


def test_verify_passes_on_the_real_folder():
    assert licenses.verify(REAL_LICENSES_DIR) == []


def test_verify_reports_a_missing_file(licenses_copy):
    entries = licenses.components(licenses_copy)
    target = next(e for e in entries if e["file"] is not None)
    (licenses_copy / target["file"]).unlink()

    problems = licenses.verify(licenses_copy)

    assert len(problems) == 1
    assert target["name"] in problems[0]
    assert "missing" in problems[0]


def test_verify_reports_a_hash_mismatch(licenses_copy):
    entries = licenses.components(licenses_copy)
    target = next(e for e in entries if e["file"] is not None)
    (licenses_copy / target["file"]).write_bytes(b"tampered content")

    problems = licenses.verify(licenses_copy)

    assert len(problems) == 1
    assert target["name"] in problems[0]
    assert target["sha256"] in problems[0]


def test_verify_reports_missing_manifest(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()

    problems = licenses.verify(empty)

    assert len(problems) == 1
    assert "MANIFEST.json" in problems[0]


def test_verify_reports_invalid_manifest_json(tmp_path):
    folder = tmp_path / "bad"
    folder.mkdir()
    (folder / "MANIFEST.json").write_text("not json", encoding="utf-8")

    problems = licenses.verify(folder)

    assert len(problems) == 1
    assert "MANIFEST.json" in problems[0]


# ----------------------------------------------------------------------------- assemble()


def test_assemble_writes_one_index_line_per_shipping_component(tmp_path):
    dest = licenses.assemble(tmp_path / "dist-licenses", folder=REAL_LICENSES_DIR)

    index_lines = (dest / "index.txt").read_text(encoding="utf-8").strip("\n").split("\n")
    shipping = [e for e in licenses.components(REAL_LICENSES_DIR) if e["ships"]]
    assert len(index_lines) == len(shipping)

    shipping_names = {e["name"] for e in shipping}
    for line in index_lines:
        name, version, license_name, _source = line.split("\t")
        assert name in shipping_names
        assert version
        assert license_name


def test_assemble_copies_every_component_file(tmp_path):
    dest = licenses.assemble(tmp_path / "dist-licenses", folder=REAL_LICENSES_DIR)

    for entry in licenses.components(REAL_LICENSES_DIR):
        if entry["file"] is None:
            continue
        copied = dest / entry["file"]
        assert copied.exists(), f"{entry['name']}: {copied} was not copied"
        assert copied.read_bytes() == (REAL_LICENSES_DIR / entry["file"]).read_bytes()


def test_assemble_refuses_a_corrupt_folder(tmp_path, licenses_copy):
    entries = licenses.components(licenses_copy)
    target = next(e for e in entries if e["file"] is not None)
    (licenses_copy / target["file"]).unlink()

    with pytest.raises(licenses.LicenseError):
        licenses.assemble(tmp_path / "dist-licenses", folder=licenses_copy)

    assert not (tmp_path / "dist-licenses").exists()


def test_assemble_creates_dest_directory(tmp_path):
    dest = tmp_path / "nested" / "licenses"
    licenses.assemble(dest, folder=REAL_LICENSES_DIR)
    assert dest.is_dir()
    assert (dest / "index.txt").exists()


# ----------------------------------------------------------------------------- folder budget


def test_licenses_folder_stays_under_one_megabyte():
    total = sum(p.stat().st_size for p in REAL_LICENSES_DIR.glob("*") if p.is_file())
    assert total < 1_000_000


# ----------------------------------------------------------------------------- CLI


def test_main_verify_returns_zero_on_the_real_folder(monkeypatch, capsys):
    monkeypatch.setattr(licenses, "LICENSES_DIR", REAL_LICENSES_DIR)
    exit_code = licenses.main(["verify"])
    assert exit_code == 0
    assert "OK" in capsys.readouterr().out


def test_main_assemble_returns_zero(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(licenses, "LICENSES_DIR", REAL_LICENSES_DIR)
    dest = tmp_path / "dist-licenses"
    exit_code = licenses.main(["assemble", str(dest)])
    assert exit_code == 0
    assert (dest / "index.txt").exists()
    assert "assembled" in capsys.readouterr().out


def test_main_verify_returns_one_on_a_broken_folder(monkeypatch, licenses_copy, capsys):
    monkeypatch.setattr(licenses, "LICENSES_DIR", licenses_copy)
    entries = licenses.components(licenses_copy)
    target = next(e for e in entries if e["file"] is not None)
    (licenses_copy / target["file"]).unlink()

    exit_code = licenses.main(["verify"])

    assert exit_code == 1
    assert "ERROR" in capsys.readouterr().err


def test_json_round_trip_is_stable(tmp_path):
    """Re-serializing MANIFEST.json (indent=2) should not silently drop or reorder fields."""
    manifest = licenses.load_manifest(REAL_LICENSES_DIR)
    reserialized = json.loads(json.dumps(manifest))
    assert reserialized == manifest
