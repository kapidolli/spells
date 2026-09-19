"""build/bootstrap_packaging.py: the pure parts of the packaging toolchain bootstrap.

Spec 19.3 steps 5 and 7. No network and no subprocess here: this file covers the smoke script
text, the version parsers, the idempotency decision against a fake toolchain tree, and the
argument lists handed to innoextract and PyInstaller. The parts that download, extract, compile
and run live in the script itself and are exercised by running it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BUILD_DIR = Path(__file__).resolve().parents[2] / "build"
sys.path.insert(0, str(BUILD_DIR))

import bootstrap_packaging as bp

# Captured verbatim from innosetup 6.2.2's ISCC.exe on the reference machine.
ISCC_HELP_OUTPUT = """Inno Setup 6 Command-Line Compiler
Copyright (C) 1997-2023 Jordan Russell. All rights reserved.
Portions Copyright (C) 2000-2023 Martijn Laan. All rights reserved.
Portions Copyright (C) 2001-2004 Alex Yackimoff. All rights reserved.
https://www.innosetup.com

Usage:  iscc [options] scriptfile.iss
"""

ISCC_COMPILE_OUTPUT = """Inno Setup 6 Command-Line Compiler
Copyright (C) 1997-2023 Jordan Russell. All rights reserved.
https://www.innosetup.com

Compiler engine version: Inno Setup 6.2.2

Preprocessing
Parsing [Setup] section, line 2
Successful compile (0.281 sec). Resulting Setup program filename is:
"""


# ----------------------------------------------------------------------------- the smoke script


def test_smoke_iss_carries_the_directives_spec_19_3_step_7_needs():
    text = bp.smoke_iss_text(output_dir=Path(r"C:\repo\build\out"),
                             output_basename="smoke-setup", payload=Path(r"C:\repo\tmp\pay.txt"))

    assert "PrivilegesRequired=lowest" in text
    assert r"DefaultDirName={localappdata}\Programs\\" in text or \
        r"DefaultDirName={localappdata}\Programs" in text
    assert "[Setup]" in text and "[Files]" in text
    assert text.index("[Setup]") < text.index("[Files]")


def test_smoke_iss_places_the_output_where_it_was_asked_to():
    text = bp.smoke_iss_text(output_dir=Path(r"C:\repo\build\out"),
                             output_basename="smoke-setup", payload=Path(r"C:\repo\tmp\pay.txt"))

    assert r"OutputDir=C:\repo\build\out" in text
    assert "OutputBaseFilename=smoke-setup" in text
    assert r'Source: "C:\repo\tmp\pay.txt"' in text


def test_smoke_iss_never_produces_an_uninstaller_or_needs_a_real_app():
    """The compiled exe is deleted and never run, so it must not want anything from the machine."""
    text = bp.smoke_iss_text(output_dir=Path("out"), output_basename="s", payload=Path("p.txt"))

    assert "Uninstallable=no" in text
    assert "[Run]" not in text
    assert "[UninstallRun]" not in text


def test_smoke_iss_is_plain_text_ending_in_one_newline():
    text = bp.smoke_iss_text(output_dir=Path("out"), output_basename="s", payload=Path("p.txt"))

    assert text.endswith("\n") and not text.endswith("\n\n")
    assert "\u2014" not in text, "no em dashes anywhere, including generated data"


# ----------------------------------------------------------------------------- version parsing


def test_iscc_help_banner_yields_the_major_version():
    assert bp.parse_iscc_version(ISCC_HELP_OUTPUT) == "6"


def test_iscc_compile_output_yields_the_exact_engine_version():
    assert bp.parse_iscc_version(ISCC_COMPILE_OUTPUT) == "6.2.2"


def test_the_engine_version_line_wins_over_the_banner():
    """Both lines appear in a compile run; the specific one is the useful answer."""
    assert bp.parse_iscc_version(ISCC_COMPILE_OUTPUT).count(".") == 2


def test_unrelated_output_has_no_version():
    assert bp.parse_iscc_version("ISCC is not recognized as a command") is None
    assert bp.parse_iscc_version("") is None


@pytest.mark.parametrize("reported,pinned,expected", [
    ("6", "6.2.2", True),
    ("6.2", "6.2.2", True),
    ("6.2.2", "6.2.2", True),
    ("7", "6.2.2", False),
    ("6.3", "6.2.2", False),
    ("6.2.22", "6.2.2", False),
    ("6.2.2.1", "6.2.2", False),
])
def test_a_reported_version_agrees_with_the_pin_only_on_whole_segments(reported, pinned, expected):
    assert bp.version_agrees(reported, pinned) is expected


@pytest.mark.parametrize("output,expected", [
    ("6.22.3\n", "6.22.3"),
    ("6.22.3", "6.22.3"),
    ("some deprecation warning\n6.22.3\n", "6.22.3"),
    ("", None),
    ("no version here\n", None),
])
def test_pyinstaller_version_is_read_from_its_own_output(output, expected):
    assert bp.parse_pyinstaller_version(output) == expected


# ----------------------------------------------------------------------------- idempotency


def _fake_tree(root: Path, parts=("innoextract", "innosetup")) -> Path:
    for part in parts:
        subdir, names = bp.PARTS[part]
        folder = root / subdir
        folder.mkdir(parents=True, exist_ok=True)
        for name in names:
            (folder / name).write_bytes(b"x")
    return root


def test_an_empty_tree_reports_every_part_as_missing(tmp_path):
    missing = bp.missing_parts(tmp_path)

    assert set(missing) == set(bp.PARTS)
    assert "innoextract.exe" in missing["innoextract"]
    assert "ISCC.exe" in missing["innosetup"]


def test_a_complete_tree_needs_no_work(tmp_path):
    assert bp.missing_parts(_fake_tree(tmp_path)) == {}


def test_a_tree_with_only_innoextract_still_needs_inno_setup(tmp_path):
    missing = bp.missing_parts(_fake_tree(tmp_path, parts=("innoextract",)))

    assert set(missing) == {"innosetup"}


def test_one_missing_support_file_names_exactly_that_file(tmp_path):
    """ISCC.exe alone cannot compile: the compiler DLL and the setup stubs must be there too."""
    _fake_tree(tmp_path)
    subdir, _ = bp.PARTS["innosetup"]
    (tmp_path / subdir / "ISCmplr.dll").unlink()

    assert bp.missing_parts(tmp_path) == {"innosetup": ["ISCmplr.dll"]}


def test_a_directory_in_place_of_a_required_file_does_not_count(tmp_path):
    _fake_tree(tmp_path)
    subdir, _ = bp.PARTS["innosetup"]
    (tmp_path / subdir / "ISCC.exe").unlink()
    (tmp_path / subdir / "ISCC.exe").mkdir()

    assert bp.missing_parts(tmp_path) == {"innosetup": ["ISCC.exe"]}


def test_the_inno_setup_part_expects_the_compiler_and_its_stubs():
    _, names = bp.PARTS["innosetup"]

    for required in ("ISCC.exe", "ISCmplr.dll", "Default.isl", "Setup.e32", "SetupLdr.e32"):
        assert required in names


# ----------------------------------------------------------------------------- argument lists


def test_innoextract_is_told_to_extract_quietly_into_the_given_directory():
    args = bp.innoextract_args(Path(r"C:\t\innoextract\innoextract.exe"),
                               Path(r"C:\c\innosetup-6.2.2.exe"), Path(r"C:\t\stage"))

    assert args[0] == r"C:\t\innoextract\innoextract.exe"
    assert args[-1] == r"C:\c\innosetup-6.2.2.exe", "the installer is the positional argument"
    assert "--extract" in args
    assert "--silent" in args
    assert args[args.index("--output-dir") + 1] == r"C:\t\stage"
    assert all(isinstance(a, str) for a in args)


def test_innoextract_is_never_asked_to_run_anything():
    args = bp.innoextract_args(Path("ie.exe"), Path("setup.exe"), Path("stage"))

    assert not any(a.startswith("--exec") or a == "--gog" for a in args)


def test_pyinstaller_builds_one_folder_without_upx():
    args = bp.pyinstaller_args(Path(r"C:\v\python.exe"), Path(r"C:\w\smoke.py"),
                               dist_dir=Path(r"C:\o\dist"), work_dir=Path(r"C:\o\work"),
                               name="smoke")

    assert args[:4] == [r"C:\v\python.exe", "-m", "PyInstaller", "--onedir"]
    assert args[-1] == r"C:\w\smoke.py"
    assert "--noupx" in args
    assert "--noconfirm" in args
    assert "--onefile" not in args
    assert args[args.index("--distpath") + 1] == r"C:\o\dist"
    assert args[args.index("--workpath") + 1] == r"C:\o\work"
    assert args[args.index("--name") + 1] == "smoke"
    assert all(isinstance(a, str) for a in args)


def test_the_smoke_app_source_is_two_lines_that_print_the_marker():
    lines = [line for line in bp.SMOKE_APP_SOURCE.splitlines() if line.strip()]

    assert len(lines) == 2
    assert bp.SMOKE_APP_MARKER in bp.SMOKE_APP_SOURCE


# ----------------------------------------------------------------------------- pins agreement


def test_the_script_reads_the_three_pins_it_is_named_after():
    assert bp.INNO_SETUP_KEY == "toolchain.inno_setup"
    assert bp.INNOEXTRACT_KEY == "toolchain.innoextract"
    assert bp.PYINSTALLER_KEY == "toolchain.pyinstaller"


def test_nothing_is_written_outside_the_repo(tmp_path):
    """Every path the script owns stays under the repo's build/ directory (spec 19.2)."""
    repo = bp.REPO_DIR
    for path in (bp.TOOLCHAIN_ROOT, bp.INNOEXTRACT_DIR, bp.INNOSETUP_DIR, bp.STAGE_DIR,
                 bp.SMOKE_DIR, bp.OUT_DIR, bp.PYINSTALLER_SMOKE_DIR):
        assert repo in path.parents, f"{path} is not inside {repo}"
        assert (repo / "build") in path.parents or path == repo / "build"
