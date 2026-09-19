"""build/package.py: the pure parts of the release build (spec 19.3 steps 4 to 8, 19.4, 19.5).

No network, no subprocess, no PyInstaller and no ISCC here. This file covers the decisions the
packaging script makes before it shells out: which Qt modules are excluded, how the Inno Setup
script is filled in, where every source file lands in dist/Spells, how the version is derived
from pyproject.toml, and the two size gates of 19.4 and 19.3 step 8. The steps that actually
build, copy and compile live in the script and are exercised by running it.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
import zipfile
from pathlib import Path

import pytest

from spells import updates
from spells.modelcatalog import Hardware, ModelKind, load_catalog, parse_catalog

REPO_DIR = Path(__file__).resolve().parents[2]
BUILD_DIR = REPO_DIR / "build"
sys.path.insert(0, str(BUILD_DIR))

import package

TEMPLATE_PATH = BUILD_DIR / "spells.iss.template"


def _template() -> str:
    return TEMPLATE_PATH.read_text(encoding="utf-8")


def _values(**overrides) -> dict:
    values = package.iss_values(
        version="0.1.0",
        output_dir=Path(r"C:\repo\dist"),
        dist_dir=Path(r"C:\repo\dist\Spells"),
        cleanup_model_name="qwen3-4b-q4_k_m.gguf",
        licenses_dir=Path(r"C:\repo\build\licenses"),
    )
    values.update(overrides)
    return values


# ----------------------------------------------------------------------------- Qt exclusions


def test_the_modules_the_ui_imports_are_kept():
    assert package.QT_KEEP == ("QtCore", "QtGui", "QtWidgets", "QtSvg")
    for name in package.QT_KEEP:
        assert name not in package.QT_DROP


def test_every_qt_exclusion_is_a_pyside6_submodule_and_none_repeats():
    excludes = package.qt_excludes()
    assert excludes == sorted(excludes)
    assert len(excludes) == len(set(excludes))
    assert all(name.startswith("PySide6.Qt") for name in excludes)


def test_the_modules_spec_19_3_step_5_names_are_all_excluded():
    """The brief's list, plus QtSvgWidgets and QtNetwork, which src/spells/ui never imports."""
    excludes = set(package.qt_excludes())
    for name in ("QtQml", "QtQuick", "QtSql", "QtTest", "QtXml", "QtOpenGL", "QtPrintSupport",
                 "QtConcurrent", "QtDBus", "QtNetwork", "QtSvgWidgets"):
        assert f"PySide6.{name}" in excludes


def test_the_installed_pyside6_has_no_extension_module_that_is_neither_kept_nor_dropped():
    """A new PySide6 release must not smuggle a module past the exclusion list unnoticed."""
    pyside6 = pytest.importorskip("PySide6")
    installed = {path.stem for path in Path(pyside6.__file__).parent.glob("*.pyd")}
    assert installed, "PySide6 is installed but ships no .pyd extension modules"
    assert installed - set(package.QT_KEEP) - set(package.QT_DROP) == set()


def test_the_qt_folders_that_are_pruned_after_the_build():
    assert package.PYSIDE_PRUNE_DIRS == ("qml", "translations")


# ----------------------------------------------------------------------------- PyInstaller arguments


def test_pyinstaller_args_are_one_folder_windowed_and_without_upx():
    args = package.pyinstaller_args(
        Path(r"C:\repo\.venv\Scripts\python.exe"),
        Path(r"C:\repo\build\entry.py"),
        dist_dir=Path(r"C:\repo\build\out\app"),
        work_dir=Path(r"C:\repo\build\out\pyinstaller-work"),
        src_dir=Path(r"C:\repo\src"),
    )

    assert args[:4] == [r"C:\repo\.venv\Scripts\python.exe", "-m", "PyInstaller", "--noconfirm"]
    for flag in ("--onedir", "--windowed", "--noupx"):
        assert flag in args
    assert "--upx-dir" not in args
    assert args[args.index("--name") + 1] == package.APP_NAME
    assert args[-1] == r"C:\repo\build\entry.py"


def test_pyinstaller_args_exclude_every_unused_qt_module():
    args = package.pyinstaller_args(
        Path("python.exe"), Path("entry.py"),
        dist_dir=Path("dist"), work_dir=Path("work"), src_dir=Path("src"))

    excluded = [args[i + 1] for i, item in enumerate(args) if item == "--exclude-module"]
    assert excluded == package.qt_excludes()


def test_pyinstaller_args_put_the_source_tree_on_the_search_path():
    args = package.pyinstaller_args(
        Path("python.exe"), Path("entry.py"),
        dist_dir=Path("dist"), work_dir=Path("work"), src_dir=Path(r"C:\repo\src"))

    assert args[args.index("--paths") + 1] == r"C:\repo\src"


# ----------------------------------------------------------------------------- version derivation


def test_version_comes_from_the_project_table_of_pyproject():
    text = '[project]\nname = "spells"\nversion = "1.2.3"\n'
    assert package.project_version(text) == "1.2.3"


def test_the_repository_version_is_readable_and_matches_the_package():
    from spells import __version__
    from spells.updates import parse_version

    assert parse_version(__version__) is not None
    assert package.project_version(package.PYPROJECT_PATH.read_text(encoding="utf-8")) == __version__


def test_a_pyproject_without_a_version_is_refused():
    with pytest.raises(package.PackageError):
        package.project_version('[project]\nname = "spells"\n')


def test_a_dev_build_carries_the_dev_suffix():
    assert package.build_version("0.1.0", dev=False) == "0.1.0"
    assert package.build_version("0.1.0", dev=True) == "0.1.0-dev"


def test_the_installer_file_name_carries_the_version():
    assert package.installer_basename("0.1.0") == "Spells-Setup-0.1.0"
    assert package.installer_basename("0.1.0-dev") == "Spells-Setup-0.1.0-dev"


# ----------------------------------------------------------------------------- the size gates


@pytest.mark.parametrize("size,verdict", [
    (0, "ok"),
    (package.APP_SIZE_TARGET_BYTES - 1, "ok"),
    (package.APP_SIZE_TARGET_BYTES, "ok"),
    (package.APP_SIZE_TARGET_BYTES + 1, "near"),
    (package.APP_SIZE_WARN_BYTES, "near"),
    (package.APP_SIZE_WARN_BYTES + 1, "over"),
])
def test_app_folder_size_verdict_boundaries(size, verdict):
    assert package.app_size_verdict(size) == verdict


def test_the_app_size_lines_are_the_ones_spec_19_4_names():
    assert package.APP_SIZE_TARGET_BYTES == 120_000_000
    assert package.APP_SIZE_WARN_BYTES == 160_000_000


@pytest.mark.parametrize("size,ok", [
    (0, True),
    (1_999_999_999, True),
    (2_000_000_000, True),
    (2_000_000_001, False),
])
def test_installer_size_boundary_is_the_one_spec_19_3_step_8_names(size, ok):
    assert package.INSTALLER_MAX_BYTES == 2_000_000_000
    assert package.installer_size_ok(size) is ok


# ----------------------------------------------------------------------------- the dist layout plan


def _plan(cleanup_name: str = "qwen3-4b-q4_k_m.gguf"):
    return package.dist_plan(
        app_dir=Path(r"C:\repo\build\out\app\Spells"),
        engines_dir=Path(r"C:\repo\build\out\engines"),
        speech_models=[Path(r"C:\repo\build\cache\models\ggml-large-v3-turbo-q8_0.bin")],
        vad_model=Path(r"C:\repo\build\cache\models\ggml-silero-v5.1.2.bin"),
        cleanup_model=Path(r"C:\repo\build\cache\models") / cleanup_name,
        data_dir=Path(r"C:\repo\data"),
    )


def test_every_source_lands_where_spec_19_3_step_6_puts_it():
    where = {item.dest: item for item in _plan()}

    assert where["."].kind == "app"
    assert where["."].source == Path(r"C:\repo\build\out\app\Spells")
    assert where["engines/vulkan"].source == Path(r"C:\repo\build\out\engines\vulkan")
    assert where["engines/cpu"].source == Path(r"C:\repo\build\out\engines\cpu")
    assert where["data"].source == Path(r"C:\repo\data")
    assert where["licenses"].kind == "licenses"
    assert where["licenses"].source is None


def test_the_model_names_are_the_ones_the_frozen_app_looks_for():
    from spells import paths as spells_paths

    dests = {item.dest for item in _plan()}
    assert f"models/{spells_paths.WHISPER_MODEL_NAME}" in dests
    assert f"models/{spells_paths.VAD_MODEL_NAME}" in dests


def test_the_cleanup_model_keeps_its_own_file_name():
    plan = _plan("gemma-3-4b-it-Q4_K_M.gguf")
    assert "models/gemma-3-4b-it-Q4_K_M.gguf" in {item.dest for item in plan}


def test_the_plan_only_names_destinations_the_installer_knows_about():
    tops = {item.dest.split("/")[0] for item in _plan()} - {"."}
    assert tops <= set(package.DIST_TOP_LEVEL)


def test_the_top_level_of_a_built_dist_is_exactly_what_the_installer_copies():
    package.check_dist_top_level(list(package.DIST_TOP_LEVEL))
    with pytest.raises(package.PackageError):
        package.check_dist_top_level([*package.DIST_TOP_LEVEL, "stray.txt"])
    with pytest.raises(package.PackageError):
        package.check_dist_top_level([name for name in package.DIST_TOP_LEVEL if name != "models"])


# ----------------------------------------------------------------------------- exactly one gguf


def test_exactly_one_gguf_ships_in_models():
    plan = _plan()
    assert package.gguf_names(plan) == ["qwen3-4b-q4_k_m.gguf"]
    package.check_one_gguf(plan)


def test_a_second_gguf_in_models_is_refused():
    """paths._llama_model picks the single stray GGUF when frozen, so two would disable cleanup."""
    plan = [*_plan(), package.DistItem("file", Path(r"C:\other.gguf"), "models/other.gguf")]
    with pytest.raises(package.PackageError):
        package.check_one_gguf(plan)


def test_no_gguf_in_models_is_refused():
    plan = [item for item in _plan() if not item.dest.endswith(".gguf")]
    with pytest.raises(package.PackageError):
        package.check_one_gguf(plan)


# ----------------------------------------------------------------------------- the Inno script


def test_the_template_exists_and_only_uses_placeholders_the_script_fills():
    template = _template()
    assert package.placeholders_in(template) == set(_values())


def test_filling_the_template_leaves_no_placeholder_behind():
    text = package.render_iss(_template(), _values())
    assert "@@" not in text


def test_rendering_refuses_a_missing_value():
    values = _values()
    values.pop("APP_VERSION")
    with pytest.raises(package.PackageError):
        package.render_iss(_template(), values)


def test_rendering_refuses_a_value_the_template_never_asks_for():
    with pytest.raises(package.PackageError):
        package.render_iss(_template(), _values(NOT_A_PLACEHOLDER="x"))


def test_the_rendered_script_carries_every_directive_spec_19_5_asks_for():
    text = package.render_iss(_template(), _values())

    assert "PrivilegesRequired=lowest" in text
    assert r"DefaultDirName={localappdata}\Programs\Spells" in text
    # Inno checks AppMutex when Setup and Uninstall start, before PrepareToInstall, so a running
    # instance would stop the wizard (and abort a silent upgrade) before --quit could close it.
    assert "AppMutex=" not in text
    assert "CloseApplications=yes" in text
    assert "[UninstallRun]" in text
    assert "--quit" in text
    assert "function PrepareToInstall" in text
    assert "CheckForMutexes('Spells')" in text
    assert "nocompression" in text
    assert "postinstall" in text and "skipifsilent" in text
    assert "REMOVEDATA" in text


def test_the_rendered_script_never_asks_for_elevation_and_never_writes_the_run_value():
    text = package.render_iss(_template(), _values())

    assert "PrivilegesRequiredOverridesAllowed" not in text
    assert "requestedExecutionLevel" not in text
    assert "[Registry]" not in text  # the app writes its own Run value (spec 15)


def test_the_uninstaller_removes_the_run_value_and_only_then_the_data_folders():
    text = package.render_iss(_template(), _values())

    assert r"Software\Microsoft\Windows\CurrentVersion\Run" in text
    assert "'Spells'" in text
    assert "{userappdata}\\Spells" in text
    assert "{localappdata}\\Spells" in text


def test_the_version_reaches_both_the_app_version_and_the_output_file_name():
    values = _values(APP_VERSION="0.1.0-dev", OUTPUT_BASENAME="Spells-Setup-0.1.0-dev")
    text = package.render_iss(_template(), values)

    assert "AppVersion=0.1.0-dev" in text
    assert "OutputBaseFilename=Spells-Setup-0.1.0-dev" in text


# ----------------------------------------------------------------------------- the Gemma page


def test_a_gemma_cleanup_model_is_recognised_by_its_file_name():
    assert package.ships_gemma("gemma-3-4b-it-Q4_K_M.gguf") is True
    assert package.ships_gemma("Gemma-3-4B-it-Q4_K_M.gguf") is True
    assert package.ships_gemma("qwen3-4b-q4_k_m.gguf") is False
    assert package.ships_gemma("") is False


def test_the_licence_page_is_on_only_for_a_gemma_model():
    gemma = package.render_iss(_template(), _values(
        **package.iss_values(
            version="0.1.0",
            output_dir=Path(r"C:\repo\dist"),
            dist_dir=Path(r"C:\repo\dist\Spells"),
            cleanup_model_name="gemma-3-4b-it-Q4_K_M.gguf",
            licenses_dir=Path(r"C:\repo\build\licenses"),
        )))
    qwen = package.render_iss(_template(), _values())

    assert "LicenseFile=" in gemma
    assert package.GEMMA_LICENSE_FILE in gemma
    assert "LicenseFile=" not in qwen
    assert package.GEMMA_LICENSE_FILE not in qwen


def test_the_gemma_terms_exist_in_the_repository_so_the_page_can_be_shown():
    assert (REPO_DIR / "build" / "licenses" / package.GEMMA_LICENSE_FILE).is_file()


# ----------------------------------------------------------------------------- the machine state check


def test_the_paths_the_installer_tests_touch_are_the_ones_spec_15_names():
    assert package.INSTALL_DIR.name == "Spells"
    assert package.INSTALL_DIR.parent.name == "Programs"
    assert package.UNINSTALL_KEY == r"Software\Microsoft\Windows\CurrentVersion\Uninstall"
    assert package.RUN_KEY == r"Software\Microsoft\Windows\CurrentVersion\Run"
    assert package.RUN_VALUE == "Spells"


# ----------------------------------------------------------------------------- loopback endpoints


def _endpoint(state, local, remote="0.0.0.0"):
    return {"State": state, "LocalAddress": local, "RemoteAddress": remote,
            "LocalPort": 8080, "RemotePort": 0, "OwningProcess": 1234}


def test_a_listening_socket_is_judged_by_where_it_is_bound():
    """Get-NetTCPConnection reports Listen with no peer, so only LocalAddress can be judged."""
    assert package.loopback_only(_endpoint("Listen", "127.0.0.1")) is True
    assert package.loopback_only(_endpoint("Listen", "::1")) is True
    assert package.loopback_only(_endpoint("Listen", "0.0.0.0")) is False


def test_an_established_connection_needs_loopback_on_both_sides():
    assert package.loopback_only(_endpoint("Established", "127.0.0.1", "127.0.0.1")) is True
    assert package.loopback_only(_endpoint("Established", "127.0.0.1", "140.82.121.4")) is False
    assert package.loopback_only(_endpoint("Established", "192.168.1.20", "127.0.0.1")) is False


def test_a_bound_socket_with_a_placeholder_remote_is_judged_by_its_local_address():
    """Bound has no peer either, so the 0.0.0.0 remote it reports must not count against it."""
    assert package.loopback_only(_endpoint("Bound", "127.0.0.1", "0.0.0.0")) is True
    assert package.loopback_only(_endpoint("Bound", "0.0.0.0", "0.0.0.0")) is False


def test_a_real_offender_is_caught():
    assert package.loopback_only(_endpoint("Established", "192.168.1.20", "104.18.32.7")) is False


def test_the_state_is_asked_for_by_name_because_convertto_json_writes_the_enum_as_a_number():
    """Listen is 2 and Established is 5 over JSON, and a number reads as an unknown state.

    An unknown state is treated as connected, so a loopback listener arriving as 2 would be
    reported as an offender. The PowerShell asks for the name to keep that from happening.
    """
    assert "$_.State.ToString()" in inspect.getsource(package.tcp_endpoints)
    assert package.loopback_only(_endpoint(2, "127.0.0.1")) is False


# ----------------------------------------------------------------------------- the exit code


@pytest.mark.parametrize("counts,code", [
    ({"PASS": 23, "FAIL": 0, "SKIP": 0}, 0),
    ({"PASS": 8, "FAIL": 0, "SKIP": 15}, 2),
    ({"PASS": 20, "FAIL": 3, "SKIP": 0}, 1),
    ({"PASS": 5, "FAIL": 3, "SKIP": 15}, 1),
])
def test_an_unverified_installer_never_exits_zero(counts, code):
    assert package.test_install_exit_code(counts) == code


def test_the_three_exit_codes_are_distinct():
    assert {package.EXIT_OK, package.EXIT_FAILED, package.EXIT_SKIPPED} == {0, 1, 2}


# ----------------------------------------------------------------------------- the signing hook


def test_nothing_is_signed_by_default():
    assert package.signtool_command(None, env={}) is None
    assert package.signtool_command("", env={}) is None
    assert package.sign_flag(None) == ""
    assert "SignTool=" not in package.sign_setup_lines(None)


def test_the_environment_variable_is_the_other_way_in():
    env = {package.SIGNTOOL_ENV: "signtool.exe sign $f"}
    assert package.signtool_command(None, env=env) == "signtool.exe sign $f"
    assert package.signtool_command("other.exe $f", env=env) == "other.exe $f"


def test_signing_turns_on_the_setup_directives_and_the_file_flag():
    lines = package.sign_setup_lines("signtool.exe sign $f")
    assert "SignTool=spells" in lines
    assert "SignedUninstaller=yes" in lines
    assert package.sign_flag("signtool.exe sign $f") == " sign"


def test_the_unsigned_template_says_why_and_the_signed_one_carries_the_directives():
    signed_values = _values(
        SIGN_SETUP_LINES=package.sign_setup_lines("signtool.exe sign $f"),
        SIGN_FLAG=package.sign_flag("signtool.exe sign $f"))
    unsigned = package.render_iss(_template(), _values())
    signed = package.render_iss(_template(), signed_values)

    assert "SignTool=" not in unsigned
    assert "SignedUninstaller" not in unsigned
    assert "Smart App Control" in unsigned
    assert "Flags: ignoreversion sign" not in unsigned
    assert "SignTool=spells" in signed
    assert "SignedUninstaller=yes" in signed
    assert "Flags: ignoreversion sign" in signed


def test_the_signing_command_must_name_the_file_the_way_inno_setup_does():
    with pytest.raises(package.PackageError):
        package.signtool_args("signtool.exe sign /fd sha256", Path(r"C:\dist\Spells\Spells.exe"))


def test_the_signing_command_is_split_with_the_file_substituted_for_the_placeholder():
    args = package.signtool_args("signtool.exe sign /fd sha256 $f",
                                 Path(r"C:\dist\Spells\Spells.exe"))
    assert args == ["signtool.exe", "sign", "/fd", "sha256", r"C:\dist\Spells\Spells.exe"]


# ----------------------------------------------------------------------------- upgrade and uninstall


def _section(text: str, name: str, until: str) -> list[str]:
    body = text.split(f"[{name}]", 1)[1].split(f"[{until}]", 1)[0]
    return [line for line in body.splitlines()
            if line.strip() and not line.strip().startswith(";")]


def test_an_upgrade_clears_only_the_pyinstaller_folder_before_writing_the_new_one():
    text = package.render_iss(_template(), _values())
    assert _section(text, "InstallDelete", "Files") == [
        'Type: filesandordirs; Name: "{app}\\_internal"']


def test_the_uninstall_delete_section_only_names_the_install_folder():
    text = package.render_iss(_template(), _values())
    assert _section(text, "UninstallDelete", "Code") == [
        'Type: filesandordirs; Name: "{app}"']


def test_the_uninstall_quit_step_is_skipped_when_the_executable_is_already_gone():
    text = package.render_iss(_template(), _values())
    line = next(line for line in text.splitlines() if '"--quit"' in line)

    assert "skipifdoesntexist" in line
    assert "waituntilterminated" in line


def _pascal_routine(text: str, header: str) -> str:
    """The source of one [Code] routine, from its header to the matching top-level end."""
    start = text.index(header)
    end = text.index("\nend;", start)
    return text[start:end + len("\nend;")]


def _pascal_const(text: str, name: str) -> str:
    code = text.split("[Code]", 1)[1]
    line = next(line for line in code.splitlines() if line.strip().startswith(f"{name} = "))
    return line.split("=", 1)[1].strip().rstrip(";")


def test_prepare_to_install_polls_the_app_mutex_for_up_to_20_seconds():
    """The mutex, not the window: an instance still starting holds it before any window exists.

    The app takes MUTEX_NAME as its first startup step and drops it as its last shutdown step.
    --quit gives up after 10 s in all; Setup allows 20 s because the instance may still be
    shutting down.
    """
    from spells.app import MUTEX_NAME

    text = package.render_iss(_template(), _values())
    prepare = _pascal_routine(text, "function PrepareToInstall(")
    wait = _pascal_routine(text, "function WaitForAppToExit(")

    assert MUTEX_NAME == "Spells"
    assert "ewNoWait" in prepare  # a hung --quit must not block Setup
    assert "if not FileExists(Existing) then" in prepare
    assert "WaitForAppToExit(QuitWaitMs)" in prepare
    assert int(_pascal_const(text, "QuitWaitMs")) == 20000
    assert int(_pascal_const(text, "QuitPollMs")) == 100
    assert f"while (Waited < TimeoutMs) and CheckForMutexes('{MUTEX_NAME}') do" in wait
    assert "Sleep(QuitPollMs);" in wait
    assert "Waited := Waited + QuitPollMs;" in wait
    assert "FindWindowByClassName" not in text
    assert "SpellsMessageWindow" not in text


def test_prepare_to_install_skips_the_quit_helper_when_no_instance_holds_the_mutex():
    """A helper started from the install folder would only put Spells.exe in use, and it takes
    the mutex itself for a moment when it finds no instance (app._quit_running_instance)."""
    text = package.render_iss(_template(), _values())
    prepare = _pascal_routine(text, "function PrepareToInstall(")

    skip = prepare.index("if not CheckForMutexes('Spells') then")
    helper = prepare.index("Exec(Existing, '--quit'")
    assert prepare.index("if not FileExists(Existing) then") < skip < helper
    assert "Exit;" in prepare[skip:helper]


def test_prepare_to_install_gives_the_helper_up_to_2_seconds_after_the_mutex_is_free():
    text = package.render_iss(_template(), _values())
    prepare = _pascal_routine(text, "function PrepareToInstall(")
    grace = _pascal_routine(text, "function WaitForHelperToLeave(")
    in_use = _pascal_routine(text, "function FileInUse(")

    assert int(_pascal_const(text, "HelperGraceMs")) == 2000
    mutex_wait = prepare.index("if WaitForAppToExit(QuitWaitMs) then")
    helper_wait = prepare.index("WaitForHelperToLeave(Existing, HelperGraceMs)")
    timed_out = prepare.index("still held after 20 s")
    # Only after the mutex is free: the grace sits in the success branch, before its else.
    assert mutex_wait < helper_wait < timed_out
    assert "while (Waited < TimeoutMs) and FileInUse(FileName) do" in grace
    assert "Sleep(QuitPollMs);" in grace
    assert "Waited := Waited + QuitPollMs;" in grace
    # fmOpenReadWrite or fmShareExclusive fails while a process still runs the executable.
    assert _pascal_const(text, "ExclusiveWriteMode") == "$0012"
    assert "TFileStream.Create(FileName, ExclusiveWriteMode)" in in_use
    assert "except" in in_use and "Result := True;" in in_use


def test_prepare_to_install_never_blocks_the_install_after_the_timeout():
    text = package.render_iss(_template(), _values())
    prepare = _pascal_routine(text, "function PrepareToInstall(")

    # A non-empty result would stop Setup on the preparing page; Restart Manager stays as the
    # second line.
    assert "Result := '';" in prepare
    assert "Result :=" not in prepare.replace("Result := '';", "")
    assert "NeedsRestart := False;" in prepare


def test_a_cancelled_data_removal_dialog_removes_nothing():
    text = package.render_iss(_template(), _values())
    assert "Result := (Form.ShowModal() = mrOk) and Box.Checked;" in text


# ----------------------------------------------------------------------------- registry scoping


def test_the_uninstall_entry_is_found_by_the_app_id_never_by_a_name_substring():
    """A substring match on the display name would find, and the cleanup delete, another app."""
    source = inspect.getsource(package.uninstall_entries)

    assert package.UNINSTALL_SUBKEY == f"{package.APP_ID}_is1"
    assert package.APP_ID in package.render_iss(_template(), _values())
    assert "UNINSTALL_SUBKEY" in source
    assert "EnumKey" not in source


# ----------------------------------------------------------------------------- the setup process


def test_the_running_setup_is_watched_by_its_temp_child_not_only_the_exe():
    """Inno's loader extracts the real setup into %TEMP% and runs it as <basename>.tmp."""
    names = package.setup_process_names(Path(r"C:\repo\dist\Spells-Setup-0.1.0-dev.exe"))

    assert "Spells-Setup-0.1.0-dev.exe" in names
    assert "Spells-Setup-0.1.0-dev.tmp" in names


# ----------------------------------------------------------------------------- the licence manifest


def test_a_cleanup_model_with_no_shipping_licence_entry_is_warned_about_not_refused():
    warning = package.warn_unlisted_cleanup_model("qwen2.5-0.5b-instruct-q4_k_m.gguf")
    assert warning and "licence" in warning.lower()


def test_a_listed_cleanup_model_draws_no_warning(monkeypatch):
    monkeypatch.setattr(package, "shipping_license_names", lambda: {"qwen3"})
    assert package.warn_unlisted_cleanup_model("Qwen3-4B-Q4_K_M.gguf") == ""


def test_the_shipping_names_come_from_the_licence_manifest():
    names = package.shipping_license_names()

    assert "whisper_cpp" in names
    assert "numpy" not in names  # recorded but never shipped (B4-25)


# ----------------------------------------------------------------------------- the models that ship

GEMMA4_FILE = "gemma-4-E2B-it-Q4_0.gguf"
GEMMA4_DRAFT = "mtp-gemma-4-E2B-it-Q4_0.gguf"


def _model(model_id, kind, languages, *, gpu, cpu, size=1, extra_files=(), license="MIT"):
    return {
        "id": model_id,
        "kind": kind,
        "display_name": model_id,
        "file": f"{model_id}.{'bin' if kind == 'asr' else 'gguf'}",
        "extra_files": list(extra_files),
        "size_bytes": size,
        "license": license,
        "runtime": "whisper-server" if kind == "asr" else "llama-server",
        "languages": languages,
        "hardware": {"gpu": gpu, "cpu": cpu},
    }


def _catalog(*entries):
    return parse_catalog({"schema_version": 1, "models": list(entries)})


def test_the_offline_bundle_is_the_processor_set_because_it_has_to_serve_any_machine():
    bundle = package.default_bundle()
    assert [model.id for model in bundle.models] == [
        "qwen3-asr-0.6b-q4_k_m",
        "whisper-large-v3-turbo-q8_0",
        "gemma-4-e2b-it-q4_0",
    ]
    assert bundle.cleanup.file == GEMMA4_FILE
    assert [model.id for model in bundle.left_out] == [
        "qwen3.5-4b-q4_k_m",
    ]
    assert [model.id for model in bundle.speech] == [
        "qwen3-asr-0.6b-q4_k_m",
        "whisper-large-v3-turbo-q8_0",
    ]


def test_the_offline_bundle_installs_inside_the_budget():
    bundle = package.default_bundle()
    models = sum(model.size_bytes for model in bundle.models) + 885_098
    assert models == 4_474_398_453
    assert bundle.footprint_bytes == 4_795_377_726
    assert bundle.footprint_bytes < package.INSTALL_BUDGET_BYTES
    package.check_bundle_budget(bundle)
    by_id = {model.id: model for model in load_catalog()}
    with_q8_0 = (bundle.footprint_bytes - by_id["qwen3-asr-0.6b-q4_k_m"].size_bytes
                 + by_id["qwen3-asr-0.6b-q8_0"].size_bytes)
    assert with_q8_0 == 5_115_911_230
    assert with_q8_0 > package.INSTALL_BUDGET_BYTES


def test_a_bundle_where_both_computers_agree_leaves_nothing_out():
    catalog = _catalog(
        _model("w", "asr", {"en": 90}, gpu=300, cpu=5000),
        _model("c", "cleanup", {"en": 90}, gpu=400, cpu=1000),
    )
    bundle = package.default_bundle(catalog, ("en",))
    assert [model.id for model in bundle.models] == ["w", "c"]
    assert bundle.left_out == ()


def test_a_bundle_without_a_processor_choice_ships_the_graphics_card_choice():
    catalog = _catalog(
        _model("w", "asr", {"en": 90}, gpu=300, cpu=5000),
        _model("c", "cleanup", {"en": 90}, gpu=400, cpu=4000),
    )
    bundle = package.default_bundle(catalog, ("en",))
    assert bundle.cleanup.id == "c"
    assert bundle.left_out == ()


def test_the_bundle_lines_name_what_ships_and_what_stays_out():
    lines = package.bundle_lines(package.default_bundle())
    text = "\n".join(lines)
    assert "English/German/Albanian" in lines[0]
    assert f"{GEMMA4_FILE}, {GEMMA4_DRAFT}" in text
    assert "left out qwen3.5-4b-q4_k_m" in text


def test_a_model_that_is_not_redistributable_never_ships():
    bundle = package.default_bundle()
    assert "whisper-large-v3-turbo-sq-flutra-v2-q8_0" in [model.id for model in bundle.personal]
    shipped = [model.id for model in bundle.models + bundle.left_out]
    assert "whisper-large-v3-turbo-sq-flutra-v2-q8_0" not in shipped
    text = "\n".join(package.bundle_lines(bundle))
    assert "never ships whisper-large-v3-turbo-sq-flutra-v2-q8_0: not redistributable" in text
    catalog = _catalog(
        _model("stock", "asr", {"en": 80}, gpu=300, cpu=5000),
        {**_model("private", "asr", {"en": 99}, gpu=300, cpu=5000), "redistributable": False},
        _model("c", "cleanup", {"en": 90}, gpu=400, cpu=1000),
    )
    assert [model.id for model in package.default_bundle(catalog, ("en",)).speech] == ["stock"]


def test_the_payload_lines_warn_only_above_the_gate():
    lines, total = package.payload_lines([("models", 1_000_000_000), ("app", 500)])
    assert total == 1_000_000_500
    assert "under the 2,000,000,000-byte gate" in lines[-1]
    lines, total = package.payload_lines([("models", 2_000_000_001)])
    assert lines[-1].strip().startswith("WARNING: 1 bytes over")


def test_a_release_ships_only_the_pinned_cleanup_model_the_selection_needs():
    bundle = package.default_bundle()
    package.check_release_cleanup(GEMMA4_FILE, bundle)
    package.check_release_cleanup(GEMMA4_FILE.lower(), bundle)
    with pytest.raises(package.PackageError, match=GEMMA4_FILE):
        package.check_release_cleanup("Qwen3-4B-Q4_K_M.gguf", bundle)
    with pytest.raises(package.PackageError):
        package.check_release_cleanup("", bundle)


def test_a_release_without_any_suitable_cleanup_model_is_refused():
    catalog = _catalog(_model("w", "asr", {"en": 90}, gpu=300, cpu=5000))
    with pytest.raises(package.PackageError):
        package.check_release_cleanup("x.gguf", package.default_bundle(catalog, ("en",)))


def test_every_bundled_speech_model_must_be_pinned():
    bundle = package.default_bundle()
    pins = package.model_pins()
    package.check_bundle_speech(pins, bundle)
    wanted = [name for model in bundle.speech for name in (model.file, *model.extra_files)]
    assert wanted == ["Qwen3-ASR-0.6B-Q4_K_M.gguf", "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf",
                      "ggml-large-v3-turbo-q8_0.bin"]
    for name in wanted:
        short = {other for other in pins if other != name.lower()}
        with pytest.raises(package.PackageError):
            package.check_bundle_speech(short, bundle)


def test_the_pins_name_every_file_the_default_bundle_ships():
    pins = package.model_pins()
    for model in package.default_bundle().models:
        for name in (model.file, *model.extra_files):
            assert name.lower() in pins, name


def test_a_speech_gguf_does_not_count_as_a_second_cleanup_model():
    models = Path(r"C:\repo\build\cache\models")
    plan = package.dist_plan(
        app_dir=Path(r"C:\repo\build\out\app\Spells"),
        engines_dir=Path(r"C:\repo\build\out\engines"),
        speech_models=[
            models / "ggml-large-v3-turbo-q8_0.bin",
            models / "Qwen3-ASR-0.6B-Q8_0.gguf",
            models / "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf",
        ],
        vad_model=models / "ggml-silero-v5.1.2.bin",
        cleanup_model=models / GEMMA4_FILE,
        data_dir=Path(r"C:\repo\data"),
        extra_model_files=[models / GEMMA4_DRAFT],
    )
    package.check_one_gguf(plan, package.catalog_extra_names() | package.catalog_speech_names())


def test_the_extra_files_of_a_catalog_cleanup_model_must_sit_beside_it(tmp_path):
    model = tmp_path / GEMMA4_FILE
    model.write_bytes(b"")
    with pytest.raises(package.PackageError, match=GEMMA4_DRAFT):
        package.extra_model_files(model)
    (tmp_path / GEMMA4_DRAFT).write_bytes(b"")
    assert package.extra_model_files(model) == (tmp_path / GEMMA4_DRAFT,)
    other = tmp_path / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    other.write_bytes(b"")
    assert package.extra_model_files(other) == ()


def test_the_extra_files_land_in_models_and_do_not_count_as_a_second_cleanup_model():
    plan = package.dist_plan(
        app_dir=Path(r"C:\repo\build\out\app\Spells"),
        engines_dir=Path(r"C:\repo\build\out\engines"),
        speech_models=[Path(r"C:\repo\build\cache\models\ggml-large-v3-turbo-q8_0.bin")],
        vad_model=Path(r"C:\repo\build\cache\models\ggml-silero-v5.1.2.bin"),
        cleanup_model=Path(r"C:\repo\build\cache\models") / GEMMA4_FILE,
        data_dir=Path(r"C:\repo\data"),
        extra_model_files=[Path(r"C:\repo\build\cache\models") / GEMMA4_DRAFT],
    )
    assert f"models/{GEMMA4_DRAFT}" in {item.dest for item in plan}
    package.check_one_gguf(plan, package.catalog_extra_names())
    with pytest.raises(package.PackageError):
        package.check_one_gguf(plan)
    stray = [*plan, package.DistItem("file", Path(r"C:\other.gguf"), "models/other.gguf")]
    with pytest.raises(package.PackageError):
        package.check_one_gguf(stray, package.catalog_extra_names())


def test_the_cleanup_model_in_dist_ignores_the_extra_files_and_the_speech_models():
    extras = package.catalog_extra_names() | package.catalog_speech_names()
    names = [GEMMA4_FILE, GEMMA4_DRAFT, "ggml-large-v3-turbo-q8_0.bin",
             "Qwen3-ASR-0.6B-Q8_0.gguf", "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf"]
    assert package.cleanup_gguf_name(names, extras) == GEMMA4_FILE
    with pytest.raises(package.PackageError):
        package.cleanup_gguf_name([*names, "other.gguf"], extras)
    with pytest.raises(package.PackageError):
        package.cleanup_gguf_name(["ggml-large-v3-turbo-q8_0.bin"], extras)


def test_the_gemma_page_follows_the_catalog_licence():
    assert package.ships_gemma(GEMMA4_FILE) is False
    assert package.ships_gemma("gemma-3-4b-it-q4_k_m.gguf") is True
    assert package.ships_gemma("gemma-2-9b-it.gguf") is True
    catalog = _catalog(_model("c", "cleanup", {"en": 90}, gpu=1, cpu=1, license="Gemma terms"))
    assert package.ships_gemma("C.GGUF", catalog) is True


def test_a_hyphenated_shipped_model_draws_no_unlisted_warning():
    """Regression: gemma-4-E2B-it-Q4_0.gguf must match a "gemma4" licence entry despite the
    hyphens the old substring check missed."""
    assert package.warn_unlisted_cleanup_model(GEMMA4_FILE) == ""
    assert package.warn_unlisted_cleanup_model("gemma-3-4b-it-Q4_K_M.gguf") != ""


def test_the_real_default_cleanup_model_has_a_listed_licence():
    bundle = package.default_bundle()
    assert package.warn_unlisted_cleanup_model(bundle.cleanup.file) == ""


# ----------------------------------------------------------------------------- disk spanning


def test_the_span_threshold_is_below_innos_own_requirement():
    """ISCC refuses a single file above 2,100,000,000 bytes; it says so itself (B5-49)."""
    assert package.INSTALLER_SPAN_REQUIRED_BYTES == 2_100_000_000
    assert package.INSTALLER_SPAN_REQUIRED_BYTES == package.DISK_SLICE_SIZE_BYTES
    assert package.INSTALLER_MAX_BYTES < package.INSTALLER_SPAN_REQUIRED_BYTES


def test_needs_spanning_follows_the_non_spanning_ceiling():
    assert package.needs_spanning(package.INSTALLER_MAX_BYTES) is False
    assert package.needs_spanning(package.INSTALLER_MAX_BYTES + 1) is True


def test_estimate_compressed_bytes_adds_the_measured_nonmodel_cost():
    assert package.estimate_compressed_bytes(0) == package.NONMODEL_COMPRESSED_ESTIMATE_BYTES
    assert (package.estimate_compressed_bytes(1_000)
            == package.NONMODEL_COMPRESSED_ESTIMATE_BYTES + 1_000)


def test_the_current_default_bundle_spans_into_setup_exe_and_two_slices():
    """Qwen3-ASR Q4_K_M is back in the bundle beside Whisper, which adds a slice; ISCC refuses
    any single file over 2,100,000,000 bytes either way (B5-45, B5-49)."""
    bundle = package.default_bundle()
    models_bytes = sum(model.size_bytes for model in bundle.models) + 885_098
    estimate = package.estimate_compressed_bytes(models_bytes)
    assert estimate == 4_558_398_453
    assert package.needs_spanning(estimate) is True
    assert package.slice_count(estimate) == 3
    assert package.offline_artifact_name("0.1.0") == "Spells-Setup-0.1.0.zip"


def test_the_q4_k_m_bundle_is_smaller_than_the_q8_0_one_would_have_been():
    bundle = package.default_bundle()
    models_bytes = sum(model.size_bytes for model in bundle.models) + 885_098
    by_id = {model.id: model for model in load_catalog()}
    with_q8_0 = (models_bytes - by_id["qwen3-asr-0.6b-q4_k_m"].size_bytes
                 + by_id["qwen3-asr-0.6b-q8_0"].size_bytes)
    estimate = package.estimate_compressed_bytes(with_q8_0)
    assert estimate == 4_878_931_957
    assert estimate - package.estimate_compressed_bytes(models_bytes) == 320_533_504
    assert package.needs_spanning(estimate) is True


def test_slice_count_stays_at_setup_exe_plus_one_or_two_bins_for_every_bundle():
    assert package.slice_count(1) == 1
    assert package.slice_count(package.DISK_SLICE_SIZE_BYTES) == 1
    assert package.slice_count(package.DISK_SLICE_SIZE_BYTES + 1) == 2
    bundle = package.default_bundle()
    models_bytes = sum(model.size_bytes for model in bundle.models) + 885_098
    by_id = {model.id: model for model in load_catalog()}
    with_q8_0 = (models_bytes - by_id["qwen3-asr-0.6b-q4_k_m"].size_bytes
                 + by_id["qwen3-asr-0.6b-q8_0"].size_bytes)
    assert package.slice_count(package.estimate_compressed_bytes(models_bytes)) == 3
    assert package.slice_count(package.estimate_compressed_bytes(with_q8_0)) == 3
    assert package.needs_spanning(package.estimate_compressed_bytes(models_bytes)) is True


def test_disk_spanning_lines_off_by_default_and_on_when_asked():
    off = package.disk_spanning_lines(False)
    on = package.disk_spanning_lines(True)
    assert "DiskSpanning=yes" not in off
    assert "DiskSliceSize=" not in off
    assert "DiskSpanning=yes" in on
    assert f"DiskSliceSize={package.DISK_SLICE_SIZE_BYTES}" in on


def test_the_template_renders_both_ways_for_disk_spanning():
    off = package.render_iss(_template(), _values())
    assert "DiskSpanning=yes" not in off
    on = package.render_iss(_template(), _values(
        DISK_SPANNING_LINES=package.disk_spanning_lines(True)))
    assert "DiskSpanning=yes" in on
    assert f"DiskSliceSize={package.DISK_SLICE_SIZE_BYTES}" in on


def test_installer_parts_is_just_the_exe_when_not_spanned(tmp_path):
    installer = tmp_path / "Spells-Setup-0.1.0.exe"
    installer.write_bytes(b"a")
    assert package.installer_parts(installer) == [installer]


def test_installer_parts_finds_its_own_slices_in_numeric_order(tmp_path):
    installer = tmp_path / "Spells-Setup-0.1.0.exe"
    installer.write_bytes(b"a")
    (tmp_path / "Spells-Setup-0.1.0-3.bin").write_bytes(b"c")
    (tmp_path / "Spells-Setup-0.1.0-2.bin").write_bytes(b"b")
    (tmp_path / "Spells-Setup-0.1.0.zip").write_bytes(b"not a slice")

    parts = package.installer_parts(installer)

    assert [part.name for part in parts] == [
        "Spells-Setup-0.1.0.exe", "Spells-Setup-0.1.0-2.bin", "Spells-Setup-0.1.0-3.bin"]


def test_zip_installer_bundles_every_slice_into_one_zip(tmp_path):
    installer = tmp_path / "Spells-Setup-0.1.0.exe"
    installer.write_bytes(b"a" * 10)
    (tmp_path / "Spells-Setup-0.1.0-2.bin").write_bytes(b"b" * 20)

    dest = package.zip_installer(installer, "0.1.0")

    assert dest == tmp_path / "Spells-Setup-0.1.0.zip"
    with zipfile.ZipFile(dest) as archive:
        assert sorted(archive.namelist()) == [
            "Spells-Setup-0.1.0-2.bin", "Spells-Setup-0.1.0.exe"]
        assert archive.read("Spells-Setup-0.1.0.exe") == b"a" * 10
        assert archive.read("Spells-Setup-0.1.0-2.bin") == b"b" * 20


def test_zip_installer_overwrites_a_stale_zip(tmp_path):
    installer = tmp_path / "Spells-Setup-0.1.0.exe"
    installer.write_bytes(b"a")
    stale = tmp_path / "Spells-Setup-0.1.0.zip"
    stale.write_bytes(b"stale")

    dest = package.zip_installer(installer, "0.1.0")

    with zipfile.ZipFile(dest) as archive:
        assert archive.namelist() == ["Spells-Setup-0.1.0.exe"]


def test_the_span_flag_is_recognised_by_the_cli(capsys):
    with pytest.raises(SystemExit):
        package.main(["--span"])
    assert "unrecognized" not in capsys.readouterr().err


def test_default_bundle_contents_are_unaffected_by_spanning():
    """Spanning packages a bigger installer; it must never change what ships (task rule)."""
    bundle = package.default_bundle()
    assert [model.id for model in bundle.models] == [
        "qwen3-asr-0.6b-q4_k_m", "whisper-large-v3-turbo-q8_0", "gemma-4-e2b-it-q4_0"]


# ----------------------------------------------------------------------------- the online flavour


ONLINE_TEMPLATE_PATH = BUILD_DIR / "spells-online.iss.template"

WHISPER = "whisper-large-v3-turbo-q8_0"
QWEN_ASR = "qwen3-asr-0.6b-q4_k_m"
QWEN_ASR_Q8 = "qwen3-asr-0.6b-q8_0"


def _online_template() -> str:
    return ONLINE_TEMPLATE_PATH.read_text(encoding="utf-8")


def _online_values(**overrides) -> dict:
    values = package.online_iss_values(
        version="0.1.0",
        output_dir=Path(r"C:\repo\dist"),
        dist_dir=Path(r"C:\repo\dist\Spells-Online"),
        manifest=package.online_manifest(),
        rows=package.install_sets(),
        cleanup_model_name="gemma-4-E2B-it-Q4_0.gguf",
        licenses_dir=Path(r"C:\repo\build\licenses"),
    )
    values.update(overrides)
    return values


def _rendered(**overrides) -> str:
    return package.render_iss(_online_template(), _online_values(**overrides))


def _by_name(manifest) -> dict:
    return {item.name: item for item in manifest}


# --- the manifest generated from the pins and the catalog


def test_every_manifest_entry_carries_what_the_download_page_needs():
    for item in package.online_manifest():
        assert item.url.startswith("https://")
        assert item.name and "/" not in item.name and "\\" not in item.name
        assert len(item.sha256) == 64 and item.sha256 == item.sha256.lower()
        assert item.size_bytes > 0
        assert item.model_id and item.purpose


def test_the_manifest_holds_every_file_any_machine_may_need():
    names = [item.name for item in package.online_manifest()]

    assert names == [
        "ggml-silero-v5.1.2.bin",
        "ggml-large-v3-turbo-q8_0.bin",
        "Qwen3-ASR-0.6B-Q4_K_M.gguf",
        "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf",
        "gemma-4-E2B-it-Q4_0.gguf",
        "mtp-gemma-4-E2B-it-Q4_0.gguf",
        "Qwen3.5-4B-Q4_K_M.gguf",
    ]


def test_the_q8_0_file_is_no_longer_downloaded_and_its_projector_is_listed_once():
    names = [item.name for item in package.online_manifest()]

    assert "Qwen3-ASR-0.6B-Q8_0.gguf" not in names
    assert names.count("mmproj-Qwen3-ASR-0.6B-Q8_0.gguf") == 1
    for row in package.install_sets():
        assert QWEN_ASR_Q8 not in row.plan.model_ids
        assert "Qwen3-ASR-0.6B-Q8_0.gguf" not in row.files


def test_the_q4_k_m_file_comes_from_the_projects_models_release():
    entry = _by_name(package.online_manifest())["Qwen3-ASR-0.6B-Q4_K_M.gguf"]

    assert entry.url == (
        "https://github.com/kapidolli/spells/releases/download/models-v1/Qwen3-ASR-0.6B-Q4_K_M.gguf")
    assert "huggingface" not in entry.url
    assert entry.sha256 == "3f711631ac7b81f223e08471838f8129b9716e4fe9935ebf99c58e320fc18633"
    assert entry.size_bytes == 484_215_744
    assert package.download_sources("https://github.com/kapidolli/spells/releases/download/models-v1/",
                                     entry.name,
                                     entry.url) == (entry.url,)


def test_a_manifest_entry_is_chosen_by_its_own_file_not_a_shared_extra_file():
    shared = "mmproj.gguf"
    catalog = _catalog(
        {**_model("big", "asr", {"en": 90}, gpu=400, cpu=1300, size=30),
         "file": "big.gguf", "extra_files": [shared]},
        {**_model("small", "asr", {"en": 90}, gpu=400, cpu=1100, size=20),
         "file": "small.gguf", "extra_files": [shared]},
        _model("c", "cleanup", {"en": 90}, gpu=100, cpu=100),
    )
    sizes = {"big.gguf": 20, "small.gguf": 10, shared: 10, "c.gguf": 1,
             "ggml-silero-v5.1.2.bin": 1}
    pins = {"models": {name: {"url": f"https://example/{name}", "sha256": "a" * 64,
                              "size_bytes": size}
                       for name, size in sizes.items()}}
    rows = package.install_sets(("en",), catalog, pins)
    manifest = package.online_manifest(("en",), catalog, pins, rows)

    assert [item.name for item in manifest] == [
        "ggml-silero-v5.1.2.bin", "small.gguf", shared, "c.gguf"]
    assert _by_name(manifest)[shared].model_id == "small"


def test_the_manifest_names_one_file_per_entry_even_when_two_models_share_it():
    """Qwen3.5 4B is a cleanup entry and a compose entry; it is downloaded once."""
    names = [item.name for item in package.online_manifest()]

    assert names.count("Qwen3.5-4B-Q4_K_M.gguf") == 1
    assert names.count("gemma-4-E2B-it-Q4_0.gguf") == 1


def test_a_purpose_says_what_goes_missing_when_a_download_is_skipped():
    entries = _by_name(package.online_manifest())

    assert entries["ggml-silero-v5.1.2.bin"].purpose == package.VAD_PURPOSE
    assert entries["gemma-4-E2B-it-Q4_0.gguf"].purpose == package.CLEANUP_PURPOSE
    assert entries["Qwen3.5-4B-Q4_K_M.gguf"].purpose == package.CLEANUP_PURPOSE
    assert entries["Qwen3-ASR-0.6B-Q4_K_M.gguf"].purpose == (
        "speech recognition for English and German")
    assert entries["ggml-large-v3-turbo-q8_0.bin"].purpose == (
        "speech recognition for English, German and Albanian")


def test_a_speech_model_installed_beside_albanian_is_not_said_to_recognise_it():
    """A processor ticking all three languages carries Qwen3-ASR for English and German only."""
    rows = package.install_sets()
    by_id = {model.id: model for model in load_catalog()}

    assert any("sq" in row.languages and "Qwen3-ASR-0.6B-Q4_K_M.gguf" in row.files
               for row in rows)
    assert package.model_languages(by_id[QWEN_ASR], rows) == ("en", "de")
    assert package.model_languages(by_id[WHISPER], rows) == ("en", "de", "sq")


def test_a_models_extra_files_carry_the_same_purpose_as_its_own_file():
    entries = _by_name(package.online_manifest())
    base = entries["Qwen3-ASR-0.6B-Q4_K_M.gguf"]
    mmproj = entries["mmproj-Qwen3-ASR-0.6B-Q8_0.gguf"]

    assert mmproj.model_id == base.model_id
    assert mmproj.purpose == base.purpose


def test_the_manifest_urls_and_hashes_are_the_pinned_ones():
    pinned = package.pinned_downloads()
    for item in package.online_manifest():
        entry = pinned[item.name.lower()]
        assert item.url == entry["url"]
        assert item.sha256 == entry["sha256"]
        assert item.size_bytes == entry["size_bytes"]


def test_the_manifest_covers_everything_the_offline_build_ships_and_a_little_more():
    """Every offline file is downloadable; the extra one is the graphics-card writing model."""
    bundle = package.default_bundle()
    shipped = {name for model in bundle.models for name in model.files}
    shipped.add(package.pinned_file_name(package.VAD_PIN))
    downloadable = {item.name for item in package.online_manifest()}

    assert shipped < downloadable
    assert downloadable - shipped == {"Qwen3.5-4B-Q4_K_M.gguf"}


def test_a_pin_without_a_size_cannot_reach_the_language_page():
    pins = {"models": {"x": {"url": "https://example/x.bin", "sha256": "a" * 64}}}

    with pytest.raises(package.PackageError, match="size_bytes"):
        package.pinned_download("x.bin", pins)


def test_a_pin_without_a_hash_cannot_reach_the_download_page():
    pins = {"models": {"x": {"url": "https://example/x.bin", "size_bytes": 12}}}

    with pytest.raises(package.PackageError, match="sha256"):
        package.pinned_download("x.bin", pins)


def test_an_unpinned_file_is_named_rather_than_silently_left_out():
    with pytest.raises(package.PackageError, match="pinned url"):
        package.pinned_download("nobody-pinned-this.gguf", {"models": {}})


def test_pinned_sizes_that_disagree_with_the_catalog_stop_the_build():
    pins = {"models": {}}
    for name, entry in (package.fetch.load_pins().get("models") or {}).items():
        copy = dict(entry)
        if str(copy.get("url", "")).endswith("mtp-gemma-4-E2B-it-Q4_0.gguf"):
            copy["size_bytes"] = copy["size_bytes"] + 1
        pins["models"][name] = copy

    with pytest.raises(package.PackageError, match="add up to"):
        package.online_manifest(pins=pins)


# --- what each machine installs, generated from the catalog

CPU = Hardware.CPU
GPU = Hardware.GPU
QWEN35_4B = "qwen3.5-4b-q4_k_m"
GEMMA4 = "gemma-4-e2b-it-q4_0"


def _row(hardware, *ticked):
    return package.install_set_for(package.install_sets(), hardware, ticked)


def test_every_machine_and_every_language_set_has_a_row():
    rows = package.install_sets()

    assert len(rows) == 14
    assert {row.hardware for row in rows} == {CPU, GPU}
    for hardware in (CPU, GPU):
        sets = {row.languages for row in rows if row.hardware is hardware}
        assert sets == set(package.language_subsets(package.ONLINE_LANGUAGES))


@pytest.mark.parametrize(
    "hardware, ticked, model_ids",
    [
        (CPU, ("en",), [QWEN_ASR, GEMMA4]),
        (CPU, ("de",), [QWEN_ASR, GEMMA4]),
        (CPU, ("en", "de"), [QWEN_ASR, GEMMA4]),
        (CPU, ("sq",), [WHISPER, GEMMA4]),
        (CPU, ("en", "sq"), [WHISPER, QWEN_ASR, GEMMA4]),
        (CPU, ("en", "de", "sq"), [QWEN_ASR, WHISPER, GEMMA4]),
        (GPU, ("en",), [QWEN_ASR, QWEN35_4B]),
        (GPU, ("en", "de"), [QWEN_ASR, QWEN35_4B]),
        (GPU, ("sq",), [WHISPER, QWEN35_4B]),
        (GPU, ("en", "de", "sq"), [WHISPER, QWEN35_4B]),
    ],
)
def test_each_row_names_the_models_that_machine_needs(hardware, ticked, model_ids):
    row = _row(hardware, *ticked)
    kept = [model.id for model in row.plan.models if model.kind is not ModelKind.COMPOSE]

    assert kept == model_ids


def test_a_graphics_card_gets_the_better_writing_model_and_a_processor_the_small_fast_one():
    graphics = _row(GPU, "en", "de", "sq")
    processor = _row(CPU, "en", "de", "sq")

    assert graphics.plan.writing.id == QWEN35_4B
    assert processor.plan.writing.id == GEMMA4
    assert graphics.reason.startswith("Your graphics card can run the better writing model")
    assert processor.reason.startswith("No graphics card was found")


def test_a_processor_that_also_dictates_albanian_keeps_the_faster_speech_model():
    """The Q4_K_M file ended the one trade the budget used to force here (B5-45, B5-46)."""
    for ticked in (("en", "de"), ("en", "sq"), ("de", "sq"), ("en", "de", "sq")):
        row = _row(CPU, *ticked)

        assert QWEN_ASR in [model.id for model in row.plan.models], ticked
        assert [model.id for model in row.plan.dropped] == [], ticked
        assert "alone" not in row.reason
        assert row.fits is True


def test_every_row_stays_inside_the_install_budget():
    rows = package.install_sets()

    for row in rows:
        assert row.footprint_bytes <= package.INSTALL_BUDGET_BYTES, row.languages
        assert row.fits is True
    package.check_install_budget(rows)


def test_a_machine_that_cannot_be_served_inside_the_budget_stops_the_build():
    catalog = _catalog(
        _model("huge-en", "asr", {"en": 90}, gpu=100, cpu=100, size=4_000_000_000),
        _model("huge-sq", "asr", {"sq": 90}, gpu=100, cpu=100, size=4_000_000_000),
        _model("c", "cleanup", {"en": 90, "sq": 90}, gpu=100, cpu=100),
    )
    pins = {"models": {name: {"url": f"https://example/{name}", "sha256": "a" * 64,
                              "size_bytes": 1}
                       for name in ("huge-en.bin", "huge-sq.bin", "c.gguf",
                                    "ggml-silero-v5.1.2.bin")}}
    rows = package.install_sets(("en", "sq"), catalog, pins)

    with pytest.raises(package.PackageError, match="budget of spec 19.4"):
        package.check_install_budget(rows)


def test_the_files_of_a_row_are_the_vad_model_and_what_that_machine_needs():
    assert _row(CPU, "sq").files == (
        "ggml-silero-v5.1.2.bin",
        "ggml-large-v3-turbo-q8_0.bin",
        "gemma-4-E2B-it-Q4_0.gguf",
        "mtp-gemma-4-E2B-it-Q4_0.gguf",
    )
    assert _row(CPU, "en").files == (
        "ggml-silero-v5.1.2.bin",
        "Qwen3-ASR-0.6B-Q4_K_M.gguf",
        "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf",
        "gemma-4-E2B-it-Q4_0.gguf",
        "mtp-gemma-4-E2B-it-Q4_0.gguf",
    )
    assert _row(CPU, "en", "de", "sq").files == (
        "ggml-silero-v5.1.2.bin",
        "Qwen3-ASR-0.6B-Q4_K_M.gguf",
        "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf",
        "ggml-large-v3-turbo-q8_0.bin",
        "gemma-4-E2B-it-Q4_0.gguf",
        "mtp-gemma-4-E2B-it-Q4_0.gguf",
    )
    assert _row(GPU, "en").files == (
        "ggml-silero-v5.1.2.bin",
        "Qwen3-ASR-0.6B-Q4_K_M.gguf",
        "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf",
        "Qwen3.5-4B-Q4_K_M.gguf",
    )


def test_the_mask_is_one_bit_per_offered_language():
    assert package.language_mask(("en",)) == 1
    assert package.language_mask(("de",)) == 2
    assert package.language_mask(("en", "sq")) == 5
    assert package.language_mask(("en", "de", "sq")) == 7
    assert {row.mask for row in package.install_sets() if row.hardware is CPU} == set(range(1, 8))


# --- the totals the language page shows


VAD_BYTES = 885_098
GEMMA4_BYTES = 2_841_481_184 + 59_235_872
QWEN35_4B_BYTES = 2_740_937_888
QWEN_ASR_BYTES = 484_215_744 + 214_392_480
WHISPER_BYTES = 874_188_075


@pytest.mark.parametrize(
    "hardware, ticked, expected",
    [
        (CPU, ("en",), VAD_BYTES + QWEN_ASR_BYTES + GEMMA4_BYTES),
        (CPU, ("en", "de"), VAD_BYTES + QWEN_ASR_BYTES + GEMMA4_BYTES),
        (CPU, ("sq",), VAD_BYTES + WHISPER_BYTES + GEMMA4_BYTES),
        (CPU, ("en", "de", "sq"), VAD_BYTES + QWEN_ASR_BYTES + WHISPER_BYTES + GEMMA4_BYTES),
        (GPU, ("en",), VAD_BYTES + QWEN_ASR_BYTES + QWEN35_4B_BYTES),
        (GPU, ("sq",), VAD_BYTES + WHISPER_BYTES + QWEN35_4B_BYTES),
        (GPU, ("en", "de", "sq"), VAD_BYTES + WHISPER_BYTES + QWEN35_4B_BYTES),
    ],
)
def test_the_download_total_per_machine_and_language_set(hardware, ticked, expected):
    assert _row(hardware, *ticked).download_bytes == expected


def test_the_download_is_smaller_than_it_was_when_both_machines_shared_one_file_set():
    """B5-27 downloaded 4,794,931,957 bytes for the whole language set, whatever the machine."""
    for row in package.install_sets():
        assert row.download_bytes <= 4_474_398_453
        assert row.download_bytes < 4_794_931_957


def test_a_processor_install_is_exactly_what_the_offline_build_ships():
    """The two flavours must never put different models on the same kind of machine."""
    bundle = package.default_bundle()
    shipped = {name for model in bundle.models for name in model.files}
    shipped.add(package.pinned_file_name(package.VAD_PIN))

    assert set(_row(CPU, "en", "de", "sq").files) == shipped
    assert _row(CPU, "en", "de", "sq").footprint_bytes == bundle.footprint_bytes


def test_language_subsets_are_every_combination_shortest_first():
    subsets = package.language_subsets(("en", "de", "sq"))

    assert len(subsets) == 7
    assert subsets[0] == ("en",)
    assert subsets[-1] == ("en", "de", "sq")
    assert len(set(subsets)) == 7


# --- where the files come from


@pytest.mark.parametrize(
    "base, expected",
    [
        ("", "https://pinned.example/a.bin"),
        ("   ", "https://pinned.example/a.bin"),
        ("https://mirror.example", "https://mirror.example/a.bin"),
        ("https://mirror.example/", "https://mirror.example/a.bin"),
        ("https://mirror.example///", "https://mirror.example/a.bin"),
        ("https://mirror.example/spells/models", "https://mirror.example/spells/models/a.bin"),
        ("https://mirror.example/spells/models/", "https://mirror.example/spells/models/a.bin"),
        ("  https://mirror.example/m/  ", "https://mirror.example/m/a.bin"),
        ("http://192.168.1.20:8080/spells", "http://192.168.1.20:8080/spells/a.bin"),
    ],
)
def test_a_mirror_address_is_joined_with_the_file_name(base, expected):
    assert package.model_url(base, "a.bin", "https://pinned.example/a.bin") == expected


def test_without_a_mirror_the_only_source_is_the_pinned_url():
    assert package.download_sources("", "a.bin", "https://pinned.example/a.bin") == (
        "https://pinned.example/a.bin",
    )


def test_with_a_mirror_the_pinned_url_stays_as_the_fallback_and_comes_second():
    assert package.download_sources(
        "https://mirror.example/m", "a.bin", "https://pinned.example/a.bin"
    ) == ("https://mirror.example/m/a.bin", "https://pinned.example/a.bin")


def test_a_mirror_that_resolves_to_the_pinned_url_is_not_tried_twice():
    assert package.download_sources(
        "https://pinned.example", "a.bin", "https://pinned.example/a.bin"
    ) == ("https://pinned.example/a.bin",)


# --- the generated Inno Setup script


def test_the_online_template_asks_for_exactly_the_values_the_build_computes():
    assert package.placeholders_in(_online_template()) == set(_online_values())


def test_the_generated_script_has_no_placeholder_left():
    assert "@@" not in _rendered()


def test_the_generated_pascal_declares_one_array_entry_per_manifest_file():
    manifest = package.online_manifest()
    text = _rendered()

    assert f"SetArrayLength(ModelFiles, {len(manifest)});" in text
    for index, item in enumerate(manifest):
        assert f"ModelFiles[{index}].Url := '{item.url}';" in text
        assert f"ModelFiles[{index}].Name := '{item.name}';" in text
        assert f"ModelFiles[{index}].Hash := '{item.sha256}';" in text
        assert f"ModelFiles[{index}].Size := StrToInt64('{item.size_bytes}');" in text
        assert f"ModelFiles[{index}].Purpose := '{item.purpose}';" in text


def test_the_generated_pascal_declares_one_row_per_machine_and_language_set():
    manifest = package.online_manifest()
    rows = package.install_sets()
    index_of = {item.name: index for index, item in enumerate(manifest)}
    text = _rendered()

    assert f"SetArrayLength(InstallSets, {len(rows)});" in text
    for index, row in enumerate(rows):
        files = ",".join(str(index_of[name]) for name in row.files)
        hardware = 1 if row.hardware is Hardware.GPU else 0
        assert f"InstallSets[{index}].Hardware := {hardware};" in text
        assert f"InstallSets[{index}].Mask := {row.mask};" in text
        assert f"InstallSets[{index}].Files := '{files}';" in text
        assert f"InstallSets[{index}].Bytes := StrToInt64('{row.download_bytes}');" in text
        assert f"InstallSets[{index}].Reason := '{row.reason}';" in text


def test_the_installer_reads_its_own_machine_and_looks_the_row_up():
    """No model reasoning in Pascal: the machine class plus the ticked boxes name a row."""
    text = _rendered()

    assert "DetectedClass := ClassGraphics" in text
    assert "DetectedClass := ClassProcessor" in text
    assert "InstallSets[I].Hardware = Hardware" in text
    assert "InstallSets[I].Mask = Mask" in text
    assert "Result := Pos(',' + IntToStr(Index) + ',', ',' + InstallSets[Row].Files + ',') > 0;"         in text


def test_the_sizes_go_through_strtoint64_because_a_model_is_past_a_pascal_integer():
    """2,841,481,184 does not fit a signed 32-bit literal, and the page adds the files up."""
    assert "ModelFiles[4].Size := StrToInt64('2841481184');" in _rendered()


def test_the_generated_language_page_offers_the_three_languages_with_english_ticked():
    text = _rendered()

    assert "SetArrayLength(LanguageCodes, 3);" in text
    for index, (code, name) in enumerate((("en", "English"), ("de", "German"),
                                          ("sq", "Albanian"))):
        assert f"LanguageCodes[{index}] := '{code}';" in text
        assert f"LanguagePage.Add('{name}');" in text
    assert "LanguagePage.Values[0] := True;" in text
    assert "LanguagePage.Values[1] := False;" in text
    assert "LanguagePage.Values[2] := False;" in text


def test_the_language_page_says_more_languages_can_be_added_in_settings():
    text = _rendered()

    assert "Which languages will you dictate in?" in text
    assert "More languages can be added later in " in text
    assert "Settings, which downloads what they need then." in text


def test_the_language_page_shows_a_running_total_that_follows_the_checkboxes():
    text = _rendered()

    assert "LanguagePage.CheckListBox.OnClickCheck := @LanguagesClicked;" in text
    assert "procedure LanguagesClicked(Sender: TObject);" in text
    assert "SizeToText(Missing)" in text


def test_the_download_page_verifies_every_file_against_its_pinned_hash():
    text = _rendered()

    assert "DownloadPage := CreateDownloadPage(" in text
    assert "DownloadPage.Add(Urls[Attempt], ModelFiles[Index].Name, ModelFiles[Index].Hash);" \
        in text
    assert "DownloadPage.Download();" in text


def test_the_download_runs_from_the_ready_page_before_anything_is_installed():
    text = _rendered()

    assert "else if CurPageID = wpReady then" in text
    assert "Result := DownloadModels();" in text


def test_a_failed_download_offers_a_retry_going_on_without_the_model_or_stopping():
    text = _rendered()

    assert "Yes: try again." in text
    assert "No: install without it. Spells will start with " in text
    assert "ModelFiles[Index].Purpose" in text
    assert "Cancel: stop the installation." in text
    assert "The installation log records what went wrong." in text


def test_a_failed_download_names_the_offline_installer_as_the_way_out():
    text = _rendered()

    assert "OfflineInstallerName = 'Spells-Setup-0.1.0.zip';" in text
    assert "which carries every model already, is the one to use on a " in text


def test_every_address_the_wizard_could_show_is_only_ever_written_to_the_log():
    """The owner's rule (B5-48): the interface names models and sizes, never an address."""
    text = _rendered()
    allowed_starts = ("ModelFiles[", "DefaultMirrorUrl = ", "; ")
    for line in text.splitlines():
        stripped = line.strip()
        if "https://" not in stripped:
            continue
        assert stripped.startswith(allowed_starts), stripped
    shown = [line.strip() for line in text.splitlines()
             if ("MsgBox(" in line or "SetText(" in line or ".Caption :=" in line)]
    for line in shown:
        assert "Url" not in line, line
        assert "Mirror" not in line, line
    question = text[text.index("Question := 'Spells could not finish a download"):
                    text.index("Answer := SuppressibleMsgBox(Question")]
    assert "Url" not in question
    assert "ModelFiles[Index].Name" not in question


def test_the_failure_detail_goes_to_the_log_with_the_address_and_the_reason():
    text = _rendered()
    body = text[text.index("function DownloadOne("):text.index("function WantedCount(")]

    assert "Log('Spells: ' + ModelFiles[Index].Name + ' failed from ' + Urls[Attempt] + ': '" \
        in body
    assert "Log('Spells: trying the next address for ' + ModelFiles[Index].Name);" in body


def test_every_address_is_tried_before_the_user_is_asked_anything():
    """A mirror that fails falls back to the pinned address silently, as B5-48 asks."""
    text = _rendered()
    body = text[text.index("function DownloadOne("):text.index("function WantedCount(")]

    assert body.index("for Attempt := 0 to Count - 1 do") < body.index("Question :=")
    assert body.count("SuppressibleMsgBox") == 1


def test_the_download_page_says_what_is_being_downloaded_and_how_far_it_has_got():
    text = _rendered()

    assert "CurrentLabel := ModelFiles[I].Title + ', ' + SizeToText(ModelFiles[I].ModelSize);" \
        in text
    assert "DownloadPage.SetText(CurrentLabel," in text
    assert "'Part ' + IntToStr(PartNumber) + ' of ' + IntToStr(PartCount) + ', ' +" in text
    assert "SizeToText(DoneBytes) + ' of ' + SizeToText(TotalBytes) + ' in all'" in text


def test_the_titles_are_the_words_the_app_uses_for_its_models():
    entries = _by_name(package.online_manifest())

    assert entries["ggml-silero-v5.1.2.bin"].title == "Voice detection model"
    assert entries["Qwen3-ASR-0.6B-Q4_K_M.gguf"].title == "Speech model for English and German"
    assert entries["ggml-large-v3-turbo-q8_0.bin"].title == (
        "Speech model for English, German and Albanian")
    assert entries["gemma-4-E2B-it-Q4_0.gguf"].title == "Writing and cleanup model"
    assert entries["Qwen3.5-4B-Q4_K_M.gguf"].title == "Writing and cleanup model"
    for item in package.online_manifest():
        assert "http" not in item.title
        assert ".gguf" not in item.title and ".bin" not in item.title


def test_an_extra_file_shows_the_size_of_the_whole_model_it_belongs_to():
    entries = _by_name(package.online_manifest())
    base = entries["Qwen3-ASR-0.6B-Q4_K_M.gguf"]
    mmproj = entries["mmproj-Qwen3-ASR-0.6B-Q8_0.gguf"]

    assert base.model_bytes == mmproj.model_bytes == 698_608_224
    assert base.size_bytes + mmproj.size_bytes == base.model_bytes


def test_the_wizard_has_no_page_that_asks_where_the_models_come_from():
    text = _rendered()

    assert "SourcePage" not in text
    assert "MirrorEdit" not in text
    assert "Where should the models come from?" not in text
    assert "MODELSOURCE" not in text


def test_the_source_is_fixed_when_the_installer_is_built():
    assert "DefaultMirrorUrl = '';" in _rendered()
    prefilled = package.render_iss(
        _online_template(),
        package.online_iss_values(
            version="0.1.0", output_dir=Path("out"), dist_dir=Path("dist"),
            manifest=package.online_manifest(), rows=package.install_sets(),
            cleanup_model_name="gemma-4-E2B-it-Q4_0.gguf",
            licenses_dir=Path("licenses"), mirror="  https://mirror.example/spells  "))

    assert "DefaultMirrorUrl = 'https://mirror.example/spells';" in prefilled
    assert "JoinModelUrl(DefaultMirrorUrl, ModelFiles[Index].Name);" in prefilled


def test_the_finish_page_says_the_models_were_downloaded_and_verified():
    text = _rendered()

    assert "FinishedLabel=Setup has finished installing Spells on your computer. The models it " \
        "needs were downloaded and checked against the hash recorded for each one." in text
    assert "https://" not in text[text.index("[Messages]"):text.index("[InstallDelete]")]


def test_the_pascal_joins_a_mirror_address_the_same_way_the_build_does():
    text = _rendered()

    assert "function JoinModelUrl(const BaseUrl, FileName: String): String;" in text
    assert "while (Length(Base) > 0) and (Base[Length(Base)] = '/') do" in text
    assert "Result := Base + '/' + FileName;" in text


def test_the_pascal_tries_the_mirror_first_and_the_pinned_url_last():
    text = _rendered()
    body = text[text.index("function SourcesFor("):text.index("procedure NoteSkipped(")]

    assert body.index("Urls[0] := Mirrored;") < body.index("Urls[1] := ModelFiles[Index].Url;")


def test_the_downloaded_files_are_moved_into_the_app_folder():
    text = _rendered()

    assert "if RenameFile(Source, Target) then" in text
    assert "else if FileCopy(Source, Target, False) then" in text
    assert "MoveDownloadedModels();" in text


def test_the_installer_seeds_the_first_run_language_file_the_app_reads():
    text = _rendered()

    assert "WriteFirstRunFile();" in text
    assert '"enabled_languages": [' in text
    assert "SaveStringToFile(ExpandConstant(" in text
    assert "first-run.json" in text


def test_the_uninstaller_removes_every_downloadable_model_and_the_first_run_file():
    text = _rendered()

    for item in package.online_manifest():
        assert f'Type: files; Name: "{{app}}\\models\\{item.name}"' in text
    assert 'Type: files; Name: "{app}\\first-run.json"' in text
    assert 'Type: dirifempty; Name: "{app}\\models"' in text
    assert 'Type: filesandordirs; Name: "{app}"' in text


def test_the_online_installer_stays_per_user_with_no_elevation():
    text = _rendered()

    assert "PrivilegesRequired=lowest" in text
    assert r"DefaultDirName={localappdata}\Programs\Spells" in text
    assert "AppId={{9D5B0F4E-3E2A-4C7A-9C3E-2A1D6B8F41C7}" in text
    assert "DiskSpanning" not in text


def test_the_online_installer_keeps_the_quit_step_and_the_uninstall_question():
    text = _rendered()

    assert "function PrepareToInstall(var NeedsRestart: Boolean): String;" in text
    assert "CloseApplications=yes" in text
    assert "RestartApplications=no" in text
    assert "function AskToRemoveUserData(): Boolean;" in text
    assert "{param:REMOVEDATA|0}" in text


def test_the_online_installer_sets_no_appmutex_for_the_same_reason_the_offline_one_does_not():
    directives = [line for line in _rendered().splitlines()
                  if line.strip().lower().startswith("appmutex")]

    assert directives == []


def test_the_online_script_carries_no_model_file_in_its_files_section():
    text = _rendered()
    files = text[text.index("[Files]"):text.index("[Icons]")]

    assert "models" not in files
    assert ".gguf" not in files
    assert ".bin" not in files


def test_the_online_dist_folder_holds_everything_but_the_models():
    plan = package.online_dist_plan(
        app_dir=Path(r"C:\repo\build\out\app\Spells"),
        engines_dir=Path(r"C:\repo\build\out\engines"),
        data_dir=Path(r"C:\repo\data"))

    assert [item.dest for item in plan] == [
        ".", "engines/vulkan", "engines/cpu", "data", "licenses"]
    assert package.ONLINE_DIST_TOP_LEVEL == (
        "Spells.exe", "_internal", "data", "engines", "licenses")
    assert "models" not in package.ONLINE_DIST_TOP_LEVEL


def test_the_online_top_level_check_refuses_a_stray_models_folder():
    with pytest.raises(package.PackageError, match="unexpected"):
        package.check_dist_top_level(
            [*package.ONLINE_DIST_TOP_LEVEL, "models"], package.ONLINE_DIST_TOP_LEVEL,
            package.DIST_ONLINE_DIR, package.ONLINE_ISS_TEMPLATE)


def test_the_two_installers_have_different_names_so_they_sit_side_by_side():
    assert package.installer_basename("0.1.0") == "Spells-Setup-0.1.0"
    assert package.online_installer_basename("0.1.0") == "Spells-Online-Setup-0.1.0"
    assert package.DIST_ONLINE_DIR != package.DIST_APP_DIR


def test_an_apostrophe_in_a_generated_string_is_doubled_for_pascal():
    assert package.pascal_string("it's") == "'it''s'"
    assert package.pascal_string("") == "''"


def test_language_names_read_as_a_sentence():
    assert package.language_phrase(()) == ""
    assert package.language_phrase(("en",)) == "English"
    assert package.language_phrase(("en", "de")) == "English and German"
    assert package.language_phrase(("en", "de", "sq")) == "English, German and Albanian"


# --- the command line


def test_the_online_flag_is_recognised_by_the_cli(capsys):
    with pytest.raises(SystemExit):
        package.main(["--online", "--step", "size", "--help"])
    assert "unrecognized" not in capsys.readouterr().err


def test_all_and_online_cannot_be_asked_for_at_once(capsys):
    with pytest.raises(SystemExit):
        package.main(["--all", "--online"])
    assert "one after the other" in capsys.readouterr().err


def test_a_dev_cleanup_model_has_no_place_in_an_online_build(capsys):
    with pytest.raises(SystemExit):
        package.main(["--online", "--cleanup-model", "x.gguf"])
    assert "no place for --cleanup-model" in capsys.readouterr().err


def test_a_mirror_without_the_online_flag_is_refused(capsys):
    with pytest.raises(SystemExit):
        package.main(["--all", "--model-mirror", "https://mirror.example"])
    assert "only means something for --online" in capsys.readouterr().err


# The version file of spec 19.7 ------------------------------------------------------------------

CHANGELOG_SAMPLE = """# Changelog

## [Unreleased]

### Added

- Nothing yet.

## [0.2.0] - 2026-09-18

### Added

- Spells can now tell you when a new version is out.
- A logo and an app icon of its own.

### Fixed

- The level meter moves with your voice again.
  It was reading raw loudness.

## [0.1.0] - 2026-09-16

- The first release.
"""


def test_the_change_lines_come_out_of_the_changelog_in_order():
    released, changes = package.changelog_release(CHANGELOG_SAMPLE, "0.2.0")

    assert released == "2026-09-18"
    assert changes == (
        "Spells can now tell you when a new version is out.",
        "A logo and an app icon of its own.",
        "The level meter moves with your voice again. It was reading raw loudness.",
    )


def test_the_section_of_another_version_is_not_read():
    _released, changes = package.changelog_release(CHANGELOG_SAMPLE, "0.1.0")

    assert changes == ("The first release.",)


def test_a_version_with_no_section_fails_the_build():
    with pytest.raises(package.PackageError):
        package.changelog_release(CHANGELOG_SAMPLE, "9.9.9")


def test_a_section_with_no_change_lines_fails_the_build():
    text = "# Changelog\n\n## [0.3.0] - 2026-10-01\n\n### Added\n\n"

    with pytest.raises(package.PackageError):
        package.changelog_release(text, "0.3.0")


def test_a_section_date_that_is_not_a_date_fails_the_build():
    text = "# Changelog\n\n## [0.3.0] - soon\n\n- Something.\n"

    with pytest.raises(package.PackageError):
        package.changelog_release(text, "0.3.0")


def test_a_change_line_longer_than_the_app_will_read_fails_the_build():
    text = f"# Changelog\n\n## [0.3.0] - 2026-10-01\n\n- {'x' * 400}\n"

    with pytest.raises(package.PackageError):
        package.changelog_release(text, "0.3.0")


def test_the_repository_changelog_has_a_section_for_this_version():
    from spells import __version__

    released, changes = package.changelog_release(
        package.CHANGELOG_PATH.read_text(encoding="utf-8"), __version__)

    assert released
    assert changes


def test_the_repository_changelog_uses_no_em_dash():
    assert "\u2014" not in package.CHANGELOG_PATH.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("base", "expected"),
    [
        ("https://spells.example.com", "https://spells.example.com/latest.json"),
        ("https://spells.example.com/", "https://spells.example.com/latest.json"),
        ("https://x.example.com/spells/", "https://x.example.com/spells/latest.json"),
    ],
)
def test_the_address_is_joined_the_same_way_every_time(base, expected):
    assert package.join_url(base, package.VERSION_FILE_NAME) == expected


@pytest.mark.parametrize(
    "base",
    ["", "spells.example.com", "file:///c:/spells/", "https:///latest", "https://x.example.com/?a=b"],
)
def test_an_update_source_that_is_not_an_address_is_refused(base):
    with pytest.raises(package.PackageError):
        package.check_update_source(base)


def test_the_version_file_is_measured_from_the_installer_this_build_made(tmp_path):
    installer = tmp_path / "Spells-Online-Setup-0.2.0.exe"
    payload = b"MZ" + b"x" * 5000
    installer.write_bytes(payload)

    body = package.version_manifest(
        version="0.2.0", released="2026-09-18", changes=["One line."],
        installer=installer, base_url="https://spells.example.com/")

    assert body["schema"] == package.VERSION_FILE_SCHEMA
    assert body["product"] == "Spells"
    assert body["version"] == "0.2.0"
    assert body["released"] == "2026-09-18"
    assert body["minimum_version"] == ""
    assert body["installer"]["url"] == (
        "https://spells.example.com/Spells-Online-Setup-0.2.0.exe")
    assert body["installer"]["size_bytes"] == len(payload)
    assert body["installer"]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert body["changes"] == ["One line."]


def test_the_version_file_the_build_writes_is_one_the_app_accepts(tmp_path):
    installer = tmp_path / "Spells-Online-Setup-0.2.0.exe"
    installer.write_bytes(b"MZ" + b"y" * 4000)
    base = "https://spells.example.com/"
    released, changes = package.changelog_release(CHANGELOG_SAMPLE, "0.2.0")

    body = package.version_manifest(version="0.2.0", released=released, changes=changes,
                                    installer=installer, base_url=base,
                                    minimum_version="0.1.0")

    release = updates.parse_manifest(json.dumps(body),
                                     source_url=package.join_url(base, "latest.json"))
    assert release.version == "0.2.0"
    assert release.changes == changes
    assert release.minimum_version == "0.1.0"
    assert updates.decide(release, "0.1.0").available


def test_the_app_folder_carries_the_address_and_nothing_else(tmp_path):
    (tmp_path / "data").mkdir()

    written = package.write_update_source(tmp_path, "https://spells.example.com/")

    assert written is not None
    assert updates.source_url(tmp_path / "data") == "https://spells.example.com/latest.json"


def test_a_build_without_an_update_source_leaves_no_address_behind(tmp_path):
    (tmp_path / "data").mkdir()
    package.write_update_source(tmp_path, "https://spells.example.com/")

    assert package.write_update_source(tmp_path, "") is None
    assert updates.source_url(tmp_path / "data") == ""


def test_the_version_file_lands_in_dist_and_is_reported(tmp_path, monkeypatch, capsys):
    installer = tmp_path / "Spells-Online-Setup-0.2.0.exe"
    installer.write_bytes(b"MZ" + b"z" * 900)
    monkeypatch.setattr(package, "DIST_DIR", tmp_path / "dist")
    monkeypatch.setattr(package, "CHANGELOG_PATH", tmp_path / "CHANGELOG.md")
    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG_SAMPLE, encoding="utf-8")

    target = package.step_latest("0.2.0", installer, "https://spells.example.com/")

    assert target == tmp_path / "dist" / "latest.json"
    assert json.loads(target.read_text(encoding="utf-8"))["version"] == "0.2.0"
    assert str(target) in capsys.readouterr().out


def test_no_version_file_is_written_without_an_update_source(tmp_path, monkeypatch):
    monkeypatch.setattr(package, "DIST_DIR", tmp_path / "dist")
    installer = tmp_path / "Spells-Online-Setup-0.2.0.exe"
    installer.write_bytes(b"MZ")

    assert package.step_latest("0.2.0", installer, "") is None
    assert not (tmp_path / "dist").exists()


def test_a_spanned_installer_is_refused_as_an_update_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(package, "DIST_DIR", tmp_path / "dist")
    monkeypatch.setattr(package, "CHANGELOG_PATH", tmp_path / "CHANGELOG.md")
    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG_SAMPLE, encoding="utf-8")
    installer = tmp_path / "Spells-Setup-0.2.0.exe"
    installer.write_bytes(b"MZ")
    (tmp_path / "Spells-Setup-0.2.0-1.bin").write_bytes(b"slice")

    with pytest.raises(package.PackageError) as caught:
        package.step_latest("0.2.0", installer, "https://spells.example.com/")

    assert "one file" in str(caught.value)


def test_an_installer_that_was_not_built_is_reported_not_invented(tmp_path, monkeypatch):
    monkeypatch.setattr(package, "DIST_DIR", tmp_path / "dist")

    assert package.step_latest("0.2.0", tmp_path / "nothing.exe",
                               "https://spells.example.com/") is None


def test_the_version_file_step_is_the_last_one():
    assert package.STEPS[-1] == "latest"


# What the online installer does on an update (spec 19.7) ----------------------------------------


def test_a_model_already_in_the_folder_is_never_downloaded_again():
    text = _rendered()

    assert "function AlreadyInstalled(Index: Integer): Boolean;" in text
    assert "function NeedsDownload(Index: Integer): Boolean;" in text
    assert "Result := FileWanted(Index) and not AlreadyInstalled(Index);" in text
    assert "Result := Size = ModelFiles[Index].Size;" in text
    assert "if NeedsDownload(I) then" in text


def test_an_update_that_needs_no_model_shows_no_download_page():
    text = _rendered()

    assert "TotalBytes := MissingBytes();" in text
    assert "if PartCount = 0 then" in text
    assert "every model this choice needs is already in the models folder" in text


@pytest.mark.parametrize("template", [package.ISS_TEMPLATE, package.ONLINE_ISS_TEMPLATE])
def test_an_update_started_from_inside_spells_brings_the_app_back(template):
    text = template.read_text(encoding="utf-8")

    assert "function RelaunchRequested(): Boolean;" in text
    assert "for Index := 1 to ParamCount do" in text
    assert "CompareText(ParamStr(Index), '/RELAUNCH') = 0" in text
    assert "CmdLineParamExists" not in text
    assert 'Flags: nowait; Check: RelaunchRequested' in text


def test_the_app_starts_the_installer_with_the_switches_the_templates_answer_to():
    assert "/SILENT" in updates.INSTALL_ARGS
    assert "/SUPPRESSMSGBOXES" in updates.INSTALL_ARGS
    assert "/NORESTART" in updates.INSTALL_ARGS
    assert "/RELAUNCH" in updates.INSTALL_ARGS


def test_signable_files_are_the_programs_outside_the_models_folder(tmp_path):
    for name in ("Spells.exe", "_internal/a.pyd", "_internal/b.DLL", "engines/cpu/llama-server.exe",
                 "models/weights.gguf", "models/stray.exe", "data/readme.txt", "licenses/x.txt"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"MZ")

    found = {path.relative_to(tmp_path).as_posix() for path in package.signable_files(tmp_path)}

    assert found == {"Spells.exe", "_internal/a.pyd", "_internal/b.DLL",
                     "engines/cpu/llama-server.exe"}


def test_the_sign_script_takes_the_file_as_inno_setup_passes_it():
    command = f"powershell -NoProfile -File {BUILD_DIR / 'sign.ps1'} $f"
    target = Path("C:/dist/Spells/Spells.exe")

    args = package.signtool_args(command, target)

    assert args[-1] == str(target)
    assert args[:3] == ["powershell", "-NoProfile", "-File"]
