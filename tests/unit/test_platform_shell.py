from __future__ import annotations

from pathlib import Path

import pytest
from PySide6 import QtWidgets

from spells import audio, pipeline
from spells.platform.base import SETTINGS_MICROPHONE_PRIVACY, SETTINGS_SOUND
from spells.platform.stub import StubShell
from spells.ui import recordings
from spells.ui.diagnostics import DiagnosticsTab
from spells.ui.tray import OPEN_SETTINGS_HINT, Tray
from spells.ui.uploadpage import TOKEN_HINT, TOKEN_UNAVAILABLE
from spells.uploadtoken import DESCRIPTION, TOKEN_FILE, TokenStore

from .fake_platform import ReversibleSecrets, fake_platform
from .test_ui_support import (
    SELECTION,
    FakeEngines,
    FakeHotkey,
    FakePipeline,
    Messages,
    Notice,
    event,
    make_config,
    qt_app,
)
from .test_ui_upload import make_page

TOKEN = "a-very-secret-token-0123456789abcdef"
KINDS = (SETTINGS_SOUND, SETTINGS_MICROPHONE_PRIVACY)


@pytest.fixture(scope="module")
def app():
    return qt_app()


class RecordingShell(StubShell):
    def __init__(self, *, settings: bool = True, blocked: bool | None = None) -> None:
        self.settings_available = settings
        self.blocked = blocked
        self.opened: list[str] = []
        self.revealed: list[Path] = []
        self.settings: list[str] = []

    def open_path(self, path):
        self.opened.append(path)

    def reveal(self, path):
        self.revealed.append(path)

    def has_settings(self, kind):
        return self.settings_available and kind in KINDS

    def open_settings(self, kind):
        self.settings.append(kind)
        return self.has_settings(kind)

    def microphone_blocked(self):
        return self.blocked


class RefusingSecrets:
    def protect(self, data, description=""):
        raise OSError("the protected storage refused")

    def unprotect(self, blob):
        raise OSError("the protected storage refused")


class RecordingIcon(QtWidgets.QSystemTrayIcon):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[tuple] = []

    def showMessage(self, *args) -> None:
        self.messages.append(args)


def make_tray(tmp_path, *, launcher=None):
    icon = RecordingIcon()
    tray = Tray(
        config=make_config(tmp_path),
        pipeline=FakePipeline(),
        hotkey=FakeHotkey(),
        engines=FakeEngines(),
        icon=icon,
        launcher=launcher,
        light_taskbar=False,
    )
    return tray, icon


def mic_event(kind: str):
    return event(
        notice=Notice.ERROR,
        notice_text="Mic is busy",
        notification="Microphone busy: another app holds it",
        notification_action=kind,
    )


def test_the_pipeline_names_the_settings_by_kind():
    assert pipeline.SOUND_SETTINGS == SETTINGS_SOUND == "sound"
    assert pipeline.PRIVACY_SETTINGS == SETTINGS_MICROPHONE_PRIVACY == "microphone_privacy"


def test_the_tray_opens_a_sound_action_through_the_platform_shell(app, tmp_path, use_platform):
    shell = RecordingShell()
    use_platform(fake_platform(shell=shell))
    launched: list[str] = []
    tray, icon = make_tray(tmp_path, launcher=launched.append)

    tray.apply_event(mic_event(SETTINGS_SOUND))

    assert OPEN_SETTINGS_HINT in icon.messages[-1][1]
    assert "Microphone busy" in tray.tooltip
    icon.messageClicked.emit()
    icon.messageClicked.emit()
    assert shell.settings == ["sound"]
    assert launched == []


def test_the_tray_opens_the_microphone_privacy_settings_by_kind(app, tmp_path, use_platform):
    shell = RecordingShell()
    use_platform(fake_platform(shell=shell))
    tray, icon = make_tray(tmp_path, launcher=lambda path: None)

    tray.notify("Microphone blocked by the privacy settings", SETTINGS_MICROPHONE_PRIVACY)
    icon.messageClicked.emit()

    assert shell.settings == ["microphone_privacy"]


def test_without_system_settings_the_toast_offers_no_open_settings_hint(
    app, tmp_path, use_platform
):
    shell = RecordingShell(settings=False)
    use_platform(fake_platform(shell=shell))
    launched: list[str] = []
    tray, icon = make_tray(tmp_path, launcher=launched.append)

    tray.apply_event(mic_event(SETTINGS_MICROPHONE_PRIVACY))

    body = icon.messages[-1][1]
    assert body == "Microphone busy: another app holds it"
    assert OPEN_SETTINGS_HINT not in body
    assert "Microphone busy" in tray.tooltip
    icon.messageClicked.emit()
    assert shell.settings == []
    assert launched == []


def test_an_action_that_is_no_settings_kind_is_no_microphone_error(app, tmp_path, use_platform):
    use_platform(fake_platform(shell=RecordingShell()))
    tray, _icon = make_tray(tmp_path, launcher=lambda path: None)

    tray.apply_event(mic_event("ms-settings:sound"))

    assert "Microphone busy" not in tray.tooltip


def test_the_tray_launches_other_actions_through_the_platform_by_default(
    app, tmp_path, use_platform
):
    shell = RecordingShell()
    use_platform(fake_platform(shell=shell))
    tray, icon = make_tray(tmp_path)
    target = str(tmp_path / "somewhere")

    tray.notify("Something to open", target)
    icon.messageClicked.emit()

    assert shell.opened == [target]
    assert shell.settings == []


def test_the_diagnostics_log_folder_opens_through_the_platform_by_default(
    app, tmp_path, use_platform
):
    shell = RecordingShell()
    use_platform(fake_platform(shell=shell))
    log_dir = tmp_path / "logs"
    tab = DiagnosticsTab(
        config=make_config(tmp_path),
        engines=FakeEngines(),
        pipeline=FakePipeline(),
        hotkey=FakeHotkey(),
        gpu_selection=SELECTION,
        log_dir=log_dir,
        notify=Messages(),
        save_dialog=lambda default: None,
    )

    tab.open_log_button.click()

    assert shell.opened == [str(log_dir)]
    tab.close()


def test_show_in_folder_reveals_through_the_platform(tmp_path, use_platform):
    shell = RecordingShell()
    use_platform(fake_platform(shell=shell))
    path = tmp_path / "000001.wav"

    recordings.default_reveal(path)

    assert shell.revealed == [path]


@pytest.mark.parametrize("blocked", [True, False, None])
def test_the_microphone_privacy_check_asks_the_platform(blocked, use_platform):
    use_platform(fake_platform(shell=RecordingShell(blocked=blocked)))

    assert audio._microphone_privacy_denied() is blocked


def test_the_stub_cannot_tell_whether_the_microphone_is_blocked(use_platform):
    use_platform(fake_platform())

    assert audio._microphone_privacy_denied() is None


def test_the_token_store_round_trips_through_the_platform_secrets(tmp_path, use_platform):
    secrets = ReversibleSecrets()
    use_platform(fake_platform(secrets=secrets))
    path = tmp_path / TOKEN_FILE
    store = TokenStore(path)

    store.write(TOKEN)

    assert path.read_bytes() == ReversibleSecrets().protect(TOKEN.encode("utf-8"))
    assert TOKEN.encode("utf-8") not in path.read_bytes()
    assert TokenStore(path).read() == TOKEN
    assert secrets.descriptions == [DESCRIPTION]


def test_a_token_file_reads_as_no_token_where_secrets_are_unavailable(
    tmp_path, use_platform, caplog
):
    use_platform(fake_platform(secrets=ReversibleSecrets()))
    path = tmp_path / TOKEN_FILE
    TokenStore(path).write(TOKEN)
    use_platform(fake_platform())
    caplog.set_level("DEBUG")

    assert TokenStore(path).read() == ""
    assert TokenStore(path).exists()
    assert "could not be decrypted on this system" in caplog.text
    assert "Windows" not in caplog.text
    assert TOKEN not in caplog.text


def test_the_upload_page_says_a_token_cannot_be_saved_on_the_stub(app, tmp_path, use_platform):
    use_platform(fake_platform())

    page, world, *_ = make_page(tmp_path, enabled=True, url="https://example.com/spells")

    assert not page.token.isEnabled()
    assert page.token_row.description_label.text() == TOKEN_UNAVAILABLE
    assert TOKEN_UNAVAILABLE == "Saving a token is not available on this system yet."
    page.close()
    world.history.close()


def test_the_upload_page_keeps_the_token_field_where_secrets_work(app, tmp_path, use_platform):
    use_platform(fake_platform(secrets=ReversibleSecrets()))

    page, world, *_ = make_page(tmp_path, enabled=True, url="https://example.com/spells")
    page.token.setText(TOKEN)
    page.token.editingFinished.emit()

    assert page.token.isEnabled()
    assert page.token_row.description_label.text() == TOKEN_HINT
    assert world.tokens.read() == TOKEN
    page.close()
    world.history.close()


def test_a_failing_secrets_probe_keeps_the_token_field(app, tmp_path, use_platform, caplog):
    use_platform(fake_platform(secrets=RefusingSecrets()))

    page, world, *_ = make_page(tmp_path, enabled=True, url="https://example.com/spells")

    assert page.token.isEnabled()
    assert page.token_row.description_label.text() == TOKEN_HINT
    assert "the protected storage probe failed" in caplog.text
    page.close()
    world.history.close()


@pytest.mark.windows
def test_windows_builds_its_shell_and_secrets_from_win32():
    from spells.platform import windows

    record = windows.build()

    assert isinstance(record.shell, windows.WindowsShell)
    assert isinstance(record.secrets, windows.WindowsSecrets)
    assert record.shell.has_settings("sound") is True
    assert record.shell.has_settings("microphone_privacy") is True
    assert record.shell.has_settings("bluetooth") is False


@pytest.mark.windows
def test_windows_opens_settings_and_paths_with_startfile(monkeypatch):
    from spells.platform import windows
    from spells.win32 import shell as win32_shell

    launched: list[str] = []
    monkeypatch.setattr(win32_shell.os, "startfile", launched.append)
    shell = windows.build().shell

    assert shell.open_settings("sound") is True
    assert shell.open_settings("microphone_privacy") is True
    assert shell.open_settings("bluetooth") is False
    shell.open_path("C:\\Spells\\logs")

    assert launched == [
        "ms-settings:sound",
        "ms-settings:privacy-microphone",
        "C:\\Spells\\logs",
    ]


@pytest.mark.windows
def test_windows_reveals_a_file_in_explorer_detached(monkeypatch):
    from spells.platform import windows
    from spells.win32 import shell as win32_shell

    started: list[tuple] = []
    monkeypatch.setattr(
        win32_shell.subprocess, "Popen", lambda args, **kwargs: started.append((args, kwargs))
    )
    path = Path("C:\\Users\\me\\Spells\\recordings\\000001.wav")

    windows.build().shell.reveal(path)

    assert started == [
        (
            ["explorer.exe", "/select,C:\\Users\\me\\Spells\\recordings\\000001.wav"],
            {"close_fds": True, "creationflags": 0x00000008 | 0x00000200},
        )
    ]


@pytest.mark.windows
def test_windows_reads_the_microphone_privacy_switch_from_win32(monkeypatch):
    from spells.platform import windows
    from spells.win32 import shell as win32_shell

    shell = windows.build().shell
    assert shell.microphone_blocked() in (True, False, None)
    monkeypatch.setattr(win32_shell, "microphone_privacy_denied", lambda: True)

    assert shell.microphone_blocked() is True


@pytest.mark.windows
def test_windows_protects_secrets_with_dpapi(monkeypatch):
    from spells.platform import windows
    from spells.win32 import dpapi

    secrets = windows.build().secrets
    blob = secrets.protect(TOKEN.encode("utf-8"), DESCRIPTION)

    assert TOKEN.encode("utf-8") not in blob
    assert secrets.unprotect(blob) == TOKEN.encode("utf-8")
    calls: list[tuple] = []
    monkeypatch.setattr(
        dpapi, "protect", lambda data, description="": calls.append((data, description)) or b"x"
    )
    assert secrets.protect(b"abc", "why") == b"x"
    assert calls == [(b"abc", "why")]
