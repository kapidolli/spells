import sys

import pytest

from spells import platform
from spells.platform import stub
from spells.platform.base import Capabilities, Platform, PlatformUnavailable
from spells.platform.paths import system_name

from .fake_platform import fake_platform


def test_system_name_maps_the_three_systems():
    assert system_name("win32") == "windows"
    assert system_name("darwin") == "macos"
    assert system_name("linux") == "linux"


def test_current_is_built_once_and_cached():
    previous = platform.swap(None)
    try:
        first = platform.current()
        assert platform.current() is first
        assert first.name == system_name(sys.platform)
    finally:
        platform.swap(previous)


def test_swap_installs_a_record_and_returns_the_old_one():
    fake = fake_platform()
    previous = platform.swap(fake)
    try:
        assert platform.current() is fake
    finally:
        assert platform.swap(previous) is fake


def test_capabilities_everything_and_missing():
    assert Capabilities().missing() == tuple(
        name for name in Capabilities.__dataclass_fields__
    )
    assert Capabilities.everything().missing() == ()


def test_stub_has_no_capabilities():
    record = stub.build("linux")
    assert isinstance(record, Platform)
    assert record.name == "linux"
    assert record.capabilities == Capabilities()
    assert record.key_hook is None


def test_stub_focus_reports_no_window():
    focus = stub.build("macos").focus
    assert focus.foreground() == 0
    assert focus.app_name(0) == ""
    assert focus.title(0) == ""
    assert focus.is_elevated(0) is False


@pytest.mark.parametrize(
    "call",
    [
        lambda p: p.keyboard.type_unicode("x"),
        lambda p: p.keyboard.send_paste(),
        lambda p: p.clipboard.set_text("x"),
        lambda p: p.secrets.protect(b"x"),
        lambda p: p.processes.spawn_hidden(["x"], stdout_path=None, stderr_path=None),
        lambda p: p.updater.apply(None),
    ],
)
def test_stub_refuses_what_it_cannot_do(call):
    with pytest.raises(PlatformUnavailable):
        call(stub.build("linux"))


def test_stub_hotkeys_never_record_and_keep_their_chords():
    service = stub.build("linux").hotkeys((), None, on_error=None)
    service.start()
    assert service.recording is False
    assert service.chords == ()
    service.set_writing(7)
    assert service.writing_id == 7
    service.stop()


def test_stub_instance_always_acquires_and_finds_nothing():
    instance = stub.build("linux").instance
    assert instance.acquire("Spells") is True
    assert instance.signal_running("SpellsMessageWindow", "quit") is False
    assert instance.find_running("SpellsMessageWindow") == 0


def test_stub_processes_have_no_window_flags_and_no_pinning():
    processes = stub.build("linux").processes
    assert processes.hidden_process_kwargs() == {}
    plan = processes.cpu_plan()
    assert plan.threads is None and plan.affinity_mask is None


@pytest.mark.windows
def test_windows_platform_claims_every_capability():
    from spells.platform import windows

    record = windows.build()
    assert record.name == "windows"
    assert record.capabilities == Capabilities.everything()
