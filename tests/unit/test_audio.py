"""Unit tests for spells.audio against a fake sounddevice module.

The fake models the owner's machine: the same three microphones appear under MME
at 44.1 kHz with names cut at 31 characters, and under WASAPI at 48 kHz with
their full names. The fake WASAPI refuses any rate but the device's own, the way
shared mode does (PaErrorCode -9997), so the fallback to MME is exercised by the
same rule the real host API applies.
"""

from __future__ import annotations

import array
import io
import logging
import wave
from types import SimpleNamespace

import pytest

from spells import audio
from spells.audio import (
    AudioDevice,
    CapturePlan,
    MicError,
    Recorder,
    capture_host_api,
    capture_plans,
    channel_options,
    downmix_to_mono,
    list_input_devices,
    match_device,
    names_match,
    pcm16_to_wav,
)

SR = 48000
MME_RATE = 44100

PA_UNANTICIPATED_HOST_ERROR = -9999
PA_INVALID_CHANNEL_COUNT = -9998
PA_INVALID_SAMPLE_RATE = -9997
PA_INVALID_DEVICE = -9996
PA_DEVICE_UNAVAILABLE = -9985

OBSBOT_MME = "OBSBOT Tiny 2 Lite Microphone ("
OBSBOT_WASAPI = "OBSBOT Tiny 2 Lite Microphone (6- OBSBOT Tiny 2 Lite Audio)"
HYPERX_MME = "Microphone (HyperX Cloud III Wi"
HYPERX_WASAPI = "Microphone (HyperX Cloud III Wireless)"
USB = "Microphone (USB Audio Device)"


def pcm(samples) -> bytes:
    return array.array("h", samples).tobytes()


def silence(ms: int) -> bytes:
    return pcm([0] * (SR * ms // 1000))


def square(ms: int, amplitude: int) -> bytes:
    n = SR * ms // 1000
    return pcm([amplitude if i % 2 == 0 else -amplitude for i in range(n)])


class FakePortAudioError(Exception):
    pass


class FakeStream:
    def __init__(self, owner, **kwargs):
        self.owner = owner
        self.kwargs = kwargs
        self.callback = kwargs["callback"]
        self.active = False
        self.closed = False
        self.start_count = 0
        owner.streams.append(self)

    def start(self):
        self.active = True
        self.start_count += 1

    def stop(self):
        self.active = False

    def abort(self):
        self.active = False

    def close(self):
        self.active = False
        self.closed = True

    def feed(self, data: bytes, status=None) -> None:
        assert self.active and not self.closed, "fed a stream that is not running"
        channels = int(self.kwargs.get("channels", 1))
        self.callback(data, len(data) // 2 // channels, None, status)


class ExplodingBuffer:
    """Stands in for a PortAudio buffer whose copy fails inside the callback."""

    def __bytes__(self) -> bytes:
        raise RuntimeError("buffer exploded")


class FakeCallbackFlags:
    def __bool__(self) -> bool:
        return True

    def __str__(self) -> str:
        return "input_overflow"


class FakeSounddevice:
    """The parts of the sounddevice module that spells.audio touches."""

    PortAudioError = FakePortAudioError

    def __init__(self, devices, hostapis, default_input, default_hostapi=0):
        self._devices = devices
        self._hostapis = hostapis
        self.default = SimpleNamespace(device=[default_input, -1], hostapi=default_hostapi)
        self.streams: list[FakeStream] = []
        self.open_error: BaseException | None = None
        self.refuse = None
        self.opened: list[dict] = []

    def query_devices(self, device=None, kind=None):
        if device is None:
            return list(self._devices)
        return self._devices[device]

    def query_hostapis(self, index=None):
        if index is None:
            return tuple(self._hostapis)
        return self._hostapis[index]

    def api_name(self, device_index: int) -> str:
        return str(self._hostapis[self._devices[device_index]["hostapi"]]["name"])

    def RawInputStream(self, **kwargs):
        self.opened.append(dict(kwargs))
        if self.open_error is not None:
            raise self.open_error
        if self.refuse is not None:
            refusal = self.refuse(kwargs)
            if refusal is not None:
                raise refusal
        info = self._devices[kwargs["device"]]
        if self.api_name(kwargs["device"]) == audio.WASAPI:
            if float(kwargs["samplerate"]) != float(info["default_samplerate"]):
                raise FakePortAudioError(
                    "Error opening RawInputStream: Invalid sample rate", PA_INVALID_SAMPLE_RATE
                )
        elif int(kwargs["channels"]) > int(info["max_input_channels"]):
            raise FakePortAudioError(
                "Error opening RawInputStream: Invalid number of channels", PA_INVALID_CHANNEL_COUNT
            )
        return FakeStream(self, **kwargs)


def dev(index, hostapi, name, inputs, outputs=0, rate=MME_RATE):
    return {
        "index": index,
        "hostapi": hostapi,
        "name": name,
        "max_input_channels": inputs,
        "max_output_channels": outputs,
        "default_samplerate": float(rate),
    }


def hostapi(name, devices, default_input, default_output=-1):
    return {
        "name": name,
        "devices": devices,
        "default_input_device": default_input,
        "default_output_device": default_output,
    }


def machine():
    """The owner's device table: three microphones under MME and under WASAPI."""
    devices = [
        dev(0, 0, "Microsoft Sound Mapper - Input", 2),
        dev(1, 0, OBSBOT_MME, 2),
        dev(2, 0, USB, 1),
        dev(3, 0, HYPERX_MME, 1),
        dev(4, 0, "Speakers (Realtek(R) Audio)", 0, 2),
        dev(5, 1, OBSBOT_WASAPI, 2, rate=SR),
        dev(6, 1, USB, 2, rate=SR),
        dev(7, 1, HYPERX_WASAPI, 2, rate=SR),
        dev(8, 1, USB, 2, rate=SR),
    ]
    hostapis = [
        hostapi("MME", [0, 1, 2, 3, 4], default_input=1, default_output=4),
        hostapi("Windows WASAPI", [5, 6, 7, 8], default_input=5),
    ]
    return devices, hostapis


@pytest.fixture
def fake_sd(monkeypatch):
    devices, hostapis = machine()
    fake = FakeSounddevice(devices, hostapis, default_input=1)
    monkeypatch.setattr(audio, "sd", fake)
    monkeypatch.setattr(audio, "_microphone_privacy_denied", lambda: None)
    return fake


@pytest.fixture
def mme_only(monkeypatch):
    """A machine PortAudio reports no WASAPI input devices for."""
    devices, hostapis = machine()
    devices = devices[:5]
    hostapis = [hostapis[0], hostapi("Windows WASAPI", [], default_input=-1)]
    fake = FakeSounddevice(devices, hostapis, default_input=1)
    monkeypatch.setattr(audio, "sd", fake)
    monkeypatch.setattr(audio, "_microphone_privacy_denied", lambda: None)
    return fake


def no_input_devices(fake: FakeSounddevice) -> None:
    fake._devices = [dev(0, 0, "Speakers", 0, 2)]
    fake._hostapis = [hostapi("MME", [0], default_input=-1, default_output=0)]
    fake.default.device = [-1, 0]


# list_input_devices


def test_list_input_devices_prefers_wasapi_with_full_names(fake_sd):
    assert list_input_devices() == [
        AudioDevice(
            index=5, name=OBSBOT_WASAPI, is_default=True, host_api="Windows WASAPI",
            sample_rate=SR, channels=2,
        ),
        AudioDevice(
            index=6, name=USB, is_default=False, host_api="Windows WASAPI",
            sample_rate=SR, channels=2,
        ),
        AudioDevice(
            index=7, name=HYPERX_WASAPI, is_default=False, host_api="Windows WASAPI",
            sample_rate=SR, channels=2,
        ),
    ]


def test_list_input_devices_lists_each_device_once(fake_sd):
    names = [device.name for device in list_input_devices()]
    assert len(names) == len(set(names))
    assert USB in names


def test_list_input_devices_falls_back_to_the_default_host_api(mme_only):
    devices = list_input_devices()
    assert [device.name for device in devices] == [
        "Microsoft Sound Mapper - Input",
        OBSBOT_MME,
        USB,
        HYPERX_MME,
    ]
    assert all(device.host_api == "MME" for device in devices)
    assert [device.is_default for device in devices] == [False, True, False, False]
    assert devices[0].sample_rate == MME_RATE


def test_list_input_devices_empty_when_no_inputs(fake_sd):
    no_input_devices(fake_sd)
    assert list_input_devices() == []


def test_capture_host_api_names_the_path(fake_sd, mme_only):
    assert capture_host_api() == "MME"


def test_capture_host_api_is_wasapi_when_there_is_one(fake_sd):
    assert capture_host_api() == "Windows WASAPI"


# Name matching and migration


@pytest.mark.parametrize(
    "stored,name,expected",
    [
        (HYPERX_MME, HYPERX_WASAPI, True),
        (HYPERX_WASAPI, HYPERX_MME, True),
        (USB, USB, True),
        (OBSBOT_MME, HYPERX_WASAPI, False),
        ("", USB, False),
        (USB, "", False),
    ],
)
def test_names_match_covers_the_truncation(stored, name, expected):
    assert names_match(stored, name) is expected


def test_match_device_prefers_an_exact_name(fake_sd):
    devices = list_input_devices()
    device, how = match_device(devices, USB)
    assert how == "exact"
    assert device.index == 6


def test_match_device_falls_back_to_a_prefix(fake_sd):
    devices = list_input_devices()
    device, how = match_device(devices, HYPERX_MME)
    assert how == "prefix"
    assert device.name == HYPERX_WASAPI


def test_match_device_falls_back_to_the_system_default(fake_sd):
    devices = list_input_devices()
    device, how = match_device(devices, "Blue Yeti")
    assert how == "default"
    assert device.name == OBSBOT_WASAPI


# capture_plans


def test_capture_plans_lead_with_wasapi_and_keep_mme_behind(fake_sd):
    plans, chosen, how = capture_plans(HYPERX_MME, None)
    assert how == "prefix"
    assert chosen.name == HYPERX_WASAPI
    assert plans == [
        CapturePlan(index=7, name=HYPERX_WASAPI, host_api="Windows WASAPI", sample_rate=SR, channels=2),
        CapturePlan(index=3, name=HYPERX_MME, host_api="MME", sample_rate=MME_RATE, channels=1),
    ]


def test_capture_plans_honour_an_explicit_rate(fake_sd):
    plans, _chosen, _how = capture_plans(HYPERX_WASAPI, 16000)
    assert [plan.sample_rate for plan in plans] == [16000, 16000]


def test_capture_plans_raise_when_there_is_no_input(fake_sd):
    no_input_devices(fake_sd)
    with pytest.raises(MicError) as info:
        capture_plans(None, None)
    assert info.value.kind == "missing"


def test_capture_plans_accept_an_index_from_any_host_api(fake_sd):
    plans, chosen, how = capture_plans(3, None)
    assert how == "index"
    assert chosen.name == HYPERX_MME
    assert plans[0].sample_rate == MME_RATE


@pytest.mark.parametrize("index", [4, 99, -1])
def test_capture_plans_reject_an_index_that_is_not_an_input(fake_sd, index):
    with pytest.raises(MicError) as info:
        capture_plans(index, None)
    assert info.value.kind == "missing"


def test_channel_options_try_mono_first():
    mono = CapturePlan(index=0, name="m", host_api="MME", sample_rate=SR, channels=1)
    stereo = CapturePlan(index=0, name="s", host_api="Windows WASAPI", sample_rate=SR, channels=2)
    assert channel_options(mono) == (1,)
    assert channel_options(stereo) == (1, 2)


# downmix


def test_downmix_averages_the_channels():
    interleaved = pcm([100, 200, -400, 0, 32767, 32767])
    assert downmix_to_mono(interleaved, 2) == pcm([150, -200, 32767])


def test_downmix_is_a_no_op_for_mono():
    data = pcm([1, 2, 3])
    assert downmix_to_mono(data, 1) is data


def test_downmix_drops_a_partial_frame():
    assert downmix_to_mono(pcm([10, 20, 30]), 2) == pcm([15])
    assert downmix_to_mono(pcm([10]), 2) == b""


# start / stop / cancel


def test_start_stop_returns_fed_bytes_and_closes_stream(fake_sd):
    rec = Recorder()
    assert not rec.is_recording
    assert rec.open_latency_ms is None
    rec.start()
    assert rec.is_recording
    stream = fake_sd.streams[-1]
    assert stream.kwargs["samplerate"] == SR
    assert stream.kwargs["channels"] == 1
    assert stream.kwargs["dtype"] == "int16"
    assert stream.kwargs["device"] == 5
    assert stream.active
    first, second = square(50, 1000), square(50, 2000)
    stream.feed(first)
    stream.feed(second)
    assert rec.stop() == first + second
    assert not rec.is_recording
    assert stream.closed


def test_the_recorder_reports_the_capture_it_settled_on(fake_sd):
    rec = Recorder()
    rec.start()
    assert rec.sample_rate == SR
    assert rec.host_api == "Windows WASAPI"
    assert rec.device_name == OBSBOT_WASAPI
    assert rec.channels == 1
    assert not rec.downmixed
    info = rec.capture_info()
    assert info.host_api == "Windows WASAPI"
    assert info.sample_rate == SR
    assert info.device == OBSBOT_WASAPI
    assert info.match == "default"
    assert info.dropped_blocks == 0
    rec.cancel()


def test_a_machine_without_wasapi_records_through_mme(mme_only):
    rec = Recorder()
    rec.start()
    assert rec.host_api == "MME"
    assert rec.sample_rate == MME_RATE
    assert mme_only.streams[-1].kwargs["device"] == 1
    rec.cancel()


def test_an_explicit_rate_wasapi_refuses_falls_back_to_mme(fake_sd, caplog):
    with caplog.at_level(logging.INFO, logger="spells.audio"):
        rec = Recorder(device=HYPERX_WASAPI, sample_rate=16000)
        rec.start()
    assert rec.host_api == "MME"
    assert rec.sample_rate == 16000
    assert rec.device_name == HYPERX_MME
    assert [call["device"] for call in fake_sd.opened] == [7, 3]
    assert "MME" in caplog.text
    rec.cancel()


def test_a_wasapi_device_that_will_not_open_falls_back_to_mme(fake_sd):
    def refuse(kwargs):
        if kwargs["device"] == 7:
            return FakePortAudioError(
                "Error opening RawInputStream: Device unavailable", PA_DEVICE_UNAVAILABLE
            )
        return None

    fake_sd.refuse = refuse
    rec = Recorder(device=HYPERX_WASAPI)
    rec.start()
    assert rec.host_api == "MME"
    assert rec.device_name == HYPERX_MME
    assert rec.sample_rate == MME_RATE
    rec.cancel()


def test_a_device_that_refuses_mono_is_downmixed(fake_sd):
    def refuse(kwargs):
        if kwargs["channels"] == 1:
            return FakePortAudioError(
                "Error opening RawInputStream: Invalid number of channels",
                PA_INVALID_CHANNEL_COUNT,
            )
        return None

    fake_sd.refuse = refuse
    rec = Recorder()
    rec.start()
    stream = fake_sd.streams[-1]
    assert stream.kwargs["channels"] == 2
    assert rec.channels == 2
    assert rec.downmixed
    stream.feed(pcm([100, 200, -400, 0]))
    assert rec.duration_s == pytest.approx(2 / SR)
    assert rec.stop() == pcm([150, -200])
    rec.close()


def test_a_stored_name_from_mme_records_and_reports_the_prefix_match(fake_sd):
    rec = Recorder(device=HYPERX_MME)
    rec.start()
    assert fake_sd.streams[-1].kwargs["device"] == 7
    assert rec.device_match == "prefix"
    assert rec.device_name == HYPERX_WASAPI
    assert rec.fallback_notice is None
    rec.cancel()


def test_an_unknown_name_records_from_the_default_with_a_notice(fake_sd, caplog):
    with caplog.at_level(logging.WARNING, logger="spells.audio"):
        rec = Recorder(device="Blue Yeti")
        rec.start()
    assert rec.device_match == "default"
    assert rec.device_name == OBSBOT_WASAPI
    assert "Blue Yeti" in (rec.fallback_notice or "")
    assert OBSBOT_WASAPI in (rec.fallback_notice or "")
    assert "Blue Yeti" in caplog.text
    assert rec.capture_info().match == "default"
    rec.cancel()


def test_open_latency_is_recorded(fake_sd):
    rec = Recorder()
    rec.start()
    assert rec.open_latency_ms is not None
    assert rec.open_latency_ms > 0
    rec.cancel()


def test_duration_is_live_and_based_on_captured_frames(fake_sd):
    rec = Recorder()
    rec.start()
    stream = fake_sd.streams[-1]
    assert rec.duration_s == 0.0
    stream.feed(square(1000, 100))
    assert rec.duration_s == pytest.approx(1.0)
    stream.feed(square(500, 100))
    assert rec.duration_s == pytest.approx(1.5)
    rec.stop()


def test_a_named_device_opens_its_own_entry(fake_sd):
    rec = Recorder(device=USB)
    rec.start()
    assert fake_sd.streams[-1].kwargs["device"] == 6
    assert fake_sd.streams[-1].kwargs["samplerate"] == SR
    rec.cancel()


def test_on_level_values_are_normalized_rms(fake_sd):
    levels: list[float] = []
    rec = Recorder(on_level=levels.append)
    rec.start()
    stream = fake_sd.streams[-1]
    stream.feed(silence(50))
    stream.feed(square(50, 16384))
    stream.feed(square(50, 32767))
    rec.stop()
    assert len(levels) == 3
    assert all(0.0 <= v <= 1.0 for v in levels)
    assert levels[0] == 0.0
    assert levels[1] == pytest.approx(0.5, abs=0.01)
    assert levels[2] == pytest.approx(1.0, abs=0.01)


def test_on_level_is_called_about_twenty_times_per_second(fake_sd):
    rec = Recorder()
    rec.start()
    assert fake_sd.streams[-1].kwargs["blocksize"] == SR // 20
    rec.cancel()


def test_the_block_size_follows_the_rate_that_was_opened(fake_sd):
    rec = Recorder(device=HYPERX_WASAPI, sample_rate=16000)
    rec.start()
    assert fake_sd.streams[-1].kwargs["blocksize"] == 16000 // 20
    rec.cancel()


def test_on_level_exception_is_recorded_and_capture_continues(fake_sd):
    calls: list[float] = []

    def bad_callback(level: float) -> None:
        calls.append(level)
        raise RuntimeError("meter crashed")

    rec = Recorder(on_level=bad_callback)
    rec.start()
    assert rec.error is None
    stream = fake_sd.streams[-1]
    stream.feed(square(50, 500))
    stream.feed(square(50, 500))
    assert len(calls) == 1
    assert isinstance(rec.error, RuntimeError)
    assert rec.is_recording
    assert rec.stop() == square(50, 500) * 2


def test_callback_exception_is_reported_not_swallowed(fake_sd, caplog):
    levels: list[float] = []
    rec = Recorder(on_level=levels.append)
    rec.start()
    stream = fake_sd.streams[-1]
    assert rec.error is None
    with caplog.at_level(logging.WARNING, logger="spells.audio"):
        stream.callback(ExplodingBuffer(), SR // 20, None, None)
    assert isinstance(rec.error, RuntimeError)
    assert str(rec.error) == "buffer exploded"
    assert "buffer exploded" in caplog.text
    assert rec.is_recording
    assert stream.active
    stream.feed(square(50, 300))
    assert levels == []
    assert rec.stop() == square(50, 300)


def test_start_resets_error(fake_sd):
    rec = Recorder()
    rec.start()
    fake_sd.streams[-1].callback(ExplodingBuffer(), SR // 20, None, None)
    assert rec.error is not None
    rec.cancel()
    rec.start()
    assert rec.error is None
    rec.cancel()


def test_status_flags_are_logged_at_debug(fake_sd, caplog):
    rec = Recorder()
    rec.start()
    stream = fake_sd.streams[-1]
    with caplog.at_level(logging.DEBUG, logger="spells.audio"):
        stream.feed(square(50, 10), status=FakeCallbackFlags())
        stream.feed(square(50, 10), status=None)
    debug_records = [
        r for r in caplog.records if r.levelno == logging.DEBUG and "input stream status" in r.getMessage()
    ]
    assert len(debug_records) == 1
    assert "input_overflow" in debug_records[0].getMessage()
    assert rec.error is None
    assert rec.stop() == square(50, 10) * 2


def test_cancel_discards_and_next_recording_starts_clean(fake_sd):
    rec = Recorder()
    rec.start()
    stream = fake_sd.streams[-1]
    stream.feed(square(50, 700))
    rec.cancel()
    assert not rec.is_recording
    assert stream.closed
    rec.start()
    stream2 = fake_sd.streams[-1]
    assert stream2 is not stream
    stream2.feed(square(50, 900))
    assert rec.stop() == square(50, 900)


def test_start_while_recording_and_stop_while_idle_raise(fake_sd):
    rec = Recorder()
    with pytest.raises(RuntimeError):
        rec.stop()
    rec.start()
    with pytest.raises(RuntimeError):
        rec.start()
    rec.cancel()
    rec.cancel()


# keep_open


def test_keep_open_reuses_the_stream_and_discards_idle_data(fake_sd):
    rec = Recorder(keep_open=True)
    rec.start()
    stream = fake_sd.streams[-1]
    stream.feed(square(50, 100))
    assert rec.stop() == square(50, 100)
    assert stream.active
    assert not stream.closed
    stream.feed(square(50, 200))
    rec.start()
    assert fake_sd.streams == [stream]
    assert stream.start_count == 1
    assert rec.duration_s == 0.0
    stream.feed(square(50, 300))
    assert rec.stop() == square(50, 300)
    rec.close()
    assert stream.closed
    assert not rec.is_recording


def test_keep_open_reports_levels_only_while_recording(fake_sd):
    levels: list[float] = []
    rec = Recorder(keep_open=True, on_level=levels.append)
    rec.start()
    stream = fake_sd.streams[-1]
    stream.feed(square(50, 100))
    rec.stop()
    stream.feed(square(50, 100))
    stream.feed(square(50, 100))
    assert len(levels) == 1
    rec.close()


def test_keep_open_reopens_a_stream_that_stopped_running(fake_sd):
    rec = Recorder(keep_open=True)
    rec.start()
    stream = fake_sd.streams[-1]
    rec.stop()
    stream.active = False
    rec.start()
    assert len(fake_sd.streams) == 2
    assert fake_sd.streams[-1].active
    rec.close()


# MicError mapping


def test_mic_error_kind_and_message():
    err = MicError("busy", "Mic in use")
    assert err.kind == "busy"
    assert str(err) == "Mic in use"
    assert isinstance(err, Exception)


def test_missing_when_no_input_devices(fake_sd):
    no_input_devices(fake_sd)
    rec = Recorder()
    with pytest.raises(MicError) as info:
        rec.start()
    assert info.value.kind == "missing"
    assert not rec.is_recording


def test_a_name_that_matches_nothing_no_longer_raises(fake_sd):
    rec = Recorder(device="Blue Yeti")
    rec.start()
    assert rec.is_recording
    rec.cancel()


def test_default_device_falls_back_to_first_input_when_none_is_flagged(fake_sd):
    fake_sd._hostapis[1]["default_input_device"] = -1
    rec = Recorder()
    rec.start()
    assert fake_sd.streams[-1].kwargs["device"] == 5
    rec.cancel()


HOST_ERROR = "Error opening RawInputStream: Unanticipated host error"
IN_USE_TEXT = "The specified device is already in use. Wait until it is free, and then try again."


@pytest.mark.parametrize(
    "error,kind",
    [
        (FakePortAudioError("Error opening RawInputStream: Invalid device", PA_INVALID_DEVICE), "missing"),
        (
            FakePortAudioError(
                HOST_ERROR,
                PA_UNANTICIPATED_HOST_ERROR,
                (0, 2, "The specified device identifier is out of range."),
            ),
            "missing",
        ),
        (ValueError("No input device matching 'x'"), "missing"),
        (
            FakePortAudioError("Error opening RawInputStream: Device unavailable", PA_DEVICE_UNAVAILABLE),
            "busy",
        ),
        (FakePortAudioError(HOST_ERROR, PA_UNANTICIPATED_HOST_ERROR, (0, 4, IN_USE_TEXT)), "busy"),
        (
            FakePortAudioError(
                HOST_ERROR, PA_UNANTICIPATED_HOST_ERROR, (2, -2004287478, "AUDCLNT_E_DEVICE_IN_USE")
            ),
            "busy",
        ),
        (
            FakePortAudioError("Error opening RawInputStream: Invalid sample rate", PA_INVALID_SAMPLE_RATE),
            "unknown",
        ),
        (RuntimeError("boom"), "unknown"),
    ],
)
def test_open_errors_map_to_kinds(fake_sd, error, kind):
    fake_sd.open_error = error
    rec = Recorder()
    with pytest.raises(MicError) as info:
        rec.start()
    assert info.value.kind == kind
    assert str(info.value)
    assert info.value.__cause__ is error
    assert not rec.is_recording


def test_every_plan_is_tried_before_the_error_is_raised(fake_sd):
    fake_sd.open_error = RuntimeError("boom")
    with pytest.raises(MicError):
        Recorder(device=HYPERX_WASAPI).start()
    assert [call["device"] for call in fake_sd.opened] == [7, 7, 3]


def test_blocked_when_only_zeros_for_500ms(fake_sd):
    rec = Recorder()
    rec.start()
    stream = fake_sd.streams[-1]
    assert not rec.silent
    for _ in range(10):
        stream.feed(silence(50))
    assert rec.silent
    with pytest.raises(MicError) as info:
        rec.stop()
    assert info.value.kind == "blocked"
    assert "privacy" in str(info.value).lower()
    assert not rec.is_recording
    assert stream.closed


def test_not_blocked_when_signal_arrives_after_initial_silence(fake_sd):
    rec = Recorder()
    rec.start()
    stream = fake_sd.streams[-1]
    for _ in range(10):
        stream.feed(silence(50))
    stream.feed(square(50, 5))
    assert not rec.silent
    assert len(rec.stop()) == 11 * SR // 20 * 2


def test_short_silence_is_returned_not_blocked(fake_sd):
    rec = Recorder()
    rec.start()
    stream = fake_sd.streams[-1]
    for _ in range(8):
        stream.feed(silence(50))
    assert not rec.silent
    assert rec.stop() == silence(400)


def test_blocked_message_names_the_privacy_setting_when_registry_denies(fake_sd, monkeypatch):
    monkeypatch.setattr(audio, "_microphone_privacy_denied", lambda: True)
    rec = Recorder()
    rec.start()
    stream = fake_sd.streams[-1]
    for _ in range(10):
        stream.feed(silence(50))
    with pytest.raises(MicError) as info:
        rec.stop()
    assert info.value.kind == "blocked"
    assert "desktop apps" in str(info.value)


def test_cancel_never_raises_blocked(fake_sd):
    rec = Recorder()
    rec.start()
    stream = fake_sd.streams[-1]
    for _ in range(10):
        stream.feed(silence(50))
    rec.cancel()
    assert not rec.is_recording


def test_microphone_privacy_denied_reads_registry_without_error():
    assert audio._microphone_privacy_denied() in (True, False, None)


# pcm16_to_wav


@pytest.mark.parametrize("rate", [16000, 44100, 48000])
def test_pcm16_to_wav_round_trip(rate):
    data = square(100, 1234)
    frames = len(data) // 2
    wav = pcm16_to_wav(data, rate)
    assert wav.startswith(b"RIFF")
    with wave.open(io.BytesIO(wav)) as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == rate
        assert w.getnframes() == frames
        assert w.readframes(frames) == data


class Overflowing(FakeCallbackFlags):
    input_overflow = True


def test_dropped_blocks_count_input_overflows_during_a_recording(fake_sd):
    rec = Recorder()
    rec.start()
    stream = fake_sd.streams[-1]
    assert rec.dropped_blocks == 0
    stream.feed(square(50, 10), status=Overflowing())
    stream.feed(square(50, 10), status=None)
    stream.feed(square(50, 10), status=Overflowing())
    assert rec.dropped_blocks == 2
    assert rec.capture_info().dropped_blocks == 2
    rec.stop()
    rec.start()
    assert rec.dropped_blocks == 0
    rec.cancel()
