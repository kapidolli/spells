"""Unit tests for spells.hotkey: the chord state machine and the hotkey thread.

Spec 6 steps 1 to 3, 13 (hook upkeep), 20.1 hotkey line, 20.2 stress test;
batch 2 decisions V1-6, V1-8, V2-8, V3-F3. The Win32 hook is replaced by
tests/unit/fake_hook.py; the clock is a FakeClock.
"""

from __future__ import annotations

import inspect
import sys
import threading
import time
from dataclasses import dataclass, field

import pytest

from spells.hotkey import (
    MASK_TAG,
    MASK_VK,
    MODIFIER_VKS,
    PROBE_TAG,
    TICK_MS,
    TIMER_PROBE,
    TIMER_PROBE_CHECK,
    TIMER_TICK,
    VK_ESCAPE,
    VK_PROBE,
    ChordState,
    ChordStateMachine,
    Decision,
    HotkeyCallbacks,
    HotkeyThread,
)
from spells.models import Chord
from unit.fake_hook import LLKHF_INJECTED, FakeClock, FakeHook, KeyEvent

VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12
VK_LWIN = 0x5B
VK_RWIN = 0x5C
VK_LSHIFT = 0xA0
VK_LCONTROL = 0xA2
VK_RCONTROL = 0xA3
VK_LMENU = 0xA4
VK_RMENU = 0xA5
VK_A = 0x41
VK_D = 0x44
VK_E = 0x45
VK_H = 0x48

MAIN = Chord((VK_LCONTROL, VK_LMENU, VK_D))
LANG_SQ = Chord((VK_LCONTROL, VK_LMENU, VK_E), language="sq")
WIN_CHORD = Chord((VK_LWIN, VK_H))

MASK_PAIR = ((MASK_VK, True, MASK_TAG), (MASK_VK, False, MASK_TAG))


@dataclass
class Recorder:
    """Records every hotkey callback in order, with its dictation id."""

    events: list[tuple] = field(default_factory=list)

    def callbacks(self) -> HotkeyCallbacks:
        return HotkeyCallbacks(
            pressed=lambda chord, did: self.events.append(("pressed", chord, did)),
            released=lambda chord, did: self.events.append(("released", chord, did)),
            latched=lambda chord, did: self.events.append(("latched", chord, did)),
            discarded=lambda did: self.events.append(("discarded", did)),
            cancelled=lambda did: self.events.append(("cancelled", did)),
        )


def raising_callbacks() -> HotkeyCallbacks:
    def boom(*_args):
        raise RuntimeError("callback failed")

    return HotkeyCallbacks(
        pressed=boom, released=boom, latched=boom, discarded=boom, cancelled=boom
    )


def ev(vk: int, keydown: bool, *, extra_info: int = 0, injected: bool = False) -> KeyEvent:
    return KeyEvent(vk, 0, LLKHF_INJECTED if injected else 0, extra_info, keydown, 0)


def press_chord(sm: ChordStateMachine, chord: Chord) -> list[Decision]:
    return [sm.on_key(ev(vk, True)) for vk in chord.keys]


def release_chord(sm: ChordStateMachine, chord: Chord) -> list[Decision]:
    return [sm.on_key(ev(vk, False)) for vk in reversed(chord.keys)]


def tap(sm: ChordStateMachine, clock: FakeClock, chord: Chord, hold_s: float = 0.1) -> None:
    press_chord(sm, chord)
    clock.advance(hold_s)
    release_chord(sm, chord)


def latch(sm: ChordStateMachine, clock: FakeClock, chord: Chord) -> None:
    tap(sm, clock, chord)
    clock.advance(0.2)
    press_chord(sm, chord)
    assert sm.state is ChordState.LATCHED
    release_chord(sm, chord)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def rec() -> Recorder:
    return Recorder()


@pytest.fixture
def sm(clock: FakeClock, rec: Recorder) -> ChordStateMachine:
    return ChordStateMachine([MAIN, LANG_SQ], rec.callbacks(), clock.now)


# Constants and types -------------------------------------------------------


def test_constants_match_the_contract():
    assert PROBE_TAG == 0x55545452
    assert MASK_TAG == 0x5554544D
    assert MASK_VK == 0xE8
    assert VK_PROBE == 0xE8
    assert VK_ESCAPE == 0x1B
    assert MODIFIER_VKS == frozenset(
        {VK_SHIFT, VK_CONTROL, VK_MENU, VK_LWIN, VK_RWIN, 0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5}
    )
    assert Decision(swallow=True) == Decision(swallow=True, inject=())
    assert [s.name for s in ChordState] == ["IDLE", "HELD", "TAP_WINDOW", "LATCHED"]


def test_default_clock_is_perf_counter():
    """time.monotonic is GetTickCount64 (15.6 ms) on this Python; QPC is needed."""
    assert inspect.signature(ChordStateMachine).parameters["now"].default is time.perf_counter
    assert inspect.signature(HotkeyThread).parameters["now"].default is time.perf_counter


def test_state_machine_works_with_the_real_clock(rec):
    sm = ChordStateMachine([MAIN], rec.callbacks())
    press_chord(sm, MAIN)
    time.sleep(0.3)
    release_chord(sm, MAIN)
    assert rec.events == [("pressed", MAIN, 1), ("released", MAIN, 1)]


def test_initial_state(sm: ChordStateMachine):
    assert sm.state is ChordState.IDLE
    assert sm.active_chord is None
    assert sm.dictation_id is None
    assert sm.recording is False
    assert sm.probe_seen is False
    assert sm.chords == (MAIN, LANG_SQ)


# Hold, tap, latch ------------------------------------------------------------


def test_hold_and_release_after_250ms_fires_pressed_then_released(sm, rec, clock):
    press_chord(sm, MAIN)
    assert rec.events == [("pressed", MAIN, 1)]
    assert sm.state is ChordState.HELD
    assert sm.active_chord == MAIN
    assert sm.dictation_id == 1
    assert sm.recording is True
    clock.advance(0.25)
    sm.on_key(ev(VK_D, False))
    assert rec.events == [("pressed", MAIN, 1), ("released", MAIN, 1)]
    assert sm.state is ChordState.IDLE
    assert sm.active_chord is None
    assert sm.dictation_id is None
    assert sm.recording is False
    # the remaining chord keys going up do nothing more
    sm.on_key(ev(VK_LMENU, False))
    sm.on_key(ev(VK_LCONTROL, False))
    assert rec.events == [("pressed", MAIN, 1), ("released", MAIN, 1)]


def test_pressed_records_the_press_time_and_release_is_measured_from_it(sm, rec, clock):
    press_chord(sm, MAIN)
    clock.advance(0.249)
    sm.on_key(ev(VK_D, False))
    assert sm.state is ChordState.TAP_WINDOW, "a release just under tap_max_s is a tap"
    assert rec.events == [("pressed", MAIN, 1)]


def test_tap_then_window_expiry_fires_pressed_then_discarded(sm, rec, clock):
    tap(sm, clock, MAIN, hold_s=0.1)
    assert rec.events == [("pressed", MAIN, 1)]
    assert sm.state is ChordState.TAP_WINDOW
    assert sm.active_chord == MAIN
    assert sm.recording is True, "recording continues silently through the tap window"
    clock.advance(0.399)
    sm.on_tick()
    assert sm.state is ChordState.TAP_WINDOW
    clock.advance(0.001)
    sm.on_tick()
    assert rec.events == [("pressed", MAIN, 1), ("discarded", 1)]
    assert sm.state is ChordState.IDLE
    assert sm.active_chord is None
    assert sm.recording is False


def test_tap_window_is_measured_from_release_not_press(sm, rec, clock):
    tap(sm, clock, MAIN, hold_s=0.2)
    clock.advance(0.3)  # 0.5 s after the press, 0.3 s after the release
    sm.on_tick()
    assert sm.state is ChordState.TAP_WINDOW
    press_chord(sm, MAIN)
    assert rec.events == [("pressed", MAIN, 1), ("latched", MAIN, 1)]


def test_double_tap_latches_and_a_later_press_ends_it_consumed(sm, rec, clock):
    tap(sm, clock, MAIN)
    clock.advance(0.2)
    press_chord(sm, MAIN)
    assert rec.events == [("pressed", MAIN, 1), ("latched", MAIN, 1)]
    assert sm.state is ChordState.LATCHED
    assert sm.active_chord == MAIN
    assert sm.dictation_id == 1
    # releasing the latching press changes nothing, however long it was held
    clock.advance(2.0)
    release_chord(sm, MAIN)
    assert sm.state is ChordState.LATCHED
    clock.advance(30.0)
    sm.on_tick()
    assert sm.state is ChordState.LATCHED
    # any later press of the chord ends the latch
    press_chord(sm, MAIN)
    assert rec.events == [("pressed", MAIN, 1), ("latched", MAIN, 1), ("released", MAIN, 1)]
    assert sm.state is ChordState.IDLE
    assert sm.active_chord is None
    # that press is consumed: its own release does nothing, whatever its hold length
    clock.advance(1.0)
    release_chord(sm, MAIN)
    assert rec.events == [("pressed", MAIN, 1), ("latched", MAIN, 1), ("released", MAIN, 1)]
    assert sm.state is ChordState.IDLE
    # and the press after that is a fresh dictation
    press_chord(sm, MAIN)
    assert rec.events[-1] == ("pressed", MAIN, 2)
    assert sm.state is ChordState.HELD


def test_latch_ended_by_a_short_tap_is_still_consumed(sm, rec, clock):
    latch(sm, clock, MAIN)
    clock.advance(1.0)
    tap(sm, clock, MAIN, hold_s=0.05)
    assert rec.events == [("pressed", MAIN, 1), ("latched", MAIN, 1), ("released", MAIN, 1)]
    assert sm.state is ChordState.IDLE
    clock.advance(0.1)
    press_chord(sm, MAIN)
    assert rec.events[-1] == ("pressed", MAIN, 2), "no tap window opens for a consumed press"
    assert sm.state is ChordState.HELD


def test_per_language_chord_press_ending_a_latch_is_consumed(sm, rec, clock):
    latch(sm, clock, MAIN)
    clock.advance(1.0)
    press_chord(sm, LANG_SQ)
    assert rec.events == [("pressed", MAIN, 1), ("latched", MAIN, 1), ("released", MAIN, 1)]
    assert sm.state is ChordState.IDLE
    clock.advance(0.5)
    release_chord(sm, LANG_SQ)
    assert rec.events[-1] == ("released", MAIN, 1)
    assert sm.state is ChordState.IDLE
    # a fresh per-language press afterwards starts a forced-language dictation
    press_chord(sm, LANG_SQ)
    assert rec.events[-1] == ("pressed", LANG_SQ, 2)
    assert sm.active_chord == LANG_SQ
    assert sm.active_chord.language == "sq"


def test_per_language_chord_uses_the_same_rules(sm, rec, clock):
    tap(sm, clock, LANG_SQ)
    clock.advance(0.1)
    press_chord(sm, LANG_SQ)
    release_chord(sm, LANG_SQ)
    assert rec.events == [("pressed", LANG_SQ, 1), ("latched", LANG_SQ, 1)]
    press_chord(sm, MAIN)
    assert rec.events[-1] == ("released", LANG_SQ, 1)
    release_chord(sm, MAIN)
    assert len(rec.events) == 3


def test_same_chord_press_after_the_window_is_a_fresh_press(sm, rec, clock):
    tap(sm, clock, MAIN)
    clock.advance(0.5)
    press_chord(sm, MAIN)  # no tick ran; the key event evaluates the expiry first
    assert rec.events == [("pressed", MAIN, 1), ("discarded", 1), ("pressed", MAIN, 2)]
    assert sm.state is ChordState.HELD


def test_different_chord_in_the_tap_window_discards_and_starts_the_new_chord(sm, rec, clock):
    tap(sm, clock, MAIN)
    clock.advance(0.1)
    press_chord(sm, LANG_SQ)
    assert rec.events == [("pressed", MAIN, 1), ("discarded", 1), ("pressed", LANG_SQ, 2)]
    assert sm.state is ChordState.HELD
    assert sm.active_chord == LANG_SQ


def test_auto_repeat_keydown_neither_refires_nor_ends_a_latch(sm, rec, clock):
    press_chord(sm, MAIN)
    for _ in range(5):
        assert sm.on_key(ev(VK_D, True)).swallow is True
    assert rec.events == [("pressed", MAIN, 1)]
    clock.advance(0.1)
    release_chord(sm, MAIN)
    clock.advance(0.1)
    press_chord(sm, MAIN)
    assert sm.state is ChordState.LATCHED
    for _ in range(5):
        sm.on_key(ev(VK_D, True))
    assert sm.state is ChordState.LATCHED
    assert rec.events == [("pressed", MAIN, 1), ("latched", MAIN, 1)]


def test_another_chord_while_one_is_held_is_ignored(sm, rec, clock):
    press_chord(sm, MAIN)
    decision = sm.on_key(ev(VK_E, True))
    assert rec.events == [("pressed", MAIN, 1)]
    assert sm.active_chord == MAIN
    assert decision.swallow is True, "a completed chord key never reaches the target app"
    sm.on_key(ev(VK_E, False))
    clock.advance(0.3)
    sm.on_key(ev(VK_D, False))
    assert rec.events == [("pressed", MAIN, 1), ("released", MAIN, 1)]


def test_dictation_ids_increase_with_every_press(sm, rec, clock):
    press_chord(sm, MAIN)
    clock.advance(0.3)
    release_chord(sm, MAIN)
    tap(sm, clock, MAIN)
    clock.advance(0.5)
    sm.on_tick()
    latch(sm, clock, MAIN)
    clock.advance(1.0)
    press_chord(sm, MAIN)
    release_chord(sm, MAIN)
    press_chord(sm, LANG_SQ)
    sm.on_key(ev(VK_ESCAPE, True))
    sm.on_key(ev(VK_ESCAPE, False))
    release_chord(sm, LANG_SQ)
    assert rec.events == [
        ("pressed", MAIN, 1),
        ("released", MAIN, 1),
        ("pressed", MAIN, 2),
        ("discarded", 2),
        ("pressed", MAIN, 3),
        ("latched", MAIN, 3),
        ("released", MAIN, 3),
        ("pressed", LANG_SQ, 4),
        ("cancelled", 4),
    ]
    assert sm.dictation_id is None


# Chord release rules -----------------------------------------------------------


@pytest.mark.parametrize("released_vk", [VK_LCONTROL, VK_LMENU, VK_D])
def test_releasing_any_single_chord_key_ends_the_chord(sm, rec, clock, released_vk):
    press_chord(sm, MAIN)
    clock.advance(0.3)
    sm.on_key(ev(released_vk, False))
    assert rec.events == [("pressed", MAIN, 1), ("released", MAIN, 1)]
    assert sm.state is ChordState.IDLE
    for vk in MAIN.keys:
        if vk != released_vk:
            sm.on_key(ev(vk, False))
    assert len(rec.events) == 2


def test_two_chords_with_a_shared_modifier_do_not_misfire(sm, rec, clock):
    sm.on_key(ev(VK_LCONTROL, True))
    sm.on_key(ev(VK_LMENU, True))
    assert rec.events == []
    assert sm.state is ChordState.IDLE
    sm.on_key(ev(VK_E, True))
    assert rec.events == [("pressed", LANG_SQ, 1)]
    clock.advance(0.3)
    sm.on_key(ev(VK_E, False))
    assert rec.events == [("pressed", LANG_SQ, 1), ("released", LANG_SQ, 1)]
    sm.on_key(ev(VK_D, True))
    assert rec.events[-1] == ("pressed", MAIN, 2)
    clock.advance(0.3)
    sm.on_key(ev(VK_LCONTROL, False))
    assert rec.events[-1] == ("released", MAIN, 2)
    sm.on_key(ev(VK_D, False))
    sm.on_key(ev(VK_LMENU, False))
    assert len(rec.events) == 4


def test_longest_matching_chord_wins_when_chords_nest(rec, clock):
    short = Chord((VK_LCONTROL, VK_LMENU))
    sm = ChordStateMachine([short, MAIN], rec.callbacks(), clock.now)
    sm.on_key(ev(VK_LCONTROL, True))
    sm.on_key(ev(VK_LMENU, True))
    assert rec.events == [("pressed", short, 1)]


def test_prefix_chords_observed_behavior(rec, clock):
    """Documents what happens when the main chord is a prefix of a per-language chord.

    The shorter chord fires as soon as its keys are down, so the longer chord can never
    fire: its extra key completes it while the main chord is held, which is ignored, and
    the extra key is swallowed. The config layer rejects subset chords; this is not fixed
    here.
    """
    main = Chord((VK_LCONTROL, VK_LWIN))
    lang = Chord((VK_LCONTROL, VK_LWIN, VK_D), language="sq")
    sm = ChordStateMachine([main, lang], rec.callbacks(), clock.now)
    assert sm.on_key(ev(VK_LCONTROL, True)).swallow is False
    assert sm.on_key(ev(VK_LWIN, True)).swallow is False
    assert rec.events == [("pressed", main, 1)]
    assert sm.on_key(ev(VK_D, True)).swallow is True
    assert rec.events == [("pressed", main, 1)]
    assert sm.active_chord == main
    clock.advance(0.3)
    assert sm.on_key(ev(VK_D, False)).swallow is True
    assert rec.events == [("pressed", main, 1)], "D is not a key of the active chord"
    assert sm.on_key(ev(VK_LWIN, False)) == Decision(swallow=False, inject=MASK_PAIR)
    assert rec.events == [("pressed", main, 1), ("released", main, 1)]
    sm.on_key(ev(VK_LCONTROL, False))
    assert len(rec.events) == 2


def test_generic_modifier_codes_in_a_chord_match_left_and_right_keys(rec, clock):
    generic = Chord((VK_CONTROL, VK_MENU, VK_D))
    sm = ChordStateMachine([generic], rec.callbacks(), clock.now)
    sm.on_key(ev(VK_RCONTROL, True))
    sm.on_key(ev(VK_LMENU, True))
    sm.on_key(ev(VK_D, True))
    assert rec.events == [("pressed", generic, 1)]
    clock.advance(0.3)
    sm.on_key(ev(VK_RCONTROL, False))
    assert rec.events == [("pressed", generic, 1), ("released", generic, 1)]


def test_update_chords_swaps_the_set(sm, rec, clock):
    new = Chord((VK_LCONTROL, VK_LSHIFT, VK_A))
    sm.set_chords([new])
    assert sm.chords == (new,)
    decisions = press_chord(sm, MAIN)
    assert rec.events == []
    assert [d.swallow for d in decisions] == [False, False, False]
    release_chord(sm, MAIN)
    press_chord(sm, new)
    assert rec.events == [("pressed", new, 1)]


def test_update_chords_keeps_the_active_dictation(sm, rec, clock):
    press_chord(sm, MAIN)
    sm.set_chords([LANG_SQ])
    clock.advance(0.3)
    sm.on_key(ev(VK_D, False))
    assert rec.events == [("pressed", MAIN, 1), ("released", MAIN, 1)]


# Esc ------------------------------------------------------------------------------


@pytest.mark.parametrize("setup", ["held", "tap_window", "latched"])
def test_escape_while_recording_cancels_and_is_swallowed(sm, rec, clock, setup):
    if setup == "held":
        press_chord(sm, MAIN)
    elif setup == "tap_window":
        tap(sm, clock, MAIN)
    else:
        latch(sm, clock, MAIN)
    before = list(rec.events)
    assert sm.recording is True
    assert sm.on_key(ev(VK_ESCAPE, True)) == Decision(swallow=True)
    assert rec.events == before + [("cancelled", 1)]
    assert sm.state is ChordState.IDLE
    assert sm.active_chord is None
    assert sm.recording is False
    assert sm.on_key(ev(VK_ESCAPE, False)).swallow is True, "the paired key-up is swallowed"
    # the cancel ended the recording: a second Esc passes through
    assert sm.on_key(ev(VK_ESCAPE, True)).swallow is False
    assert sm.on_key(ev(VK_ESCAPE, False)).swallow is False
    # chord keys still physically down produce nothing when they go up
    release_chord(sm, MAIN)
    assert rec.events == before + [("cancelled", 1)]


def test_escape_when_idle_passes_through(sm, rec):
    assert sm.on_key(ev(VK_ESCAPE, True)) == Decision(swallow=False)
    assert sm.on_key(ev(VK_ESCAPE, False)) == Decision(swallow=False)
    assert rec.events == []
    assert sm.state is ChordState.IDLE


def test_escape_passes_through_after_end_recording(sm, rec):
    press_chord(sm, MAIN)
    sm.end_recording(1)
    assert sm.on_key(ev(VK_ESCAPE, True)).swallow is False
    assert rec.events == [("pressed", MAIN, 1)]


# end_recording -----------------------------------------------------------------------


def test_end_recording_in_held_ends_silently_and_keeps_swallowing_held_keys(sm, rec, clock):
    press_chord(sm, MAIN)
    sm.end_recording(1)
    assert sm.state is ChordState.IDLE
    assert sm.active_chord is None
    assert sm.dictation_id is None
    assert sm.recording is False
    assert rec.events == [("pressed", MAIN, 1)], "no callback: the pipeline ended it itself"
    assert sm.on_key(ev(VK_D, True)).swallow is True, "repeat of the still-held chord key"
    clock.advance(0.3)
    assert sm.on_key(ev(VK_D, False)).swallow is True, "swallowed until its own key-up"
    assert sm.on_key(ev(VK_LMENU, False)).swallow is False
    assert rec.events == [("pressed", MAIN, 1)], "no released for an ended dictation"
    # pressing the chord again is a fresh dictation with the next id
    sm.on_key(ev(VK_LMENU, True))
    assert sm.on_key(ev(VK_D, True)).swallow is True
    assert rec.events[-1] == ("pressed", MAIN, 2)


def test_end_recording_in_tap_window_ends_silently(sm, rec, clock):
    tap(sm, clock, MAIN)
    sm.end_recording(1)
    assert sm.state is ChordState.IDLE
    clock.advance(0.1)
    press_chord(sm, MAIN)
    assert rec.events == [("pressed", MAIN, 1), ("pressed", MAIN, 2)], "a fresh press, no latch"
    clock.advance(0.5)
    sm.on_tick()
    assert sm.state is ChordState.HELD, "no discarded for the ended tap"


def test_end_recording_in_latched_ends_silently(sm, rec, clock):
    """The 10-minute cap is owned by pipeline (V1-8); after it ends a latched recording
    the next chord press must start a new dictation, not end a stale latch."""
    latch(sm, clock, MAIN)
    sm.end_recording(1)
    assert sm.state is ChordState.IDLE
    assert rec.events == [("pressed", MAIN, 1), ("latched", MAIN, 1)]
    press_chord(sm, MAIN)
    assert rec.events[-1] == ("pressed", MAIN, 2)
    assert sm.state is ChordState.HELD


def test_end_recording_with_a_stale_id_is_ignored(sm, rec, clock):
    press_chord(sm, MAIN)
    clock.advance(0.3)
    release_chord(sm, MAIN)
    press_chord(sm, MAIN)
    assert sm.dictation_id == 2
    sm.end_recording(1)
    assert sm.state is ChordState.HELD, "a late end of dictation 1 must not kill dictation 2"
    sm.end_recording(99)
    assert sm.state is ChordState.HELD
    clock.advance(0.3)
    sm.on_key(ev(VK_D, False))
    assert rec.events[-1] == ("released", MAIN, 2)
    sm.end_recording(2)  # already over: nothing to do
    assert sm.state is ChordState.IDLE
    assert rec.events == [("pressed", MAIN, 1), ("released", MAIN, 1), ("pressed", MAIN, 2),
                          ("released", MAIN, 2)]  # fmt: skip


def test_end_recording_when_idle_is_a_no_op(sm, rec):
    sm.end_recording(1)
    assert sm.state is ChordState.IDLE
    assert rec.events == []


def test_end_recording_keeps_the_win_mask_pending(rec, clock):
    sm = ChordStateMachine([WIN_CHORD], rec.callbacks(), clock.now)
    sm.on_key(ev(VK_LWIN, True))
    sm.on_key(ev(VK_H, True))
    sm.end_recording(1)
    sm.on_key(ev(VK_H, False))
    assert sm.on_key(ev(VK_LWIN, False)) == Decision(swallow=False, inject=MASK_PAIR)


# Swallowing --------------------------------------------------------------------------


def test_non_modifier_chord_keys_are_swallowed_and_modifiers_pass(sm, rec, clock):
    assert sm.on_key(ev(VK_LCONTROL, True)).swallow is False
    assert sm.on_key(ev(VK_LMENU, True)).swallow is False
    assert sm.on_key(ev(VK_D, True)).swallow is True
    assert sm.on_key(ev(VK_D, True)).swallow is True, "auto-repeat while held"
    assert sm.on_key(ev(VK_A, True)).swallow is False, "keys outside the chord pass"
    assert sm.on_key(ev(VK_A, False)).swallow is False
    clock.advance(0.3)
    assert sm.on_key(ev(VK_D, False)).swallow is True, "the paired key-up is swallowed"
    assert sm.on_key(ev(VK_LMENU, False)).swallow is False
    assert sm.on_key(ev(VK_LCONTROL, False)).swallow is False
    assert sm.on_key(ev(VK_D, True)).swallow is False, "D alone types normally"
    assert sm.on_key(ev(VK_D, False)).swallow is False
    assert rec.events == [("pressed", MAIN, 1), ("released", MAIN, 1)]


def test_swallowed_key_stays_swallowed_until_its_own_key_up(sm, rec, clock):
    press_chord(sm, MAIN)
    clock.advance(0.3)
    assert sm.on_key(ev(VK_LCONTROL, False)).swallow is False  # chord released
    assert sm.state is ChordState.IDLE
    assert sm.on_key(ev(VK_D, True)).swallow is True, "repeat of the still-held D"
    assert sm.on_key(ev(VK_D, False)).swallow is True
    assert sm.on_key(ev(VK_D, True)).swallow is False


def test_non_modifier_pressed_before_the_modifiers_has_its_repeats_swallowed(sm, rec, clock):
    assert sm.on_key(ev(VK_D, True)).swallow is False, "no chord yet: the app gets this key"
    assert sm.on_key(ev(VK_D, True)).swallow is False, "and its repeats before the chord"
    assert sm.on_key(ev(VK_LCONTROL, True)).swallow is False
    completing = sm.on_key(ev(VK_LMENU, True))
    assert completing == Decision(swallow=False, inject=((VK_D, False, MASK_TAG),)), (
        "the completing key is a modifier, and D gets a synthetic key-up (B3-11)"
    )
    assert rec.events == [("pressed", MAIN, 1)]
    assert sm.on_key(ev(VK_D, True)).swallow is True, "repeats after completion are swallowed"
    clock.advance(0.3)
    assert sm.on_key(ev(VK_D, False)).swallow is True, "and so is its physical key-up"
    assert rec.events == [("pressed", MAIN, 1), ("released", MAIN, 1)]
    # D alone types normally again once the modifiers are up and the chord cannot refire
    sm.on_key(ev(VK_LMENU, False))
    sm.on_key(ev(VK_LCONTROL, False))
    assert sm.on_key(ev(VK_D, True)).swallow is False


def test_the_synthetic_key_up_passes_back_through_the_hook_untouched(sm, rec, clock):
    """The injected key-up carries our tag, so the callback lets it reach the app and
    leaves the swallowed set alone: the physical key is still down and still swallowed."""
    sm.on_key(ev(VK_D, True))
    sm.on_key(ev(VK_LCONTROL, True))
    sm.on_key(ev(VK_LMENU, True))
    echo = ev(VK_D, False, extra_info=MASK_TAG, injected=True)
    assert sm.on_key(echo) == Decision(swallow=False), "our own key-up is never swallowed"
    assert sm.state is ChordState.HELD, "and it does not end the chord"
    assert sm.on_key(ev(VK_D, True)).swallow is True, "the physical key is still swallowed"
    clock.advance(0.3)
    assert sm.on_key(ev(VK_D, False)).swallow is True
    assert rec.events == [("pressed", MAIN, 1), ("released", MAIN, 1)]


def test_no_synthetic_key_up_when_the_non_modifier_key_completes_the_chord(sm, rec):
    """D's key-down is the one being swallowed, so the app never saw it and is owed
    no key-up. This is the ordinary case: the modifiers go down first."""
    decisions = press_chord(sm, MAIN)
    assert decisions == [Decision(swallow=False), Decision(swallow=False), Decision(swallow=True)]
    assert rec.events == [("pressed", MAIN, 1)]


def test_no_second_synthetic_key_up_for_an_already_swallowed_key(sm, rec, clock):
    """D was swallowed by the first chord, so the app never saw its key-down and a
    second completion while it is still held owes the app nothing."""
    press_chord(sm, MAIN)
    clock.advance(0.3)
    assert sm.on_key(ev(VK_LCONTROL, False)).swallow is False, "the chord ends, D stays down"
    assert sm.state is ChordState.IDLE
    assert sm.on_key(ev(VK_LCONTROL, True)) == Decision(swallow=False), "completes MAIN again"
    assert rec.events[-1] == ("pressed", MAIN, 2)


def test_each_pre_chord_key_gets_its_own_synthetic_key_up(rec, clock):
    multi = Chord((VK_LCONTROL, VK_D, VK_E))
    sm = ChordStateMachine([multi], rec.callbacks(), clock.now)
    assert sm.on_key(ev(VK_D, True)).swallow is False
    assert sm.on_key(ev(VK_E, True)).swallow is False
    completing = sm.on_key(ev(VK_LCONTROL, True))
    assert completing == Decision(
        swallow=False, inject=((VK_D, False, MASK_TAG), (VK_E, False, MASK_TAG))
    ), "one synthetic key-up each, in chord order"
    assert rec.events == [("pressed", multi, 1)]


def test_win_key_up_after_a_win_chord_injects_the_mask_pair_before_passing(rec, clock):
    sm = ChordStateMachine([WIN_CHORD], rec.callbacks(), clock.now)
    assert sm.on_key(ev(VK_LWIN, True)) == Decision(swallow=False)
    assert sm.on_key(ev(VK_H, True)) == Decision(swallow=True)
    clock.advance(0.3)
    assert sm.on_key(ev(VK_H, False)) == Decision(swallow=True)
    assert sm.on_key(ev(VK_LWIN, False)) == Decision(swallow=False, inject=MASK_PAIR)
    assert rec.events == [("pressed", WIN_CHORD, 1), ("released", WIN_CHORD, 1)]
    # a lone Win tap is not masked: the Start menu opens as usual
    sm.on_key(ev(VK_LWIN, True))
    assert sm.on_key(ev(VK_LWIN, False)) == Decision(swallow=False)


CTRL_WIN = Chord((VK_LCONTROL, VK_LWIN))


def live_release(sm: ChordStateMachine, *vks: int) -> list[Decision]:
    return [sm.on_key(ev(vk, False, extra_info=MASK_TAG, injected=True)) for vk in vks]


def test_after_live_typing_releases_the_chord_its_repeats_stay_hidden(rec, clock):
    sm = ChordStateMachine([CTRL_WIN], rec.callbacks(), clock.now)
    sm.on_key(ev(VK_LCONTROL, True))
    assert sm.on_key(ev(VK_LWIN, True)) == Decision(swallow=False)
    assert sm.on_key(ev(VK_LWIN, True)) == Decision(swallow=False), "repeat before any typing"
    assert live_release(sm, VK_LWIN, VK_CONTROL, VK_LCONTROL) == [Decision(swallow=False)] * 3
    for _ in range(5):
        assert sm.on_key(ev(VK_LWIN, True)) == Decision(swallow=True)
    assert sm.on_key(ev(VK_LCONTROL, True)) == Decision(swallow=True)
    assert sm.state is ChordState.HELD


def test_the_real_release_of_a_key_live_typing_released_is_hidden_and_unmasked(rec, clock):
    sm = ChordStateMachine([CTRL_WIN], rec.callbacks(), clock.now)
    sm.on_key(ev(VK_LCONTROL, True))
    sm.on_key(ev(VK_LWIN, True))
    clock.advance(1.0)
    live_release(sm, VK_LWIN, VK_CONTROL, VK_LCONTROL)
    assert sm.on_key(ev(VK_LWIN, False)) == Decision(swallow=True)
    assert sm.on_key(ev(VK_LCONTROL, False)) == Decision(swallow=True)
    assert rec.events == [("pressed", CTRL_WIN, 1), ("released", CTRL_WIN, 1)]
    sm.on_key(ev(VK_LWIN, True))
    assert sm.on_key(ev(VK_LWIN, False)) == Decision(swallow=False), "a later lone Win tap"


def test_without_live_typing_the_win_release_is_masked_as_before(rec, clock):
    sm = ChordStateMachine([CTRL_WIN], rec.callbacks(), clock.now)
    sm.on_key(ev(VK_LCONTROL, True))
    sm.on_key(ev(VK_LWIN, True))
    sm.on_key(ev(VK_LWIN, True))
    clock.advance(1.0)
    assert sm.on_key(ev(VK_LWIN, False)) == Decision(swallow=False, inject=MASK_PAIR)
    assert sm.on_key(ev(VK_LCONTROL, False)) == Decision(swallow=False)


def test_a_synthetic_release_of_a_key_not_held_changes_nothing(rec, clock):
    sm = ChordStateMachine([CTRL_WIN], rec.callbacks(), clock.now)
    live_release(sm, VK_LWIN, VK_CONTROL)
    assert sm.on_key(ev(VK_LWIN, True)) == Decision(swallow=False)
    assert sm.on_key(ev(VK_LWIN, True)) == Decision(swallow=False)
    assert sm.on_key(ev(VK_LWIN, False)) == Decision(swallow=False)


def test_resync_forgets_releases_of_keys_no_longer_down(rec, clock):
    sm = ChordStateMachine([CTRL_WIN], rec.callbacks(), clock.now)
    sm.on_key(ev(VK_LCONTROL, True))
    sm.on_key(ev(VK_LWIN, True))
    live_release(sm, VK_LWIN, VK_LCONTROL)
    sm.resync(lambda vk: False)
    sm.on_key(ev(VK_LWIN, True))
    assert sm.on_key(ev(VK_LWIN, True)) == Decision(swallow=False)


def test_win_mask_on_every_completion_including_latch_and_latch_end(rec, clock):
    sm = ChordStateMachine([WIN_CHORD], rec.callbacks(), clock.now)
    win_ups = []
    for _ in range(3):
        sm.on_key(ev(VK_LWIN, True))
        sm.on_key(ev(VK_H, True))
        clock.advance(0.1)
        sm.on_key(ev(VK_H, False))
        win_ups.append(sm.on_key(ev(VK_LWIN, False)))
        clock.advance(0.2)
    assert all(d == Decision(swallow=False, inject=MASK_PAIR) for d in win_ups)
    assert rec.events == [
        ("pressed", WIN_CHORD, 1),
        ("latched", WIN_CHORD, 1),
        ("released", WIN_CHORD, 1),
    ]


def test_win_mask_when_the_win_key_goes_up_first(rec, clock):
    sm = ChordStateMachine([WIN_CHORD], rec.callbacks(), clock.now)
    sm.on_key(ev(VK_LWIN, True))
    sm.on_key(ev(VK_H, True))
    clock.advance(0.3)
    assert sm.on_key(ev(VK_LWIN, False)) == Decision(swallow=False, inject=MASK_PAIR)
    assert rec.events == [("pressed", WIN_CHORD, 1), ("released", WIN_CHORD, 1)]
    assert sm.on_key(ev(VK_H, False)) == Decision(swallow=True)


def test_right_win_chord_is_masked_too(rec, clock):
    sm = ChordStateMachine([Chord((VK_RWIN,))], rec.callbacks(), clock.now)
    assert sm.on_key(ev(VK_RWIN, True)) == Decision(swallow=False)
    clock.advance(0.3)
    assert sm.on_key(ev(VK_RWIN, False)) == Decision(swallow=False, inject=MASK_PAIR)


# Probe and mask events ----------------------------------------------------------


def test_probe_key_with_probe_tag_is_swallowed_and_sets_probe_seen(sm, rec):
    down = ev(VK_PROBE, True, extra_info=PROBE_TAG, injected=True)
    up = ev(VK_PROBE, False, extra_info=PROBE_TAG, injected=True)
    assert sm.probe_seen is False
    assert sm.on_key(down) == Decision(swallow=True)
    assert sm.probe_seen is True
    assert sm.on_key(up) == Decision(swallow=True)
    assert sm.consume_probe() is True
    assert sm.probe_seen is False
    assert sm.consume_probe() is False
    assert rec.events == []
    assert sm.state is ChordState.IDLE


def test_untagged_key_e8_is_not_a_probe(sm):
    assert sm.on_key(ev(VK_PROBE, True, injected=True)).swallow is False
    assert sm.probe_seen is False


def test_probe_without_the_injected_flag_is_not_a_probe(sm):
    """Spec 13: the probe is recognised by VK, the private tag and LLKHF_INJECTED."""
    assert sm.on_key(ev(VK_PROBE, True, extra_info=PROBE_TAG)).swallow is False
    assert sm.on_key(ev(VK_PROBE, False, extra_info=PROBE_TAG)).swallow is False
    assert sm.probe_seen is False


def test_probe_tag_on_another_vk_is_not_a_probe(sm, rec):
    """B3-25's third condition: the vk has to be VK_PROBE as well."""
    assert sm.on_key(ev(VK_A, True, extra_info=PROBE_TAG, injected=True)).swallow is False
    assert sm.on_key(ev(VK_A, False, extra_info=PROBE_TAG, injected=True)).swallow is False
    assert sm.probe_seen is False
    assert rec.events == []


def test_mask_tagged_events_pass_through(sm, rec):
    down = ev(MASK_VK, True, extra_info=MASK_TAG, injected=True)
    up = ev(MASK_VK, False, extra_info=MASK_TAG, injected=True)
    assert sm.on_key(down) == Decision(swallow=False)
    assert sm.on_key(up) == Decision(swallow=False)
    assert sm.probe_seen is False
    assert rec.events == []
    assert sm.state is ChordState.IDLE


def test_probe_and_mask_events_do_not_disturb_an_active_chord(sm, rec, clock):
    press_chord(sm, MAIN)
    sm.on_key(ev(VK_PROBE, True, extra_info=PROBE_TAG, injected=True))
    sm.on_key(ev(VK_PROBE, False, extra_info=PROBE_TAG, injected=True))
    sm.on_key(ev(MASK_VK, True, extra_info=MASK_TAG, injected=True))
    sm.on_key(ev(MASK_VK, False, extra_info=MASK_TAG, injected=True))
    assert sm.state is ChordState.HELD
    clock.advance(0.3)
    sm.on_key(ev(VK_D, False))
    assert rec.events == [("pressed", MAIN, 1), ("released", MAIN, 1)]


# Callback exceptions -------------------------------------------------------------------


def test_callback_exception_is_caught_counted_and_state_stays_consistent(clock):
    sm = ChordStateMachine([MAIN], raising_callbacks(), clock.now)
    decisions = press_chord(sm, MAIN)
    assert decisions[-1].swallow is True
    assert sm.state is ChordState.HELD
    assert sm.callback_exceptions == 1
    clock.advance(0.3)
    sm.on_key(ev(VK_D, False))
    assert sm.state is ChordState.IDLE
    assert sm.callback_exceptions == 2


def test_callback_exception_is_reported_to_on_exception(clock):
    seen: list[BaseException] = []
    sm = ChordStateMachine([MAIN], raising_callbacks(), clock.now, on_exception=seen.append)
    press_chord(sm, MAIN)
    assert len(seen) == 1
    assert isinstance(seen[0], RuntimeError)


def test_resync_drops_keys_that_are_no_longer_physically_down(sm, rec):
    sm.on_key(ev(VK_LCONTROL, True))
    sm.on_key(ev(VK_LMENU, True))
    sm.resync(lambda vk: vk == VK_LMENU)
    sm.on_key(ev(VK_D, True))
    assert rec.events == [], "Ctrl went up while the hook was dead"
    sm.on_key(ev(VK_D, False))
    sm.on_key(ev(VK_LCONTROL, True))
    sm.on_key(ev(VK_D, True))
    assert rec.events == [("pressed", MAIN, 1)]


# HotkeyThread ------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def restore_switch_interval():
    saved = sys.getswitchinterval()
    yield
    sys.setswitchinterval(saved)


@pytest.fixture
def hook() -> FakeHook:
    return FakeHook()


@pytest.fixture
def started(hook: FakeHook, rec: Recorder):
    thread = HotkeyThread([MAIN, WIN_CHORD], rec.callbacks(), hook, now=hook.clock.now)
    thread.start()
    assert hook.loop_started.wait(1.0)
    yield thread
    thread.stop()


def press_main(hook: FakeHook) -> bool:
    hook.press(VK_LCONTROL)
    hook.press(VK_LMENU)
    return hook.press(VK_D)


def test_thread_installs_hook_registers_timers_and_raises_priority(started, hook):
    assert started.daemon is True
    assert hook.install_log == [1]
    assert hook.callback is not None
    assert hook.timers == {TIMER_TICK: TICK_MS, TIMER_PROBE: 15000}
    assert TICK_MS == 50
    assert hook.priority_calls == [started.ident]
    assert sys.getswitchinterval() == pytest.approx(0.001)
    assert started.stats.probes_sent == 0
    assert started.stats.probes_missed == 0
    assert started.stats.reinstalls == 0
    assert started.stats.hook_exceptions == 0
    assert started.stats.install_failures == 0
    assert started.stats.last_error == ""


def test_thread_probe_interval_is_configurable(hook, rec):
    thread = HotkeyThread([MAIN], rec.callbacks(), hook, probe_interval_s=2.5, probe_check_s=0.2)
    thread.start()
    assert hook.loop_started.wait(1.0)
    try:
        assert hook.timers[TIMER_PROBE] == 2500
        hook.fire_timer(TIMER_PROBE)
        assert hook.timers[TIMER_PROBE_CHECK] == 200
    finally:
        thread.stop()


def test_probe_sent_only_while_not_recording(started, hook):
    hook.fire_timer(TIMER_PROBE)
    assert hook.sent == [(VK_PROBE, True, PROBE_TAG), (VK_PROBE, False, PROBE_TAG)]
    assert started.stats.probes_sent == 1
    assert hook.timers[TIMER_PROBE_CHECK] == 100
    hook.fire_timer(TIMER_PROBE_CHECK)
    assert TIMER_PROBE_CHECK not in hook.timers

    press_main(hook)  # HELD
    hook.fire_timer(TIMER_PROBE)
    assert len(hook.sent) == 2
    assert TIMER_PROBE_CHECK not in hook.timers
    hook.clock.advance(0.1)
    hook.release(VK_D)  # TAP_WINDOW
    hook.fire_timer(TIMER_PROBE)
    assert len(hook.sent) == 2
    hook.clock.advance(0.1)
    hook.press(VK_D)  # LATCHED
    hook.release(VK_D)
    hook.fire_timer(TIMER_PROBE)
    assert len(hook.sent) == 2
    hook.press(VK_D)  # ends the latch
    hook.release(VK_D)
    hook.fire_timer(TIMER_PROBE)
    assert len(hook.sent) == 4
    assert started.stats.probes_sent == 2


def test_seen_probe_causes_no_reinstall(started, hook):
    hook.fire_timer(TIMER_PROBE)
    assert ("event", VK_PROBE, True, True) in hook.trace, "the probe echoed and was swallowed"
    assert ("event", VK_PROBE, False, True) in hook.trace
    hook.fire_timer(TIMER_PROBE_CHECK)
    assert started.stats.reinstalls == 0
    assert started.stats.probes_missed == 0
    assert hook.install_log == [1]
    assert hook.uninstall_log == []
    assert ("kill", TIMER_PROBE_CHECK, None) in hook.timer_log


def test_missed_probe_causes_exactly_one_reinstall(started, hook):
    hook.drop_hook()
    hook.fire_timer(TIMER_PROBE)
    assert ("lost", VK_PROBE, True, PROBE_TAG) in hook.trace
    hook.fire_timer(TIMER_PROBE_CHECK)
    assert started.stats.reinstalls == 1
    assert started.stats.probes_missed == 1
    assert hook.install_log == [1, 2]
    assert hook.callback is not None
    assert started.stats.hook_exceptions == 0, "a failed unhook of a dead handle is expected"
    # the new hook works and the next probe is seen: no further reinstall
    hook.fire_timer(TIMER_PROBE)
    hook.fire_timer(TIMER_PROBE_CHECK)
    assert started.stats.reinstalls == 1
    assert started.stats.probes_missed == 1
    assert started.stats.probes_sent == 2
    assert press_main(hook) is True


def test_reinstall_resyncs_physical_key_state(started, hook, rec):
    hook.press(VK_LCONTROL)
    hook.drop_hook()
    hook.keys_down.discard(VK_LCONTROL)  # released while the hook was dead
    started.request_reinstall()
    hook.fire_timer(TIMER_TICK)
    assert started.stats.reinstalls == 1
    hook.press(VK_LMENU)
    hook.press(VK_D)
    assert rec.events == []
    hook.release(VK_D)
    hook.press(VK_LCONTROL)
    hook.press(VK_D)
    assert rec.events == [("pressed", MAIN, 1)]


def test_request_reinstall_is_handled_on_the_next_tick(started, hook):
    started.request_reinstall()
    assert started.stats.reinstalls == 0
    assert hook.install_log == [1]
    hook.fire_timer(TIMER_TICK)
    assert started.stats.reinstalls == 1
    assert hook.uninstall_log == [1]
    assert hook.install_log == [1, 2]
    hook.fire_timer(TIMER_TICK)
    assert started.stats.reinstalls == 1, "one request, one reinstall"


def test_install_failure_is_counted_reported_and_retried_on_ticks(hook, rec):
    hook.install_failures_left = 2
    errors: list[str] = []
    thread = HotkeyThread([MAIN], rec.callbacks(), hook, now=hook.clock.now, on_error=errors.append)
    thread.start()
    assert hook.loop_started.wait(1.0)
    try:
        assert thread.is_alive(), "an install failure never kills the thread"
        assert thread.stats.install_failures == 1
        assert hook.install_log == []
        assert hook.callback is None
        hook.fire_timer(TIMER_TICK)
        assert thread.stats.install_failures == 2
        assert len(errors) == 1, "the same error every 50 ms is one tray warning"
        assert "SetWindowsHookEx" in errors[0]
        assert thread.stats.last_error == errors[0]
        hook.fire_timer(TIMER_TICK)
        assert thread.stats.install_failures == 2
        assert hook.install_log == [1]
        assert thread.stats.reinstalls == 0, "the first successful install is not a reinstall"
        assert press_main(hook) is True
        assert rec.events == [("pressed", MAIN, 1)]
        hook.fire_timer(TIMER_TICK)
        assert hook.install_log == [1], "no further attempts once installed"
    finally:
        thread.stop()


def test_a_different_install_error_is_reported_again(hook, rec):
    hook.install_failures_left = 3
    errors: list[str] = []
    thread = HotkeyThread([MAIN], rec.callbacks(), hook, now=hook.clock.now, on_error=errors.append)
    thread.start()
    assert hook.loop_started.wait(1.0)
    try:
        assert len(errors) == 1
        hook.fire_timer(TIMER_TICK)
        assert len(errors) == 1, "the same message again is not a second warning"
        hook.install_error = "SetWindowsHookEx failed: out of hooks"
        hook.fire_timer(TIMER_TICK)
        assert len(errors) == 2, "a different reason is worth telling the user about"
        assert "out of hooks" in errors[1]
        assert thread.stats.last_error == errors[1]
        assert thread.stats.install_failures == 3
    finally:
        thread.stop()


def test_failed_reinstall_is_retried_on_the_next_tick(started, hook):
    hook.drop_hook()
    hook.install_failures_left = 1
    hook.fire_timer(TIMER_PROBE)
    hook.fire_timer(TIMER_PROBE_CHECK)
    assert started.stats.probes_missed == 1
    assert started.stats.install_failures == 1
    assert started.stats.reinstalls == 0
    assert hook.callback is None
    hook.fire_timer(TIMER_TICK)
    assert started.stats.reinstalls == 1
    assert hook.install_log == [1, 2]
    assert press_main(hook) is True


def test_on_error_exceptions_do_not_break_the_thread(hook, rec):
    hook.install_failures_left = 1

    def bad_handler(_message: str) -> None:
        raise RuntimeError("tray is gone")

    thread = HotkeyThread([MAIN], rec.callbacks(), hook, on_error=bad_handler)
    thread.start()
    assert hook.loop_started.wait(1.0)
    try:
        assert thread.stats.install_failures == 1
        hook.fire_timer(TIMER_TICK)
        assert hook.install_log == [1]
    finally:
        thread.stop()


def test_tick_drives_the_tap_window_on_the_thread(started, hook, rec):
    press_main(hook)
    hook.clock.advance(0.1)
    hook.release(VK_D)
    hook.fire_timer(TIMER_TICK)
    assert rec.events == [("pressed", MAIN, 1)]
    hook.clock.advance(0.4)
    hook.fire_timer(TIMER_TICK)
    assert rec.events == [("pressed", MAIN, 1), ("discarded", 1)]


def test_end_recording_is_applied_on_the_hotkey_thread_at_the_next_tick(started, hook, rec):
    assert started.recording is False
    press_main(hook)
    assert rec.events == [("pressed", MAIN, 1)]
    assert started.recording is True
    started.end_recording(1)
    hook.fire_timer(TIMER_PROBE)
    assert hook.sent == [], "still recording until the tick applies the end"
    assert started.recording is True, "the caller's thread does not touch the state machine"
    hook.fire_timer(TIMER_TICK)
    assert started.recording is False
    hook.fire_timer(TIMER_PROBE)
    assert len(hook.sent) == 2, "the end was applied: probes resume"
    hook.clock.advance(0.3)
    assert hook.release(VK_D) is True, "held chord keys stay swallowed to their key-up"
    assert rec.events == [("pressed", MAIN, 1)], "no released for the ended dictation"
    assert hook.press(VK_D) is True
    assert rec.events[-1] == ("pressed", MAIN, 2)


def test_end_recording_after_stop_is_dropped(hook, rec):
    """No tick will ever run to apply it, so the queue must not keep growing."""
    thread = HotkeyThread([MAIN], rec.callbacks(), hook, now=hook.clock.now)
    thread.start()
    assert hook.loop_started.wait(1.0)
    press_main(hook)
    thread.stop()
    thread.end_recording(1)
    thread.end_recording(2)
    # the queue is private, and after stop() there is no other way to observe it
    assert thread._pending_ends == []


def test_synthetic_key_up_is_sent_before_the_completing_key_and_passes_through(started, hook, rec):
    hook.press(VK_D)
    hook.press(VK_LCONTROL)
    assert hook.press(VK_LMENU) is False, "the completing modifier reaches the app"
    assert hook.sent == [(VK_D, False, MASK_TAG)], "D is released for the app (B3-11)"
    assert rec.events == [("pressed", MAIN, 1)]
    sent = hook.trace.index(("sent", VK_D, False, MASK_TAG))
    assert sent < hook.trace.index(("event", VK_LMENU, True, False)), "injected first"
    assert ("event", VK_D, False, False) in hook.trace, "and passed straight through the hook"
    assert hook.press(VK_D) is True, "the physical key is still held and still swallowed"
    hook.clock.advance(0.3)
    assert hook.release(VK_D) is True, "and so is its physical key-up"
    assert rec.events == [("pressed", MAIN, 1), ("released", MAIN, 1)]
    assert hook.sent == [(VK_D, False, MASK_TAG)], "exactly one synthetic key-up"


def test_end_recording_with_a_stale_id_on_the_thread_is_ignored(started, hook, rec):
    press_main(hook)
    hook.clock.advance(0.3)
    hook.release(VK_D)
    hook.press(VK_D)
    assert rec.events[-1] == ("pressed", MAIN, 2)
    started.end_recording(1)
    hook.fire_timer(TIMER_TICK)
    hook.clock.advance(0.3)
    hook.release(VK_D)
    assert rec.events[-1] == ("released", MAIN, 2)


def test_win_mask_is_sent_before_the_win_key_up_passes_on(started, hook, rec):
    hook.press(VK_LWIN)
    assert hook.press(VK_H) is True
    hook.clock.advance(0.3)
    assert hook.release(VK_H) is True
    assert hook.release(VK_LWIN) is False
    assert hook.sent == list(MASK_PAIR)
    sent_down = hook.trace.index(("sent", MASK_VK, True, MASK_TAG))
    assert hook.trace[sent_down + 1] == ("sent", MASK_VK, False, MASK_TAG)
    assert sent_down < hook.trace.index(("event", VK_LWIN, False, False))
    # the echoed mask keys pass through the hook unswallowed
    assert ("event", MASK_VK, True, False) in hook.trace
    assert ("event", MASK_VK, False, False) in hook.trace
    assert rec.events == [("pressed", WIN_CHORD, 1), ("released", WIN_CHORD, 1)]


def test_update_chords_on_the_thread(started, hook, rec):
    started.update_chords([LANG_SQ])
    hook.press(VK_LCONTROL)
    hook.press(VK_LMENU)
    assert hook.press(VK_D) is False
    hook.release(VK_D)
    assert hook.press(VK_E) is True
    assert rec.events == [("pressed", LANG_SQ, 1)]
    assert hook.press(VK_ESCAPE) is True
    assert rec.events == [("pressed", LANG_SQ, 1), ("cancelled", 1)]


def test_callback_exception_is_counted_and_never_raised_into_the_hook(hook):
    thread = HotkeyThread([MAIN], raising_callbacks(), hook, now=hook.clock.now)
    thread.start()
    assert hook.loop_started.wait(1.0)
    try:
        assert press_main(hook) is True
        assert thread.stats.hook_exceptions == 1
        assert thread.is_alive()
    finally:
        thread.stop()


def test_hook_callback_survives_a_broken_state_machine(started, hook, monkeypatch):
    def broken(self, event):
        raise ValueError("state machine broke")

    monkeypatch.setattr(ChordStateMachine, "on_key", broken)
    assert hook.press(VK_D) is False, "pass the key through rather than fail the hook"
    assert started.stats.hook_exceptions == 1


def test_timer_handler_exceptions_are_counted(started, hook, monkeypatch):
    def broken(self):
        raise ValueError("tick broke")

    monkeypatch.setattr(ChordStateMachine, "on_tick", broken)
    hook.fire_timer(TIMER_TICK)
    assert started.stats.hook_exceptions == 1
    assert started.is_alive()


def test_stop_joins_within_one_second(hook, rec):
    thread = HotkeyThread([MAIN], rec.callbacks(), hook)
    thread.start()
    assert hook.loop_started.wait(1.0)
    t0 = time.perf_counter()
    thread.stop()
    assert time.perf_counter() - t0 < 1.0
    assert not thread.is_alive()
    assert hook.loop_exited.is_set()
    assert hook.uninstall_log == [1]
    assert hook.hooks == {}
    thread.stop()  # idempotent


def test_stop_before_start_is_harmless(hook, rec):
    thread = HotkeyThread([MAIN], rec.callbacks(), hook)
    thread.stop()
    assert not thread.is_alive()


# Stress -----------------------------------------------------------------------------------


def _spin(stop_at: float) -> None:
    x = 1
    while time.perf_counter() < stop_at:
        x = (x * 3 + 1) % 1000003


def _stress_script(hook: FakeHook) -> list[KeyEvent]:
    """A realistic 20-event cycle: a hold, a tap, unrelated typing, a probe, a Win chord."""
    e = hook.event
    return [
        e(VK_LCONTROL, True), e(VK_LMENU, True), e(VK_D, True),
        e(VK_D, True), e(VK_D, False), e(VK_LMENU, False), e(VK_LCONTROL, False),
        e(VK_A, True), e(VK_A, False),
        e(VK_PROBE, True, injected=True, extra_info=PROBE_TAG),
        e(VK_PROBE, False, injected=True, extra_info=PROBE_TAG),
        e(VK_LWIN, True), e(VK_H, True), e(VK_H, False), e(VK_LWIN, False),
        e(VK_LCONTROL, True), e(VK_LMENU, True), e(VK_D, True), e(VK_D, False),
        e(VK_LCONTROL, False),
    ]  # fmt: skip


def test_stress_per_event_handling_p99_under_1ms_with_a_cpu_bound_thread(started, hook):
    """In-process budget: the state machine's own per-event cost stays under 1 ms while
    another Python thread is CPU-bound, with the 1 ms switch interval the thread sets.

    The 300 ms end-to-end budget with the real WH_KEYBOARD_LL hook (spec 20.2, decision
    V4-11) is an integration test that belongs to a later task; it is not covered here.

    Between events the feeder yields the GIL to the spinner so that each timed region
    starts on a fresh GIL grant: what is measured is handling cost, not the interpreter's
    switch-interval scheduling, which the real-hook stress test covers.
    """
    script = _stress_script(hook)
    events = [script[i % len(script)] for i in range(2000)]
    assert sys.getswitchinterval() == pytest.approx(0.001)
    spinner = threading.Thread(target=_spin, args=(time.perf_counter() + 1.0,), daemon=True)
    samples_ns: list[int] = []
    handled_during_spin = 0
    spinner.start()
    for event in events:
        if spinner.is_alive():
            time.sleep(0.0005)
            handled_during_spin += 1
        t0 = time.perf_counter_ns()
        hook.feed(event)
        samples_ns.append(time.perf_counter_ns() - t0)
    spinner.join()
    assert len(samples_ns) == 2000
    assert handled_during_spin > 0
    p99 = sorted(samples_ns)[int(0.99 * len(samples_ns)) - 1]
    assert p99 < 1_000_000, f"p99 was {p99 / 1e6:.3f} ms"
    assert started.stats.hook_exceptions == 0
    # the whole script was handled correctly, not just quickly
    assert started.stats.probes_sent == 0
    assert hook.sent.count((MASK_VK, True, MASK_TAG)) == 100


def test_chords_property_reads_the_current_set(started):
    assert started.chords == (MAIN, WIN_CHORD)
    started.update_chords([MAIN])
    assert started.chords == (MAIN,)
    started.update_chords(())
    assert started.chords == ()


def test_a_successful_install_after_a_failure_clears_the_error(hook, rec):
    hook.install_failures_left = 1
    errors: list[str] = []
    thread = HotkeyThread([MAIN], rec.callbacks(), hook, now=hook.clock.now, on_error=errors.append)
    thread.start()
    assert hook.loop_started.wait(1.0)
    try:
        assert len(errors) == 1 and errors[0]
        assert thread.stats.last_error == errors[0]
        hook.fire_timer(TIMER_TICK)
        assert hook.install_log == [1]
        assert errors == [errors[0], ""], "the recovery is reported once, as an empty message"
        assert thread.stats.last_error == ""
        hook.fire_timer(TIMER_TICK)
        assert len(errors) == 2, "a healthy tick reports nothing"
        assert thread.stats.install_failures == 1
    finally:
        thread.stop()


def test_a_clean_first_install_reports_nothing(hook, rec):
    errors: list[str] = []
    thread = HotkeyThread([MAIN], rec.callbacks(), hook, now=hook.clock.now, on_error=errors.append)
    thread.start()
    assert hook.loop_started.wait(1.0)
    try:
        hook.fire_timer(TIMER_TICK)
        assert errors == []
        assert thread.stats.last_error == ""
    finally:
        thread.stop()


# --- Esc while the writing model works (spec 8.5) ---


def test_escape_cancels_the_composition_the_pipeline_armed(sm, rec):
    sm.set_writing(7)
    assert sm.recording is False
    assert sm.writing_id == 7
    assert sm.on_key(ev(VK_ESCAPE, True)) == Decision(swallow=True)
    assert rec.events == [("cancelled", 7)]
    assert sm.writing_id is None
    assert sm.on_key(ev(VK_ESCAPE, False)).swallow is True


def test_a_second_escape_after_a_cancelled_composition_passes_through(sm, rec):
    sm.set_writing(7)
    sm.on_key(ev(VK_ESCAPE, True))
    sm.on_key(ev(VK_ESCAPE, False))
    assert sm.on_key(ev(VK_ESCAPE, True)).swallow is False
    assert rec.events == [("cancelled", 7)]


def test_disarming_the_composition_lets_escape_through_again(sm, rec):
    sm.set_writing(7)
    sm.set_writing(None)
    assert sm.on_key(ev(VK_ESCAPE, True)) == Decision(swallow=False)
    assert rec.events == []


def test_a_recording_wins_over_a_composition_still_being_written(sm, rec, clock):
    sm.set_writing(7)
    press_chord(sm, MAIN)
    clock.advance(1.0)
    assert sm.on_key(ev(VK_ESCAPE, True)) == Decision(swallow=True)
    assert rec.events[-1] == ("cancelled", 1)
    assert sm.writing_id == 7


def test_a_chord_still_fires_while_a_composition_is_being_written(sm, rec, clock):
    sm.set_writing(7)
    press_chord(sm, MAIN)
    assert rec.events[-1] == ("pressed", MAIN, 1)
    assert sm.writing_id == 7


def test_the_thread_arms_the_composition_from_any_thread():
    hook = FakeHook()
    thread = HotkeyThread([MAIN], Recorder().callbacks(), hook, now=hook.clock.now)
    thread.set_writing(11)
    assert thread.writing_id == 11
    thread.set_writing(None)
    assert thread.writing_id is None
