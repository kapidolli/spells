"""bench/synth_clips.py: pure helpers only, no TTS, no network, no audio devices."""

import random
import sys
from array import array
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parents[2] / "bench"
sys.path.insert(0, str(BENCH_DIR))

import prompts as bench_prompts
import synth_clips


def pcm(samples):
    return array("h", samples).tobytes()


def samples(data):
    return array("h", data).tolist()


def row(client, path, sentence, up="2", down="0"):
    return {
        "client_id": client,
        "path": path,
        "sentence": sentence,
        "up_votes": up,
        "down_votes": down,
    }


def test_noise_is_seeded_and_stays_under_the_peak():
    first = synth_clips.noise(0.1, random.Random(3))
    second = synth_clips.noise(0.1, random.Random(3))
    assert first == second
    assert len(first) == 2 * 1600
    assert max(abs(v) for v in samples(first)) <= synth_clips.NOISE_PEAK


def test_pad_adds_lead_and_tail_noise():
    body = pcm([1000] * 800)
    padded = synth_clips.pad(body, random.Random(1))
    lead = int(synth_clips.LEAD_S * 16000)
    tail = int(synth_clips.TAIL_S * 16000)
    assert len(padded) == len(body) + 2 * (lead + tail)
    assert padded[2 * lead : 2 * lead + len(body)] == body


def test_trim_silence_keeps_speech_with_a_margin():
    quiet = [0] * 16000
    loud = [8000, -8000] * 4000
    trimmed = samples(synth_clips.trim_silence(pcm(quiet + loud + quiet)))
    margin = int(synth_clips.TRIM_MARGIN_S * 16000)
    frame = 480
    assert len(loud) <= len(trimmed) <= len(loud) + 2 * (margin + frame)
    assert 8000 in trimmed


def test_trim_silence_of_pure_silence_is_empty():
    assert synth_clips.trim_silence(pcm([0] * 16000)) == b""


def test_join_puts_a_gap_between_segments():
    a = pcm([100] * 160)
    b = pcm([200] * 160)
    joined = synth_clips.join([a, b], random.Random(2))
    gap = int(synth_clips.GAP_S * 16000)
    assert len(joined) == len(a) + len(b) + 2 * gap
    assert joined.startswith(a) and joined.endswith(b)


def test_select_cv_sets_groups_one_speaker_per_clip_within_bounds():
    rows = [
        row("s1", "a1.mp3", "Një."),
        row("s1", "a2.mp3", "Dy."),
        row("s1", "a3.mp3", "Tre."),
        row("s2", "b1.mp3", "Katër."),
        row("s2", "b2.mp3", "Pesë.", down="1"),
        row("s2", "b3.mp3", "Gjashtë."),
        row("s3", "c1.mp3", "Shtatë.", up="1"),
    ]
    durations = {"a1.mp3": 9000, "a2.mp3": 9000, "a3.mp3": 9000, "b1.mp3": 8000, "b3.mp3": 9000}
    sets = synth_clips.select_cv_sets(rows, durations, count=5, min_s=16.0, max_s=30.0)
    assert [[r["path"] for r in s] for s in sets] == [
        ["a1.mp3", "a2.mp3"],
        ["b1.mp3", "b3.mp3"],
    ]


def test_select_cv_sets_skips_a_sentence_that_would_pass_the_maximum():
    rows = [row("s1", "a1.mp3", "Një."), row("s1", "a2.mp3", "Dy."), row("s1", "a3.mp3", "Tre.")]
    durations = {"a1.mp3": 12000, "a2.mp3": 20000, "a3.mp3": 6000}
    sets = synth_clips.select_cv_sets(rows, durations, count=1, min_s=16.0, max_s=30.0)
    assert [r["path"] for r in sets[0]] == ["a1.mp3", "a3.mp3"]


def test_select_cv_sets_honours_the_count():
    rows = [row(f"s{i}", f"{i}.mp3", "Po.") for i in range(6)]
    durations = {f"{i}.mp3": 20000 for i in range(6)}
    assert len(synth_clips.select_cv_sets(rows, durations, count=3, min_s=16.0, max_s=30.0)) == 3


def test_merge_entries_replaces_by_id_and_keeps_the_rest():
    def entry(clip_id, text):
        return bench_prompts.ManifestEntry(
            id=clip_id,
            language="sq",
            category="long",
            reference=text,
            text=text,
            duration_s=12.0,
            recorded_at="now",
            file=f"{clip_id}.wav",
        )

    merged = synth_clips.merge_entries([entry("a", "old"), entry("b", "keep")], [entry("a", "new")])
    assert {e.id: e.text for e in merged} == {"a": "new", "b": "keep"}


def test_cv_ids_are_file_safe_and_carry_language_and_category():
    assert synth_clips.cv_id("long", 3) == "sq-cvlong-03"
    assert synth_clips.cv_id("short", 12) == "sq-cvshort-12"
