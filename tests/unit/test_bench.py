"""bench/prompts.py and bench/run.py: pure logic only, no engines, no network, no audio."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

BENCH_DIR = Path(__file__).resolve().parents[2] / "bench"
BUILD_DIR = Path(__file__).resolve().parents[2] / "build"
sys.path.insert(0, str(BENCH_DIR))
sys.path.insert(0, str(BUILD_DIR))

import fetch as build_fetch
import prompts as bench_prompts
import run as bench_run

EM_DASH = "\u2014"

# ------------------------------------------------------------------ normalisation


def test_normalise_words_lowercases_and_strips_punctuation():
    assert bench_run.normalise_words("Hello, World!") == ["hello", "world"]


def test_normalise_words_keeps_digits_and_internal_apostrophes():
    assert bench_run.normalise_words("Bahnhofstrasse 14, don't stop") == [
        "bahnhofstrasse",
        "14",
        "don't",
        "stop",
    ]


def test_normalise_words_keeps_diacritics_and_collapses_whitespace():
    assert bench_run.normalise_words("  Zürich \n  hätte  ") == ["zürich", "hätte"]


def test_normalise_words_on_empty_text():
    assert bench_run.normalise_words("   ...  ") == []


# ------------------------------------------------------------------ levenshtein


def test_levenshtein_identical_is_zero():
    assert bench_run.levenshtein(["a", "b", "c"], ["a", "b", "c"]) == 0


def test_levenshtein_one_substitution():
    assert bench_run.levenshtein(["a", "b", "c"], ["a", "x", "c"]) == 1


def test_levenshtein_one_insertion():
    assert bench_run.levenshtein(["a", "b"], ["a", "x", "b"]) == 1


def test_levenshtein_one_deletion():
    assert bench_run.levenshtein(["a", "b", "c"], ["a", "c"]) == 1


def test_levenshtein_against_empty_is_the_other_length():
    assert bench_run.levenshtein([], ["a", "b", "c"]) == 3
    assert bench_run.levenshtein(["a", "b", "c"], []) == 3
    assert bench_run.levenshtein([], []) == 0


def test_levenshtein_is_symmetric():
    a, b = ["one", "two", "three"], ["one", "three", "four", "five"]
    assert bench_run.levenshtein(a, b) == bench_run.levenshtein(b, a)


# ------------------------------------------------------------------ word error rate


def test_wer_identical_is_zero():
    assert bench_run.wer("Push it to main.", "push it to MAIN") == 0.0


def test_wer_one_substitution_in_four_words():
    assert bench_run.wer("push it to main", "push it to master") == pytest.approx(0.25)


def test_wer_insertion_counts_against_the_reference_length():
    assert bench_run.wer("push it to main", "push it up to main") == pytest.approx(0.25)


def test_wer_deletion_counts_against_the_reference_length():
    assert bench_run.wer("push it to main", "push to main") == pytest.approx(0.25)


def test_wer_can_exceed_one_when_the_hypothesis_runs_away():
    assert bench_run.wer("yes", "yes and here is a long answer") == pytest.approx(6.0)


def test_wer_is_none_for_an_empty_reference():
    """Silence clips have no reference words, so the rate is undefined (spec 18 step 3)."""
    assert bench_run.wer("", "thanks for watching") is None
    assert bench_run.wer("   ", "") is None


def test_wer_of_an_empty_hypothesis_is_one():
    assert bench_run.wer("push it to main", "") == pytest.approx(1.0)


# ------------------------------------------------------------------ percentile


def test_percentile_of_one_sample_is_that_sample():
    assert bench_run.percentile([42.0], 50) == pytest.approx(42.0)
    assert bench_run.percentile([42.0], 95) == pytest.approx(42.0)


def test_percentile_of_no_samples_is_none():
    assert bench_run.percentile([], 50) is None


def test_percentile_interpolates_between_samples():
    values = [10.0, 20.0, 30.0, 40.0]
    assert bench_run.percentile(values, 50) == pytest.approx(25.0)
    assert bench_run.percentile(values, 0) == pytest.approx(10.0)
    assert bench_run.percentile(values, 100) == pytest.approx(40.0)


def test_percentile_ignores_input_order():
    assert bench_run.percentile([30.0, 10.0, 20.0], 50) == pytest.approx(20.0)


def test_percentile_p95_of_many_samples():
    values = [float(n) for n in range(1, 101)]
    assert bench_run.percentile(values, 95) == pytest.approx(95.05)


# ------------------------------------------------------------------ duration buckets


@pytest.mark.parametrize(
    "duration_s,expected",
    [
        (0.4, "short"),
        (1.999, "short"),
        (2.0, "other"),
        (9.9, "other"),
        (10.0, "long"),
        (18.5, "long"),
        (30.0, "long"),
        (30.1, "other"),
    ],
)
def test_duration_bucket(duration_s, expected):
    assert bench_prompts.duration_bucket(duration_s) == expected


# ------------------------------------------------------------------ spec 12 targets


def _clip(**kwargs):
    defaults = {
        "id": "x",
        "language": "en",
        "category": "long",
        "duration_s": 15.0,
        "reference": "one two three",
        "spoken": "one two three",
        "raw_text": "one two three",
        "raw_ms": 500.0,
        "auto_text": "one two three",
        "auto_ms": 520.0,
        "auto_language": "en",
        "fallback_used": False,
        "cleaned_text": "One two three.",
        "cleaned_ms": 1500.0,
        "clean_reason": "ok",
        "used_llm": True,
        "skipped_ms": 600.0,
        "skipped_reason": "disabled",
        "error": "",
    }
    defaults.update(kwargs)
    return bench_run.ClipResult(**defaults)


def test_target_rows_pass_when_every_median_is_under_the_target():
    rows = bench_run.target_rows([_clip(id="a"), _clip(id="b")])
    assert [(row.name, row.passed) for row in rows] == [
        ("Release to raw transcript", True),
        ("Release to cleaned text, cleanup skipped", True),
        ("Release to cleaned text, cleanup run", True),
    ]
    assert [row.target_ms for row in rows] == [700.0, 800.0, 1800.0]
    assert rows[0].measured_ms == pytest.approx(500.0)
    assert rows[0].samples == 2


def test_target_rows_fail_on_the_median_not_the_worst_clip():
    clips = [_clip(id="a", raw_ms=100.0), _clip(id="b", raw_ms=200.0), _clip(id="c", raw_ms=9000.0)]
    rows = bench_run.target_rows(clips)
    assert rows[0].measured_ms == pytest.approx(200.0)
    assert rows[0].passed is True


def test_target_rows_fail_when_the_median_is_over():
    clips = [_clip(id="a", cleaned_ms=2000.0), _clip(id="b", cleaned_ms=2500.0)]
    rows = bench_run.target_rows(clips)
    assert rows[2].passed is False
    assert rows[2].measured_ms == pytest.approx(2250.0)


def test_target_rows_only_count_long_clips():
    clips = [_clip(id="short", duration_s=1.2, raw_ms=9000.0), _clip(id="long", raw_ms=300.0)]
    rows = bench_run.target_rows(clips)
    assert rows[0].samples == 1
    assert rows[0].measured_ms == pytest.approx(300.0)


def test_target_rows_without_long_clips_are_undecided():
    rows = bench_run.target_rows([_clip(id="short", duration_s=1.2)])
    assert all(row.samples == 0 and row.measured_ms is None for row in rows)
    assert all(row.passed is None for row in rows)


def test_cleanup_rows_are_invalid_while_llama_serves_from_its_cpu_build():
    """Spec 8.1 skips cleanup on the CPU build, so a CPU measurement is not a pass."""
    rows = bench_run.target_rows([_clip(id="a"), _clip(id="b")], llama_variant="cpu")
    assert rows[0].passed is True and rows[0].note == ""
    for row in rows[1:]:
        assert row.passed is None
        assert row.measured_ms is None
        assert row.note == bench_run.CPU_CLEANUP_INVALID
    assert bench_run.candidate_verdict(rows) == bench_run.CPU_CLEANUP_INVALID


def test_allow_cpu_cleanup_measures_the_rows_and_marks_them():
    rows = bench_run.target_rows(
        [_clip(id="a"), _clip(id="b")], llama_variant="cpu", allow_cpu_cleanup=True
    )
    assert [row.passed for row in rows] == [True, True, True]
    assert rows[2].measured_ms == pytest.approx(1500.0)
    assert rows[2].note == bench_run.CPU_CLEANUP_MEASURED
    assert bench_run.candidate_verdict(rows) == f"pass ({bench_run.CPU_CLEANUP_MEASURED})"


def _result(**kwargs):
    defaults = {
        "name": "cand",
        "model_file": "cand.gguf",
        "whisper_model": "whisper.bin",
        "gpu_name": "GPU",
        "clips": (_clip(id="a"), _clip(id="b")),
    }
    defaults.update(kwargs)
    return bench_run.CandidateResult(**defaults)


def test_cleanup_is_invalid_only_on_cpu_without_the_flag():
    assert bench_run.cleanup_is_invalid(_result()) is False
    assert bench_run.cleanup_is_invalid(_result(llama_variant="cpu")) is True
    assert (
        bench_run.cleanup_is_invalid(_result(llama_variant="cpu", allow_cpu_cleanup=True)) is False
    )


def test_a_cpu_candidate_is_not_listed_as_passing_in_the_summary():
    summary = bench_run.render_summary([_result(llama_variant="cpu")], "2026-09-17")
    assert "Candidates meeting the spec 12 targets: none" in summary
    assert bench_run.CPU_CLEANUP_INVALID in summary
    assert "--allow-cpu-cleanup" in summary


def test_the_candidate_report_says_why_the_cleaned_rows_are_missing():
    report = bench_run.render_candidate_report(_result(llama_variant="cpu"), "2026-09-17")
    assert "llama served from its CPU build" in report
    assert "not valid" in report


def test_candidate_passes_only_when_every_decided_row_passes():
    good = [_clip(id="a"), _clip(id="b")]
    bad = [_clip(id="a", cleaned_ms=2500.0), _clip(id="b", cleaned_ms=2600.0)]
    assert bench_run.candidate_passes(bench_run.target_rows(good)) is True
    assert bench_run.candidate_passes(bench_run.target_rows(bad)) is False
    assert bench_run.candidate_passes(bench_run.target_rows([_clip(duration_s=1.0)])) is None


# ------------------------------------------------------------------ fallback rate


def test_fallback_rate_counts_long_clips_only():
    """Spec 7.2 step 1 never reaches step 3 below 2 s, and 7.2 puts its 10% line on long clips."""
    clips = [
        _clip(id="l1", duration_s=15.0, fallback_used=True),
        _clip(id="l2", duration_s=12.0, fallback_used=False),
        _clip(id="s1", duration_s=1.0, fallback_used=True),
        _clip(id="s2", duration_s=1.2, fallback_used=True),
        _clip(id="o1", duration_s=5.0, fallback_used=True),
    ]
    assert bench_run._fallback_rate(clips, "en") == (1, 2)


def test_fallback_rate_ignores_clips_that_could_not_be_measured():
    clips = [_clip(id="a", fallback_used=True), _clip(id="b", fallback_used=True, error="boom")]
    assert bench_run._fallback_rate(clips, "en") == (1, 1)


def test_fallback_rate_of_a_language_without_long_clips_is_zero_of_zero():
    assert bench_run._fallback_rate([_clip(duration_s=1.0)], "en") == (0, 0)


# ------------------------------------------------------------------ language WER


def test_language_wer_is_a_micro_average_over_reference_words():
    """A long clip has to weigh more than a short one: total edits over total words."""
    clips = [
        _clip(id="long", reference="a b c d e f g h i j", raw_text="a b c d e f g h i x"),
        _clip(id="short", reference="k l", raw_text="x y"),
    ]
    # 1 edit in 10 words plus 2 edits in 2 words: 3 / 12, not the mean of 10% and 100%.
    assert bench_run._language_wer(clips, "en") == pytest.approx(3 / 12)


def test_language_wer_skips_clips_without_reference_words():
    clips = [
        _clip(id="silence", reference="", raw_text="thanks for watching"),
        _clip(id="spoken", reference="a b c d", raw_text="a b c x"),
    ]
    assert bench_run._language_wer(clips, "en") == pytest.approx(0.25)


def test_language_wer_is_none_when_no_clip_has_a_reference():
    assert bench_run._language_wer([_clip(reference="")], "en") is None


def test_language_wer_against_the_spoken_text_uses_the_spoken_form():
    clips = [_clip(reference="a b", spoken="a b um", raw_text="a b um")]
    assert bench_run._language_wer(clips, "en") == pytest.approx(0.5)
    assert bench_run._language_wer(clips, "en", spoken=True) == pytest.approx(0.0)


# ------------------------------------------------------------------ gate and guard reasons


def test_cleanup_attempts_count_accepted_and_guard_rejected_clips_only():
    clips = [
        _clip(id="a", clean_reason="ok"),
        _clip(id="b", clean_reason="preamble"),
        _clip(id="c", clean_reason="short_clean"),
        _clip(id="d", clean_reason="no_transcript"),
    ]
    assert bench_run.cleanup_attempts(clips) == 2
    assert bench_run.reason_counts(clips)["short_clean"] == 1


def test_the_report_splits_gate_reasons_from_guard_reasons():
    clips = (
        _clip(id="a", clean_reason="ok"),
        _clip(id="b", clean_reason="preamble"),
        _clip(id="c", clean_reason="short_clean"),
    )
    report = bench_run.render_candidate_report(_result(clips=clips), "2026-09-17")
    assert "## Cleanup gate (spec 8.1, cleanup never ran)" in report
    assert "## Cleanup guards (spec 8.3, the output was rejected)" in report
    assert "Guard rejection rate: 1 of 2 cleanup attempts (50%)." in report
    assert "| short_clean | 1 | 33% |" in report
    assert "| preamble | 1 | 50% |" in report


def test_gate_and_guard_reason_names_cover_every_reason_the_app_can_report():
    from spells import models

    documented = set(bench_run.GATE_REASONS) | set(bench_run.GUARD_REASONS) | {"ok"}
    assert "cpu_fallback" in documented and "language_switch" in documented
    assert documented <= set(models.CleanResult.__doc__.replace('"', " ").replace(",", " ").split())


# ------------------------------------------------------------------ manifest round trip


def test_manifest_round_trip(tmp_path):
    entries = [
        bench_prompts.ManifestEntry(
            id="de-long-3",
            language="de",
            category="long",
            reference="Der Termin mit Herrn Krasniqi steht am elften März.",
            text="Der Termin mit Herrn Krasniqi steht am elften März.",
            duration_s=14.25,
            recorded_at="2026-09-17T10:00:00",
            file="de-long-3.wav",
        ),
        bench_prompts.ManifestEntry(
            id="en-silence-1",
            language="en",
            category="silence",
            reference="",
            text=bench_prompts.SILENCE_TEXT,
            duration_s=3.0,
            recorded_at="2026-09-17T10:01:00",
            file="en-silence-1.wav",
        ),
    ]
    path = tmp_path / "clips" / "manifest.json"
    bench_prompts.write_manifest(path, entries)
    assert bench_prompts.read_manifest(path) == sorted(entries, key=lambda e: e.id)


def test_read_manifest_of_a_missing_file_is_empty(tmp_path):
    assert bench_prompts.read_manifest(tmp_path / "nope.json") == []


def test_write_manifest_keeps_non_ascii_readable(tmp_path):
    path = tmp_path / "manifest.json"
    bench_prompts.write_manifest(
        path,
        [
            bench_prompts.ManifestEntry(
                id="de-short-1",
                language="de",
                category="short",
                reference="Passt, bis später.",
                text="Passt, bis später.",
                duration_s=1.1,
                recorded_at="2026-09-17T10:00:00",
                file="de-short-1.wav",
            )
        ],
    )
    assert "später" in path.read_text(encoding="utf-8")


# ------------------------------------------------------------------ placeholders


def test_no_prompt_is_a_placeholder():
    for prompt in bench_prompts.PROMPTS:
        assert not bench_prompts.is_placeholder(prompt), prompt.id


def test_is_placeholder_detects_an_empty_text():
    empty = bench_prompts.Prompt(id="x", language="sq", category="short", text="  ", reference="")
    assert bench_prompts.is_placeholder(empty)


def test_is_placeholder_is_false_once_the_text_is_written():
    written = bench_prompts.Prompt(
        id="sq-short-1", language="sq", category="short", text="Mirë, shihemi.", reference="Mirë."
    )
    assert not bench_prompts.is_placeholder(written)


# ------------------------------------------------------------------ prompt set invariants


def test_prompt_ids_are_unique_and_file_safe():
    ids = [p.id for p in bench_prompts.PROMPTS]
    assert len(ids) == len(set(ids))
    assert bench_prompts.WARMUP_PROMPT.id not in ids
    for prompt_id in [*ids, bench_prompts.WARMUP_PROMPT.id]:
        assert prompt_id and all(ch.isalnum() or ch == "-" for ch in prompt_id), prompt_id


def test_prompt_set_is_about_thirty_prompts_across_three_languages():
    assert 27 <= len(bench_prompts.PROMPTS) <= 33
    languages = {p.language for p in bench_prompts.PROMPTS}
    assert languages == set(bench_prompts.LANGUAGES)


def test_every_category_is_present_in_every_language():
    for language in bench_prompts.LANGUAGES:
        found = {p.category for p in bench_prompts.PROMPTS if p.language == language}
        assert found == set(bench_prompts.CATEGORIES), language


def test_categories_are_known_and_ids_carry_language_and_category():
    for prompt in bench_prompts.PROMPTS:
        assert prompt.category in bench_prompts.CATEGORIES
        assert prompt.id.startswith(f"{prompt.language}-{prompt.category}-")


def test_references_are_non_empty_except_silence():
    for prompt in bench_prompts.PROMPTS:
        if prompt.category != "silence":
            assert prompt.reference.strip(), prompt.id


def test_silence_prompts_have_an_empty_reference_in_every_language():
    silence = [p for p in bench_prompts.PROMPTS if p.category == "silence"]
    assert len(silence) == len(bench_prompts.LANGUAGES)
    assert all(p.reference == "" for p in silence)


def test_filler_prompts_mark_fillers_and_drop_them_from_the_reference():
    filler = [p for p in bench_prompts.PROMPTS if p.category == "filler"]
    assert filler
    for prompt in filler:
        assert "[" in prompt.text and "]" in prompt.text, prompt.id
        assert len(prompt.reference.split()) < len(prompt.text.split()), prompt.id


def test_correction_prompts_name_a_self_correction_phrase():
    corrections = {
        "en-correction-1": "sorry I mean",
        "de-correction-1": "nein warte",
        "sq-correction-1": "jo prit",
    }
    for prompt_id, phrase in corrections.items():
        prompt = bench_prompts.by_id(prompt_id)
        assert prompt is not None and phrase.lower() in prompt.text.lower()
        assert phrase.lower() not in prompt.reference.lower()


def test_spoken_text_keeps_the_filler_words_and_drops_the_brackets():
    prompt = bench_prompts.by_id("en-filler-1")
    spoken = bench_prompts.spoken_text(prompt)
    assert "[" not in spoken and "]" not in spoken
    assert "um" in bench_run.normalise_words(spoken)


def test_spoken_text_of_a_silence_prompt_is_empty():
    assert bench_prompts.spoken_text(bench_prompts.by_id("en-silence-1")) == ""


def test_warmup_prompt_is_a_short_neutral_english_sentence():
    warmup = bench_prompts.WARMUP_PROMPT
    assert warmup.language == "en"
    assert warmup.reference == warmup.text
    assert 6 <= len(warmup.text.split()) <= 14


def test_by_id_returns_none_for_an_unknown_id():
    assert bench_prompts.by_id("nope") is None


def test_no_em_dashes_anywhere_in_the_prompt_set():
    for prompt in (*bench_prompts.PROMPTS, bench_prompts.WARMUP_PROMPT):
        assert EM_DASH not in prompt.text and EM_DASH not in prompt.reference, prompt.id


# ------------------------------------------------------------------ report rendering


def _fixed_results():
    """A small deterministic result set: one long English clip, one short German clip."""
    clips = [
        bench_run.ClipResult(
            id="en-long-1",
            language="en",
            category="long",
            duration_s=18.0,
            reference="hold the line",
            spoken="hold the line",
            raw_text="hold the lion",
            raw_ms=480.0,
            auto_text="hold the lion for me",
            auto_ms=505.0,
            auto_language="en",
            fallback_used=False,
            cleaned_text="Hold the line.",
            cleaned_ms=1420.0,
            clean_reason="ok",
            used_llm=True,
            skipped_ms=560.0,
            skipped_reason="disabled",
            error="",
        ),
        bench_run.ClipResult(
            id="de-short-1",
            language="de",
            category="short",
            duration_s=1.4,
            reference="Bis später.",
            spoken="Bis später.",
            raw_text="Bis später.",
            raw_ms=190.0,
            auto_text="Passt, bis später.",
            auto_ms=640.0,
            auto_language="de",
            fallback_used=True,
            cleaned_text="Bis später.",
            cleaned_ms=205.0,
            clean_reason="short_clean",
            used_llm=False,
            skipped_ms=198.0,
            skipped_reason="disabled",
            error="",
        ),
    ]
    return bench_run.CandidateResult(
        name="qwen3_4b_q4_k_m",
        model_file="Qwen3-4B-Q4_K_M.gguf",
        whisper_model="ggml-large-v3-turbo-q8_0.bin",
        gpu_name="NVIDIA GeForce RTX 5060 Laptop GPU",
        clips=tuple(clips),
        notes=("smoke run", ),
    )


GOLDEN_REPORT = """# Benchmark: qwen3_4b_q4_k_m

- Date: 2026-09-17
- Cleanup model: Qwen3-4B-Q4_K_M.gguf
- Whisper model: ggml-large-v3-turbo-q8_0.bin
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU
- Clips: 2
- Verdict against spec 12: pass
- Note: smoke run

## Latency targets (spec 12, long clips, median)

| Row | Target | Median | Result | Clips |
|---|---|---|---|---|
| Release to raw transcript | < 700 ms | 480 ms | pass | 1 |
| Release to cleaned text, cleanup skipped | < 800 ms | 560 ms | pass | 1 |
| Release to cleaned text, cleanup run | < 1800 ms | 1420 ms | pass | 1 |

## Latency by duration bucket

| Bucket | Clips | Raw p50 | Raw p95 | Skipped p50 | Skipped p95 | Cleaned p50 | Cleaned p95 |
|---|---|---|---|---|---|---|---|
| short | 1 | 190 ms | 190 ms | 198 ms | 198 ms | 205 ms | 205 ms |
| long | 1 | 480 ms | 480 ms | 560 ms | 560 ms | 1420 ms | 1420 ms |

## Latency by language and category

| Language | Category | Clips | Raw p50 | Raw p95 | Cleaned p50 | Cleaned p95 |
|---|---|---|---|---|---|---|
| en | long | 1 | 480 ms | 480 ms | 1420 ms | 1420 ms |
| de | short | 1 | 190 ms | 190 ms | 205 ms | 205 ms |

## Raw word error rate (micro average over the language's clips)

| Language | Scored clips | WER against the reference | WER against the spoken text |
|---|---|---|---|
| en | 1 | 33.3% | 33.3% |
| de | 1 | 0.0% | 0.0% |

## Auto mode language fallback (spec 7.2 step 3, long clips)

A recording under 2 s never reaches step 3, so only long clips are counted.

| Language | Long clips | Fallback | Rate | Auto raw p50 | Fallback raw p50 |
|---|---|---|---|---|---|
| en | 1 | 0 | 0% | 505 ms | n/a |

## Cleanup gate (spec 8.1, cleanup never ran)

| Reason | Clips | Share of clips |
|---|---|---|
| short_clean | 1 | 50% |

## Cleanup guards (spec 8.3, the output was rejected)

Guard rejection rate: 0 of 1 cleanup attempts (0%).

| Reason | Clips | Share of cleanup attempts |
|---|---|---|
| ok (accepted) | 1 | 100% |

## Side by side (manual judgement, spec 18 step 3)

| Clip | Language | Category | WER | Reference | Raw | Cleaned | Reason |
|---|---|---|---|---|---|---|---|
| en-long-1 | en | long | 33.3% | hold the line | hold the lion | Hold the line. | ok |
| de-short-1 | de | short | 0.0% | Bis später. | Bis später. | Bis später. | short_clean |
"""


def test_render_candidate_report_matches_the_golden_string():
    assert bench_run.render_candidate_report(_fixed_results(), "2026-09-17") == GOLDEN_REPORT


def test_render_summary_has_one_row_per_candidate():
    one = _fixed_results()
    two = bench_run.CandidateResult(
        name="qwen3_1_7b_q8_0",
        model_file="Qwen3-1.7B-Q8_0.gguf",
        whisper_model="ggml-large-v3-turbo-q8_0.bin",
        gpu_name="NVIDIA GeForce RTX 5060 Laptop GPU",
        clips=one.clips,
        notes=(),
    )
    summary = bench_run.render_summary([one, two], "2026-09-17")
    assert "| qwen3_4b_q4_k_m |" in summary
    assert "| qwen3_1_7b_q8_0 |" in summary
    assert summary.count("\n| ") >= 2
    assert "Selection (spec 18 step 4) is manual" in summary


# ------------------------------------------------------------------ fetch and select


def _pins_file(tmp_path, candidate_sha=""):
    pins = {
        "models": {
            "whisper_large_v3_turbo_q8_0": {
                "url": "https://example.invalid/ggml-large-v3-turbo-q8_0.bin",
                "version": "v1",
                "sha256": "",
                "notes": "test",
            },
            "silero_vad": {
                "url": "https://example.invalid/ggml-silero-v5.1.2.bin",
                "version": "v5.1.2",
                "sha256": "b" * 64,
                "notes": "test",
            },
        },
        "bench_candidates": [
            {
                "name": "cand_a",
                "url": "https://example.invalid/Cand-A-Q4_K_M.gguf",
                "version": "cand-a@main",
                "sha256": candidate_sha,
                "notes": "test",
            }
        ],
    }
    path = tmp_path / "pins.json"
    path.write_text(json.dumps(pins), encoding="utf-8")
    return path


def test_fetch_all_asks_build_fetch_for_the_candidates_and_both_models(tmp_path, monkeypatch):
    """--fetch reuses build/fetch.py rather than copying its verify-or-record logic."""
    calls = []

    def fake_fetch(key, *, pins_path, cache_dir, record):
        calls.append((key, record))
        return cache_dir / key

    monkeypatch.setattr(build_fetch, "fetch", fake_fetch)
    assert bench_run.fetch_all(["cand_a"], _pins_file(tmp_path), tmp_path / "cache") == 0
    assert calls == [
        ("bench_candidates.cand_a", True),
        ("models.whisper_large_v3_turbo_q8_0", True),
        ("models.silero_vad", True),
    ]


def test_fetch_all_reports_a_failure_without_stopping_the_others(tmp_path, monkeypatch):
    seen = []

    def fake_fetch(key, *, pins_path, cache_dir, record):
        seen.append(key)
        if key.startswith("bench_candidates."):
            raise build_fetch.UnpinnedHash("no hash")
        return cache_dir / key

    monkeypatch.setattr(build_fetch, "fetch", fake_fetch)
    assert bench_run.fetch_all(["cand_a"], _pins_file(tmp_path), tmp_path / "cache") == 1
    assert len(seen) == 3


def test_select_model_writes_the_chosen_candidate_into_pins(tmp_path):
    pins_path = _pins_file(tmp_path, candidate_sha="c" * 64)
    assert bench_run.select_model("cand_a", pins_path, tmp_path / "cache") == 0
    entry = build_fetch.load_pins(pins_path)["models"]["cleanup_model"]
    assert entry["url"] == "https://example.invalid/Cand-A-Q4_K_M.gguf"
    assert entry["version"] == "cand-a@main"
    assert entry["sha256"] == "c" * 64
    assert "spec 18 step 4" in entry["notes"]


def test_select_model_hashes_the_cached_file_when_the_pin_is_still_empty(tmp_path):
    pins_path = _pins_file(tmp_path)
    cached = tmp_path / "cache" / "models" / "Cand-A-Q4_K_M.gguf"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"gguf payload")
    assert bench_run.select_model("cand_a", pins_path, tmp_path / "cache") == 0
    entry = build_fetch.load_pins(pins_path)["models"]["cleanup_model"]
    assert entry["sha256"] == hashlib.sha256(b"gguf payload").hexdigest()


def test_select_model_refuses_an_unknown_candidate(tmp_path):
    assert bench_run.select_model("nope", _pins_file(tmp_path), tmp_path / "cache") == 1


def test_select_model_refuses_when_nothing_is_downloaded_and_no_hash_is_pinned(tmp_path):
    pins_path = _pins_file(tmp_path)
    assert bench_run.select_model("cand_a", pins_path, tmp_path / "cache") == 1
    assert "cleanup_model" not in build_fetch.load_pins(pins_path)["models"]


# ------------------------------------------------------------------ measure_clip with fakes


import smoke_clips

from .fake_clients import FakeLlamaAsrClient, FakeLlamaClient, FakeWhisperClient

LONG_LINE = "please move the standup to nine fifteen because half the team is on the train"


def _wav(path, seconds=15.0, sample_rate=16000):
    import wave

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * int(seconds * sample_rate))
    return path


def _entry(clip_id="en-long-1", language="en", category="long", reference=LONG_LINE):
    return bench_prompts.ManifestEntry(
        id=clip_id,
        language=language,
        category=category,
        reference=reference,
        text=reference,
        duration_s=15.0,
        recorded_at="2026-09-17T10:00:00",
        file=f"{clip_id}.wav",
    )


def _measure(tmp_path, whisper, llama, entry=None, seconds=15.0):
    entry = entry or _entry()
    _wav(tmp_path / entry.file, seconds)
    return bench_run.measure_clip(
        entry,
        tmp_path,
        whisper,
        llama,
        {"en": ["um", "uh"]},
        {"en": ["no wait"]},
        bench_run.asr.LanguagePolicyState(last_accepted="en"),
    )


def test_measure_clip_runs_auto_then_locked_then_cleanup_then_the_gate_off_pass(tmp_path):
    whisper = FakeWhisperClient(responses=[{"text": LONG_LINE, "language": "english"}])
    llama = FakeLlamaClient(content="Please move the standup to nine fifteen.")
    result = _measure(tmp_path, whisper, llama)

    assert result.error == ""
    assert result.raw_text == LONG_LINE
    assert result.auto_text == LONG_LINE
    assert result.cleaned_text == "Please move the standup to nine fifteen."
    assert result.clean_reason == "ok" and result.used_llm is True
    assert result.skipped_reason == "disabled"
    assert [call["language"] for call in whisper.calls] == ["auto", "en"]
    assert len(llama.calls) == 1, "the gate-off pass must not reach the engine"
    assert result.cleaned_ms >= result.raw_ms >= 0.0
    assert result.skipped_ms >= result.raw_ms


def test_measure_clip_records_the_auto_mode_fallback(tmp_path):
    whisper = FakeWhisperClient(
        responses=[
            {"text": LONG_LINE, "language": "french"},
            {"language_probabilities": {"de": 0.8, "en": 0.1}},
            {"text": LONG_LINE, "language": "german"},
            {"text": LONG_LINE, "language": "english"},
        ]
    )
    result = _measure(tmp_path, whisper, FakeLlamaClient(content=LONG_LINE))
    assert result.fallback_used is True
    assert result.auto_language == "de"
    assert len(whisper.calls) == 4


def test_measure_clip_reports_an_empty_transcript_as_no_transcript(tmp_path):
    whisper = FakeWhisperClient(responses=[{"text": "", "language": "english"}])
    llama = FakeLlamaClient(content="never called")
    result = _measure(tmp_path, whisper, llama)
    assert result.raw_text == ""
    assert result.clean_reason == "no_transcript" and result.skipped_reason == "no_transcript"
    assert llama.calls == []
    assert result.cleaned_ms == result.raw_ms


def test_measure_clip_carries_a_whisper_failure_into_the_error_field(tmp_path):
    whisper = FakeWhisperClient(responses=[bench_run.asr.WhisperError("engine gone")])
    result = _measure(tmp_path, whisper, FakeLlamaClient())
    assert "engine gone" in result.error
    assert result.raw_ms is None and result.clean_reason == "not_measured"


def test_measure_clip_reports_a_missing_clip_file(tmp_path):
    result = bench_run.measure_clip(
        _entry(),
        tmp_path,
        FakeWhisperClient(),
        FakeLlamaClient(),
        {},
        {},
        bench_run.asr.LanguagePolicyState(last_accepted="en"),
    )
    assert result.error and result.raw_ms is None


def test_measure_clip_takes_the_duration_from_the_audio_not_the_manifest(tmp_path):
    whisper = FakeWhisperClient(responses=[{"text": LONG_LINE, "language": "english"}])
    result = _measure(tmp_path, whisper, FakeLlamaClient(content=LONG_LINE), seconds=1.0)
    assert result.duration_s == pytest.approx(1.0)
    assert result.bucket == "short"


def test_measure_clip_keeps_the_raw_text_when_a_guard_rejects_the_output(tmp_path):
    whisper = FakeWhisperClient(responses=[{"text": LONG_LINE, "language": "english"}])
    llama = FakeLlamaClient(content="Sure, here is the cleaned text for you now.")
    result = _measure(tmp_path, whisper, llama)
    assert result.clean_reason == "preamble"
    assert result.cleaned_text == LONG_LINE and result.used_llm is False


# ------------------------------------------------------------------ SAPI quoting


def test_ps_literal_escapes_a_single_quote_and_leaves_a_dollar_sign_alone():
    assert smoke_clips._ps_literal("it's $env:PATH") == "'it''s $env:PATH'"


def test_ps_literal_round_trips_through_powershell_quoting_rules():
    values = (
        "plain",
        "with 'quotes'",
        r"C:\path\to\file.wav",
        '$x `tick` "dq"',
    )
    for value in values:
        literal = smoke_clips._ps_literal(value)
        assert literal.startswith("'") and literal.endswith("'")
        assert literal[1:-1].replace("''", "'") == value


def test_the_smoke_short_clip_is_short_enough_to_land_in_the_short_bucket():
    """SAPI reads slowly, so the short smoke line has to be very few words."""
    short = next(text for name, _, text in smoke_clips.SMOKE_TEXTS if name == "smoke-short-1")
    assert len(short.split()) <= 5


# ------------------------------------------------------------------ the app configuration (--hardware)


from spells.engines import SpeechSlot
from spells.modelcatalog import Hardware, select_models
from spells.models import Engine, EngineId

QWEN_ID = EngineId(Engine.WHISPER, 0)
FLUTRA_ID = EngineId(Engine.WHISPER, 1)


class FakeAppSupervisor:
    def __init__(self, *, cpu_only=True, llama_url="http://127.0.0.1:9300", variant="cpu",
                 cpu_cleanup_allowed=True, cleanup_languages=frozenset({"en", "de", "sq"})):
        self.cpu_only = cpu_only
        self.llama_url = llama_url
        self._variant = variant
        self.cpu_cleanup_allowed = cpu_cleanup_allowed
        self.cleanup_languages = cleanup_languages
        self.urls = {QWEN_ID: "http://127.0.0.1:9101", FLUTRA_ID: None}

    def speech_engines(self):
        return (
            SpeechSlot(QWEN_ID, "llama-asr", ("de", "en")),
            SpeechSlot(FLUTRA_ID, "whisper-server", ("sq",)),
        )

    def url(self, engine):
        return self.urls[engine]

    def variant(self, engine):
        return self._variant


def test_the_app_configuration_builds_the_engines_the_app_would_run(tmp_path):
    selection = select_models(["en", "de", "sq"], Hardware.CPU)
    models = tmp_path / "models"
    paths = bench_run.app_engine_paths(
        selection, models, tmp_path / "engines", models / "vad.bin", tmp_path / "logs"
    )
    assert paths.cpu_dir == tmp_path / "engines" / "cpu"
    assert paths.whisper_model == models / "Qwen3-ASR-0.6B-Q4_K_M.gguf"
    assert paths.whisper_runtime == "llama-asr"
    assert [spec.model.name for spec in paths.extra_speech] == [
        "ggml-large-v3-turbo-sq-flutra-v2-q8_0.bin"
    ]
    assert paths.llama_model == models / "gemma-4-E2B-it-Q4_0.gguf"
    assert paths.llama_extra_files == (models / "mtp-gemma-4-E2B-it-Q4_0.gguf",)
    assert paths.log_dir == tmp_path / "logs"


def test_the_app_configuration_needs_a_speech_model(tmp_path):
    selection = select_models(["en"], Hardware.CPU, frozenset())
    with pytest.raises(ValueError):
        bench_run.app_engine_paths(selection, tmp_path, tmp_path, tmp_path / "v", tmp_path)


def test_routes_and_clients_follow_the_supervisor():
    supervisor = FakeAppSupervisor()
    routes = bench_run.speech_routes(supervisor)
    assert [(route.engine, route.runtime, route.languages) for route in routes] == [
        (QWEN_ID, "llama-asr", ("de", "en")),
        (FLUTRA_ID, "whisper-server", ("sq",)),
    ]
    connect = bench_run.speech_connector(supervisor)
    qwen = connect(routes[0])
    assert isinstance(qwen, bench_run.asr.LlamaAsrClient)
    assert qwen.base_url == "http://127.0.0.1:9101"
    assert connect(routes[0]) is qwen
    assert connect(routes[1]) is None
    supervisor.urls[FLUTRA_ID] = "http://127.0.0.1:9102"
    assert isinstance(connect(routes[1]), bench_run.asr.WhisperClient)


def test_the_app_gate_carries_the_processor_rule_and_the_scored_languages():
    gate = bench_run.app_gate(FakeAppSupervisor())
    assert gate.cleanup_enabled and gate.profile_cleanup and gate.engine_available
    assert gate.cpu_fallback is False
    assert gate.cpu_selected is True
    assert gate.cleanup_languages == frozenset({"en", "de", "sq"})
    unplanned = bench_run.app_gate(
        FakeAppSupervisor(cpu_only=False, cpu_cleanup_allowed=False, llama_url=None)
    )
    assert unplanned.cpu_fallback is True and unplanned.engine_available is False


def _routed_measure(tmp_path, qwen, flutra, entry, gate):
    supervisor = FakeAppSupervisor()
    supervisor.urls[FLUTRA_ID] = "http://127.0.0.1:9102"
    clients = {QWEN_ID: qwen, FLUTRA_ID: flutra}
    routes = bench_run.speech_routes(supervisor)
    transcriber = bench_run.routed_transcriber(
        routes, lambda route: clients[route.engine], ["en", "de", "sq"]
    )
    _wav(tmp_path / entry.file, 15.0)
    return bench_run.measure_clip(
        entry,
        tmp_path,
        None,
        FakeLlamaClient(content="Never used."),
        {"en": ["um"], "sq": ["pra"]},
        {"en": ["no wait"]},
        bench_run.asr.LanguagePolicyState(last_accepted="en"),
        transcriber=transcriber,
        gate=gate,
        enabled_languages=["en", "de", "sq"],
    )


def test_a_clip_routed_like_the_pipeline_records_the_engine_that_served_it(tmp_path):
    qwen = FakeLlamaAsrClient(
        bench_run.asr.AsrReply("", "hu", "Hungarian"), reports="Hungarian"
    )
    flutra = FakeWhisperClient(responses=[{"text": "Mirëdita të gjithëve sot", "language": "sq"}])
    entry = _entry("sq-long-1", "sq", reference="Mirëdita të gjithëve sot")
    gate = bench_run.app_gate(FakeAppSupervisor())
    result = _routed_measure(tmp_path, qwen, flutra, entry, gate)
    assert result.error == ""
    assert (result.engine, result.auto_engine) == ("whisper-2", "whisper-2")
    assert result.auto_language == "sq"
    assert qwen.stopped == 1
    assert result.clean_reason == "clean_text" and result.used_llm is False
    assert result.cleaned_ms >= result.raw_ms


def test_an_english_clip_stays_on_the_fast_engine(tmp_path):
    qwen = FakeLlamaAsrClient(bench_run.asr.AsrReply(LONG_LINE, "en", "English"), reports="English")
    flutra = FakeWhisperClient(responses=[AssertionError("Whisper must not be asked")])
    gate = bench_run.app_gate(FakeAppSupervisor())
    result = _routed_measure(tmp_path, qwen, flutra, _entry(), gate)
    assert result.engine == "whisper" and result.auto_engine == "whisper"
    assert qwen.calls[1]["language"] == "en"
    assert flutra.calls == []


def test_an_unavailable_engine_is_an_error_for_that_clip(tmp_path):
    entry = _entry("sq-long-1", "sq", reference="Mirëdita")
    _wav(tmp_path / entry.file, 15.0)
    routes = bench_run.speech_routes(FakeAppSupervisor())
    transcriber = bench_run.routed_transcriber(routes, lambda route: None, ["en", "de", "sq"])
    result = bench_run.measure_clip(
        entry, tmp_path, None, FakeLlamaClient(), {}, {},
        bench_run.asr.LanguagePolicyState(last_accepted="sq"), transcriber=transcriber,
    )
    assert "not serving" in result.error


def _app_result(clips, languages=("en", "sq")):
    return bench_run.AppResult(
        hardware="cpu",
        languages=languages,
        speech=(
            ("whisper", "qwen3-asr-0.6b-q8_0", "llama-asr", ("de", "en")),
            ("whisper-2", "whisper-large-v3-turbo-sq-flutra-v2-q8_0", "whisper-server", ("sq",)),
        ),
        cleanup_model="gemma-4-e2b-it-q4_0",
        gpu_name="",
        cpu_mask="5555",
        clips=tuple(clips),
        notes=("a note",),
    )


def test_language_rows_split_cleaned_clips_from_raw_ones_on_long_clips_only():
    clips = [
        _clip(id="en-1", raw_ms=600.0, cleaned_ms=1400.0, used_llm=True, engine="whisper"),
        _clip(id="en-2", raw_ms=800.0, cleaned_ms=800.5, used_llm=False, clean_reason="clean_text",
              engine="whisper"),
        _clip(id="en-3", raw_ms=100.0, cleaned_ms=100.0, used_llm=False, duration_s=1.0,
              engine="whisper"),
        _clip(id="sq-1", language="sq", raw_ms=5800.0, cleaned_ms=5800.2, used_llm=False,
              clean_reason="clean_text", engine="whisper-2"),
    ]
    en, sq, de = bench_run.app_language_rows(clips, ["en", "sq", "de"])
    assert (en.clips, en.engines, en.raw_ms) == (2, ("whisper",), pytest.approx(700.0))
    assert en.raw_passed is False
    assert (en.cleaned_run, en.cleaned_run_ms, en.cleaned_run_passed) == (1, 1400.0, True)
    assert en.cleaned_skipped_ms == pytest.approx(800.5) and en.cleaned_skipped_passed is False
    assert sq.engines == ("whisper-2",) and sq.cleaned_run == 0 and sq.cleaned_run_ms is None
    assert sq.cleaned_run_passed is None
    assert de.clips == 0 and de.raw_ms is None and de.raw_passed is None


def test_the_app_report_names_engines_medians_and_the_serving_engine_per_clip():
    clips = [
        _clip(id="en-long-1", raw_ms=650.0, cleaned_ms=651.0, used_llm=False,
              clean_reason="clean_text", engine="whisper", auto_engine="whisper"),
        _clip(id="sq-long-1", language="sq", raw_ms=5900.0, cleaned_ms=5901.0, used_llm=False,
              clean_reason="clean_text", engine="whisper-2", auto_engine="whisper-2",
              auto_language="sq"),
    ]
    report = bench_run.render_app_report(_app_result(clips), "2026-09-17")
    assert report.startswith("# App configuration benchmark: cpu")
    assert "- Speech engine whisper-2: whisper-large-v3-turbo-sq-flutra-v2-q8_0 (whisper-server)" in report
    assert "- Performance core mask: 5555" in report
    assert "| en | 1 | whisper | 650 ms (pass) | 0 | n/a | 651 ms (pass) |" in report
    assert "| sq | 1 | whisper-2 | 5900 ms (FAIL) | 0 | n/a | 5901 ms (FAIL) |" in report
    assert "| sq-long-1 | sq | long | whisper-2 | whisper-2 | sq |" in report
    assert "| clean_text | 2 |" in report
    assert EM_DASH not in report


def test_hardware_mode_cannot_be_combined_with_the_smoke_run():
    with pytest.raises(SystemExit):
        bench_run.main(["--hardware", "cpu", "--smoke"])

