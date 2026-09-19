"""Model selection benchmark: fetch, measure, report, select (spec 18, targets from spec 12).

The benchmark shares the ASR request, the language policy, the cleanup request, the guards, the
fixed prompt and the GPU device selection with the app (spells.asr, spells.cleanup,
spells.cleanup_prompt, spells.gpu, spells.engines), so both are measured on identical logic.

Usage:
    py bench/run.py --fetch [--candidates a,b]
    py bench/run.py [--candidates a,b] [--clips id,id]
    py bench/run.py --hardware cpu|gpu [--languages en,de,sq] [--clips id,id]
    py bench/run.py --smoke
    py bench/run.py --select <candidate>

--hardware runs the app's own configuration: the model selection from the catalog for the
enabled languages, every chosen engine through the supervisor (the CPU builds pinned to the
performance cores for cpu), the pipeline's speech routing and the cleanup gate.

The clips run in id order and Auto mode carries the last accepted language between them, as
the app does between dictations (spec 7.2), so a run is reproducible rather than dependent on
the manifest's file order.

Stdlib plus the spells package only. Nothing here writes audio to disk and no report carries a
clip file, only the transcripts the benchmark itself produced.
"""

from __future__ import annotations

import argparse
import datetime
import sys
import time
import wave
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_DIR = Path(__file__).resolve().parents[1]
BENCH_DIR = REPO_DIR / "bench"
for _extra in (REPO_DIR / "src", BENCH_DIR, REPO_DIR / "build"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import prompts as prompt_set

from spells import asr, cleanup, gpu
from spells.config import DEFAULT_ENABLED_LANGUAGES
from spells.engines import EnginePaths, EngineSupervisor
from spells.modelcatalog import (
    Hardware,
    Selection,
    installed_ids,
    load_catalog,
    select_models,
)
from spells.models import CpuPlan, Engine, EngineState, LangMode
from spells.profiles import DEFAULT_PROFILE
from spells.textutil import normalize_for_match

# Spec 12, long category (10 to 30 s), medians. (report row, ClipResult field, target in ms)
LONG_TARGETS = (
    ("Release to raw transcript", "raw_ms", 700.0),
    ("Release to cleaned text, cleanup skipped", "skipped_ms", 800.0),
    ("Release to cleaned text, cleanup run", "cleaned_ms", 1800.0),
)
# The two cleaned rows are only meaningful while llama serves from its Vulkan build: spec 8.1
# skips cleanup entirely on the CPU build, so a CPU measurement describes a state the app never
# reaches. --allow-cpu-cleanup measures them anyway and marks them as such.
CPU_CLEANUP_INVALID = "invalid (llama on CPU)"
CPU_CLEANUP_MEASURED = "measured on CPU"
# Spec 8.1 gate reasons (cleanup never ran) and spec 8.3 guard reasons (the output was rejected).
GATE_REASONS = (
    "disabled",
    "profile_off",
    "language_unscored",
    "short_clean",
    "clean_text",
    "engine_not_ready",
    "cpu_fallback",
)
APP_RAW_TARGET_MS = 700.0
APP_SKIPPED_TARGET_MS = 800.0
APP_CLEANED_TARGET_MS = 1800.0
GUARD_REASONS = (
    "timeout",
    "error",
    "empty",
    "length_ratio",
    "finish_length",
    "preamble",
    "language_switch",
)
# Spec 7.2: the welcome page recommends per-language chords above this fallback rate.
FALLBACK_RATE_NOTE = 0.10
WHISPER_MODEL_PIN = "models.whisper_large_v3_turbo_q8_0"
VAD_MODEL_PIN = "models.silero_vad"
# Spec 19.1 pins two test-only models that never ship; --smoke runs on exactly those.
SMOKE_WHISPER_PIN = "models.whisper_tiny_test"
SMOKE_CANDIDATE_PIN = "models.qwen2_5_0_5b_instruct_q4_k_m_test"


# ----------------------------------------------------------------------------- measurements


@dataclass(frozen=True)
class ClipResult:
    """One clip measured against one cleanup candidate.

    raw_* is the locked-mode pass (one whisper request, the language the prompt was recorded
    in). auto_* is the Auto-mode pass of spec 7.2, which may send the audio three times when
    the detected language falls outside the enabled set; spec 12 reports that path separately,
    so the target rows use the locked pass.
    """

    id: str
    language: str
    category: str
    duration_s: float
    reference: str
    spoken: str
    raw_text: str
    raw_ms: float | None
    auto_text: str
    auto_ms: float | None
    auto_language: str
    fallback_used: bool
    cleaned_text: str
    cleaned_ms: float | None
    clean_reason: str
    used_llm: bool
    skipped_ms: float | None
    skipped_reason: str
    error: str = ""
    engine: str = ""
    auto_engine: str = ""

    @property
    def bucket(self) -> str:
        return prompt_set.duration_bucket(self.duration_s)

    @property
    def wer(self) -> float | None:
        """Raw word error rate against the intended clean text (spec 18 step 3)."""
        return wer(self.reference, self.raw_text)

    @property
    def wer_spoken(self) -> float | None:
        """Raw word error rate against what was actually said, fillers included.

        Filler and self-correction prompts are read with words their reference deliberately
        drops, so this second rate separates recognition errors from the cleanup work.
        """
        return wer(self.spoken, self.raw_text)


@dataclass(frozen=True)
class CandidateResult:
    name: str
    model_file: str
    whisper_model: str
    gpu_name: str
    clips: tuple[ClipResult, ...]
    notes: tuple[str, ...] = ()
    llama_variant: str = "vulkan"
    allow_cpu_cleanup: bool = False


@dataclass(frozen=True)
class TargetRow:
    name: str
    target_ms: float
    measured_ms: float | None
    passed: bool | None
    samples: int
    note: str = ""


# ----------------------------------------------------------------------------- small statistics


def normalise_words(text: str) -> list[str]:
    """Lowercased words with punctuation stripped, the form both sides of a WER are compared in.

    spells.textutil.normalize_for_match is the same normalisation the gate, the guards and the
    hallucination filter use, so the benchmark scores text the way the app matches it.
    """
    return normalize_for_match(text).split()


def levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    """Edit distance in whole words: substitutions, insertions and deletions cost one each."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, item in enumerate(a, start=1):
        current = [i]
        for j, other in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (0 if item == other else 1),
                )
            )
        previous = current
    return previous[-1]


def wer(reference: str, hypothesis: str) -> float | None:
    """Word error rate of hypothesis against reference, or None when the reference has no words.

    A silence prompt has no reference words, so its rate is undefined; those clips are judged
    by whether anything was transcribed at all.
    """
    ref = normalise_words(reference)
    if not ref:
        return None
    return levenshtein(ref, normalise_words(hypothesis)) / len(ref)


def percentile(values: Sequence[float], p: float) -> float | None:
    """Linear interpolation between the closest ranks; None for no samples."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (p / 100.0)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def target_rows(
    clips: Sequence[ClipResult],
    *,
    llama_variant: str = "vulkan",
    allow_cpu_cleanup: bool = False,
) -> list[TargetRow]:
    """The spec 12 pass or fail table: medians over the long category only.

    While llama serves from its CPU build the two cleaned rows describe a state the app never
    reaches, so they are left undecided and marked invalid, unless allow_cpu_cleanup says to
    measure them anyway.
    """
    long_clips = [clip for clip in clips if clip.bucket == "long" and not clip.error]
    cpu = llama_variant == "cpu"
    rows = []
    for index, (name, field_name, target) in enumerate(LONG_TARGETS):
        values = [
            getattr(clip, field_name)
            for clip in long_clips
            if getattr(clip, field_name) is not None
        ]
        measured = percentile(values, 50)
        passed = None if measured is None else measured < target
        note = ""
        if cpu and index > 0:
            note = CPU_CLEANUP_MEASURED if allow_cpu_cleanup else CPU_CLEANUP_INVALID
            if not allow_cpu_cleanup:
                measured, passed = None, None
        rows.append(
            TargetRow(
                name=name,
                target_ms=target,
                measured_ms=measured,
                passed=passed,
                samples=len(values),
                note=note,
            )
        )
    return rows


def candidate_passes(rows: Sequence[TargetRow]) -> bool | None:
    """True when every decided row passes, None when no long clip decided anything."""
    decided = [row for row in rows if row.passed is not None]
    if not decided:
        return None
    return all(row.passed for row in decided)


def rows_for(result: CandidateResult) -> list[TargetRow]:
    """The target rows of one candidate, with its llama variant applied."""
    return target_rows(
        result.clips,
        llama_variant=result.llama_variant,
        allow_cpu_cleanup=result.allow_cpu_cleanup,
    )


def candidate_verdict(rows: Sequence[TargetRow]) -> str:
    """The one word verdict for the summary, with the CPU cleanup state folded in."""
    notes = {row.note for row in rows if row.note}
    if CPU_CLEANUP_INVALID in notes:
        return CPU_CLEANUP_INVALID
    verdict = _verdict(candidate_passes(rows))
    if CPU_CLEANUP_MEASURED in notes:
        return f"{verdict} ({CPU_CLEANUP_MEASURED})"
    return verdict


def cleanup_is_invalid(result: CandidateResult) -> bool:
    """True when the cleaned rows could not be judged because llama served from its CPU build."""
    return result.llama_variant == "cpu" and not result.allow_cpu_cleanup


# ----------------------------------------------------------------------------- report rendering


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f} ms"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _cell(text: str) -> str:
    """One table cell: newlines folded, pipes escaped, empty text made visible."""
    folded = " ".join(text.split())
    return folded.replace("|", "\\|") if folded else "(empty)"


def _verdict(passed: bool | None) -> str:
    return "undecided" if passed is None else ("pass" if passed else "FAIL")


def _language_wer(
    clips: Sequence[ClipResult], language: str, spoken: bool = False
) -> float | None:
    """Micro average: total edits over total reference words across that language's clips.

    Averaging the per-clip rates would give a three word clip the same weight as a sixty word
    one. The per-clip rate stays in the side by side table.
    """
    edits = 0
    reference_words = 0
    for clip in clips:
        if clip.language != language or clip.error:
            continue
        reference = normalise_words(clip.spoken if spoken else clip.reference)
        if not reference:
            continue
        edits += levenshtein(reference, normalise_words(clip.raw_text))
        reference_words += len(reference)
    return edits / reference_words if reference_words else None


def _fallback_rate(clips: Sequence[ClipResult], language: str) -> tuple[int, int]:
    """Spec 7.2 step 3 over the long clips only, the rate spec 7.2 puts its 10% line on.

    A recording under 2 s never reaches step 3 (step 1 sends the last accepted language), so
    counting short clips would only dilute the rate.
    """
    scored = [
        clip
        for clip in clips
        if clip.language == language and not clip.error and clip.bucket == "long"
    ]
    return sum(1 for clip in scored if clip.fallback_used), len(scored)


def reason_counts(clips: Sequence[ClipResult]) -> dict[str, int]:
    """How many clips ended on each CleanResult.reason."""
    counted: dict[str, int] = {}
    for clip in clips:
        counted[clip.clean_reason] = counted.get(clip.clean_reason, 0) + 1
    return counted


def cleanup_attempts(clips: Sequence[ClipResult]) -> int:
    """Clips where the gate let the request through, so a guard could reject the output."""
    counted = reason_counts(clips)
    return counted.get("ok", 0) + sum(counted.get(reason, 0) for reason in GUARD_REASONS)


def render_candidate_report(result: CandidateResult, date: str) -> str:
    """The per-candidate markdown report of spec 18 step 3."""
    clips = list(result.clips)
    rows = rows_for(result)
    out: list[str] = []
    out.append(f"# Benchmark: {result.name}")
    out.append("")
    out.append(f"- Date: {date}")
    out.append(f"- Cleanup model: {result.model_file}")
    out.append(f"- Whisper model: {result.whisper_model}")
    out.append(f"- GPU: {result.gpu_name or 'none'}")
    out.append(f"- Clips: {len(clips)}")
    out.append(f"- Verdict against spec 12: {candidate_verdict(rows)}")
    if result.llama_variant == "cpu":
        out.append(
            "- Note: llama served from its CPU build. Spec 8.1 skips cleanup entirely there, so "
            + (
                "the two cleaned rows below are measured but not comparable."
                if result.allow_cpu_cleanup
                else "the two cleaned rows below are not valid; rerun on the Vulkan build."
            )
        )
    for note in result.notes:
        out.append(f"- Note: {note}")
    out.append("")

    out.append("## Latency targets (spec 12, long clips, median)")
    out.append("")
    out.append("| Row | Target | Median | Result | Clips |")
    out.append("|---|---|---|---|---|")
    for row in rows:
        verdict = f"{_verdict(row.passed)} ({row.note})" if row.note else _verdict(row.passed)
        out.append(
            f"| {row.name} | < {row.target_ms:.0f} ms | {_ms(row.measured_ms)} | "
            f"{verdict} | {row.samples} |"
        )
    out.append("")

    out.append("## Latency by duration bucket")
    out.append("")
    out.append(
        "| Bucket | Clips | Raw p50 | Raw p95 | Skipped p50 | Skipped p95 | "
        "Cleaned p50 | Cleaned p95 |"
    )
    out.append("|---|---|---|---|---|---|---|---|")
    for bucket in ("short", "long", "other"):
        group = [clip for clip in clips if clip.bucket == bucket and not clip.error]
        if not group:
            continue
        out.append(f"| {bucket} | {len(group)} | " + " | ".join(_field_percentiles(group)) + " |")
    out.append("")

    out.append("## Latency by language and category")
    out.append("")
    out.append("| Language | Category | Clips | Raw p50 | Raw p95 | Cleaned p50 | Cleaned p95 |")
    out.append("|---|---|---|---|---|---|---|")
    for language in prompt_set.LANGUAGES:
        for category in prompt_set.CATEGORIES:
            group = [
                clip
                for clip in clips
                if clip.language == language and clip.category == category and not clip.error
            ]
            if not group:
                continue
            raw = [clip.raw_ms for clip in group if clip.raw_ms is not None]
            cleaned = [clip.cleaned_ms for clip in group if clip.cleaned_ms is not None]
            out.append(
                f"| {language} | {category} | {len(group)} | {_ms(percentile(raw, 50))} | "
                f"{_ms(percentile(raw, 95))} | {_ms(percentile(cleaned, 50))} | "
                f"{_ms(percentile(cleaned, 95))} |"
            )
    out.append("")

    out.append("## Raw word error rate (micro average over the language's clips)")
    out.append("")
    out.append(
        "| Language | Scored clips | WER against the reference | WER against the spoken text |"
    )
    out.append("|---|---|---|---|")
    for language in prompt_set.LANGUAGES:
        scored = [
            clip
            for clip in clips
            if clip.language == language and not clip.error and clip.wer is not None
        ]
        if not scored:
            continue
        out.append(
            f"| {language} | {len(scored)} | {_pct(_language_wer(clips, language))} | "
            f"{_pct(_language_wer(clips, language, spoken=True))} |"
        )
    out.append("")

    out.append("## Auto mode language fallback (spec 7.2 step 3, long clips)")
    out.append("")
    out.append("A recording under 2 s never reaches step 3, so only long clips are counted.")
    out.append("")
    out.append("| Language | Long clips | Fallback | Rate | Auto raw p50 | Fallback raw p50 |")
    out.append("|---|---|---|---|---|---|")
    for language in prompt_set.LANGUAGES:
        used, total = _fallback_rate(clips, language)
        if not total:
            continue
        group = [
            clip
            for clip in clips
            if clip.language == language and not clip.error and clip.bucket == "long"
        ]
        auto = [clip.auto_ms for clip in group if clip.auto_ms is not None]
        fell = [clip.auto_ms for clip in group if clip.fallback_used and clip.auto_ms is not None]
        out.append(
            f"| {language} | {total} | {used} | {used / total * 100:.0f}% | "
            f"{_ms(percentile(auto, 50))} | {_ms(percentile(fell, 50))} |"
        )
    out.append("")

    counted = reason_counts(clips)
    attempts = cleanup_attempts(clips)
    out.append("## Cleanup gate (spec 8.1, cleanup never ran)")
    out.append("")
    out.append("| Reason | Clips | Share of clips |")
    out.append("|---|---|---|")
    for reason in GATE_REASONS:
        if counted.get(reason):
            share = counted[reason] / len(clips) if clips else 0.0
            out.append(f"| {reason} | {counted[reason]} | {share * 100:.0f}% |")
    if not any(counted.get(reason) for reason in GATE_REASONS):
        out.append("| none | 0 | 0% |")
    out.append("")

    rejected = sum(counted.get(reason, 0) for reason in GUARD_REASONS)
    out.append("## Cleanup guards (spec 8.3, the output was rejected)")
    out.append("")
    share = rejected / attempts if attempts else 0.0
    out.append(
        f"Guard rejection rate: {rejected} of {attempts} cleanup attempts ({share * 100:.0f}%)."
    )
    out.append("")
    out.append("| Reason | Clips | Share of cleanup attempts |")
    out.append("|---|---|---|")
    accepted = counted.get("ok", 0)
    accepted_share = accepted / attempts if attempts else 0.0
    out.append(f"| ok (accepted) | {accepted} | {accepted_share * 100:.0f}% |")
    for reason in GUARD_REASONS:
        if counted.get(reason):
            reason_share = counted[reason] / attempts if attempts else 0.0
            out.append(f"| {reason} | {counted[reason]} | {reason_share * 100:.0f}% |")
    out.append("")
    other = {
        reason: count
        for reason, count in counted.items()
        if reason not in GATE_REASONS and reason not in GUARD_REASONS and reason != "ok"
    }
    if other:
        listed = ", ".join(f"{reason} {count}" for reason, count in sorted(other.items()))
        out.append(f"Clips that never reached the gate: {listed}.")
        out.append("")

    silence = [clip for clip in clips if clip.category == "silence" and not clip.error]
    if silence:
        spoke = sum(1 for clip in silence if normalise_words(clip.raw_text))
        out.append(
            f"Silence clips: {len(silence)}, of which {spoke} produced text "
            f"that the hallucination filter (spec 7.4) let through."
        )
        out.append("")

    out.append("## Side by side (manual judgement, spec 18 step 3)")
    out.append("")
    out.append("| Clip | Language | Category | WER | Reference | Raw | Cleaned | Reason |")
    out.append("|---|---|---|---|---|---|---|---|")
    for clip in clips:
        out.append(
            f"| {clip.id} | {clip.language} | {clip.category} | {_pct(clip.wer)} | "
            f"{_cell(clip.reference)} | {_cell(clip.raw_text)} | {_cell(clip.cleaned_text)} | "
            f"{clip.clean_reason} |"
        )
    failed = [clip for clip in clips if clip.error]
    if failed:
        out.append("")
        out.append("## Clips that could not be measured")
        out.append("")
        for clip in failed:
            out.append(f"- {clip.id}: {_cell(clip.error)}")
    out.append("")
    return "\n".join(out)


def _field_percentiles(group: Sequence[ClipResult]) -> list[str]:
    cells = []
    for field_name in ("raw_ms", "skipped_ms", "cleaned_ms"):
        values = [
            getattr(clip, field_name) for clip in group if getattr(clip, field_name) is not None
        ]
        cells.append(_ms(percentile(values, 50)))
        cells.append(_ms(percentile(values, 95)))
    # Column order is raw p50, raw p95, skipped p50, skipped p95, cleaned p50, cleaned p95.
    return cells


def render_summary(results: Sequence[CandidateResult], date: str) -> str:
    """One row per candidate plus the rows spec 18 step 4 wants judged by hand."""
    out: list[str] = []
    out.append(f"# Benchmark summary {date}")
    out.append("")
    out.append(f"Candidates: {len(results)}")
    out.append("")
    out.append(
        "| Candidate | Clips | Raw p50 | Skipped p50 | Cleaned p50 | Spec 12 | "
        "WER en | WER de | WER sq | Fallback sq |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for result in results:
        rows = rows_for(result)
        used, total = _fallback_rate(result.clips, "sq")
        rate = "n/a" if not total else f"{used / total * 100:.0f}%"
        out.append(
            f"| {result.name} | {len(result.clips)} | {_ms(rows[0].measured_ms)} | "
            f"{_ms(rows[1].measured_ms)} | {_ms(rows[2].measured_ms)} | "
            f"{candidate_verdict(rows)} | "
            f"{_pct(_language_wer(result.clips, 'en'))} | "
            f"{_pct(_language_wer(result.clips, 'de'))} | "
            f"{_pct(_language_wer(result.clips, 'sq'))} | {rate} |"
        )
    out.append("")
    passing = [
        r.name for r in results if not cleanup_is_invalid(r) and candidate_passes(rows_for(r))
    ]
    out.append(f"Candidates meeting the spec 12 targets: {', '.join(passing) or 'none'}")
    out.append("")
    invalid = [r.name for r in results if cleanup_is_invalid(r)]
    if invalid:
        out.append(
            f"Cleanup rows not judged for {', '.join(invalid)}: llama served from its CPU build, "
            "where spec 8.1 skips cleanup. Rerun on the Vulkan build, or pass "
            "`--allow-cpu-cleanup` to measure them anyway."
        )
        out.append("")
    out.append(
        "Selection (spec 18 step 4) is manual: among the candidates above, pick the one judged "
        "best on Albanian and German in the rows below, ties broken by lower latency, then run "
        "`py bench/run.py --select <candidate>` to pin it."
    )
    out.append("")
    for language, title in (("sq", "Albanian"), ("de", "German")):
        out.append(f"## {title} rows for judgement")
        out.append("")
        out.append("| Candidate | Clip | Category | WER | Reference | Raw | Cleaned | Reason |")
        out.append("|---|---|---|---|---|---|---|---|")
        for result in results:
            for clip in result.clips:
                if clip.language != language:
                    continue
                out.append(
                    f"| {result.name} | {clip.id} | {clip.category} | {_pct(clip.wer)} | "
                    f"{_cell(clip.reference)} | {_cell(clip.raw_text)} | "
                    f"{_cell(clip.cleaned_text)} | {clip.clean_reason} |"
                )
        out.append("")
    if any(_fallback_rate(r.clips, "sq")[1] for r in results):
        worst = max(
            (_fallback_rate(r.clips, "sq")[0] / max(1, _fallback_rate(r.clips, "sq")[1]))
            for r in results
        )
        if worst > FALLBACK_RATE_NOTE:
            out.append(
                f"The Albanian Auto-mode fallback rate reaches {worst * 100:.0f}%, above the 10% "
                "line of spec 7.2, so the welcome page should recommend per-language chords for "
                "Albanian."
            )
            out.append("")
    return "\n".join(out)


# ----------------------------------------------------------------------------- pins and fetching


def _load_fetch():
    """build/fetch.py, imported rather than copied so both use one verify-or-record path."""
    import fetch

    return fetch


def candidate_names(pins: dict) -> list[str]:
    return [entry["name"] for entry in pins.get("bench_candidates", [])]


def fetch_all(names: Sequence[str], pins_path: Path, cache_dir: Path) -> int:
    """Download every candidate plus the whisper and VAD models through build/fetch.py.

    record=True lets an entry whose pinned sha256 is still empty be downloaded; fetch.py then
    writes the hash back into pins.json and prints its own warning.
    """
    fetch = _load_fetch()
    keys = [f"bench_candidates.{name}" for name in names] + [WHISPER_MODEL_PIN, VAD_MODEL_PIN]
    failures = []
    for key in keys:
        try:
            fetch.fetch(key, pins_path=pins_path, cache_dir=cache_dir, record=True)
        except (fetch.PinError, OSError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            failures.append(key)
    if failures:
        print(f"{len(failures)} of {len(keys)} downloads failed", file=sys.stderr)
        return 1
    return 0


def select_model(name: str, pins_path: Path, cache_dir: Path) -> int:
    """Write the chosen candidate's url and sha256 into pins.json models.cleanup_model (18.4)."""
    fetch = _load_fetch()
    pins = fetch.load_pins(pins_path)
    entry = next(
        (item for item in pins.get("bench_candidates", []) if item.get("name") == name), None
    )
    if entry is None:
        print(f"ERROR: no benchmark candidate named {name!r}", file=sys.stderr)
        return 1
    sha256 = (entry.get("sha256") or "").strip().lower()
    if not sha256:
        cached = cache_dir / "models" / fetch.filename_for(entry)
        if not cached.is_file():
            print(
                f"ERROR: {name} has no pinned sha256 and {cached} is not downloaded. "
                "Run --fetch first.",
                file=sys.stderr,
            )
            return 1
        sha256 = fetch.sha256_file(cached)
    pins.setdefault("models", {})["cleanup_model"] = {
        "url": entry["url"],
        "version": entry["version"],
        "sha256": sha256,
        "notes": (
            f"Cleanup model chosen by the benchmark (spec 18 step 4) from candidate {name!r}. "
            "Written by bench/run.py --select; the report in bench/reports/ records why."
        ),
    }
    fetch.save_pins(pins, pins_path)
    print(f"models.cleanup_model pinned to {name} ({sha256})")
    return 0


# ----------------------------------------------------------------------------- running a candidate


def read_wav(path: Path) -> tuple[bytes, int]:
    """Raw 16-bit mono PCM and its sample rate. Raises ValueError on any other format."""
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError(f"{path.name}: expected 16-bit mono, got {handle.getparams()}")
        return handle.readframes(handle.getnframes()), handle.getframerate()


Transcriber = Callable[[bytes, int, LangMode, asr.LanguagePolicyState], Any]


def measure_clip(
    entry: prompt_set.ManifestEntry,
    clips_dir: Path,
    whisper_client,
    llama_client,
    fillers: dict,
    corrections: dict,
    auto_state: asr.LanguagePolicyState,
    *,
    transcriber: Transcriber | None = None,
    gate: cleanup.GateInput | None = None,
    enabled_languages: Sequence[str] = DEFAULT_ENABLED_LANGUAGES,
) -> ClipResult:
    """One clip through the shared app logic: Auto pass, locked pass, cleanup, cleanup skipped."""
    enabled = list(enabled_languages)

    def whisper_only(pcm, sample_rate, mode, state):
        return asr.transcribe(pcm, sample_rate, mode, [], enabled, state, whisper_client)

    transcribe = transcriber or whisper_only
    prompt = prompt_set.by_id(entry.id)
    spoken = prompt_set.spoken_text(prompt) if prompt is not None else entry.reference
    blank = {
        "id": entry.id,
        "language": entry.language,
        "category": entry.category,
        "duration_s": entry.duration_s,
        "reference": entry.reference,
        "spoken": spoken,
        "raw_text": "",
        "raw_ms": None,
        "auto_text": "",
        "auto_ms": None,
        "auto_language": "",
        "fallback_used": False,
        "cleaned_text": "",
        "cleaned_ms": None,
        "clean_reason": "not_measured",
        "used_llm": False,
        "skipped_ms": None,
        "skipped_reason": "not_measured",
    }
    try:
        pcm, sample_rate = read_wav(clips_dir / entry.file)
    except (OSError, ValueError, wave.Error) as exc:
        return ClipResult(**blank, error=str(exc))
    duration_s = len(pcm) / (2 * sample_rate)

    try:
        started = time.perf_counter()
        auto = transcribe(pcm, sample_rate, LangMode("auto"), auto_state)
        auto_ms = (time.perf_counter() - started) * 1000.0

        locked_state = asr.LanguagePolicyState(last_accepted=entry.language)
        started = time.perf_counter()
        locked = transcribe(pcm, sample_rate, LangMode("locked", entry.language), locked_state)
        raw_ms = (time.perf_counter() - started) * 1000.0
    except (asr.WhisperError, asr.SpeechEngineUnavailable) as exc:
        return ClipResult(**{**blank, "duration_s": duration_s}, error=str(exc))

    if locked is None:
        # Nothing usable: an empty result or a known silence hallucination (spec 7.4).
        return ClipResult(
            **{
                **blank,
                "duration_s": duration_s,
                "raw_ms": raw_ms,
                "auto_text": auto.text if auto else "",
                "auto_ms": auto_ms,
                "auto_language": auto.language if auto else "",
                "fallback_used": bool(auto and auto.fallback_used),
                "cleaned_ms": raw_ms,
                "clean_reason": "no_transcript",
                "skipped_ms": raw_ms,
                "skipped_reason": "no_transcript",
                "auto_engine": auto.engine if auto else "",
            }
        )

    on_gate = gate or cleanup.GateInput(
        cleanup_enabled=True, profile_cleanup=True, engine_available=True, cpu_fallback=False
    )
    started = time.perf_counter()
    cleaned = cleanup.clean(
        locked,
        DEFAULT_PROFILE,
        [],
        fillers,
        corrections,
        enabled,
        on_gate,
        llama_client,
    )
    cleaned_ms = raw_ms + (time.perf_counter() - started) * 1000.0

    off_gate = cleanup.GateInput(
        cleanup_enabled=False, profile_cleanup=True, engine_available=True, cpu_fallback=False
    )
    started = time.perf_counter()
    skipped = cleanup.clean(
        locked,
        DEFAULT_PROFILE,
        [],
        fillers,
        corrections,
        enabled,
        off_gate,
        llama_client,
    )
    skipped_ms = raw_ms + (time.perf_counter() - started) * 1000.0

    return ClipResult(
        id=entry.id,
        language=entry.language,
        category=entry.category,
        duration_s=duration_s,
        reference=entry.reference,
        spoken=spoken,
        raw_text=locked.text,
        raw_ms=raw_ms,
        auto_text=auto.text if auto else "",
        auto_ms=auto_ms,
        auto_language=auto.language if auto else "",
        fallback_used=bool(auto and auto.fallback_used),
        cleaned_text=cleaned.text,
        cleaned_ms=cleaned_ms,
        clean_reason=cleaned.reason,
        used_llm=cleaned.used_llm,
        skipped_ms=skipped_ms,
        skipped_reason=skipped.reason,
        engine=locked.engine,
        auto_engine=auto.engine if auto else "",
    )


def run_candidate(
    name: str,
    llama_model: Path,
    whisper_model: Path,
    vad_model: Path,
    engines_dir: Path,
    entries: Sequence[prompt_set.ManifestEntry],
    clips_dir: Path,
    startup_timeout_s: float,
    allow_cpu_cleanup: bool = False,
) -> CandidateResult:
    """Launch the engines on this candidate, measure every clip, shut the engines down."""
    vulkan_dir = engines_dir / "vulkan"
    paths = EnginePaths(
        vulkan_dir=vulkan_dir,
        cpu_dir=engines_dir / "cpu",
        whisper_model=whisper_model,
        vad_model=vad_model,
        llama_model=llama_model,
        log_dir=REPO_DIR / "build" / "out" / "bench-logs" / name,
    )
    paths.log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] selecting the GPU device", file=sys.stderr)
    selection = gpu.select_device(
        vulkan_dir / "llama-server.exe", whisper_server=vulkan_dir / "whisper-server.exe"
    )
    print(f"[{name}] GPU: {selection.name or 'none'} (raw index {selection.raw_index})",
          file=sys.stderr)

    def on_status(engine: Engine, state: EngineState, reason: str) -> None:
        print(f"[{name}] {engine.value}: {state.value} ({reason})", file=sys.stderr)

    notes: list[str] = []
    supervisor = EngineSupervisor(paths, selection, on_status, startup_timeout_s=startup_timeout_s)
    clips: list[ClipResult] = []
    llama_variant = "vulkan"
    supervisor.start()
    try:
        for engine in (Engine.WHISPER, Engine.LLAMA):
            if not supervisor.wait_ready(engine, startup_timeout_s):
                notes.append(f"{engine.value} never reached a serving state")
        llama_variant = supervisor.variant(Engine.LLAMA)
        if supervisor.variant(Engine.WHISPER) == "cpu":
            notes.append("whisper served from its CPU build; the latencies are not comparable")
        whisper_url, llama_url = supervisor.whisper_url, supervisor.llama_url
        if not whisper_url or not llama_url:
            notes.append("an engine exposed no URL; nothing was measured")
            return CandidateResult(
                name=name,
                model_file=llama_model.name,
                whisper_model=whisper_model.name,
                gpu_name=selection.name,
                clips=(),
                notes=tuple(notes),
                llama_variant=llama_variant,
                allow_cpu_cleanup=allow_cpu_cleanup,
            )
        whisper_client = asr.WhisperClient(whisper_url)
        llama_client = cleanup.LlamaClient(llama_url)
        fillers = cleanup.default_fillers()
        corrections = cleanup.default_corrections()
        # Spec 7.2: the app seeds the policy with the first enabled language and carries the
        # last accepted one forward. The clips run in a fixed order (sorted by id) so the same
        # carry-over happens on every run instead of following the manifest's file order.
        auto_state = asr.LanguagePolicyState(last_accepted=DEFAULT_ENABLED_LANGUAGES[0])
        for index, entry in enumerate(entries, start=1):
            print(f"[{name}] {index}/{len(entries)} {entry.id}", file=sys.stderr)
            clips.append(
                measure_clip(
                    entry, clips_dir, whisper_client, llama_client, fillers, corrections, auto_state
                )
            )
    finally:
        supervisor.stop()
    return CandidateResult(
        name=name,
        model_file=llama_model.name,
        whisper_model=whisper_model.name,
        gpu_name=selection.name,
        clips=tuple(clips),
        notes=tuple(notes),
        llama_variant=llama_variant,
        allow_cpu_cleanup=allow_cpu_cleanup,
    )


# ----------------------------------------------------------------------------- the app configuration


@dataclass(frozen=True)
class AppLanguageRow:
    language: str
    clips: int
    engines: tuple[str, ...]
    raw_ms: float | None
    cleaned_run: int
    cleaned_run_ms: float | None
    cleaned_skipped_ms: float | None

    @property
    def raw_passed(self) -> bool | None:
        return None if self.raw_ms is None else self.raw_ms < APP_RAW_TARGET_MS

    @property
    def cleaned_run_passed(self) -> bool | None:
        return None if self.cleaned_run_ms is None else self.cleaned_run_ms < APP_CLEANED_TARGET_MS

    @property
    def cleaned_skipped_passed(self) -> bool | None:
        if self.cleaned_skipped_ms is None:
            return None
        return self.cleaned_skipped_ms < APP_SKIPPED_TARGET_MS


@dataclass(frozen=True)
class AppResult:
    hardware: str
    languages: tuple[str, ...]
    speech: tuple[tuple[str, str, str, tuple[str, ...]], ...]
    cleanup_model: str
    gpu_name: str
    cpu_mask: str
    clips: tuple[ClipResult, ...]
    notes: tuple[str, ...] = ()


def app_engine_paths(
    selection: Selection, models_dir: Path, engines_dir: Path, vad_model: Path, log_dir: Path
) -> EnginePaths:
    if not selection.asr:
        raise ValueError("no installed speech model serves these languages")
    base = EnginePaths(
        vulkan_dir=engines_dir / "vulkan",
        cpu_dir=engines_dir / "cpu",
        whisper_model=models_dir / selection.asr[0].file,
        vad_model=vad_model,
        llama_model=None,
        log_dir=log_dir,
    )
    return base.with_speech(models_dir, selection.asr).with_cleanup(models_dir, selection.cleanup)


def speech_routes(supervisor) -> tuple[asr.SpeechRoute, ...]:
    return tuple(
        asr.SpeechRoute(slot.engine, slot.runtime, slot.languages)
        for slot in supervisor.speech_engines()
    )


def speech_connector(supervisor) -> Callable[[asr.SpeechRoute], Any]:
    clients: dict[Any, Any] = {}

    def connect(route: asr.SpeechRoute):
        url = supervisor.url(route.engine)
        if url is None:
            return None
        key = (route.engine, url)
        if key not in clients:
            if route.runtime == asr.LLAMA_ASR:
                clients[key] = asr.LlamaAsrClient(url)
            else:
                clients[key] = asr.WhisperClient(url)
        return clients[key]

    return connect


def routed_transcriber(routes, connect, enabled_languages: Sequence[str]) -> Transcriber:
    enabled = list(enabled_languages)

    def transcribe(pcm, sample_rate, mode, state):
        return asr.transcribe_routed(pcm, sample_rate, mode, [], enabled, state, routes, connect)

    return transcribe


def app_gate(supervisor) -> cleanup.GateInput:
    return cleanup.GateInput(
        cleanup_enabled=True,
        profile_cleanup=True,
        engine_available=supervisor.llama_url is not None,
        cpu_fallback=supervisor.variant(Engine.LLAMA) == "cpu"
        and not supervisor.cpu_cleanup_allowed,
        cpu_selected=bool(supervisor.cpu_only),
        cleanup_languages=supervisor.cleanup_languages,
    )


def app_language_rows(
    clips: Sequence[ClipResult], languages: Sequence[str]
) -> list[AppLanguageRow]:
    rows = []
    for language in languages:
        group = [
            clip
            for clip in clips
            if clip.language == language and clip.bucket == "long" and not clip.error
        ]
        engines = tuple(sorted({clip.engine for clip in group if clip.engine}))
        raw = [clip.raw_ms for clip in group if clip.raw_ms is not None]
        ran = [clip.cleaned_ms for clip in group if clip.used_llm and clip.cleaned_ms is not None]
        skipped = [
            clip.cleaned_ms
            for clip in group
            if not clip.used_llm and clip.cleaned_ms is not None
        ]
        rows.append(
            AppLanguageRow(
                language=language,
                clips=len(group),
                engines=engines,
                raw_ms=percentile(raw, 50),
                cleaned_run=len(ran),
                cleaned_run_ms=percentile(ran, 50),
                cleaned_skipped_ms=percentile(skipped, 50),
            )
        )
    return rows


def _judged(value: float | None, passed: bool | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.0f} ms ({_verdict(passed)})"


def render_app_report(result: AppResult, date: str) -> str:
    clips = list(result.clips)
    out: list[str] = []
    out.append(f"# App configuration benchmark: {result.hardware}")
    out.append("")
    out.append(f"- Date: {date}")
    out.append(f"- Hardware: {result.hardware}")
    out.append(f"- Languages: {', '.join(result.languages)}")
    for key, model_id, runtime, languages in result.speech:
        out.append(f"- Speech engine {key}: {model_id} ({runtime}) for {', '.join(languages)}")
    out.append(f"- Cleanup model: {result.cleanup_model}")
    out.append(f"- GPU: {result.gpu_name or 'none'}")
    out.append(f"- Performance core mask: {result.cpu_mask or 'none'}")
    out.append(f"- Clips: {len(clips)}")
    for note in result.notes:
        out.append(f"- Note: {note}")
    out.append("")
    out.append("## Per language against spec 12 (long clips, medians)")
    out.append("")
    out.append(
        "Release to raw transcript is the locked pass through the app's routing. Release to "
        "cleaned text adds the gate and, when the gate lets it through, the cleanup call; the "
        "two cleaned columns split the clips the gate cleaned from the ones it delivered raw."
    )
    out.append("")
    out.append(
        f"| Language | Long clips | Engines | Raw (< {APP_RAW_TARGET_MS:.0f} ms) | Cleanup ran | "
        f"Cleaned, cleanup run (< {APP_CLEANED_TARGET_MS:.0f} ms) | "
        f"Cleaned, cleanup skipped (< {APP_SKIPPED_TARGET_MS:.0f} ms) |"
    )
    out.append("|---|---|---|---|---|---|---|")
    for row in app_language_rows(clips, result.languages):
        out.append(
            f"| {row.language} | {row.clips} | {', '.join(row.engines) or 'none'} | "
            f"{_judged(row.raw_ms, row.raw_passed)} | {row.cleaned_run} | "
            f"{_judged(row.cleaned_run_ms, row.cleaned_run_passed)} | "
            f"{_judged(row.cleaned_skipped_ms, row.cleaned_skipped_passed)} |"
        )
    out.append("")
    out.append("## Auto mode (spec 7.2, long clips)")
    out.append("")
    out.append("| Language | Long clips | Right language | Fallback | Auto raw p50 |")
    out.append("|---|---|---|---|---|")
    for language in result.languages:
        group = [
            clip
            for clip in clips
            if clip.language == language and clip.bucket == "long" and not clip.error
        ]
        if not group:
            continue
        right = sum(1 for clip in group if clip.auto_language == language)
        used = sum(1 for clip in group if clip.fallback_used)
        auto = [clip.auto_ms for clip in group if clip.auto_ms is not None]
        out.append(
            f"| {language} | {len(group)} | {right} | {used} | {_ms(percentile(auto, 50))} |"
        )
    out.append("")
    counted = reason_counts(clips)
    out.append("## Cleanup gate and guards")
    out.append("")
    out.append("| Reason | Clips |")
    out.append("|---|---|")
    for reason, count in sorted(counted.items()):
        out.append(f"| {reason} | {count} |")
    out.append("")
    out.append("## Clips")
    out.append("")
    out.append(
        "| Clip | Language | Category | Locked engine | Auto engine | Auto language | Raw | "
        "Cleaned | Reason | WER | Raw text |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for clip in clips:
        out.append(
            f"| {clip.id} | {clip.language} | {clip.category} | {clip.engine or 'none'} | "
            f"{clip.auto_engine or 'none'} | {clip.auto_language or 'none'} | "
            f"{_ms(clip.raw_ms)} | {_ms(clip.cleaned_ms)} | {clip.clean_reason} | "
            f"{_pct(clip.wer)} | {_cell(clip.raw_text)} |"
        )
    failed = [clip for clip in clips if clip.error]
    if failed:
        out.append("")
        out.append("## Clips that could not be measured")
        out.append("")
        for clip in failed:
            out.append(f"- {clip.id}: {_cell(clip.error)}")
    out.append("")
    return "\n".join(out)


def _default_cpu_plan() -> CpuPlan:
    from spells.win32.cpu import detect_cpu_plan

    return detect_cpu_plan()


def run_app(
    hardware: Hardware,
    languages: Sequence[str],
    engines_dir: Path,
    models_dir: Path,
    vad_model: Path,
    entries: Sequence[prompt_set.ManifestEntry],
    clips_dir: Path,
    startup_timeout_s: float,
    *,
    cpu_plan: Callable[[], CpuPlan] = _default_cpu_plan,
) -> AppResult:
    catalog = load_catalog()
    selection = select_models(languages, hardware, installed_ids(models_dir, catalog), catalog)
    log_dir = REPO_DIR / "build" / "out" / "bench-logs" / f"app-{hardware.value}"
    log_dir.mkdir(parents=True, exist_ok=True)
    paths = app_engine_paths(selection, models_dir, engines_dir, vad_model, log_dir)
    plan = cpu_plan()
    if hardware is Hardware.CPU:
        selection_gpu = gpu.NO_GPU
    else:
        vulkan_dir = engines_dir / "vulkan"
        selection_gpu = gpu.select_device(
            vulkan_dir / "llama-server.exe", whisper_server=vulkan_dir / "whisper-server.exe"
        )
    speech = tuple(
        (f"whisper-{index + 1}" if index else "whisper", choice.model_id, choice.runtime,
         choice.languages)
        for index, choice in enumerate(selection.asr)
    )
    cleanup_model = selection.cleanup.model_id if selection.cleanup else "none"
    label = f"app-{hardware.value}"
    notes = list(selection.notes)

    def on_status(engine: Engine, state: EngineState, reason: str) -> None:
        print(f"[{label}] {engine.value}: {state.value} ({reason})", file=sys.stderr)

    supervisor = EngineSupervisor(
        paths,
        selection_gpu,
        on_status,
        startup_timeout_s=startup_timeout_s,
        cpu_only=hardware is Hardware.CPU,
        cpu_cleanup_allowed=selection.cpu_cleanup_allowed,
        cpu_plan=plan,
    )
    clips: list[ClipResult] = []
    supervisor.start()
    try:
        for role in (Engine.WHISPER, Engine.LLAMA):
            if role is Engine.LLAMA and selection.cleanup is None:
                notes.append("no cleanup model suits these languages; raw text is delivered")
                continue
            if not supervisor.wait_ready(role, startup_timeout_s):
                notes.append(f"{role.value} never reached a serving state")
        routes = speech_routes(supervisor)
        connect = speech_connector(supervisor)
        llama_url = supervisor.llama_url
        llama_client = cleanup.LlamaClient(llama_url or "http://127.0.0.1:0")
        gate = app_gate(supervisor)
        fillers = cleanup.default_fillers()
        corrections = cleanup.default_corrections()
        auto_state = asr.LanguagePolicyState(last_accepted=languages[0])
        transcriber = routed_transcriber(routes, connect, languages)
        for index, entry in enumerate(entries, start=1):
            print(f"[{label}] {index}/{len(entries)} {entry.id}", file=sys.stderr)
            clips.append(
                measure_clip(
                    entry,
                    clips_dir,
                    None,
                    llama_client,
                    fillers,
                    corrections,
                    auto_state,
                    transcriber=transcriber,
                    gate=gate,
                    enabled_languages=languages,
                )
            )
    finally:
        supervisor.stop()
    return AppResult(
        hardware=hardware.value,
        languages=tuple(languages),
        speech=speech,
        cleanup_model=cleanup_model,
        gpu_name=selection_gpu.name,
        cpu_mask=plan.cpu_mask_hex or "",
        clips=tuple(clips),
        notes=tuple(notes),
    )


# ----------------------------------------------------------------------------- cli


def _resolve_clips(clips_dir: Path, wanted: Sequence[str]) -> list[prompt_set.ManifestEntry]:
    """The manifest clips, sorted by id.

    Auto mode carries the last accepted language from clip to clip (spec 7.2), exactly as the
    app does between dictations, so the order decides the result. Sorting by id makes that
    carry-over the same on every run rather than hiding it behind the manifest's file order.
    """
    entries = prompt_set.read_manifest(clips_dir / "manifest.json")
    if wanted:
        keep = set(wanted)
        entries = [entry for entry in entries if entry.id in keep]
    return sorted(entries, key=lambda entry: entry.id)


def _split(value: str | None) -> list[str]:
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Spells model selection benchmark (spec 18)")
    parser.add_argument("--fetch", action="store_true",
                        help="download the candidates plus the whisper and VAD models")
    parser.add_argument("--select", metavar="CANDIDATE",
                        help="pin this candidate as models.cleanup_model in build/pins.json")
    parser.add_argument("--candidates", help="comma separated candidate names, default all")
    parser.add_argument("--clips", help="comma separated clip ids, default every recorded clip")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run with ggml-tiny.bin and the test GGUF against whatever clips exist",
    )
    parser.add_argument(
        "--hardware",
        choices=[hardware.value for hardware in Hardware],
        help="run the app's configuration for this hardware instead of the candidates",
    )
    parser.add_argument(
        "--languages",
        help="comma separated enabled languages for --hardware, default en,de,sq",
    )
    parser.add_argument(
        "--allow-cpu-cleanup",
        action="store_true",
        help="measure the cleaned rows even while llama serves from its CPU build",
    )
    parser.add_argument("--clips-dir", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=None,
                        help="default bench/reports, or bench/reports/smoke with --smoke")
    parser.add_argument("--engines-dir", type=Path, default=REPO_DIR / "build" / "out" / "engines")
    parser.add_argument("--models-dir", type=Path, default=REPO_DIR / "build" / "cache" / "models")
    parser.add_argument("--pins", type=Path, default=REPO_DIR / "build" / "pins.json")
    parser.add_argument("--cache", type=Path, default=REPO_DIR / "build" / "cache")
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument(
        "--date",
        default=datetime.datetime.now(datetime.UTC).astimezone().date().isoformat(),
    )
    args = parser.parse_args(argv)

    fetch = _load_fetch()
    pins = fetch.load_pins(args.pins)
    wanted_candidates = _split(args.candidates) or candidate_names(pins)

    if args.select:
        return select_model(args.select, args.pins, args.cache)
    if args.fetch:
        return fetch_all(wanted_candidates, args.pins, args.cache)
    if args.hardware and args.smoke:
        parser.error("--hardware runs the real configuration; it cannot be combined with --smoke")

    clips_dir = args.clips_dir
    if clips_dir is None:
        smoke_dir = BENCH_DIR / "clips" / "smoke"
        use_smoke_dir = args.smoke and (smoke_dir / "manifest.json").is_file()
        clips_dir = smoke_dir if use_smoke_dir else BENCH_DIR / "clips"
    reports_dir = args.reports_dir
    if reports_dir is None:
        # A smoke report is throwaway, so it stays out of the committed bench/reports/ folder.
        reports_dir = BENCH_DIR / "reports" / "smoke" if args.smoke else BENCH_DIR / "reports"
    entries = _resolve_clips(clips_dir, _split(args.clips))
    if not entries:
        print(
            f"ERROR: no clips in {clips_dir / 'manifest.json'}. Record them with "
            "`py bench/record.py`, or synthesise smoke clips with `py bench/smoke_clips.py`.",
            file=sys.stderr,
        )
        return 1

    def pinned_model(key: str) -> Path:
        return args.models_dir / fetch.filename_for(fetch.resolve(pins, key))

    vad_model = pinned_model(VAD_MODEL_PIN)
    if args.hardware:
        return run_app_mode(args, entries, clips_dir, reports_dir, vad_model)
    if args.smoke:
        whisper_model = pinned_model(SMOKE_WHISPER_PIN)
        candidates = [("smoke", pinned_model(SMOKE_CANDIDATE_PIN))]
    else:
        whisper_model = pinned_model(WHISPER_MODEL_PIN)
        candidates = [
            (name, pinned_model(f"bench_candidates.{name}")) for name in wanted_candidates
        ]

    missing = [str(path) for _, path in candidates if not path.is_file()]
    missing += [str(path) for path in (whisper_model, vad_model) if not path.is_file()]
    if missing:
        print("ERROR: missing model files:\n  " + "\n  ".join(missing), file=sys.stderr)
        print("Run `py bench/run.py --fetch` first.", file=sys.stderr)
        return 1

    reports_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name, model_file in candidates:
        result = run_candidate(
            name,
            model_file,
            whisper_model,
            vad_model,
            args.engines_dir,
            entries,
            clips_dir,
            args.startup_timeout,
            allow_cpu_cleanup=args.allow_cpu_cleanup,
        )
        results.append(result)
        report_path = reports_dir / f"{args.date}-{name}.md"
        report_path.write_text(render_candidate_report(result, args.date), encoding="utf-8")
        print(f"wrote {report_path}")
    summary_path = reports_dir / f"{args.date}-summary.md"
    summary_path.write_text(render_summary(results, args.date), encoding="utf-8")
    print(f"wrote {summary_path}")
    invalid = [result.name for result in results if cleanup_is_invalid(result)]
    if invalid:
        print(
            f"ERROR: the cleaned rows of {', '.join(invalid)} are not valid: llama served from "
            "its CPU build, where spec 8.1 skips cleanup. Rerun on the Vulkan build, or pass "
            "--allow-cpu-cleanup to measure them anyway.",
            file=sys.stderr,
        )
        return 1
    return 0


def run_app_mode(args, entries, clips_dir: Path, reports_dir: Path, vad_model: Path) -> int:
    hardware = Hardware(args.hardware)
    languages = _split(args.languages) or list(DEFAULT_ENABLED_LANGUAGES)
    if not vad_model.is_file():
        print(f"ERROR: missing {vad_model}", file=sys.stderr)
        return 1
    try:
        result = run_app(
            hardware,
            languages,
            args.engines_dir,
            args.models_dir,
            vad_model,
            entries,
            clips_dir,
            args.startup_timeout,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{args.date}-app-{hardware.value}.md"
    report_path.write_text(render_app_report(result, args.date), encoding="utf-8")
    print(f"wrote {report_path}")
    return 0 if any(not clip.error for clip in result.clips) else 1


if __name__ == "__main__":
    sys.exit(main())
