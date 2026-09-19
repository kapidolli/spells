"""Build the benchmark clip set without a microphone (spec 18 step 1, batch 5).

English prompts are spoken by Windows SAPI, German and Albanian prompts by the Piper voices
de_DE-thorsten-high and sq_AL-edon-medium. A second Albanian set comes from real Common Voice
speakers (public domain): sentences of one speaker joined into 10 to 30 s clips, plus single
sentences under 2 s. Every clip is 16 kHz mono 16-bit with low noise around the speech, as a
hotkey recording has.

Usage:
    py bench/synth_clips.py
    py bench/synth_clips.py --languages sq --cv-long 8 --cv-short 4
"""

from __future__ import annotations

import argparse
import csv
import datetime
import random
import sys
import urllib.request
import wave
from array import array
from collections import defaultdict
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
BENCH_DIR = REPO_DIR / "bench"
for _extra in (REPO_DIR / "src", BENCH_DIR):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import prompts as prompt_set

SAMPLE_RATE = 16000
LEAD_S = 0.3
TAIL_S = 0.4
GAP_S = 0.35
NOISE_PEAK = 40
TRIM_FRAME_S = 0.03
TRIM_MARGIN_S = 0.09
TRIM_RELATIVE = 0.06
SHORT_TRIMMED_MAX_S = 1.95
SHORT_RAW_MAX_MS = 2800
PIPER_VOICES = {"de": "de_DE-thorsten-high.onnx", "sq": "sq_AL-edon-medium.onnx"}
CV_CLIP_URL = (
    "https://huggingface.co/datasets/edyrkaj/albanian-common-voice-v1.0/resolve/main/sq/clips/"
)


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).astimezone().isoformat(timespec="seconds")


def noise(seconds: float, rng: random.Random) -> bytes:
    count = round(seconds * SAMPLE_RATE)
    return array("h", (rng.randint(-NOISE_PEAK, NOISE_PEAK) for _ in range(count))).tobytes()


def pad(pcm: bytes, rng: random.Random) -> bytes:
    return noise(LEAD_S, rng) + pcm + noise(TAIL_S, rng)


def join(segments: list[bytes], rng: random.Random) -> bytes:
    out = bytearray()
    for index, segment in enumerate(segments):
        if index:
            out += noise(GAP_S, rng)
        out += segment
    return bytes(out)


def trim_silence(pcm: bytes) -> bytes:
    values = array("h", pcm)
    frame = int(TRIM_FRAME_S * SAMPLE_RATE)
    energies = []
    for start in range(0, len(values), frame):
        chunk = values[start : start + frame]
        energies.append((sum(v * v for v in chunk) / max(1, len(chunk))) ** 0.5)
    if not energies or max(energies) < 4 * NOISE_PEAK:
        return b""
    threshold = max(energies) * TRIM_RELATIVE
    loud = [i for i, energy in enumerate(energies) if energy >= threshold]
    margin = int(TRIM_MARGIN_S * SAMPLE_RATE)
    begin = max(0, loud[0] * frame - margin)
    end = min(len(values), (loud[-1] + 1) * frame + margin)
    return values[begin:end].tobytes()


def select_cv_sets(
    rows: list[dict], durations_ms: dict[str, int], count: int, min_s: float, max_s: float
) -> list[list[dict]]:
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for item in rows:
        if item["path"] not in durations_ms:
            continue
        if int(item.get("down_votes") or 0) != 0 or int(item.get("up_votes") or 0) < 2:
            continue
        by_speaker[item["client_id"]].append(item)
    speakers = sorted(by_speaker, key=lambda cid: (-len(by_speaker[cid]), cid))
    sets: list[list[dict]] = []
    for speaker in speakers:
        if len(sets) >= count:
            break
        chosen: list[dict] = []
        total = 0.0
        for item in by_speaker[speaker]:
            seconds = durations_ms[item["path"]] / 1000
            if total + seconds > max_s:
                continue
            chosen.append(item)
            total += seconds
            if total >= min_s:
                break
        if total >= min_s:
            sets.append(chosen)
    return sets


def merge_entries(
    existing: list[prompt_set.ManifestEntry], new: list[prompt_set.ManifestEntry]
) -> list[prompt_set.ManifestEntry]:
    merged = {entry.id: entry for entry in existing}
    merged.update({entry.id: entry for entry in new})
    return sorted(merged.values(), key=lambda entry: entry.id)


def cv_id(category: str, index: int) -> str:
    return f"sq-cv{category}-{index:02d}"


def read_wav(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError(f"{path} is not mono 16-bit")
        return handle.readframes(handle.getnframes()), handle.getframerate()


def write_wav(path: Path, pcm: bytes) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm)
    return len(pcm) / 2 / SAMPLE_RATE


def to_16k(pcm: bytes, rate: int) -> bytes:
    if rate == SAMPLE_RATE:
        return pcm
    import miniaudio

    fmt = miniaudio.SampleFormat.SIGNED16
    return bytes(miniaudio.convert_frames(fmt, 1, rate, pcm, fmt, 1, SAMPLE_RATE))


class PiperSpeaker:
    def __init__(self, voices_dir: Path):
        self._voices_dir = voices_dir
        self._loaded: dict[str, object] = {}

    def speak(self, language: str, text: str) -> bytes:
        from piper import PiperVoice

        if language not in self._loaded:
            self._loaded[language] = PiperVoice.load(self._voices_dir / PIPER_VOICES[language])
        voice = self._loaded[language]
        chunks = list(voice.synthesize(text))
        rate = chunks[0].sample_rate
        return to_16k(b"".join(chunk.audio_int16_bytes for chunk in chunks), rate)


def sapi_speak(text: str, scratch: Path) -> bytes:
    import smoke_clips

    if not smoke_clips.synthesise(text, scratch):
        raise RuntimeError("SAPI could not speak the English prompt")
    pcm, rate = read_wav(scratch)
    scratch.unlink(missing_ok=True)
    return to_16k(pcm, rate)


def fetch_cv_clip(name: str, clips_dir: Path) -> Path:
    destination = clips_dir / name
    if destination.is_file():
        return destination
    clips_dir.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(name + ".part")
    with urllib.request.urlopen(CV_CLIP_URL + name, timeout=60) as response:
        partial.write_bytes(response.read())
    partial.replace(destination)
    return destination


def decode_mp3(path: Path) -> bytes:
    import miniaudio

    decoded = miniaudio.decode_file(
        str(path), miniaudio.SampleFormat.SIGNED16, nchannels=1, sample_rate=SAMPLE_RATE
    )
    return decoded.samples.tobytes()


def entry_for(
    clip_id: str, language: str, category: str, reference: str, text: str, duration_s: float
) -> prompt_set.ManifestEntry:
    return prompt_set.ManifestEntry(
        id=clip_id,
        language=language,
        category=category,
        reference=reference,
        text=text,
        duration_s=round(duration_s, 3),
        recorded_at=_now(),
        file=f"{clip_id}.wav",
    )


def build_prompt_clips(
    languages: set[str], out: Path, voices_dir: Path, rng: random.Random
) -> list[prompt_set.ManifestEntry]:
    speaker = PiperSpeaker(voices_dir)
    entries = []
    for prompt in prompt_set.PROMPTS:
        if prompt.language not in languages or prompt_set.is_placeholder(prompt):
            continue
        spoken = prompt_set.spoken_text(prompt)
        if prompt.category == "silence":
            pcm = noise(prompt_set.SILENCE_SECONDS, rng)
        elif prompt.language == "en":
            pcm = pad(trim_silence(sapi_speak(spoken, out / f".{prompt.id}.sapi.wav")), rng)
        else:
            pcm = pad(trim_silence(speaker.speak(prompt.language, spoken)), rng)
        duration = write_wav(out / f"{prompt.id}.wav", pcm)
        print(f"{prompt.id}: {duration:.1f} s")
        entries.append(
            entry_for(prompt.id, prompt.language, prompt.category, prompt.reference, spoken,
                      duration)
        )
    return entries


def build_cv_clips(
    cv_dir: Path, out: Path, long_count: int, short_count: int, rng: random.Random
) -> list[prompt_set.ManifestEntry]:
    with (cv_dir / "test.tsv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE))
    with (cv_dir / "clip_durations.tsv").open(encoding="utf-8", newline="") as handle:
        durations = {r["clip"]: int(r["duration[ms]"]) for r in csv.DictReader(handle, delimiter="\t")}
    entries = []
    long_sets = select_cv_sets(rows, durations, long_count, min_s=17.0, max_s=32.0)
    for index, chosen in enumerate(long_sets, start=1):
        segments = [trim_silence(decode_mp3(fetch_cv_clip(r["path"], cv_dir / "clips")))
                    for r in chosen]
        pcm = pad(join([s for s in segments if s], rng), rng)
        clip_id = cv_id("long", index)
        duration = write_wav(out / f"{clip_id}.wav", pcm)
        text = " ".join(r["sentence"].strip() for r in chosen)
        print(f"{clip_id}: {duration:.1f} s from {len(chosen)} sentences")
        entries.append(entry_for(clip_id, "sq", "long", text, text, duration))
    used_speakers = {s[0]["client_id"] for s in long_sets}
    short_rows = [
        r for r in rows
        if durations.get(r["path"], 10**6) <= SHORT_RAW_MAX_MS
        and int(r.get("down_votes") or 0) == 0
        and r["client_id"] not in used_speakers
    ]
    index = 0
    seen_speakers: set[str] = set()
    for item in short_rows:
        if index >= short_count:
            break
        if item["client_id"] in seen_speakers:
            continue
        speech = trim_silence(decode_mp3(fetch_cv_clip(item["path"], cv_dir / "clips")))
        if not speech or len(speech) / 2 / SAMPLE_RATE > SHORT_TRIMMED_MAX_S - LEAD_S - TAIL_S:
            continue
        index += 1
        seen_speakers.add(item["client_id"])
        clip_id = cv_id("short", index)
        duration = write_wav(out / f"{clip_id}.wav", pad(speech, rng))
        text = item["sentence"].strip()
        print(f"{clip_id}: {duration:.1f} s")
        entries.append(entry_for(clip_id, "sq", "short", text, text, duration))
    return entries


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=BENCH_DIR / "clips")
    parser.add_argument("--voices", type=Path, default=REPO_DIR / "build" / "cache" / "voices")
    parser.add_argument("--cv-dir", type=Path,
                        default=REPO_DIR / "build" / "cache" / "commonvoice-sq")
    parser.add_argument("--languages", default="en,de,sq")
    parser.add_argument("--cv-long", type=int, default=8)
    parser.add_argument("--cv-short", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    languages = {code.strip() for code in args.languages.split(",") if code.strip()}
    entries = build_prompt_clips(languages, args.out, args.voices, rng)
    if "sq" in languages and (args.cv_long or args.cv_short):
        entries += build_cv_clips(args.cv_dir, args.out, args.cv_long, args.cv_short, rng)
    manifest = args.out / "manifest.json"
    merged = merge_entries(prompt_set.read_manifest(manifest), entries)
    prompt_set.write_manifest(manifest, merged)
    print(f"wrote {manifest} with {len(merged)} clip(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
