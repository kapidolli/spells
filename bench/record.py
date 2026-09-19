"""Record the benchmark clips on the console (spec 18 step 1).

Walks the prompt set of bench/prompts.py, records each one through spells.audio.Recorder at
16 kHz mono, and writes bench/clips/<id>.wav plus bench/clips/manifest.json. The run is
resumable: a prompt whose clip already exists is skipped unless it is named in --redo.

Usage:
    py bench/record.py
    py bench/record.py --redo en-long-2,de-filler-1
    py bench/record.py --language sq
    py bench/record.py --warmup-out data/warmup.wav
    py bench/record.py --list-devices

Per prompt: Enter starts the recording, Enter stops it, r records it again, s skips it, q ends
the session. Silence prompts record three seconds on their own with nothing spoken. A prompt
whose text is still a TODO(owner) placeholder is skipped with a notice.

After each Albanian clip the script asks which fillers and which self-correction phrase were
actually spoken and collects the answers in bench/clips/sq_vocab.json. Those are candidates for
data/fillers/sq.txt and data/corrections/sq.txt (spec 8.1); nothing is written into data/ here.

Audio never leaves bench/clips/, which is gitignored (spec 17: recordings are not committed).
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
BENCH_DIR = REPO_DIR / "bench"
for _extra in (REPO_DIR / "src", BENCH_DIR):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import prompts as prompt_set

from spells import audio

SAMPLE_RATE = 16000
VOCAB_NOTE = (
    "Candidates for data/fillers/sq.txt and data/corrections/sq.txt (spec 8.1), collected while "
    "recording. Copy the ones you really use into those files by hand; bench/record.py never "
    "writes into data/."
)


class Quit(Exception):
    """The user asked to end the session."""


def _now() -> str:
    """Local wall clock with its offset, for the manifest's recorded_at."""
    return datetime.datetime.now(datetime.UTC).astimezone().isoformat(timespec="seconds")


def _hint(prompt: prompt_set.Prompt) -> str:
    return {
        "short": "Short clip: keep it under two seconds.",
        "long": "Long clip: aim for 10 to 30 seconds, at your normal pace.",
        "filler": "Speak every bracketed word out loud as a filler; drop the brackets.",
        "correction": "Say the correction out loud, exactly as written.",
        "question": "Just read the question. Cleanup must never answer it.",
        "silence": "Say nothing. The recorder stops on its own.",
    }.get(prompt.category, "")


def _ask(question: str, allowed: str) -> str:
    """Read one console answer, lowercased and stripped; '' means the bare Enter."""
    keys = tuple(allowed)
    while True:
        try:
            answer = input(question).strip().lower()
        except EOFError as exc:
            raise Quit("stdin closed") from exc
        if answer == "q":
            raise Quit("user quit")
        if answer == "" or answer in keys:
            return answer
        print(f"  Please answer with one of: Enter, {', '.join(keys)}, q")


def _duration_note(prompt: prompt_set.Prompt, duration_s: float) -> str:
    bucket = prompt_set.duration_bucket(duration_s)
    if prompt.category == "short" and bucket != "short":
        return "  Note: over two seconds, so this will not count as a short clip."
    if prompt.category in ("long", "filler") and bucket != "long":
        return (
            "  Note: outside 10 to 30 seconds, so this will not count against the spec 12 "
            "targets."
        )
    return ""


def _record_once(recorder: audio.Recorder, prompt: prompt_set.Prompt, silence_s: float) -> bytes:
    """One take. Returns the PCM, or b'' when the microphone refused or delivered silence."""
    try:
        recorder.start()
    except audio.MicError as exc:
        print(f"  Microphone error: {exc}")
        return b""
    if prompt.category == "silence":
        print(f"  Recording {silence_s:.0f} s of silence, say nothing ...")
        time.sleep(silence_s)
    else:
        print("  Recording. Press Enter to stop.")
        try:
            input()
        except EOFError as exc:
            recorder.cancel()
            raise Quit("stdin closed") from exc
    try:
        return recorder.stop()
    except audio.MicError as exc:
        print(f"  Microphone error: {exc}")
        return b""


def _collect_sq_vocabulary(clips_dir: Path, prompt: prompt_set.Prompt) -> None:
    """Ask which Albanian fillers and correction phrase were spoken and store the answers."""
    if prompt.language != "sq" or prompt.category == "silence":
        return
    try:
        fillers = input("  Fillers you actually said (comma separated, Enter for none): ")
        corrections = input("  Self-correction phrase you used (comma separated, Enter for none): ")
    except EOFError as exc:
        raise Quit("stdin closed") from exc
    entry = {
        "id": prompt.id,
        "fillers": [part.strip() for part in fillers.split(",") if part.strip()],
        "corrections": [part.strip() for part in corrections.split(",") if part.strip()],
    }
    path = clips_dir / "sq_vocab.json"
    payload = {"version": 1, "note": VOCAB_NOTE, "clips": []}
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["note"] = VOCAB_NOTE
    clips = [item for item in payload.get("clips", []) if item.get("id") != prompt.id]
    clips.append(entry)
    clips.sort(key=lambda item: item["id"])
    payload["clips"] = clips
    for key in ("fillers", "corrections"):
        seen: list[str] = []
        for item in clips:
            for word in item.get(key, []):
                if word.lower() not in [s.lower() for s in seen]:
                    seen.append(word)
        payload[key] = seen
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  Noted in {path.name}.")


def _walk(
    to_record: list[prompt_set.Prompt],
    recorder: audio.Recorder,
    clips_dir: Path,
    entries: dict[str, prompt_set.ManifestEntry],
    manifest_path: Path,
    silence_s: float,
    warmup_out: Path | None,
) -> None:
    total = len(to_record)
    for index, prompt in enumerate(to_record, start=1):
        is_warmup = prompt.id == prompt_set.WARMUP_PROMPT.id
        print()
        print("=" * 78)
        print(f"[{index}/{total}] {prompt.id}  ({prompt.language}, {prompt.category})")
        print(f"  {_hint(prompt)}")
        print()
        print(f"  {prompt.text}")
        print()
        answer = _ask("  Enter to start, s to skip, q to quit: ", "s")
        if answer == "s":
            print("  Skipped.")
            continue
        while True:
            pcm = _record_once(recorder, prompt, silence_s)
            if not pcm:
                answer = _ask("  Nothing was captured. Enter to try again, s to skip: ", "s")
                if answer == "s":
                    break
                continue
            duration_s = len(pcm) / (2 * SAMPLE_RATE)
            print(f"  Captured {duration_s:.1f} s.")
            note = _duration_note(prompt, duration_s)
            if note:
                print(note)
            answer = _ask("  Enter to keep, r to record again, s to skip: ", "rs")
            if answer == "r":
                continue
            if answer == "s":
                print("  Skipped.")
                break
            destination = warmup_out if is_warmup and warmup_out else clips_dir / f"{prompt.id}.wav"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(audio.pcm16_to_wav(pcm, SAMPLE_RATE))
            print(f"  Wrote {destination}.")
            if not is_warmup:
                entries[prompt.id] = prompt_set.ManifestEntry(
                    id=prompt.id,
                    language=prompt.language,
                    category=prompt.category,
                    reference=prompt.reference,
                    text=prompt.text,
                    duration_s=round(duration_s, 3),
                    recorded_at=_now(),
                    file=destination.name,
                )
                prompt_set.write_manifest(manifest_path, list(entries.values()))
                _collect_sq_vocabulary(clips_dir, prompt)
            break


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Record the Spells benchmark clips (spec 18)")
    parser.add_argument("--clips-dir", type=Path, default=BENCH_DIR / "clips")
    parser.add_argument("--redo", default="", help="comma separated ids to record again")
    parser.add_argument("--only", default="", help="comma separated ids to walk, default all")
    parser.add_argument("--language", default="", help="limit to one language code")
    parser.add_argument("--warmup-out", type=Path, default=None,
                        help="also record the warm-up prompt and write the WAV here "
                             "(spec 13 uses data/warmup.wav)")
    parser.add_argument("--device", default=None, help="input device name or index")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--silence-seconds", type=float, default=prompt_set.SILENCE_SECONDS)
    args = parser.parse_args(argv)

    if args.list_devices:
        for device in audio.list_input_devices():
            print(f"{device.index:>3}  {device.name}")
        return 0

    clips_dir = args.clips_dir
    manifest_path = clips_dir / "manifest.json"
    entries = {entry.id: entry for entry in prompt_set.read_manifest(manifest_path)}
    redo = {part.strip() for part in args.redo.split(",") if part.strip()}
    only = {part.strip() for part in args.only.split(",") if part.strip()}

    to_record: list[prompt_set.Prompt] = []
    skipped_placeholder: list[str] = []
    skipped_done: list[str] = []
    for prompt in prompt_set.PROMPTS:
        if only and prompt.id not in only:
            continue
        if args.language and prompt.language != args.language:
            continue
        if prompt_set.is_placeholder(prompt):
            skipped_placeholder.append(prompt.id)
            continue
        if (clips_dir / f"{prompt.id}.wav").is_file() and prompt.id not in redo:
            skipped_done.append(prompt.id)
            continue
        to_record.append(prompt)
    if args.warmup_out is not None:
        to_record.append(prompt_set.WARMUP_PROMPT)

    if skipped_placeholder:
        print(
            f"Skipping {len(skipped_placeholder)} prompt(s) whose text is still a "
            f"{prompt_set.PLACEHOLDER_MARK} placeholder: {', '.join(skipped_placeholder)}."
        )
        print("Write them in bench/prompts.py first, then run this again.")
        print()
    if skipped_done:
        print(
            f"Already recorded, skipping {len(skipped_done)}: {', '.join(skipped_done)}. "
            "Use --redo <id> to record one again."
        )
        print()
    if not to_record:
        print("Nothing left to record.")
        return 0

    device = args.device
    if device is not None and device.strip().lstrip("-").isdigit():
        device = int(device)
    recorder = audio.Recorder(device=device, sample_rate=SAMPLE_RATE)
    try:
        _walk(
            to_record,
            recorder,
            clips_dir,
            entries,
            manifest_path,
            args.silence_seconds,
            args.warmup_out,
        )
    except Quit as exc:
        print(f"\nStopped ({exc}). Recorded clips are kept; run again to continue.")
    except KeyboardInterrupt:
        print("\nInterrupted. Recorded clips are kept; run again to continue.")
    finally:
        recorder.close()
    print(f"\n{len(entries)} clip(s) in {manifest_path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
