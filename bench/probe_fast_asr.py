from __future__ import annotations

import argparse
import base64
import json
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
BENCH_DIR = REPO_DIR / "bench"
for _extra in (BENCH_DIR,):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import prompts as prompt_set
from run import levenshtein, normalise_words

CLIPS_DIR = BENCH_DIR / "clips"
MANIFEST_PATH = CLIPS_DIR / "manifest.json"
ASR_FAST_DIR = REPO_DIR / "build" / "cache" / "asr-fast"
RESULTS_DIR = ASR_FAST_DIR / "results"
ENGINES_CPU_DIR = REPO_DIR / "build" / "out" / "engines" / "cpu"
THREADS = 6
REPEATS = 2

CANDIDATE_LANGUAGES = {
    "parakeet": ("en", "de"),
    "qwen3-asr": ("en", "de"),
    "omnilingual-300m": ("en", "de", "sq"),
    "omnilingual-1b": ("en", "de", "sq"),
}

QWEN3_ASR_PREFIX = re.compile(r"^language\s+\S+\s*<asr_text>", re.IGNORECASE)
QWEN3_ASR_SUFFIX = re.compile(r"</asr_text>\s*$", re.IGNORECASE)


def load_wav_samples(path):
    import numpy as np

    with wave.open(str(path), "rb") as handle:
        sample_rate = handle.getframerate()
        frame_count = handle.getnframes()
        raw = handle.readframes(frame_count)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sample_rate


def clip_entries(languages):
    entries = prompt_set.read_manifest(MANIFEST_PATH)
    return [entry for entry in entries if entry.language in languages]


def process_rss_mb(pid=None):
    import psutil

    proc = psutil.Process(pid) if pid is not None else psutil.Process()
    return proc.memory_info().rss / (1024 * 1024)


def summarize_times(values_ms):
    if not values_ms:
        return {"median_ms": None, "max_ms": None, "min_ms": None, "n": 0}
    return {
        "median_ms": statistics.median(values_ms),
        "max_ms": max(values_ms),
        "min_ms": min(values_ms),
        "n": len(values_ms),
    }


def micro_wer(records, text_key, reference_key):
    edits = 0
    reference_words_total = 0
    for record in records:
        reference_words = normalise_words(record[reference_key])
        if not reference_words:
            continue
        hyp_words = normalise_words(record[text_key])
        edits += levenshtein(reference_words, hyp_words)
        reference_words_total += len(reference_words)
    if reference_words_total == 0:
        return None
    return edits / reference_words_total


def build_report(candidate_name, load_s, rss_mb, records):
    languages = sorted({record["language"] for record in records})
    long_records = [record for record in records if record["bucket"] == "long"]
    by_language = {}
    for language in languages:
        language_records = [record for record in records if record["language"] == language]
        language_long = [record for record in language_records if record["bucket"] == "long"]
        by_language[language] = {
            "wer_reference": micro_wer(language_records, "hyp", "reference"),
            "wer_spoken": micro_wer(language_records, "hyp", "spoken"),
            "wer_reference_long": micro_wer(language_long, "hyp", "reference"),
            "timing_long": summarize_times(
                [t for record in language_long for t in record["times_ms"]]
            ),
        }
    return {
        "candidate": candidate_name,
        "load_s": load_s,
        "rss_mb": rss_mb,
        "threads": THREADS,
        "repeats": REPEATS,
        "languages": languages,
        "timing_long_all": summarize_times(
            [t for record in long_records for t in record["times_ms"]]
        ),
        "by_language": by_language,
        "records": records,
    }


def run_sherpa_transducer(candidate_name, model_dir):
    import sherpa_onnx

    languages = CANDIDATE_LANGUAGES[candidate_name]
    entries = clip_entries(languages)

    t0 = time.perf_counter()
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(model_dir / "encoder.int8.onnx"),
        decoder=str(model_dir / "decoder.int8.onnx"),
        joiner=str(model_dir / "joiner.int8.onnx"),
        tokens=str(model_dir / "tokens.txt"),
        num_threads=THREADS,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        model_type="nemo_transducer",
    )
    load_s = time.perf_counter() - t0

    warmup_entry = next((e for e in entries if e.category == "long"), entries[0])
    samples, sample_rate = load_wav_samples(CLIPS_DIR / warmup_entry.file)
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    recognizer.decode_stream(stream)

    records = []
    for entry in entries:
        samples, sample_rate = load_wav_samples(CLIPS_DIR / entry.file)
        times_ms = []
        hyp_text = ""
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            stream = recognizer.create_stream()
            stream.accept_waveform(sample_rate, samples)
            recognizer.decode_stream(stream)
            times_ms.append((time.perf_counter() - t0) * 1000)
            hyp_text = stream.result.text
        records.append(
            {
                "id": entry.id,
                "language": entry.language,
                "category": entry.category,
                "bucket": prompt_set.duration_bucket(entry.duration_s),
                "duration_s": entry.duration_s,
                "reference": entry.reference,
                "spoken": entry.text,
                "hyp": hyp_text,
                "times_ms": times_ms,
            }
        )

    rss_mb = process_rss_mb()
    return build_report(candidate_name, load_s, rss_mb, records)


def run_omnilingual(candidate_name, model_dir):
    import sherpa_onnx

    languages = CANDIDATE_LANGUAGES[candidate_name]
    entries = clip_entries(languages)

    t0 = time.perf_counter()
    recognizer = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
        model=str(model_dir / "model.int8.onnx"),
        tokens=str(model_dir / "tokens.txt"),
        num_threads=THREADS,
    )
    load_s = time.perf_counter() - t0

    warmup_entry = next((e for e in entries if e.category == "long"), entries[0])
    samples, sample_rate = load_wav_samples(CLIPS_DIR / warmup_entry.file)
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    recognizer.decode_stream(stream)

    records = []
    for entry in entries:
        samples, sample_rate = load_wav_samples(CLIPS_DIR / entry.file)
        times_ms = []
        hyp_text = ""
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            stream = recognizer.create_stream()
            stream.accept_waveform(sample_rate, samples)
            recognizer.decode_stream(stream)
            times_ms.append((time.perf_counter() - t0) * 1000)
            hyp_text = stream.result.text
        records.append(
            {
                "id": entry.id,
                "language": entry.language,
                "category": entry.category,
                "bucket": prompt_set.duration_bucket(entry.duration_s),
                "duration_s": entry.duration_s,
                "reference": entry.reference,
                "spoken": entry.text,
                "hyp": hyp_text,
                "times_ms": times_ms,
            }
        )

    rss_mb = process_rss_mb()
    return build_report(candidate_name, load_s, rss_mb, records)


def wait_for_health(port, timeout_s=90):
    deadline = time.perf_counter() + timeout_s
    url = f"http://127.0.0.1:{port}/health"
    while time.perf_counter() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(0.5)
    return False


def qwen3_asr_request(port, wav_path, extra_text=None, timeout_s=60):
    with open(wav_path, "rb") as handle:
        audio_b64 = base64.b64encode(handle.read()).decode("ascii")
    content = [{"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}}]
    if extra_text:
        content.append({"type": "text", "text": extra_text})
    payload = {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": content},
        ],
        "temperature": 0.0,
        "max_tokens": 256,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = json.loads(resp.read())
    elapsed_ms = (time.perf_counter() - t0) * 1000
    raw_content = body["choices"][0]["message"]["content"]
    cleaned = QWEN3_ASR_PREFIX.sub("", raw_content)
    cleaned = QWEN3_ASR_SUFFIX.sub("", cleaned).strip()
    return elapsed_ms, cleaned, raw_content


def run_qwen3_asr(port=8811):
    languages = CANDIDATE_LANGUAGES["qwen3-asr"]
    entries = clip_entries(languages)
    model_path = ASR_FAST_DIR / "models" / "qwen3-asr-0.6b" / "Qwen3-ASR-0.6B-Q8_0.gguf"
    mmproj_path = ASR_FAST_DIR / "models" / "qwen3-asr-0.6b" / "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf"
    exe = ENGINES_CPU_DIR / "llama-server.exe"

    t0 = time.perf_counter()
    proc = subprocess.Popen(
        [
            str(exe),
            "-m",
            str(model_path),
            "--mmproj",
            str(mmproj_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "-t",
            str(THREADS),
            "-ngl",
            "0",
            "-c",
            "4096",
            "--no-webui",
        ],
        cwd=str(REPO_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        if not wait_for_health(port):
            raise RuntimeError("llama-server did not become healthy in time")
        load_s = time.perf_counter() - t0

        warmup_entry = next((e for e in entries if e.category == "long"), entries[0])
        qwen3_asr_request(port, CLIPS_DIR / warmup_entry.file)

        records = []
        for entry in entries:
            wav_path = CLIPS_DIR / entry.file
            times_ms = []
            hyp_text = ""
            for _ in range(REPEATS):
                elapsed_ms, cleaned, _raw = qwen3_asr_request(port, wav_path)
                times_ms.append(elapsed_ms)
                hyp_text = cleaned
            records.append(
                {
                    "id": entry.id,
                    "language": entry.language,
                    "category": entry.category,
                    "bucket": prompt_set.duration_bucket(entry.duration_s),
                    "duration_s": entry.duration_s,
                    "reference": entry.reference,
                    "spoken": entry.text,
                    "hyp": hyp_text,
                    "times_ms": times_ms,
                }
            )

        rss_mb = process_rss_mb(proc.pid)
        return build_report("qwen3-asr", load_s, rss_mb, records)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        required=True,
        choices=["parakeet", "qwen3-asr", "omnilingual-300m", "omnilingual-1b"],
    )
    parser.add_argument("--port", type=int, default=8811)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.candidate == "parakeet":
        report = run_sherpa_transducer(
            "parakeet", ASR_FAST_DIR / "models" / "parakeet-tdt-0.6b-v3-int8"
        )
    elif args.candidate == "omnilingual-300m":
        report = run_omnilingual(
            "omnilingual-300m",
            ASR_FAST_DIR
            / "models"
            / "sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-v2-int8-2026-02-05",
        )
    elif args.candidate == "omnilingual-1b":
        report = run_omnilingual(
            "omnilingual-1b",
            ASR_FAST_DIR
            / "models"
            / "sherpa-onnx-omnilingual-asr-1600-languages-1B-ctc-v2-int8-2026-02-05",
        )
    else:
        report = run_qwen3_asr(port=args.port)

    out_path = RESULTS_DIR / f"{args.candidate}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"load_s={report['load_s']:.2f} rss_mb={report['rss_mb']:.0f}")
    print("timing_long_all:", report["timing_long_all"])
    for language, stats in report["by_language"].items():
        print(language, stats["wer_reference"], stats["wer_reference_long"], stats["timing_long"])


if __name__ == "__main__":
    sys.exit(main())
