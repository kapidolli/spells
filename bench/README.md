# Model selection benchmark (spec 18)

Picks the cleanup model and checks it against the spec 12 targets. It calls `spells.asr`,
`spells.cleanup`, `spells.cleanup_prompt`, `spells.gpu` and `spells.engines` directly, so it
measures the code the app runs. `PY` is `.venv/Scripts/python.exe`, never `python`.

## 1. Write the Albanian prompts

`bench/prompts.py` holds 30 prompts in EN, DE and SQ across the six spec 18 kinds (short, long,
filler, correction, question, silence). English and German are written; the Albanian ones are
stubs marked `TODO(owner)`, which `record.py` skips with a notice. Put what you will read aloud
in `text` with fillers in square brackets, and the clean version in `reference`.

## 2. Record

    PY bench/record.py

Enter starts, Enter stops, `r` records it again, `s` skips it, `q` ends the session. Silence
prompts record three seconds on their own. Clips land in `bench/clips/<id>.wav` with
`bench/clips/manifest.json`; an existing clip is skipped unless you pass `--redo <id>`. Also
`--language sq`, `--only <ids>`, `--device`, `--list-devices`. After each Albanian clip it asks
which fillers and correction phrase you really said, into `bench/clips/sq_vocab.json`; copy the
useful ones into `data/fillers/sq.txt` and `data/corrections/sq.txt` by hand (spec 8.1).

Record the warm-up clip once too (it replaces the 2 s clip the supervisor sends whisper, 13).
`bench/clips/` is gitignored, so never commit audio.

    PY bench/record.py --warmup-out data/warmup.wav

## 3. Fetch, run, smoke

    PY bench/run.py --fetch
    PY bench/run.py [--candidates a,b] [--clips id,id]

`--fetch` pulls every `bench_candidates` entry plus the whisper and Silero models into
`build/cache/models/` through `build/fetch.py`, recording a still-empty sha256 back into
`build/pins.json` with a warning (roughly 10 GB). A run launches the engines per candidate on the
GPU `spells.gpu.select_device` picks and puts every clip through Auto mode, locked mode, cleanup
and cleanup with the gate forced off, into `bench/reports/<date>-<candidate>.md` and
`<date>-summary.md`. Clips run in id order and Auto mode carries the last accepted language
between them as the app does (7.2), so a run is reproducible rather than following the manifest's
file order. If llama lands on its CPU build the two cleaned rows are reported invalid and the run
exits non-zero, because 8.1 skips cleanup there; `--allow-cpu-cleanup` measures them and marks
them. To exercise the script without recordings, in about a minute:

    PY bench/smoke_clips.py
    PY bench/run.py --smoke

`smoke_clips.py` speaks three English clips with Windows SAPI into `bench/clips/smoke/` (falling
back to copies of `data/warmup.wav`, with a notice), and `--smoke` runs with `ggml-tiny.bin` and
the test GGUF into the gitignored `bench/reports/smoke/`. Its numbers mean nothing.

## 4. Select

Selection is manual (spec 18 step 4). The summary lists which candidates met the targets and
prints the Albanian and German rows side by side, for judgement on meaning kept, fillers removed,
corrections applied and questions not answered. Pick the best, ties broken by lower latency; then
`PY bench/run.py --select <candidate>` writes its url, version and sha256 into `build/pins.json`
under `models.cleanup_model`, to commit with the reports.
