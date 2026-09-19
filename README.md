# Spells

Spells is a local, offline, per-user Windows 11 desktop app for voice dictation in the style of Wispr Flow. Hold a hotkey, speak English, German, or Albanian, release, and cleaned-up text appears at the cursor in whatever app has focus. Speech recognition (Whisper) and text cleanup (a small LLM) run entirely on the local GPU. There is no subscription, no network access, no admin requirement, and no manual setup: a single installer contains everything.

## Development setup

```bash
py install 3.12
py -3.12 -V
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -e .[dev]
```

## Running tests

Unit tests (no GPU, integration tests are skipped by default):

```bash
.venv/Scripts/python.exe -m pytest
```

Engine integration tests (need a GPU and the engine binaries):

```bash
.venv/Scripts/python.exe -m pytest -m integration
```
