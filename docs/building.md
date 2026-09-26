# Building from source

The build runs as a normal user: no admin rights, no registry changes, nothing added to `PATH`, and apart from Python itself everything it downloads stays inside the repository folder. You need Windows 11 x64, Git, the Python install manager (`py`) and a network connection. Commands are for PowerShell.

## Set up and test

```powershell
git clone https://github.com/kapidolli/spells.git
cd spells
py install 3.12
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -e .[dev]
.venv/Scripts/python.exe -m pytest
```

The unit tests need no graphics card, no network and no engines. The integration tests (`.venv/Scripts/python.exe -m pytest -m integration`) need the engines and models below, and some of them take over the keyboard focus and the clipboard for a few minutes, so run them on an unlocked desktop you are not using.

## Build the engines

```powershell
py build/bootstrap_toolchain.py
py build/build_whisper.py
```

`bootstrap_toolchain.py` downloads a pinned MSYS2 UCRT64 toolchain and Kitware's signed CMake into `build/toolchain` and proves the chain works; `--offline` installs from package files already cached in `build/cache`. `build_whisper.py` builds `whisper-server` from the pinned whisper.cpp tag, once for Vulkan and once for the processor, and unpacks the pinned llama.cpp release builds beside it into `build/out/engines`. It takes `--variant vulkan|cpu|all`, `--no-llama` and `--jobs N`.

## Fetch the models and run

```powershell
py build/fetch.py models.whisper_large_v3_turbo_q8_0 models.silero_vad models.qwen3_asr_0_6b_q4_k_m models.qwen3_asr_0_6b_mmproj models.cleanup_model models.cleanup_model_mtp_draft
.venv/Scripts/python.exe -m spells
```

`build/fetch.py` downloads pinned inputs from `build/pins.json` into `build/cache` and refuses any file whose SHA-256 does not match; `--all` fetches every pinned input. Run from a checkout, Spells uses the engines in `build/out/engines` and the models in `build/cache/models`.

## Build the installers

```powershell
py build/bootstrap_packaging.py
py build/package.py --all
py build/package.py --online
```

`bootstrap_packaging.py` unpacks the pinned Inno Setup compiler with innoextract (the Inno Setup installer is never run) and installs the pinned PyInstaller into `.venv`. `package.py --all` runs the unit tests, freezes the app, assembles `dist/Spells` and builds the offline installer, `dist/Spells-Setup-<version>.exe`, splitting it into `.bin` slices when it is too large for one file (and zipping them into `dist/Spells-Setup-<version>.zip`); the models it ships are downloaded and hash-checked on the way if they are not cached yet. `--online` builds `dist/Spells-Online-Setup-<version>.exe` instead, plus a copy named `Spells-Online-Setup.exe`, which every release carries so that the download link on the website always gets the newest version.

| Flag | What it does |
|---|---|
| `--step tests\|app\|dist\|installer\|size\|latest` | Run one step; repeat it to run several, in order |
| `--span` | Force the split installer even when it would fit in one file |
| `--cleanup-model <path>` | Ship another GGUF as the cleanup model; the build is marked `-dev` |
| `--model-mirror <base url>` | With `--online`: download the models from this folder first, with the pinned addresses as the fallback |
| `--update-source <base url>` | Fix the address this build checks for updates, and write `dist/latest.json` for it. Without it the build cannot check for updates |
| `--minimum-version <version>` | With `--update-source`: the oldest version that can update to this one directly |
| `--signtool "<command with $f>"` | Sign every unsigned `.exe`, `.dll` and `.pyd` in the app folder, the installer and the uninstaller with this command (or set `SPELLS_SIGNTOOL`). `build/sign.ps1 $f` is a ready command for a code signing certificate in your user certificate store, named by `SPELLS_SIGN_THUMBPRINT` |
| `--test-install [--installer <path>]` | Install, upgrade and uninstall silently on this machine, check the result, and put the machine back as it was. A check that cannot run here counts as skipped and the run exits with 2 |

## Benchmark

```powershell
py bench/synth_clips.py
py bench/run.py --hardware cpu
```

`bench/synth_clips.py` builds the benchmark clips without a microphone: English spoken by Windows' own voices, German and Albanian by Piper voices, plus real Albanian speakers from Common Voice. The German and Albanian parts need the `piper-tts` package and the voices `de_DE-thorsten-high` and `sq_AL-edon-medium` in `build/cache/voices`, and the Common Voice clips need the Albanian Common Voice metadata in `build/cache/commonvoice-sq`; `--languages en` needs none of these. `bench/run.py --hardware cpu|gpu` runs the app's own configuration over the clips and writes a report to `bench/reports`; `--smoke` runs a quick check with tiny test models. `bench/record.py` records your own voice instead.
