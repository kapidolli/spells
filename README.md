# Spells

**Private voice dictation for Windows 11.** Hold a hotkey, speak, let go, and clean text appears at your cursor. Everything runs on your own PC.

**[Download for Windows](https://github.com/kapidolli/spells/releases/latest/download/Spells-Online-Setup.exe)** (online installer, about 100 MB) · [All downloads](https://github.com/kapidolli/spells/releases/latest) · [Website](https://kapidolli.github.io/spells/)

Spells is a dictation app that works in any Windows program: Outlook, Word, Slack, Teams, a browser, a code editor. Hold the hotkey and talk. When you release it, Spells turns your speech into text, tidies it up (the "um"s go, and "no wait, I mean Wednesday" becomes just "Wednesday") and types the result where your cursor is. Speech recognition and cleanup run in local AI models on your processor or graphics card. There is no account, no subscription and no cloud service: your voice and your text never leave your computer. If you know cloud dictation tools such as Wispr Flow, this is the same idea, done locally and open source.

- English, German and Albanian, plus any other language Whisper knows
- Hold to talk, or tap twice for hands-free
- Words appear while you are still speaking
- A second hotkey writes for you: say what you want, get an email
- Runs on the processor; a graphics card makes it faster
- Install it just for you without admin rights, or for everyone on the PC
- Free software (GPL-3.0-or-later)

## Contents

- [What it does](#what-it-does)
- [Privacy](#privacy)
- [Download and install](#download-and-install)
- [System requirements](#system-requirements)
- [Models and credits](#models-and-credits)
- [Building from source](#building-from-source)
- [Project status](#project-status)
- [Licence](#licence)

## What it does

### Dictate with a hotkey

The default hotkey is **Ctrl+Win**. You can record any other combination in Settings > General.

| You do | Spells does |
|---|---|
| Hold the hotkey, speak, release | Records while you hold it, then inserts the text |
| Tap the hotkey twice quickly | Keeps recording hands-free (the pill shows a lock) until you press the hotkey again |
| Press Esc while recording | Throws the recording away; nothing is typed |
| Tap it once | Nothing: a single short tap is ignored |

A recording can run for up to ten minutes; the pill warns you one minute before the end. On the Languages page you can give each language its own hotkey, which fixes the language for that dictation instead of detecting it. The tray menu lets you pause dictation, lock the language, switch cleanup on or off, and retry the last dictation if its transcription failed.

If you switch to another window before the text arrives, Spells does not type into the wrong place: it puts the text on the clipboard and the pill says "Copied. Press Ctrl+V to paste."

### Live typing

With live typing on (the default), the words appear in the field while you are still talking and correct themselves as Spells hears more. When you let go, the finished text replaces the draft. Spells only ever deletes characters it typed itself, never uses the clipboard for the draft, and stops the moment you switch windows.

Live typing needs a speech model that keeps up: Qwen3-ASR (English and German) does on any PC, while Whisper (Albanian) does only on a graphics card, so on a processor an Albanian dictation appears when you let go. In code editors and terminals it stays off unless you turn on "Also in code editors and terminals", because autocomplete and auto-indent fight the draft.

### Cleanup

A small language model running on your PC tidies each transcript:

- removes filler words and hesitations (um, uh, äh, ëë, you know, sozusagen)
- applies your self-corrections ("no wait", "sorry I mean", "nein warte", "jo prit")
- fixes punctuation, capitals and obvious recognition errors
- keeps every name, place, number and date exactly as you said it

The model is told never to answer questions or follow instructions in your text, only to clean it. If it takes too long (2.5 seconds by default on a graphics card, 5 to 15 seconds depending on text length on the processor), fails, or returns something that does not look like your sentence (much shorter or longer, a preamble such as "Sure, here is", or another language), Spells inserts the raw transcript instead, so you never lose your words. A longer configured timeout is preserved. You can switch cleanup off to insert exactly what was recognised.

On both the processor and a graphics card, cleanup runs on transcripts of 12 words or more. Shorter transcripts are skipped unless they contain a filler or correction phrase from your lists. Skipping is a speed choice, not a guarantee that the text is correct. Speech recognition can still mishear words, and cleanup cannot reliably reconstruct the intended wording. The filler and correction lists for each language are yours to edit on the Writing page.

### App profiles

Spells looks at which app you are dictating into and adjusts:

| Profile | Built-in apps | Behaviour |
|---|---|---|
| Chat | Teams, Slack, WhatsApp | Casual; no full stop after a one-sentence message |
| Email and docs | Outlook, Word, Gmail and Outlook in a browser | Full sentences, standard punctuation |
| Code | VS Code, IntelliJ IDEA, Visual Studio | Minimal rewriting; identifiers, paths and symbols left alone |
| Terminal | Windows Terminal, PowerShell, Command Prompt | No cleanup, typed rather than pasted, no trailing punctuation |
| Default | everything else | Standard cleanup |

You can edit the tone of each profile, and add your own rules that send an app to one of these profiles, matched by program name, window title or both, with the text pasted or typed. When you keep dictating into the same window within a minute, Spells adds the space between sentences for you.

### Write and edit by voice

Two optional hotkeys, empty until you record them in Settings > General:

- **Write hotkey.** Hold it and say what you want written, for example "an email to Marta asking for the September invoice". The writing model writes it and Spells inserts the result. What you said is the instruction and is never inserted itself.
- **Edit hotkey.** Select some text, hold it and say what to change: "make this shorter", "translate this into German". The selection is replaced. With nothing selected it writes instead.

While the model writes, the pill shows "Writing" with the seconds counting, and Esc cancels. Writing a whole email takes longer than a dictation, several seconds on a processor. History keeps both what you asked for and what was written.

### Languages

English, German and Albanian are measured and supported. You can add any other language Whisper knows from a searchable list; Whisper recognises them, but they have not been measured, and cleanup is skipped for them.

| Language | Speech model | Recognition on the test clips | Cleanup |
|---|---|---|---|
| English | Qwen3-ASR 0.6B | about 8 words in 100 wrong | Excellent |
| German | Qwen3-ASR 0.6B | about 12 words in 100 wrong | Excellent to very good |
| Albanian | Whisper large-v3 turbo | about 52 words in 100 wrong | Cautious: often leaves Albanian fillers in (Gemma 4 E2B); better at corrections but now and then changes a word (Qwen3.5 4B) |
| Others | Whisper large-v3 turbo | not measured | Skipped |

Be warned about Albanian: it works, but expect to correct a fair amount. Moving Albanian to the fast speech model is planned. On a PC with a graphics card, as soon as any language other than English and German is enabled, Whisper serves English and German as well (about 13 and 15 words in 100 wrong), because one engine for everything is fast enough there.

These figures are provisional. They come from the project's own small benchmark (synthetic speech plus real Albanian speakers from Mozilla Common Voice), not from a large public test set. The Languages page in Settings shows which model Spells picked for each of your languages, and why.

### Vocabulary

- **Terms.** Names and jargon you use. They steer recognition toward the right spelling and tell the cleanup model how to write them.
- **Replacements.** Find and replace applied to every transcript, such as "git hub" to "GitHub".
- **Snippets.** Say a trigger on its own ("signature") and Spells inserts the longer text you saved for it.

### History

Spells keeps your recent dictations on your PC: the last 100 by default, or the last 7 or 30 days, or you switch history off. You can search them, copy or delete one, clear everything, and export as JSON, CSV or a readable Markdown log. Each row has a quality badge, and the page shows your speaking statistics (median words per minute, fillers per 100 words, how often cleanup changed the text). "Check this transcript" asks the local model whether a transcript reads like real speech, for when a dictation looks wrong.

If you want, Spells can also keep the recording of each dictation so you can play it back. That is off by default and has its own limit (the newest 200 recordings within 1000 MB, both adjustable).

### The pill

A small pill at the bottom of the screen you are dictating on shows what is happening: a level meter that moves with your voice, a lock when you are hands-free, a spinner while it works, "Starting engines" while the models load, "Writing" with a timer, and short notices when something needs your attention. It never takes focus from the window you are typing into.

### Graphics card or processor

Spells runs the models on your processor, or on any graphics card with a Vulkan driver (NVIDIA, AMD or Intel; no CUDA needed).

- **Automatic choice.** At start-up Spells lists your graphics devices, prefers a graphics card of its own over graphics built into the processor, and picks the models that suit your languages on that hardware. The Languages page explains the choice, and the Diagnostics page lets you override the device.
- **Integrated graphics are measured, not guessed.** On a PC whose only graphics are built into the processor (such as Intel UHD or Iris Xe, the Intel Arc graphics of Core Ultra laptops, or AMD Radeon integrated graphics), Spells measures once, in the background on the first start, whether the graphics or the processor transcribes a built-in sample clip faster. The graphics are used only if they are at least 20% faster and transcribe the sample correctly. The result is remembered.
- **Fallback.** If the graphics build of an engine fails to start, Spells falls back to the processor build, so dictation keeps working.
- **Hybrid processors.** On a processor with performance and efficiency cores, the engines are pinned to the performance cores.
- **Free the memory.** An idle timer can unload the models after a number of minutes; they load again when you dictate. It is off by default.

### Everything else

- Starts with Windows (on by default; switch it off in Settings > General).
- A welcome tour on the first start, with a microphone test.
- Settings follow the Windows light or dark theme.
- A Diagnostics page with engine state, speed of the last dictations, the log folder and a diagnostics bundle you can save.
- An About page with the licences, a Check for updates button and an optional weekly check (see [Privacy](#privacy)).

## Privacy

**Your voice and your text never leave your computer.** Speech recognition, cleanup and writing run in engines that Spells starts itself and that listen only on 127.0.0.1, the loopback address that no other computer can reach. Spells talks to them directly, never through a proxy server, even when Windows or an environment variable has one configured. Spells has no account, no telemetry, no analytics and no crash reporting.

The only network traffic Spells can cause is this, and you start both:

1. **The online installer downloading the models.** It downloads the model files your languages and hardware need, from this project's GitHub Releases and from Hugging Face, and checks each one against a SHA-256 hash pinned into the installer. If you would rather have no network at all, use the offline installer: it carries its models and never connects.
2. **Checking for updates, when you ask.** The About page has a Check for updates button, and you can switch on a weekly check, which is off by default and never runs during a dictation. A check fetches one small version file, `latest.json`, from this project's latest GitHub release. It sends nothing but the request itself, with the user agent `Spells/<version>`: no identifier, no settings, no text. If there is a newer version and you press Install this version, Spells downloads that installer from the same server and checks it against the SHA-256 in the version file; a file that does not match is deleted unrun, and only a matching one is offered for installing. An update fetches the installer only, never your models. As with any web request, the server sees your IP address.

That covers Spells itself. Windows and other software on your PC (SmartScreen, Windows Error Reporting, your antivirus) behave as they always do.

### What stays on your disk

| What | Where | What it holds |
|---|---|---|
| Settings | `%APPDATA%\Spells\settings.json` | Hotkeys, languages, app rules, your vocabulary, filler lists, tones. No dictated text |
| Speed measurements | `%APPDATA%\Spells\calibration.json` | Only on integrated graphics: model names, the graphics device name and how fast each ran on the built-in sample clip |
| History | `%LOCALAPPDATA%\Spells\history.db` | Your recent dictations (last 100 by default): the text as recognised, cleaned and inserted, the app and window title, language, timings. Set it to off, or clear it, any time |
| Recordings | `%LOCALAPPDATA%\Spells\recordings\` | Only if you turn on "Keep the recordings" (off by default) |
| Logs | `%LOCALAPPDATA%\Spells\logs\` | App and engine logs, rotated at 5 MB, five files per log. See below |

- **Audio.** Your recording is held in memory for the current dictation and sent to the local speech engine over loopback. Spells writes it to disk only if you turn on "Keep the recordings". After a failed transcription it stays in memory until you retry or start the next dictation.
- **Logs.** At the default level the logs hold no dictated text: timings, model and device names, errors, and now and then the title of a window (when live typing stops because you switched windows). Debug logging, off by default on the Diagnostics page, can include transcript text. The diagnostics bundle is a zip of the logs and the settings file, never the history or recordings, saved where you choose and sent nowhere.
- **Other windows.** Spells reads the program name and title of the window you dictate into, to pick the app profile and to make sure the text goes back to the same window. It does not read what is in your windows, with one exception you trigger yourself: the edit hotkey copies your selection.
- **Clipboard.** To paste, Spells keeps a copy of your clipboard in memory, puts the text on the clipboard, presses Ctrl+V and a moment later puts your own clipboard back. Text that Spells places on the clipboard is marked so Windows leaves it out of clipboard history and cloud clipboard sync. Live typing never touches the clipboard. The edit hotkey copies your selection with Ctrl+C just as you would, and your previous clipboard is restored afterwards; because the app you are in makes that copy, Windows clipboard history and cloud sync, if you use them, treat it like any other copy.
- **Keyboard.** Spells installs a keyboard hook to notice your hotkeys. It reacts only to your hotkeys and to Esc while recording, and never logs or stores what you type.

When you uninstall, Spells asks whether to remove your settings, history and logs as well. They stay unless you tick the box.

## Download and install

Every release on the [Releases page](https://github.com/kapidolli/spells/releases) has two installers. They install the same app and differ only in how the models arrive. The [Download for Windows](https://github.com/kapidolli/spells/releases/latest/download/Spells-Online-Setup.exe) link always fetches the newest online installer.

| | Online installer | Offline installer |
|---|---|---|
| Files | `Spells-Online-Setup-<version>.exe` | `Spells-Setup-<version>.exe` plus its `.bin` files |
| Download | about 100 MB, then the models (3.4 to 4.5 GB) | about 4.5 GB in total |
| Network during setup | Yes | None |
| Models | Only what your languages and hardware need | English, German and Albanian, the set for a PC without a graphics card |
| Best for | Most people | A PC without network, or keeping a complete copy |

### Online installer

Run `Spells-Online-Setup-<version>.exe`. It asks which languages you dictate in, checks whether your PC has a graphics card (one with at least 4 GB of its own memory) and tells you which writing model that gets you; with a graphics card you can pick the smaller, faster one instead. Then it downloads only the files that choice needs and checks each against its pinned SHA-256 hash. If a download fails you can retry it, carry on without it, or cancel.

The model files come from this project's [`models-v1` release](https://github.com/kapidolli/spells/releases/tag/models-v1). The two files larger than GitHub's 2 GiB per-file limit, the Gemma 4 E2B and Qwen3.5 4B models, come from the Hugging Face repositories they are published in.

### Just for you, or for everyone

Both installers start by asking how to install:

- **Install for me only** (the default) puts Spells in `%LOCALAPPDATA%\Programs\Spells` and needs no admin rights.
- **Install for all users** puts it in `C:\Program Files\Spells` for everyone who uses the PC and asks for admin rights once. Each person still has their own hotkeys, settings, history and autostart, kept in their own profile.

An update keeps the choice you made. One copy at a time: to switch, uninstall the other one first, and the installer tells you so if you forget. Adding a language later runs the online installer again, which for an installation for all users asks for admin rights again. For a silent install, `/CURRENTUSER` and `/ALLUSERS` pick the mode on the command line.

### Offline installer

The complete bundle is about 4.5 GB and GitHub accepts at most 2 GiB per file, so the offline installer is split into parts (Inno Setup disk spanning): `Spells-Setup-<version>.exe` plus `Spells-Setup-<version>-1.bin`, `-2.bin` and so on. Download the `.exe` and every `.bin` file of the same version into one folder, then run the `.exe`.

It carries the models for all three languages as chosen for a PC without a graphics card: Qwen3-ASR for English and German, Whisper for Albanian, and Gemma 4 E2B for cleanup and writing. They run on a graphics card too; the larger Qwen3.5 4B writing model comes only with the online installer.

### "Windows protected your PC"

The installers are not code-signed yet, so Windows warns about them:

- **SmartScreen** shows "Windows protected your PC". Click **More info**, then **Run anyway**.
- **Smart App Control.** On a PC where Smart App Control is on, Windows blocks unsigned apps such as Spells outright, and there is no Run anyway. Signing is being worked on.
- **Antivirus.** Some antivirus programs are suspicious of Spells because it watches the keyboard for your hotkey and types into other windows, which is exactly what it is for. The source is all here, and each release lists the SHA-256 checksums of its files so you can check that what you downloaded is what was published.

### Uninstall

Settings > Apps > Installed apps > Spells > Uninstall. It asks whether to remove your settings, history and logs too.

## System requirements

| | |
|---|---|
| Windows | Windows 11, 64-bit (x64). Spells is built and tested on Windows 11 only; Windows 10 is not a target and has not been tried. Windows on ARM is not supported |
| Account | A standard user account; no admin rights needed to install for yourself. Installing for all users of the PC asks for them once |
| Disk | 3.8 to 4.8 GB once installed, depending on your languages and hardware. Spells keeps every installation at or under 5 GB (5,000,000,000 bytes) |
| Memory | Not measured yet. As a rough guide, the models Spells keeps loaded add up to 3.4 to 4.5 GB |
| Processor | A 64-bit Intel or AMD processor with AVX2, which the Whisper engine needs (most processors from the last ten years; some low-end Pentium and Celeron chips lack it). The speed figures below are from an Intel Core i7-14700K |
| Graphics | Optional. Any graphics card with a Vulkan driver makes it faster |
| Microphone | Any input device Windows sees |

Spells cannot dictate into windows running as administrator: Windows does not let a normal program's hotkey or typing reach them.

### How fast

Measured on an Intel Core i7-14700K without its graphics card, with the engines pinned to its performance cores (8 threads), for a dictation of about 15 seconds. The graphics card column is an estimate, not a measurement: the shipped models have not been benchmarked on a graphics card yet.

| Step | Model | Processor (measured) | Graphics card (estimate) |
|---|---|---|---|
| English speech to text | Qwen3-ASR 0.6B | 1.1 s | 0.4 s |
| German speech to text | Qwen3-ASR 0.6B | 1.2 s | 0.4 s |
| Albanian speech to text | Whisper large-v3 turbo | about 4.5 s | 0.3 s |
| Cleanup, when there is something to clean | Gemma 4 E2B | 1.0 s (0.7 s for English) | 0.4 s |
| Cleanup | Qwen3.5 4B | 2.7 s | 0.9 s |

Times run from the moment you let go. On a processor an English or German dictation lands in roughly one to two seconds, depending on whether cleanup runs; Albanian takes around five. Where live typing runs, most of the text is already on screen before you let go.

## Models and credits

Spells stands on the work of others. Every third-party licence text ships in the `licenses` folder of each installation and is shown on the About page.

### Models

| Model | Published by | Licence | Used for |
|---|---|---|---|
| Qwen3-ASR 0.6B, Q4_K_M | Qwen ([Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)) | Apache-2.0 | Speech recognition for English and German |
| Qwen3-ASR 0.6B audio projector, Q8_0 | ggml-org ([Qwen3-ASR-0.6B-GGUF](https://huggingface.co/ggml-org/Qwen3-ASR-0.6B-GGUF)) | Apache-2.0 | The audio encoder Qwen3-ASR needs |
| Whisper large-v3 turbo, q8_0 | OpenAI; ggml conversion by [whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp) | MIT | Speech recognition for Albanian and other languages |
| Silero VAD v5.1.2 | Silero; ggml conversion by [ggml-org/whisper-vad](https://huggingface.co/ggml-org/whisper-vad) | MIT | Finding the speech in a recording for Whisper |
| Gemma 4 E2B, Q4_0, with its MTP draft head | Google; GGUF by [ggml-org](https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF) | Apache-2.0 | Cleanup and writing on a PC without a graphics card, and in the offline installer |
| Qwen3.5 4B, Q4_K_M | Qwen; GGUF by [Unsloth](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF) | Apache-2.0 | Cleanup and writing on a PC with a graphics card |

Nobody publishes Qwen3-ASR 0.6B at Q4_K_M, so the file Spells uses is this project's own quantisation of [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) (Apache-2.0), made from the bf16 weights with llama.cpp b10997 (`bench/convert_qwen3_asr.py`). It recognises as well as the Q8_0 file and is 320 MB smaller.

### Engines and libraries

| Component | Licence | Role |
|---|---|---|
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp) v1.9.4, with ggml | MIT | `whisper-server`, built from source for Vulkan and for the processor |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) b10997 | MIT | `llama-server`, the official Windows Vulkan and processor builds |
| cpp-httplib 0.20.0, nlohmann/json 3.11.2 | MIT | Compiled into `whisper-server` |
| GCC runtime (libgcc, libstdc++) | GPLv3 with the GCC Runtime Library Exception 3.1 | Statically linked into `whisper-server` |
| winpthreads | MIT and BSD-3-Clause | Statically linked into `whisper-server` |
| LLVM OpenMP | Apache-2.0 with LLVM exception | `libomp.dll` beside `llama-server` |
| Python 3.12 | PSF License | The app runtime |
| PySide6 and Qt 6.11 | LGPLv3 | The user interface |
| sounddevice 0.5.6 and PortAudio | MIT, MIT-style | Microphone capture |
| PyInstaller 6.22.3 bootloader | GPL-2.0-or-later with the bootloader exception | Starts the packaged app |

The installers are built with [Inno Setup](https://jrsoftware.org/isinfo.php). The Vulkan loader comes with your graphics driver and is not shipped.

## Building from source

The build runs as a normal user: no admin rights, no registry changes, nothing added to `PATH`, and apart from Python itself everything it downloads stays inside the repository folder. You need Windows 11 x64, Git, the Python install manager (`py`) and a network connection. Commands are for PowerShell.

### Set up and test

```powershell
git clone https://github.com/kapidolli/spells.git
cd spells
py install 3.12
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -e .[dev]
.venv/Scripts/python.exe -m pytest
```

The unit tests need no graphics card, no network and no engines. The integration tests (`.venv/Scripts/python.exe -m pytest -m integration`) need the engines and models below, and some of them take over the keyboard focus and the clipboard for a few minutes, so run them on an unlocked desktop you are not using.

### Build the engines

```powershell
py build/bootstrap_toolchain.py
py build/build_whisper.py
```

`bootstrap_toolchain.py` downloads a pinned MSYS2 UCRT64 toolchain and Kitware's signed CMake into `build/toolchain` and proves the chain works; `--offline` installs from package files already cached in `build/cache`. `build_whisper.py` builds `whisper-server` from the pinned whisper.cpp tag, once for Vulkan and once for the processor, and unpacks the pinned llama.cpp release builds beside it into `build/out/engines`. It takes `--variant vulkan|cpu|all`, `--no-llama` and `--jobs N`.

### Fetch the models and run

```powershell
py build/fetch.py models.whisper_large_v3_turbo_q8_0 models.silero_vad models.qwen3_asr_0_6b_q4_k_m models.qwen3_asr_0_6b_mmproj models.cleanup_model models.cleanup_model_mtp_draft
.venv/Scripts/python.exe -m spells
```

`build/fetch.py` downloads pinned inputs from `build/pins.json` into `build/cache` and refuses any file whose SHA-256 does not match; `--all` fetches every pinned input. Run from a checkout, Spells uses the engines in `build/out/engines` and the models in `build/cache/models`.

### Build the installers

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

### Benchmark

```powershell
py bench/synth_clips.py
py bench/run.py --hardware cpu
```

`bench/synth_clips.py` builds the benchmark clips without a microphone: English spoken by Windows' own voices, German and Albanian by Piper voices, plus real Albanian speakers from Common Voice. The German and Albanian parts need the `piper-tts` package and the voices `de_DE-thorsten-high` and `sq_AL-edon-medium` in `build/cache/voices`, and the Common Voice clips need the Albanian Common Voice metadata in `build/cache/commonvoice-sq`; `--languages en` needs none of these. `bench/run.py --hardware cpu|gpu` runs the app's own configuration over the clips and writes a report to `bench/reports`; `--smoke` runs a quick check with tiny test models. `bench/record.py` records your own voice instead.

## Project status

Spells is a personal project, shared in case it is useful to others. It is young (version 0.x) and changes quickly. Issues and pull requests are welcome; please run the unit tests and `.venv/Scripts/python.exe -m ruff check .` before sending a pull request. There are no support promises and no release schedule.

Planned next: a signed installer, Albanian on the fast speech model, and reading the few characters before the cursor so spacing and capitals follow the text already there. [CHANGELOG.md](CHANGELOG.md) lists what changed in each release.

## Licence

Copyright (C) 2026 kapidolli

Spells is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed in the hope that it will be useful, but without any warranty; see the [GNU General Public License](LICENSE) for details.

The models, engines and libraries Spells uses keep their own licences, listed above. Their full texts are in the `licenses` folder of every installation and in `build/licenses` in this repository.
