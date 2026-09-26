# Download and install

Every release on the [Releases page](https://github.com/kapidolli/spells/releases) has two installers. They install the same app and differ only in how the models arrive. The [Download for Windows](https://github.com/kapidolli/spells/releases/latest/download/Spells-Online-Setup.exe) link always fetches the newest online installer.

| | Online installer | Offline installer |
|---|---|---|
| Files | `Spells-Online-Setup-<version>.exe` | `Spells-Setup-<version>.exe` plus its `.bin` files |
| Download | about 100 MB, then the models (3.4 to 4.5 GB) | about 4.5 GB in total |
| Network during setup | Yes | None |
| Models | Only what your languages and hardware need | English, German and Albanian, the set for a PC without a graphics card |
| Best for | Most people | A PC without network, or keeping a complete copy |

## Online installer

Run `Spells-Online-Setup-<version>.exe`. It asks which languages you dictate in, checks whether your PC has a graphics card (one with at least 4 GB of its own memory) and tells you which writing model that gets you; with a graphics card you can pick the smaller, faster one instead. Then it downloads only the files that choice needs and checks each against its pinned SHA-256 hash. If a download fails you can retry it, carry on without it, or cancel.

The model files come from this project's [`models-v1` release](https://github.com/kapidolli/spells/releases/tag/models-v1). The two files larger than GitHub's 2 GiB per-file limit, the Gemma 4 E2B and Qwen3.5 4B models, come from the Hugging Face repositories they are published in.

## Just for you, or for everyone

Both installers start by asking how to install:

- **Install for me only** (the default) puts Spells in `%LOCALAPPDATA%\Programs\Spells` and needs no admin rights.
- **Install for all users** puts it in `C:\Program Files\Spells` for everyone who uses the PC and asks for admin rights once. Each person still has their own hotkeys, settings, history and autostart, kept in their own profile.

An update keeps the choice you made. One copy at a time: to switch, uninstall the other one first, and the installer tells you so if you forget. Adding a language later runs the online installer again, which for an installation for all users asks for admin rights again. For a silent install, `/CURRENTUSER` and `/ALLUSERS` pick the mode on the command line.

## Offline installer

The complete bundle is about 4.5 GB and GitHub accepts at most 2 GiB per file, so the offline installer is split into parts (Inno Setup disk spanning): `Spells-Setup-<version>.exe` plus `Spells-Setup-<version>-1.bin`, `-2.bin` and so on. Download the `.exe` and every `.bin` file of the same version into one folder, then run the `.exe`.

It carries the models for all three languages as chosen for a PC without a graphics card: Qwen3-ASR for English and German, Whisper for Albanian, and Gemma 4 E2B for cleanup and writing. They run on a graphics card too; the larger Qwen3.5 4B writing model comes only with the online installer.

## "Windows protected your PC"

The installers are not code-signed yet, so Windows warns about them:

- **SmartScreen** shows "Windows protected your PC". Click **More info**, then **Run anyway**.
- **Smart App Control.** On a PC where Smart App Control is on, Windows blocks unsigned apps such as Spells outright, and there is no Run anyway. Signing is being worked on.
- **Antivirus.** Some antivirus programs are suspicious of Spells because it watches the keyboard for your hotkey and types into other windows, which is exactly what it is for. The source is all here, and each release lists the SHA-256 checksums of its files so you can check that what you downloaded is what was published.

## Uninstall

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
