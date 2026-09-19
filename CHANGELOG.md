# Changelog

What changed in each release of Spells, in the words the app itself uses. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the versions follow
[semantic versioning](https://semver.org/spec/v2.0.0.html).

The bullet lines of a released section are also what the app shows on its About page: the
build reads them out of this file into the version file it serves beside the installers, so
the two can never say different things.

## [Unreleased]

### Planned

- Albanian on the fast speech model, once the gated model is available.
- A signed installer, so Windows stops warning about it.
- Reading the few characters before the cursor, so spacing and capitals follow the text already there.

## [0.5.0] - 2026-09-19

Install it for everyone on the PC, and a privacy fix for computers with a proxy.

### Added

- The installers ask whether to install Spells for you only, which needs no administrator rights, or for everyone who uses this PC, which asks for them once. An installation for everyone goes to Program Files, and each person keeps their own settings and history.

### Fixed

- With a proxy server set by hand in Windows or in an environment variable, Spells sent the requests to its own engines on this computer through that proxy, recordings and text included. They now always go straight to the engines and never leave the PC.
- When a model download fails, the online installer now names the offline installer as it is published: the .exe with its .bin parts.
- The live typing switch no longer says it needs a graphics card, since it runs on the processor too.

## [0.4.0] - 2026-09-19

The first public release. Spells is now open source under the GNU General Public License, version 3 or later.

### Changed

- Spells now lives on GitHub. The installers, the version file the update check reads and the model files are all published there.
- The online installer downloads the models from the project's own GitHub release, and the two largest from Hugging Face, where their publishers host them. Every file is still checked against its pinned SHA-256 hash.
- The offline installer comes as Setup.exe plus a few .bin parts, because GitHub accepts files of up to 2 GiB. Download every part into one folder and run the .exe.

## [0.3.3] - 2026-09-18

A quieter pill.

### Changed

- The pill no longer says "Typing live" while the words are typed into your window. It shows the level meter alone, and the words appearing are the sign that it is working.

## [0.3.2] - 2026-09-18

A fix for a Windows key hotkey together with live typing.

### Fixed

- With a hotkey that includes the Windows key, the Start menu no longer opens while the words are being typed live, and the text keeps going into your window instead of stopping halfway.

## [0.3.1] - 2026-09-18

The release for laptops without a graphics card of their own.

### Added

- On a computer whose graphics are built into the processor, Spells measures once whether the graphics or the processor transcribes faster, and then uses the faster one. It runs in the background on the first start and is remembered.

### Changed

- Live typing now also runs without a graphics card when the speech model can stop a pass halfway, which Qwen3-ASR can. The words arrive every second or two while you speak instead of only at the end.
- With Albanian enabled beside English or German, the words still appear live for English and German. An Albanian dictation appears when you let go, as before.
- On a computer without a graphics card, an install for Albanian together with English or German now keeps Qwen3-ASR for English and German, so those take about a second instead of five or more.
- The Qwen3-ASR model is about 320 MB smaller at the same accuracy, which is what makes it fit in the 5 GB install.

### Fixed

- A long dictation on a slow processor no longer fails with "Transcription failed" while the speech engine is still working. Spells now waits as long as the measured speed needs.

## [0.3.0] - 2026-09-18

The release where the text appears while you speak, and where a second hotkey writes for you.

### Added

- The words now appear in the field while you are still speaking, and correct themselves as Spells hears more. It types only what it typed itself, never touches your clipboard, and stops the moment you switch windows.
- Live typing runs where a speech pass is quick, which today means a computer with a graphics card. There is a switch for it on the General page.
- A writing hotkey: hold it, say what you want written, and the written text is inserted. What you said is the instruction and is never inserted itself.
- An editing hotkey: select some text, say what to do with it, and the selection is replaced. With nothing selected it writes instead, and your clipboard is put back exactly as it was.
- The writing model is chosen for your machine like the others, and the Writing page names it and says why.
- The pill shows a Writing state with the seconds counting, and Escape cancels the request.
- History tells a written row from a dictated one, keeping both what you asked for and what was written.

### Fixed

- The quality badge sits in the middle of its row in History instead of at the top.

## [0.2.0] - 2026-09-18

The release that made Spells fast on a computer without a graphics card, gave it a face,
and learned to tell you when there is a newer version.

### Added

- Spells can now tell you when a new version is out. The About page has a Check for updates button that always works, and a weekly check you can switch on. It is off to begin with, and Spells opens no connection until you ask it to.
- An update downloads the installer, checks it against the hash the release was published with, and installs it in one click. It never downloads the speech or writing models again.
- A logo and an app icon of its own, in the window, the tray and the installer.
- A Languages page. Tick the languages you dictate in and Spells picks the models for them, and says in plain words how well each one does each language and how fast it is on your computer.
- A History page: search your dictations, see how each one came out, read your speaking statistics, and export everything as JSON, as a spreadsheet file or as a readable log.
- Spells can keep the recording beside the text of each dictation and play it back. It is off by default and stays on your computer.
- A Check this transcript action that asks the writing model whether a transcript reads like real speech, for when a dictation looks wrong and you want a second opinion.

### Changed

- English and German are transcribed by Qwen3-ASR 0.6B, which is about four times faster than Whisper on a processor. Albanian and every other language keep Whisper, which is still the best of the two at them.
- Cleanup is done by Gemma 4 E2B, or by Qwen3.5 4B on a computer with a graphics card. Albanian fillers and self-corrections are now part of what the model is told to look for, and it is told to keep every name, place, number and date exactly as you said it.
- On a computer without a graphics card, cleanup is skipped when there is nothing in the transcript to clean, so most dictations arrive a second sooner.
- The settings, welcome and diagnostics windows were rebuilt around a navigation rail, with cards, switches and keycaps, in the app's own accent colour and following the light or dark theme of Windows.
- The speech and writing engines run on the performance cores of a hybrid processor, which took about 400 ms off a long English dictation.
- There is a second installer of about 97 MB that downloads the models your languages and your computer need while it installs. The full installer, which carries every model and needs no network at all, is still there for a machine that has none.

### Fixed

- The level meter moves with your voice again. It was reading raw loudness, where speech sits at a few per cent of full scale, so it looked flat while you were speaking.
- The microphone list no longer cuts off long device names.
- A microphone that delivers no sound now says so instead of reporting a failed transcription.
- Blocks of sound the card drops mid-recording are counted and shown, so a dictation that is missing words says why.

[Unreleased]: https://github.com/kapidolli/spells/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/kapidolli/spells/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/kapidolli/spells/releases/tag/v0.4.0
