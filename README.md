# Spells

**Talk instead of type. Private voice dictation for Windows 11.**

Hold a hotkey, speak, let go, and clean, punctuated text appears wherever your cursor is: Outlook, Word, Slack, Teams, your browser, your code editor. The AI models run on your own PC, so there is no account, no subscription and no cloud. Like Wispr Flow, but local and open source.

**[Download for Windows](https://github.com/kapidolli/spells/releases/latest/download/Spells-Online-Setup.exe)** · [All downloads](https://github.com/kapidolli/spells/releases/latest) · [Website](https://kapidolli.github.io/spells/)

## Why Spells

- **Private by design.** Your voice and your text stay on your computer. No account, no telemetry, no analytics.
- **Fast.** English and German land about a second or two after you let go, even without a graphics card. With live typing, the words appear while you are still talking.
- **Clean text, not a transcript.** The "um"s go, and "no wait, I mean Wednesday" becomes just "Wednesday".
- **Fits every app.** Casual in chat, full sentences in email, hands off your identifiers in a code editor.
- **Writes for you.** Say "an email to Marta asking for the September invoice" and get the email. Select text and say "make this shorter".
- **Knows your words.** Teach it names and jargon, add replacements and snippets.
- **Speaks your language.** English, German and Albanian, plus any other language Whisper knows.
- **Yours to keep.** Search your history on your PC, or send it to a server of your own, daily or weekly.
- **Free software.** GPL-3.0-or-later, with every line of source here.

## How it works

| | |
|---|---|
| **Hold Ctrl+Win and speak** | Spells listens while you hold the hotkey |
| **Let go** | The text is cleaned up and typed where your cursor is |
| **Tap twice** | Hands-free until you press the hotkey again |
| **Press Esc** | Nothing is typed |

Every hotkey can be changed in Settings. The [guide](docs/guide.md) covers live typing, cleanup, app profiles, writing and editing by voice, languages, vocabulary and history.

## Privacy

Speech recognition, cleanup and writing run in engines on your own PC that no other computer can reach. Spells goes online only when you ask it to: the online installer downloads the models, you check for updates, or you set up an upload to a server you run (off until you switch it on). Read [exactly what Spells does and stores](docs/privacy.md).

## Get it

Windows 11, 64-bit, a processor with AVX2 and about 5 GB of disk. A graphics card is optional and makes it faster (any Vulkan driver: NVIDIA, AMD or Intel). No admin rights needed.

The installer is not code-signed yet, so Windows SmartScreen may say "Windows protected your PC": click **More info**, then **Run anyway**. [Install options, requirements and speed](docs/install.md).

## Learn more

- [Guide](docs/guide.md): everything Spells can do
- [Privacy](docs/privacy.md): what goes where, and what stays on your disk
- [Download and install](docs/install.md): online or offline installer, requirements, speed
- [Upload to your own server](docs/upload.md): the protocol and a small receiver to start from
- [Building from source](docs/building.md): development setup, engines, installers, benchmark
- [Models and credits](docs/credits.md): the models, engines and libraries Spells uses
- [Changelog](CHANGELOG.md)

## Status and licence

Spells is a personal project, shared in case it is useful to others. It is young (version 0.x) and changes quickly. Issues and pull requests are welcome; please run the unit tests and `ruff check` first.

Copyright (C) 2026 kapidolli. Spells is free software under the [GNU General Public License](LICENSE), version 3 or later, without any warranty. The models, engines and libraries it uses keep their own licences, listed in [credits](docs/credits.md) and shipped in the `licenses` folder of every installation.
