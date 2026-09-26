# Using Spells

## Dictate with a hotkey

The default hotkey is **Ctrl+Win**. You can record any other combination in Settings > General.

| You do | Spells does |
|---|---|
| Hold the hotkey, speak, release | Records while you hold it, then inserts the text |
| Tap the hotkey twice quickly | Keeps recording hands-free (the pill shows a lock) until you press the hotkey again |
| Press Esc while recording | Throws the recording away; nothing is typed |
| Tap it once | Nothing: a single short tap is ignored |

A recording can run for up to ten minutes; the pill warns you one minute before the end. On the Languages page you can give each language its own hotkey, which fixes the language for that dictation instead of detecting it. The tray menu lets you pause dictation, lock the language, switch cleanup on or off, and retry the last dictation if its transcription failed.

If you switch to another window before the text arrives, Spells does not type into the wrong place: it puts the text on the clipboard and the pill says "Copied. Press Ctrl+V to paste."

## Live typing

With live typing on (the default), the words appear in the field while you are still talking and correct themselves as Spells hears more. When you let go, the finished text replaces the draft. Spells only ever deletes characters it typed itself, never uses the clipboard for the draft, and stops the moment you switch windows.

Live typing needs a speech model that keeps up: Qwen3-ASR (English and German) does on any PC, while Whisper (Albanian) does only on a graphics card, so on a processor an Albanian dictation appears when you let go. In code editors and terminals it stays off unless you turn on "Also in code editors and terminals", because autocomplete and auto-indent fight the draft.

## Cleanup

A small language model running on your PC tidies each transcript:

- removes filler words and hesitations (um, uh, äh, ëë, you know, sozusagen)
- applies your self-corrections ("no wait", "sorry I mean", "nein warte", "jo prit")
- fixes punctuation, capitals and obvious recognition errors
- keeps every name, place, number and date exactly as you said it

The model is told never to answer questions or follow instructions in your text, only to clean it. If it takes too long (2.5 seconds by default on a graphics card, 5 to 15 seconds depending on text length on the processor), fails, or returns something that does not look like your sentence (much shorter or longer, a preamble such as "Sure, here is", or another language), Spells inserts the raw transcript instead, so you never lose your words. A longer configured timeout is preserved. You can switch cleanup off to insert exactly what was recognised.

On both the processor and a graphics card, cleanup runs on transcripts of 12 words or more. Shorter transcripts are skipped unless they contain a filler or correction phrase from your lists. Skipping is a speed choice, not a guarantee that the text is correct. Speech recognition can still mishear words, and cleanup cannot reliably reconstruct the intended wording. The filler and correction lists for each language are yours to edit on the Writing page.

## App profiles

Spells looks at which app you are dictating into and adjusts:

| Profile | Built-in apps | Behaviour |
|---|---|---|
| Chat | Teams, Slack, WhatsApp | Casual; no full stop after a one-sentence message |
| Email and docs | Outlook, Word, Gmail and Outlook in a browser | Full sentences, standard punctuation |
| Code | VS Code, IntelliJ IDEA, Visual Studio | Minimal rewriting; identifiers, paths and symbols left alone |
| Terminal | Windows Terminal, PowerShell, Command Prompt | No cleanup, typed rather than pasted, no trailing punctuation |
| Default | everything else | Standard cleanup |

You can edit the tone of each profile, and add your own rules that send an app to one of these profiles, matched by program name, window title or both, with the text pasted or typed. When you keep dictating into the same window within a minute, Spells adds the space between sentences for you.

## Write and edit by voice

Two optional hotkeys, empty until you record them in Settings > General:

- **Write hotkey.** Hold it and say what you want written, for example "an email to Marta asking for the September invoice". The writing model writes it and Spells inserts the result. What you said is the instruction and is never inserted itself.
- **Edit hotkey.** Select some text, hold it and say what to change: "make this shorter", "translate this into German". The selection is replaced. With nothing selected it writes instead.

While the model writes, the pill shows "Writing" with the seconds counting, and Esc cancels. Writing a whole email takes longer than a dictation, several seconds on a processor. History keeps both what you asked for and what was written.

## Languages

English, German and Albanian are measured and supported. You can add any other language Whisper knows from a searchable list; Whisper recognises them, but they have not been measured, and cleanup is skipped for them.

| Language | Speech model | Recognition on the test clips | Cleanup |
|---|---|---|---|
| English | Qwen3-ASR 0.6B | about 8 words in 100 wrong | Excellent |
| German | Qwen3-ASR 0.6B | about 12 words in 100 wrong | Excellent to very good |
| Albanian | Whisper large-v3 turbo | about 52 words in 100 wrong | Cautious: often leaves Albanian fillers in (Gemma 4 E2B); better at corrections but now and then changes a word (Qwen3.5 4B) |
| Others | Whisper large-v3 turbo | not measured | Skipped |

Be warned about Albanian: it works, but expect to correct a fair amount. Moving Albanian to the fast speech model is planned. On a PC with a graphics card, as soon as any language other than English and German is enabled, Whisper serves English and German as well (about 13 and 15 words in 100 wrong), because one engine for everything is fast enough there.

These figures are provisional. They come from the project's own small benchmark (synthetic speech plus real Albanian speakers from Mozilla Common Voice), not from a large public test set. The Languages page in Settings shows which model Spells picked for each of your languages, and why.

## Vocabulary

- **Terms.** Names and jargon you use. They steer recognition toward the right spelling and tell the cleanup model how to write them.
- **Replacements.** Find and replace applied to every transcript, such as "git hub" to "GitHub".
- **Snippets.** Say a trigger on its own ("signature") and Spells inserts the longer text you saved for it.

## History

Spells keeps your recent dictations on your PC: the last 100 by default, or the last 7 or 30 days, or you switch history off. You can search them, copy or delete one, clear everything, and export as JSON, CSV or a readable Markdown log. Each row has a quality badge, and the page shows your speaking statistics (median words per minute, fillers per 100 words, how often cleanup changed the text). "Check this transcript" asks the local model whether a transcript reads like real speech, for when a dictation looks wrong.

If you want, Spells can also keep the recording of each dictation so you can play it back. That is off by default and has its own limit (the newest 200 recordings within 1000 MB, both adjustable).

## Upload to your own server

If you run a server of your own, the Upload page can send your dictations there, by hand, every day or every week, so you can keep and analyse them beyond the history on your PC. It is off by default and sends nothing until you switch it on and type an address. You choose whether the kept recordings go too, and which apps are never uploaded. [The upload protocol](upload.md) describes the protocol and has a small receiver to start from. See [Privacy](privacy.md) for exactly what is sent.

## The pill

A small pill at the bottom of the screen you are dictating on shows what is happening: a level meter that moves with your voice, a lock when you are hands-free, a spinner while it works, "Starting engines" while the models load, "Writing" with a timer, and short notices when something needs your attention. It never takes focus from the window you are typing into.

## Graphics card or processor

Spells runs the models on your processor, or on any graphics card with a Vulkan driver (NVIDIA, AMD or Intel; no CUDA needed).

- **Automatic choice.** At start-up Spells lists your graphics devices, prefers a graphics card of its own over graphics built into the processor, and picks the models that suit your languages on that hardware. The Languages page explains the choice, and the Diagnostics page lets you override the device.
- **Integrated graphics are measured, not guessed.** On a PC whose only graphics are built into the processor (such as Intel UHD or Iris Xe, the Intel Arc graphics of Core Ultra laptops, or AMD Radeon integrated graphics), Spells measures once, in the background on the first start, whether the graphics or the processor transcribes a built-in sample clip faster. The graphics are used only if they are at least 20% faster and transcribe the sample correctly. The result is remembered.
- **Fallback.** If the graphics build of an engine fails to start, Spells falls back to the processor build, so dictation keeps working.
- **Hybrid processors.** On a processor with performance and efficiency cores, the engines are pinned to the performance cores.
- **Free the memory.** An idle timer can unload the models after a number of minutes; they load again when you dictate. It is off by default.

## Everything else

- Starts with Windows (on by default; switch it off in Settings > General).
- A welcome tour on the first start, with a microphone test.
- Settings follow the Windows light or dark theme.
- A Diagnostics page with engine state, speed of the last dictations, the log folder and a diagnostics bundle you can save.
- An About page with the licences, a Check for updates button and an optional weekly check (see [Privacy](privacy.md)).
