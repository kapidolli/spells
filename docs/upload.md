# Uploading to your own server

Spells can send your dictation history to a server you run, so you can keep it, search it or analyse it somewhere else. It is off by default, and nothing is sent until you switch it on and type an address in Settings > Upload. This page describes exactly what Spells sends, so you can write the receiving end in whatever language you like. The protocol is called `spells-upload/1`; nothing in it is tied to a particular server.

## On the Upload page

| Setting | What it does |
|---|---|
| Upload my dictations to a server | The switch. Off by default |
| Server address | The `http://` or `https://` address Spells posts to, exactly as you type it. An `http://` address is sent unencrypted, and the page says so |
| Token | Sent as `Authorization: Bearer <token>`. Kept in `%LOCALAPPDATA%\Spells\upload-token.bin`, encrypted for your Windows account with DPAPI, never in `settings.json` |
| When to upload | Manual, daily or weekly. Daily and weekly run in the background, never during a dictation |
| Include the recordings | Also sends the recordings Spells keeps, as WAV files. Switching it on also sends again every dictation that was sent without its recording and still has one. Spells keeps recordings only with "Keep the recordings" on the History page |
| Never upload from these apps | Process names such as `keepassxc.exe`, compared without regard to case. Dictations into them are never sent |
| Name of this computer | Sent with every request so the server can tell your computers apart. Empty means the name Windows gives the computer |

**Test connection** sends an empty batch and shows the answer. **Upload now** sends what is waiting straight away, whatever the schedule.

The first upload sends every dictation still in your history, oldest first. After that, only new dictations go, plus any whose on-demand quality check finished after they were sent. Changing the address sends everything still on the computer again, to the new server.

While uploading is on, dictations not sent yet are kept for up to 30 days, even past your history limit (the last 100, or 7 or 30 days), and so are their recordings. Past 30 days the usual limits apply; a dictation removed before it could be sent is counted, and the count goes with the next request as `lost`.

## The protocol: spells-upload/1

### Request

`POST` to the configured address, exactly as typed.

| Header | Value |
|---|---|
| `Authorization` | `Bearer <token>` when a token is set; no header at all when it is empty |
| `Content-Type` | `application/json; charset=utf-8` |
| `User-Agent` | `Spells/<version>` |

The body is one JSON object:

```json
{
  "format": "spells-upload/1",
  "sentAt": "2026-09-26T21:04:05.120+02:00",
  "app": { "name": "Spells", "version": "0.6.0" },
  "device": { "installId": "5f0c7d2e-8b1a-4c1e-9d55-3b8f0a6e2c41", "name": "DESKTOP-1" },
  "lost": 0,
  "entries": [
    {
      "key": "5f0c7d2e-8b1a-4c1e-9d55-3b8f0a6e2c41:123",
      "id": 123,
      "createdAt": "2026-09-26T20:01:02.345+02:00",
      "mode": "dictate",
      "language": "en",
      "app": { "process": "slack.exe", "title": "Slack | general" },
      "text": { "raw": "...", "cleaned": "...", "delivered": "..." },
      "instruction": "",
      "selectionChars": 0,
      "usedLlm": true,
      "cleanupReason": "",
      "outcome": "delivered",
      "engines": { "asr": "qwen3-asr", "cleanup": "gemma" },
      "timings": { "release_to_transcript_ms": 812.0, "extra": { "audio_s": 6.2 } },
      "signals": { "audio_s": 6.2, "word_count": 17, "words_per_minute": 164.5, "filler_count": 1, "correction_count": 0 },
      "quality": { "label": "", "reason": "", "checkVerdict": "", "checkReason": "", "checkedAt": null },
      "audio": {
        "format": "wav", "sampleRate": 16000, "channels": 1, "sampleWidth": 2,
        "seconds": 6.2, "bytes": 198444, "sha256": "<hex of the WAV file>", "data": "<base64 of the WAV file>"
      }
    }
  ]
}
```

- `installId` is a random UUID kept in the history database. It changes only when that database is deleted. `key` is `<installId>:<id>` and never repeats for one installation, so it is the natural primary key on the server.
- `createdAt` is local time with its offset and milliseconds. `checkedAt` has the same form, or is `null` when the transcript was never checked.
- `mode` is `dictate`, `compose` (the write hotkey) or `edit` (the edit hotkey). For `compose` and `edit`, `instruction` is what you asked for and `text.delivered` is what was written; `selectionChars` is the length of the selection an edit replaced.
- `text.raw` is what the speech model heard, `text.cleaned` what cleanup made of it, and `text.delivered` what was typed or pasted.
- `timings` and `signals` are the objects Spells stores, with their field names. More fields may appear in later versions; treat unknown fields as opaque.
- `audio` is `null` when the dictation has no kept recording, when recordings are not included, or after a 413. In that last case the entry also carries `"audioSkipped": "too large"`.
- `lost` counts dictations removed on the computer before they could be sent, since the last request the server accepted.
- `entries: []` is a valid request. Test connection sends one, and a scheduled run with nothing new sends one too.

### Batches

A request carries at most 200 entries and at most 8 MB of JSON; a single entry larger than that goes on its own. Entries go oldest first. Spells waits up to 120 seconds for an answer.

### Response

Spells reads only the status code, plus the `error` field of a 400.

| Status | Spells does |
|---|---|
| 2xx | The whole batch is stored: every entry in it counts as sent, `lost` starts again from 0, and the next batch goes |
| 400 | Stops the run and shows the `error` field of a JSON body (its first 200 characters), or the status |
| 401, 403 | Stops and shows "The server refused the token". Scheduled uploads pause until you change the address or the token, or press Upload now |
| 413 | Splits the batch in half and tries again. A single entry that still gets 413 is sent once more without its audio, then Spells moves on and tries it again next time |
| 429, 5xx, no answer, timeout | Stops the run; the next hourly tick tries again |
| Anything else, including a redirect | Stops the run like a 5xx |

Spells never follows a redirect: a `POST` that turned into a `GET` could look like success without storing anything, and the token would go along to wherever the redirect points. Type the final address instead.

Read the whole request body before you answer, even when you refuse it. A server that answers and closes the connection while Spells is still sending makes Spells see a network error instead of your status, so a 413 would never be split.

### Storing idempotently

Store entries by `key`, replacing what you have (an upsert). The same entry can arrive again: after a later quality check, after a failed run, when the address changes, or with its recording when "Include the recordings" is switched on. A re-send without audio should keep audio stored earlier.

## A minimal receiver

This one uses only the Python standard library. It checks the token, refuses bodies over 32 MB, and keeps each entry as a JSON file and each recording as a WAV file, in a folder per installation:

```python
import base64, hmac, json, os, re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TOKEN = os.environ["SPELLS_TOKEN"].encode()
STORE = Path(os.environ.get("SPELLS_STORE", "spells-uploads"))
LIMIT = 32 * 1024 * 1024
INSTALL_ID = re.compile(r"[A-Za-z0-9-]{8,64}")


class Receiver(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        if length > LIMIT:
            return self.reply(413, {"error": "too large"})
        if not hmac.compare_digest(self.headers.get("Authorization", "").encode(), b"Bearer " + TOKEN):
            return self.reply(401, {"error": "unknown token"})
        batch = json.loads(body)
        install_id = str(batch.get("device", {}).get("installId", ""))
        if batch.get("format") != "spells-upload/1" or not INSTALL_ID.fullmatch(install_id):
            return self.reply(400, {"error": "not a spells-upload/1 batch"})
        folder = STORE / install_id
        folder.mkdir(parents=True, exist_ok=True)
        for entry in batch["entries"]:
            name = str(int(entry["id"]))
            audio = entry.get("audio") or {}
            if audio.get("data"):
                (folder / f"{name}.wav").write_bytes(base64.b64decode(audio.pop("data")))
            (folder / f"{name}.json").write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
        self.reply(200, {"ok": True, "stored": len(batch["entries"])})

    def reply(self, status, answer):
        data = json.dumps(answer).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


ThreadingHTTPServer(("0.0.0.0", 8080), Receiver).serve_forever()
```

Start it with a long random token, then give Spells the same token and the address `http://<that computer>:8080/`:

```powershell
$env:SPELLS_TOKEN = "a-long-random-token"
py receiver.py
```

Writing the entry again with the same name is the upsert, and a re-send without audio leaves the WAV file from before in place. It speaks plain `http`, so run it on your own network or behind a reverse proxy that adds `https`. A real server would also check the size and hash of each recording against `audio.bytes` and `audio.sha256`, and write files under a temporary name before moving them into place.
