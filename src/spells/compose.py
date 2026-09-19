"""Compose and edit mode: the transcript is an instruction, not text (spec 8.5, B5-64 to B5-71).

A compose dictation records exactly like a normal one and skips cleanup entirely. The transcript
goes to the writing model with the fixed system prompt of `compose_prompt`, temperature 0 and
`cache_prompt` so that prompt stays cached beside the cleanup one, and whatever comes back is
delivered by the normal delivery path of spec 11. In edit mode the selection copied from the
target window travels with it and the answer replaces the selection, which is still selected.

Three parts live here and nothing else does:

* the request, with a timeout and a completion bound of its own (an instruction is short, the
  text it asks for is not, so the cleanup bound of 8.2 would cut every answer off);
* the guards of 8.5, which are not the cleanup guards: a composed email is far longer than the
  instruction and "translate this into German" changes the language on purpose, so the length
  ratio and the language switch are gone and a refusal check and an echo check take their place.
  A rejection delivers nothing, because delivering the raw instruction would be worse;
* the selection read, which sends Ctrl+C through `win32.input` under the clipboard snapshot and
  sequence number rules of spec 11 and puts the user's clipboard back exactly as it was.

The clipboard is read as material, never as instructions: what comes back is a block in the user
message and the system prompt says so.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request

from spells.cleanup import starts_with_preamble
from spells.compose_prompt import system_prompt, user_message
from spells.loopback import urlopen
from spells.models import ChordMode, ComposeResult
from spells.textutil import normalize_for_match, word_count

log = logging.getLogger(__name__)

COMPOSE_TIMEOUT_MS = 60_000

COMPOSE_MIN_TOKENS = 256
COMPOSE_MAX_TOKENS = 1024
TOKENS_PER_INSTRUCTION_WORD = 8
TOKENS_PER_SELECTION_WORD = 2
COMPOSE_TOKEN_SLACK = 64

ECHO_LENGTH_PERCENT = 120

SELECTION_WAIT_S = 0.4
SELECTION_POLL_S = 0.02

WRITING_TEXT = "Writing"
ELAPSED_AFTER_S = 2.0
NOTHING_SELECTED_TEXT = "Nothing selected"

REFUSAL_OPENINGS = (
    "i cannot",
    "i can not",
    "i can't",
    "i am unable",
    "i'm unable",
    "i will not",
    "i won't",
    "i am not able",
    "i'm not able",
    "as an ai",
    "as a language model",
    "as an assistant",
    "i am sorry",
    "i'm sorry",
    "sorry i cannot",
    "sorry i can't",
    "unfortunately i cannot",
    "unfortunately i can't",
    "sure here is",
    "sure here's",
    "ich kann nicht",
    "ich kann leider",
    "als ki",
    "als sprachmodell",
    "es tut mir leid",
    "leider kann ich",
    "nuk mund",
    "me vjen keq",
    "më vjen keq",
)

# One short line each: the pill caps at 320 px and elides past it (spec 14.2), so every
# line here is measured to fit whole. The log carries the detail.
REJECTION_TEXT = {
    "engine_not_ready": "Writing model not ready",
    "no_instruction": "Nothing was said",
    "timeout": "Writing timed out",
    "error": "Writing failed",
    "cancelled": "Writing cancelled",
    "finish_length": "The text was cut off",
    "empty": "Nothing was written",
    "preamble": "The model answered",
    "refusal": "The model refused",
    "echo": "The model repeated it",
}

GUARD_REASONS = ("finish_length", "empty", "preamble", "refusal", "echo")


class ComposeError(Exception):
    """A writing request failed. reason is "timeout", "error" or "cancelled"."""

    def __init__(self, message: str, reason: str = "error") -> None:
        super().__init__(message)
        self.reason = reason


class CancelHandle:
    """Closes the connection of one in-flight writing request from another thread.

    Esc cancels a composition: the response object is closed under the worker, the socket goes
    with it, and the engine sees the client hang up instead of finishing text nobody wants. A
    request started after cancel() never leaves the process.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._closer: Callable[[], None] | None = None
        self._cancelled = False

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def attach(self, response: Any) -> bool:
        """Hold the response so cancel() can close it; False when cancel() already ran."""
        with self._lock:
            if self._cancelled:
                return False
            self._closer = getattr(response, "close", None)
            return True

    def detach(self) -> None:
        with self._lock:
            self._closer = None

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True
            closer, self._closer = self._closer, None
        if closer is not None:
            try:
                closer()
            except Exception:
                log.debug("closing the writing request failed", exc_info=True)


class ComposeClient:
    """The text engine asked to write instead of to clean.

    Same server as cleanup and the same cached system prompt slot, a timeout of its own because
    writing is not on the dictation latency path, and a CancelHandle so Esc can close the
    connection mid-answer. Override chat() in tests.
    """

    def __init__(
        self,
        base_url: str,
        timeout_s: float = COMPOSE_TIMEOUT_MS / 1000.0,
        cancel: CancelHandle | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.cancel = cancel

    def chat(self, system: str, user: str, max_tokens: int) -> tuple[str, str]:
        """POST one chat completion and return (content, finish_reason). Raises ComposeError."""
        handle = self.cancel
        if handle is not None and handle.cancelled:
            raise ComposeError("the writing request was cancelled", reason="cancelled")
        payload = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "max_tokens": max_tokens,
            "stream": False,
        }
        request = Request(
            self.base_url + "/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        raw = self._send(request, handle)
        try:
            body = json.loads(raw.decode("utf-8"))
            choice = body["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
        except (UnicodeDecodeError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise ComposeError("llama-server returned an unexpected body") from exc
        return ("" if content is None else str(content), str(finish_reason or ""))

    def _send(self, request: Request, handle: CancelHandle | None) -> bytes:
        try:
            response = urlopen(request, timeout=self.timeout_s)
        except HTTPError as exc:
            raise self._failed(handle, exc, f"llama-server returned HTTP {exc.code}") from exc
        except TimeoutError as exc:
            raise self._failed(handle, exc, "the writing request timed out", "timeout") from exc
        except URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise self._failed(
                    handle, exc, "the writing request timed out", "timeout"
                ) from exc
            raise self._failed(handle, exc, f"the writing request failed: {exc.reason}") from exc
        except OSError as exc:
            raise self._failed(handle, exc, f"the writing request failed: {exc}") from exc
        try:
            if handle is not None and not handle.attach(response):
                raise ComposeError("the writing request was cancelled", reason="cancelled")
            return response.read()
        except OSError as exc:
            raise self._failed(handle, exc, f"the writing request failed: {exc}") from exc
        finally:
            if handle is not None:
                handle.detach()
            try:
                response.close()
            except Exception:
                log.debug("closing the writing response failed", exc_info=True)

    @staticmethod
    def _failed(
        handle: CancelHandle | None, exc: BaseException, message: str, reason: str = "error"
    ) -> ComposeError:
        if handle is not None and handle.cancelled:
            return ComposeError("the writing request was cancelled", reason="cancelled")
        return ComposeError(message)


def max_tokens_for(instruction: str, selection: str = "") -> int:
    """The completion bound for one writing request (spec 8.5).

    Cleanup bounds the answer by the transcript, because a cleaned sentence is about as long as
    the raw one. Composing is the opposite: "write an email to Marta asking for the September
    invoice" is ten words and the email is a hundred and fifty. So the bound starts at a floor of
    256 tokens, which covers a short email or note whatever the instruction weighs, grows by 8
    tokens per instruction word (a longer instruction asks for a longer text) and by 2 tokens per
    selected word (an edit answers at about the length of its material, with room for a
    translation that runs longer), and stops at 1024 tokens. That ceiling is roughly 750 words,
    more than any spoken instruction asks for, and it is what keeps a runaway generation from
    holding the engine and the user for minutes on a processor.
    """
    wanted = (
        TOKENS_PER_INSTRUCTION_WORD * word_count(instruction)
        + TOKENS_PER_SELECTION_WORD * word_count(selection)
        + COMPOSE_TOKEN_SLACK
    )
    return max(COMPOSE_MIN_TOKENS, min(COMPOSE_MAX_TOKENS, wanted))


def starts_with_refusal(output: str) -> bool:
    """Whether the output opens with a refusal or an assistant self-description (spec 8.5)."""
    tokens = normalize_for_match(output).split()
    for opening in REFUSAL_OPENINGS:
        needle = normalize_for_match(opening).split()
        if needle and tokens[: len(needle)] == needle:
            return True
    return False


def echoes_instruction(instruction: str, output: str) -> bool:
    """Whether the model handed the instruction back instead of carrying it out.

    Equal after normalisation, or the instruction sitting whole inside an output no more than a
    fifth longer than it. A longer answer that happens to quote the instruction is not an echo:
    a note that opens by naming what it is about is a real answer.
    """
    needle = normalize_for_match(instruction).split()
    if not needle:
        return False
    tokens = normalize_for_match(output).split()
    if tokens == needle:
        return True
    if len(tokens) * 100 > ECHO_LENGTH_PERCENT * len(needle):
        return False
    span = len(needle)
    return any(tokens[i : i + span] == needle for i in range(len(tokens) - span + 1))


def guard(instruction: str, output: str, finish_reason: str) -> str | None:
    """Spec 8.5 output guards. Returns the rejection reason, or None when the output is kept.

    Empty output, a length finish and the preamble list are the cleanup guards of 8.3, which
    still hold: an assistant opening is an assistant answering rather than writing. The length
    ratio and the language switch are dropped here on purpose. The refusal and echo checks are
    this mode's own.
    """
    if finish_reason == "length":
        return "finish_length"
    written = output.strip()
    if not written:
        return "empty"
    if starts_with_preamble(written):
        return "preamble"
    if starts_with_refusal(written):
        return "refusal"
    if echoes_instruction(instruction, written):
        return "echo"
    return None


def compose(
    instruction: str,
    client: ComposeClient | None,
    *,
    selection: str = "",
    tone: str = "",
    mode: ChordMode = ChordMode.COMPOSE,
) -> ComposeResult:
    """One writing request with its guards. Never raises; a failure delivers nothing.

    `client` is None when the writing engine is not serving. A rejected or failed result carries
    no text at all, because the only other thing to deliver would be the raw instruction.
    """
    spoken = instruction.strip()

    def result(text: str, ok: bool, reason: str) -> ComposeResult:
        return ComposeResult(
            text=text,
            ok=ok,
            reason=reason,
            mode=mode,
            instruction=spoken,
            selection_chars=len(selection),
        )

    if not spoken:
        return result("", False, "no_instruction")
    if client is None:
        return result("", False, "engine_not_ready")
    try:
        output, finish_reason = client.chat(
            system_prompt(),
            user_message(spoken, selection, tone),
            max_tokens_for(spoken, selection),
        )
    except ComposeError as exc:
        log.info("writing request failed: %s", exc.reason)
        return result("", False, exc.reason)
    rejection = guard(spoken, output, finish_reason)
    if rejection is not None:
        log.info("writing output rejected: %s", rejection)
        return result("", False, rejection)
    return result(keep_edges(selection, tidy(output, spoken)), True, "ok")


def keep_edges(selection: str, text: str) -> str:
    core = selection.strip()
    if not core or not text:
        return text
    start = selection.index(core)
    return selection[:start] + text + selection[start + len(core):]


_SUBJECT_LINE = re.compile(r"^\s*(subject|betreff|subjekti|tema)\s*:", re.IGNORECASE)
_SUBJECT_WORDS = re.compile(r"\b(subject|betreff|subjekt\w*|tem\w*)\b", re.IGNORECASE)
_PLACEHOLDER_LINE = re.compile(r"^\s*\[[^\[\]\n]{1,40}\]\s*[,.]?\s*$")


def tidy(output: str, instruction: str) -> str:
    lines = output.strip().split("\n")
    if lines and _SUBJECT_LINE.match(lines[0]) and not _SUBJECT_WORDS.search(instruction):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]
    lines = [line for line in lines if not _PLACEHOLDER_LINE.match(line)]
    text = "\n".join(lines).strip()
    return re.sub(r"\n{3,}", "\n\n", text)


def rejection_text(reason: str) -> str:
    """The one short line the pill shows when nothing was delivered (spec 14.2)."""
    return REJECTION_TEXT.get(reason, "Writing failed")


def writing_text(instruction: str = "", elapsed_s: float = 0.0) -> str:
    """The pill's writing line: the state, the seconds after two of them, the instruction.

    The pill caps at 320 px and elides, so the seconds and the state sit in front where they are
    always readable and the instruction takes whatever room is left.
    """
    head = WRITING_TEXT
    if elapsed_s >= ELAPSED_AFTER_S:
        head = f"{WRITING_TEXT} {int(elapsed_s)}s"
    said = " ".join(instruction.split())
    return f"{head}: {said}" if said else head


# The selection ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Selection:
    """What Ctrl+C brought back, and what happened to the user's clipboard.

    `copied` is False when the clipboard sequence number never moved, which is what nothing
    selected looks like from here: edit mode then falls back to composing and says so on the
    pill. `restored` is None when there was nothing to put back.
    """

    text: str = ""
    copied: bool = False
    restored: bool | None = None
    detail: str = ""

    @property
    def chars(self) -> int:
        return len(self.text)


def read_selection(
    backends: Any = None,
    *,
    wait_s: float = SELECTION_WAIT_S,
    poll_s: float = SELECTION_POLL_S,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.perf_counter,
) -> Selection:
    """Copy the target window's selection and put the user's clipboard back (spec 11, 8.5).

    Snapshot the clipboard, release any held modifiers, send Ctrl+C, then watch the clipboard
    sequence number for at most `wait_s`. A number that never moves means the target had nothing
    selected (or refused to copy), the clipboard was never touched and there is nothing to
    restore. A number that moves means the text is ours to read, and the snapshot goes back under
    the same sequence rule delivery uses, so the user's clipboard survives an edit exactly.

    Never raises: any Win32 failure reads as nothing selected, which falls back to composing.
    """
    if backends is None:
        from spells.inject import DEFAULT_BACKENDS

        backends = DEFAULT_BACKENDS
    clipboard = backends.clipboard
    try:
        snap = clipboard.snapshot()
        before = int(clipboard.sequence_number())
    except Exception as exc:
        log.warning("the clipboard could not be snapshotted for an edit", exc_info=True)
        return Selection(detail=repr(exc))
    try:
        backends.input.release_held_modifiers()
        backends.input.send_ctrl_c()
    except Exception as exc:
        log.warning("Ctrl+C could not be sent for an edit", exc_info=True)
        return Selection(detail=repr(exc))
    if not _wait_for_change(clipboard, before, wait_s, poll_s, sleeper, clock):
        log.info("nothing was selected: the clipboard did not change within %.0f ms", wait_s * 1000)
        return Selection(detail="the clipboard did not change")
    try:
        text = clipboard.get_text() or ""
    except Exception as exc:
        log.warning("the copied selection could not be read", exc_info=True)
        text = ""
        detail = repr(exc)
    else:
        detail = ""
    restored, note = _restore(clipboard, snap)
    return Selection(text=text, copied=True, restored=restored, detail=detail or note)


def _wait_for_change(
    clipboard: Any,
    before: int,
    wait_s: float,
    poll_s: float,
    sleeper: Callable[[float], None],
    clock: Callable[[], float],
) -> bool:
    """Poll the sequence number for at most `wait_s`, bounded by a poll count as well.

    Both bounds are real: the clock ends the wait on a machine under load, and the count
    ends it where the clock does not move, which is every test with a hand-driven clock and
    a sleeper that does not sleep.
    """
    step = max(poll_s, 0.001)
    polls = max(1, int(max(0.0, wait_s) / step))
    deadline = clock() + max(0.0, wait_s)
    for attempt in range(polls + 1):
        try:
            if int(clipboard.sequence_number()) != before:
                return True
        except Exception:
            log.debug("the clipboard sequence number could not be read", exc_info=True)
            return False
        if attempt >= polls or clock() >= deadline:
            return False
        sleeper(step)
    return False


def _restore(clipboard: Any, snap: Any) -> tuple[bool, str]:
    try:
        sequence = int(clipboard.sequence_number())
        restored = bool(clipboard.restore(snap, sequence))
    except Exception as exc:
        log.warning("the clipboard could not be restored after an edit", exc_info=True)
        return False, f"clipboard not restored: {exc!r}"
    if restored:
        return True, "clipboard restored"
    return False, "clipboard not restored: it changed after the copy"
