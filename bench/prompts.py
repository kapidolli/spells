"""The benchmark prompt set and the clip manifest, shared by record.py and run.py (spec 18).

Why a Python module and not prompts.json: record.py and run.py need the same small helpers
(the Prompt shape, the filler marker convention, placeholder detection, the duration bucket of
spec 12, the manifest reader and writer), and the brief allows exactly one prompts file. A
module carries the data and those helpers together, and it keeps the notes for the Albanian
entries next to the entries themselves, which JSON cannot do.

Prompt fields:

- ``id``      stable, used as the clip file name (bench/clips/<id>.wav).
- ``language``"en", "de" or "sq" (the codes whisper.cpp uses, spec 7.2).
- ``category``one of CATEGORIES, the spec 18 step 1 kinds. It is the semantic kind of the
              prompt and is not the same axis as the duration bucket of spec 12, which comes
              from the measured length of the recording (duration_bucket below).
- ``text``    what to read aloud. A word in square brackets is a filler to speak as written,
              for example "[um]"; the brackets are only a cue for the reader.
- ``reference`` the intended clean text: fillers dropped, self-corrections applied. This is
              what the raw transcript is scored against (spec 18 step 3) and what the cleaned
              output is judged against by hand.

The Albanian prompts mirror the English and German ones in standard written Albanian with a
Kosovo register. record.py still skips a prompt whose text is a TODO(owner) placeholder and says
so. The silence prompts need no language, so they are real prompts in all three languages.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

CATEGORIES = ("short", "long", "filler", "correction", "question", "silence")
LANGUAGES = ("en", "de", "sq")
PLACEHOLDER_MARK = "TODO(owner)"

# Spec 12: short is under 2 s, long is 10 to 30 s; everything else is reported but never
# counted against a target.
SHORT_MAX_S = 2.0
LONG_MIN_S = 10.0
LONG_MAX_S = 30.0

# Silence clips record this long with nothing spoken (spec 18 step 1).
SILENCE_SECONDS = 3.0

SILENCE_TEXT = "Say nothing at all. The recorder runs on its own for three seconds."


@dataclass(frozen=True)
class Prompt:
    id: str
    language: str
    category: str
    text: str
    reference: str


@dataclass(frozen=True)
class ManifestEntry:
    """One recorded clip in bench/clips/manifest.json."""

    id: str
    language: str
    category: str
    reference: str
    text: str
    duration_s: float
    recorded_at: str
    file: str


# The warm-up clip is not part of the scored set: its recording replaces data/warmup.wav, the
# 2 s speech clip the engine supervisor sends to whisper after launch (spec 13).
WARMUP_PROMPT = Prompt(
    id="warmup",
    language="en",
    category="short",
    text="This is a short microphone check, nothing more.",
    reference="This is a short microphone check, nothing more.",
)


PROMPTS: tuple[Prompt, ...] = (
    # ------------------------------------------------------------------ English
    Prompt(
        id="en-short-1",
        language="en",
        category="short",
        text="Sounds good, see you at six.",
        reference="Sounds good, see you at six.",
    ),
    Prompt(
        id="en-short-2",
        language="en",
        category="short",
        text="Push it to main.",
        reference="Push it to main.",
    ),
    Prompt(
        id="en-long-1",
        language="en",
        category="long",
        text=(
            "Hi Marta, thanks for sending the draft over. I read it on the train this morning "
            "and I only have two small comments, both on page four. Could we go through them "
            "on Thursday at half past two? I will book the small meeting room."
        ),
        reference=(
            "Hi Marta, thanks for sending the draft over. I read it on the train this morning "
            "and I only have two small comments, both on page four. Could we go through them "
            "on Thursday at half past two? I will book the small meeting room."
        ),
    ),
    Prompt(
        id="en-long-2",
        language="en",
        category="long",
        text=(
            "Add a comment above the retry loop explaining that the backoff is one, two, five, "
            "ten and then thirty seconds, repeating, and that the counter resets once the "
            "engine reports ready. Then run docker compose up minus d and tail the logs."
        ),
        reference=(
            "Add a comment above the retry loop explaining that the backoff is one, two, five, "
            "ten and then thirty seconds, repeating, and that the counter resets once the "
            "engine reports ready. Then run docker compose up minus d and tail the logs."
        ),
    ),
    Prompt(
        id="en-long-3",
        language="en",
        category="long",
        text=(
            "Shopping list for Saturday: two litres of milk, a kilo of flour, six eggs, olive "
            "oil and a bag of coffee. Then drop the parcel at Bahnhofstrasse 14, 8001 Zurich, "
            "before five, and text me when it is done."
        ),
        reference=(
            "Shopping list for Saturday: two litres of milk, a kilo of flour, six eggs, olive "
            "oil and a bag of coffee. Then drop the parcel at Bahnhofstrasse 14, 8001 Zurich, "
            "before five, and text me when it is done."
        ),
    ),
    Prompt(
        id="en-filler-1",
        language="en",
        category="filler",
        text=(
            "So [um] I was thinking we could [uh] move the standup to nine fifteen, because "
            "[you know] half the team is still on the train at nine."
        ),
        reference=(
            "So I was thinking we could move the standup to nine fifteen, because half the "
            "team is still on the train at nine."
        ),
    ),
    Prompt(
        id="en-filler-2",
        language="en",
        category="filler",
        text=(
            "Right, [erm] the deploy failed again, and [um] I think it is the same certificate "
            "problem we had in July, [I mean] the one where the runner cannot reach the "
            "registry. Can you [uh] check the token scopes before you run it again?"
        ),
        reference=(
            "Right, the deploy failed again, and I think it is the same certificate problem we "
            "had in July, the one where the runner cannot reach the registry. Can you check "
            "the token scopes before you run it again?"
        ),
    ),
    Prompt(
        id="en-correction-1",
        language="en",
        category="correction",
        text=(
            "Send the invoice to Arben on Tuesday, no wait, sorry I mean send it to Marta on "
            "Wednesday, she handles the billing now."
        ),
        reference="Send the invoice to Marta on Wednesday, she handles the billing now.",
    ),
    Prompt(
        id="en-question-1",
        language="en",
        category="question",
        text="What is the capital of Albania, and how many people live in Prishtina these days?",
        reference=(
            "What is the capital of Albania, and how many people live in Prishtina these days?"
        ),
    ),
    Prompt(
        id="en-silence-1",
        language="en",
        category="silence",
        text=SILENCE_TEXT,
        reference="",
    ),
    # ------------------------------------------------------------------- German
    Prompt(
        id="de-short-1",
        language="de",
        category="short",
        text="Passt, bis später.",
        reference="Passt, bis später.",
    ),
    Prompt(
        id="de-short-2",
        language="de",
        category="short",
        text="Schick mir das bitte.",
        reference="Schick mir das bitte.",
    ),
    Prompt(
        id="de-long-1",
        language="de",
        category="long",
        text=(
            "Hallo Frau Berger, vielen Dank für die schnelle Rückmeldung. Ich habe den Entwurf "
            "heute Morgen gelesen und hätte nur zwei Anmerkungen, beide auf Seite vier. Können "
            "wir das am Donnerstag um halb drei kurz durchgehen? Ich reserviere den kleinen "
            "Besprechungsraum."
        ),
        reference=(
            "Hallo Frau Berger, vielen Dank für die schnelle Rückmeldung. Ich habe den Entwurf "
            "heute Morgen gelesen und hätte nur zwei Anmerkungen, beide auf Seite vier. Können "
            "wir das am Donnerstag um halb drei kurz durchgehen? Ich reserviere den kleinen "
            "Besprechungsraum."
        ),
    ),
    Prompt(
        id="de-long-2",
        language="de",
        category="long",
        text=(
            "Schreib bitte einen Kommentar über die Warteschleife, dass der Backoff eine, "
            "zwei, fünf, zehn und dann dreißig Sekunden beträgt und sich danach wiederholt. "
            "Anschließend führst du git pull und npm run build aus und schaust in die "
            "Logdatei."
        ),
        reference=(
            "Schreib bitte einen Kommentar über die Warteschleife, dass der Backoff eine, "
            "zwei, fünf, zehn und dann dreißig Sekunden beträgt und sich danach wiederholt. "
            "Anschließend führst du git pull und npm run build aus und schaust in die "
            "Logdatei."
        ),
    ),
    Prompt(
        id="de-long-3",
        language="de",
        category="long",
        text=(
            "Der Termin mit Herrn Krasniqi steht jetzt am elften März um neun Uhr dreißig, und "
            "zwar in der Seestrasse 27 in 8002 Zürich. Bitte trag das im Kalender ein und sag "
            "Arben Bescheid, er fährt am Vortag nach Prishtina."
        ),
        reference=(
            "Der Termin mit Herrn Krasniqi steht jetzt am elften März um neun Uhr dreißig, und "
            "zwar in der Seestrasse 27 in 8002 Zürich. Bitte trag das im Kalender ein und sag "
            "Arben Bescheid, er fährt am Vortag nach Prishtina."
        ),
    ),
    Prompt(
        id="de-filler-1",
        language="de",
        category="filler",
        text=(
            "[Also] [ähm] ich glaube, wir sollten das Standup auf viertel nach neun "
            "verschieben, weil [halt] die Hälfte vom Team um neun noch im Zug sitzt."
        ),
        reference=(
            "Ich glaube, wir sollten das Standup auf viertel nach neun verschieben, weil die "
            "Hälfte vom Team um neun noch im Zug sitzt."
        ),
    ),
    Prompt(
        id="de-filler-2",
        language="de",
        category="filler",
        text=(
            "[Äh] der Deploy ist schon wieder fehlgeschlagen, und [ähm] ich glaube, es ist "
            "dasselbe Zertifikatsproblem wie im Juli, [quasi] das, wo der Runner die Registry "
            "nicht erreicht. Kannst du [äh] bitte die Token-Berechtigungen prüfen, bevor du es "
            "noch einmal startest?"
        ),
        reference=(
            "Der Deploy ist schon wieder fehlgeschlagen, und ich glaube, es ist dasselbe "
            "Zertifikatsproblem wie im Juli, das, wo der Runner die Registry nicht erreicht. "
            "Kannst du bitte die Token-Berechtigungen prüfen, bevor du es noch einmal startest?"
        ),
    ),
    Prompt(
        id="de-correction-1",
        language="de",
        category="correction",
        text=(
            "Schick die Rechnung am Dienstag an Arben, nein warte, ich meine, schick sie am "
            "Mittwoch an Marta, sie macht jetzt die Buchhaltung."
        ),
        reference="Schick die Rechnung am Mittwoch an Marta, sie macht jetzt die Buchhaltung.",
    ),
    Prompt(
        id="de-question-1",
        language="de",
        category="question",
        text=(
            "Wie viele Einwohner hat Zürich eigentlich, und wie weit ist es von dort nach "
            "Prishtina?"
        ),
        reference=(
            "Wie viele Einwohner hat Zürich eigentlich, und wie weit ist es von dort nach "
            "Prishtina?"
        ),
    ),
    Prompt(
        id="de-silence-1",
        language="de",
        category="silence",
        text=SILENCE_TEXT,
        reference="",
    ),
    # ----------------------------------------------------------------- Albanian
    Prompt(
        id="sq-short-1",
        language="sq",
        category="short",
        text="Mirë, shihemi në gjashtë.",
        reference="Mirë, shihemi në gjashtë.",
    ),
    Prompt(
        id="sq-short-2",
        language="sq",
        category="short",
        text="Ma dërgo, të lutem.",
        reference="Ma dërgo, të lutem.",
    ),
    Prompt(
        id="sq-long-1",
        language="sq",
        category="long",
        text=(
            "Përshëndetje Vjosa, faleminderit që ma dërgove dokumentin. E lexova sot në mëngjes "
            "në tren dhe kam vetëm dy vërejtje të vogla, që të dyja në faqen katër. A mund t'i "
            "kalojmë bashkë të enjten në orën dy e gjysmë? Unë e rezervoj sallën e vogël të "
            "takimeve."
        ),
        reference=(
            "Përshëndetje Vjosa, faleminderit që ma dërgove dokumentin. E lexova sot në mëngjes "
            "në tren dhe kam vetëm dy vërejtje të vogla, që të dyja në faqen katër. A mund t'i "
            "kalojmë bashkë të enjten në orën dy e gjysmë? Unë e rezervoj sallën e vogël të "
            "takimeve."
        ),
    ),
    Prompt(
        id="sq-long-2",
        language="sq",
        category="long",
        text=(
            "Takimi me zotin Krasniqi tani është më njëmbëdhjetë mars në orën nëntë e tridhjetë, "
            "në rrugën Nëna Terezë 27 në Prishtinë. Të lutem shënoje në kalendar dhe lajmëroje "
            "Arbenin, sepse ai udhëton për në Cyrih një ditë më herët."
        ),
        reference=(
            "Takimi me zotin Krasniqi tani është më njëmbëdhjetë mars në orën nëntë e tridhjetë, "
            "në rrugën Nëna Terezë 27 në Prishtinë. Të lutem shënoje në kalendar dhe lajmëroje "
            "Arbenin, sepse ai udhëton për në Cyrih një ditë më herët."
        ),
    ),
    Prompt(
        id="sq-long-3",
        language="sq",
        category="long",
        text=(
            "Të lutem shto një koment mbi retry loop-in, që backoff-i është një, dy, pesë, dhjetë "
            "e pastaj tridhjetë sekonda dhe përsëritet. Pastaj bëj git pull, nise npm run build "
            "dhe shiko logun nëse deploy-i kalon pa error."
        ),
        reference=(
            "Të lutem shto një koment mbi retry loop-in, që backoff-i është një, dy, pesë, dhjetë "
            "e pastaj tridhjetë sekonda dhe përsëritet. Pastaj bëj git pull, nise npm run build "
            "dhe shiko logun nëse deploy-i kalon pa error."
        ),
    ),
    Prompt(
        id="sq-filler-1",
        language="sq",
        category="filler",
        text=(
            "[Ëë] mendova që [domethanë] ta shtyjmë takimin e mëngjesit në nëntë e një çerek, "
            "sepse [a e di] gjysma e ekipit në nëntë është ende në tren."
        ),
        reference=(
            "Mendova që ta shtyjmë takimin e mëngjesit në nëntë e një çerek, sepse gjysma e "
            "ekipit në nëntë është ende në tren."
        ),
    ),
    Prompt(
        id="sq-filler-2",
        language="sq",
        category="filler",
        text=(
            "Deploy-i [ëhm] dështoi përsëri, dhe [pra] mendoj që është i njëjti problem me "
            "certifikatën si në korrik, [si me thënë] ai kur runner-i nuk e arrin registry-n. "
            "A mundesh [ëë] t'i kontrollosh të drejtat e token-it para se ta nisësh prapë?"
        ),
        reference=(
            "Deploy-i dështoi përsëri, dhe mendoj që është i njëjti problem me certifikatën si "
            "në korrik, ai kur runner-i nuk e arrin registry-n. A mundesh t'i kontrollosh të "
            "drejtat e token-it para se ta nisësh prapë?"
        ),
    ),
    Prompt(
        id="sq-correction-1",
        language="sq",
        category="correction",
        text=(
            "Dërgoja faturën Arbenit të martën, jo prit, desha të them dërgoja Vjosës të "
            "mërkurën, se ajo merret me kontabilitetin tani."
        ),
        reference="Dërgoja faturën Vjosës të mërkurën, se ajo merret me kontabilitetin tani.",
    ),
    Prompt(
        id="sq-question-1",
        language="sq",
        category="question",
        text=(
            "Sa banorë ka Tirana sot, dhe sa zgjat rruga me makinë nga Prishtina deri në "
            "Shkodër?"
        ),
        reference=(
            "Sa banorë ka Tirana sot, dhe sa zgjat rruga me makinë nga Prishtina deri në "
            "Shkodër?"
        ),
    ),
    Prompt(
        id="sq-silence-1",
        language="sq",
        category="silence",
        text=SILENCE_TEXT,
        reference="",
    ),
)


def by_id(prompt_id: str) -> Prompt | None:
    """The prompt with this id, the warm-up included, or None."""
    for prompt in (*PROMPTS, WARMUP_PROMPT):
        if prompt.id == prompt_id:
            return prompt
    return None


def is_placeholder(prompt: Prompt) -> bool:
    """True while the text is still a TODO(owner) stub, so record.py must skip the prompt."""
    return PLACEHOLDER_MARK in prompt.text or not prompt.text.strip()


def spoken_text(prompt: Prompt) -> str:
    """What is actually said aloud: the filler markers dropped, the filler words kept.

    Silence prompts speak nothing, so their spoken text is empty. This is a secondary
    reference for filler and self-correction prompts, whose ``reference`` deliberately differs
    from what the microphone hears.
    """
    if prompt.category == "silence":
        return ""
    return prompt.text.replace("[", "").replace("]", "")


def duration_bucket(duration_s: float) -> str:
    """The spec 12 latency bucket of a recording: "short", "long" or "other".

    Short is under 2 s and long is 10 to 30 s inclusive; a clip between or beyond those is
    reported as "other" and never counted against a target.
    """
    if duration_s < SHORT_MAX_S:
        return "short"
    if LONG_MIN_S <= duration_s <= LONG_MAX_S:
        return "long"
    return "other"


def write_manifest(path: Path, entries: list[ManifestEntry]) -> None:
    """Write bench/clips/manifest.json, sorted by id, atomically."""
    payload = {
        "version": 1,
        "clips": [asdict(entry) for entry in sorted(entries, key=lambda e: e.id)],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_manifest(path: Path) -> list[ManifestEntry]:
    """Read a manifest written by write_manifest; [] when the file does not exist."""
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        ManifestEntry(
            id=str(clip["id"]),
            language=str(clip["language"]),
            category=str(clip["category"]),
            reference=str(clip.get("reference", "")),
            text=str(clip.get("text", "")),
            duration_s=float(clip.get("duration_s", 0.0)),
            recorded_at=str(clip.get("recorded_at", "")),
            file=str(clip.get("file", f"{clip['id']}.wav")),
        )
        for clip in payload.get("clips", [])
    ]
