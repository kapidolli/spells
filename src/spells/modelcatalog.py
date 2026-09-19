"""Model choice from the dictation languages and the hardware (spec batch 5, language picker).

The user picks the languages they dictate in; Spells assesses the models it knows for those
languages on the hardware it runs on and chooses the speech recognition model per language,
the cleanup model and the writing model compose mode uses. This module is the contract the
settings page, the engine wiring and both installers build against; the scores come from the
benchmark (spec 18) through data/models/catalog.json.

It also owns the size rule (spec 19.4, B5-45): an install is the app, both engine folders and
every model one machine needs, and it must stay at or below INSTALL_BUDGET_BYTES. plan_install
is what an installer asks for the models a machine of one hardware class needs for one set of
languages, already inside that budget.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from functools import lru_cache
from pathlib import Path

from spells.datafiles import data_path
from spells.gpu import GpuSelection

CATALOG_SCHEMA_VERSION = 1
CLEANUP_TARGET_MS = 1800
SAFETY_FLOOR = 50
ASR_MARGIN = 5
ASR_SPEED_FACTOR = 2
ASR_SPEED_MIN_SAVED_MS = 500
MAX_SPEECH_ENGINES = 2
INSTALL_BUDGET_BYTES = 5_000_000_000
BASE_INSTALL_BYTES = 320_979_273
VAD_MODEL_BYTES = 885_098
ANY_LANGUAGE = "*"
SMALL_GPU_MEMORY_MB = 4096
WHISPER_SERVER = "whisper-server"
LLAMA_ASR = "llama-asr"
LLAMA_SERVER = "llama-server"
EXTRA_FILE_PLACEHOLDER = re.compile(r"\{extra:(\d+)\}")


class Hardware(str, Enum):
    GPU = "gpu"
    CPU = "cpu"


class ModelKind(str, Enum):
    ASR = "asr"
    CLEANUP = "cleanup"
    COMPOSE = "compose"


RUNTIMES = {
    ModelKind.ASR: frozenset({WHISPER_SERVER, LLAMA_ASR}),
    ModelKind.CLEANUP: frozenset({LLAMA_SERVER}),
    ModelKind.COMPOSE: frozenset({LLAMA_SERVER}),
}


class CatalogError(ValueError):
    pass


@dataclass(frozen=True, eq=False)
class CatalogModel:
    id: str
    kind: ModelKind
    display_name: str
    file: str
    extra_files: tuple[str, ...]
    size_bytes: int
    license: str
    runtime: str
    languages: Mapping[str, int]
    hardware: Mapping[str, int | None]
    engine_args: tuple[str, ...]
    notes: str = ""
    redistributable: bool = True

    @property
    def files(self) -> tuple[str, ...]:
        return (self.file, *self.extra_files)

    @property
    def file_key(self) -> tuple[str, ...]:
        """What two entries for the same weights share, so bytes are counted once."""
        return tuple(sorted(name.lower() for name in self.files))

    def score(self, language: str) -> int | None:
        if language in self.languages:
            return self.languages[language]
        return self.languages.get(ANY_LANGUAGE)

    def measured(self, language: str) -> bool:
        return language in self.languages

    def latency_ms(self, hardware: Hardware) -> int | None:
        return self.hardware.get(hardware.value)


@dataclass(frozen=True)
class ModelChoice:
    kind: ModelKind
    model_id: str
    display_name: str
    languages: tuple[str, ...]
    reason: str
    installed: bool
    file: str = ""
    extra_files: tuple[str, ...] = ()
    engine_args: tuple[str, ...] = ()
    runtime: str = ""
    latency_ms: int | None = None
    unscored: tuple[str, ...] = ()


@dataclass(frozen=True)
class Selection:
    hardware: Hardware
    asr: tuple[ModelChoice, ...]
    cleanup: ModelChoice | None
    notes: tuple[str, ...]
    compose: ModelChoice | None = None
    integrated: bool = False

    def asr_for(self, language: str) -> ModelChoice | None:
        for choice in self.asr:
            if language in choice.languages:
                return choice
        return None

    @property
    def primary_asr(self) -> ModelChoice | None:
        return self.asr[0] if self.asr else None

    @property
    def cpu_cleanup_allowed(self) -> bool:
        cleanup = self.cleanup
        return (
            self.hardware is Hardware.CPU
            and cleanup is not None
            and cleanup.latency_ms is not None
            and cleanup.latency_ms <= CLEANUP_TARGET_MS
        )

    @property
    def one_text_model(self) -> bool:
        """True when cleanup and composing share one model, so one engine serves both."""
        cleanup, compose = self.cleanup, self.compose
        if cleanup is None or compose is None:
            return False
        return cleanup.file == compose.file and cleanup.extra_files == compose.extra_files


def default_catalog_path() -> Path:
    return data_path("models", "catalog.json")


@lru_cache(maxsize=1)
def _default_catalog() -> tuple[CatalogModel, ...]:
    return load_catalog(default_catalog_path())


def load_catalog(path: Path | None = None) -> tuple[CatalogModel, ...]:
    if path is None:
        return _default_catalog()
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise CatalogError(f"{path}: {exc}") from exc
    return parse_catalog(raw)


def parse_catalog(raw: object) -> tuple[CatalogModel, ...]:
    if not isinstance(raw, dict):
        raise CatalogError("catalog: expected an object")
    if raw.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise CatalogError(
            f"catalog: schema_version {raw.get('schema_version')!r} is not "
            f"{CATALOG_SCHEMA_VERSION}"
        )
    entries = raw.get("models")
    if not isinstance(entries, list):
        raise CatalogError("catalog.models: expected a list")
    models = tuple(_parse_model(entry, index) for index, entry in enumerate(entries))
    seen: set[str] = set()
    for model in models:
        if model.id in seen:
            raise CatalogError(f"catalog: the id {model.id!r} appears twice")
        seen.add(model.id)
    return models


def _parse_model(entry: object, index: int) -> CatalogModel:
    where = f"catalog.models[{index}]"
    if not isinstance(entry, dict):
        raise CatalogError(f"{where}: expected an object")

    def text(key: str, *, required: bool = True) -> str:
        value = entry.get(key, None if required else "")
        if not isinstance(value, str) or (required and not value):
            raise CatalogError(f"{where}.{key}: expected a non-empty string")
        return value

    def file_name(value: object, label: str) -> str:
        if not isinstance(value, str) or not value or "/" in value or "\\" in value:
            raise CatalogError(f"{where}.{label}: expected a bare file name")
        return value

    kind = text("kind")
    if kind not in {k.value for k in ModelKind}:
        raise CatalogError(f"{where}.kind: expected asr, cleanup or compose, got {kind!r}")
    extra = entry.get("extra_files", [])
    if not isinstance(extra, list):
        raise CatalogError(f"{where}.extra_files: expected a list")
    size = entry.get("size_bytes")
    if not _is_int(size) or size < 0:
        raise CatalogError(f"{where}.size_bytes: expected a non-negative integer")
    languages = entry.get("languages")
    if not isinstance(languages, dict):
        raise CatalogError(f"{where}.languages: expected an object")
    for code, score in languages.items():
        if not _is_int(score) or not 0 <= score <= 100:
            raise CatalogError(f"{where}.languages.{code}: expected an integer from 0 to 100")
    hardware = entry.get("hardware")
    if not isinstance(hardware, dict):
        raise CatalogError(f"{where}.hardware: expected an object")
    latencies: dict[str, int | None] = {}
    for key in (Hardware.GPU.value, Hardware.CPU.value):
        if key not in hardware:
            raise CatalogError(f"{where}.hardware.{key}: missing")
        value = hardware[key]
        if value is not None and (not _is_int(value) or value < 0):
            raise CatalogError(f"{where}.hardware.{key}: expected milliseconds or null")
        latencies[key] = value
    args = entry.get("engine_args", [])
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        raise CatalogError(f"{where}.engine_args: expected a list of strings")
    for arg in args:
        for named in extra_file_indexes(arg):
            if named >= len(extra):
                raise CatalogError(
                    f"{where}.engine_args: {arg!r} names extra file {named}, but the model "
                    f"has {len(extra)}"
                )
    redistributable = entry.get("redistributable", True)
    if not isinstance(redistributable, bool):
        raise CatalogError(f"{where}.redistributable: expected true or false")
    return CatalogModel(
        id=text("id"),
        kind=ModelKind(kind),
        display_name=text("display_name"),
        file=file_name(entry.get("file"), "file"),
        extra_files=tuple(
            file_name(name, f"extra_files[{position}]") for position, name in enumerate(extra)
        ),
        size_bytes=size,
        license=text("license"),
        runtime=text("runtime"),
        languages=dict(languages),
        hardware=latencies,
        engine_args=tuple(args),
        notes=text("notes", required=False),
        redistributable=redistributable,
    )


def extra_file_indexes(arg: str) -> tuple[int, ...]:
    return tuple(int(match.group(1)) for match in EXTRA_FILE_PLACEHOLDER.finditer(arg))


def resolve_extra_files(args: Sequence[str], extra_files: Sequence[Path]) -> tuple[str, ...]:
    def replace(match: re.Match[str]) -> str:
        index = int(match.group(1))
        if index >= len(extra_files):
            raise ValueError(f"{match.group(0)} names a missing extra file")
        return str(Path(extra_files[index]).absolute())

    return tuple(EXTRA_FILE_PLACEHOLDER.sub(replace, arg) for arg in args)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def installed_ids(
    models_dir: Path, catalog: Iterable[CatalogModel] | None = None
) -> frozenset[str]:
    models = load_catalog() if catalog is None else catalog
    folder = Path(models_dir)
    return frozenset(
        model.id for model in models if all(_is_file(folder / name) for name in model.files)
    )


def _is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def hardware_from_gpu(selection: GpuSelection | None) -> Hardware:
    if selection is None or selection.raw_index is None:
        return Hardware.CPU
    device = next((d for d in selection.devices if d.raw_index == selection.raw_index), None)
    uma = device.uma if device is not None else None
    if uma is False:
        return Hardware.GPU
    if uma is True:
        return Hardware.CPU
    memory = device.memory_mb if device is not None else selection.memory_mb
    return Hardware.GPU if memory >= SMALL_GPU_MEMORY_MB else Hardware.CPU


def with_latencies(
    catalog: Sequence[CatalogModel], hardware: Hardware, latencies: Mapping[str, int]
) -> tuple[CatalogModel, ...]:
    return tuple(
        replace(model, hardware={**model.hardware, hardware.value: latencies[model.id]})
        if model.id in latencies
        else model
        for model in catalog
    )


def select_models(
    enabled_languages: Sequence[str],
    hardware: Hardware,
    installed_ids: frozenset[str] | None = None,
    catalog: Sequence[CatalogModel] | None = None,
) -> Selection:
    languages = tuple(dict.fromkeys(enabled_languages))
    models = tuple(load_catalog() if catalog is None else catalog)

    def installed(model: CatalogModel) -> bool:
        return installed_ids is None or model.id in installed_ids

    asr, asr_notes = _select_asr(languages, hardware, models, installed)
    cleanup, cleanup_notes = _select_cleanup(languages, hardware, models, installed)
    compose, compose_notes = _select_compose(languages, hardware, models, installed, cleanup)
    return Selection(
        hardware=hardware,
        asr=asr,
        cleanup=cleanup,
        notes=tuple(asr_notes + cleanup_notes + compose_notes),
        compose=compose,
    )


def _runnable(models: Sequence[CatalogModel], kind: ModelKind) -> list[CatalogModel]:
    return [model for model in models if model.kind is kind and model.runtime in RUNTIMES[kind]]


def _choice(
    model: CatalogModel,
    languages: Sequence[str],
    reason: str,
    installed: bool,
    hardware: Hardware,
    unscored: Sequence[str] = (),
) -> ModelChoice:
    return ModelChoice(
        kind=model.kind,
        model_id=model.id,
        display_name=model.display_name,
        languages=tuple(languages),
        reason=reason,
        installed=installed,
        file=model.file,
        extra_files=model.extra_files,
        engine_args=model.engine_args,
        runtime=model.runtime,
        latency_ms=model.latency_ms(hardware),
        unscored=tuple(unscored),
    )


def _select_asr(languages, hardware, models, installed):
    runnable = [m for m in _runnable(models, ModelKind.ASR) if m.latency_ms(hardware) is not None]
    available = [m for m in runnable if installed(m)]
    order = {model.id: index for index, model in enumerate(models)}
    near: dict[str, list[CatalogModel]] = {}
    best: dict[str, int] = {}
    uncovered: list[str] = []
    for language in languages:
        scored = [(m.score(language), m) for m in available if m.score(language) is not None]
        if not scored:
            uncovered.append(language)
            continue
        best[language] = max(score for score, _ in scored)
        near[language] = [m for score, m in scored if score >= best[language] - ASR_MARGIN]

    remaining = [language for language in languages if language in near]
    groups = _group_by_coverage(remaining, near, hardware, order)
    groups, speed_notes = _split_for_speed(groups, near, hardware, order)

    choices = []
    notes: list[str] = []
    for model, covered, faster_than in groups:
        compromise = any((model.score(lang) or 0) < best[lang] for lang in covered)
        reason = _asr_reason(model, covered, hardware, alone=len(available) == 1,
                             compromise=compromise and len(covered) > 1, faster_than=faster_than)
        choices.append(_choice(model, covered, reason, True, hardware))
        notes.extend(_asr_better_notes(model, covered, hardware, runnable, installed))
    notes.extend(speed_notes)
    notes.extend(_asr_uncovered_notes(uncovered, runnable, installed))
    return tuple(choices), notes


def _group_by_coverage(remaining, near, hardware, order):
    groups: list[tuple[CatalogModel, list[str]]] = []
    remaining = list(remaining)
    while remaining:
        candidates = {m.id: m for language in remaining for m in near[language]}

        def rank(model: CatalogModel, remaining=remaining):
            covered = [lang for lang in remaining if model in near[lang]]
            return (
                -len(covered),
                -sum(model.score(lang) or 0 for lang in covered),
                model.latency_ms(hardware),
                model.size_bytes,
                order[model.id],
            )

        chosen = min(candidates.values(), key=rank)
        covered = [language for language in remaining if chosen in near[language]]
        groups.append((chosen, covered))
        remaining = [language for language in remaining if language not in covered]
    return groups


def _meaningfully_faster(fast_latency, slow_latency):
    if fast_latency is None or slow_latency is None:
        return False
    return (
        slow_latency >= fast_latency * ASR_SPEED_FACTOR
        and slow_latency - fast_latency >= ASR_SPEED_MIN_SAVED_MS
    )


def _split_for_speed(groups, near, hardware, order):
    if len(groups) != 1 or MAX_SPEECH_ENGINES < 2:
        return [(model, covered, None) for model, covered in groups], []
    model, covered = groups[0]
    model_latency = model.latency_ms(hardware)
    considered: dict[str, CatalogModel] = {}
    for language in covered:
        for alt in near[language]:
            if alt.id != model.id:
                considered.setdefault(alt.id, alt)

    best = None
    best_rank = None
    skip_notes: list[str] = []
    for alt in considered.values():
        served = [language for language in covered if alt in near[language]]
        if not served:
            continue
        alt_latency = alt.latency_ms(hardware)
        if alt_latency is None or model_latency is None or alt_latency >= model_latency:
            continue
        if _meaningfully_faster(alt_latency, model_latency):
            rank = (
                -len(served),
                -sum(alt.score(lang) or 0 for lang in served),
                alt_latency,
                alt.size_bytes,
                order[alt.id],
            )
            if best_rank is None or rank < best_rank:
                best_rank, best = rank, (alt, served)
        elif len(served) == len(covered):
            skip_notes.append(
                f"{alt.display_name} would also recognise {_names(served)} but is not fast "
                f"enough here to justify a second engine."
            )

    if best is None:
        return [(model, covered, None)], skip_notes
    fast_model, fast_languages = best
    remainder = [language for language in covered if language not in fast_languages]
    if not remainder:
        return [(fast_model, covered, model)], []
    ordered = sorted(
        [(fast_model, fast_languages, model), (model, remainder, None)],
        key=lambda item: (-len(item[1]), order[item[0].id]),
    )
    return ordered, []


def _times_word(times):
    words = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
             9: "nine", 10: "ten"}
    return words.get(times, str(times))


def _speed_clause(model, other, hardware):
    fast = model.latency_ms(hardware) or 1
    slow = other.latency_ms(hardware) or 0
    times = max(ASR_SPEED_FACTOR, round(slow / fast))
    return f"about {_times_word(times)} times faster than {other.display_name} here"


def _asr_reason(model, languages, hardware, *, alone, compromise, faster_than=None):
    measured = [lang for lang in languages if model.measured(lang)]
    unmeasured = [lang for lang in languages if not model.measured(lang)]
    sentences = []
    if measured:
        clause = _quality_phrase(model, measured, _asr_word)
        if faster_than is not None:
            sentences.append(
                f"Recognises {clause}, and is {_speed_clause(model, faster_than, hardware)}."
            )
        else:
            sentences.append(f"Recognises {clause}.")
    if unmeasured:
        sentences.append(f"Not measured yet for {_names(unmeasured)}.")
    if alone:
        sentences.append("It is the only speech model installed.")
    elif compromise:
        sentences.append("One speech engine for all of them saves memory and start time.")
    sentences.append(
        f"About {_seconds(model.latency_ms(hardware))} to transcribe a 15 s dictation "
        f"{_on(hardware)}."
    )
    return " ".join(sentences)


def _asr_better_notes(chosen, languages, hardware, runnable, installed):
    notes = []
    for model in runnable:
        if model.id == chosen.id or installed(model):
            continue
        better = [
            lang
            for lang in languages
            if model.measured(lang)
            and (model.score(lang) or 0) > (chosen.score(lang) or 0) + ASR_MARGIN
        ]
        if better:
            notes.append(
                f"{model.display_name} would recognise {_names(better)} better "
                "but is not installed."
            )
    return notes


def _asr_uncovered_notes(languages, runnable, installed):
    groups: dict[str, list[str]] = {}
    names: dict[str, str] = {}
    for language in languages:
        model = next(
            (m for m in runnable if not installed(m) and m.score(language) is not None), None
        )
        key = model.id if model is not None else ""
        names[key] = model.display_name if model is not None else ""
        groups.setdefault(key, []).append(language)
    notes = []
    for key, codes in groups.items():
        if key:
            notes.append(f"{names[key]} would recognise {_names(codes)} but is not installed.")
        else:
            notes.append(f"No installed speech model recognises {_names(codes)}.")
    return notes


def _select_cleanup(languages, hardware, models, installed):
    if not languages:
        return None, []
    runnable = _runnable(models, ModelKind.CLEANUP)
    order = {model.id: index for index, model in enumerate(models)}

    def scored(model: CatalogModel) -> list[str]:
        return [language for language in languages if model.score(language) is not None]

    def min_score(model: CatalogModel) -> int:
        return min((model.score(language) or 0 for language in scored(model)), default=0)

    def safe(model: CatalogModel) -> bool:
        covered = scored(model)
        return bool(covered) and all(
            (model.score(language) or 0) >= SAFETY_FLOOR for language in covered
        )

    def suitable(model: CatalogModel) -> bool:
        return model.latency_ms(hardware) is not None

    def fast(model: CatalogModel) -> bool:
        latency = model.latency_ms(hardware)
        return latency is not None and latency <= CLEANUP_TARGET_MS

    def quality(model: CatalogModel):
        return (-len(scored(model)), -min_score(model))

    def rank(model: CatalogModel):
        return (*quality(model), model.latency_ms(hardware) or 0, model.size_bytes,
                order[model.id])

    eligible = [m for m in runnable if safe(m) and fast(m) and installed(m)]
    if eligible:
        chosen = min(eligible, key=rank)
        covered = scored(chosen)
        unscored = [language for language in languages if language not in covered]
        reason = _cleanup_reason(chosen, covered, hardware)
        notes = []
        if unscored:
            notes.append(
                f"{chosen.display_name} has not been measured for {_names(unscored)}, so "
                f"{_names(unscored)} dictations are delivered without cleanup."
            )
        better = sorted((m for m in runnable if safe(m) and quality(m) < quality(chosen)),
                        key=rank)
        if better:
            notes.append(_cleanup_better_note(better[0], chosen, languages, hardware,
                                              suitable, fast))
        return _choice(chosen, covered, reason, True, hardware, unscored), notes

    everything = _names(languages)
    local = [m for m in runnable if installed(m)]
    notes = []
    if not local:
        notes.append(f"No cleanup model is installed, so raw transcripts are delivered for "
                     f"{everything}.")
    else:
        safe_local = sorted((m for m in local if safe(m)), key=rank)
        if safe_local:
            best_safe = safe_local[0]
            if suitable(best_safe):
                notes.append(
                    f"{best_safe.display_name} is safe for {everything} but takes about "
                    f"{_seconds(best_safe.latency_ms(hardware))} {_on(hardware)}, over the "
                    f"{_seconds(CLEANUP_TARGET_MS)} goal, so raw transcripts are delivered."
                )
            else:
                notes.append(
                    f"{best_safe.display_name} is safe for {everything} but cannot run "
                    f"{_on(hardware)}, so raw transcripts are delivered."
                )
        elif not any(scored(m) for m in local):
            notes.append(
                f"No installed cleanup model has been measured for {everything}, so raw "
                "transcripts are delivered."
            )
        else:
            unsafe = [
                lang
                for lang in languages
                if any(m.score(lang) is not None for m in local)
                and not any((m.score(lang) or 0) >= SAFETY_FLOOR for m in local)
            ]
            if unsafe:
                notes.append(
                    f"No installed cleanup model is safe for {_names(unsafe)}, so raw "
                    f"transcripts are delivered for {everything}."
                )
            else:
                notes.append(
                    f"No installed cleanup model is safe for {everything} together, so raw "
                    "transcripts are delivered."
                )
    missing = sorted((m for m in runnable if safe(m) and fast(m) and not installed(m)), key=rank)
    if missing:
        notes.append(
            f"{missing[0].display_name} would clean {everything} safely but is not installed."
        )
    return None, notes


def _cleanup_reason(model, languages, hardware):
    return (
        f"Cleans {_quality_phrase(model, languages, _cleanup_word)}. "
        f"About {_seconds(model.latency_ms(hardware))} per dictation {_on(hardware)}."
    )


def _cleanup_better_note(model, chosen, languages, hardware, suitable, fast):
    def value(candidate, lang):
        score = candidate.score(lang)
        return -1 if score is None else score

    better = [lang for lang in languages if value(model, lang) > value(chosen, lang)]
    head = f"{model.display_name} would clean {_names(better)} better"
    if not suitable(model):
        return f"{head} but cannot run {_on(hardware)}."
    if not fast(model):
        return (
            f"{head} but takes about {_seconds(model.latency_ms(hardware))} {_on(hardware)}, "
            f"over the {_seconds(CLEANUP_TARGET_MS)} goal."
        )
    return f"{head} but is not installed."


def _scored_languages(model: CatalogModel, languages: Sequence[str]) -> list[str]:
    return [language for language in languages if model.score(language) is not None]


def compose_candidates(
    languages: Sequence[str], hardware: Hardware, models: Sequence[CatalogModel]
) -> tuple[tuple[tuple[int, int], CatalogModel], ...]:
    """Every writing model that suits these languages on this hardware, best writer first.

    The rule is the cleanup rule of B5-19 without its latency bar (B5-44): a model is out when a
    language it scores is below the safety floor, when it scores none of the enabled languages,
    or when it cannot run on this hardware at all. A slow writer is kept, because composing an
    email is not on the dictation latency path. Order: more languages scored, then the highest
    lowest score, then the smaller file, then catalog order.
    """
    order = {model.id: index for index, model in enumerate(models)}
    found = []
    for model in _runnable(models, ModelKind.COMPOSE):
        covered = _scored_languages(model, languages)
        if not covered or model.latency_ms(hardware) is None:
            continue
        if any((model.score(language) or 0) < SAFETY_FLOOR for language in covered):
            continue
        quality = (-len(covered), -min(model.score(language) or 0 for language in covered))
        found.append((quality, model.size_bytes, order[model.id], model))
    found.sort(key=lambda item: (item[0], item[1], item[2]))
    return tuple((item[0], item[3]) for item in found)


def _select_compose(languages, hardware, models, installed, cleanup):
    if not languages or not _runnable(models, ModelKind.COMPOSE):
        return None, []
    candidates = compose_candidates(languages, hardware, models)
    everything = _names(languages)
    here = [(quality, model) for quality, model in candidates if installed(model)]
    absent = [(quality, model) for quality, model in candidates if not installed(model)]
    if not here:
        notes = [_no_compose_note(_runnable(models, ModelKind.COMPOSE), languages, hardware,
                                  installed, everything)]
        if absent:
            notes.append(f"{absent[0][1].display_name} would write {everything} but is not "
                         f"installed.")
        return None, notes

    quality, chosen = here[0]
    covered = _scored_languages(chosen, languages)
    unscored = [language for language in languages if language not in covered]
    shared = cleanup is not None and cleanup.file == chosen.file
    notes = []
    if unscored:
        notes.append(
            f"{chosen.display_name} has not been measured for writing in {_names(unscored)}, so "
            f"composing in {_names(unscored)} is a guess."
        )
    better = [model for rank, model in absent if rank < quality]
    if better:
        improved = [lang for lang in languages
                    if (better[0].score(lang) or -1) > (chosen.score(lang) or -1)]
        notes.append(f"{better[0].display_name} would write {_names(improved)} better but is not "
                     f"installed.")
    return (
        _choice(chosen, covered, _compose_reason(chosen, covered, hardware, shared), True,
                hardware, unscored),
        notes,
    )


def _no_compose_note(runnable, languages, hardware, installed, everything):
    local = [m for m in runnable if installed(m)]
    if not local:
        return f"No installed model writes text, so compose mode is off for {everything}."
    unsafe = [
        lang
        for lang in languages
        if any(m.score(lang) is not None for m in local)
        and not any((m.score(lang) or 0) >= SAFETY_FLOOR for m in local)
    ]
    if unsafe:
        return (f"No installed writing model is safe for {_names(unsafe)}, so compose mode is "
                f"off for {everything}.")
    if not any(m.latency_ms(hardware) is not None for m in local):
        return (f"No installed writing model can run {_on(hardware)}, so compose mode is off "
                f"for {everything}.")
    if not any(_scored_languages(m, languages) for m in local):
        return (f"No installed writing model has been measured for {everything}, so compose "
                f"mode is off.")
    return f"No installed writing model suits {everything} together, so compose mode is off."


def _compose_reason(model, languages, hardware, shared: bool) -> str:
    sentences = [
        f"Writes {_quality_phrase(model, languages, _cleanup_word)}.",
        f"About {_seconds(model.latency_ms(hardware))} for a short email {_on(hardware)}.",
    ]
    if shared:
        sentences.append("It is the cleanup model as well, so nothing else is loaded.")
    return " ".join(sentences)


def _asr_word(score: int) -> str:
    if score >= 85:
        return "very well"
    if score >= 70:
        return "well"
    if score >= 50:
        return "fairly well"
    return "poorly"


def _cleanup_word(score: int) -> str:
    if score >= 90:
        return "very well"
    if score >= 75:
        return "well"
    return "cautiously"


def _quality_phrase(model, languages, word) -> str:
    groups: dict[str, list[str]] = {}
    for language in languages:
        groups.setdefault(word(model.score(language) or 0), []).append(language)
    parts = [f"{_names(codes)} {label}" for label, codes in groups.items()]
    return _join(parts)


def _on(hardware: Hardware) -> str:
    return "on the graphics card" if hardware is Hardware.GPU else "on the processor"


def _seconds(ms: int | None) -> str:
    value = round((ms or 0) / 1000.0, 1)
    text = f"{value:.1f}".removesuffix(".0")
    return f"{text} s"


@lru_cache(maxsize=1)
def _language_names() -> dict[str, str]:
    try:
        table = json.loads(data_path("whisper_languages.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    names: dict[str, str] = {}
    for name, code in table.items():
        names.setdefault(code, name.title())
    return names


def language_name(code: str) -> str:
    return _language_names().get(code, code)


def _name(code: str) -> str:
    return language_name(code)


def _names(codes: Sequence[str]) -> str:
    return _join([_name(code) for code in codes])


def _join(parts: Sequence[str]) -> str:
    items = list(parts)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} and {items[-1]}"


def model_bytes(models: Iterable[CatalogModel]) -> int:
    """What a set of models occupies on disk, counting two entries for one file once."""
    seen: set[tuple[str, ...]] = set()
    total = 0
    for model in models:
        if model.file_key in seen:
            continue
        seen.add(model.file_key)
        total += model.size_bytes
    return total


def installed_footprint(
    models: Iterable[CatalogModel], base_bytes: int = BASE_INSTALL_BYTES + VAD_MODEL_BYTES
) -> int:
    """The whole installation: the app, both engine folders, the data files and these models.

    ``base_bytes`` is everything an install carries whatever the languages are: the measured
    dist folder of spec 19.4 (app 124,909,062, engines 195,805,484, data and licences 264,727)
    plus the Silero VAD model, which every language needs and which the catalog does not list.
    """
    return base_bytes + model_bytes(models)


@dataclass(frozen=True)
class InstallPlan:
    """The models one machine needs for one set of languages, inside the size budget."""

    hardware: Hardware
    languages: tuple[str, ...]
    models: tuple[CatalogModel, ...]
    writing: CatalogModel | None
    dropped: tuple[CatalogModel, ...]
    footprint_bytes: int
    reason: str

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(model.id for model in self.models)

    @property
    def files(self) -> tuple[str, ...]:
        found: list[str] = []
        for model in self.models:
            for name in model.files:
                if name not in found:
                    found.append(name)
        return tuple(found)

    @property
    def fits(self) -> bool:
        return self.footprint_bytes <= INSTALL_BUDGET_BYTES


def plan_install(
    languages: Sequence[str],
    hardware: Hardware,
    catalog: Sequence[CatalogModel] | None = None,
    base_bytes: int = BASE_INSTALL_BYTES + VAD_MODEL_BYTES,
) -> InstallPlan:
    """What an installer should put on a machine of this class for these languages (B5-46).

    Three rules, in order. The speech models are the ones ``select_models`` chooses for this
    hardware. One text model is installed and never two: the best writer that this hardware can
    also run as the cleanup model, which is why a graphics card gets Qwen3.5 4B for both jobs
    and a processor keeps Gemma 4 E2B for both. Then the size budget: while the installation is
    over ``INSTALL_BUDGET_BYTES``, a speech model the speed rule of B5-26 added is dropped, the
    largest first, because every language it serves is within the quality margin of a speech
    model that stays, so the install loses speed and never a language.
    """
    codes = tuple(dict.fromkeys(languages))
    models = tuple(
        model for model in (load_catalog() if catalog is None else catalog)
        if model.redistributable
    )
    by_id = {model.id: model for model in models}
    speech_ids = [choice.model_id for choice in select_models(codes, hardware, None, models).asr]
    chosen = [by_id[model_id] for model_id in speech_ids]
    text = _text_models(codes, hardware, models, speech_ids)
    chosen.extend(text)

    dropped: list[CatalogModel] = []
    footprint = installed_footprint(chosen, base_bytes)
    while footprint > INSTALL_BUDGET_BYTES:
        spare = _spare_speech(chosen, codes)
        if spare is None:
            break
        chosen.remove(spare)
        dropped.append(spare)
        footprint = installed_footprint(chosen, base_bytes)
    writing = next((model for model in text if model.kind is ModelKind.CLEANUP), None)
    return InstallPlan(
        hardware=hardware,
        languages=codes,
        models=tuple(chosen),
        writing=writing,
        dropped=tuple(dropped),
        footprint_bytes=footprint,
        reason=_plan_reason(hardware, writing, tuple(dropped), tuple(chosen), codes),
    )


def _text_models(languages, hardware, models, speech_ids) -> tuple[CatalogModel, ...]:
    """The one model that cleans and writes here, as its catalog entries for both jobs."""
    for _quality, candidate in compose_candidates(languages, hardware, models):
        twins = tuple(model for model in models if model.file_key == candidate.file_key)
        installed = frozenset(speech_ids) | {model.id for model in twins}
        trial = select_models(languages, hardware, installed, models)
        if trial.one_text_model:
            return twins
    cleanup = select_models(languages, hardware, None, models).cleanup
    if cleanup is None:
        return ()
    return tuple(model for model in models if model.file == cleanup.file)


def _spare_speech(chosen: Sequence[CatalogModel], languages: Sequence[str]
                  ) -> CatalogModel | None:
    """The largest speech model whose languages another chosen speech model covers anyway."""
    speech = [model for model in chosen if model.kind is ModelKind.ASR]
    best = {}
    for language in languages:
        scores = [model.score(language) for model in speech
                  if model.score(language) is not None]
        if scores:
            best[language] = max(scores)
    spare = []
    for model in speech:
        served = [language for language in languages
                  if model.score(language) is not None
                  and (model.score(language) or 0) >= best[language] - ASR_MARGIN]
        others = [other for other in speech if other is not model]
        if not served or not others:
            continue
        if all(
            any((other.score(language) or -1) >= best[language] - ASR_MARGIN for other in others)
            for language in served
        ):
            spare.append(model)
    if not spare:
        return None
    return max(spare, key=lambda model: (model.size_bytes, model.id))


def _plan_reason(hardware, writing, dropped, chosen, languages) -> str:
    if writing is None:
        return "No writing model suits these languages, so this install cleans nothing."
    if hardware is Hardware.GPU:
        sentence = (f"Your graphics card can run the better writing model, "
                    f"{writing.display_name}, {_gigabytes(writing.size_bytes)}.")
    else:
        sentence = (f"No graphics card was found, so Spells takes the small fast model, "
                    f"{writing.display_name}, {_gigabytes(writing.size_bytes)}.")
    if dropped:
        kept = [model for model in chosen if model.kind is ModelKind.ASR]
        names = _join([model.display_name for model in kept])
        sentence += (f" {_names(languages)} together need more than 5 GB with "
                     f"{dropped[0].display_name} as well, so speech runs on {names} alone.")
    return sentence


def _gigabytes(size: int) -> str:
    return f"{size / 1_000_000_000:.1f} GB"
