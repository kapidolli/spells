"""Model choice from the dictation languages and the hardware (batch 5, language picker)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from spells import modelcatalog
from spells.gpu import NO_GPU, GpuDevice, GpuSelection
from spells.modelcatalog import (
    CLEANUP_TARGET_MS,
    SAFETY_FLOOR,
    CatalogError,
    Hardware,
    ModelChoice,
    ModelKind,
    Selection,
    hardware_from_gpu,
    installed_ids,
    language_name,
    load_catalog,
    parse_catalog,
    resolve_extra_files,
    select_models,
)

GPU = Hardware.GPU
CPU = Hardware.CPU
WHISPER = "whisper-large-v3-turbo-q8_0"
QWEN_ASR = "qwen3-asr-0.6b-q4_k_m"
QWEN_ASR_Q8 = "qwen3-asr-0.6b-q8_0"
FLUTRA = "whisper-large-v3-turbo-sq-flutra-v2-q8_0"
GEMMA4 = "gemma-4-e2b-it-q4_0"
QWEN35_4B = "qwen3.5-4b-q4_k_m"
QWEN35_4B_WRITE = "qwen3.5-4b-q4_k_m-compose"
GEMMA4_WRITE = "gemma-4-e2b-it-q4_0-compose"
QWEN3_2507 = "qwen3-4b-instruct-2507-q4_k_m"
QWEN35_2B = "qwen3.5-2b-q4_k_m"
GEMMA3 = "gemma-3-4b-it-q4_k_m"
EM_DASH = "\u2014"


def asr(model_id, languages, *, gpu=300, cpu=5000, runtime="whisper-server", size=1, **extra):
    return {
        "id": model_id,
        "kind": "asr",
        "display_name": extra.pop("display_name", model_id),
        "file": extra.pop("file", f"{model_id}.bin"),
        "extra_files": extra.pop("extra_files", []),
        "size_bytes": size,
        "license": "MIT",
        "runtime": runtime,
        "languages": languages,
        "hardware": {"gpu": gpu, "cpu": cpu},
        "engine_args": extra.pop("engine_args", []),
        "notes": "",
    }


def cleanup(model_id, languages, *, gpu=500, cpu=1000, runtime="llama-server", size=1, **extra):
    entry = asr(model_id, languages, gpu=gpu, cpu=cpu, runtime=runtime, size=size, **extra)
    entry["kind"] = "cleanup"
    entry["file"] = extra.get("file", f"{model_id}.gguf")
    return entry


def compose(model_id, languages, *, gpu=1500, cpu=6000, runtime="llama-server", size=1, **extra):
    entry = cleanup(model_id, languages, gpu=gpu, cpu=cpu, runtime=runtime, size=size, **extra)
    entry["kind"] = "compose"
    return entry


def catalog(*entries):
    return parse_catalog({"schema_version": 1, "models": list(entries)})


def with_speech(*entries):
    return catalog(asr("speech", {"*": 80}), *entries)


def all_ids(models):
    return frozenset(model.id for model in models)


# The shipped catalog ----------------------------------------------------------------------------


def test_the_shipped_catalog_parses_and_names_every_measured_model():
    models = load_catalog()
    assert [model.id for model in models] == [
        WHISPER,
        QWEN_ASR,
        QWEN_ASR_Q8,
        FLUTRA,
        GEMMA4,
        QWEN35_4B,
        QWEN3_2507,
        QWEN35_2B,
        GEMMA3,
        QWEN35_4B_WRITE,
        GEMMA4_WRITE,
    ]
    assert load_catalog() is load_catalog()


def test_the_shipped_catalog_says_its_numbers_are_provisional():
    raw = json.loads(modelcatalog.default_catalog_path().read_text(encoding="utf-8"))
    assert "provisional" in raw["notes"].lower()
    for model in load_catalog():
        assert model.notes.startswith("Provisional.")
        assert model.runtime in modelcatalog.RUNTIMES[model.kind]


def test_the_shipped_catalog_carries_the_measurements():
    models = {model.id: model for model in load_catalog()}
    whisper = models[WHISPER]
    assert whisper.file == "ggml-large-v3-turbo-q8_0.bin"
    assert whisper.size_bytes == 874188075
    assert (whisper.score("en"), whisper.score("de"), whisper.score("sq")) == (87, 85, 48)
    assert whisper.latency_ms(CPU) == 4500
    assert whisper.redistributable is True
    qwen = models[QWEN_ASR]
    assert qwen.runtime == "llama-asr"
    assert qwen.file == "Qwen3-ASR-0.6B-Q4_K_M.gguf"
    assert qwen.extra_files == ("mmproj-Qwen3-ASR-0.6B-Q8_0.gguf",)
    assert qwen.size_bytes == 484215744 + 214392480
    assert qwen.license == "Apache-2.0"
    assert dict(qwen.languages) == {"en": 92, "de": 88}
    assert (qwen.latency_ms(CPU), qwen.latency_ms(GPU)) == (1150, 400)
    assert qwen.engine_args == ("--mmproj", "{extra:0}", "--no-webui")
    assert qwen.redistributable is True
    q8 = models[QWEN_ASR_Q8]
    assert q8.runtime == "llama-asr"
    assert q8.file == "Qwen3-ASR-0.6B-Q8_0.gguf"
    assert q8.extra_files == qwen.extra_files
    assert q8.size_bytes == 804749248 + 214392480
    assert q8.license == "Apache-2.0"
    assert dict(q8.languages) == dict(qwen.languages)
    assert (q8.latency_ms(CPU), q8.latency_ms(GPU)) == (1300, 400)
    assert q8.engine_args == qwen.engine_args
    assert q8.redistributable is True
    flutra = models[FLUTRA]
    assert flutra.runtime == "whisper-server"
    assert flutra.size_bytes == 874188075
    assert dict(flutra.languages) == {"sq": 62, "en": 87, "de": 68}
    assert (flutra.latency_ms(CPU), flutra.latency_ms(GPU)) == (4500, 300)
    assert flutra.license == "unknown (personal use only)"
    assert flutra.redistributable is False
    gemma = models[GEMMA4]
    assert gemma.extra_files == ("mtp-gemma-4-E2B-it-Q4_0.gguf",)
    assert gemma.size_bytes == 2841481184 + 59235872
    assert gemma.license == "Apache-2.0"
    assert gemma.latency_ms(CPU) == 1024
    assert gemma.engine_args == ("--spec-type", "draft-mtp", "-md", "{extra:0}")
    assert models[QWEN35_4B].latency_ms(CPU) == 2700
    assert models[QWEN3_2507].latency_ms(CPU) == 2160
    assert models[QWEN35_2B].latency_ms(CPU) == 1340
    assert models[GEMMA3].latency_ms(CPU) == 4050
    assert models[GEMMA3].license == "Gemma Terms of Use"
    ngram = (
        "--spec-type",
        "ngram-simple",
        "--spec-ngram-simple-size-n",
        "2",
        "--spec-ngram-simple-size-m",
        "24",
    )
    for model in models.values():
        if model.kind is ModelKind.CLEANUP and model.id != GEMMA4:
            assert model.engine_args == ngram


def test_the_shipped_scores_keep_albanian_harm_below_the_safety_floor():
    models = {model.id: model for model in load_catalog()}
    for harmful in (QWEN3_2507, QWEN35_2B, GEMMA3):
        assert models[harmful].score("sq") < SAFETY_FLOOR
    for safe in (GEMMA4, QWEN35_4B):
        assert models[safe].score("sq") >= SAFETY_FLOOR
    assert models[QWEN35_4B].score("sq") > models[GEMMA4].score("sq")


def test_the_catalog_file_has_no_em_dash():
    assert EM_DASH not in modelcatalog.default_catalog_path().read_text(encoding="utf-8")


# Parsing -------------------------------------------------------------------------------------


GOOD = asr("w", {"en": 90})


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw.update(schema_version=2),
        lambda raw: raw.update(models={}),
        lambda raw: raw["models"].append(copy.deepcopy(raw["models"][0])),
        lambda raw: raw["models"][0].update(kind="tts"),
        lambda raw: raw["models"][0].update(id=""),
        lambda raw: raw["models"][0].update(file="models/w.bin"),
        lambda raw: raw["models"][0].update(extra_files=["a\\b.gguf"]),
        lambda raw: raw["models"][0].update(size_bytes=-1),
        lambda raw: raw["models"][0].update(size_bytes=True),
        lambda raw: raw["models"][0].update(languages={"en": 101}),
        lambda raw: raw["models"][0].update(languages={"en": 50.5}),
        lambda raw: raw["models"][0].update(hardware={"gpu": 300}),
        lambda raw: raw["models"][0].update(hardware={"gpu": "fast", "cpu": None}),
        lambda raw: raw["models"][0].update(engine_args="--spec-type ngram-simple"),
        lambda raw: raw["models"][0].update(engine_args=["--mmproj", "{extra:0}"]),
        lambda raw: raw["models"][0].update(extra_files=["a.gguf"], engine_args=["{extra:1}"]),
        lambda raw: raw["models"][0].update(redistributable="no"),
    ],
)
def test_a_malformed_catalog_is_refused(mutate):
    raw = {"schema_version": 1, "models": [copy.deepcopy(GOOD)]}
    mutate(raw)
    with pytest.raises(CatalogError):
        parse_catalog(raw)


def test_a_catalog_that_is_not_an_object_is_refused():
    with pytest.raises(CatalogError):
        parse_catalog([GOOD])


def test_an_unreadable_catalog_file_is_a_catalog_error(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text("{ broken", encoding="utf-8")
    with pytest.raises(CatalogError):
        load_catalog(path)
    with pytest.raises(CatalogError):
        load_catalog(tmp_path / "missing.json")


def test_a_null_latency_means_unsuitable_and_notes_are_optional():
    entry = asr("w", {"en": 90}, cpu=None)
    del entry["notes"]
    (model,) = parse_catalog({"schema_version": 1, "models": [entry]})
    assert model.latency_ms(CPU) is None
    assert model.latency_ms(GPU) == 300
    assert model.notes == ""


def test_an_extra_file_placeholder_and_the_redistributable_flag_parse():
    entry = asr("q", {"en": 90}, runtime="llama-asr", extra_files=["mmproj.gguf"],
                engine_args=["--mmproj", "{extra:0}"])
    entry["redistributable"] = False
    (model,) = parse_catalog({"schema_version": 1, "models": [entry]})
    assert model.engine_args == ("--mmproj", "{extra:0}")
    assert model.redistributable is False
    (default,) = catalog(asr("w", {"en": 90}))
    assert default.redistributable is True


def test_extra_file_placeholders_resolve_to_absolute_paths(tmp_path):
    extras = (tmp_path / "mmproj.gguf", tmp_path / "draft.gguf")
    resolved = resolve_extra_files(["--mmproj", "{extra:0}", "-md={extra:1}", "-c"], extras)
    assert resolved == ("--mmproj", str(extras[0]), f"-md={extras[1]}", "-c")
    relative = resolve_extra_files(["{extra:0}"], (Path("models") / "x.gguf",))
    assert Path(relative[0]).is_absolute()
    with pytest.raises(ValueError):
        resolve_extra_files(["{extra:2}"], extras)


def test_the_star_scores_every_unmeasured_language():
    (model,) = catalog(asr("w", {"en": 90, "*": 50}))
    assert model.score("en") == 90
    assert model.score("fr") == 50
    assert model.measured("en") and not model.measured("fr")
    (strict,) = catalog(cleanup("c", {"en": 90}))
    assert strict.score("fr") is None


# Installed models ------------------------------------------------------------------------------


def test_a_model_is_installed_when_its_file_and_every_extra_file_are_present(tmp_path):
    models = catalog(
        asr("w", {"en": 90}),
        cleanup("c", {"en": 90}, extra_files=["c-draft.gguf"]),
    )
    (tmp_path / "w.bin").write_bytes(b"")
    (tmp_path / "c.gguf").write_bytes(b"")
    assert installed_ids(tmp_path, models) == frozenset({"w"})
    (tmp_path / "c-draft.gguf").write_bytes(b"")
    assert installed_ids(tmp_path, models) == frozenset({"w", "c"})


def test_a_directory_with_a_model_name_does_not_count(tmp_path):
    models = catalog(asr("w", {"en": 90}))
    (tmp_path / "w.bin").mkdir()
    assert installed_ids(tmp_path, models) == frozenset()


def test_a_missing_models_folder_installs_nothing(tmp_path):
    assert installed_ids(tmp_path / "nowhere") == frozenset()


def test_installed_ids_reads_the_shipped_catalog_by_default(tmp_path):
    (tmp_path / "ggml-large-v3-turbo-q8_0.bin").write_bytes(b"")
    (tmp_path / "gemma-4-E2B-it-Q4_0.gguf").write_bytes(b"")
    assert installed_ids(tmp_path) == frozenset({WHISPER})


# Hardware ------------------------------------------------------------------------------------


def gpu_selection(uma, memory_mb, raw_index=0):
    device = GpuDevice(raw_index, "Vulkan0", "Device", memory_mb, memory_mb, uma=uma)
    return GpuSelection(raw_index=raw_index, name="Device", memory_mb=memory_mb, devices=(device,))


def test_no_device_is_the_processor():
    assert hardware_from_gpu(None) is CPU
    assert hardware_from_gpu(NO_GPU) is CPU


def test_a_discrete_gpu_is_the_graphics_card_whatever_its_memory():
    assert hardware_from_gpu(gpu_selection(False, 7810)) is GPU
    assert hardware_from_gpu(gpu_selection(False, 2048)) is GPU


def test_an_integrated_gpu_is_the_processor_whatever_its_shared_memory():
    assert hardware_from_gpu(gpu_selection(True, 32000)) is CPU


def test_an_unknown_gpu_counts_as_a_graphics_card_only_with_enough_memory():
    assert hardware_from_gpu(gpu_selection(None, 7810)) is GPU
    assert hardware_from_gpu(gpu_selection(None, 4096)) is GPU
    assert hardware_from_gpu(gpu_selection(None, 4095)) is CPU


def test_the_chosen_device_decides_not_the_best_one():
    discrete = GpuDevice(0, "Vulkan0", "RTX", 7810, 7000, uma=False)
    integrated = GpuDevice(1, "Vulkan1", "Radeon", 16000, 15000, uma=True)
    override = GpuSelection(raw_index=1, name="Radeon", memory_mb=16000,
                            devices=(discrete, integrated))
    assert hardware_from_gpu(override) is CPU


def test_a_device_missing_from_the_list_falls_back_to_the_selection_memory():
    lonely = GpuSelection(raw_index=3, name="Ghost", memory_mb=8000, devices=())
    assert hardware_from_gpu(lonely) is GPU
    assert hardware_from_gpu(GpuSelection(raw_index=3, name="Ghost", memory_mb=1024,
                                          devices=())) is CPU


# Worked examples on the shipped catalog ------------------------------------------------------------


@pytest.mark.parametrize(
    ("languages", "hardware", "speech", "cleanup_id", "cpu_ok"),
    [
        (["en", "de"], GPU, [(QWEN_ASR, ("en", "de"))], GEMMA4, False),
        (["en", "de"], CPU, [(QWEN_ASR, ("en", "de"))], GEMMA4, True),
        (["sq"], GPU, [(FLUTRA, ("sq",))], QWEN35_4B, False),
        (["sq"], CPU, [(FLUTRA, ("sq",))], GEMMA4, True),
        (["en", "de", "sq"], GPU, [(QWEN_ASR, ("en", "de")), (FLUTRA, ("sq",))], QWEN35_4B,
         False),
        (["en", "de", "sq"], CPU, [(QWEN_ASR, ("en", "de")), (FLUTRA, ("sq",))], GEMMA4, True),
    ],
)
def test_the_worked_examples(languages, hardware, speech, cleanup_id, cpu_ok):
    selection = select_models(languages, hardware)
    assert [(choice.model_id, choice.languages) for choice in selection.asr] == speech
    assert selection.cleanup is not None
    assert selection.cleanup.model_id == cleanup_id
    assert selection.cleanup.languages == tuple(languages)
    assert selection.cleanup.unscored == ()
    assert selection.cpu_cleanup_allowed is cpu_ok
    assert selection.hardware is hardware


def test_without_the_albanian_whisper_qwen_speeds_up_english_and_german_on_the_processor():
    installed = frozenset({WHISPER, QWEN_ASR, GEMMA4})
    selection = select_models(["en", "de", "sq"], CPU, installed)
    assert [(c.model_id, c.languages) for c in selection.asr] == [
        (QWEN_ASR, ("en", "de")),
        (WHISPER, ("sq",)),
    ]
    assert selection.primary_asr.model_id == QWEN_ASR
    assert selection.asr[0].reason == (
        "Recognises English and German very well, and is about four times faster than "
        "Whisper large-v3 turbo here. About 1.1 s to transcribe a 15 s dictation on the "
        "processor."
    )
    assert selection.asr[1].reason == (
        "Recognises Albanian poorly. About 4.5 s to transcribe a 15 s dictation on the processor."
    )


def test_the_speed_split_is_independent_of_language_order():
    installed = frozenset({WHISPER, QWEN_ASR, GEMMA4})
    one = select_models(["en", "de", "sq"], CPU, installed)
    two = select_models(["sq", "de", "en"], CPU, installed)
    assert {c.model_id: set(c.languages) for c in one.asr} == {
        c.model_id: set(c.languages) for c in two.asr
    }
    assert {c.model_id: set(c.languages) for c in one.asr} == {
        QWEN_ASR: {"en", "de"},
        WHISPER: {"sq"},
    }


def test_on_a_graphics_card_whisper_stays_the_only_engine_and_the_skip_is_explained():
    installed = frozenset({WHISPER, QWEN_ASR, GEMMA4})
    selection = select_models(["en", "de", "sq"], GPU, installed)
    assert [(c.model_id, c.languages) for c in selection.asr] == [
        (WHISPER, ("en", "de", "sq")),
    ]


def test_a_faster_but_not_meaningfully_faster_alternative_is_named_as_skipped():
    selection = select_models(["en", "de"], GPU)
    assert [c.model_id for c in selection.asr] == [QWEN_ASR]
    assert selection.notes == (
        (
            "Whisper large-v3 turbo would also recognise English and German but is not fast "
            "enough here to justify a second engine."
        ),
    )


@pytest.mark.parametrize("hardware", [CPU, GPU])
@pytest.mark.parametrize("languages", [["en", "de", "sq"], ["en", "de"], ["en"]])
def test_a_machine_holding_both_quantisations_uses_the_q4_k_m_one(languages, hardware):
    installed = frozenset({WHISPER, QWEN_ASR, QWEN_ASR_Q8, GEMMA4, GEMMA4_WRITE})
    chosen = [c.model_id for c in select_models(languages, hardware, installed).asr]
    assert QWEN_ASR_Q8 not in chosen
    planned = [c.model_id for c in select_models(languages, hardware).asr]
    assert QWEN_ASR_Q8 not in planned
    if hardware is CPU or languages != ["en", "de", "sq"]:
        assert QWEN_ASR in chosen
        assert QWEN_ASR in planned


@pytest.mark.parametrize("hardware", [CPU, GPU])
@pytest.mark.parametrize("languages", [["en", "de", "sq"], ["en", "de"], ["en"]])
def test_an_install_that_holds_only_the_q8_0_file_keeps_using_it(languages, hardware):
    installed = frozenset({WHISPER, QWEN_ASR_Q8, GEMMA4, GEMMA4_WRITE})
    with_q4 = installed - {QWEN_ASR_Q8} | {QWEN_ASR}
    before = select_models(languages, hardware, installed)
    after = select_models(languages, hardware, with_q4)
    swap = {QWEN_ASR: QWEN_ASR_Q8}
    assert [(c.model_id, c.languages) for c in before.asr] == [
        (swap.get(c.model_id, c.model_id), c.languages) for c in after.asr
    ]
    assert not any("Qwen3-ASR" in note for note in before.notes)


def test_on_the_processor_the_slower_albanian_model_is_named_in_a_note():
    selection = select_models(["en", "de", "sq"], CPU)
    assert selection.notes == (
        (
            "Qwen3.5 4B would clean English and Albanian better but takes about 2.7 s on the "
            "processor, over the 1.8 s goal."
        ),
    )


def test_a_graphics_card_with_only_the_bundled_model_names_the_better_one_as_not_installed():
    installed = frozenset({WHISPER, GEMMA4, GEMMA4_WRITE})
    selection = select_models(["en", "de", "sq"], GPU, installed)
    assert selection.cleanup.model_id == GEMMA4
    assert selection.compose.model_id == GEMMA4_WRITE
    assert selection.notes == (
        "Whisper large-v3 turbo Albanian would recognise Albanian better but is not installed.",
        "Qwen3.5 4B would clean English and Albanian better but is not installed.",
        "Qwen3.5 4B would write English, German and Albanian better but is not installed.",
    )


def test_the_reasons_are_plain_sentences():
    selection = select_models(["en", "de", "sq"], CPU)
    assert selection.asr[0].reason == (
        "Recognises English and German very well. "
        "About 1.1 s to transcribe a 15 s dictation on the processor."
    )
    assert selection.asr[1].reason == (
        "Recognises Albanian fairly well. "
        "About 4.5 s to transcribe a 15 s dictation on the processor."
    )
    assert selection.cleanup.reason == (
        "Cleans English and German very well and Albanian cautiously. "
        "About 1 s per dictation on the processor."
    )
    alone = select_models(["en", "de", "sq"], CPU, frozenset({WHISPER, GEMMA4}))
    assert alone.asr[0].reason == (
        "Recognises English and German very well and Albanian poorly. "
        "It is the only speech model installed. "
        "About 4.5 s to transcribe a 15 s dictation on the processor."
    )
    gpu = select_models(["en", "de", "sq"], GPU)
    assert gpu.cleanup.reason == (
        "Cleans English very well, German well and Albanian cautiously. "
        "About 0.9 s per dictation on the graphics card."
    )


def test_no_reason_or_note_uses_jargon_or_an_em_dash():
    for languages in (["en"], ["de", "sq"], ["fr"], ["en", "de", "sq"]):
        for hardware in (GPU, CPU):
            selection = select_models(languages, hardware, frozenset({WHISPER, GEMMA4}))
            texts = [choice.reason for choice in selection.asr] + list(selection.notes)
            if selection.cleanup is not None:
                texts.append(selection.cleanup.reason)
            for text in texts:
                assert "score" not in text.lower()
                assert EM_DASH not in text
                assert text.endswith(".")


def test_a_chosen_model_carries_what_the_engines_need():
    selection = select_models(["en"], CPU)
    choice = selection.cleanup
    assert choice.installed is True
    assert choice.file == "gemma-4-E2B-it-Q4_0.gguf"
    assert choice.extra_files == ("mtp-gemma-4-E2B-it-Q4_0.gguf",)
    assert choice.engine_args == ("--spec-type", "draft-mtp", "-md", "{extra:0}")
    assert choice.runtime == "llama-server"
    assert choice.latency_ms == 1024
    speech = selection.asr[0]
    assert speech.file == "Qwen3-ASR-0.6B-Q4_K_M.gguf"
    assert speech.extra_files == ("mmproj-Qwen3-ASR-0.6B-Q8_0.gguf",)
    assert speech.runtime == "llama-asr"
    assert speech.engine_args == ("--mmproj", "{extra:0}", "--no-webui")


def test_an_unmeasured_language_keeps_speech_and_skips_cleanup_for_that_language_only():
    selection = select_models(["en", "fr"], GPU)
    writing_note = (
        "Qwen3.5 4B has not been measured for writing in French, so composing in French is a "
        "guess."
    )
    assert selection.asr_for("fr").model_id == WHISPER
    assert "Not measured yet for French." in selection.asr[0].reason
    assert selection.cleanup.model_id == QWEN35_4B
    assert selection.cleanup.languages == ("en",)
    assert selection.cleanup.unscored == ("fr",)
    assert selection.notes == (
        (
            "Qwen3.5 4B has not been measured for French, so French dictations are delivered "
            "without cleanup."
        ),
        writing_note,
    )


def test_a_language_no_installed_cleanup_model_measured_turns_cleanup_off():
    selection = select_models(["fr"], GPU, frozenset({WHISPER, GEMMA4, GEMMA4_WRITE}))
    assert selection.cleanup is None
    assert selection.compose is None
    assert selection.notes == (
        (
            "No installed cleanup model has been measured for French, so raw transcripts are "
            "delivered."
        ),
        "No installed writing model has been measured for French, so compose mode is off.",
    )


# Speech recognition rule ------------------------------------------------------------------------


def test_each_language_gets_the_best_speech_model_and_languages_share_one_choice():
    models = catalog(
        asr("general", {"en": 87, "de": 85, "sq": 48}, display_name="General"),
        asr("albanian", {"sq": 80}, display_name="Albanian"),
    )
    selection = select_models(["en", "sq", "de"], GPU, all_ids(models), models)
    assert [(c.model_id, c.languages) for c in selection.asr] == [
        ("general", ("en", "de")),
        ("albanian", ("sq",)),
    ]
    assert selection.primary_asr.model_id == "general"
    assert selection.asr_for("sq").model_id == "albanian"
    assert selection.asr_for("fr") is None


def test_a_speech_model_within_the_margin_keeps_one_engine():
    models = catalog(
        asr("general", {"en": 87, "sq": 76}, display_name="General"),
        asr("albanian", {"sq": 80}, display_name="Albanian"),
    )
    selection = select_models(["en", "sq"], GPU, all_ids(models), models)
    assert [(c.model_id, c.languages) for c in selection.asr] == [("general", ("en", "sq"))]
    assert "One speech engine for all of them saves memory and start time." in (
        selection.asr[0].reason
    )


def test_the_speech_choice_does_not_depend_on_the_language_order():
    models = catalog(
        asr("general", {"en": 87, "de": 85, "sq": 48}),
        asr("albanian", {"sq": 80, "en": 84}),
    )
    one = select_models(["sq", "en", "de"], GPU, all_ids(models), models)
    two = select_models(["de", "en", "sq"], GPU, all_ids(models), models)
    assert {c.model_id: set(c.languages) for c in one.asr} == {
        c.model_id: set(c.languages) for c in two.asr
    }
    assert {c.model_id: set(c.languages) for c in one.asr} == {
        "general": {"en", "de"},
        "albanian": {"sq"},
    }


def test_speech_ties_go_to_the_faster_then_the_smaller_then_the_earlier_model():
    faster = catalog(asr("a", {"en": 80}, gpu=500), asr("b", {"en": 80}, gpu=300))
    assert select_models(["en"], GPU, None, faster).asr[0].model_id == "b"
    smaller = catalog(asr("a", {"en": 80}, size=9), asr("b", {"en": 80}, size=3))
    assert select_models(["en"], GPU, None, smaller).asr[0].model_id == "b"
    earlier = catalog(asr("a", {"en": 80}), asr("b", {"en": 80}))
    assert select_models(["en"], GPU, None, earlier).asr[0].model_id == "a"


def test_a_speech_model_unsuitable_for_the_hardware_is_never_chosen():
    models = catalog(asr("gpu-only", {"en": 95}, cpu=None), asr("both", {"en": 70}))
    assert select_models(["en"], CPU, None, models).asr[0].model_id == "both"
    assert select_models(["en"], GPU, None, models).asr[0].model_id == "gpu-only"


def test_a_speech_model_for_an_unsupported_runtime_is_never_chosen():
    models = catalog(asr("onnx", {"en": 99}, runtime="sherpa-onnx"), asr("w", {"en": 70}))
    assert select_models(["en"], GPU, None, models).asr[0].model_id == "w"


def test_an_uninstalled_better_speech_model_is_a_note_never_a_choice():
    models = catalog(
        asr("general", {"en": 87, "sq": 48}, display_name="Whisper"),
        asr("albanian", {"sq": 80}, display_name="Albanian Whisper"),
    )
    selection = select_models(["en", "sq"], GPU, frozenset({"general"}), models)
    assert [c.model_id for c in selection.asr] == ["general"]
    assert "Albanian Whisper would recognise Albanian better but is not installed." in (
        selection.notes
    )


def test_languages_without_an_installed_speech_model_are_named():
    models = catalog(asr("w", {"en": 87}, display_name="Whisper"), asr("x", {"sq": 60},
                                                                         display_name="Xs"))
    selection = select_models(["en", "sq", "de"], GPU, frozenset({"w"}), models)
    assert [c.languages for c in selection.asr] == [("en",)]
    assert "Xs would recognise Albanian but is not installed." in selection.notes
    assert "No installed speech model recognises German." in selection.notes


def test_speech_quality_words_follow_the_scores():
    models = catalog(asr("w", {"en": 85, "de": 70, "sq": 50, "it": 49}))
    reason = select_models(["en", "de", "sq", "it"], GPU, None, models).asr[0].reason
    assert reason.startswith(
        "Recognises English very well, German well, Albanian fairly well and Italian poorly."
    )


# Cleanup rule ------------------------------------------------------------------------------------


def test_a_model_harmful_for_any_enabled_language_is_excluded():
    models = catalog(
        cleanup("strong", {"en": 95, "sq": SAFETY_FLOOR - 1}, cpu=500),
        cleanup("steady", {"en": 70, "sq": SAFETY_FLOOR}, cpu=900),
    )
    assert select_models(["en"], CPU, None, models).cleanup.model_id == "strong"
    assert select_models(["en", "sq"], CPU, None, models).cleanup.model_id == "steady"


def test_a_model_without_a_score_for_a_language_still_cleans_the_scored_ones():
    models = catalog(cleanup("c", {"en": 95}))
    selection = select_models(["en", "de"], CPU, None, models)
    assert selection.cleanup.model_id == "c"
    assert selection.cleanup.languages == ("en",)
    assert selection.cleanup.unscored == ("de",)


def test_a_model_scoring_more_of_the_languages_ranks_first():
    models = catalog(
        cleanup("narrow", {"en": 99}),
        cleanup("broad", {"en": 70, "de": 70}),
    )
    assert select_models(["en", "de"], GPU, None, models).cleanup.model_id == "broad"
    assert select_models(["en"], GPU, None, models).cleanup.model_id == "narrow"


def test_a_harmful_score_still_excludes_a_model_that_misses_another_language():
    models = catalog(cleanup("c", {"en": 95, "sq": 20}))
    assert select_models(["en", "sq", "de"], GPU, None, models).cleanup is None


def test_a_speech_model_on_the_llama_asr_runtime_is_chosen():
    models = catalog(
        asr("fast", {"en": 92}, runtime="llama-asr", cpu=1300),
        asr("slow", {"en": 80}, cpu=4500),
    )
    choice = select_models(["en"], CPU, None, models).asr[0]
    assert (choice.model_id, choice.runtime) == ("fast", "llama-asr")


def test_the_highest_lowest_language_quality_wins():
    models = catalog(
        cleanup("even", {"en": 80, "de": 80}),
        cleanup("lopsided", {"en": 99, "de": 75}),
    )
    assert select_models(["en", "de"], GPU, None, models).cleanup.model_id == "even"


def test_cleanup_ties_go_to_the_faster_then_the_smaller_then_the_earlier_model():
    faster = catalog(cleanup("a", {"en": 80}, gpu=700), cleanup("b", {"en": 80}, gpu=400))
    assert select_models(["en"], GPU, None, faster).cleanup.model_id == "b"
    smaller = catalog(cleanup("a", {"en": 80}, size=9), cleanup("b", {"en": 80}, size=3))
    assert select_models(["en"], GPU, None, smaller).cleanup.model_id == "b"
    earlier = catalog(cleanup("a", {"en": 80}), cleanup("b", {"en": 80}))
    assert select_models(["en"], GPU, None, earlier).cleanup.model_id == "a"


def test_a_model_over_the_cleanup_target_on_this_hardware_is_not_chosen():
    models = with_speech(
        cleanup("slow", {"en": 95}, display_name="Slow", cpu=CLEANUP_TARGET_MS + 1),
        cleanup("quick", {"en": 60}, display_name="Quick", cpu=CLEANUP_TARGET_MS),
    )
    selection = select_models(["en"], CPU, None, models)
    assert selection.cleanup.model_id == "quick"
    assert selection.cpu_cleanup_allowed is True
    assert selection.notes == (
        (
            "Slow would clean English better but takes about 1.8 s on the processor, "
            "over the 1.8 s goal."
        ),
    )


def test_a_better_model_that_cannot_run_here_is_named_as_such():
    models = with_speech(
        cleanup("gpu-only", {"en": 95}, display_name="Big", cpu=None),
        cleanup("small", {"en": 60}, display_name="Small"),
    )
    selection = select_models(["en"], CPU, None, models)
    assert selection.notes == ("Big would clean English better but cannot run on the processor.",)


def test_nothing_safe_means_no_cleanup_and_raw_transcripts():
    models = with_speech(cleanup("c", {"en": 90, "sq": 20}, display_name="C"))
    selection = select_models(["en", "sq"], GPU, None, models)
    assert selection.cleanup is None
    assert selection.cpu_cleanup_allowed is False
    assert selection.notes == (
        (
            "No installed cleanup model is safe for Albanian, so raw transcripts are delivered for "
            "English and Albanian."
        ),
    )


def test_models_safe_for_each_language_but_not_together_are_explained():
    models = with_speech(cleanup("a", {"en": 90, "sq": 10}), cleanup("b", {"en": 10, "sq": 90}))
    selection = select_models(["en", "sq"], GPU, None, models)
    assert selection.cleanup is None
    assert selection.notes == (
        (
            "No installed cleanup model is safe for English and Albanian together, so raw "
            "transcripts are delivered."
        ),
    )


def test_no_installed_cleanup_model_suggests_one_that_would_work():
    models = with_speech(cleanup("c", {"en": 90}, display_name="Clean"))
    selection = select_models(["en"], CPU, frozenset({"speech"}), models)
    assert selection.cleanup is None
    assert selection.notes == (
        "No cleanup model is installed, so raw transcripts are delivered for English.",
        "Clean would clean English safely but is not installed.",
    )


def test_only_slow_safe_models_installed_explains_the_goal():
    models = with_speech(cleanup("slow", {"en": 90}, display_name="Slow", cpu=2700))
    selection = select_models(["en"], CPU, None, models)
    assert selection.cleanup is None
    assert selection.notes == (
        (
            "Slow is safe for English but takes about 2.7 s on the processor, over the 1.8 s goal, "
            "so raw transcripts are delivered."
        ),
    )


def test_only_unsuitable_safe_models_installed_explains_the_hardware():
    models = with_speech(cleanup("big", {"en": 90}, display_name="Big", cpu=None))
    selection = select_models(["en"], CPU, None, models)
    assert selection.notes == (
        (
            "Big is safe for English but cannot run on the processor, so raw transcripts are "
            "delivered."
        ),
    )


def test_an_uninstalled_model_is_never_chosen():
    models = catalog(cleanup("best", {"en": 99}), cleanup("fine", {"en": 60}))
    selection = select_models(["en"], GPU, frozenset({"fine"}), models)
    assert selection.cleanup.model_id == "fine"


def test_a_cleanup_model_for_an_unsupported_runtime_is_never_chosen():
    models = catalog(cleanup("onnx", {"en": 99}, runtime="sherpa-onnx"))
    assert select_models(["en"], GPU, None, models).cleanup is None


def test_cpu_cleanup_is_allowed_only_for_a_processor_selection_within_the_target():
    models = catalog(cleanup("c", {"en": 90}, gpu=400, cpu=1130))
    assert select_models(["en"], CPU, None, models).cpu_cleanup_allowed is True
    assert select_models(["en"], GPU, None, models).cpu_cleanup_allowed is False


def test_duplicate_languages_count_once_and_no_language_selects_nothing():
    selection = select_models(["en", "en"], GPU)
    assert selection.asr[0].languages == ("en",)
    empty = select_models([], GPU)
    assert empty.asr == () and empty.cleanup is None and empty.notes == ()


# Contract --------------------------------------------------------------------------------------


def test_the_contract_names_the_ui_builds_against_stay():
    choice = ModelChoice(
        kind=ModelKind.ASR,
        model_id="m",
        display_name="M",
        languages=("en",),
        reason="r",
        installed=True,
    )
    selection = Selection(hardware=GPU, asr=(choice,), cleanup=None, notes=())
    assert selection.asr_for("en") is choice
    assert Hardware("gpu") is GPU and ModelKind("cleanup") is ModelKind.CLEANUP


def test_language_names_come_from_the_whisper_table():
    assert language_name("sq") == "Albanian"
    assert language_name("de") == "German"
    assert language_name("xx") == "xx"


def test_default_catalog_path_is_in_the_data_folder():
    assert modelcatalog.default_catalog_path().parts[-2:] == ("models", "catalog.json")
    assert isinstance(modelcatalog.default_catalog_path(), Path)


# The writing model for compose mode (B5-44) ------------------------------------------------------


def test_the_shipped_catalog_carries_a_writing_entry_beside_each_usable_cleanup_model():
    models = {model.id: model for model in load_catalog()}
    qwen = models[QWEN35_4B_WRITE]
    assert qwen.kind is ModelKind.COMPOSE
    assert qwen.runtime == "llama-server"
    assert qwen.files == models[QWEN35_4B].files
    assert qwen.size_bytes == models[QWEN35_4B].size_bytes == 2740937888
    assert dict(qwen.languages) == {"en": 90, "de": 85, "sq": 70}
    assert (qwen.latency_ms(CPU), qwen.latency_ms(GPU)) == (9000, 1800)
    gemma = models[GEMMA4_WRITE]
    assert gemma.files == models[GEMMA4].files
    assert gemma.size_bytes == models[GEMMA4].size_bytes == 2900717056
    assert dict(gemma.languages) == {"en": 75, "de": 75, "sq": 55}
    assert gemma.engine_args == models[GEMMA4].engine_args
    for model in (qwen, gemma):
        assert model.score("sq") >= SAFETY_FLOOR


def test_the_writing_scores_say_they_are_a_judgement_and_name_the_missing_benchmark():
    raw = json.loads(modelcatalog.default_catalog_path().read_text(encoding="utf-8"))
    assert "judgement" in raw["notes"]
    assert "writing benchmark is still missing" in raw["notes"]
    models = {model.id: model for model in load_catalog()}
    for model_id in (QWEN35_4B_WRITE, GEMMA4_WRITE):
        notes = models[model_id].notes
        assert "judgement from the cleanup benchmark, not a measurement" in notes
        assert "A writing benchmark is still missing." in notes


def test_the_best_writer_wins_on_both_kinds_of_computer_when_it_is_installed():
    for hardware in (GPU, CPU):
        selection = select_models(["en", "de", "sq"], hardware)
        assert selection.compose.model_id == QWEN35_4B_WRITE
        assert selection.compose.languages == ("en", "de", "sq")
        assert selection.compose.file == "Qwen3.5-4B-Q4_K_M.gguf"


def test_a_slow_writer_is_kept_because_composing_is_not_on_the_dictation_path():
    selection = select_models(["en", "de", "sq"], CPU)
    assert selection.compose.latency_ms == 9000
    assert selection.compose.latency_ms > CLEANUP_TARGET_MS
    assert selection.cleanup.model_id == GEMMA4


def test_the_processor_writes_with_its_cleanup_model_when_nothing_stronger_is_installed():
    installed = frozenset({WHISPER, QWEN_ASR, GEMMA4, GEMMA4_WRITE})
    selection = select_models(["en", "de", "sq"], CPU, installed)
    assert selection.cleanup.model_id == GEMMA4
    assert selection.compose.model_id == GEMMA4_WRITE
    assert selection.one_text_model is True
    assert selection.compose.reason == (
        "Writes English and German well and Albanian cautiously. "
        "About 6 s for a short email on the processor. "
        "It is the cleanup model as well, so nothing else is loaded."
    )


def test_a_stronger_writer_already_installed_is_used_even_on_a_processor():
    installed = frozenset({WHISPER, QWEN_ASR, GEMMA4, GEMMA4_WRITE, QWEN35_4B, QWEN35_4B_WRITE})
    selection = select_models(["en", "de", "sq"], CPU, installed)
    assert selection.cleanup.model_id == GEMMA4
    assert selection.compose.model_id == QWEN35_4B_WRITE
    assert selection.one_text_model is False


def test_a_writer_that_cannot_run_on_this_hardware_is_not_chosen():
    models = with_speech(
        compose("here", {"en": 90}, display_name="Here", cpu=None, gpu=4000),
        compose("elsewhere", {"en": 70}, display_name="Elsewhere", cpu=5000, gpu=5000),
    )
    assert select_models(["en"], CPU, all_ids(models), models).compose.model_id == "elsewhere"
    assert select_models(["en"], GPU, all_ids(models), models).compose.model_id == "here"


def test_a_writer_that_harms_an_enabled_language_is_never_chosen():
    models = with_speech(
        compose("harmful", {"en": 95, "sq": 20}, display_name="Harmful"),
        compose("safe", {"en": 70, "sq": 60}, display_name="Safe"),
    )
    selection = select_models(["en", "sq"], CPU, all_ids(models), models)
    assert selection.compose.model_id == "safe"


def test_the_smaller_writer_wins_a_tie():
    models = with_speech(
        compose("big", {"en": 80}, display_name="Big", size=900),
        compose("small", {"en": 80}, display_name="Small", size=100),
    )
    assert select_models(["en"], CPU, all_ids(models), models).compose.model_id == "small"


def test_no_writing_model_installed_says_so_without_mentioning_raw_transcripts():
    models = with_speech(
        cleanup("c", {"en": 90}, display_name="Clean"),
        compose("w", {"en": 80}, display_name="Writer"),
    )
    selection = select_models(["en"], CPU, frozenset({"speech", "c"}), models)
    assert selection.compose is None
    assert selection.notes == (
        "No installed model writes text, so compose mode is off for English.",
        "Writer would write English but is not installed.",
    )


def test_a_catalog_without_writing_models_says_nothing_about_compose_mode():
    models = with_speech(cleanup("c", {"en": 90}, display_name="Clean"))
    selection = select_models(["en"], CPU, all_ids(models), models)
    assert selection.compose is None
    assert selection.notes == ()


def test_one_text_model_is_true_only_when_the_two_choices_are_the_same_file():
    assert select_models(["en", "de", "sq"], GPU).one_text_model is True
    assert select_models(["en", "de", "sq"], CPU).one_text_model is False
    assert Selection(hardware=CPU, asr=(), cleanup=None, notes=()).one_text_model is False


# The size budget and the install plan (B5-46, B5-45) ---------------------------------------------


def test_two_entries_for_the_same_file_are_counted_once():
    models = {model.id: model for model in load_catalog()}
    pair = (models[QWEN35_4B], models[QWEN35_4B_WRITE])
    assert modelcatalog.model_bytes(pair) == 2740937888
    assert modelcatalog.model_bytes((models[GEMMA4], models[GEMMA4_WRITE])) == 2900717056
    assert modelcatalog.model_bytes((models[QWEN35_4B], models[GEMMA4])) == 2740937888 + 2900717056


def test_the_footprint_is_the_app_the_engines_the_vad_model_and_the_models():
    assert modelcatalog.BASE_INSTALL_BYTES == 320_979_273
    assert modelcatalog.VAD_MODEL_BYTES == 885_098
    assert modelcatalog.installed_footprint(()) == 321_864_371
    models = {model.id: model for model in load_catalog()}
    assert modelcatalog.installed_footprint((models[WHISPER],)) == 321_864_371 + 874188075


def test_the_budget_is_five_billion_bytes():
    assert modelcatalog.INSTALL_BUDGET_BYTES == 5_000_000_000


@pytest.mark.parametrize(
    ("languages", "hardware", "model_ids", "footprint"),
    [
        (["en"], GPU, [QWEN_ASR, QWEN35_4B], 3_761_410_483),
        (["en"], CPU, [QWEN_ASR, GEMMA4], 3_921_189_651),
        (["en", "de"], GPU, [QWEN_ASR, QWEN35_4B], 3_761_410_483),
        (["en", "de"], CPU, [QWEN_ASR, GEMMA4], 3_921_189_651),
        (["sq"], GPU, [WHISPER, QWEN35_4B], 3_936_990_334),
        (["sq"], CPU, [WHISPER, GEMMA4], 4_096_769_502),
        (["en", "de", "sq"], GPU, [WHISPER, QWEN35_4B], 3_936_990_334),
        (["en", "de", "sq"], CPU, [QWEN_ASR, WHISPER, GEMMA4], 4_795_377_726),
    ],
)
def test_every_offered_install_fits_the_budget(languages, hardware, model_ids, footprint):
    plan = modelcatalog.plan_install(languages, hardware)
    kept = [model.id for model in plan.models if model.kind is not ModelKind.COMPOSE]
    assert kept == model_ids
    assert plan.footprint_bytes == footprint
    assert plan.fits is True
    assert plan.footprint_bytes <= modelcatalog.INSTALL_BUDGET_BYTES


def test_a_graphics_card_installs_one_model_for_cleaning_and_writing():
    plan = modelcatalog.plan_install(["en", "de", "sq"], GPU)
    assert plan.writing.id == QWEN35_4B
    assert plan.files == ("ggml-large-v3-turbo-q8_0.bin", "Qwen3.5-4B-Q4_K_M.gguf")
    selection = select_models(["en", "de", "sq"], GPU, frozenset(plan.model_ids))
    assert selection.cleanup.model_id == QWEN35_4B
    assert selection.compose.model_id == QWEN35_4B_WRITE
    assert selection.one_text_model is True


def test_a_processor_installs_the_small_fast_model_for_both_jobs():
    plan = modelcatalog.plan_install(["en", "de"], CPU)
    assert plan.writing.id == GEMMA4
    selection = select_models(["en", "de"], CPU, frozenset(plan.model_ids))
    assert selection.cleanup.model_id == GEMMA4
    assert selection.compose.model_id == GEMMA4_WRITE
    assert selection.one_text_model is True


def test_the_processor_keeps_the_fast_english_and_german_model_beside_albanian():
    plan = modelcatalog.plan_install(["en", "de", "sq"], CPU)
    assert [model.id for model in plan.dropped] == []
    assert plan.footprint_bytes == 4_795_377_726
    assert plan.fits is True
    assert plan.files == (
        "Qwen3-ASR-0.6B-Q4_K_M.gguf",
        "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf",
        "ggml-large-v3-turbo-q8_0.bin",
        "gemma-4-E2B-it-Q4_0.gguf",
        "mtp-gemma-4-E2B-it-Q4_0.gguf",
    )
    selection = select_models(["en", "de", "sq"], CPU, frozenset(plan.model_ids))
    assert [(c.model_id, c.languages) for c in selection.asr] == [
        (QWEN_ASR, ("en", "de")),
        (WHISPER, ("sq",)),
    ]
    assert selection.cleanup.model_id == GEMMA4
    assert plan.reason == (
        "No graphics card was found, so Spells takes the small fast model, Gemma 4 E2B, 2.9 GB.")


@pytest.mark.parametrize("hardware", [CPU, GPU])
def test_no_plan_installs_the_q8_0_file_any_more(hardware):
    for languages in (["en"], ["de"], ["sq"], ["en", "de"], ["en", "sq"], ["de", "sq"],
                      ["en", "de", "sq"]):
        plan = modelcatalog.plan_install(languages, hardware)
        assert QWEN_ASR_Q8 not in plan.model_ids
        assert "Qwen3-ASR-0.6B-Q8_0.gguf" not in plan.files
        assert plan.fits is True


def test_the_budget_drops_a_speed_model_and_never_a_language():
    before_q4 = tuple(model for model in load_catalog() if model.id != QWEN_ASR)
    plan = modelcatalog.plan_install(["en", "de", "sq"], CPU, before_q4)
    assert [model.id for model in plan.dropped] == [QWEN_ASR_Q8]
    assert plan.footprint_bytes == 4_096_769_502
    selection = select_models(["en", "de", "sq"], CPU, frozenset(plan.model_ids), before_q4)
    assert [choice.languages for choice in selection.asr] == [("en", "de", "sq")]
    assert selection.cleanup is not None
    assert plan.reason.endswith(
        "English, German and Albanian together need more than 5 GB with Qwen3-ASR 0.6B as well, "
        "so speech runs on Whisper large-v3 turbo alone.")


def test_the_q8_0_processor_set_is_what_the_budget_refuses_and_the_q4_k_m_set_fits():
    models = {model.id: model for model in load_catalog()}
    untrimmed = (models[WHISPER], models[QWEN_ASR_Q8], models[GEMMA4], models[GEMMA4_WRITE])
    assert modelcatalog.installed_footprint(untrimmed) == 5_115_911_230
    assert modelcatalog.installed_footprint(untrimmed) > modelcatalog.INSTALL_BUDGET_BYTES
    smaller = (models[WHISPER], models[QWEN_ASR], models[GEMMA4], models[GEMMA4_WRITE])
    assert modelcatalog.installed_footprint(smaller) == 4_795_377_726
    assert modelcatalog.installed_footprint(smaller) <= modelcatalog.INSTALL_BUDGET_BYTES


def test_a_plan_never_names_a_model_that_may_not_ship():
    for hardware in (GPU, CPU):
        plan = modelcatalog.plan_install(["sq"], hardware)
        assert FLUTRA not in plan.model_ids
        assert plan.model_ids[0] == WHISPER


def test_a_plan_that_cannot_fit_says_so_rather_than_dropping_a_language():
    models = catalog(
        asr("huge-en", {"en": 90}, cpu=1000, size=4_000_000_000, display_name="Huge English"),
        asr("huge-sq", {"sq": 90}, cpu=1000, size=4_000_000_000, display_name="Huge Albanian"),
        cleanup("c", {"en": 90, "sq": 90}, display_name="Clean", size=1),
        compose("w", {"en": 90, "sq": 90}, display_name="Writer", size=1),
    )
    plan = modelcatalog.plan_install(["en", "sq"], CPU, models)
    assert plan.fits is False
    assert [model.id for model in plan.dropped] == []
    assert set(plan.model_ids) >= {"huge-en", "huge-sq"}


def test_the_plan_reason_is_one_plain_line_with_the_model_and_its_size():
    assert modelcatalog.plan_install(["en"], GPU).reason == (
        "Your graphics card can run the better writing model, Qwen3.5 4B, 2.7 GB.")
    assert modelcatalog.plan_install(["en"], CPU).reason == (
        "No graphics card was found, so Spells takes the small fast model, Gemma 4 E2B, 2.9 GB.")
    for languages in (["en"], ["sq"], ["en", "de", "sq"]):
        for hardware in (GPU, CPU):
            reason = modelcatalog.plan_install(languages, hardware).reason
            assert EM_DASH not in reason
            assert reason.endswith(".")
