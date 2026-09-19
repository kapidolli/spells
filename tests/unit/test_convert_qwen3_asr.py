"""bench/convert_qwen3_asr.py: pure logic only, no network, no torch import at test time."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import ClassVar

import pytest

BENCH_DIR = Path(__file__).resolve().parents[2] / "bench"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import convert_qwen3_asr as C

EM_DASH = "—"


class FakeRunner:
    def __init__(self, result=None):
        self.calls = []
        self.result = result

    def __call__(self, cmd, check=True):
        self.calls.append((list(cmd), check))
        return self.result


def _write(path, content=b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _make_hf_dir(tmp_path, *, vocab="split", weights="safetensors"):
    model_dir = tmp_path / "hf-model"
    _write(model_dir / "config.json")
    _write(model_dir / "tokenizer_config.json")
    if vocab == "split":
        _write(model_dir / "vocab.json")
        _write(model_dir / "merges.txt")
    elif vocab == "fast":
        _write(model_dir / "tokenizer.json")
    if weights == "safetensors":
        _write(model_dir / "model.safetensors")
    elif weights == "bin":
        _write(model_dir / "pytorch_model.bin")
    return model_dir


# ------------------------------------------------------------------ em dash guard


def test_module_source_has_no_em_dash():
    text = (BENCH_DIR / "convert_qwen3_asr.py").read_text(encoding="utf-8")
    assert EM_DASH not in text


# ------------------------------------------------------------------ input folder checks


def test_missing_input_files_on_complete_folder_is_empty(tmp_path):
    model_dir = _make_hf_dir(tmp_path)
    assert C.missing_input_files(model_dir) == []


def test_missing_input_files_on_fast_tokenizer_folder_is_empty(tmp_path):
    model_dir = _make_hf_dir(tmp_path, vocab="fast")
    assert C.missing_input_files(model_dir) == []


def test_missing_input_files_on_bin_weights_is_empty(tmp_path):
    model_dir = _make_hf_dir(tmp_path, weights="bin")
    assert C.missing_input_files(model_dir) == []


def test_missing_input_files_names_missing_config(tmp_path):
    model_dir = tmp_path / "empty"
    model_dir.mkdir()
    missing = C.missing_input_files(model_dir)
    assert "config.json" in missing
    assert "tokenizer_config.json" in missing
    assert any("tokenizer" in item for item in missing)
    assert any("weights" in item for item in missing)


def test_missing_input_files_names_missing_weights_only(tmp_path):
    model_dir = _make_hf_dir(tmp_path)
    (model_dir / "model.safetensors").unlink()
    missing = C.missing_input_files(model_dir)
    assert missing == ["model weights (*.safetensors or pytorch_model.bin)"]


def test_require_input_dir_passes_on_complete_folder(tmp_path):
    model_dir = _make_hf_dir(tmp_path)
    C.require_input_dir(model_dir)  # does not raise


def test_require_input_dir_refuses_politely_and_names_files(tmp_path):
    model_dir = tmp_path / "empty"
    model_dir.mkdir()
    with pytest.raises(C.ConvertError) as excinfo:
        C.require_input_dir(model_dir)
    message = str(excinfo.value)
    assert "config.json" in message
    assert "tokenizer_config.json" in message


def test_require_input_dir_refuses_when_not_a_folder(tmp_path):
    missing = tmp_path / "does-not-exist"
    with pytest.raises(C.ConvertError):
        C.require_input_dir(missing)


# ------------------------------------------------------------------ naming


def test_default_prefix_from_input_dir_uses_folder_name(tmp_path):
    model_dir = tmp_path / "Qwen3-ASR-0.6B"
    assert C.default_prefix_from_input_dir(model_dir) == "Qwen3-ASR-0.6B"


def test_default_prefix_from_input_dir_sanitises_odd_characters(tmp_path):
    model_dir = tmp_path / "Kushtrim Qwen3-ASR-0.6B Albanian 728h!!"
    prefix = C.default_prefix_from_input_dir(model_dir)
    assert " " not in prefix
    assert "!" not in prefix


def test_slugify_id_is_lowercase_and_keeps_dots_and_underscores():
    assert C.slugify_id("Qwen3-ASR-0.6B-Albanian-728h", "Q4_K_M") == (
        "qwen3-asr-0.6b-albanian-728h-q4_k_m"
    )


def test_slugify_id_matches_the_shipped_catalog_style():
    assert C.slugify_id("Qwen3-ASR-0.6B", "Q8_0") == "qwen3-asr-0.6b-q8_0"


def test_slugify_id_collapses_repeated_separators():
    assert "--" not in C.slugify_id("a  b", "Q4_K_M")


# ------------------------------------------------------------------ sizes and hashes


@pytest.mark.parametrize(
    "num_bytes,expected",
    [
        (0, "0 B"),
        (512, "512 B"),
        (1536, "1.5 KB"),
        (5 * 1024 * 1024, "5.0 MB"),
        (2 * 1024 * 1024 * 1024, "2.0 GB"),
    ],
)
def test_human_size_formats_common_ranges(num_bytes, expected):
    assert C.human_size(num_bytes) == expected


def test_sha256_file_streams_and_matches_hashlib(tmp_path):
    payload = tmp_path / "payload.bin"
    payload.write_bytes(bytes(range(256)) * 4096)
    expected = hashlib.sha256(payload.read_bytes()).hexdigest()
    assert C.sha256_file(payload) == expected
    assert len(C.sha256_file(payload)) == 64


# ------------------------------------------------------------------ pins.json tag reading


def test_read_llama_cpp_tag_reads_pinned_version(tmp_path):
    pins_path = tmp_path / "pins.json"
    pins_path.write_text(
        json.dumps({"binaries": {"llama_cpp_cpu": {"version": "b10997"}}}), encoding="utf-8"
    )
    assert C.read_llama_cpp_tag(pins_path) == "b10997"


def test_read_llama_cpp_tag_refuses_when_missing(tmp_path):
    pins_path = tmp_path / "pins.json"
    pins_path.write_text(json.dumps({"binaries": {}}), encoding="utf-8")
    with pytest.raises(C.ConvertError):
        C.read_llama_cpp_tag(pins_path)


def test_read_llama_cpp_tag_refuses_on_bad_json(tmp_path):
    pins_path = tmp_path / "pins.json"
    pins_path.write_text("not json", encoding="utf-8")
    with pytest.raises(C.ConvertError):
        C.read_llama_cpp_tag(pins_path)


def test_llama_cpp_archive_url_uses_the_tag():
    url = C.llama_cpp_archive_url("b10997")
    assert url == "https://github.com/ggml-org/llama.cpp/archive/refs/tags/b10997.tar.gz"


# ------------------------------------------------------------------ command construction


def test_text_convert_command_order():
    outfile = Path("out/text-bf16.gguf")
    cmd = C.text_convert_command(Path("py.exe"), Path("convert.py"), Path("hf"), outfile)
    assert cmd == [
        "py.exe", "convert.py", "hf",
        "--outfile", str(outfile), "--outtype", "bf16",
    ]


def test_mmproj_convert_command_order():
    outfile = Path("out/mmproj-q8_0.gguf")
    cmd = C.mmproj_convert_command(Path("py.exe"), Path("convert.py"), Path("hf"), outfile)
    assert cmd == [
        "py.exe", "convert.py", "hf", "--mmproj",
        "--outfile", str(outfile), "--outtype", "q8_0",
    ]


def test_quantize_command_order():
    cmd = C.quantize_command(Path("quantize.exe"), Path("in.gguf"), Path("out.gguf"), "Q4_K_M", 6)
    assert cmd == ["quantize.exe", "in.gguf", "out.gguf", "Q4_K_M", "6"]


def test_server_command_order():
    cmd = C.server_command(Path("server.exe"), Path("text.gguf"), Path("mmproj.gguf"), 8831, 6)
    assert cmd == [
        "server.exe",
        "-m", "text.gguf",
        "--mmproj", "mmproj.gguf",
        "--host", "127.0.0.1",
        "--port", "8831",
        "-t", "6",
        "-ngl", "0",
        "-c", "4096",
        "--no-webui",
    ]


# ------------------------------------------------------------------ runner-based steps


def test_run_text_convert_creates_out_dir_and_calls_runner(tmp_path):
    outfile = tmp_path / "out" / "text-bf16.gguf"
    runner = FakeRunner()
    result = C.run_text_convert(Path("py.exe"), Path("convert.py"), tmp_path / "hf", outfile, runner)
    assert outfile.parent.is_dir()
    assert result == outfile
    assert len(runner.calls) == 1
    assert runner.calls[0][0][0] == "py.exe"
    assert runner.calls[0][1] is True


def test_run_mmproj_convert_creates_out_dir_and_calls_runner(tmp_path):
    outfile = tmp_path / "out" / "mmproj-q8_0.gguf"
    runner = FakeRunner()
    result = C.run_mmproj_convert(Path("py.exe"), Path("convert.py"), tmp_path / "hf", outfile, runner)
    assert outfile.parent.is_dir()
    assert result == outfile
    assert "--mmproj" in runner.calls[0][0]


def test_run_quantize_creates_out_dir_and_calls_runner(tmp_path):
    dst = tmp_path / "out" / "text-Q4_K_M.gguf"
    runner = FakeRunner()
    result = C.run_quantize(Path("quantize.exe"), tmp_path / "text-bf16.gguf", dst, "Q4_K_M", 6, runner)
    assert dst.parent.is_dir()
    assert result == dst
    assert runner.calls[0][0] == [
        "quantize.exe", str(tmp_path / "text-bf16.gguf"), str(dst), "Q4_K_M", "6",
    ]


# ------------------------------------------------------------------ fetching llama.cpp (fakes only)


def test_conversion_ready_false_when_nothing_cached(tmp_path):
    assert C.conversion_ready(tmp_path, "b10997") is False


def test_conversion_ready_true_once_all_three_pieces_exist(tmp_path):
    convert_script, gguf_py, conversion_pkg = C.conversion_paths(tmp_path, "b10997")
    _write(convert_script)
    _write(gguf_py / "gguf" / "__init__.py")
    _write(conversion_pkg / "__init__.py")
    assert C.conversion_ready(tmp_path, "b10997") is True


def test_fetch_llama_cpp_skips_download_when_already_cached(tmp_path):
    convert_script, gguf_py, conversion_pkg = C.conversion_paths(tmp_path, "b10997")
    _write(convert_script)
    _write(gguf_py / "gguf" / "__init__.py")
    _write(conversion_pkg / "__init__.py")

    def fail_downloader(url, dest):
        raise AssertionError("should not download when already cached")

    result = C.fetch_llama_cpp(tmp_path, "b10997", downloader=fail_downloader)
    assert result == convert_script


def test_fetch_llama_cpp_downloads_and_extracts_when_missing(tmp_path):
    calls = {}

    def fake_downloader(url, dest):
        calls["url"] = url
        dest.write_bytes(b"fake archive bytes")

    def fake_extractor(archive_path, tag, cache_dir):
        calls["archive_path"] = archive_path
        calls["tag"] = tag
        convert_script, gguf_py, conversion_pkg = C.conversion_paths(cache_dir, tag)
        _write(convert_script)
        _write(gguf_py / "gguf" / "__init__.py")
        _write(conversion_pkg / "__init__.py")

    result = C.fetch_llama_cpp(
        tmp_path, "b10997", downloader=fake_downloader, extractor=fake_extractor
    )
    assert calls["url"] == C.llama_cpp_archive_url("b10997")
    assert calls["tag"] == "b10997"
    assert result.is_file()


def test_fetch_llama_cpp_refuses_when_extraction_leaves_pieces_missing(tmp_path):
    def fake_downloader(url, dest):
        dest.write_bytes(b"fake")

    def fake_extractor(archive_path, tag, cache_dir):
        pass  # extracts nothing

    with pytest.raises(C.ConvertError):
        C.fetch_llama_cpp(tmp_path, "b10997", downloader=fake_downloader, extractor=fake_extractor)


def test_extract_conversion_sources_writes_only_wanted_members(tmp_path):
    import tarfile

    archive_path = tmp_path / "src.tar.gz"
    src_dir = tmp_path / "src"
    _write(src_dir / "llama.cpp-b10997" / "convert_hf_to_gguf.py", b"print('hi')")
    _write(src_dir / "llama.cpp-b10997" / "gguf-py" / "gguf" / "__init__.py")
    _write(src_dir / "llama.cpp-b10997" / "conversion" / "__init__.py")
    _write(src_dir / "llama.cpp-b10997" / "tools" / "unwanted.cpp")
    with tarfile.open(archive_path, mode="w:gz") as archive:
        archive.add(src_dir / "llama.cpp-b10997", arcname="llama.cpp-b10997")

    cache_dir = tmp_path / "cache"
    C.extract_conversion_sources(archive_path, "b10997", cache_dir)

    convert_script, gguf_py, conversion_pkg = C.conversion_paths(cache_dir, "b10997")
    assert convert_script.read_bytes() == b"print('hi')"
    assert (gguf_py / "gguf" / "__init__.py").is_file()
    assert (conversion_pkg / "__init__.py").is_file()
    assert not (cache_dir / "llama.cpp-b10997" / "tools").exists()


# ------------------------------------------------------------------ server verification (fakes only)


class FakePopen:
    instances: ClassVar[list] = []

    def __init__(self, cmd, **kwargs):
        self.cmd = cmd
        self.kwargs = kwargs
        self.terminated = False
        self.waited = False
        FakePopen.instances.append(self)

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        self.waited = True

    def kill(self):
        pass


def test_verify_loads_returns_true_and_terminates_process(tmp_path):
    FakePopen.instances.clear()
    result = C.verify_loads(
        Path("server.exe"), Path("text.gguf"), Path("mmproj.gguf"),
        port=8831, threads=6, popen=FakePopen, health_check=lambda port: True,
    )
    assert result is True
    assert len(FakePopen.instances) == 1
    assert FakePopen.instances[0].terminated is True
    assert FakePopen.instances[0].waited is True


def test_verify_loads_returns_false_when_unhealthy_but_still_terminates(tmp_path):
    FakePopen.instances.clear()
    result = C.verify_loads(
        Path("server.exe"), Path("text.gguf"), Path("mmproj.gguf"),
        port=8831, threads=6, popen=FakePopen, health_check=lambda port: False,
    )
    assert result is False
    assert FakePopen.instances[0].terminated is True


def test_verify_loads_kills_process_on_timeout(tmp_path):
    class SlowPopen(FakePopen):
        def wait(self, timeout=None):
            self.waited = True
            if not getattr(self, "killed", False):
                raise subprocess.TimeoutExpired(cmd="server.exe", timeout=timeout)

        def kill(self):
            self.killed = True

    FakePopen.instances.clear()
    C.verify_loads(
        Path("server.exe"), Path("text.gguf"), Path("mmproj.gguf"),
        port=8831, threads=6, popen=SlowPopen, health_check=lambda port: True,
    )
    assert FakePopen.instances[0].killed is True


# ------------------------------------------------------------------ catalog entry rendering


def test_catalog_entry_fields_has_exactly_the_requested_keys():
    fields = C.catalog_entry_fields(
        "Qwen3-ASR-0.6B-Albanian-728h", "Q4_K_M",
        "Qwen3-ASR-0.6B-Albanian-728h-Q4_K_M.gguf",
        "mmproj-Qwen3-ASR-0.6B-Albanian-728h-Q8_0.gguf",
        400_000_000, 214_000_000,
    )
    assert set(fields) == {"id", "file", "extra_files", "size_bytes", "runtime", "engine_args"}
    assert fields["file"] == "Qwen3-ASR-0.6B-Albanian-728h-Q4_K_M.gguf"
    assert fields["extra_files"] == ["mmproj-Qwen3-ASR-0.6B-Albanian-728h-Q8_0.gguf"]
    assert fields["size_bytes"] == 614_000_000
    assert fields["runtime"] == "llama-asr"
    assert fields["engine_args"] == ["--mmproj", "{extra:0}", "--no-webui"]


def test_render_catalog_entry_is_parseable_json():
    fields = C.catalog_entry_fields("prefix", "Q4_K_M", "a.gguf", "b.gguf", 1, 2)
    rendered = C.render_catalog_entry(fields)
    assert json.loads(rendered) == fields


# ------------------------------------------------------------------ argument parsing


def test_parse_args_required_fields():
    args = C.parse_args(["--input-dir", "hf", "--output-dir", "out"])
    assert args.input_dir == Path("hf")
    assert args.output_dir == Path("out")
    assert args.quant == "Q4_K_M"
    assert args.name_prefix is None
    assert args.threads == 6
    assert args.skip_verify is False


def test_parse_args_overrides():
    args = C.parse_args(
        [
            "--input-dir", "hf",
            "--output-dir", "out",
            "--quant", "Q5_K_M",
            "--name-prefix", "custom-prefix",
            "--threads", "4",
            "--skip-verify",
        ]
    )
    assert args.quant == "Q5_K_M"
    assert args.name_prefix == "custom-prefix"
    assert args.threads == 4
    assert args.skip_verify is True


def test_parse_args_requires_input_and_output_dir():
    with pytest.raises(SystemExit):
        C.parse_args([])


# ------------------------------------------------------------------ main() orchestration (stubbed)


def test_main_refuses_politely_on_incomplete_input_dir(tmp_path, capsys):
    model_dir = tmp_path / "incomplete"
    model_dir.mkdir()
    rc = C.main(["--input-dir", str(model_dir), "--output-dir", str(tmp_path / "out")])
    assert rc == 2
    captured = capsys.readouterr()
    assert "config.json" in captured.err


def test_main_runs_the_full_pipeline_and_prints_catalog_entry(tmp_path, monkeypatch, capsys):
    model_dir = _make_hf_dir(tmp_path)
    output_dir = tmp_path / "out"
    pins_path = tmp_path / "pins.json"
    pins_path.write_text(
        json.dumps({"binaries": {"llama_cpp_cpu": {"version": "b10997"}}}), encoding="utf-8"
    )

    monkeypatch.setattr(C, "fetch_llama_cpp", lambda cache_dir, tag: Path("convert_hf_to_gguf.py"))

    def fake_text_convert(python_exe, convert_script, input_dir, outfile, runner=subprocess.run):
        outfile.parent.mkdir(parents=True, exist_ok=True)
        outfile.write_bytes(b"0" * 4096)
        return outfile

    def fake_mmproj_convert(python_exe, convert_script, input_dir, outfile, runner=subprocess.run):
        outfile.parent.mkdir(parents=True, exist_ok=True)
        outfile.write_bytes(b"1" * 2048)
        return outfile

    def fake_quantize(quantize_exe, src, dst, quant, threads, runner=subprocess.run):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"2" * 1024)
        return dst

    monkeypatch.setattr(C, "run_text_convert", fake_text_convert)
    monkeypatch.setattr(C, "run_mmproj_convert", fake_mmproj_convert)
    monkeypatch.setattr(C, "run_quantize", fake_quantize)
    monkeypatch.setattr(C, "verify_loads", lambda *a, **k: True)

    rc = C.main(
        [
            "--input-dir", str(model_dir),
            "--output-dir", str(output_dir),
            "--pins-path", str(pins_path),
            "--cache-dir", str(tmp_path / "cache"),
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "sha256" in captured.out
    assert "catalog entry fields:" in captured.out
    assert '"runtime": "llama-asr"' in captured.out
    assert "Q4_K_M.gguf" in captured.out
    assert "mmproj-" in captured.out


def test_main_returns_nonzero_when_verification_fails(tmp_path, monkeypatch, capsys):
    model_dir = _make_hf_dir(tmp_path)
    pins_path = tmp_path / "pins.json"
    pins_path.write_text(
        json.dumps({"binaries": {"llama_cpp_cpu": {"version": "b10997"}}}), encoding="utf-8"
    )

    monkeypatch.setattr(C, "fetch_llama_cpp", lambda cache_dir, tag: Path("convert_hf_to_gguf.py"))

    def fake_text_convert(python_exe, convert_script, input_dir, outfile, runner=subprocess.run):
        outfile.parent.mkdir(parents=True, exist_ok=True)
        outfile.write_bytes(b"0")
        return outfile

    def fake_mmproj_convert(python_exe, convert_script, input_dir, outfile, runner=subprocess.run):
        outfile.parent.mkdir(parents=True, exist_ok=True)
        outfile.write_bytes(b"1")
        return outfile

    def fake_quantize(quantize_exe, src, dst, quant, threads, runner=subprocess.run):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"2")
        return dst

    monkeypatch.setattr(C, "run_text_convert", fake_text_convert)
    monkeypatch.setattr(C, "run_mmproj_convert", fake_mmproj_convert)
    monkeypatch.setattr(C, "run_quantize", fake_quantize)
    monkeypatch.setattr(C, "verify_loads", lambda *a, **k: False)

    rc = C.main(
        [
            "--input-dir", str(model_dir),
            "--output-dir", str(tmp_path / "out"),
            "--pins-path", str(pins_path),
            "--cache-dir", str(tmp_path / "cache"),
        ]
    )
    assert rc == 3
    assert "did not load cleanly" in capsys.readouterr().err


def test_main_skips_verification_when_flag_set(tmp_path, monkeypatch):
    model_dir = _make_hf_dir(tmp_path)
    pins_path = tmp_path / "pins.json"
    pins_path.write_text(
        json.dumps({"binaries": {"llama_cpp_cpu": {"version": "b10997"}}}), encoding="utf-8"
    )

    monkeypatch.setattr(C, "fetch_llama_cpp", lambda cache_dir, tag: Path("convert_hf_to_gguf.py"))

    def fake_step(python_exe, convert_script, input_dir, outfile, runner=subprocess.run):
        outfile.parent.mkdir(parents=True, exist_ok=True)
        outfile.write_bytes(b"0")
        return outfile

    def fake_quantize(quantize_exe, src, dst, quant, threads, runner=subprocess.run):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"0")
        return dst

    monkeypatch.setattr(C, "run_text_convert", fake_step)
    monkeypatch.setattr(C, "run_mmproj_convert", fake_step)
    monkeypatch.setattr(C, "run_quantize", fake_quantize)

    def fail_verify(*a, **k):
        raise AssertionError("verify_loads should not run with --skip-verify")

    monkeypatch.setattr(C, "verify_loads", fail_verify)

    rc = C.main(
        [
            "--input-dir", str(model_dir),
            "--output-dir", str(tmp_path / "out"),
            "--pins-path", str(pins_path),
            "--cache-dir", str(tmp_path / "cache"),
            "--skip-verify",
        ]
    )
    assert rc == 0
