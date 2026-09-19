import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR / "bench") not in sys.path:
    sys.path.insert(0, str(REPO_DIR / "bench"))

import convert_whisper_hf as C


class FakeRunner:
    def __init__(self):
        self.calls = []

    def __call__(self, cmd, check=True):
        self.calls.append(list(cmd))


def test_convert_command_order():
    cmd = C.convert_command(
        Path("py.exe"), Path("convert.py"), Path("hf"), Path("whisper"), Path("out")
    )
    assert cmd == ["py.exe", "convert.py", "hf", "whisper", "out"]


def test_quantize_command_order():
    cmd = C.quantize_command(Path("quantize.exe"), Path("f16.bin"), Path("q8.bin"), "q8_0")
    assert cmd == ["quantize.exe", "f16.bin", "q8.bin", "q8_0"]


def test_run_conversion_creates_out_dir_and_calls_runner(tmp_path):
    out_dir = tmp_path / "ggml" / "candidate"
    runner = FakeRunner()
    result = C.run_conversion(
        Path("py.exe"), Path("convert.py"), tmp_path / "hf", tmp_path / "whisper", out_dir, runner
    )
    assert out_dir.is_dir()
    assert result == out_dir / "ggml-model.bin"
    assert len(runner.calls) == 1
    assert runner.calls[0][0] == "py.exe"
    assert runner.calls[0][-1] == str(out_dir)


def test_run_quantize_creates_parent_and_calls_runner(tmp_path):
    f16 = tmp_path / "ggml-model.bin"
    q_bin = tmp_path / "quantized" / "ggml-model-q8_0.bin"
    runner = FakeRunner()
    result = C.run_quantize(Path("quantize.exe"), f16, q_bin, "q8_0", runner)
    assert q_bin.parent.is_dir()
    assert result == q_bin
    assert runner.calls == [["quantize.exe", str(f16), str(q_bin), "q8_0"]]


def test_convert_and_quantize_chains_both_steps(tmp_path):
    out_dir = tmp_path / "out"
    runner = FakeRunner()
    result = C.convert_and_quantize(
        Path("py.exe"),
        Path("convert.py"),
        tmp_path / "hf",
        tmp_path / "whisper",
        out_dir,
        Path("quantize.exe"),
        "q8_0",
        runner,
    )
    assert result == out_dir / "ggml-model-q8_0.bin"
    assert len(runner.calls) == 2
    assert runner.calls[0][0] == "py.exe"
    assert runner.calls[1][0] == "quantize.exe"
    assert runner.calls[1][1] == str(out_dir / "ggml-model.bin")


def test_parse_args_required_fields():
    a = C.parse_args(
        [
            "--python-exe",
            "py.exe",
            "--convert-script",
            "convert.py",
            "--hf-dir",
            "hf",
            "--whisper-repo",
            "whisper",
            "--out-dir",
            "out",
            "--quantize-exe",
            "quantize.exe",
        ]
    )
    assert a.qtype == "q8_0"
    assert a.hf_dir == Path("hf")


def test_main_invokes_conversion_and_prints_path(tmp_path, capsys, monkeypatch):
    calls = []

    def fake_convert_and_quantize(*args, **kwargs):
        calls.append((args, kwargs))
        return tmp_path / "ggml-model-q8_0.bin"

    monkeypatch.setattr(C, "convert_and_quantize", fake_convert_and_quantize)
    rc = C.main(
        [
            "--python-exe",
            "py.exe",
            "--convert-script",
            "convert.py",
            "--hf-dir",
            "hf",
            "--whisper-repo",
            "whisper",
            "--out-dir",
            "out",
            "--quantize-exe",
            "quantize.exe",
        ]
    )
    assert rc == 0
    assert len(calls) == 1
    out = capsys.readouterr().out
    assert "ggml-model-q8_0.bin" in out
