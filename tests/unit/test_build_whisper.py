"""build/build_whisper.py: pure helpers (CMake output parser, objdump DLL checker)."""

import sys
from pathlib import Path

BUILD_DIR = Path(__file__).resolve().parents[2] / "build"
sys.path.insert(0, str(BUILD_DIR))

import build_whisper as bw

# Lines exactly as ggml/src/ggml-vulkan/CMakeLists.txt in whisper.cpp v1.9.4 prints them:
#   message(STATUS "${EXTENSION_NAME} supported by glslc")
#   message(STATUS "${EXTENSION_NAME} not supported by glslc")
ALL_SUPPORTED = """\
-- Vulkan found
-- GL_KHR_cooperative_matrix supported by glslc
-- GL_NV_cooperative_matrix2 supported by glslc
-- GL_NV_cooperative_matrix_decode_vector supported by glslc
-- GL_EXT_integer_dot_product supported by glslc
-- GL_EXT_bfloat16 supported by glslc
-- GL_EXT_float_e2m1 supported by glslc
-- GL_EXT_float_e4m3 supported by glslc
-- Configuring done (12.3s)
"""

SOME_UNSUPPORTED = """\
-- Vulkan found
-- GL_KHR_cooperative_matrix supported by glslc
-- GL_NV_cooperative_matrix2 not supported by glslc
-- GL_NV_cooperative_matrix_decode_vector supported by glslc
-- GL_EXT_integer_dot_product not supported by glslc
-- GL_EXT_bfloat16 not supported by glslc
"""

ONLY_DECODE_VECTOR = """\
-- Vulkan found
-- GL_NV_cooperative_matrix_decode_vector supported by glslc
"""


def test_parser_reports_all_required_features_supported():
    result = bw.parse_vulkan_feature_checks(ALL_SUPPORTED)
    assert result == {"coopmat": True, "coopmat2": True, "integer_dot": True}
    assert bw.missing_vulkan_features(result) == []


def test_parser_reports_unsupported_features():
    result = bw.parse_vulkan_feature_checks(SOME_UNSUPPORTED)
    assert result == {"coopmat": True, "coopmat2": False, "integer_dot": False}
    assert bw.missing_vulkan_features(result) == ["coopmat2", "integer_dot"]


def test_parser_treats_absent_lines_as_missing():
    result = bw.parse_vulkan_feature_checks("-- Vulkan found\n-- Configuring done\n")
    assert result == {"coopmat": None, "coopmat2": None, "integer_dot": None}
    assert bw.missing_vulkan_features(result) == ["coopmat", "coopmat2", "integer_dot"]


def test_decode_vector_line_does_not_count_as_coopmat2():
    result = bw.parse_vulkan_feature_checks(ONLY_DECODE_VECTOR)
    assert result["coopmat2"] is None


OBJDUMP_SYSTEM_ONLY = """

C:/x/whisper-server.exe:     file format pei-x86-64

Characteristics 0x22
\texecutable
\tlarge address aware

The Import Tables (interpreted .idata section contents)
 vma:            Hint    Time      Forward  DLL       First
                 Table   Stamp     Chain    Name      Thunk
 00000000004f2000 004f2ae0 00000000 00000000 004f3c60 004f2c58

\tDLL Name: KERNEL32.dll
\tvma:     Ordinal  Hint  Member-Name  Bound-To
\t4f3d4c\t      0\t  16f  CloseHandle

\tDLL Name: api-ms-win-crt-runtime-l1-1-0.dll
\tvma:     Ordinal  Hint  Member-Name  Bound-To
\t4f3f2c\t      0\t   1a  __p___argc

\tDLL Name: WS2_32.dll
\tvma:     Ordinal  Hint  Member-Name  Bound-To
\t4f4034\t      0\t    2  WSAStartup

\tDLL Name: vulkan-1.dll
\tvma:     Ordinal  Hint  Member-Name  Bound-To
\t4f40a0\t      0\t   10  vkCreateInstance
"""


def test_parse_dll_names_from_objdump_output():
    assert bw.parse_dll_names(OBJDUMP_SYSTEM_ONLY) == [
        "KERNEL32.dll",
        "api-ms-win-crt-runtime-l1-1-0.dll",
        "WS2_32.dll",
        "vulkan-1.dll",
    ]


def test_system_only_dll_list_is_accepted():
    names = [
        "KERNEL32.dll",
        "msvcrt.dll",
        "api-ms-win-crt-math-l1-1-0.dll",
        "WS2_32.dll",
        "ADVAPI32.dll",
        "bcrypt.dll",
    ]
    assert bw.foreign_dlls(names) == []
    assert bw.foreign_dlls(names, allow_vulkan=True) == []


def test_vulkan_loader_is_accepted_only_for_the_vulkan_variant():
    names = ["KERNEL32.dll", "WS2_32.dll", "vulkan-1.dll"]
    assert bw.foreign_dlls(names, allow_vulkan=True) == []
    assert bw.foreign_dlls(names, allow_vulkan=False) == ["vulkan-1.dll"]
    assert bw.is_system_dll("VULKAN-1.DLL", allow_vulkan=True) is True
    assert bw.is_system_dll("VULKAN-1.DLL", allow_vulkan=False) is False


def test_libstdcxx_is_rejected():
    names = ["KERNEL32.dll", "msvcrt.dll", "libstdc++-6.dll"]
    assert bw.foreign_dlls(names) == ["libstdc++-6.dll"]


def test_every_mingw_runtime_dll_is_rejected():
    names = [
        "KERNEL32.dll",
        "libgcc_s_seh-1.dll",
        "libwinpthread-1.dll",
        "libgomp-1.dll",
        "LIBSTDC++-6.DLL",
    ]
    assert bw.foreign_dlls(names) == [
        "libgcc_s_seh-1.dll",
        "libwinpthread-1.dll",
        "libgomp-1.dll",
        "LIBSTDC++-6.DLL",
    ]


def _write_cache(build_dir: Path, home_directory: str) -> Path:
    build_dir.mkdir(parents=True, exist_ok=True)
    cache = build_dir / "CMakeCache.txt"
    cache.write_text(
        "# This is the CMakeCache file.\n"
        "CMAKE_BUILD_TYPE:STRING=Release\n"
        f"CMAKE_HOME_DIRECTORY:INTERNAL={home_directory}\n",
        encoding="utf-8",
    )
    return cache


def test_stale_cache_from_a_moved_repository_is_removed(tmp_path):
    source_dir = tmp_path / "now" / "whisper.cpp-1.9.4"
    source_dir.mkdir(parents=True)
    build_dir = tmp_path / "whisper-build-cpu"
    _write_cache(build_dir, "C:/Users/someone/ws/before/build/cache/whisper.cpp-1.9.4")

    bw.discard_stale_cache(source_dir, build_dir)

    assert not build_dir.exists()


def test_matching_cache_is_kept(tmp_path):
    source_dir = tmp_path / "whisper.cpp-1.9.4"
    source_dir.mkdir()
    build_dir = tmp_path / "whisper-build-cpu"
    _write_cache(build_dir, source_dir.as_posix())
    (build_dir / "build.ninja").write_text("", encoding="utf-8")

    bw.discard_stale_cache(source_dir, build_dir)

    assert (build_dir / "build.ninja").exists()


def test_missing_cache_is_not_an_error(tmp_path):
    bw.discard_stale_cache(tmp_path / "src", tmp_path / "never-configured")


def test_prefix_path_points_at_the_ucrt64_root():
    args = bw.prefix_path_args()
    assert len(args) == 1
    prefix = args[0].removeprefix("-DCMAKE_PREFIX_PATH=")
    assert args[0].startswith("-DCMAKE_PREFIX_PATH=")
    assert prefix.endswith("/ucrt64"), prefix


# The llama.cpp Windows release zips are flat: llama-server.exe and a dozen other tools, each
# tool's own <name>-impl.dll, the shared libraries, and the bundled OpenMP licence.
FAKE_LLAMA_ZIP = [
    "llama-server.exe",
    "llama-server-impl.dll",
    "llama-cli.exe",
    "llama-cli-impl.dll",
    "llama-bench.exe",
    "llama-bench-impl.dll",
    "llama-quantize.exe",
    "llama-quantize-impl.dll",
    "llama-tokenize.exe",
    "llama.exe",
    "ggml-rpc-server.exe",
    "llama.dll",
    "llama-common.dll",
    "mtmd.dll",
    "libomp.dll",
    "ggml.dll",
    "ggml-base.dll",
    "ggml-rpc.dll",
    "ggml-vulkan.dll",
    "ggml-cpu-haswell.dll",
    "LICENSE-LLVM-OpenMP",
]


def test_only_llama_server_and_its_libraries_are_shipped():
    kept = [name for name in FAKE_LLAMA_ZIP if bw.is_shipped_llama_file(name)]
    assert kept == [
        "llama-server.exe",
        "llama-server-impl.dll",
        "llama.dll",
        "llama-common.dll",
        "mtmd.dll",
        "libomp.dll",
        "ggml.dll",
        "ggml-base.dll",
        "ggml-rpc.dll",
        "ggml-vulkan.dll",
        "ggml-cpu-haswell.dll",
        "LICENSE-LLVM-OpenMP",
    ]


def test_other_tools_and_their_impl_dlls_are_dropped():
    dropped = [name for name in FAKE_LLAMA_ZIP if not bw.is_shipped_llama_file(name)]
    assert dropped == [
        "llama-cli.exe",
        "llama-cli-impl.dll",
        "llama-bench.exe",
        "llama-bench-impl.dll",
        "llama-quantize.exe",
        "llama-quantize-impl.dll",
        "llama-tokenize.exe",
        "llama.exe",
        "ggml-rpc-server.exe",
    ]


def test_whisper_server_is_not_a_llama_file():
    assert bw.is_shipped_llama_file("whisper-server.exe") is False


def test_nested_or_escaping_names_are_never_shipped():
    for name in ("bin/llama-server.exe", r"..\llama-server.exe", "../llama.dll", "", "C:llama.dll"):
        assert bw.is_shipped_llama_file(name) is False, name


def test_strip_prefix_handles_flat_and_wrapped_zips():
    assert bw.strip_prefix(FAKE_LLAMA_ZIP) == ""
    assert bw.strip_prefix(["build/llama-server.exe", "build/llama.dll"]) == "build/"
