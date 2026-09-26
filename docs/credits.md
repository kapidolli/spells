# Models and credits

Spells stands on the work of others. Every third-party licence text ships in the `licenses` folder of each installation and is shown on the About page.

## Models

| Model | Published by | Licence | Used for |
|---|---|---|---|
| Qwen3-ASR 0.6B, Q4_K_M | Qwen ([Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)) | Apache-2.0 | Speech recognition for English and German |
| Qwen3-ASR 0.6B audio projector, Q8_0 | ggml-org ([Qwen3-ASR-0.6B-GGUF](https://huggingface.co/ggml-org/Qwen3-ASR-0.6B-GGUF)) | Apache-2.0 | The audio encoder Qwen3-ASR needs |
| Whisper large-v3 turbo, q8_0 | OpenAI; ggml conversion by [whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp) | MIT | Speech recognition for Albanian and other languages |
| Silero VAD v5.1.2 | Silero; ggml conversion by [ggml-org/whisper-vad](https://huggingface.co/ggml-org/whisper-vad) | MIT | Finding the speech in a recording for Whisper |
| Gemma 4 E2B, Q4_0, with its MTP draft head | Google; GGUF by [ggml-org](https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF) | Apache-2.0 | Cleanup and writing on a PC without a graphics card, and in the offline installer |
| Qwen3.5 4B, Q4_K_M | Qwen; GGUF by [Unsloth](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF) | Apache-2.0 | Cleanup and writing on a PC with a graphics card |

Nobody publishes Qwen3-ASR 0.6B at Q4_K_M, so the file Spells uses is this project's own quantisation of [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) (Apache-2.0), made from the bf16 weights with llama.cpp b10997 (`bench/convert_qwen3_asr.py`). It recognises as well as the Q8_0 file and is 320 MB smaller.

## Engines and libraries

| Component | Licence | Role |
|---|---|---|
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp) v1.9.4, with ggml | MIT | `whisper-server`, built from source for Vulkan and for the processor |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) b10997 | MIT | `llama-server`, the official Windows Vulkan and processor builds |
| cpp-httplib 0.20.0, nlohmann/json 3.11.2 | MIT | Compiled into `whisper-server` |
| GCC runtime (libgcc, libstdc++) | GPLv3 with the GCC Runtime Library Exception 3.1 | Statically linked into `whisper-server` |
| winpthreads | MIT and BSD-3-Clause | Statically linked into `whisper-server` |
| LLVM OpenMP | Apache-2.0 with LLVM exception | `libomp.dll` beside `llama-server` |
| Python 3.12 | PSF License | The app runtime |
| PySide6 and Qt 6.11 | LGPLv3 | The user interface |
| sounddevice 0.5.6 and PortAudio | MIT, MIT-style | Microphone capture |
| PyInstaller 6.22.3 bootloader | GPL-2.0-or-later with the bootloader exception | Starts the packaged app |

The installers are built with [Inno Setup](https://jrsoftware.org/isinfo.php). The Vulkan loader comes with your graphics driver and is not shipped.
