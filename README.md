# pc-nsf-hifigan.cpp

> **Languages:** [English](README.md) | [中文](README_CN.md)

**NSF-HiFiGAN (mini_nsf) vocoder on ggml**

An independent vocoder library for the DiffSinger `diffsinger.cpp` pipeline. This repository owns only the **ggml vocoder body**; the mini-nsf sine source generator is delegated to [KakaruHayate/libmininsf](https://github.com/KakaruHayate/libmininsf).

---

## Feature highlights

- **Native ggml ops** — sub-pixel upsample (phase-major, exact) via F32 `im2col + mul_mat` + graph interleave (default); `ggml_conv_transpose_1d` kept as legacy fallback; `ggml_mul_mat` for the source conv
- **Multi-backend** — weights auto-uploaded to backend buffers (CPU/Vulkan/CUDA/Metal); Metal uses a fused F32 implicit-GEMM direct convolution instead of materializing im2col
- **F16/F32 dual precision (no quantization)** — F32 line (weights+compute fp32, exact baseline) and F16 line (fp16 weights, reserved for future fp16/bf16 training pilots)
- **Mel front-ends** — `mel_nvstft` (DiffSinger hifigan front-end) + `MelExtractor` (ecosystem API parity)
- **CLI + CTest** — single/batch vocode + CTest golden checks (pure numpy reference, no model assets)

## Quick start

```bash
# 1. convert (PT checkpoint -> GGUF; needs torch + gguf)
python converter/convert_hifigan.py --ckpt model.ckpt --config config.json --out hifigan_f32.gguf --dtype F32
# The official OpenVPI pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.ckpt has
# a built-in preset, because its release archive does not include config.json:
python converter/convert_hifigan.py --ckpt pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.ckpt --out hifigan_f32.gguf --dtype F32

# 2. build (see BUILDING.md)
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release
cmake --build build -j

# 3. vocode (mel.bin + f0.bin -> out.wav)
build/bin/hifigan_cli hifigan_f32.gguf mel.bin f0.bin out.wav
#    fp16 line (needs a --dtype F16 GGUF):
# HF_PRECISION=F16 build/bin/hifigan_cli hifigan_f16.gguf mel.bin f0.bin out_f16.wav
```

## Ecosystem positioning

This repo lives in a small C++ ecosystem for DiffSinger-style edge deployment.
Training-side repos never use ggml — these C++ repos exist for inference-only
deployment.  The canonical upstreams are:

| Repository | Component | Relationship |
|---|---|---|
| **pc-nsf-hifigan.cpp** | NSF-HiFiGAN vocoder (this repo) | Core engine; depends on `libmininsf` for the sine source |
| [libmininsf](https://github.com/KakaruHayate/libmininsf) | mini-nsf sine source | Base component (pulled via FetchContent) |
| [ggml-audio-patch](https://github.com/KakaruHayate/ggml-audio-patch) | ggml patch set + audio backends | Provides `ggml_conv_direct_1d`, `ADD_LEAKY_RELU`, and Vulkan/CUDA/Metal kernels consumed by this repo |
| [game.cpp](https://github.com/KakaruHayate/game.cpp) | GAME (DiffSinger V3, score recognition) | Independent downstream that consumes this repo as a vocoder |

If you are cloning this repo to build, you also need the two dependency repos
above — see `docs/integrating_zh.md` §8 "Build from source".

## Model release

This repo is the **quantization exception** in the ecosystem: the vocoder is
numerically sensitive, so only two precisions are ever produced
(`converter/convert_hifigan.py --dtype F32|F16`); there is **no** `rec`/Q8
tier.

| Asset | Precision | When to use |
|---|---|---|
| `hifigan_f16.gguf` | F16 (fp16 line) | normal deployments (weights fp16) |
| `hifigan_f32.gguf` | F32 (exact line) | golden-grade regression baseline |

- Publish via GitHub **Releases**, never commit weights to the repo.
- Names always carry the precision suffix (`f16` / `f32`) — consumers glob by
  size (e.g. OpenUtau), so a bare `hifigan.gguf` would silently shadow the
  other precision.
- Release checklist: convert → F16 wav vs torch reference (CTest t01/t02 +
  `tests/gen_hifigan_golden.py`) → attach both assets to the release notes.
- Future fp16 *training* pilots are validated torch-side first; the inference
  interface stays F16/F32 only.

## Deployment note (speed baseline = ONNX/DML)

The **lowest-latency deployment path for this vocoder is an ONNX export on DirectML**
(about 282 ms for a 20 s clip, ≈71× realtime on an RTX 2070-class GPU; the raw
engine-to-engines number is in `docs/benchmarks.md` §1).  We invest in this ggml
engine not to chase that latency, but to stay a **reference / pure-inference,
backend-portable** implementation (CPU/Vulkan/CUDA/Metal) that is **numerically
equivalent to the torch reference at the EP-precision level** — corr 0.9999998460
on the fp32 line, inside the legitimate EP-noise band on both golden frames. The
remaining ~1.5× gap to DML is a deliberate, investigated, **frozen** decision (see
`docs/benchmarks.md` §6): each further speed-up either breaks the precision
contract or sinks into infra we do not want to own. Correctness, reproducibility,
no-GPU/edge deployment, and ecosystem integration are the goals; wall-clock parity
with ONNX/DML is explicitly not (pulled by `hifisampler`/`hachitune`).

## Integration (typical case: mel + f0 in, wav out)

See **[docs/integrating_zh.md](docs/integrating_zh.md)** for the hh-typical
contract (mel `[T, 128]` natural-log + f0 `[T]` Hz, both frame-aligned at hop=512
@44.1 kHz → wav `T*512` samples), the adapted model's parameter table, and the
common pitfalls (log10 vs ln, frame count mismatch, unvoiced=0). The integration
surface is `pc_nsf_hifigan::HifiganModel` + `hifigan_run`, exercised end-to-end by
`examples/external_consumer/`.

## Development guide

### For contributors

- **No quantization path** — this repository deliberately **does not quantize**: the vocoder is numerically sensitive; F16/F32 only. Future fp16 training pilots are validated torch-side first.
- **Dependency policy (D2-revised, 2026-08-30)** — ggml changes not accepted upstream go to `ggml-patch` (patch set). This repo consumes patch-provided ops (`ggml_conv_direct_1d[_fused]`, `ggml_add_leaky_relu`). Nine patches live in `./patches/` and are applied idempotently to stock ggml v0.19.0 on every configure (`cmake/ApplyGgmlPatches.cmake`), including upgrades of existing build trees. Patch 9 addresses Vulkan device-memory selection and convolution device limits; see [the investigation and validation](docs/vulkan_device_memory.md). Local pre-patched trees are detected and adopted (stamp directory `.pcnsf-patches/`).
- **Numeric gate before commit** — run `tests/` golden comparison (`gen_hifigan_golden.py` + CTest t01/t02) before committing; wav output must match the torch reference.
- **mininsf source sync** — source-generator changes live in `libmininsf`; this repo only consumes (FetchContent).

### Known pitfalls

- `ggml_conv_transpose_1d` output-length semantics differ from torch ConvTranspose1d padding conventions — verify lengths.
- Two mel front-ends, don't mix: `mel_nvstft` (DiffSinger-specific, reflect pad + center=False) vs `MelExtractor` (htk/slaney general).

## License

Code [MPL-2.0](LICENSE) (this repo), except the `patches/` directory, which is dual-licensed
`MIT OR Apache-2.0` (see `patches/LICENSE`) so the diffs stay acceptable to upstream ggml.

Model weights retain their own licenses: the DiffSinger official NSF-HiFiGAN weights are
CC BY-NC-SA 4.0 (see [NOTICE.md](NOTICE.md)); note the non-commercial restriction.

## Project layout

```
include/pc_nsf_hifigan/  public headers (gguf_model.h, mel.h, hifigan.h)
src/                     impl (gguf_model.cpp, mel.cpp, hifigan.cpp)
tools/                   CLI (hifigan_cli.cpp)
converter/               weight conversion scripts (convert_hifigan.py)
tests/                   golden comparison tests (numpy reference, no model assets)
examples/external_consumer  third-party integration example
third_party/             pocketfft_hdronly.h (vendored single header)
cmake/Dependencies.cmake dependency management (FetchContent)
```

[Building](BUILDING.md) · [Architecture](docs/hifigan.md) · [Benchmarks](docs/benchmarks.md) · [Integration](docs/integrating_zh.md) · [Tests](tests/README.md)
