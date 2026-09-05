# Vulkan device-memory regression and device limits

## Problem and fix

On Windows with an RTX 4060, a 512-frame F32 vocoder call took approximately
15.6 seconds. GPU timestamps attributed 15.58 of 15.90 seconds to the 92
direct-convolution operations. Disabling direct convolution reduced that to
approximately 5.3 seconds, but was not the underlying fix.

The same compiled convolution SPIR-V ran in approximately 0.066 ms in an
independent Vulkan harness, versus 2.7–4.9 ms for the same shape through ggml.
Choosing ordinary device-local VRAM in ggml removed that difference. An
independent allocation comparison reproduced the slowdown using memory type 4
(`DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT`), including when the allocation
was never mapped. Thus the observed issue follows the memory type, not merely
the map/unmap operation. Both types reported a device-local heap; this does
not establish why the driver implements their effective access differently.

Patch 9 changes discrete-GPU allocation preference to device-only local memory,
ranking compatible non-host-visible types first regardless of enumeration
order. The existing staging transfers handle host uploads and downloads.
Compatible host-visible device-local types remain fallback candidates, including
after allocation failure. UMA devices retain their existing host-visible
policy, and explicit host/system-memory overrides retain their roles.

This changes no convolution arithmetic, precision, shader source or public
vocoder API. Direct convolution remains enabled: with the memory issue removed,
it is faster than the im2col alternative on the tested device.

## Supporting different devices

The fix uses Vulkan memory flags and device limits, not a GPU-model allowlist.

| Device capability | Policy |
|---|---|
| Discrete GPU with device-only VRAM | Prefer that memory for model and graph buffers; stage host transfers. |
| No compatible device-only type, or allocation fails | Try other compatible device-local types. |
| UMA / integrated-memory device | Retain host-visible device-local preference. |
| 256 threads and at least 18,560 bytes of shared memory | Retain the preferred w128 convolution tile for larger channel counts. |
| 128 threads / 16 KiB shared memory | Use the existing e64 convolution tile, requiring 15,488 bytes. |
| Smaller supported tile required | Try e32/e16; report unsupported if none fits. |
| Apple Silicon using Metal | No change to backend selection or Metal implementation. |

The convolution shader uses ordinary F32 arithmetic; the fix does not require
NVIDIA cooperative-matrix extensions. Tile indexing uses logical groups of 32
threads, not hardware subgroup shuffle operations. Capability tests check the
device's workgroup invocation count, local X limit and shared-memory size.

This is capability-aware selection, not runtime autotuning. It does not
guarantee that a GPU beats CPU on every device/input, nor that a single tile is
optimal on every driver. Use the probe below for additional physical hardware;
do not infer AMD/Intel/Qualcomm performance from the NVIDIA measurements.

## Measurements

Windows, NVIDIA GeForce RTX 4060, driver 620.02, Vulkan SDK 1.4.341.1,
MSVC 19.51 Release; ggml v0.19.0 plus the repository patch stack.
Model: `pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.gguf`, F32, 128 mel bins,
hop 512, sample rate 44100. CPU comparison uses 9 threads and AVX2.

Timings cover complete `hifigan_run` calls, including cleanup, excluding model
loading. Caches were not cleared, so the first call is not a cold-driver test.
Cases ran sequentially on a normal desktop, not an isolated benchmark machine.

| Case | Input | First call | Repeated calls |
|---|---|---:|---:|
| Before fix, default Vulkan | 512 frames, constant mel/F0 | 16,444.7 ms | 15,663.8 ms |
| Before fix, direct convolution disabled | Same | 5,553.4 ms | 5,323.0 ms |
| Fixed default Vulkan | Same | 168.5 ms | 91.4 / 90.8 ms |
| Fixed CPU | Same | 1,234.2 ms | 1,211.2 / 1,224.9 ms |
| Fixed Vulkan | 7 frames, varying mel/F0 with unvoiced frames | 74.3 ms | 14.2 / 14.8 ms |
| Fixed Vulkan | 33 frames, varying | 78.0 ms | 15.7 / 14.8 ms |
| Fixed Vulkan | 513 frames, varying | 154.8 ms | 92.3 / 96.9 ms |

512 frames produce 262144 samples, approximately 5.94 seconds of audio.
The 512-frame repeated Vulkan call improves by approximately 170x in this
measurement and is approximately 13x faster than CPU. These are observations
on this configuration, not performance guarantees.

An A/B using the original binary's `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1`
also produced 97.7 / 94.2 ms repeated calls. With device-only memory and direct
convolution disabled it took 196.9 / 200.6 ms: the direct path itself benefits
once the allocation policy is corrected.

## Correctness and reproducible checks

The fixed/default Vulkan output is byte-identical to the old default Vulkan
output for 7, 33 and 513 varying frames, and for 512 constant frames. Fixed CPU
also matches the old CPU output byte-for-byte for the 512-frame input.
The 512-frame Vulkan SHA-256 is:

```text
54c1523fe5ff45c74da9cb7a962fac607e70a22f3d7bcf8b3dd26b54f4899ca8
```

`vocoder_probe` is built with `PCNSF_BUILD_CLI=ON`. It calls the library without
the separate `hifigan_cli` program's cooperative-matrix environment override:

```sh
vocoder_probe MODEL.gguf 512 3 default.f32 constant
vocoder_probe MODEL.gguf 513 3 varying.f32 varying
```

Set `PCNSF_BACKEND=cpu` for CPU comparison. For an allocation A/B on the fixed
binary, set `GGML_VK_PREFER_HOST_VISIBLE_VIDMEM=1`, rerun the same input, and
compare the raw files. Use separate processes: Vulkan initialization captures
the allocation preference. These GGML switches are presence-based; remove
them rather than assigning `0` when restoring defaults.

Model-free checks:

```sh
ctest --test-dir build --output-on-failure -R 't08_vulkan_device_policy|t09_vulkan_convolution'
build/bin/test_audio_op_regressions Vulkan0
```

The memory/limit test runs without a GPU and covers reversed memory-type
enumeration, incompatible types, undersized heaps, UMA, explicit overrides,
device-local fallback, and low workgroup/shared-memory limits. The GPU test
compares nine convolution shapes to a double-precision reference, covering
partial tiles, K=3/7/11, dilation=1/3/5 and fused bias/residual/activations.
The Mesa CI job runs both tests; software-driver correctness is not physical
GPU performance evidence. The existing populated-cache upgrade test checks
patch 9 and repeated configuration, including Windows UTF-8 paths/text.

Set `PCNSF_MODEL_GGUF` before configuring to add real-model CPU and Vulkan
probe checks, along with the existing F16 and mixed-kernel tests. CPU/Vulkan
byte comparisons above are change-regression checks, not a new claim of
PyTorch golden parity or equivalence between different backends.

On the validation machine, all 16 registered tests passed across the final
suite run and the isolated patch-upgrade rerun. The latter needed execution
outside the restricted sandbox because Git's shell helpers could not run
inside it; the test used only local dependency clones and temporary directories.
The standalone Vulkan audio-op suite additionally reported 9 passed and 3
unsupported (the latter are not passing cases). The e64 shader was also run
directly with K=11, IC=32, OC=64, T=129, dilation=5: reported shared memory was
15,488 bytes and maximum error versus the double reference was 9.69e-9.

## Boundaries

Physical execution for this fix was on Windows/RTX 4060. Other vendors, Intel
Mac, Apple Silicon and DAW hosts were not physically tested in this change.
The CI workflow has been extended, but its remote run is not claimed here.
CPU and Metal implementations are unchanged. Existing ggml warnings remain;
new policy/probe/regression sources compile without new warnings on MSVC.

The engine's existing per-call graph/context allocation and the embedding
application's choice of render range are separate issues. This patch does not
change their public contracts or claim that plugin-wide latency/host behavior
has been validated. Its purpose is to remove the demonstrated Vulkan memory
regression and avoid dispatching convolution tiles beyond device limits.
