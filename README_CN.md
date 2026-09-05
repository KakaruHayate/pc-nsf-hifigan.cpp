# pc-nsf-hifigan.cpp

> **语言：** [English](README.md) | [中文](README_CN.md)

**基于 ggml 的 NSF-HiFiGAN 声码器推理库**

面向 DiffSinger `diffsinger.cpp` 链路的独立 vocoder 库。本仓库只负责 **ggml vocoder body**；mini-nsf 正弦源生成委托给 [KakaruHayate/libmininsf](https://github.com/KakaruHayate/libmininsf)。

---

## 特性

- **原生 ggml 算子** — sub-pixel 上采样（相位主序，精确）走 F32 `im2col+mul_mat` + graph interleave（默认）；`ggml_conv_transpose_1d` 仅作 legacy 回退 + `ggml_mul_mat`（source conv）
- **多后端** — 权重自动上传到后端 buffer（CPU/Vulkan/CUDA/Metal 通用）；Metal 使用融合 F32 implicit-GEMM direct convolution，不再物化 im2col
- **F16/F32 双线路（不量化）** — F32 线路（权重+计算 fp32，精确基线）；F16 线路（权重 fp16，为未来 fp16/bf16 训练试点预留）
- **Mel 前端** — `mel_nvstft`（DiffSinger hifigan 前端）+ `MelExtractor`（生态 API parity）
- **CLI + CTest** — 单条/批量 vocode + CTest golden 校验（纯 numpy 参考，无需模型资产）

## 快速上手

```bash
# 1. 转换（PT checkpoint -> GGUF；需 torch + gguf）
python converter/convert_hifigan.py --ckpt model.ckpt --config config.json --out hifigan_f32.gguf --dtype F32
# OpenVPI 官方 pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.ckpt 的发布包
# 不含 config.json，转换器已内置该版本参数：
python converter/convert_hifigan.py --ckpt pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.ckpt --out hifigan_f32.gguf --dtype F32

# 2. 构建（见 BUILDING.md）
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release
cmake --build build -j

# 3. vocode（mel.bin + f0.bin -> out.wav）
build/bin/hifigan_cli hifigan_f32.gguf mel.bin f0.bin out.wav
#    F16 线路（需 --dtype F16 的 GGUF）：
# HF_PRECISION=F16 build/bin/hifigan_cli hifigan_f16.gguf mel.bin f0.bin out_f16.wav
```

## 生态定位

本仓库属于一个面向 DiffSinger 风格端侧部署的小型 C++ 生态。训练侧仓库不使用 ggml —— 这些 C++ 仓库只做推理部署。当前上游如下：

| 仓库 | 组件 | 关系 |
|---|---|---|
| **pc-nsf-hifigan.cpp** | NSF-HiFiGAN vocoder（本仓库） | 核心引擎；依赖 `libmininsf` 提供正弦源 |
| [libmininsf](https://github.com/KakaruHayate/libmininsf) | mini-nsf 正弦源 | 基础组件（FetchContent 自动拉取） |
| [ggml-audio-patch](https://github.com/KakaruHayate/ggml-audio-patch) | ggml 补丁集 + 音频后端 | 提供 `ggml_conv_direct_1d`、`ADD_LEAKY_RELU` 及 Vulkan/CUDA/Metal kernel，被本仓库消费 |
| [game.cpp](https://github.com/KakaruHayate/game.cpp) | GAME（DiffSinger V3, score recognition） | 独立下游，把本仓库当作 vocoder 拉取 |

如果你克隆本仓库准备构建，还需要上面两个依赖仓库 —— 见 `docs/integrating_zh.md` §8 "从源码构建"。

## 模型发布

本仓库是生态中的**量化例外**：vocoder 对数值敏感，只产出两种精度
（`converter/convert_hifigan.py --dtype F32|F16`），**没有** `rec`/Q8 档。

| 资产 | 精度 | 使用场景 |
|---|---|---|
| `hifigan_f16.gguf` | F16（fp16 线路） | 常规部署（权重 fp16） |
| `hifigan_f32.gguf` | F32（精确线路） | golden 级回归基线 |

- 通过 GitHub **Releases** 发布，权重不提交进仓库。
- 文件名必须带精度后缀（`f16` / `f32`）——消费者按 size glob（如
  OpenUtau），裸 `hifigan.gguf` 会静默遮蔽另一精度。
- 发布清单：转换 → F16 wav 对 torch 参考验证（CTest t01/t02 +
  `tests/gen_hifigan_golden.py`）→ 双精度资产附到 release notes。
- 未来 fp16 *训练*试点先在 torch 侧验证；推理接口保持仅 F16/F32。

## 部署说明（速度基线 = ONNX/DML）

本声码器的**最低延迟部署路径是导出 ONNX 跑 DirectML**（20 s 音节约 282 ms，≈71× 实时，RTX 2070 级；裸基线数值见 `docs/benchmarks.md` §1）。本 ggml 引擎**不追平 ONNX 的延迟**，而是作为**参考 / 纯推理实现、后端可移植**（CPU/Vulkan/CUDA/Metal）+ **数值与 torch 参考在 EP 精度档内等价**（fp32 线路 corr 0.9999998460，合法 EP 噪声带内，双黄金帧一致）。与 DML 剩余的 ~1.5× 差距是**刻意止步、已冻结**（见 `docs/benchmarks.md` §6 终态裁决）：每再快一步，要么违反精度合同、要么落入我们不愿意持有的引擎级基建。正确性 / 可复现性 / 无 GPU 边缘场景 / 生态接入是目标；延迟追齐 ONNX/DML 不是（被 `hifisampler`/`hachitune` 拉取的方式见"生态定位"）。

## 集成（典型场景：mel + f0 进,wav 出）

逐帧契约、模型参数表与常见坑见 **[docs/integrating_zh.md](docs/integrating_zh.md)**（mel `[T, 128]` 自然对数 + f0 `[T]` Hz，帧按 hop=512 @ 44.1 kHz 逐帧对齐 → wav `T*512` 采样）。接入面对 `pc_nsf_hifigan::HifiganModel` + `hifigan_run`，端到端的极简模板在 `examples/external_consumer/`。

## 开发指引

### 贡献者

- **无量化路径** — 本仓库**刻意不量化**：vocoder 对数值敏感，仅保留 F16/F32；未来 fp16 训练试点先在 torch 侧验证
- **依赖策略（D2-修订,2026-08-30）** — 上游不接受的 ggml 修改放在补丁集中。本仓库消费补丁提供的新算子（`ggml_conv_direct_1d[_fused]`、`ggml_add_leaky_relu`），9 枚补丁保存在 `./patches/`，每次配置时由 `cmake/ApplyGgmlPatches.cmake` 幂等应用到 stock ggml v0.19.0，并支持已有构建目录升级。第 9 枚修复 Vulkan 显存选择及卷积分块的设备限制，详见[调查与验证记录](docs/vulkan_device_memory.md)。已手工打过补丁的目录树会被识别并收养（stamp: `.pcnsf-patches/`）。
- **提交前过数值门槛** — 提交前跑 `tests/` golden 对比（`gen_hifigan_golden.py` + CTest t01/t02），wav 输出须与 torch 参考一致
- **mininsf 源同步** — source generator 改动在 `libmininsf` 仓库，本仓库只消费（FetchContent）

### 已知坑

- `ggml_conv_transpose_1d` 的输出长度语义与 torch ConvTranspose1d 的 padding 约定不同——注意核对长度
- 两个 mel 前端勿混用：`mel_nvstft`（DiffSinger 专用，reflect pad + center=False）vs `MelExtractor`（htk/slaney 通用）

## 许可

代码 [MPL-2.0](LICENSE)（本仓库）；**例外**：`patches/` 目录采用 `MIT OR Apache-2.0` 双许可
（见 `patches/LICENSE`），以保证这些补丁仍可被上游 ggml 接纳。

模型权重视各自许可：DiffSinger 官方 NSF-HiFiGAN 权重为 CC BY-NC-SA 4.0（见
[NOTICE.md](NOTICE.md)）；请注意 non-commercial 限制。

## 目录结构

```
include/pc_nsf_hifigan/  公共头文件（gguf_model.h, mel.h, hifigan.h）
src/                     实现（gguf_model.cpp, mel.cpp, hifigan.cpp）
tools/                   CLI（hifigan_cli.cpp）
converter/               权重转换脚本（convert_hifigan.py）
tests/                   golden 对比测试（numpy 参考，无需模型资产）
examples/external_consumer  第三方集成示例
third_party/             pocketfft_hdronly.h（vendored 单头文件）
cmake/Dependencies.cmake 依赖管理（FetchContent）
```

[构建](BUILDING.md) · [架构](docs/hifigan.md) · [基准实测](docs/benchmarks.md) · [集成](docs/integrating_zh.md) · [测试](tests/README.md)
