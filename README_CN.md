<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/img/logo_dark.png">
    <img src="docs/img/logo.png" alt="Token Rush" width="180">
  </picture>
  <h1>Token Rush</h1>
  <p><em>单张 RTX 5090 上最快的 Qwen3.8-27B 推理引擎 —— 为一个用户、一张卡、一条流而生。</em></p>
  <p><a href="README.md">English</a> | 简体中文</p>
</div>

## 你能得到什么

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/decode_vs_context_dark.svg">
  <img src="docs/img/decode_vs_context.svg" alt="单流解码速度随上下文长度的变化：Token Rush 对比 llama.cpp、vLLM、SGLang 和 ExLlamaV3（单张 RTX 5090）">
</picture>

所有引擎都在同一张卡、同一天、同一批提示词下测得。短上下文、贪心解码，单位 tok/s：

| 引擎（各取其最快配置） | 文章 | 代码 | 数学 |
|---|---|---|---|
| **Token Rush**（DFlash2 草稿，图内投机） | **229** | **358** | **379** |
| SGLang + DSpark | 106 | 138 | 207 |
| ExLlamaV3 + MTP ×2 | 132 | 142 | 166 |
| ollama（其默认 MTP 链） | 129 | 135 | 166 |
| llama.cpp + MTP | 124 | 115 | 160 |
| vLLM（原始模式；其投机路径更慢） | 78 | 78 | 78 |

在 200k token 上下文下，开投机解码仍能以 200–220 tok/s 生成散文，不开为 70 tok/s，而次优的引擎只有 60。完整的 256k 窗口装得下也跑得通：262k token 处的大海捞针检索，显存占用 26 GB。

它也是一台服务器。`python -m tokenrush.serve` 同时说 OpenAI 和 Anthropic 两套 API，会话常驻内存，一轮对话只需为其新增的 token 付费，Claude Code 可以端到端跑在它上面。

## 你放弃了什么

- **单条流。**批量恒为 1，一次只服务一个请求；第二个客户端只能排队。用高负载下的吞吐换延迟，这是处处刻意为之的取舍：没有调度器、没有分页 KV、任何 kernel 里都没有批处理。
- **每权重 4.25 bit。**权重为 int4，每 128 个权重一组 bf16 的 scale 和 min，用 GPTQ 校准。相对 bf16：平均 KL 0.023，top-1 一致率 94.2%，WikiText-2 困惑度 6.37 对 6.26，GSM8K 97.0% 对 96.0%。我们所知最好的 4-bit 量化（ExLlamaV3）KL 为 0.013；我们还没到那里。细节、对手与完整配方见[量化](#量化)。
- **一个模型、一张卡、纯文本。**只做 Qwen3.8-27B 的文本路径，只跑 RTX 5090（`sm_120`）；视觉塔被丢弃。没有任何东西是通用的。

## 用法

需要一张带 CUDA 13 驱动的 RTX 5090，以及 [`uv`](https://docs.astral.sh/uv/)。第一条命令会创建环境（torch cu130、Triton、GDN kernel），首次运行会把检查点（17 GB）和 DFlash2 草稿（3.9 GB）下载到 Hub 缓存：

```bash
uv run python -m tokenrush.run --chat --prompt "Explain speculative decoding in three sentences."
```

起服务。一个端口同时提供 OpenAI Chat / Completions 和 Anthropic Messages；Claude Code 只需两个环境变量：

```bash
uv run python -m tokenrush.serve --port 8000
```

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8000 ANTHROPIC_AUTH_TOKEN=anything claude
```

`--draft mtp` 切换到自带的 MTP 头作草稿（默认会对中文提示词选它），`--no-spec` / `--draft raw` 关掉投机，`--max-len` 设上下文窗口（两个草稿都要时 256k 约需 30 GB）。更多见 [docs/serving.md](docs/serving.md)。

## 怎么做到的

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/progress_dark.svg">
  <img src="docs/img/progress.svg" alt="构建过程中的解码速度：原始解码与投机解码在文章、代码、数学上的 tok/s，逐步演进">
</picture>

引擎约 6000 行 PyTorch、Triton 和一个 CUDA kernel，专为这一个模型和这张卡而写。真正把数字推上去的步骤（每一步及其测量值，见 [docs/progress.md](docs/progress.md)）：

| 步骤 | 改了什么 | tok/s |
|---|---|---|
| 2 | 模型写成显式状态上的普通函数：连续 KV、FP32 递归状态、int4 权重即时反量化 | 3 |
| 3 | Triton 自写 int4 GEMV，投影融合 | 36 |
| 4 | **整个解码步进一张 CUDA graph**：64 层、lm_head、采样，token 在设备上直接回填 | 68 |
| 6 | 48 个 Gated DeltaNet 层各融合为一个 kernel（卷积、门控、delta 规则、归一化、输出门） | 83 |
| 7 | 融合的 flash-decoding attention，只算实际长度，不再有上下文分桶 | 100 |
| 8–9 | split-K GEMV，部分和在归一化中累加；FP8 KV cache 与 prefill attention kernel，256k 装得下了 | 102 |
| 13–14 | **投机解码进图**：MTP 头的草稿链、M 行 verify 步、接受与提交都在设备上完成 | 183 / 254 / 258 |
| 20 | 草稿只读 lm_head 的 128k 行切片，而非全部 248k 行 | 186 / 261 / 263 |
| 25–27 | **DFlash2 当草稿**：块扩散模型一次草稿前向出 7 个 token，跑在我们的 kernel 上，int4，环形缓存 | 217 / 355 / 350 |
| 28 | verify 步用 Marlin 级 int4 GEMM：验证 7 个 token 只花 1.13 倍单步开销 | 224 / 373 / 373 |

其中三步扛起了大部分结果。CUDA graph 从 10 ms 的预算里砍掉了 9 ms 的 kernel 启动时间——混合架构的 48 个递归层是一串串小算子，任何带动态批处理的引擎都无法把它们完整捕获进图。标题数字则来自投机解码：批量 1 时张量核是闲着的，验证 7 个草稿 token 比解码 1 个贵不了多少，而通用引擎在批处理之下付不起这个取舍。每一个 kernel 都对着这张卡上实测的字节数调优，因为别的任何东西都喂不饱 Blackwell 的带宽。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/wall_fraction_dark.svg">
  <img src="docs/img/wall_fraction.svg" alt="各引擎原始解码速度占带宽墙的比例，随上下文长度变化">
</picture>

单流解码受限于显存带宽：每一步都要把全部权重和 KV cache 读一遍，所以 `tok/s ≈ 1701 GB/s ÷ 每步字节数`，其中 1701 是这张卡的实测读带宽。4.25 bit 下权重共 14.3 GB，空上下文时上限 119 tok/s。上图是每个引擎的原始解码速度占*它自己*上限的比例，因此量的是引擎而不是量化：扣掉启动开销、同步和跑不满带宽的 kernel 之后还剩多少。我们稳在 80–85% 且随上下文上升；llama.cpp 在 240k 处掉到 54%，因为它的 attention 路径随长度劣化，而递归层——64 层中的 48 层——状态大小恒定，不随上下文变贵。

由此有两件事。原始解码在我们动手之前就已接近解决：vLLM 稳在 75–81%，单是 cuBLAS 就能跑出 96%，所以原始模式的余量不过是墙上的几个百分点加上少读几个字节。唯一能*穿过*墙的办法是每步产出多于一个 token——这正是第一张图上半部分展示的东西。

## 量化

每权重比特数是仅次于投机的第二大杠杆：它决定上限。格式是分组非对称 int4：每 128 个权重配一对 bf16 的 scale 和 min，401 个大矩阵上每权重 4.25 bit；embedding、归一化和小投影保持 bf16。格式在第 2 天就冻结且从未改动，因此所有 kernel、graph 和草稿都与码字怎么选无关，更好的量化器可以直接替换进来。

码字由 GPTQ 选取——256 条 2048 token 的校准序列（WikiText-103、torch 源码、GSM8K 训练集），逐层顺序量化，`lm_head` 最后在量化后主体的隐藏状态上量化——另外每组做一次 MSE 范围搜索，裁掉少数离群点，为其余权重换精度。约 200 行代码，单卡 20 分钟。

在同一份留出文本（WikiText-2 测试集、代码、数学）的 81,920 个位置上对着 bf16 logits 测量，所有候选用同一次前向：

| 检查点 | bit/权重 | 对 bf16 的 KL | top-1 | WikiText-2 困惑度（bf16 6.255） |
|---|---|---|---|---|
| llama.cpp UD-Q4_K_M | 4.80 | 0.0093 | 0.966 | 6.270 |
| ExLlamaV3 4.00 bpw | 4.10 | 0.0128 | 0.960 | 6.274 |
| **Token Rush，GPTQ + MSE** | **4.25** | **0.0232** | 0.942 | 6.365 |
| NVFP4（QUASAR QAT） | 5.07 | 0.0231 | 0.943 | 6.369 |
| RedHatAI INT4（AWQ + GPTQ） | 4.71 | 0.0458 | 0.924 | 6.460 |
| Token Rush，四舍五入 | 4.25 | 0.0546 | 0.905 | 6.495 |

校准把 KL 从 0.055 压到 0.023；与 ExLlamaV3 的差距来自均匀 16 级网格本身，它的 trellis 编码避开了这一点，代价是另一套 kernel。经引擎跑 GSM8K：194/200，bf16 为 192/200。

检查点发布在 [`zyhector/Qwen3.8-27B-TokenRush-int4g128`](https://huggingface.co/zyhector/Qwen3.8-27B-TokenRush-int4g128)，引擎下载的就是它。要从 bf16 权重重建它，或用同一把尺子量别的量化：

```bash
bash scripts/quantize/build.sh
```

完整脉络——量尺、各家格式、每步改进值多少——见 [docs/quantization.md](docs/quantization.md)。

## 正确性

两道关卡。bf16 路径在贪心解码下与 HF transformers 逐 token 一致；每个融合 kernel 都对着 torch 参考实现做差分测试；CUDA graph 回放与 eager 执行一致；投机贪心输出与原始贪心输出严格相等——这是唯一真正严格的检验。长上下文由 131k 和 262k 处的大海捞针验证。`pytest tests/` 跑全部 65 个测试；kernel 测试需要那张卡，协议与会话测试不需要。

## 数据从哪来

这里的每个数字都来自同一台机器同一次开机，2026-09-12，每个对手当天按它自己的配方重跑：日志在 [results/2026-09-12-machine-59052/](results/2026-09-12-machine-59052/)，表格由 `scripts/sweep_table.py` 和 `scripts/matrix_table.py` 从日志生成，图由 `scripts/plot_sweep.py` 和 `scripts/plot_progress.py` 生成。对手的版本、参数和坑见 [docs/baselines.md](docs/baselines.md)；机器见 [docs/environment.md](docs/environment.md)；逐步记录见 [docs/progress.md](docs/progress.md)。

## 目录结构

| | |
|---|---|
| `tokenrush/` | 引擎：`model.py`（文本路径）、`fused.py` / `ops.py`（Triton kernel）、`csrc/`（Marlin 移植）、`spec.py` / `mtp.py` / `dflash.py`（投机）、`gptq.py` / `quant.py`（量化）、`run.py`、`serve.py` |
| `bench/` | 测量：解码、上下文扫描、质量、大海捞针、GSM8K |
| `scripts/` | 对手基准、量化配方、表格与图的生成器 |
| `tests/` | 差分与协议测试 |
| `docs/` | 基线、环境、进展、量化、服务、坑 |
| `results/` | 每次报告运行的原始日志 |

## 作者

我是 USC 计算机科学硕士生，2027 年 6 月毕业，正在找 AI 基础设施方向的工作——推理引擎、kernel、服务系统——地点在湾区、西雅图或洛杉矶。如果你所在的团队在做这类工作，或知道适合它的团队，欢迎在 [LinkedIn](https://www.linkedin.com/in/hectorzhu/) 上联系我。
