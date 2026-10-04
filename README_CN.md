# extend-nano-vLLM

[English](README.md) | 简体中文

`extend-nano-vLLM` 是一个基于
[nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 扩展而来的实验性 GPU
推理运行时。它保留了上游项目紧凑的离线生成 API，并补充了模型执行、内存管理、低精度
计算和解码相关能力，用于研究现代大语言模型推理系统。

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](pyproject.toml)
[![CI](https://github.com/GeeeekExplorer/extend-nano-vllm/actions/workflows/quality.yml/badge.svg)](https://github.com/GeeeekExplorer/extend-nano-vllm/actions/workflows/quality.yml)

## 特性

以下同时列出继承自 nano-vLLM 的核心机制与本项目的扩展，并标明实验路径及当前限制。

### 模型执行与生成

- **Dense Qwen3 / Qwen3-MoE**：按 model_type 注册模型，加载本地 Hugging Face
  checkpoint，并按受支持并行布局仅加载本 rank 所需权重。
- **离线批量生成**：支持文本或 token-ID prompts、共享或逐请求 SamplingParams，
  按输入顺序返回 text 和生成的 token_ids。
- **Greedy 与 temperature 采样**：temperature=0 使用 argmax，正值使用类别概率采样，
  支持 EOS 停止、max_tokens 和 ignore_eos。
- **请求生命周期接口**：提供 add_request()、step()、is_finished()、generate()，
  以及 exit() 显式退出与资源清理。

### Attention、KV cache 与调度

- **分页 KV cache**：固定 256-token block、逻辑到物理 block table、引用计数，
  按 GPU 显存预算分配容量，decode 使用分页 attention。
- **FlashAttention 2**：支持变长 causal prefill 与 KV-cache decode，
  Triton kernel 将新增 K/V 写入对应物理 slot。
- **Prefix caching**：跨请求复用计算完成的完整前缀 block，校验 hash/token 内容，
  清理陈旧 hash，并统计命中数与复用 token 数。
- **迭代级 decode batching**：每步批量推进活跃请求，完成后释放资源；
  KV block 不足时抢占请求并在后续重新 prefill。
- **Chunked prefill**：按 max_num_batched_tokens 切分长 prompt、复用已有 KV，
  prefill 后给 running decode 请求推进机会。
- **多请求 prefill batching**：多个 waiting 请求共享本步 token budget；
  Qwen3-MoE 默认开启，dense Qwen3 可显式开启。

### 分布式执行

- **张量并行（TP）**：vocab embedding/LM head、QKV 与 dense MLP 投影分片，
  使用显式 TP group 和 collective。
- **流水线并行（PP）**：连续层与本地 KV 分区、stage 间传递 activation、末 stage
  回传采样 token；当前同步执行单 microbatch，要求 eager，没有流水重叠。
- **数据并行（DP）**：独立副本引擎、scheduler 和 KV pool，离线 round-robin
  分发请求并恢复原输出顺序。
- **组合并行**：支持 TP×PP 和 DP×TP；开启专家并行后自动派生 **EP=DP×TP**，
  不增加独立进程轴。跨 DP EP 当前要求 PP=1。
- **跨 DP EP 协调**：不同 token 数自动 padding 通信、路由前去掉填充、输出裁回；
  先完成副本执行 dummy forward 直到全局完成，保留各副本独立的 prefix cache。

### MoE 路由与专家执行

- **Top-k 路由与本地专家**：FP32 softmax、可选 top-k 概率归一化、
  本地专家权重分片和专家输出加权归并。
- **Triton grouped GEMM**：按专家排列路由并对齐到 16 行，两次 grouped GEMM
  计算 gate/up 和 down，GPU 执行 SwiGLU 与按路由顺序的输出归并。
- **紧凑 Graph 专家任务**：固定容量索引、GPU 有效长度、M×top_k 行中间激活，
  无效 GEMM tile 在加载权重和矩阵计算前退出。
- **通信 backend**：复制输入后归约；跨 DP all-gather 后 all-reduce，或
  reduce-scatter 加本地 TP all-gather；另有 eager all-to-all dispatch/combine 实验路径。
- **Shared experts 与 capacity**：支持 shared-expert 执行、固定专家容量或
  按 token 数计算的 capacity factor，以及容量溢出时的 token dropping。
- **专家放置与迁移**：支持逐层显式 expert ownership 和 relocate_experts()
  受控迁移；当前不支持跨 DP 动态迁移。

### CUDA Graph、编译与低精度

- **完整 decode CUDA Graph**：按多个 batch bucket 捕获包含 attention/MoE 的
  forward，每 GPU 上限 512 tokens，最大配置共 36 个 Graph；
  跨 DP MoE 汇集上限 DP×512，logits/采样在图外。
- **Piecewise prefill CUDA Graph**：静态片段围绕 eager attention 捕获，dense
  Qwen3 默认开启；MoE 需显式指定 capture sizes，片段之间的 attention/MoE 保持 eager。
- **局部 torch.compile**：编译带装饰器的 normalization、activation 与 sampling
  等算子，编译与 CUDA Graph 是独立控制的优化路径。
- **W4A16 权重量化**：dense Qwen3 在线 group-wise packed INT4 Linear 权重，
  activation 保持 BF16/FP16，使用 Triton GEMM。
- **FP8 W8A8 Linear**：dense Qwen3 在线 E4M3 weight/activation，默认
  per-tensor activation scale，可选 per-token；per-token prefill 使用 eager，decode 可用 Graph。
- **FP8 E4M3 KV cache**：可选单 GPU 存储、Triton page 写入与 paged decode
  attention，prefill 临时反量化后使用 FA2；教学路径要求 head_dim=128，
  每层 K/V scale 固定为 1。

### 投机解码与验证

- **EAGLE3 speculative decoding**：独立 draft model、greedy proposal/target
  verification、接受前缀与 replacement token、独立 KV 状态、checkpoint/rollback，
  支持批量请求。当前要求 DP=TP=PP=1、关闭 prefix cache、target verification
  eager，且不支持 FP8 KV 组合。
- **运行诊断**：并行 rank、参数字节数、通信与 Graph replay 计数、prefix reuse
  和 speculative acceptance 统计。
- **Tests / benchmarks**：临时 config-only checkpoint 的 CPU 合约测试、GPU
  layer/Graph 回归、分布式集成，以及区分 warmup 的延迟、吞吐与 GPU profile 测量。

### 当前范围

目标环境是 CUDA/NVIDIA 和本地 Qwen3 checkpoint。尚无 OpenAI 兼容服务、流式 API、
ROCm/SDPA fallback、top-k/top-p 采样、自动 EPLB、CP 或 PD 分离。调度采用
prefill/decode 阶段交替，没有同一次 forward 的混合 batch；跨 DP 不等长批次的
本步完整 forward 回退 eager。尚不支持预量化 checkpoint 与量化 MoE，
性能收益与长输出数值一致性需要按 workload 验证。

项目聚焦本地 CUDA 推理和 Qwen3 checkpoint，是研究型代码库，不是在线服务平台。
默认值、组合限制与验证范围见 [运行指南](docs/runtime.md)；实现完成不代表所有负载均已验收。

## 安装

### 环境要求

- Linux 和支持 CUDA 的 NVIDIA GPU
- Python 3.10、3.11 或 3.12
- 与 CUDA 匹配的 PyTorch
- 与当前 PyTorch/CUDA 匹配构建的 FlashAttention 2

```bash
git clone https://github.com/GeeeekExplorer/extend-nano-vllm.git
cd extend-nano-vllm

# 先为本机 CUDA 环境安装 PyTorch。
# FlashAttention 使用该 PyTorch 构建，因此关闭 build isolation。
pip install flash-attn --no-build-isolation
pip install -e .
```

下载示例使用的 Qwen3 checkpoint：

```bash
huggingface-cli download Qwen/Qwen3-0.6B --local-dir ./models/Qwen3-0.6B
```

## 快速开始

```python
from nanovllm import LLM, SamplingParams

llm = LLM("./models/Qwen3-0.6B")
outputs = llm.generate(
    ["用一句话解释 paged KV cache。"],
    SamplingParams(temperature=0.0, max_tokens=64),
    use_tqdm=False,
)
print(outputs[0]["text"])
llm.exit()
```

仓库中也提供相同流程的命令行示例：

```bash
python example.py ./models/Qwen3-0.6B
```

## 运行选项

### CUDA Graph 执行

Dense Qwen3 和 Qwen3-MoE 的受支持 decode 路径默认启用 CUDA Graph，包括相应 TP/EP
布局。每 GPU 捕获上限为 min(max_num_seqs,512)，max_num_seqs>=512 时共 36 个
bucket；完整 forward 包含 attention 和 MoE，logits 与采样在图外。

Dense Qwen3 默认 piecewise prefill，静态片段捕获而 attention 保持 eager，缓存按实际
forward 的 token 总数区分。MoE prefill 需显式开启 moe_prefill_piece 并指定
moe_prefill_piece_capture_sizes。PP 要求 eager；跨 DP token 数或执行阶段不同的本步
完整 forward 统一 eager，后续协调一致时恢复 Graph。enforce_eager=True 关闭 Graph，
但局部带 torch.compile 装饰器的函数仍可编译。

紧凑 MoE Graph 使用固定容量路由索引、GPU 有效长度和 M×top_k 行中间激活，
无效 GEMM tile 在加载权重前退出。跨 DP 汇集输入上限为 DP×512，每 GPU 上限为 512。

### Prefix cache 与 prefill 调度

Prefix cache 默认开启，跨 DP EP 同样保留。各副本 KV 独立，通信 padding 处理命中
差异，不改变本地 attention 输入。先完成的副本执行 dummy forward，直至全局完成。

max_num_batched_tokens 限制本步 prefill 预算；enable_prefill_batching=True 时可批量
推进多个 waiting 请求，Qwen3-MoE 默认开启、dense 默认关闭。当前没有同一次 forward
的混合 prefill/decode batch，也不是异步在线服务调度器。

### 低精度执行

```python
# Group-wise packed INT4 weight，activation 保持 BF16/FP16。
w4a16 = LLM("./models/Qwen3-0.6B", quantization="w4a16")

# FP8 E4M3 Linear weight 和 activation，默认使用 per-tensor scale。
fp8 = LLM("./models/Qwen3-0.6B", quantization="fp8")

# 需要更细 activation 动态范围时可选择 per-token。
# 此模式 prefill 使用 eager，decode 仍使用 CUDA Graph。
fp8_per_token = LLM(
    "./models/Qwen3-0.6B", quantization="fp8", fp8_format="per_token"
)

# 使用 FP8 E4M3 存储 paged K/V tensor，KV 容量约增加一倍；吞吐影响依模型和 workload 而定。
fp8_cache = LLM("./models/Qwen3-0.6B", kv_cache_dtype="fp8")
```

### EAGLE3 speculative decoding

```python
llm = LLM(
    "./models/Qwen3-8B",
    speculative_config={
        "method": "eagle3",
        "model": "./models/Qwen3-8B-eagle3",
        "num_speculative_tokens": 8,
    },
)
```

### 分布式布局

LLM 接受 tensor_parallel_size、pipeline_parallel_size 和 data_parallel_size。
Qwen3-MoE 开启 enable_expert_parallel 后，EP 自动等于 DP×TP；普通层仍使用
本副本 TP 组，专家跨 DP×TP 分片。显式 expert_parallel_size 只能用于一致性检查。
跨 DP EP 当前要求 PP=1；支持 shared experts、capacity 和显式专家放置，尚无自动 EPLB。

```python
llm = LLM(
    "./models/Qwen3-0.6B",
    tensor_parallel_size=2,
    pipeline_parallel_size=1,
)
```

离线 DP 使用公共 helper，并放在多进程入口保护内：

```python
from nanovllm import SamplingParams
from nanovllm.engine.data_parallel import generate_data_parallel

if __name__ == "__main__":
    outputs, replica_ids = generate_data_parallel(
        "./models/Qwen3-30B-A3B-Base",
        [[1000, 1001], [1002, 1003, 1004]],
        SamplingParams(temperature=0, ignore_eos=True, max_tokens=4),
        data_parallel_size=2,
        tensor_parallel_size=1,
        enable_expert_parallel=True,
        max_model_len=32,
    )
```

该示例需要两张 GPU，EP=2。全局 EP helper 要求 token-ID prompts、均衡请求数、
ignore_eos=True 和相同正数 max_tokens，允许 prompt 长度和缓存命中不同。
helper 每次创建并退出引擎，总调用耗时包含启动；部署稳态计时应在已初始化引擎中进行。

## 验证与基准测试

在仓库根目录运行以下 GPU 检查：

```bash
python -m tests.test_quantization all
python -m tests.test_fp8_kv_cache
python -m tests.test_cuda_graph_piece_integration --model ./models/Qwen3-0.6B
python -m tests.test_moe_kernel
python -m tests.test_quantized_qwen3 w4a16 --model ./models/Qwen3-0.6B
python -m pytest -q tests/test_moe_graph_capture.py tests/test_moe_capacity_graph.py \
  tests/test_moe_graph_limits.py tests/test_moe_compact_layout.py
```

CPU 配置测试使用临时 config-only checkpoint，无需下载模型权重：

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q \
  tests/test_feature_contracts.py tests/test_prefill_batch_scheduler.py \
  tests/test_moe_dp_generation.py tests/test_moe_capacity_config.py \
  tests/test_moe_topology_contracts.py tests/test_moe_prefill_batching_default.py \
  tests/test_moe_prefill_piece_config.py
```

benchmark driver 均显式指定模型路径和 workload：

```bash
python -m benchmarks.bench --model ./models/Qwen3-0.6B
python -m benchmarks.bench_quantization fp8 --model ./models/Qwen3-0.6B
python -m benchmarks.bench_cuda_graph_piece --model ./models/Qwen3-0.6B \
  --prompt-tokens 128 --runs 5
python -m benchmarks.bench_moe_kernel --tokens 256
python -m benchmarks.bench_moe_graph --tokens 512 --warmup 10 --repeats 50
python -m benchmarks.bench_moe_compact_engine \
  --model ./models/Qwen3-30B-A3B-Base --dp 1 --tp 1
python -m benchmarks.bench_prefix_cache --model ./models/Qwen3-0.6B
python -m benchmarks.bench_chunked_prefill --model ./models/Qwen3-0.6B
python -m benchmarks.bench_tp_scaling --model ./models/Qwen3-0.6B --tp 1 2 4
python -m benchmarks.bench_spec_decode_eagle3 nanovllm \
  --target-model ./models/Qwen3-8B \
  --draft-model ./models/Qwen3-8B-eagle3
```

多 GPU 测试前显式选择可见设备。随机权重单层 benchmark 不是端到端模型性能；
只有明确说明的计时才排除初始化。BF16 的并行归约及 batch composition 可改变 greedy
输出，短测试不能证明任意长输出严格一致。最新紧凑 MoE Graph 已有单 GPU 重放验证，
其 NCCL 多卡回归和配对端到端加速仍待补验收。

## 项目结构

```text
nanovllm/
  engine/         请求生命周期、调度、KV block 管理、model runner
  layers/         Attention、Linear、quantization、MoE 和 CUDA/Triton kernel
  models/         Qwen3、Qwen3-MoE 和 EAGLE3 adapter
  distributed/    并行进程组与 collective helper
benchmarks/       可复现的本地性能测试 driver
tests/            GPU correctness 和 integration check
docs/runtime.md   默认配置、并行/Graph 行为与已知限制
example.py        最小离线生成示例
```

## 致谢与归属

本项目构建于 [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 之上，并参考了
[vLLM](https://github.com/vllm-project/vllm)、
[FlashAttention](https://github.com/Dao-AILab/flash-attention)、
[Hugging Face Transformers](https://github.com/huggingface/transformers) 和
[EAGLE](https://github.com/SafeAILab/EAGLE) 的设计与 API。来自 vLLM 的适配 kernel
结构均在源代码中标注，许可证归属见 [NOTICE](NOTICE)。

## 贡献

欢迎改进实现、验证或可复现性的贡献，详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可证

`extend-nano-vLLM` 使用 [MIT License](LICENSE)。
