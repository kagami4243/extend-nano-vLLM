# extend-nano-vLLM

[English](README.md) | 简体中文

`extend-nano-vLLM` 是一个基于
[nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 扩展而来的实验性 GPU
推理运行时。它保留了上游项目紧凑的离线生成 API，并补充了模型执行、内存管理、低精度
计算和解码相关能力，用于研究现代大语言模型推理系统。

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](pyproject.toml)
[![CI](https://github.com/GeeeekExplorer/extend-nano-vllm/actions/workflows/quality.yml/badge.svg)](https://github.com/GeeeekExplorer/extend-nano-vllm/actions/workflows/quality.yml)

## 相比 nano-vLLM 的扩展

| 方向 | 本项目新增内容 |
| --- | --- |
| 模型支持 | 在 dense Qwen3 之外，支持 Qwen3-MoE 加载和 EAGLE3 draft model 执行 |
| 并行执行 | TP/PP/DP 进程组与 EP=DP×TP，不增加独立 EP 轴 |
| Cache 与调度 | Prefix reuse、chunked/batched prefill、跨 DP token padding/完成状态协调与 FP8 E4M3 KV cache |
| 解码 | 支持 target/draft KV cache 协同的 greedy EAGLE3 speculative decoding |
| 低精度 | 面向 dense Qwen3 的在线 W4A16 和 FP8 W8A8 Linear quantization，FP8 默认 per-tensor |
| GPU kernel | Triton FP8 KV cache kernel 与紧凑 grouped MoE GEMM，Graph 下跳过无效 tile |
| 验证 | 为新增路径提供 GPU correctness check 和独立 benchmark driver |

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
