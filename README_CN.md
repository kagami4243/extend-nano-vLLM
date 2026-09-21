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
| 并行执行 | Tensor、Pipeline、Data 和 Expert Parallel 的进程组布局 |
| Cache 与调度 | Prefix reuse、chunked prefill、piecewise CUDA Graph prefill 以及 FP8 E4M3 KV cache 存储 |
| 解码 | 支持 target/draft KV cache 协同的 greedy EAGLE3 speculative decoding |
| 低精度 | 面向 dense Qwen3 的在线 W4A16 和 FP8 W8A8 Linear quantization，FP8 默认 per-tensor |
| GPU kernel | Triton FP8 KV cache store/decode kernel 与 grouped MoE expert GEMM |
| 验证 | 为新增路径提供 GPU correctness check 和独立 benchmark driver |

项目聚焦本地 CUDA 推理和 Qwen3 checkpoint，是研究型代码库，不是在线服务平台。

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

llm = LLM("./models/Qwen3-0.6B", enforce_eager=True)
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

支持的单进程 dense Qwen3 workload 默认启用 CUDA Graph。prefill 会捕获 embedding、Linear、
normalization 和 MLP 等静态片段，attention 因 metadata 动态而保持 eager。图缓存按 prompt
长度区分；调试时可设置 `enforce_eager=True` 关闭 graph capture。

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

`LLM` 构造函数接受 `tensor_parallel_size`、`pipeline_parallel_size` 和
`data_parallel_size`。对于 Qwen3-MoE，设置 `enable_expert_parallel=True` 会在模型
并行 rank 之间启用 local-expert placement。

```python
llm = LLM(
    "./models/Qwen3-0.6B",
    tensor_parallel_size=2,
    pipeline_parallel_size=1,
)
```

## 验证与基准测试

在仓库根目录运行以下 GPU 检查：

```bash
python -m tests.test_quantization all
python -m tests.test_fp8_kv_cache
python -m tests.test_cuda_graph_piece_integration --model ./models/Qwen3-0.6B
python -m tests.test_moe_kernel
python -m tests.test_quantized_qwen3 w4a16 --model ./models/Qwen3-0.6B
```

benchmark driver 均显式指定模型路径和 workload：

```bash
python -m benchmarks.bench --model ./models/Qwen3-0.6B
python -m benchmarks.bench_quantization fp8 --model ./models/Qwen3-0.6B
python -m benchmarks.bench_cuda_graph_piece --model ./models/Qwen3-0.6B \
  --prompt-tokens 128 --runs 5
python -m benchmarks.bench_moe_kernel --tokens 256
python -m benchmarks.bench_spec_decode_eagle3 nanovllm \
  --target-model ./models/Qwen3-8B \
  --draft-model ./models/Qwen3-8B-eagle3
```

## 项目结构

```text
nanovllm/
  engine/         请求生命周期、调度、KV block 管理、model runner
  layers/         Attention、Linear、quantization、MoE 和 CUDA/Triton kernel
  models/         Qwen3、Qwen3-MoE 和 EAGLE3 adapter
  distributed/    并行进程组与 collective helper
benchmarks/       可复现的本地性能测试 driver
tests/            GPU correctness 和 integration check
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
