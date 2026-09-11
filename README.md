# extend-nano-vLLM

English | [简体中文](README_CN.md)

`extend-nano-vLLM` is an experimental GPU inference runtime built by extending
[nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm). It keeps the compact
offline-generation API of the base project while adding model execution,
memory-management, low-precision, and decoding paths that are useful when
studying modern LLM serving systems.

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](pyproject.toml)
[![CI](https://github.com/GeeeekExplorer/extend-nano-vllm/actions/workflows/quality.yml/badge.svg)](https://github.com/GeeeekExplorer/extend-nano-vllm/actions/workflows/quality.yml)

## Extensions over nano-vLLM

| Area | Added in this project |
| --- | --- |
| Model support | Qwen3-MoE model loading and EAGLE3 draft-model execution alongside dense Qwen3 |
| Parallel execution | Tensor, pipeline, data, and expert-parallel process-group layouts |
| Cache and scheduling | Prefix reuse, chunked prefill, and FP8 E4M3 KV-cache storage |
| Decoding | Greedy EAGLE3 speculative decoding with target/draft KV-cache coordination |
| Low precision | Online W4A16 weight-only and FP8 linear-weight quantization for dense Qwen3 |
| GPU kernels | Triton FP8 KV-cache store/decode kernels and grouped MoE expert GEMMs |
| Validation | GPU correctness checks and standalone benchmark drivers for the added paths |

The implementation is intentionally focused on local CUDA inference and Qwen3
checkpoints. It is a research codebase, not a hosted serving platform.

## Installation

### Requirements

- Linux with a CUDA-capable NVIDIA GPU
- Python 3.10, 3.11, or 3.12
- A CUDA-compatible PyTorch build
- FlashAttention 2 built for the installed PyTorch/CUDA combination

```bash
git clone https://github.com/GeeeekExplorer/extend-nano-vllm.git
cd extend-nano-vllm

# Install PyTorch for the local CUDA environment first.
# Install FlashAttention without build isolation so it uses that PyTorch build.
pip install flash-attn --no-build-isolation
pip install -e .
```

Download a Qwen3 checkpoint for the examples:

```bash
huggingface-cli download Qwen/Qwen3-0.6B --local-dir ./models/Qwen3-0.6B
```

## Quick start

```python
from nanovllm import LLM, SamplingParams

llm = LLM("./models/Qwen3-0.6B", enforce_eager=True)
outputs = llm.generate(
    ["Explain paged KV cache in one sentence."],
    SamplingParams(temperature=0.0, max_tokens=64),
    use_tqdm=False,
)
print(outputs[0]["text"])
llm.exit()
```

The repository also provides the same flow as a command-line example:

```bash
python example.py ./models/Qwen3-0.6B
```

## Runtime options

### Low-precision execution

```python
# Group-wise packed INT4 weights with BF16/FP16 activations.
w4a16 = LLM("./models/Qwen3-0.6B", quantization="w4a16")

# FP8 linear weights and activations.
fp8 = LLM("./models/Qwen3-0.6B", quantization="fp8")

# Store paged K/V tensors in FP8 E4M3.
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

### Distributed layouts

The `LLM` constructor accepts `tensor_parallel_size`,
`pipeline_parallel_size`, and `data_parallel_size`. For Qwen3-MoE,
`enable_expert_parallel=True` enables local-expert placement across the model
parallel ranks.

```python
llm = LLM(
    "./models/Qwen3-0.6B",
    tensor_parallel_size=2,
    pipeline_parallel_size=1,
)
```

## Verification and benchmarks

Run the focused GPU checks from the repository root:

```bash
python -m tests.test_quantization all
python -m tests.test_fp8_kv_cache
python -m tests.test_moe_kernel
python -m tests.test_quantized_qwen3 w4a16 --model ./models/Qwen3-0.6B
```

Benchmark drivers keep model paths and workload sizes explicit:

```bash
python -m benchmarks.bench --model ./models/Qwen3-0.6B
python -m benchmarks.bench_quantization fp8 --model ./models/Qwen3-0.6B
python -m benchmarks.bench_moe_kernel --tokens 256
python -m benchmarks.bench_spec_decode_eagle3 nanovllm \
  --target-model ./models/Qwen3-8B \
  --draft-model ./models/Qwen3-8B-eagle3
```

## Project layout

```text
nanovllm/
  engine/         Request lifecycle, scheduling, KV block management, model runner
  layers/         Attention, linear, quantization, MoE, and CUDA/Triton kernels
  models/         Qwen3, Qwen3-MoE, and EAGLE3 adapters
  distributed/    Parallel process-group and collective helpers
benchmarks/       Reproducible local performance drivers
tests/            GPU correctness and integration checks
example.py        Minimal offline generation example
```

## Attribution

This project builds on [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)
and references ideas and APIs from
[vLLM](https://github.com/vllm-project/vllm),
[FlashAttention](https://github.com/Dao-AILab/flash-attention),
[Hugging Face Transformers](https://github.com/huggingface/transformers), and
[EAGLE](https://github.com/SafeAILab/EAGLE). Adapted vLLM kernel structures are
identified in source comments; see [NOTICE](NOTICE) for license attribution.

## Contributing

Contributions that improve the implementation, validation, or reproducibility
are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

`extend-nano-vLLM` is released under the [MIT License](LICENSE).
