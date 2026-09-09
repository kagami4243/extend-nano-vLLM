<p align="center">
  <img width="280" src="assets/logo.png" alt="nano-vLLM logo">
</p>

<h1 align="center">nano-vLLM</h1>

<p align="center">
  A readable, single-node inference engine for learning how vLLM-style systems work.
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="https://github.com/GeeeekExplorer/nano-vllm"><img src="https://img.shields.io/badge/status-educational-orange.svg" alt="Educational project"></a>
</p>

> [!WARNING]
> This is an educational implementation, not a production replacement for
> [vLLM](https://github.com/vllm-project/vllm). It intentionally favors small,
> inspectable implementations over complete model coverage, portability, and
> production-grade performance.

## Why this project?

`nano-vLLM` turns core inference-engine ideas into a compact codebase that can
be read end-to-end. It is useful for studying the connection between scheduler,
paged KV cache, attention kernels, model execution, and decoding optimizations.

The primary reference model is **Qwen3**. The project also contains focused,
explicitly documented experiments rather than a broad compatibility layer.

## What is implemented?

| Area | Included teaching implementation |
| --- | --- |
| Serving loop | Request lifecycle, paged KV cache, prefix caching, chunked prefill, decode batching, and recompute preemption |
| Attention | FlashAttention-backed paged KV attention; FP8 E4M3 KV storage with Triton page-store and decode kernels |
| Parallelism | Tensor, pipeline, data, and expert parallel experiments; their supported combinations are documented in [TODO](TODO) |
| Models | Qwen3 dense and Qwen3-MoE; Qwen3 + EAGLE3 speculative decoding |
| MoE | Local-expert routing and a minimal vLLM-style Triton grouped GEMM path |
| Quantization | Online W4A16 weight-only and FP8 linear experiments for dense Qwen3 |
| Validation | Focused GPU correctness tests and reproducible benchmark scripts |

## Quick start

### Prerequisites

- Linux with an NVIDIA GPU and a CUDA-compatible PyTorch installation
- Python 3.10–3.12
- FlashAttention 2 built for the installed PyTorch/CUDA combination
- A local Hugging Face Qwen3 checkpoint

Create an environment, install a matching PyTorch build and FlashAttention,
then install this project in editable mode:

```bash
git clone https://github.com/GeeeekExplorer/nano-vllm.git
cd nano-vllm

# Install PyTorch and FlashAttention for *your* CUDA environment first.
# See their installation instructions; do not use a wheel built on another host.
pip install -e .
```

Download a small reference checkpoint, for example:

```bash
huggingface-cli download Qwen/Qwen3-0.6B --local-dir ./models/Qwen3-0.6B
```

Run offline generation:

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

The same example is available in [example.py](example.py).

## Focused experiments

### FP8 KV cache

FP8 cache storage approximately doubles the number of resident KV pages under
the same cache-memory budget. It uses a Triton paged decode kernel on Ada GPUs
and an eager prefill fallback; it is intentionally not CUDA-graph compatible
yet.

```python
llm = LLM("./models/Qwen3-0.6B", kv_cache_dtype="fp8")
```

See [FP8 KV cache](docs/fp8_kv_cache.md) for limitations and validation.

### Quantization

```python
LLM("./models/Qwen3-0.6B", quantization="w4a16")  # INT4 weight, A16 GEMM
LLM("./models/Qwen3-0.6B", quantization="fp8")    # FP8 weight and activation GEMM
```

These are online, learning-oriented formats, not GPTQ/AWQ checkpoint support.
Read [quantization](docs/quantization.md) before using them.

### MoE and speculative decoding

- [MoE grouped kernel](docs/moe_kernel.md): local-expert execution with a
  minimal Triton grouped GEMM.
- [EAGLE3 benchmark](benchmarks/bench_spec_decode_eagle3.py): Qwen3 target +
  EAGLE3 greedy speculative decoding.

## Verify and benchmark

GPU tests are modules so they can run in an environment with the intended CUDA
and FlashAttention builds:

```bash
python -m tests.test_fp8_kv_cache
python -m tests.test_moe_kernel
python -m tests.test_quantization all
python -m tests.test_quantized_qwen3 w4a16 --model ./models/Qwen3-0.6B
```

Selected benchmarks:

```bash
python -m benchmarks.bench_moe_kernel --tokens 256
python -m benchmarks.bench_quantization none --model ./models/Qwen3-0.6B
python -m benchmarks.bench_spec_decode_eagle3 --help
```

Benchmark results are hardware-, model-, prompt-, and dependency-specific.
This repository does not claim parity with vLLM; use the scripts to measure the
configuration you care about.

## Repository map

```text
nanovllm/
  engine/         Request state, scheduler, block manager, model runner
  layers/         Attention, cache kernels, linear, MoE, quantization
  models/         Qwen3, Qwen3-MoE, and EAGLE3 model adapters
  distributed/    Teaching TP/PP/DP/EP process-group helpers
benchmarks/       Reproducible performance experiments
tests/            GPU correctness and integration tests
docs/             Design notes, experiments, and implementation limitations
```

## Documentation and roadmap

- [Engine walkthrough](docs/ENGINE_OVERVIEW.md)
- [Documentation index](docs/README.md)
- [Current capabilities and limitations](TODO)
- [Detailed learning tasks](nanovllm/tasks.md)

The most valuable next single-GPU step is full continuous batching: mixed
prefill/decode scheduling under a shared token budget. See [TODO](TODO) for
the remaining work and explicit non-goals.

## Contributing

Contributions that improve clarity, tests, reproducibility, or documented
learning value are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md).

## Acknowledgements

This project learns from and references [vLLM](https://github.com/vllm-project/vllm),
[FlashAttention](https://github.com/Dao-AILab/flash-attention),
[Hugging Face Transformers](https://github.com/huggingface/transformers), and
the EAGLE speculative-decoding work. Some small Triton kernel structures are
adapted from vLLM under Apache-2.0; see [NOTICE](NOTICE).

## License

`nano-vLLM` is released under the [MIT License](LICENSE).
