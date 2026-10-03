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
| Parallel execution | Tensor/pipeline/data groups and expert parallel derived as EP=DP×TP, without an independent EP axis |
| Cache and scheduling | Prefix reuse, chunked and batched prefill, cross-DP token padding/completion coordination, and FP8 E4M3 KV-cache storage |
| Decoding | Greedy EAGLE3 speculative decoding with target/draft KV-cache coordination |
| Low precision | Online W4A16 and FP8 W8A8 linear quantization for dense Qwen3, with per-tensor FP8 as the default |
| GPU kernels | Triton FP8 KV-cache kernels and compact grouped MoE GEMMs that skip inactive tiles under CUDA Graph |
| Validation | GPU correctness checks and standalone benchmark drivers for the added paths |

The implementation is intentionally focused on local CUDA inference and Qwen3
checkpoints. It is a research codebase, not a hosted serving platform.
See [the runtime guide](docs/runtime.md) for defaults, supported combinations,
and the distinction between implemented behavior and completed verification.

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

llm = LLM("./models/Qwen3-0.6B")
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

### CUDA Graph execution

CUDA Graph is enabled by default for supported dense Qwen3 and Qwen3-MoE
decode paths, including supported TP/EP layouts. Each GPU captures decode
buckets up to `min(max_num_seqs, 512)`; `max_num_seqs>=512` gives 36 graphs.
The complete model forward, including attention and MoE, is captured; logits
and sampling remain outside the graph.

Dense Qwen3 also uses piecewise prefill: static pieces are captured around eager
attention, with a cache keyed by the number of tokens actually forwarded.
Qwen3-MoE prefill pieces are opt-in via `moe_prefill_piece=True` and explicit
`moe_prefill_piece_capture_sizes`. PP requires eager execution. Unequal cross-DP
token counts or execution phases make the entire affected forward eager;
compatible later steps resume graph replay. `enforce_eager=True` disables
graphs, while locally decorated `torch.compile` functions can still compile.

MoE graphs use fixed-capacity route indices and a GPU effective length. Their
activations occupy `M*top_k` rows, and inactive GEMM tiles exit before loading
weights. Cross-DP input capacity is `DP*512`, rather than a shared 512-token cap.

### Prefix cache and prefill scheduling

Prefix caching defaults to enabled and remains available with global EP.
Each DP replica owns its KV cache; communication padding handles different
cache hits without changing local attention inputs. A replica that finishes
early runs a dummy forward until all replicas finish.

`max_num_batched_tokens` bounds each prefill step. Multiple waiting requests
can share that budget with `enable_prefill_batching=True`, which is the default
for Qwen3-MoE; dense Qwen3 defaults to one prefill request at a time. This is
not a mixed prefill/decode batch or an asynchronous serving scheduler.

### Low-precision execution

```python
# Group-wise packed INT4 weights with BF16/FP16 activations.
w4a16 = LLM("./models/Qwen3-0.6B", quantization="w4a16")

# FP8 E4M3 linear weights and activations. Per-tensor scale is the default.
fp8 = LLM("./models/Qwen3-0.6B", quantization="fp8")

# Per-token activation scales are available when their finer dynamic range is
# required. Prefill runs eagerly for this mode; decode still uses CUDA Graph.
fp8_per_token = LLM(
    "./models/Qwen3-0.6B", quantization="fp8", fp8_format="per_token"
)

# Store paged K/V tensors in FP8 E4M3. This approximately doubles KV capacity;
# its throughput effect depends on the model and workload.
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

The `LLM` constructor accepts `tensor_parallel_size`, `pipeline_parallel_size`,
and `data_parallel_size`. For Qwen3-MoE, `enable_expert_parallel=True` derives
`EP=DP*TP`. Ordinary layers keep their local TP groups; experts span DP×TP.
An explicit `expert_parallel_size` must match this derived size.
Cross-DP EP currently requires PP=1. Shared experts, capacity limits and
explicit expert placement are supported; automatic EPLB is not implemented.

```python
llm = LLM(
    "./models/Qwen3-0.6B",
    tensor_parallel_size=2,
    pipeline_parallel_size=1,
)
```

For offline DP, use the public helper inside a multiprocessing entry guard:

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

This uses two GPUs and EP=2. The global-EP helper requires token-ID prompts,
a balanced request count, `ignore_eos=True`, and equal positive `max_tokens`;
prompt lengths and prefix-cache hits may differ. It creates engines per call,
so its total call time includes startup. Use a persistent initialized engine
when measuring deployment latency.

## Verification and benchmarks

Run the focused GPU checks from the repository root:

```bash
python -m tests.test_quantization all
python -m tests.test_fp8_kv_cache
python -m tests.test_cuda_graph_piece_integration --model ./models/Qwen3-0.6B
python -m tests.test_moe_kernel
python -m tests.test_quantized_qwen3 w4a16 --model ./models/Qwen3-0.6B
python -m pytest -q tests/test_moe_graph_capture.py tests/test_moe_capacity_graph.py \
  tests/test_moe_graph_limits.py tests/test_moe_compact_layout.py
```

CPU contracts use temporary config-only checkpoints, without model weights:

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q \
  tests/test_feature_contracts.py tests/test_prefill_batch_scheduler.py \
  tests/test_moe_dp_generation.py tests/test_moe_capacity_config.py \
  tests/test_moe_topology_contracts.py tests/test_moe_prefill_batching_default.py \
  tests/test_moe_prefill_piece_config.py
```

Benchmark drivers keep model paths and workload sizes explicit:

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

Select the visible GPUs before multi-GPU tests. Layer benchmarks with random
weights do not measure full-model inference. Performance comparisons exclude
startup only where explicitly stated. BF16 parallel reductions and batch
composition can change greedy tokens; short checks do not establish arbitrary
long-output equality. The compact MoE graph has single-GPU replay coverage;
its latest NCCL layout and paired end-to-end speedup still need validation.

## Project layout

```text
nanovllm/
  engine/         Request lifecycle, scheduling, KV block management, model runner
  layers/         Attention, linear, quantization, MoE, and CUDA/Triton kernels
  models/         Qwen3, Qwen3-MoE, and EAGLE3 adapters
  distributed/    Parallel process-group and collective helpers
benchmarks/       Reproducible local performance drivers
tests/            GPU correctness and integration checks
docs/runtime.md   Runtime defaults, parallel/graph behavior, and known limits
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
