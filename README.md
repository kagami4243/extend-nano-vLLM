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

## Features

The list includes mechanisms inherited from nano-vLLM and the extensions in
this project. Experimental paths and their current constraints are identified
alongside each feature.

### Model execution and generation

- **Dense Qwen3 and Qwen3-MoE**: model-type registry, local Hugging Face
  checkpoint loading, and per-rank weight loading for supported parallel layouts.
- **Offline batched generation**: text or token-ID prompts, shared or per-request
  sampling parameters, and ordered results containing text and generated token IDs.
- **Greedy and temperature sampling**: `temperature=0` selects argmax;
  positive temperatures use categorical sampling, with EOS and output-length limits.
- **Request lifecycle APIs**: `add_request()`, `step()`, `is_finished()` and
  `generate()`, plus explicit engine cleanup through `exit()`.

### Attention, KV cache and scheduling

- **Paged KV cache**: fixed 256-token blocks, logical-to-physical block tables,
  reference counting, GPU-memory-based capacity allocation, and paged decode attention.
- **FlashAttention 2**: variable-length causal prefill and KV-cache decode;
  Triton kernels write new K/V values into physical cache slots.
- **Prefix caching**: reuse complete computed prefix blocks across requests,
  validate hash/token contents, remove stale entries, and report cache hits and reused tokens.
- **Iteration-level decode batching**: advance active requests together, release
  completed requests, and preempt/recompute requests when KV blocks are exhausted.
- **Chunked prefill**: split long prompts by `max_num_batched_tokens`, reuse
  existing KV, and give running decode requests a turn after a prefill step.
- **Multi-request prefill batching**: several waiting requests can share one
  token budget; enabled by default for MoE and opt-in for dense Qwen3.

### Distributed execution

- **Tensor parallelism (TP)**: shard vocab embeddings/LM head, QKV and dense
  MLP projections, with explicit TP groups and collective operations.
- **Pipeline parallelism (PP)**: partition consecutive layers and local KV
  storage, transfer activations between stages, and return sampled tokens;
  currently synchronous, one microbatch at a time, with eager execution.
- **Data parallelism (DP)**: independent replica engines, schedulers and KV
  pools; offline round-robin request routing and restoration of input order.
- **Combined layouts**: support TP×PP and DP×TP; enabled expert parallelism
  derives **EP=DP×TP**, with no additional process axis. Cross-DP EP requires PP=1.
- **Cross-DP EP coordination**: pad unequal token counts for communication,
  remove padding before routing, trim local outputs, and use dummy forwards on
  early-finished replicas until global completion. Prefix caching stays enabled.

### MoE routing and expert execution

- **Top-k routing and local experts**: FP32 softmax, optional top-k probability
  normalization, local expert weight shards, and weighted combination of expert outputs.
- **Grouped Triton GEMMs**: expert-major route packing, 16-row alignment,
  gate/up and down projections, GPU SwiGLU, and route-order output combination.
- **Compact graph-safe expert tasks**: fixed-capacity route indices, GPU
  effective lengths, `M*top_k` activation rows, and early exit for inactive GEMM tiles.
- **Dispatch backends**: replicated inputs with output reduction; cross-DP
  all-gather with all-reduce or reduce-scatter/local TP all-gather;
  optional eager all-to-all dispatch/combine experiments.
- **Shared experts and capacity controls**: shared-expert execution, fixed
  per-expert capacity or a token-scaled capacity factor, and overflow token dropping.
- **Expert placement and migration**: explicit per-layer expert ownership and
  controlled relocation via `relocate_experts()`; cross-DP migration is unsupported.

### Graphs, compilation and low precision

- **Full-forward decode CUDA Graphs**: capture attention and MoE in multiple
  batch buckets, up to 512 tokens per GPU; 36 buckets at the maximum, with
  cross-DP MoE capacity of `DP*512`. Logits and sampling remain outside the graphs.
- **Piecewise prefill CUDA Graphs**: capture static pieces around eager
  attention; dense Qwen3 enables this by default, while MoE requires explicit
  capture sizes and keeps attention/MoE calls eager between pieces.
- **Local `torch.compile` optimization**: compile decorated normalization,
  activation and sampling operations; graph capture and compilation are separate controls.
- **W4A16 weight quantization**: online group-wise packed INT4 linear weights
  for dense Qwen3, with BF16/FP16 activations and Triton GEMMs.
- **FP8 W8A8 linear execution**: online E4M3 weights/activations for dense Qwen3,
  with per-tensor activation scaling by default and optional per-token scaling;
  per-token prefill is eager, while decode can use CUDA Graphs.
- **FP8 E4M3 KV cache**: optional single-GPU storage with Triton page writes
  and paged decode attention; prefill dequantizes KV for FlashAttention 2.
  The teaching path uses `head_dim=128` and fixed per-layer K/V scales of 1.

### Speculative decoding and validation

- **EAGLE3 speculative decoding**: independent draft-model execution, greedy
  proposal/target verification, accepted-prefix and replacement-token handling,
  separate KV state, checkpoint/rollback, and batched request support.
  Currently DP=TP=PP=1, prefix cache off, target verification eager, and no FP8 KV.
- **Runtime diagnostics**: parallel ranks, parameter bytes, communication and
  graph replay counts, prefix reuse, and speculative acceptance statistics.
- **Tests and benchmarks**: CPU contracts with config-only checkpoints, GPU
  layer/graph checks, distributed integration drivers, and warmup-aware latency,
  throughput and GPU-profile measurements for supported features.

### Current scope

CUDA/NVIDIA and local Qwen3 checkpoints are the supported target. There is no
OpenAI-compatible server, streaming API, ROCm/SDPA fallback, top-k/top-p sampler,
automatic EPLB, CP or PD separation. Scheduling alternates prefill and decode
phases rather than mixing them in one forward. Unequal cross-DP batches fall
back to eager for the affected forward. Quantized checkpoints and quantized MoE
are not supported; performance and long-output numerical equality require
workload-specific verification.

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
