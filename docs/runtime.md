# Runtime Guide

This guide describes the current offline runtime. The implementation uses CUDA,
Triton, and local Hugging Face checkpoints; it does not provide an online
request server. Model loading can require substantially more memory than the
number of parameters activated per token suggests for a MoE checkpoint.

## Defaults

| Setting | Default / behavior |
| --- | --- |
| `enforce_eager` | `False`; use available CUDA Graph paths |
| `enable_prefix_caching` | `True`; EAGLE3 disables it |
| `enable_prefill_batching` | Derived when `None`: enabled for Qwen3-MoE, disabled for dense Qwen3 |
| `max_num_seqs` | 512; decode capture stops at 512 per GPU |
| `max_num_batched_tokens` | 16384; prefill token budget per step |
| `max_model_len` | 4096; prompt plus completion must fit |
| `kvcache_block_size` | Fixed at 256 |
| `moe_dispatch_backend` | `replicated`; automatically `allgather_reduce` for enabled cross-DP EP |
| `moe_prefill_piece` | `False`; enabling it requires explicit, positive capture sizes |

These are constructor defaults, not universally optimal workload settings.
Reducing `max_num_seqs` limits decode capture sizes and startup memory. Explicit
prefill capture sizes restrict the experimental MoE prefill cache.

## Parallel Topology

```text
world_size = DP * PP * TP
global_rank = (dp_rank * PP + pp_rank) * TP + tp_rank
EP = DP * TP                    # only when expert parallel is enabled
ep_rank = dp_rank * TP + tp_rank # within the same PP stage
```

There is no independent EP process axis. `expert_parallel_size`, if supplied,
must equal DP×TP; the obsolete `moe_global_dp` constructor switch is rejected.
Dense models use ordinary TP groups. Qwen3-MoE with TP>1 requires EP enabled.
DP=1 EP uses the TP group; cross-DP EP currently requires PP=1.

Cross-DP MoE gathers tokens across the expert group and removes duplicate TP
copies before routing. The default combines outputs with all-reduce. The
`allgather_reducescatter` alternative uses reduce-scatter followed by local TP
all-gather. Eager all-to-all backends are experiments and cannot use the graph
path. Shared experts and fixed/proportional capacity are supported. Explicit
placement must assign equal expert counts per rank; controlled migration is
supported outside cross-DP execution, without an automatic EPLB controller.

PP splits layers and local KV storage, but processes one microbatch
synchronously and requires `enforce_eager=True`. It does not overlap pipeline
stages across requests.

## Prefix Cache and Scheduling

Only complete computed KV blocks can be reused. Hash, token contents, and a
ready flag are checked. If the whole prompt is cached, its last token is still
forwarded to obtain logits. Partial prefill keeps its existing block table.
Each DP replica has its own prefix/KV cache; KV is not shared across replicas.

The scheduler spends a prefill token budget on waiting requests in order. With
batching enabled, multiple requests can advance; running decode gets a turn
after a prefill step. A forward contains one phase, not a mixed prefill/decode
batch. Configurable long/short admission, priority, and asynchronous scheduling
are not implemented.

Global EP coordinates actual token counts, phases, and prefill graph state.
Communication buffers are padded to the largest local token count; padding is
removed before routing so it does not consume expert capacity. Outputs are
trimmed back to each replica. Early-finished replicas run dummy forwards that
participate in communication without touching request KV, until global finish.

The public `generate_data_parallel` helper starts and exits engines per call,
so call time includes initialization. Global EP requires token-ID prompts,
balanced request counts, `ignore_eos=True`, and the same positive `max_tokens`.
Prompt lengths, chunk counts, and prefix hits may differ. Online cancellation,
independent arrivals, early EOS, and worker failure recovery remain unverified.

## Graph and Compile Paths

Decode captures complete model forward graphs at `1,2,4,8,16,32,...,512`,
restricted by `max_num_seqs`, plus its exact upper limit. The 512 upper limit
gives 36 buckets. Replay uses the smallest fitting bucket and stable input/KV
metadata buffers. Attention and MoE are inside the graph; logits, sampling,
and DP step coordination are outside it.

MoE uses fixed-capacity sorted route indices, per-expert padded counts, and a
device effective length. Activations have `M*top_k` rows; unused GEMM tiles exit
before loading weights or performing dot products. For a per-replica bucket B,
cross-DP routing sees M=DP×B; the total token bound is DP×512. This avoids
reserving B activation rows for every expert.

Dense prefill uses piecewise graphs around eager attention. MoE prefill is
opt-in and captures only specified token totals; its attention and MoE calls
remain eager between static pieces. Unequal cross-DP token counts, differing
phases, or inconsistent prefill capture state force the entire affected
forward eager, rather than only disabling MoE graphs. Later compatible steps
resume graphs. Padding does not yet let unequal real batches share a graph.

`enforce_eager=True` disables graphs; it does not disable locally decorated
`torch.compile` functions. `TORCHDYNAMO_DISABLE=1` is a separate compiler switch.
Per-token FP8 prefill uses eager execution, while decode can still use graphs.
EAGLE3 requires DP=TP=PP=1 and greedy sampling; target verification stays eager
and batch-independent numerical behavior is not guaranteed.

## Verification and Measurement

CPU configuration tests create temporary Hugging Face config-only directories;
they do not load model weights. Use `pytest` for the new contract tests and
module CLI entry points for real-model integration tests. Model defaults use
`./models/...`; pass `--model` where supported to select another checkpoint.
Install test dependencies with `pip install -e '.[test]'`.

| Check | Entry point |
| --- | --- |
| Prefix and chunk invariants | `tests/test_feature_contracts.py` |
| Batched prefill scheduling | `tests/test_prefill_batch_scheduler.py` |
| EP topology/configuration | `tests/test_moe_topology_contracts.py` |
| DP padding and communication | `tests/test_moe_dp_padding.py` |
| Prefix differences with global EP | `python -m tests.test_moe_dp_prefix_cache` (four GPUs, local MoE checkpoint) |
| Graph limits and compact activations | `tests/test_moe_graph_limits.py`, `tests/test_moe_compact_layout.py` |
| Real single-GPU graph generation | `benchmarks.bench_moe_compact_engine --dp 1 --tp 1 --model ...` |
| Random-weight MoE layer timing | `benchmarks.bench_moe_graph --tokens 512 --warmup 10 --repeats 50` |
| Prefix/chunk/TP measurements | `benchmarks.bench_prefix_cache`, `benchmarks.bench_chunked_prefill`, `benchmarks.bench_tp_scaling` |
| Real four-GPU global EP timing | `benchmarks.bench_moe_global_dp --model ...` |

The compact graph has prior single-GPU replay checks, capacity/skewed/empty
route coverage, and full Qwen3-30B-A3B generation at batches 1/17/512. A
single-GPU 1024-token shard simulation verifies layout math, not NCCL.
The latest compact implementation has not been rerun under multi-GPU NCCL or
paired against its previous version for full-model performance. A known BF16
down-projection/routing-weight rounding test remains a strict expected failure.
Parallel reductions and batch composition can change greedy tokens, including
observed dense TP divergences; long-output equivalence remains unproven.

Benchmark startup, compilation, capture, and warmup separately from steady
requests. State model, DP/TP/EP, graph mode, input/output sizes, cache reuse,
and timing boundaries. Random-weight layer timings do not establish full-model
speedup; profiler GPU activity and synchronized request wall time are different
measurements and cannot be subtracted to obtain exact CPU time.
