# EAGLE3 Acceptance Ablation

## Short-prompt experiment

Date: 2026-08-21.  Configuration is the earlier short benchmark: Qwen3-8B
target, the same Qwen3-8B EAGLE3 checkpoint, batch size 1, prompt 1,024
tokens, response 256 tokens, `k=8`, greedy sampling, `temperature=0`,
`ignore_eos=True`, `enforce_eager=True`, and prefix caching disabled.

The three requested combinations were measured in separate processes. The
"old EAGLE" implementation is a temporary compatibility model in
`/data0/fwy/tmp/ablate_eagle_verify.py`: separate Q/K/V projections, separate
gate/up projections, standalone RMSNorm, and the original single-request draft
cache path. The checkpoint and target model are otherwise unchanged.

| Draft model | Target verify | Drafted | Accepted | Acceptance rate | Mean acceptance length |
| --- | --- | ---: | ---: | ---: | ---: |
| Current Qwen3-shared EAGLE | Per-token | 275 | 221 | **80.36%** | 7.31 |
| Previous-style EAGLE | Current multi-token varlen | 276 | 221 | **80.07%** | 7.31 |
| Previous-style EAGLE | Per-token | 685 | 170 | **24.82%** | 2.98 |

An additional control, current Qwen3-shared EAGLE with current multi-token
varlen verification, was `275 drafted / 221 accepted = 80.36%` on the same
short prompt.

## Interpretation

The short-prompt result does **not** support the hypothesis that changing the
EAGLE projections to Qwen3-shared kernels alone caused the 80% to 20% drop. On
this workload, both current and previous-style draft models reach about 80%
when paired with the current verification path.

The previous-style draft paired with the per-token verification path reaches
only 24.82%. This isolates a semantic interaction in the old verification
path: its one-token prefill/attention/KV update sequence is not equivalent to
the current varlen verification layout for this draft state. A rejection at an
early proposal position rejects the rest of that proposal prefix, so the
per-token mismatch compounds into a large acceptance-rate reduction.

The long CUDA Graph benchmark (`prompt=4096, response=1024`) measured the
current/current combination at 21.03%. Therefore the long-context result is
not explained by the Qwen3-shared draft model alone. It additionally includes
long-context KV/attention numerical differences and the current multi-token
verification versus target-only decode behavior.

The temporary old-model loader is only for this controlled acceptance
experiment; it is not a production implementation and is intentionally kept
outside the repository.
