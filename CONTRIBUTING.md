# Contributing to nano-vLLM

Thanks for improving this learning project.

## Scope

Prefer changes that make an inference-engine concept easier to inspect,
measure, or validate. Keep a feature small and document its deliberate limits;
this repository does not aim to duplicate all of vLLM.

## Development guidelines

1. Keep CUDA, model, and checkpoint assumptions explicit in code and docs.
2. Add a focused test under `tests/` for behavior changes. GPU tests should be
   runnable as `python -m tests.<module>`.
3. Put reproducible timing scripts in `benchmarks/`; report hardware, model,
   workload, warm-up, and what the measurement excludes.
4. Preserve numerical reference paths when adding a kernel where practical.
5. Attribute copied or adapted third-party code and retain its license notice.

## Before opening a pull request

```bash
git diff --check
python -m compileall -q nanovllm tests benchmarks
```

Run the relevant GPU test modules when compatible hardware is available. State
clearly when a test was not run and why.

## Issues

Please include the model path layout (without private data), GPU model, CUDA,
PyTorch, FlashAttention, and Triton versions, together with the smallest
reproduction command.
