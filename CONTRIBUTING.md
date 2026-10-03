# Contributing to extend-nano-vLLM

Thanks for improving extend-nano-vLLM.

## Scope

Prefer changes that improve inference behavior, validation, or reproducible
measurement. Keep each feature focused and preserve a clear numerical or
functional reference path when practical.

## Development guidelines

1. Keep CUDA, model, and checkpoint assumptions explicit in code and docs.
2. Add a focused test under `tests/` for behavior changes. Contract and layer
   tests use pytest; real-model GPU integration drivers can use
   `python -m tests.<module>`. Configuration contracts use temporary config-only
   checkpoints rather than requiring private model paths.
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
