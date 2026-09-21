"""Run a small, reproducible Qwen3 FP8 backend comparison.

The vLLM path is intentionally separate from the local kernel microbenchmark:
this driver measures end-to-end request latency using the same token workload.
"""

import argparse
import json
from time import perf_counter


def summarize_result(
    *,
    backend: str,
    quantization: str,
    batch_size: int,
    prompt_tokens: int,
    decode_tokens: int,
    elapsed_ms: float,
    fp8_format: str = "per_token",
) -> dict:
    return {
        "backend": backend,
        "quantization": quantization,
        "fp8_format": fp8_format,
        "batch_size": batch_size,
        "prompt_tokens": prompt_tokens,
        "decode_tokens": decode_tokens,
        "elapsed_ms": float(elapsed_ms),
    }


def validate_result(result: dict) -> None:
    if result.get("backend") not in {"vllm", "nanovllm"}:
        raise ValueError("backend must be vllm or nanovllm")
    if result.get("quantization") != "fp8":
        raise ValueError("this benchmark only compares fp8")
    if result.get("fp8_format") not in {"per_tensor", "per_token"}:
        raise ValueError("fp8_format must be per_tensor or per_token")
    for key in ("batch_size", "prompt_tokens", "decode_tokens"):
        if int(result.get(key, 0)) <= 0:
            raise ValueError(f"{key} must be positive")
    if float(result.get("elapsed_ms", 0.0)) <= 0:
        raise ValueError("elapsed_ms must be positive")


def validate_backend(
    backend: str, quantization: str, fp8_format: str = "per_token"
) -> None:
    if backend not in {"vllm", "nanovllm"}:
        raise ValueError("backend must be vllm or nanovllm")
    if backend == "vllm" and quantization != "fp8":
        raise ValueError("vllm comparison only supports fp8")
    if backend == "vllm" and fp8_format != "per_token":
        raise ValueError("vllm comparison uses per_token activation quantization")


def measure_vllm(args) -> dict:
    # vLLM's V1 engine starts an EngineCore process. CUDA contexts cannot be
    # inherited through fork, so this driver must opt into spawn before the
    # engine is constructed.
    import multiprocessing as mp
    import os

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    mp.set_start_method("spawn", force=True)
    import torch
    from vllm import LLM, SamplingParams, TokensPrompt

    engine = None
    try:
        engine = LLM(
            model=args.model,
            tokenizer=args.model,
            dtype="bfloat16",
            quantization="fp8",
            trust_remote_code=True,
            enforce_eager=True,
            max_model_len=args.prompt_tokens + args.decode_tokens,
            max_num_batched_tokens=args.prompt_tokens + args.decode_tokens,
            max_num_seqs=args.batch_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_prefix_caching=False,
        )
        prompts = [
            TokensPrompt(prompt_token_ids=[1] * args.prompt_tokens)
            for _ in range(args.batch_size)
        ]
        sampling_params = SamplingParams(
            temperature=0.0,
            ignore_eos=True,
            max_tokens=args.decode_tokens,
        )
        engine.generate(prompts, sampling_params)
        torch.cuda.synchronize()
        start = perf_counter()
        engine.generate(prompts, sampling_params)
        torch.cuda.synchronize()
        result = summarize_result(
            backend="vllm",
            quantization="fp8",
            batch_size=args.batch_size,
            prompt_tokens=args.prompt_tokens,
            decode_tokens=args.decode_tokens,
            elapsed_ms=(perf_counter() - start) * 1000,
            fp8_format="per_token",
        )
        validate_result(result)
        return result
    finally:
        if engine is not None:
            engine.llm_engine.engine_core.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data1/model/qwen/Qwen/Qwen3-8B")
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    args = parser.parse_args()
    print(json.dumps(measure_vllm(args), indent=2))


if __name__ == "__main__":
    main()
