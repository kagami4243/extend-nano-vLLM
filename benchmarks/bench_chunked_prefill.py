"""Compare single-request TTFT with full and chunked graph prefill.

Run ``CUDA_VISIBLE_DEVICES=1 python -m benchmarks.bench_chunked_prefill``.
Prompts differ in their first token so enabled prefix caching cannot hide work.
"""

import argparse
import json
from statistics import median
import subprocess
import sys
from time import perf_counter

import torch


def measure_mode(args, budget: int) -> dict:
    from nanovllm import LLM, SamplingParams

    llm = LLM(
        args.model,
        enforce_eager=False,
        enable_prefix_caching=True,
        max_model_len=args.prompt_tokens + 8,
        max_num_batched_tokens=budget,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling = SamplingParams(temperature=0, ignore_eos=True, max_tokens=1)
    tail = [100 + i % 97 for i in range(args.prompt_tokens - 1)]
    try:
        # Capture the measured graph shape before timing.
        llm.generate([[700] + tail], sampling, use_tqdm=False)
        samples = []
        outputs = []
        before_hits = llm.cache_stats["reused_tokens"]
        before_steps = llm.scheduler.stats["prefill_steps"]
        for i in range(args.runs):
            torch.cuda.synchronize()
            start = perf_counter()
            output = llm.generate([[800 + i] + tail], sampling, use_tqdm=False)
            torch.cuda.synchronize()
            samples.append(1000 * (perf_counter() - start))
            outputs.append(output[0]["token_ids"])
        result = {
            "token_budget": budget,
            "latency_ms": samples,
            "median_ttft_ms": median(samples),
            "prefill_steps": llm.scheduler.stats["prefill_steps"] - before_steps,
            "reused_tokens": llm.cache_stats["reused_tokens"] - before_hits,
            "cuda_graph_decode": bool(getattr(llm.model_runner, "graphs", {})),
            "prefill_piece_enabled": llm.model_runner.prefill_piece_enabled,
            "outputs": outputs,
        }
        return result
    finally:
        llm.exit()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./models/Qwen3-0.6B")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--chunk-tokens", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--child-budget", type=int)
    args = parser.parse_args()
    if args.prompt_tokens < 2 or args.chunk_tokens < 1 or args.runs < 1:
        parser.error("prompt length must be >=2; chunk size and runs must be positive")
    if args.chunk_tokens >= args.prompt_tokens:
        parser.error("chunk size must be smaller than prompt length")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.child_budget is not None:
        print(json.dumps(measure_mode(args, args.child_budget)))
        return
    results = []
    for budget in (args.prompt_tokens, args.chunk_tokens):
        completed = subprocess.run(
            [sys.executable, "-m", "benchmarks.bench_chunked_prefill",
             "--model", args.model, "--prompt-tokens", str(args.prompt_tokens),
             "--chunk-tokens", str(args.chunk_tokens), "--runs", str(args.runs),
             "--gpu-memory-utilization", str(args.gpu_memory_utilization),
             "--child-budget", str(budget)],
            check=False, capture_output=True, text=True, timeout=600,
        )
        if completed.returncode:
            raise RuntimeError(f"budget={budget} failed:\n{completed.stderr}")
        results.append(json.loads(completed.stdout.strip().splitlines()[-1]))
    full, chunked = results
    assert full["outputs"] == chunked["outputs"]
    assert full["reused_tokens"] == chunked["reused_tokens"] == 0
    assert full["prefill_steps"] == args.runs
    assert chunked["prefill_steps"] == args.runs * ((args.prompt_tokens + args.chunk_tokens - 1) // args.chunk_tokens)
    assert all(result["cuda_graph_decode"] and result["prefill_piece_enabled"] for result in results)
    print(json.dumps({
        "model": args.model,
        "prompt_tokens": args.prompt_tokens,
        "chunk_tokens": args.chunk_tokens,
        "runs": args.runs,
        "full": {key: value for key, value in full.items() if key != "outputs"},
        "chunked": {key: value for key, value in chunked.items() if key != "outputs"},
        "speedup": full["median_ttft_ms"] / chunked["median_ttft_ms"],
    }, indent=2))


if __name__ == "__main__":
    main()
