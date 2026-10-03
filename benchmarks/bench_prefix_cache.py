"""Measure repeated-prefix inference with and without the KV prefix cache.

Run ``CUDA_VISIBLE_DEVICES=1 python -m benchmarks.bench_prefix_cache``.
The reported speedup is measured, never assumed. Each mode has its own process.
"""

import argparse
import json
from statistics import median
import subprocess
import sys
from time import perf_counter

import torch


def measure_mode(args, enabled: bool) -> dict:
    from nanovllm import LLM, SamplingParams

    print(f"cache={'on' if enabled else 'off'}: initializing", file=sys.stderr, flush=True)
    llm = LLM(
        args.model,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=enabled,
        max_model_len=args.prefix_tokens + args.suffix_tokens + args.output_tokens + 8,
        max_num_batched_tokens=args.prefix_tokens + args.suffix_tokens,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling = SamplingParams(temperature=0, ignore_eos=True, max_tokens=args.output_tokens)
    prefix = [100 + i % 97 for i in range(args.prefix_tokens)]
    try:
        # Warm kernels and populate the cached prefix before timing.
        print("warming", file=sys.stderr, flush=True)
        llm.generate([prefix + [700] * args.suffix_tokens], sampling, use_tqdm=False)
        # Cache hits shrink the prefill from full prompt to suffix-only. Capture
        # that second graph shape before the steady-state timing window.
        llm.generate([prefix + [701] * args.suffix_tokens], sampling, use_tqdm=False)
        samples = []
        outputs = []
        before = llm.cache_stats
        for i in range(args.runs):
            print(f"run {i + 1}/{args.runs}", file=sys.stderr, flush=True)
            prompt = prefix + [800 + i] * args.suffix_tokens
            torch.cuda.synchronize()
            start = perf_counter()
            output = llm.generate([prompt], sampling, use_tqdm=False)
            torch.cuda.synchronize()
            samples.append(1000 * (perf_counter() - start))
            outputs.append(output[0]["token_ids"])
        after = llm.cache_stats
        return {
            "enabled": enabled,
            "cuda_graph_decode": bool(getattr(llm.model_runner, "graphs", {})),
            "prefill_piece_enabled": llm.model_runner.prefill_piece_enabled,
            "latency_ms": samples,
            "median_ms": median(samples),
            "outputs": outputs,
            "reused_tokens": after["reused_tokens"] - before["reused_tokens"],
            "prefix_cache_hits": after["prefix_cache_hits"] - before["prefix_cache_hits"],
        }
    finally:
        llm.exit()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./models/Qwen3-0.6B")
    parser.add_argument("--prefix-tokens", type=int, default=256)
    parser.add_argument("--suffix-tokens", type=int, default=16)
    parser.add_argument("--output-tokens", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--child-mode", choices=("off", "on"))
    args = parser.parse_args()
    if args.prefix_tokens < 256 or args.suffix_tokens < 1 or args.output_tokens < 1 or args.runs < 1:
        parser.error("prefix must be >=256 and suffix, output, runs must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.child_mode:
        print(json.dumps(measure_mode(args, args.child_mode == "on")))
        return
    results = {}
    for mode in ("off", "on"):
        command = [
            sys.executable, "-m", "benchmarks.bench_prefix_cache",
            "--model", args.model, "--prefix-tokens", str(args.prefix_tokens),
            "--suffix-tokens", str(args.suffix_tokens), "--output-tokens", str(args.output_tokens),
            "--runs", str(args.runs), "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--child-mode", mode,
        ]
        if args.enforce_eager:
            command.append("--enforce-eager")
        completed = subprocess.run(
            command,
            check=True, stdout=subprocess.PIPE, text=True, timeout=600,
        )
        results[mode] = json.loads(completed.stdout.strip().splitlines()[-1])
    assert results["off"]["outputs"] == results["on"]["outputs"]
    assert results["off"]["cuda_graph_decode"] is not args.enforce_eager
    assert results["on"]["cuda_graph_decode"] is not args.enforce_eager
    assert results["off"]["reused_tokens"] == 0
    assert results["on"]["reused_tokens"] >= args.runs * (args.prefix_tokens // 256) * 256
    print(json.dumps({
        "model": args.model,
        "prefix_tokens": args.prefix_tokens,
        "suffix_tokens": args.suffix_tokens,
        "output_tokens": args.output_tokens,
        "runs": args.runs,
        "enforce_eager": args.enforce_eager,
        "cache_off": {key: value for key, value in results["off"].items() if key != "outputs"},
        "cache_on": {key: value for key, value in results["on"].items() if key != "outputs"},
        "speedup": results["off"]["median_ms"] / results["on"]["median_ms"],
    }, indent=2))


if __name__ == "__main__":
    main()
