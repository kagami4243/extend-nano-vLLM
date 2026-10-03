"""Compare TP=1/2/4 end-to-end throughput on a fixed greedy workload.

Run ``CUDA_VISIBLE_DEVICES=1,2,3,4 python -m benchmarks.bench_tp_scaling``.
TP may improve capacity without improving latency, especially on small models.
``TORCHDYNAMO_DISABLE=1`` can isolate CUDA Graph execution when first-use
Inductor compilation is slow; the JSON result records whether it was set.
"""

import argparse
import json
import os
from statistics import median
import subprocess
import sys
from time import perf_counter

import torch


def measure_tp(args, tp: int) -> dict:
    from nanovllm import LLM, SamplingParams

    llm = LLM(
        args.model,
        tensor_parallel_size=tp,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=True,
        max_model_len=args.prompt_tokens + args.output_tokens + 8,
        max_num_batched_tokens=args.prompt_tokens,
        max_num_seqs=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    prompts = (
        [args.prompt_text] * args.batch_size
        if args.prompt_text is not None
        else [[100 + (i + request) % 97 for i in range(args.prompt_tokens)]
              for request in range(args.batch_size)]
    )
    actual_prompt_tokens = (
        len(llm.tokenizer.encode(args.prompt_text))
        if args.prompt_text is not None else args.prompt_tokens
    )
    sampling = SamplingParams(temperature=0, ignore_eos=True, max_tokens=args.output_tokens)
    try:
        llm.generate(prompts, sampling, use_tqdm=False)
        samples = []
        outputs = None
        for _ in range(args.runs):
            torch.cuda.synchronize()
            start = perf_counter()
            current = llm.generate(prompts, sampling, use_tqdm=False)
            torch.cuda.synchronize()
            samples.append(perf_counter() - start)
            token_ids = [output["token_ids"] for output in current]
            if outputs is not None:
                assert token_ids == outputs
            outputs = token_ids
        seconds = median(samples)
        return {
            "tp": tp,
            "actual_prompt_tokens": actual_prompt_tokens,
            "cuda_graph_decode": bool(getattr(llm.model_runner, "graphs", {})),
            "prefill_piece_enabled": llm.model_runner.prefill_piece_enabled,
            "rank_diagnostics": llm.model_runner.call("get_diagnostics"),
            "latency_ms": [1000 * value for value in samples],
            "median_ms": 1000 * seconds,
            "output_tokens_per_s": args.batch_size * args.output_tokens / seconds,
            "outputs": outputs,
        }
    finally:
        llm.exit()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./models/Qwen3-0.6B")
    parser.add_argument("--prompt-text")
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--tp", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--child-tp", type=int)
    args = parser.parse_args()
    if min(args.prompt_tokens, args.output_tokens, args.batch_size, args.runs) < 1:
        parser.error("workload dimensions and runs must be positive")
    if any(tp not in (1, 2, 4) for tp in args.tp):
        parser.error("TP sizes must be 1, 2, or 4")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.child_tp:
        print(json.dumps(measure_tp(args, args.child_tp)))
        return
    results = {}
    for tp in args.tp:
        command = [
            sys.executable, "-m", "benchmarks.bench_tp_scaling",
            "--model", args.model, "--prompt-tokens", str(args.prompt_tokens),
            "--output-tokens", str(args.output_tokens), "--batch-size", str(args.batch_size),
            "--runs", str(args.runs), "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--child-tp", str(tp),
        ]
        if args.enforce_eager:
            command.append("--enforce-eager")
        if args.prompt_text is not None:
            command.extend(("--prompt-text", args.prompt_text))
        completed = subprocess.run(
            command,
            check=False, capture_output=True, text=True, timeout=900,
        )
        if completed.returncode:
            raise RuntimeError(f"TP={tp} failed:\n{completed.stderr}")
        results[tp] = json.loads(completed.stdout.strip().splitlines()[-1])
    baseline = results[args.tp[0]]["outputs"]
    for result in results.values():
        assert result["outputs"] == baseline, result["tp"]
        assert result["cuda_graph_decode"] is not args.enforce_eager
        assert result["prefill_piece_enabled"] is not args.enforce_eager
        assert len(result["rank_diagnostics"]) == result["tp"]
    print(json.dumps({
        "model": args.model,
        "batch_size": args.batch_size,
        "prompt_tokens": args.prompt_tokens,
        "output_tokens": args.output_tokens,
        "runs": args.runs,
        "enforce_eager": args.enforce_eager,
        "torchdynamo_disabled": os.environ.get("TORCHDYNAMO_DISABLE") == "1",
        "results": [
            {
                **{key: value for key, value in result.items()
                   if key not in ("outputs", "rank_diagnostics")},
                "parameter_bytes_per_rank": [rank["parameter_bytes"] for rank in result["rank_diagnostics"]],
                "kv_cache_bytes_per_rank": [rank["kv_cache_bytes"] for rank in result["rank_diagnostics"]],
                "speedup_vs_first": results[args.tp[0]]["median_ms"] / result["median_ms"],
            }
            for result in results.values()
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
