"""Measure extend-nano-vLLM BF16, W4A16, and FP8 inference under one workload.

Run as ``python -m benchmarks.bench_quantization ...`` from the repository
root. Run one quantization mode per process so allocator state from an earlier
model does not affect the next result. The default workload deliberately uses
a large prefill matrix (16 x 512 tokens) and a 16-sequence decode batch: those
are materially more favorable to quantized GEMMs than batch-1 decoding.
"""

import argparse
import gc
import json
from types import SimpleNamespace
from time import perf_counter

import torch


def measure(callable_):
    torch.cuda.synchronize()
    start = perf_counter()
    result = callable_()
    torch.cuda.synchronize()
    return result, perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("quantization", choices=("none", "w4a16", "fp8"))
    parser.add_argument("--model", required=True, help="Local Hugging Face model directory")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument(
        "--backend", choices=("nanovllm", "vllm"), default="nanovllm"
    )
    parser.add_argument(
        "--fp8-format", choices=("per_tensor", "per_token"), default="per_tensor"
    )
    parser.add_argument(
        "--kv-cache-dtype",
        choices=("auto", "fp8"),
        default="auto",
        help="KV cache storage; 'fp8' measures the FP8 paged KV cache",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="disable CUDA graphs; deployment-style graph execution is the default",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.prompt_tokens < 1 or args.decode_tokens < 1:
        parser.error("batch size, prompt tokens, and decode tokens must be positive")

    if args.backend == "vllm":
        if args.quantization != "fp8":
            parser.error("the vllm backend currently supports only fp8")
        if args.fp8_format != "per_token":
            parser.error("the vllm backend uses per_token FP8 activation quantization")
        from benchmarks.bench_fp8_compare import measure_vllm

        result = measure_vllm(
            SimpleNamespace(
                model=args.model,
                prompt_tokens=args.prompt_tokens,
                decode_tokens=args.decode_tokens,
                batch_size=args.batch_size,
                gpu_memory_utilization=0.8,
                fp8_format=args.fp8_format,
            )
        )
        print(json.dumps(result, indent=2))
        return

    from nanovllm import LLM, SamplingParams

    quantization = None if args.quantization == "none" else args.quantization
    max_model_len = args.prompt_tokens + args.batch_size + args.decode_tokens + 1
    llm = LLM(
        args.model,
        quantization=quantization,
        fp8_format=args.fp8_format,
        kv_cache_dtype=args.kv_cache_dtype,
        enforce_eager=args.enforce_eager,
        max_model_len=max_model_len,
        # The current scheduler admits one prefill sequence per step. Keeping
        # this equal to a single prompt avoids an unrealistically large model
        # warmup allocation while preserving the measured execution shape.
        max_num_batched_tokens=args.prompt_tokens,
        max_num_seqs=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
    )
    try:
        # Use the measured prompt shape for warmup and exclude first graph
        # capture from steady-state prefill throughput.
        llm.add_request(
            [index % 10000 for index in range(args.prompt_tokens)],
            SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=1),
        )
        while not llm.is_finished():
            llm.step()
        prompts = [
            [index % 10000 for index in range(args.prompt_tokens)]
            for _ in range(args.batch_size)
        ]
        # The current scheduler admits one prefill sequence at a time and
        # interleaves decode steps while later prompts wait. Reserve enough
        # output tokens for that admission phase plus the measured decode.
        sampling = SamplingParams(
            temperature=0.0,
            ignore_eos=True,
            max_tokens=args.batch_size + args.decode_tokens + 1,
        )
        for prompt in prompts:
            llm.add_request(prompt, sampling)

        prefill_seconds = 0.0
        scheduled_prefill_tokens = 0
        while llm.scheduler.waiting:
            (_, num_tokens), step_seconds = measure(llm.step)
            if num_tokens > 0:
                prefill_seconds += step_seconds
                scheduled_prefill_tokens += num_tokens
        if scheduled_prefill_tokens != args.batch_size * args.prompt_tokens:
            raise RuntimeError(
                "prefill did not schedule the full requested batch: "
                f"got {scheduled_prefill_tokens}"
            )

        def decode_steps():
            for _ in range(args.decode_tokens):
                llm.step()

        _, decode_seconds = measure(decode_steps)
        result = {
            "model": args.model,
            "quantization": args.quantization,
            "fp8_format": args.fp8_format,
            "kv_cache_dtype": args.kv_cache_dtype,
            "requested_enforce_eager": args.enforce_eager,
            "enforce_eager": bool(llm.model_runner.enforce_eager),
            "prefill_piece_enabled": bool(
                llm.model_runner.prefill_piece_enabled
            ),
            "batch_size": args.batch_size,
            "prompt_tokens_per_request": args.prompt_tokens,
            "decode_tokens_per_request": args.decode_tokens,
            "prefill_steps": args.batch_size,
            "prefill_ms": 1000 * prefill_seconds,
            "prefill_tokens_per_s": scheduled_prefill_tokens / prefill_seconds,
            "decode_ms": 1000 * decode_seconds,
            "decode_tokens_per_s": (
                args.batch_size * args.decode_tokens / decode_seconds
            ),
        }
        print(json.dumps(result, indent=2))
    finally:
        llm.exit()
        del llm
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
