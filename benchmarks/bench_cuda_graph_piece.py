"""Compare eager and piecewise-prefill latency on the same Qwen3 workload.

Two metrics are available. ``ttft`` is the wall time from admitting a request to
observing its first output token, which is what a client sees. ``prefill-step``
times only the step that computes the prompt and samples the first token, so it
excludes the polling and synchronization the TTFT loop adds around it.
"""

import argparse
import gc
import json
from statistics import median
from time import perf_counter

import torch


def summarize(mode: str, samples: list[float], prompt_tokens: int) -> dict:
    if not samples:
        raise ValueError("at least one latency sample is required")
    return {
        "mode": mode,
        "runs": len(samples),
        "prompt_tokens": prompt_tokens,
        "ttft_ms_median": median(samples),
    }


def validate_result(result: dict) -> None:
    if result.get("mode") not in {"eager", "piece"}:
        raise ValueError("mode must be eager or piece")
    if int(result.get("runs", 0)) <= 0:
        raise ValueError("runs must be positive")
    if float(result.get("ttft_ms_median", 0.0)) <= 0:
        raise ValueError("ttft_ms_median must be positive")


def measure_mode(args, mode: str) -> dict:
    from nanovllm import LLM, SamplingParams

    llm = LLM(
        args.model,
        enforce_eager=mode == "eager",
        max_model_len=args.prompt_tokens + args.output_tokens,
        max_num_batched_tokens=args.prompt_tokens + args.output_tokens,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
    )
    prompt = [1] * args.prompt_tokens
    sampling = SamplingParams(
        temperature=0.0, ignore_eos=True, max_tokens=args.output_tokens
    )
    try:
        llm.add_request(prompt, sampling)
        while not llm.is_finished():
            llm.step()
        samples = []
        for _ in range(args.runs):
            llm.add_request(prompt, sampling)
            if args.metric == "prefill-step":
                # The scheduler admits one request per step, so this window is
                # exactly one prompt computation as long as the step also emits
                # the first token, which only happens after the full prompt has
                # been processed.
                seq = llm.scheduler.waiting[-1]
                torch.cuda.synchronize()
                start = perf_counter()
                llm.step()
                torch.cuda.synchronize()
                if seq.num_completion_tokens < 1:
                    raise RuntimeError(
                        "prefill-step expects the prompt step to emit the first "
                        f"token, got {seq.num_completion_tokens} completion tokens"
                    )
                samples.append((perf_counter() - start) * 1000)
                while not llm.is_finished():
                    llm.step()
            else:
                seq = llm.scheduler.waiting[-1]
                torch.cuda.synchronize()
                start = perf_counter()
                first = None
                while not llm.is_finished():
                    llm.step()
                    torch.cuda.synchronize()
                    if first is None and seq.num_completion_tokens:
                        first = perf_counter()
                if first is None:
                    raise RuntimeError("request emitted no token")
                samples.append((first - start) * 1000)
        result = summarize(mode, samples, args.prompt_tokens)
        validate_result(result)
        result["metric"] = args.metric
        if args.metric == "prefill-step":
            result["prefill_step_ms_median"] = result.pop("ttft_ms_median")
        result["prefill_piece_enabled"] = bool(
            llm.model_runner.prefill_piece_enabled
        )
        result["prefill_piece_token_counts"] = sorted(
            llm.model_runner.prefill_piece_graphs
        )
        return result
    finally:
        llm.exit()
        del llm
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local Qwen3 model directory")
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument(
        "--metric", choices=("ttft", "prefill-step"), default="ttft"
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    result = {
        "device": torch.cuda.get_device_name(),
        "metric": args.metric,
        "eager": measure_mode(args, "eager"),
        "piece": measure_mode(args, "piece"),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
