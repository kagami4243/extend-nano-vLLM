"""Profile one warmed Qwen3-MoE request on nano-vLLM rank 0."""

import argparse
import json
from time import perf_counter

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

from nanovllm import LLM, SamplingParams


def prompts(seed, batch_size, prompt_tokens):
    return [
        [1000 + seed * batch_size + request]
        + [100 + (position + request) % 97 for position in range(prompt_tokens - 1)]
        for request in range(batch_size)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./models/Qwen3-30B-A3B-Base")
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--tp", type=int)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--independent-ep", action="store_true")
    parser.add_argument("--placement-file")
    parser.add_argument("--dispatch-backend", choices=("replicated", "all_to_all", "all_to_all_reduce"),
                        default="replicated")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--moe-prefill-piece", action="store_true")
    parser.add_argument("--enable-prefill-batching", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--prompt-tokens", type=int, default=16)
    parser.add_argument("--output-tokens", type=int, default=4)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--control-runs", type=int, default=0)
    parser.add_argument("--result-file")
    parser.add_argument("--trace-file")
    args = parser.parse_args()
    if args.tp is None:
        args.tp = args.ep
    if min(args.ep, args.tp, args.pp, args.batch_size, args.prompt_tokens,
           args.output_tokens, args.warmup_runs) < 1:
        parser.error("all dimensions must be positive")
    if args.independent_ep or args.tp != args.ep:
        parser.error("independent EP axes are removed; this DP=1 profile uses EP=TP")
    topology = {"tensor_parallel_size": args.ep, "pipeline_parallel_size": args.pp}
    llm = LLM(
        args.model,
        **topology,
        enable_expert_parallel=True,
        moe_dispatch_backend=args.dispatch_backend,
        moe_expert_placement=args.placement_file,
        enforce_eager=not args.graph,
        enable_prefix_caching=True,
        max_model_len=args.prompt_tokens + args.output_tokens + 8,
        max_num_batched_tokens=args.batch_size * args.prompt_tokens,
        max_num_seqs=args.batch_size,
        enable_prefill_batching=args.enable_prefill_batching,
        moe_prefill_piece=args.moe_prefill_piece,
        moe_prefill_piece_capture_sizes=(
            (args.batch_size * args.prompt_tokens,)
            if args.moe_prefill_piece else ()
        ),
        gpu_memory_utilization=0.9,
    )
    sampling = SamplingParams(
        temperature=0, ignore_eos=True, max_tokens=args.output_tokens
    )
    try:
        for warmup in range(args.warmup_runs):
            llm.generate(prompts(1 + warmup, args.batch_size, args.prompt_tokens),
                         sampling, use_tqdm=False)
        def measure_controls(seed_start):
            latencies = []
            for repeat in range(args.control_runs):
                torch.cuda.synchronize()
                start = perf_counter()
                llm.generate(prompts(seed_start + repeat, args.batch_size,
                                     args.prompt_tokens), sampling, use_tqdm=False)
                torch.cuda.synchronize()
                latencies.append((perf_counter() - start) * 1000)
            return latencies

        control_before_ms = measure_controls(100)
        profile_seed = 100 + args.control_runs
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            start = perf_counter()
            cache_before = llm.cache_stats
            llm.generate(prompts(profile_seed, args.batch_size, args.prompt_tokens),
                         sampling, use_tqdm=False)
            torch.cuda.synchronize()
            wall_ms = (perf_counter() - start) * 1000
            cache_after = llm.cache_stats
        if args.trace_file:
            prof.export_chrome_trace(args.trace_file)
        control_after_ms = measure_controls(profile_seed + 1)
        averages = prof.key_averages()
        device_events = [
            event for event in prof.events()
            if event.device_type == DeviceType.CUDA
        ]
        nccl_events = [
            event for event in device_events
            if "nccl" in event.name.lower()
        ]
        intervals = sorted(
            (event.time_range.start, event.time_range.end)
            for event in device_events
        )
        union_us = 0.0
        if intervals:
            start, end = intervals[0]
            for next_start, next_end in intervals[1:]:
                if next_start > end:
                    union_us += end - start
                    start, end = next_start, next_end
                else:
                    end = max(end, next_end)
            union_us += end - start

        def operation(item):
            return {
                "name": item.key,
                "calls": item.count,
                "self_cpu_ms": item.self_cpu_time_total / 1000,
                "self_gpu_ms": item.self_device_time_total / 1000,
            }
        result = {
            "ep": args.ep,
            "tp": args.tp if args.independent_ep else args.ep,
            "pp": args.pp,
            "independent_ep": args.independent_ep,
            "placement_file": args.placement_file,
            "dispatch_backend": args.dispatch_backend,
            "cuda_graph_decode": args.graph,
            "moe_prefill_piece": args.moe_prefill_piece,
            "enable_prefill_batching": args.enable_prefill_batching,
            "batch_size": args.batch_size,
            "prompt_tokens": args.prompt_tokens,
            "output_tokens": args.output_tokens,
            "rank0_profile_only": True,
            "wall_ms_with_profiler": wall_ms,
            "profile_seed": profile_seed,
            "profile_cache_hits": (
                cache_after["prefix_cache_hits"] - cache_before["prefix_cache_hits"]
            ),
            "control_before_ms": control_before_ms,
            "control_after_ms": control_after_ms,
            "cpu_self_ms_sum": sum(item.self_cpu_time_total for item in averages) / 1000,
            "gpu_device_event_ms_sum": sum(
                event.self_device_time_total for event in device_events
            ) / 1000,
            "gpu_active_union_ms": union_us / 1000,
            "gpu_device_event_count": len(device_events),
            "nccl_device_event_count": len(nccl_events),
            "nccl_device_ms_sum": sum(
                event.self_device_time_total for event in nccl_events
            ) / 1000,
            "trace_file": args.trace_file,
            "top_cpu": [operation(item) for item in sorted(
                averages, key=lambda item: item.self_cpu_time_total, reverse=True
            )[:20]],
            "top_gpu": [operation(item) for item in sorted(
                averages, key=lambda item: item.self_device_time_total, reverse=True
            )[:20]],
            "selected_ops": [operation(item) for item in averages if any(
                term in item.key.lower() for term in (
                    "local_scalar", "item", "bincount", "sort", "all_reduce",
                    "all_to_all", "broadcast", "grouped_moe", "cudadevicesynchronize"
                )
            )],
        }
        if args.result_file:
            with open(args.result_file, "w") as output:
                json.dump(result, output, indent=2)
                output.write("\n")
        print(json.dumps(result))
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
