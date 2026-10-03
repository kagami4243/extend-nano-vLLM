"""Synchronized offline DP=2, TP=2, EP=4 MoE benchmark on four GPUs."""

import argparse
import json
import multiprocessing as mp
import socket
from statistics import median
from time import perf_counter
import traceback


MODEL = "./models/Qwen3-30B-A3B-Base"


def prompts(seed, rank, prompt_tokens):
    return [
        [1000 + seed * 4 + request]
        + [100 + (position + request) % 97 for position in range(prompt_tokens - 1)]
        for request in range(4) if request % 2 == rank
    ]


def worker(rank, port, barrier, output, args):
    from nanovllm import LLM, SamplingParams

    llm = None
    try:
        llm = LLM(
            args.model,
            data_parallel_size=2,
            data_parallel_rank=rank,
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            moe_dispatch_backend=args.dispatch_backend,
            master_port=port,
            run_id=f"moe_global_dp_{port}_{rank}",
            enforce_eager=False,
            enable_prefill_batching=True,
            enable_prefix_caching=not args.disable_prefix_cache,
            moe_prefill_piece=args.moe_prefill_piece,
            moe_prefill_piece_capture_sizes=(
                (args.prompt_tokens * 2,) if args.moe_prefill_piece else ()
            ),
            max_model_len=args.prompt_tokens + args.output_tokens + 8,
            max_num_batched_tokens=args.prompt_tokens * 2,
            max_num_seqs=2,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        diagnostics = llm.model_runner.call("get_diagnostics")
        sampling = SamplingParams(
            temperature=0, ignore_eos=True, max_tokens=args.output_tokens
        )
        for seed in range(args.warmup_runs):
            barrier.wait(timeout=args.timeout)
            llm.generate(prompts(seed, rank, args.prompt_tokens), sampling,
                         use_tqdm=False)
        samples = []
        first_outputs = None
        for seed in range(args.runs):
            barrier.wait(timeout=args.timeout)
            start = perf_counter()
            results = llm.generate(
                prompts(100 + seed, rank, args.prompt_tokens), sampling,
                use_tqdm=False,
            )
            samples.append((perf_counter() - start) * 1000)
            if first_outputs is None:
                first_outputs = [item["token_ids"] for item in results]
        profile_result = None
        if args.profile_trace:
            barrier.wait(timeout=args.timeout)
            profile_prompts = prompts(999, rank, args.prompt_tokens)
            if rank == 0:
                import torch
                from torch.autograd import DeviceType
                from torch.profiler import ProfilerActivity, profile

                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    profile_start = perf_counter()
                    llm.generate(profile_prompts, sampling, use_tqdm=False)
                    torch.cuda.synchronize()
                    profiled_wall_ms = (perf_counter() - profile_start) * 1000
                prof.export_chrome_trace(args.profile_trace)
                device_events = [
                    event for event in prof.events()
                    if event.device_type == DeviceType.CUDA
                ]
                nccl_events = [
                    event for event in device_events
                    if "nccl" in event.name.lower()
                ]
                profile_result = {
                    "trace": args.profile_trace,
                    "profiled_wall_ms": profiled_wall_ms,
                    "device_event_count": len(device_events),
                    "device_time_ms": sum(
                        event.self_device_time_total for event in device_events
                    ) / 1000,
                    "nccl_event_count": len(nccl_events),
                    "nccl_time_ms": sum(
                        event.self_device_time_total for event in nccl_events
                    ) / 1000,
                }
            else:
                llm.generate(profile_prompts, sampling, use_tqdm=False)
        after = llm.model_runner.call("get_diagnostics")
        output.put({
            "rank": rank,
            "samples_ms": samples,
            "first_outputs": first_outputs,
            "diagnostics": diagnostics,
            "decode_graph_replays": [
                later["decode_graph_replay_count"]
                - earlier["decode_graph_replay_count"]
                for earlier, later in zip(diagnostics, after)
            ],
            "dispatch_backend": llm.config.moe_dispatch_backend,
            "prefill_piece_captured_sizes": sorted(
                llm.model_runner.prefill_piece_graphs
            ),
            "rank0_profile": profile_result,
        })
    except Exception:
        output.put({"rank": rank, "error": traceback.format_exc()})
        raise
    finally:
        if llm is not None:
            llm.exit()


def run(args):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    context = mp.get_context("spawn")
    barrier = context.Barrier(2)
    output = context.Queue()
    processes = [
        context.Process(target=worker, args=(rank, port, barrier, output, args))
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        observations = [output.get(timeout=args.timeout) for _ in processes]
        for process in processes:
            process.join(timeout=30)
        if any("error" in item for item in observations):
            raise RuntimeError(str(observations))
        if any(process.exitcode != 0 for process in processes):
            raise RuntimeError("global DP worker exited unsuccessfully")
        observations.sort(key=lambda item: item["rank"])
        round_ms = [max(times) for times in zip(*(
            item["samples_ms"] for item in observations
        ))]
        return {
            "backend": "nano",
            "model": args.model,
            "dp": 2,
            "tp": 2,
            "effective_ep": 4,
            "global_dp_expert_group": True,
            "dispatch_backend": args.dispatch_backend,
            "batch_size": 4,
            "prompt_tokens": args.prompt_tokens,
            "output_tokens": args.output_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enable_prefill_batching": True,
            "enable_prefix_caching": not args.disable_prefix_cache,
            "cuda_graph_enabled": True,
            "moe_prefill_piece": args.moe_prefill_piece,
            "latency_scope": "maximum synchronous generate wall over DP ranks",
            "latency_ms": round_ms,
            "latency_ms_median": median(round_ms),
            "rank_results": observations,
        }
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--prompt-tokens", type=int, default=16)
    parser.add_argument("--output-tokens", type=int, default=4)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--dispatch-backend",
        choices=("allgather_reducescatter", "allgather_reduce"),
        default="allgather_reduce",
    )
    parser.add_argument("--disable-prefix-cache", action="store_true")
    parser.add_argument(
        "--moe-prefill-piece", action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--profile-trace")
    parser.add_argument("--result-file")
    args = parser.parse_args()
    if min(args.prompt_tokens, args.output_tokens, args.warmup_runs, args.runs) < 1:
        parser.error("workload dimensions and runs must be positive")
    result = run(args)
    if args.result_file:
        with open(args.result_file, "w") as file:
            json.dump(result, file, indent=2)
            file.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
