"""Measure fixed-shape Qwen3-MoE layer eager and CUDA Graph decode latency."""

import argparse
import json
from statistics import median
from time import perf_counter

import torch

from nanovllm.layers.moe import ExpertParallelMoE, MOE_MAX_GRAPH_TOKENS_PER_RANK


@torch.inference_mode()
def measure(run, repeats, warmup):
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = perf_counter()
        run()
        torch.cuda.synchronize()
        samples.append((perf_counter() - start) * 1000)
    return median(samples)


def profile_gpu(run):
    from torch.profiler import profile, ProfilerActivity

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        run()
        torch.cuda.synchronize()
    events = [event for event in prof.events()
              if event.device_type == torch.autograd.DeviceType.CUDA]
    return {"gpu_activity_ms": sum(event.device_time_total for event in events) / 1000,
            "grouped_gemm_ms": sum(event.device_time_total for event in events
                                   if "grouped_moe_gemm" in event.name) / 1000,
            "gpu_activity_count": len(events)}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--profile-gpu", action="store_true")
    parser.add_argument("--result-file")
    args = parser.parse_args()
    if (not 1 <= args.tokens <= MOE_MAX_GRAPH_TOKENS_PER_RANK
            or args.repeats < 1 or args.warmup < 1):
        parser.error(f"tokens must be 1-{MOE_MAX_GRAPH_TOKENS_PER_RANK}; repeats/warmup must be positive")

    torch.manual_seed(17)
    module = ExpertParallelMoE(2048, 768, 128, 8, True).cuda().bfloat16()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(0, 0.02)
    static_hidden = torch.randn(
        args.tokens, 2048, device="cuda", dtype=torch.bfloat16
    )
    for _ in range(5):
        module(static_hidden)
    eager_output = module(static_hidden)
    eager_ms = measure(lambda: module(static_hidden), args.repeats, args.warmup)

    module.graph_safe_decode = True
    static_eager_ms = measure(lambda: module(static_hidden), args.repeats, args.warmup)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        module(static_hidden)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    result = {"tokens": args.tokens, "eager_ms": eager_ms,
              "static_eager_ms": static_eager_ms, "warmup": args.warmup,
              "repeats": args.repeats, "layout": "compact_token_route",
              "activation_rows": args.tokens * 8}
    try:
        with torch.cuda.graph(graph):
            output = module(static_hidden)
        graph.replay()
        torch.cuda.synchronize()
        result["graph_ms"] = measure(graph.replay, args.repeats, args.warmup)
        result["speedup"] = eager_ms / result["graph_ms"]
        result["output_finite"] = bool(torch.isfinite(output).all().item())
        result["matches_eager"] = bool(torch.allclose(
            output, eager_output, rtol=1e-2, atol=1e-3
        ))
        if args.profile_gpu:
            profile_results = {}
            for name, enabled, run in (
                ("dynamic_eager", False, lambda: module(static_hidden)),
                ("compact_eager", True, lambda: module(static_hidden)),
                ("compact_graph", True, graph.replay),
            ):
                module.graph_safe_decode = enabled
                profile_results[name] = profile_gpu(run)
            result["profile_one_iteration"] = profile_results
    except RuntimeError as error:
        result["graph_error"] = str(error).splitlines()[0]
    if args.result_file:
        with open(args.result_file, "w") as file:
            json.dump(result, file, indent=2)
            file.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
