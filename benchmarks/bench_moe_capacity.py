"""Single-layer capacity/drop cost on a Qwen3-30B-A3B shaped MoE layer."""

import argparse
import json
import math
from statistics import median
from time import perf_counter

import torch

from nanovllm.layers.moe import ExpertParallelMoE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--capacity", type=int, default=8)
    parser.add_argument("--capacity-factor", type=float)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--result-file")
    args = parser.parse_args()
    if min(args.tokens, args.warmup_runs, args.runs) < 1 or args.capacity < 1:
        parser.error("all dimensions must be positive")
    if args.capacity_factor is not None and (
        not math.isfinite(args.capacity_factor) or args.capacity_factor <= 0
    ):
        parser.error("capacity factor must be finite and positive")
    if args.graph and args.tokens > 32:
        parser.error("graph-safe decode supports at most 32 tokens")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    torch.manual_seed(41)
    baseline = ExpertParallelMoE(2048, 768, 128, 8, True).cuda().bfloat16()
    limited = ExpertParallelMoE(
        2048, 768, 128, 8, True,
        expert_capacity=None if args.capacity_factor is not None else args.capacity,
        expert_capacity_factor=args.capacity_factor,
    ).cuda().bfloat16()
    with torch.no_grad():
        for parameter in baseline.parameters():
            parameter.normal_(0, 0.02)
        limited.load_state_dict(baseline.state_dict())
    hidden = torch.randn(args.tokens, 2048, device="cuda", dtype=torch.bfloat16)

    @torch.inference_mode()
    def measure(module):
        module.graph_safe_decode = args.graph
        for _ in range(args.warmup_runs):
            module(hidden)
        if args.graph:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                module(hidden)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                module(hidden)
            run = graph.replay
        else:
            run = lambda: module(hidden)
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.runs):
            start = perf_counter()
            run()
            torch.cuda.synchronize()
            samples.append((perf_counter() - start) * 1000)
        return samples

    baseline_samples = measure(baseline)
    limited_samples = measure(limited)
    result = {
        "tokens": args.tokens,
        "experts": 128,
        "top_k": 8,
        "capacity": (
            args.capacity if args.capacity_factor is None else
            max(1, math.ceil(args.capacity_factor * args.tokens * 8 / 128))
        ),
        "capacity_factor": args.capacity_factor,
        "cuda_graph": args.graph,
        "baseline_ms": median(baseline_samples),
        "limited_ms": median(limited_samples),
        "speedup": median(baseline_samples) / median(limited_samples),
        "baseline_assignments": baseline.dispatch_assignment_count,
        "limited_assignments": limited.dispatch_assignment_count,
        "dropped_assignments": limited.dropped_assignment_count,
    }
    if args.result_file:
        with open(args.result_file, "w") as file:
            json.dump(result, file, indent=2)
            file.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
