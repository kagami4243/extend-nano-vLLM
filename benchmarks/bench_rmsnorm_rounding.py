"""CUDA Graph and eager timing for the fused residual RMSNorm."""

import argparse
import json
import os
from statistics import median
from time import perf_counter

import torch

from nanovllm.layers.layernorm import RMSNorm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--replays-per-run", type=int, default=100)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--without-residual", action="store_true")
    parser.add_argument("--deterministic-cuda", action="store_true")
    parser.add_argument("--q-heads", type=int, default=0)
    parser.add_argument("--result-file")
    args = parser.parse_args()
    if min(args.tokens, args.hidden_size, args.runs, args.replays_per_run) < 1 or args.q_heads < 0:
        parser.error("all dimensions and repeat counts must be positive")
    if args.q_heads and not args.without_residual:
        parser.error("--q-heads requires --without-residual")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")

    torch.manual_seed(17)
    norm = RMSNorm(
        args.hidden_size, deterministic_cuda=args.deterministic_cuda
    ).cuda().bfloat16().eval()
    x = (
        torch.randn(args.tokens, args.q_heads + 8, args.hidden_size,
                    device="cuda", dtype=torch.bfloat16)[:, :args.q_heads]
        if args.q_heads else torch.randn(
            args.tokens, args.hidden_size, device="cuda", dtype=torch.bfloat16
        )
    )
    residual = torch.randn_like(x)
    def execute():
        return norm(x) if args.without_residual else norm(x, residual)

    with torch.inference_mode():
        for _ in range(5):
            execute()
        torch.cuda.synchronize()
        graph = None
        if not args.enforce_eager:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = execute()
            graph.replay()
            torch.cuda.synchronize()

        samples = []
        gpu_samples = []
        for _ in range(args.runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            wall_start = perf_counter()
            start_event.record()
            for _ in range(args.replays_per_run):
                if graph is None:
                    result = execute()
                else:
                    graph.replay()
            end_event.record()
            torch.cuda.synchronize()
            samples.append((perf_counter() - wall_start) * 1000 / args.replays_per_run)
            gpu_samples.append(start_event.elapsed_time(end_event) / args.replays_per_run)

    output = {
        "tokens": args.tokens,
        "hidden_size": args.hidden_size,
        "q_heads": args.q_heads,
        "dtype": "bfloat16",
        "cuda_graph": graph is not None,
        "without_residual": args.without_residual,
        "deterministic_cuda": args.deterministic_cuda,
        "torchdynamo_disabled": os.environ.get("TORCHDYNAMO_DISABLE") == "1",
        "runs": args.runs,
        "replays_per_run": args.replays_per_run,
        "host_wall_ms_median": median(samples),
        "gpu_event_ms_median": median(gpu_samples),
        "output_checksum": float((result if args.without_residual else result[0]).float().sum()),
        "residual_checksum": None if args.without_residual else float(result[1].float().sum()),
    }
    if args.result_file:
        with open(args.result_file, "w") as file:
            json.dump(output, file, indent=2)
            file.write("\n")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
