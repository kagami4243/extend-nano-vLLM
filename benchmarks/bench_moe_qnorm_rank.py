"""Check cross-rank Q/K RMSNorm parity and graph replay cost on identical data."""

import argparse
import json
import socket
from statistics import median
from time import perf_counter

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nanovllm.layers.layernorm import RMSNorm


def worker(rank, world_size, port, tokens, heads, runs, replays, input_file,
           results):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}",
        rank=rank, world_size=world_size,
        device_id=torch.device("cuda", rank),
    )
    completed = False
    try:
        with torch.inference_mode():
            torch.manual_seed(2718)
            source = (
                torch.empty(tokens, heads + 8, 128, device="cuda", dtype=torch.bfloat16)
                if input_file else torch.randn(
                    tokens, heads + 8, 128, device="cuda", dtype=torch.bfloat16
                )
            )
            norm = RMSNorm(128).cuda().bfloat16()
            norm.weight.data.uniform_(0.5, 1.5)
            if input_file and rank == 0:
                vectors = torch.load(
                    input_file, map_location="cpu", weights_only=True
                )["vectors"]
                captured = vectors["1.qnorm_input_full"]
                if tuple(captured.shape) != (tokens, heads, 128):
                    raise ValueError("captured Q norm input shape does not match benchmark")
                source.zero_()
                source[:, :heads].copy_(captured)
                norm.weight.data.copy_(vectors["1.qnorm_weight"])
            dist.broadcast(source, src=0)
            dist.broadcast(norm.weight.data, src=0)
            q = source[:, :heads]
            functions = {
                "compiled": lambda: norm.rms_forward(q),
                "eager": lambda: RMSNorm.rms_forward.__wrapped__(norm, q),
                "deterministic": lambda: norm.rms_forward_deterministic(q),
            }
            modes = {}
            eager_output = functions["eager"]()
            for name, fn in functions.items():
                for _ in range(3):
                    fn()
                torch.cuda.synchronize()
                first = fn()
                second = fn()
                repeat_exact = torch.equal(first, second)
                reference = first.clone()
                dist.broadcast(reference, src=0)
                difference = (first.float() - reference.float()).abs()
                eager_difference = (first.float() - eager_output.float()).abs()

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = fn()
                graph.replay()
                torch.cuda.synchronize()
                graph_exact = torch.equal(captured, first)
                wall_samples = []
                gpu_samples = []
                for _ in range(runs):
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    start = perf_counter()
                    start_event.record()
                    for _ in range(replays):
                        graph.replay()
                    end_event.record()
                    torch.cuda.synchronize()
                    wall_samples.append((perf_counter() - start) * 1000 / replays)
                    gpu_samples.append(start_event.elapsed_time(end_event) / replays)
                modes[name] = {
                    "rank_different_elements": int(torch.count_nonzero(difference)),
                    "rank_max_abs": float(difference.max()),
                    "eager_different_elements": int(torch.count_nonzero(eager_difference)),
                    "eager_max_abs": float(eager_difference.max()),
                    "repeat_exact": repeat_exact,
                    "graph_exact": graph_exact,
                    "graph_host_ms": median(wall_samples),
                    "graph_gpu_ms": median(gpu_samples),
                }
            results.put({
                "rank": rank,
                "input_shape": list(q.shape),
                "input_stride": list(q.stride()),
                "modes": modes,
            })
            completed = True
    finally:
        if completed:
            dist.barrier()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--replays-per-run", type=int, default=100)
    parser.add_argument("--input-file")
    parser.add_argument("--result-file")
    args = parser.parse_args()
    if min(args.tokens, args.heads, args.runs, args.replays_per_run) < 1:
        parser.error("all dimensions and repeat counts must be positive")
    world_size = torch.cuda.device_count()
    if world_size < 2:
        parser.error("at least two visible GPUs are required")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    processes = [
        ctx.Process(target=worker, args=(
            rank, world_size, port, args.tokens, args.heads,
            args.runs, args.replays_per_run, args.input_file, queue,
        ))
        for rank in range(world_size)
    ]
    try:
        for process in processes:
            process.start()
        observations = sorted(
            (queue.get(timeout=180) for _ in processes),
            key=lambda item: item["rank"],
        )
        result = {
            "tokens": args.tokens,
            "heads": args.heads,
            "world_size": world_size,
            "cuda_graph": True,
            "runs": args.runs,
            "replays_per_run": args.replays_per_run,
            "input_file": args.input_file,
            "observations": observations,
        }
        if args.result_file:
            with open(args.result_file, "w") as file:
                json.dump(result, file, indent=2)
                file.write("\n")
        print(json.dumps(result))
    finally:
        for process in processes:
            process.join(timeout=60)
            if process.is_alive():
                process.terminate()
                process.join()
        if any(process.exitcode != 0 for process in processes):
            raise RuntimeError(
                f"Q/K norm rank benchmark worker exit codes: "
                f"{[process.exitcode for process in processes]}"
            )


if __name__ == "__main__":
    main()
