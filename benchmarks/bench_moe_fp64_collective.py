"""Compare full FP64 all-reduce with FP64 reduce-scatter/BF16 all-gather."""

import argparse
import json
import socket
from statistics import median
from time import perf_counter

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, port, tokens, hidden_size, repeats, results):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}",
        rank=rank, world_size=4, device_id=torch.device("cuda", rank),
    )
    try:
        torch.manual_seed(100 + rank)
        numel = tokens * hidden_size
        if numel % 4:
            raise ValueError("tensor elements must divide across four ranks")
        partial = torch.randn(
            numel, device="cuda", dtype=torch.bfloat16
        ).double() * 0.015625
        work = torch.empty_like(partial)
        reduced = torch.empty(numel // 4, device="cuda", dtype=torch.float64)
        gathered = torch.empty(numel, device="cuda", dtype=torch.bfloat16)

        def full():
            dist.all_reduce(work)
            return work.bfloat16()

        def staged():
            dist.reduce_scatter_tensor(reduced, partial)
            dist.all_gather_into_tensor(gathered, reduced.bfloat16())
            return gathered

        for _ in range(3):
            work.copy_(partial)
            full()
            staged()
        work.copy_(partial)
        expected = full()
        actual = staged()
        difference = (expected.float() - actual.float()).abs()
        observations = {}
        for name, run in (("all_reduce_fp64", full),
                          ("reduce_scatter_fp64_gather_bf16", staged)):
            samples = []
            for _ in range(repeats):
                if name == "all_reduce_fp64":
                    work.copy_(partial)
                torch.cuda.synchronize()
                start = perf_counter()
                run()
                torch.cuda.synchronize()
                samples.append(1000 * (perf_counter() - start))
            observations[name] = {
                "median_ms": median(samples), "samples_ms": samples,
            }
        results.put({
            "rank": rank,
            "tokens": tokens,
            "hidden_size": hidden_size,
            "different_elements": int(torch.count_nonzero(difference)),
            "max_abs": float(difference.max()),
            "timings": observations,
        })
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=16384)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--result-file")
    args = parser.parse_args()
    if args.tokens < 1 or args.hidden_size < 1 or args.repeats < 1:
        parser.error("tokens, hidden size and repeats must be positive")
    if torch.cuda.device_count() != 4:
        parser.error("set CUDA_VISIBLE_DEVICES to exactly four GPUs")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    ctx = mp.get_context("spawn")
    results = ctx.Queue()
    processes = [
        ctx.Process(target=worker, args=(rank, port, args.tokens,
                                        args.hidden_size, args.repeats, results))
        for rank in range(4)
    ]
    try:
        for process in processes:
            process.start()
        observations = sorted(
            (results.get(timeout=180) for _ in processes),
            key=lambda item: item["rank"],
        )
        if args.result_file:
            with open(args.result_file, "w") as file:
                json.dump(observations, file, indent=2)
                file.write("\n")
        print(json.dumps(observations))
        if any(item["different_elements"] for item in observations):
            raise AssertionError("staged reduction differs from FP64 all-reduce")
    finally:
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join()
        if any(process.exitcode != 0 for process in processes):
            raise RuntimeError("collective benchmark worker failed")


if __name__ == "__main__":
    main()
