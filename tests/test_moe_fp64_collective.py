"""Four-rank exactness and CUDA Graph replay for staged FP64 EP reduction."""

import json
import socket
import gc

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nanovllm.layers.moe import reduce_fp64_to_bf16


def worker(rank, port, results):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}",
        rank=rank, world_size=4, device_id=torch.device("cuda", rank),
    )
    try:
        torch.manual_seed(100 + rank)
        local = torch.randn(
            32, 128, device="cuda", dtype=torch.bfloat16
        ).double() * 0.015625

        def reference():
            expected = local.clone()
            dist.all_reduce(expected)
            return expected.bfloat16()

        expected = reference()
        actual = reduce_fp64_to_bf16(local, dist.group.WORLD)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            reduce_fp64_to_bf16(local, dist.group.WORLD)
        torch.cuda.current_stream().wait_stream(stream)
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = reduce_fp64_to_bf16(local, dist.group.WORLD)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)

        local.copy_(torch.randn_like(local).bfloat16().double() * 0.015625)
        expected = reference()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)
        del graph, captured
        gc.collect()
        results.put({"rank": rank, "exact": True, "graph_replays": 2})
        dist.barrier()
    finally:
        dist.destroy_process_group()


def main():
    if torch.cuda.device_count() != 4:
        raise RuntimeError("set CUDA_VISIBLE_DEVICES to exactly four GPUs")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    ctx = mp.get_context("spawn")
    results = ctx.Queue()
    processes = [
        ctx.Process(target=worker, args=(rank, port, results))
        for rank in range(4)
    ]
    try:
        for process in processes:
            process.start()
        observations = sorted(
            (results.get(timeout=120) for _ in processes),
            key=lambda item: item["rank"],
        )
        assert all(item["exact"] and item["graph_replays"] == 2
                   for item in observations)
        print(json.dumps(observations))
    finally:
        for process in processes:
            process.join(timeout=60)
            if process.is_alive():
                process.terminate()
                process.join()
        assert all(process.exitcode == 0 for process in processes)


if __name__ == "__main__":
    main()
