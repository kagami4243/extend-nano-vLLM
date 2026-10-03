"""Two-rank numerical contract for token all-to-all MoE dispatch."""

import multiprocessing as mp
import socket

import pytest


def worker(rank, port, output, backend, capacity, capacity_factor):
    import torch
    import torch.distributed as dist

    from nanovllm.distributed.parallel_state import (
        destroy_model_parallel,
        initialize_model_parallel,
    )
    from nanovllm.layers.moe import ExpertParallelMoE

    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", world_size=2, rank=rank,
    )
    initialize_model_parallel(2, 1, True)
    try:
        torch.manual_seed(13)
        full = ExpertParallelMoE(
            16, 32, 4, 2, True, expert_capacity=capacity,
            expert_capacity_factor=capacity_factor,
        )
        with torch.no_grad():
            for parameter in full.parameters():
                parameter.normal_(0, 0.1)
        shard = ExpertParallelMoE(
            16, 32, 4, 2, True, rank, 2, dispatch_backend=backend,
            expert_capacity=capacity, expert_capacity_factor=capacity_factor,
        )
        with torch.no_grad():
            shard.gate.weight.copy_(full.gate.weight)
            shard.gate_up_proj.copy_(full.gate_up_proj[rank * 2:(rank + 1) * 2])
            shard.down_proj.copy_(full.down_proj[rank * 2:(rank + 1) * 2])
        hidden = torch.randn(9, 16)
        _, selected = full._route(hidden)
        expected_token_packets = sum(
            len({int(expert // 2) for expert in expert_ids if expert < 4})
            for expert_ids in selected
        )
        for _ in range(2):
            reference = full(hidden)
            actual = shard(hidden)
            torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-6)
        assert shard.ep_all_reduce_count == (2 if backend == "all_to_all_reduce" else 0)
        assert shard.ep_all_to_all_count == (4 if backend == "all_to_all_reduce" else 6)
        assert shard.dispatch_assignment_count == shard.return_assignment_count
        result = (rank, shard.dispatch_assignment_count,
                  shard.total_assignment_count, shard.ep_all_to_all_count,
                  getattr(shard, "total_dispatch_token_count", None),
                  2 * expected_token_packets, shard.dropped_assignment_count)
        full = full.bfloat16()
        shard = shard.bfloat16()
        bf_hidden = hidden.bfloat16()
        torch.testing.assert_close(
            shard(bf_hidden), full(bf_hidden), rtol=3e-2, atol=3e-3
        )
        large_hidden = bf_hidden.repeat(8, 1)
        torch.testing.assert_close(
            shard(large_hidden), full(large_hidden), rtol=3e-2, atol=3e-3
        )
        assert shard.ep_all_to_all_count == (9 if backend == "all_to_all_reduce" else 13)
        assert shard.ep_all_reduce_count == (4 if backend == "all_to_all_reduce" else 0)
        single_hidden = torch.ones(1, 16, dtype=torch.bfloat16)
        for first, second in ((0, 2), (0, 1), (2, 3)):
            with torch.no_grad():
                gate = torch.full_like(full.gate.weight, -0.1)
                gate[first].fill_(0.1)
                gate[second].fill_(0.05)
                full.gate.weight.copy_(gate)
                shard.gate.weight.copy_(gate)
            torch.testing.assert_close(
                shard(single_hidden), full(single_hidden), rtol=3e-2, atol=3e-3
            )
        assert shard.ep_all_to_all_count == (15 if backend == "all_to_all_reduce" else 22)
        output.put(result)
    finally:
        destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.timeout(90)
@pytest.mark.parametrize("backend", ["all_to_all", "all_to_all_reduce"])
@pytest.mark.parametrize("capacity,capacity_factor", [
    (None, None), (1, None), (None, 1.25),
])
def test_all_to_all_matches_unsharded_moe(backend, capacity, capacity_factor):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    context = mp.get_context("spawn")
    output = context.Queue()
    processes = [
        context.Process(target=worker, args=(
            rank, port, output, backend, capacity, capacity_factor,
        ))
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        results = sorted(output.get(timeout=60) for _ in processes)
        assert sum(item[1] for item in results) == results[0][2] - results[0][6]
        assert results[1][2] == 0
        if backend == "all_to_all_reduce":
            assert results[0][4] == results[0][5]
            assert results[0][4] < results[0][2]
    finally:
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join()
        assert all(process.exitcode == 0 for process in processes)
