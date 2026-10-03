"""Contracts for per-layer offline expert placement across EP ranks."""

import json
import multiprocessing as mp
import socket

import pytest
import torch

from nanovllm.config import Config


MODEL = "./models/Qwen3-30B-A3B-Base"


def placement_file(tmp_path, ep_size=4):
    path = tmp_path / "placement.json"
    path.write_text(json.dumps({
        "model": MODEL,
        "num_experts": 128,
        "expert_parallel_size": ep_size,
        "layers": {
            str(layer): [expert % 4 for expert in range(128)]
            for layer in range(48)
        },
    }))
    return path


def test_config_loads_noncontiguous_placement(tmp_path):
    path = placement_file(tmp_path)
    config = Config(
        MODEL,
        enable_expert_parallel=True,
        tensor_parallel_size=4,
        expert_parallel_size=4,
        moe_expert_placement=str(path),
        enforce_eager=True,
        max_model_len=32,
        max_num_batched_tokens=16,
    )
    assert config.moe_placement_by_layer["0"][1] == 1
    assert len(config.moe_placement_by_layer) == 48


def test_config_rejects_wrong_ep_size(tmp_path):
    path = placement_file(tmp_path, ep_size=2)
    with pytest.raises(ValueError, match="placement|EP"):
        Config(
            MODEL,
            enable_expert_parallel=True,
            tensor_parallel_size=4,
            expert_parallel_size=4,
            moe_expert_placement=str(path),
            enforce_eager=True,
            max_model_len=32,
            max_num_batched_tokens=16,
        )


def test_config_rejects_missing_model_path(tmp_path):
    path = placement_file(tmp_path)
    placement = json.loads(path.read_text())
    del placement["model"]
    path.write_text(json.dumps(placement))
    with pytest.raises(ValueError, match="placement model"):
        Config(
            MODEL,
            enable_expert_parallel=True,
            tensor_parallel_size=4,
            expert_parallel_size=4,
            moe_expert_placement=str(path),
            enforce_eager=True,
            max_model_len=32,
            max_num_batched_tokens=16,
        )


def test_noncontiguous_route_mapping_handles_padding():
    import torch

    from nanovllm.layers.moe import ExpertParallelMoE

    layer = ExpertParallelMoE(
        16, 32, 4, 2, True,
        ep_rank=0, ep_size=2, expert_owners=[0, 1, 1, 0],
    )
    mask, local_ids = layer._local_routes(torch.tensor([-1, 0, 1, 2, 3, 4]))
    assert mask.tolist() == [False, True, False, False, True, False]
    assert local_ids[mask].tolist() == [0, 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_route_mapping_replays_changed_input():
    from nanovllm.layers.moe import ExpertParallelMoE

    layer = ExpertParallelMoE(
        16, 32, 4, 2, True,
        ep_rank=0, ep_size=2, expert_owners=[0, 1, 1, 0],
    ).cuda()
    routes = torch.tensor([-1, 0, 1, 2, 3, 4], device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mask, local_ids = layer._local_routes(routes)
    for values, expected_mask, expected_ids in (
        ([-1, 0, 1, 2, 3, 4], [False, True, False, False, True, False], [0, 1]),
        ([4, 1, 3, 0, 2, -1], [False, False, True, True, False, False], [1, 0]),
    ):
        routes.copy_(torch.tensor(values, device="cuda"))
        graph.replay()
        torch.cuda.synchronize()
        assert mask.tolist() == expected_mask
        assert local_ids[mask].tolist() == expected_ids


def placement_worker(rank, port, backend, output):
    import torch
    import torch.distributed as dist

    from nanovllm.distributed.parallel_state import (
        destroy_model_parallel, initialize_model_parallel,
    )
    from nanovllm.layers.moe import ExpertParallelMoE

    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}",
        world_size=2, rank=rank,
    )
    initialize_model_parallel(2, 1, True)
    try:
        torch.manual_seed(19)
        full = ExpertParallelMoE(16, 32, 4, 2, True)
        with torch.no_grad():
            for parameter in full.parameters():
                parameter.normal_(0, 0.1)
        owners = [0, 1, 1, 0]
        local_ids = [expert for expert, owner in enumerate(owners) if owner == rank]
        shard = ExpertParallelMoE(
            16, 32, 4, 2, True,
            ep_rank=rank, ep_size=2,
            dispatch_backend=backend,
            expert_owners=owners,
        )
        assert shard.local_expert_ids == tuple(local_ids)
        with torch.no_grad():
            shard.gate.weight.copy_(full.gate.weight)
            shard.gate_up_proj.copy_(full.gate_up_proj[local_ids])
            shard.down_proj.copy_(full.down_proj[local_ids])
        hidden = torch.randn(9, 16)
        for _ in range(2):
            torch.testing.assert_close(
                shard(hidden), full(hidden), rtol=2e-5, atol=2e-6
            )
        output.put(rank)
    finally:
        destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.parametrize("backend", ["replicated", "all_to_all", "all_to_all_reduce"])
@pytest.mark.timeout(90)
def test_noncontiguous_placement_matches_unsharded_moe(backend):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    context = mp.get_context("spawn")
    output = context.Queue()
    processes = [
        context.Process(target=placement_worker, args=(rank, port, backend, output))
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=60)
        assert all(process.exitcode == 0 for process in processes)
        assert sorted(output.get_nowait() for _ in processes) == [0, 1]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
