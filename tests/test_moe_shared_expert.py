"""Shared MoE experts contribute once per token across EP ranks."""

import multiprocessing as mp
import socket

import pytest
import torch
import torch.nn.functional as F

from nanovllm.layers.moe import ExpertParallelMoE


def _make_module(ep_rank=0, ep_size=1, backend="replicated"):
    module = ExpertParallelMoE(
        16, 32, 4, 2, True, ep_rank=ep_rank, ep_size=ep_size,
        dispatch_backend=backend, shared_expert_intermediate_size=8,
    )
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(0, 0.1)
    return module


def _shared_reference(module, hidden):
    gate, up = F.linear(hidden, module.shared_expert.gate_up_proj.weight).chunk(2, -1)
    activated = F.silu(gate) * up
    projected = F.linear(activated, module.shared_expert.down_proj.weight)
    return torch.sigmoid(F.linear(hidden, module.shared_expert_gate.weight)) * projected


def test_shared_expert_matches_independent_formula():
    torch.manual_seed(17)
    module = _make_module()
    hidden = torch.randn(5, 16)
    routed = ExpertParallelMoE(16, 32, 4, 2, True)
    with torch.no_grad():
        routed.gate.weight.copy_(module.gate.weight)
        routed.gate_up_proj.copy_(module.gate_up_proj)
        routed.down_proj.copy_(module.down_proj)
    expected = routed(hidden) + _shared_reference(module, hidden)
    torch.testing.assert_close(module(hidden), expected, rtol=2e-5, atol=2e-6)


def test_shared_expert_checkpoint_mapping_loads_replicated_weights(tmp_path):
    from safetensors.torch import save_file

    from nanovllm.layers.linear import ReplicatedLinear
    from nanovllm.layers.moe import SharedExpertMLP
    from nanovllm.models.qwen3_moe import Qwen3MoeForCausalLM
    from nanovllm.utils.loader import load_model

    model = torch.nn.Module()
    model.packed_modules_mapping = Qwen3MoeForCausalLM.packed_modules_mapping
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleDict({"0": torch.nn.Module()})
    model.model.layers["0"].mlp = torch.nn.Module()
    mlp = model.model.layers["0"].mlp
    mlp.shared_expert = SharedExpertMLP(16, 8)
    mlp.shared_expert_gate = ReplicatedLinear(16, 1)
    prefix = "model.layers.0.mlp."
    weights = {
        prefix + "shared_expert.gate_proj.weight": torch.randn(8, 16),
        prefix + "shared_expert.up_proj.weight": torch.randn(8, 16),
        prefix + "shared_expert.down_proj.weight": torch.randn(16, 8),
        prefix + "shared_expert_gate.weight": torch.randn(1, 16),
    }
    save_file(weights, tmp_path / "weights.safetensors")
    load_model(model, str(tmp_path))
    torch.testing.assert_close(
        mlp.shared_expert.gate_up_proj.weight,
        torch.cat((
            weights[prefix + "shared_expert.gate_proj.weight"],
            weights[prefix + "shared_expert.up_proj.weight"],
        )),
        rtol=0, atol=0,
    )
    torch.testing.assert_close(
        mlp.shared_expert.down_proj.weight,
        weights[prefix + "shared_expert.down_proj.weight"],
        rtol=0, atol=0,
    )
    torch.testing.assert_close(
        mlp.shared_expert_gate.weight,
        weights[prefix + "shared_expert_gate.weight"],
        rtol=0, atol=0,
    )


def _distributed_worker(rank, port, backend, queue):
    import torch.distributed as dist

    from nanovllm.distributed.parallel_state import (
        destroy_model_parallel, initialize_model_parallel,
    )

    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
        world_size=2,
    )
    initialize_model_parallel(2, 1, True)
    try:
        torch.manual_seed(29)
        full = _make_module()
        shard = _make_module(rank, 2, backend)
        with torch.no_grad():
            shard.gate.weight.copy_(full.gate.weight)
            shard.gate_up_proj.copy_(full.gate_up_proj[rank * 2:(rank + 1) * 2])
            shard.down_proj.copy_(full.down_proj[rank * 2:(rank + 1) * 2])
            shard.shared_expert.load_state_dict(full.shared_expert.state_dict())
            shard.shared_expert_gate.weight.copy_(full.shared_expert_gate.weight)
        hidden = torch.randn(7, 16)
        expected = full(hidden)
        actual = shard(hidden)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        queue.put(rank)
    finally:
        destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.timeout(90)
@pytest.mark.parametrize("backend", ["replicated", "all_to_all", "all_to_all_reduce"])
def test_shared_expert_is_added_once_across_ep(backend):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [context.Process(target=_distributed_worker, args=(
        rank, port, backend, queue,
    )) for rank in range(2)]
    try:
        for process in processes:
            process.start()
        assert sorted(queue.get(timeout=60) for _ in processes) == [0, 1]
    finally:
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join()
        assert all(process.exitcode == 0 for process in processes)


@pytest.mark.timeout(90)
@torch.inference_mode()
def test_shared_expert_graph_replays_changed_inputs():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    torch.manual_seed(41)
    module = _make_module().cuda().bfloat16()
    first = torch.randn(16, 16, device="cuda", dtype=torch.bfloat16)
    second = torch.randn_like(first)
    module.graph_safe_decode = False
    expected_first = module(first)
    expected_second = module(second)
    module.graph_safe_decode = True
    static_input = first.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        module(static_input)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = module(static_input)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected_first, rtol=1e-2, atol=1e-3)
    static_input.copy_(second)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected_second, rtol=1e-2, atol=1e-3)
    assert not torch.equal(expected_first, expected_second)


def _tp_ep_worker(rank, port, queue):
    import torch.distributed as dist

    from nanovllm.distributed.parallel_state import (
        destroy_model_parallel, initialize_model_parallel,
    )

    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
        world_size=4,
    )
    torch.manual_seed(53)
    full = _make_module()
    initialize_model_parallel(2, 1, True, data_parallel_size=2)
    try:
        ep_rank, tp_rank = rank, rank % 2
        shard = ExpertParallelMoE(
            16, 32, 4, 2, True, ep_rank=ep_rank, ep_size=4,
            dispatch_backend="allgather_reduce",
            shared_expert_intermediate_size=8,
            shared_expert_tp_sharded=True,
        )
        width = shard.shared_expert.gate_up_proj.weight.size(0) // 2
        with torch.no_grad():
            shard.gate.weight.copy_(full.gate.weight)
            shard.gate_up_proj.copy_(
                full.gate_up_proj[ep_rank:ep_rank + 1]
            )
            shard.down_proj.copy_(
                full.down_proj[ep_rank:ep_rank + 1]
            )
            for part in range(2):
                shard.shared_expert.gate_up_proj.weight[
                    part * width:(part + 1) * width
                ].copy_(full.shared_expert.gate_up_proj.weight[
                    part * 8 + tp_rank * width:part * 8 + (tp_rank + 1) * width
                ])
            shard.shared_expert.down_proj.weight.copy_(
                full.shared_expert.down_proj.weight[
                    :, tp_rank * width:(tp_rank + 1) * width
                ]
            )
            shard.shared_expert_gate.weight.copy_(full.shared_expert_gate.weight)
        torch.manual_seed(71 + rank // 2)
        hidden = torch.randn(6, 16)
        with torch.inference_mode():
            torch.testing.assert_close(
                shard(hidden), full(hidden), rtol=2e-5, atol=2e-6,
            )
        queue.put(rank)
    finally:
        destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.timeout(90)
def test_shared_expert_dp2_tp2_ep4_matches_each_replicas_reference():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [context.Process(target=_tp_ep_worker, args=(rank, port, queue))
                 for rank in range(4)]
    try:
        for process in processes:
            process.start()
        assert sorted(queue.get(timeout=60) for _ in processes) == [0, 1, 2, 3]
    finally:
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join()
        assert all(process.exitcode == 0 for process in processes)
