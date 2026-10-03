"""Runtime expert migration must preserve weights and captured graph pointers."""

import multiprocessing as mp
import socket
from types import SimpleNamespace

import pytest
import torch

from nanovllm.config import Config
from nanovllm.engine.llm_engine import LLMEngine


MODEL = "./models/Qwen3-30B-A3B-Base"


def test_dynamic_placement_supports_default_ep_equals_tp():
    with pytest.raises(ValueError, match="dynamic.*EP|expert parallel"):
        Config(MODEL, moe_dynamic_placement=True)
    legacy = Config(
        MODEL, enable_expert_parallel=True, tensor_parallel_size=2,
        moe_dynamic_placement=True, max_model_len=32,
        max_num_batched_tokens=16,
    )
    assert legacy.effective_expert_parallel_size == 2
    assert legacy.moe_dynamic_placement
    config = Config(
        MODEL, enable_expert_parallel=True, tensor_parallel_size=2,
        expert_parallel_size=2,
        moe_dynamic_placement=True, max_model_len=32,
        max_num_batched_tokens=16,
    )
    assert config.moe_dynamic_placement


def test_engine_rejects_migration_with_pending_requests():
    engine = object.__new__(LLMEngine)
    engine.config = SimpleNamespace(moe_dynamic_placement=True)
    engine.scheduler = SimpleNamespace(is_finished=lambda: False)
    with pytest.raises(RuntimeError, match="idle|pending"):
        engine.relocate_experts({})


def test_engine_validates_complete_placement_before_broadcast():
    engine = object.__new__(LLMEngine)
    engine.config = SimpleNamespace(
        moe_dynamic_placement=True,
        effective_expert_parallel_size=2,
        hf_config=SimpleNamespace(
            num_hidden_layers=2, mlp_only_layers=[], decoder_sparse_step=1,
            num_experts=4,
        ),
    )
    engine.scheduler = SimpleNamespace(is_finished=lambda: True)
    calls = []
    engine.model_runner = SimpleNamespace(
        call=lambda method, mapping: calls.append((method, mapping)) or "ok"
    )
    with pytest.raises(ValueError, match="every MoE layer"):
        engine.relocate_experts({"0": [0, 0, 1, 1]})
    with pytest.raises(ValueError, match="layer 1"):
        engine.relocate_experts({"0": [0, 0, 1, 1], "1": [0, 0, 0, 1]})
    assert calls == []
    assert engine.relocate_experts({
        "0": [0, 0, 1, 1], "1": [0, 1, 0, 1],
    }) == "ok"
    assert calls == [("relocate_moe_experts", {
        "0": (0, 0, 1, 1), "1": (0, 1, 0, 1),
    })]


def _worker(rank, port, result):
    import torch.distributed as dist

    from nanovllm.distributed.parallel_state import (
        destroy_model_parallel, initialize_model_parallel,
    )
    from nanovllm.layers.moe import ExpertParallelMoE

    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}",
        rank=rank, world_size=2,
    )
    initialize_model_parallel(2, 1, True)
    try:
        torch.manual_seed(49)
        full = ExpertParallelMoE(16, 32, 4, 2, True)
        with torch.no_grad():
            for parameter in full.parameters():
                parameter.normal_(0, 0.1)
        initial = (0, 0, 1, 1)
        target = (0, 1, 0, 1)
        shard = ExpertParallelMoE(
            16, 32, 4, 2, True, ep_rank=rank, ep_size=2,
            expert_owners=initial, dynamic_placement=True,
        )
        local_ids = [expert for expert, owner in enumerate(initial) if owner == rank]
        with torch.no_grad():
            shard.gate.weight.copy_(full.gate.weight)
            shard.gate_up_proj.copy_(full.gate_up_proj[local_ids])
            shard.down_proj.copy_(full.down_proj[local_ids])
        x = torch.randn(9, 16)
        expected = full(x)
        before = shard(x)
        pointers = (shard.gate_up_proj.data_ptr(), shard.down_proj.data_ptr(),
                    shard.expert_owner_tensor.data_ptr())

        with pytest.raises(ValueError, match="equal|placement"):
            shard.relocate_experts((0, 0, 0, 1))
        assert shard.local_expert_ids == tuple(local_ids)
        assert shard.relocate_experts(target) == 2
        assert shard.local_expert_ids == tuple(
            expert for expert, owner in enumerate(target) if owner == rank
        )
        assert pointers == (
            shard.gate_up_proj.data_ptr(), shard.down_proj.data_ptr(),
            shard.expert_owner_tensor.data_ptr(),
        )
        torch.testing.assert_close(shard(x), before, rtol=0, atol=0)
        torch.testing.assert_close(shard(x), expected, rtol=2e-5, atol=2e-6)
        assert shard.relocate_experts(target) == 0
        assert shard.relocate_experts(initial) == 2
        torch.testing.assert_close(shard(x), before, rtol=0, atol=0)
        result.put(rank)
    finally:
        destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.timeout(90)
def test_runtime_expert_migration_preserves_output_and_storage():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    context = mp.get_context("spawn")
    result = context.Queue()
    workers = [context.Process(target=_worker, args=(rank, port, result))
               for rank in range(2)]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=70)
        assert all(worker.exitcode == 0 for worker in workers)
        assert sorted(result.get_nowait() for _ in workers) == [0, 1]
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join()


def _disagree_worker(rank, port, result):
    import torch.distributed as dist

    from nanovllm.distributed.parallel_state import (
        destroy_model_parallel, initialize_model_parallel,
    )
    from nanovllm.layers.moe import ExpertParallelMoE

    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}",
        rank=rank, world_size=2,
    )
    initialize_model_parallel(2, 1, True)
    try:
        layer = ExpertParallelMoE(
            16, 32, 4, 2, True, ep_rank=rank, ep_size=2,
            dynamic_placement=True,
        )
        proposals = ((0, 0, 1, 1), (0, 1, 0, 1))
        with pytest.raises(ValueError, match="same expert placement"):
            layer.relocate_experts(proposals[rank])
        result.put(rank)
    finally:
        destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.timeout(30)
def test_mismatched_rank_proposals_fail_without_deadlock():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    context = mp.get_context("spawn")
    result = context.Queue()
    workers = [
        context.Process(target=_disagree_worker, args=(rank, port, result))
        for rank in range(2)
    ]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=15)
        assert all(worker.exitcode == 0 for worker in workers)
        assert sorted(result.get_nowait() for _ in workers) == [0, 1]
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join()
