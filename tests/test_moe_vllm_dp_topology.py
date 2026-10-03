"""Target topology for vLLM-style DP x TP expert sharding."""

import multiprocessing as mp
import socket

import pytest

from nanovllm.config import Config


MODEL = "./models/Qwen3-30B-A3B-Base"


def test_default_ep_reuses_tp_ranks():
    config = Config(
        MODEL, tensor_parallel_size=2, enable_expert_parallel=True,
        max_model_len=32, max_num_batched_tokens=16,
    )
    assert config.effective_expert_parallel_size == 2
    assert config.moe_dispatch_backend == "replicated"


def test_dp2_tp2_effective_ep_spans_all_four_ranks():
    config = Config(
        MODEL, data_parallel_size=2, tensor_parallel_size=2,
        enable_expert_parallel=True,
        master_port=29591, max_model_len=32, max_num_batched_tokens=16,
    )
    assert config.effective_expert_parallel_size == 4
    assert config.moe_dispatch_backend == "allgather_reduce"


def test_dp2_tp2_cannot_keep_an_independent_ep_group():
    config = Config(
        MODEL, data_parallel_size=2, tensor_parallel_size=2,
        enable_expert_parallel=True, max_model_len=32,
        max_num_batched_tokens=16, master_port=29591,
    )
    assert config.effective_expert_parallel_size == 4
    assert config.moe_dispatch_backend == "allgather_reduce"
    assert config.moe_global_dp


def test_global_dp_requires_shared_port():
    with pytest.raises(ValueError, match="master_port"):
        Config(
            MODEL, data_parallel_size=2, tensor_parallel_size=2,
            enable_expert_parallel=True,
            max_model_len=32, max_num_batched_tokens=16,
        )


def test_global_dp_accepts_reducescatter_with_prefill_graph():
    config = Config(
        MODEL, data_parallel_size=2, tensor_parallel_size=2,
        enable_expert_parallel=True,
        moe_dispatch_backend="allgather_reducescatter",
        moe_prefill_piece=True, moe_prefill_piece_capture_sizes=(32,),
        master_port=29591, max_model_len=32, max_num_batched_tokens=32,
    )
    assert config.moe_dispatch_backend == "allgather_reducescatter"
    assert config.moe_prefill_piece_capture_sizes == (32,)


def test_reducescatter_requires_global_dp():
    with pytest.raises(ValueError, match="requires global DP MoE"):
        Config(
            MODEL, tensor_parallel_size=2, enable_expert_parallel=True,
            moe_dispatch_backend="allgather_reducescatter",
            max_model_len=32, max_num_batched_tokens=32,
        )


def _group_worker(rank, port, output):
    import torch.distributed as dist

    from nanovllm.distributed.parallel_state import (
        destroy_model_parallel, get_moe_group_ranks, get_tp_group_ranks,
        initialize_model_parallel,
    )

    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}",
        rank=rank, world_size=4,
    )
    try:
        initialize_model_parallel(
            2, 1, True, data_parallel_size=2,
        )
        output.put((rank, get_tp_group_ranks(), get_moe_group_ranks()))
    finally:
        destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.timeout(90)
def test_dp2_tp2_has_local_tp_groups_and_global_ep_group():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    context = mp.get_context("spawn")
    output = context.Queue()
    workers = [context.Process(target=_group_worker, args=(rank, port, output))
               for rank in range(4)]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=60)
        assert all(worker.exitcode == 0 for worker in workers)
        groups = sorted(output.get_nowait() for _ in workers)
        for rank, tp_ranks, moe_ranks in groups:
            assert tp_ranks == ((0, 1) if rank < 2 else (2, 3))
            assert moe_ranks == (0, 1, 2, 3)
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join()


def _global_moe_worker(rank, port, output, backend):
    import torch
    import torch.distributed as dist

    from nanovllm.distributed.parallel_state import (
        destroy_model_parallel, get_replica_group, initialize_model_parallel,
    )
    from nanovllm.layers.moe import ExpertParallelMoE

    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}",
        rank=rank, world_size=4,
    )
    try:
        initialize_model_parallel(
            2, 1, True, data_parallel_size=2,
        )
        assert dist.get_world_size(get_replica_group()) == 2
        torch.manual_seed(13)
        full = ExpertParallelMoE(16, 32, 4, 2, True)
        with torch.no_grad():
            for parameter in full.parameters():
                parameter.normal_(0, 0.1)
        shard = ExpertParallelMoE(
            16, 32, 4, 2, True, rank, 4,
            dispatch_backend=backend,
        )
        with torch.no_grad():
            shard.gate.weight.copy_(full.gate.weight)
            shard.gate_up_proj.copy_(full.gate_up_proj[rank:rank + 1])
            shard.down_proj.copy_(full.down_proj[rank:rank + 1])
        with torch.inference_mode():
            for seed in (21, 22):
                torch.manual_seed(seed + rank // 2)
                hidden = torch.randn(3, 16)
                torch.testing.assert_close(
                    shard(hidden), full(hidden), rtol=2e-5, atol=2e-6,
                )
        if backend == "allgather_reducescatter":
            assert shard.ep_all_gather_count == 4
            assert shard.ep_reduce_scatter_count == 2
            assert shard.ep_all_reduce_count == 0
        else:
            assert shard.ep_all_gather_count == 2
            assert shard.ep_all_reduce_count == 2
        full = full.bfloat16()
        shard = shard.bfloat16()
        with torch.inference_mode():
            torch.testing.assert_close(
                shard(hidden.bfloat16()), full(hidden.bfloat16()),
                rtol=3e-2, atol=3e-3,
            )
        output.put(rank)
    finally:
        destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.timeout(90)
@pytest.mark.parametrize("backend", ["allgather_reduce", "allgather_reducescatter"])
def test_global_dp_moe_returns_each_replicas_own_tokens(backend):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    context = mp.get_context("spawn")
    output = context.Queue()
    workers = [
        context.Process(target=_global_moe_worker, args=(rank, port, output, backend))
        for rank in range(4)
    ]
    try:
        for worker in workers:
            worker.start()
        assert sorted(output.get(timeout=60) for _ in workers) == list(range(4))
    finally:
        for worker in workers:
            worker.join(timeout=10)
            if worker.is_alive():
                worker.terminate()
                worker.join()
        assert all(worker.exitcode == 0 for worker in workers)
