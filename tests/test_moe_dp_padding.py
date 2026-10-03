"""Unequal DP token counts, transport padding, capacity, and graph fallback."""

import argparse
import json
import multiprocessing as mp
import os
import socket

import pytest


def test_dummy_attention_does_not_modify_kv_cache():
    import torch
    from nanovllm.layers.attention import Attention
    from nanovllm.utils.context import get_context, reset_context

    layer = Attention(2, 4, 0.5, 2)
    layer.k_cache = torch.ones(1, 2, 4)
    layer.v_cache = torch.ones_like(layer.k_cache)
    reset_context()
    get_context().is_dummy = True
    try:
        value = torch.randn(1, 2, 4)
        torch.testing.assert_close(layer(value, value, value), torch.zeros_like(value))
        assert torch.all(layer.k_cache == 1) and torch.all(layer.v_cache == 1)
    finally:
        reset_context()


def test_idle_runner_executes_without_autograd(monkeypatch):
    import torch
    from types import SimpleNamespace
    from nanovllm.engine.model_runner import ModelRunner
    from nanovllm.utils.context import get_context

    zeros = torch.zeros
    monkeypatch.setattr(torch, "zeros", lambda *args, **kwargs:
                        zeros(*args, **dict(kwargs, device="cpu")))
    runner = ModelRunner.__new__(ModelRunner)
    runner.config = SimpleNamespace(moe_global_dp=True)
    runner.coordinate_ep_batch = lambda *args: None
    calls = []

    def model(input_ids, positions):
        assert torch.is_inference_mode_enabled()
        assert get_context().is_dummy
        calls.append(input_ids.numel())

    runner.model = model
    assert runner.run([], True) == []
    assert calls == [1]
    assert not get_context().is_dummy


def worker(rank, port, tp, backend, output=None, cuda=False):
    import torch
    import torch.distributed as dist

    from nanovllm.distributed.parallel_state import (
        destroy_model_parallel, initialize_model_parallel,
    )
    from nanovllm.engine.model_runner import ModelRunner
    from nanovllm.layers.moe import ExpertParallelMoE
    from nanovllm.utils.context import get_context, reset_context

    if cuda:
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        dist.init_process_group("nccl", device_id=device)
    else:
        device = torch.device("cpu")
        dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                                rank=rank, world_size=4)
    dp = 4 // tp
    initialize_model_parallel(tp, enable_expert_parallel=True, data_parallel_size=dp)
    runner = ModelRunner.__new__(ModelRunner)
    runner.ep_size = 4
    runner.prefill_piece_graphs = {}
    layouts = [(3,) * dp, tuple(range(1, dp + 1)),
               tuple(reversed(range(1, dp + 1))),
               (0,) + (5,) * (dp - 1), (5,) + (0,) * (dp - 1), (0,) * dp]
    try:
        with torch.inference_mode():
            for capacity in (None, 2):
                torch.manual_seed(17)
                full = ExpertParallelMoE(16, 32, 4, 2, True, expert_capacity=capacity)
                for parameter in full.parameters():
                    parameter.normal_(0, 0.1)
                shard = ExpertParallelMoE(
                    16, 32, 4, 2, True, rank, 4, dispatch_backend=backend,
                    expert_capacity=capacity, graph_safe_decode=cuda,
                )
                shard.gate.weight.copy_(full.gate.weight)
                shard.gate_up_proj.copy_(full.gate_up_proj[rank:rank + 1])
                shard.down_proj.copy_(full.down_proj[rank:rank + 1])
                for dtype in (torch.float32, torch.bfloat16):
                    full = full.to(device=device, dtype=dtype)
                    shard = shard.to(device=device, dtype=dtype)
                    for counts in layouts:
                        torch.manual_seed(31)
                        inputs = [torch.randn(count, 16, device=device, dtype=dtype)
                                  for count in counts]
                        dp_rank = rank // tp
                        hidden = inputs[dp_rank]
                        # An idle engine uses a one-row placeholder without touching KV.
                        if counts[dp_rank] == 0:
                            hidden = torch.zeros(1, 16, device=device, dtype=dtype)
                        reset_context()
                        get_context().is_dummy = counts[dp_rank] == 0
                        runner.coordinate_ep_batch(counts[dp_rank], True, device)
                        uneven = len(set(counts)) > 1 or 0 in counts
                        assert get_context().force_eager == uneven
                        assert get_context().moe_token_counts == (counts if uneven else None)
                        expected = full(torch.cat(inputs)).split(counts)[dp_rank]
                        actual = shard(hidden)
                        if counts[dp_rank] == 0:
                            torch.testing.assert_close(actual, torch.zeros_like(hidden))
                        else:
                            tolerance = dict(rtol=3e-2, atol=3e-3) if dtype == torch.bfloat16 else dict(rtol=2e-5, atol=2e-6)
                            torch.testing.assert_close(actual, expected, **tolerance)
                        assert actual.shape == hidden.shape

            reset_context()
            runner.prefill_piece_graphs = {3: {None: object()}} if rank // tp == 0 else {}
            runner.coordinate_ep_batch(3, True, device)
            assert get_context().force_eager
            runner.prefill_piece_graphs = {}
            runner.coordinate_ep_batch(3, rank // tp == 0, device)
            assert get_context().force_eager
            reset_context()
            assert runner.has_unfinished_dp(rank == 0)
            assert not runner.has_unfinished_dp(False)
        if output is not None:
            output.put(rank)
        elif rank == 0:
            print(json.dumps({"tp": tp, "ep": 4, "backend": backend,
                              "padding_capacity_and_idle": "passed"}), flush=True)
    finally:
        reset_context()
        destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.timeout(150)
@pytest.mark.parametrize("tp", [1, 2])
@pytest.mark.parametrize("backend", ["allgather_reduce", "allgather_reducescatter"])
def test_unequal_dp_batches_are_padded_and_trimmed(tp, backend):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    context = mp.get_context("spawn")
    output = context.Queue()
    workers = [context.Process(target=worker, args=(rank, port, tp, backend, output))
               for rank in range(4)]
    try:
        for process in workers:
            process.start()
        assert sorted(output.get(timeout=110) for _ in workers) == list(range(4))
    finally:
        for process in workers:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join()
        assert all(process.exitcode == 0 for process in workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, choices=(1, 2), default=2)
    parser.add_argument("--backend", choices=("allgather_reduce", "allgather_reducescatter"),
                        default="allgather_reduce")
    args = parser.parse_args()
    worker(int(os.environ["LOCAL_RANK"]), None, args.tp, args.backend, cuda=True)
