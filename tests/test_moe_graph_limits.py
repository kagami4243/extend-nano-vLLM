"""Per-rank Graph limits, multiple buckets, and gathered DP MoE captures."""

import argparse
import json
import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nanovllm.layers import moe
from nanovllm.layers.moe import ExpertParallelMoE, MOE_MAX_GRAPH_TOKENS_PER_RANK


@pytest.mark.parametrize("backend", ["replicated", "allgather_reduce", "allgather_reducescatter"])
@pytest.mark.parametrize("tp_size,ep_size", [(1, 1), (1, 4), (2, 4), (4, 4)])
def test_static_graph_limit_is_per_dp_replica(monkeypatch, backend, tp_size, ep_size):
    monkeypatch.setattr(moe, "get_tp_world_size", lambda: tp_size)
    layer = ExpertParallelMoE(16, 32, 8, 2, True, ep_size=ep_size,
                             dispatch_backend=backend, graph_safe_decode=True)
    dp_size = ep_size // tp_size if backend != "replicated" else 1
    limit = dp_size * MOE_MAX_GRAPH_TOKENS_PER_RANK
    assert layer.graph_token_limit == limit
    monkeypatch.setattr(layer, "_can_use_triton_kernel", lambda _: True)
    monkeypatch.setattr(layer, "_forward_static_decode", lambda *args: ("static", 0))
    monkeypatch.setattr(layer, "_forward_triton", lambda *args: ("dynamic", 0))
    assert layer._execute_local(torch.empty(limit, 16), None, None)[0] == "static"
    assert layer._execute_local(torch.empty(limit + 1, 16), None, None)[0] == "dynamic"


class SmallMoEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(256, 32)
        self.model = ExpertParallelMoE(32, 64, 8, 2, True, graph_safe_decode=True)
        for parameter in self.parameters():
            parameter.data.normal_(0, 0.02)

    def forward(self, input_ids, positions):
        hidden = self.embed(input_ids)
        return self.model(hidden + positions.to(hidden.dtype)[:, None] * 0.001)

    def compute_logits(self, hidden):
        return hidden


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_runner_captures_multiple_moe_graphs_through_512():
    from nanovllm.engine.model_runner import ModelRunner
    from nanovllm.utils.context import reset_context, set_context

    device = torch.device("cuda", torch.cuda.current_device())
    previous_device = torch.get_default_device()
    previous_dtype = torch.get_default_dtype()
    torch.set_default_device(device)
    torch.set_default_dtype(torch.bfloat16)
    try:
        torch.manual_seed(19)
        runner = ModelRunner.__new__(ModelRunner)
        runner.config = SimpleNamespace(
            max_num_seqs=513, max_model_len=256,
            hf_config=SimpleNamespace(model_type="qwen3_moe", hidden_size=32),
            moe_prefill_piece=False, quantization=None,
        )
        runner.block_size = 256
        runner.pp_size = 1
        runner.enforce_eager = False
        runner.eagle3_model = None
        runner.decode_graph_replay_count = 0
        runner.model = SmallMoEModel().to(device=device, dtype=torch.bfloat16)
        runner.capture_cudagraph()
        assert {1, 2, 4, 8, 16, 32, 256, 512}.issubset(runner.graphs)
        assert max(runner.graphs) == 512
        assert len({id(graph) for graph in runner.graphs.values()}) == len(runner.graphs)

        for batch_size in (1, 17, 33, 129, 512, 513):
            ids = torch.randint(0, 256, (batch_size,), device=device)
            positions = torch.arange(batch_size, device=device)
            set_context(False, slot_mapping=torch.full((batch_size,), -1, dtype=torch.int32),
                        context_lens=torch.ones(batch_size, dtype=torch.int32),
                        block_tables=torch.zeros(batch_size, 1, dtype=torch.int32))
            before = runner.decode_graph_replay_count
            runner.model.model.graph_safe_decode = False
            expected = runner.model(ids, positions)
            runner.model.model.graph_safe_decode = True
            actual = runner.run_model(ids, positions, False)
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-3)
            assert runner.decode_graph_replay_count - before == int(batch_size <= 512)
        assert runner.decode_graph_replay_count == 5
    finally:
        reset_context()
        torch.set_default_device(previous_device)
        torch.set_default_dtype(previous_dtype)


@torch.inference_mode()
def distributed_graph_worker(tp_size, backend):
    import torch.distributed as dist
    from nanovllm.distributed.parallel_state import (
        destroy_model_parallel, initialize_model_parallel,
    )
    from nanovllm.utils.context import reset_context

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", device_id=device)
    ep_size = dist.get_world_size()
    dp_size = ep_size // tp_size
    initialize_model_parallel(tp_size, enable_expert_parallel=True, data_parallel_size=dp_size)
    results = []
    try:
        for capacity_options in ({}, {"expert_capacity": 2}, {"expert_capacity_factor": 0.5}):
            torch.manual_seed(41)
            full = ExpertParallelMoE(32, 64, 128, 8, True, **capacity_options)
            for parameter in full.parameters():
                parameter.normal_(0, 0.02)
            shard = ExpertParallelMoE(
                32, 64, 128, 8, True, rank, ep_size, dispatch_backend=backend,
                graph_safe_decode=True, **capacity_options,
            )
            shard.gate.weight.copy_(full.gate.weight)
            start = rank * shard.num_local_experts
            end = start + shard.num_local_experts
            shard.gate_up_proj.copy_(full.gate_up_proj[start:end])
            shard.down_proj.copy_(full.down_proj[start:end])
            full = full.to(device=device, dtype=torch.bfloat16)
            shard = shard.to(device=device, dtype=torch.bfloat16)
            assert shard.graph_token_limit == dp_size * 512

            def reject_dynamic(*args):
                raise AssertionError("captured DP batch entered dynamic MoE packing")

            shard._forward_triton = reject_dynamic
            graphs = {}
            pool = None
            for batch_size in (512, 257, 17, 1):
                torch.manual_seed(100 + batch_size)
                first = torch.randn(dp_size * batch_size, 32, device=device, dtype=torch.bfloat16)
                second = torch.randn_like(first)
                dp_rank = rank // tp_size
                static = first.chunk(dp_size)[dp_rank].clone()
                reset_context()
                shard(static)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    shard(static)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    output = shard(static)
                if pool is None:
                    pool = graph.pool()
                graphs[batch_size] = (graph, static, output, first, second)

            # Switch sizes and routing data after every graph has been captured.
            for batch_size in (1, 512, 17, 257, 512):
                graph, static, output, first, second = graphs[batch_size]
                for hidden in (first, second):
                    expected = full(hidden).chunk(dp_size)[rank // tp_size]
                    static.copy_(hidden.chunk(dp_size)[rank // tp_size])
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-3)
            results.append({"capacity": capacity_options, "batch_sizes": sorted(graphs),
                            "global_token_limit": shard.graph_token_limit, "replays": 10})
            del graphs, shard, full, graph, static, output, first, second, pool
            torch.cuda.empty_cache()
        if rank == 0:
            print(json.dumps({"dp": dp_size, "tp": tp_size, "ep": ep_size,
                              "backend": backend, "passed": results}), flush=True)
    finally:
        reset_context()
        destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--backend", choices=("allgather_reduce", "allgather_reducescatter"),
                        default="allgather_reduce")
    args = parser.parse_args()
    distributed_graph_worker(args.tp, args.backend)
