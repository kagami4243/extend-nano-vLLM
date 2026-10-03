"""Compact route activations and device-side lengths under Graph replay."""

import pytest
import torch

from nanovllm.layers.moe import ExpertParallelMoE


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@torch.inference_mode()
def test_compact_activations_have_only_token_route_rows(monkeypatch):
    torch.manual_seed(17)
    layer = ExpertParallelMoE(32, 64, 128, 8, True).cuda().bfloat16()
    for parameter in layer.parameters():
        parameter.normal_(0, 0.02)
    hidden = torch.randn(512, 32, device="cuda", dtype=torch.bfloat16)
    weights, experts = layer._route(hidden)
    expected, _ = layer._forward_triton(hidden, weights, experts)
    calls = []
    original = layer._run_grouped_gemm

    def record(*args, **kwargs):
        output = original(*args, **kwargs)
        calls.append({"rows": output.size(0), "index_capacity": args[2].numel(),
                      "length": kwargs["num_tokens_post_padded"].item(),
                      "indices": args[2].clone()})
        return output

    monkeypatch.setattr(layer, "_run_grouped_gemm", record)
    actual, _ = layer._forward_static_decode(hidden, weights, experts)
    counts = torch.bincount(experts.reshape(-1), minlength=128)
    effective_rows = int((((counts + 15) // 16) * 16).sum().item())
    assert len(calls) == 2
    for call in calls:
        assert call["rows"] == 4096
        assert call["index_capacity"] == 6016
        assert call["length"] == effective_rows
        indices = call["indices"][:effective_rows]
        routes = indices[indices < 4096].sort().values
        torch.testing.assert_close(routes, torch.arange(4096, device="cuda", dtype=torch.int32))
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-3)


@torch.inference_mode()
def test_compact_graph_changes_between_empty_and_skewed_local_routes():
    torch.manual_seed(43)
    layer = ExpertParallelMoE(32, 64, 8, 2, True, ep_rank=1, ep_size=2).cuda().bfloat16()
    for parameter in layer.parameters():
        parameter.normal_(0, 0.02)
    hidden = torch.randn(512, 32, device="cuda", dtype=torch.bfloat16)
    selected = torch.tensor([0, 1], device="cuda").expand(512, 2).clone()
    weights = torch.full((512, 2), 0.5, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        layer._forward_static_decode(hidden, weights, selected)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, _ = layer._forward_static_decode(hidden, weights, selected)
    for routes in ([0, 1], [4, 5], [0, 1], [7, 4]):
        selected.copy_(torch.tensor(routes, device="cuda").expand(512, 2))
        hidden.copy_(torch.randn_like(hidden))
        expected, _ = layer._forward_triton(hidden, weights, selected)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize("capacity", [None, 0.5])
@torch.inference_mode()
def test_gathered_1024_token_graph_shards_on_one_gpu(monkeypatch, capacity):
    from nanovllm.layers import moe

    monkeypatch.setattr(moe, "get_tp_world_size", lambda: 2)
    torch.manual_seed(47)
    options = {} if capacity is None else {"expert_capacity_factor": capacity}
    full = ExpertParallelMoE(32, 64, 128, 8, True, **options).cuda().bfloat16()
    for parameter in full.parameters():
        parameter.normal_(0, 0.02)
    first = torch.randn(1024, 32, device="cuda", dtype=torch.bfloat16)
    second = torch.randn_like(first)
    owners = tuple(expert % 4 for expert in range(128))
    partials = [[], []]
    for rank in range(4):
        shard = ExpertParallelMoE(
            32, 64, 128, 8, True, ep_rank=rank, ep_size=4,
            dispatch_backend="allgather_reduce", graph_safe_decode=True,
            expert_owners=owners, **options,
        ).cuda().bfloat16()
        shard.gate.weight.copy_(full.gate.weight)
        shard.gate_up_proj.copy_(full.gate_up_proj[list(shard.local_expert_ids)])
        shard.down_proj.copy_(full.down_proj[list(shard.local_expert_ids)])
        assert shard.graph_token_limit == 1024
        hidden = first.clone()

        def run():
            weights, experts = shard._route(hidden)
            return shard._execute_local(hidden, weights, experts)[0]

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run()
        for index, inputs in enumerate((first, second)):
            hidden.copy_(inputs)
            shard.graph_safe_decode = False
            expected = run()
            shard.graph_safe_decode = True
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-3)
            partials[index].append(output.clone())
        del graph, shard, output, expected
    for inputs, outputs in zip((first, second), partials):
        combined = torch.stack(outputs).float().sum(dim=0).bfloat16()
        torch.testing.assert_close(combined, full(inputs), rtol=1e-2, atol=1e-3)
