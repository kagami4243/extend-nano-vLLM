"""Capture and replay fixed-shape MoE decode with changed token data."""

import pytest
import torch

from nanovllm.layers.moe import ExpertParallelMoE


def test_graph_safe_decode_rejects_dynamic_dispatch():
    with pytest.raises(ValueError, match="graph-safe dispatch"):
        ExpertParallelMoE(
            128, 256, 8, 2, True,
            dispatch_backend="all_to_all", graph_safe_decode=True,
        )


@pytest.mark.parametrize("num_tokens", [1, 4, 16, 17, 32])
@torch.inference_mode()
def test_moe_decode_cuda_graph_replays_changed_inputs(num_tokens):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    torch.manual_seed(17)
    module = ExpertParallelMoE(
        128, 256, 8, 2, True, graph_safe_decode=True
    ).cuda().bfloat16()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(0, 0.02)
    hidden = torch.randn(num_tokens, 128, device="cuda", dtype=torch.bfloat16)
    first = hidden.clone()
    second = torch.randn_like(hidden)
    module.graph_safe_decode = False
    expected_first = module(first)
    expected_second = module(second)
    module.graph_safe_decode = True
    torch.testing.assert_close(module(first), expected_first, rtol=1e-2, atol=1e-3)
    torch.testing.assert_close(module(second), expected_second, rtol=1e-2, atol=1e-3)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        module(hidden)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = module(hidden)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected_first, rtol=1e-2, atol=1e-3)
    hidden.copy_(second)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected_second, rtol=1e-2, atol=1e-3)
    assert not torch.equal(expected_first, expected_second)


@pytest.mark.parametrize("num_tokens", [1, 4, 16, 17, 32])
@torch.inference_mode()
def test_static_decode_matches_dynamic_local_expert_shards(num_tokens):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    torch.manual_seed(100 + num_tokens)
    hidden = torch.randn(
        num_tokens, 128, device="cuda", dtype=torch.bfloat16
    )
    for ep_rank in range(4):
        module = ExpertParallelMoE(
            128, 256, 128, 8, True, ep_rank=ep_rank, ep_size=4
        ).cuda().bfloat16()
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.normal_(0, 0.02)
        weights, experts = module._route(hidden)
        dynamic, dynamic_count = module._execute_local(hidden, weights, experts)
        module.graph_safe_decode = True
        static, static_count = module._execute_local(hidden, weights, experts)
        assert static_count == dynamic_count
        torch.testing.assert_close(static, dynamic, rtol=1e-2, atol=1e-3)


@torch.inference_mode()
def test_static_decode_handles_expert_with_more_than_one_tile():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    torch.manual_seed(132)
    module = ExpertParallelMoE(128, 256, 8, 2, True).cuda().bfloat16()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(0, 0.02)
    hidden = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
    selected = torch.tensor([0, 1], device="cuda").expand(32, 2)
    weights = torch.full((32, 2), 0.5, device="cuda", dtype=torch.bfloat16)
    dynamic, _ = module._forward_triton(hidden, weights, selected)
    static, _ = module._forward_static_decode(hidden, weights, selected)
    torch.testing.assert_close(static, dynamic, rtol=1e-2, atol=1e-3)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        module._forward_static_decode(hidden, weights, selected)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed, _ = module._forward_static_decode(hidden, weights, selected)
    hidden.copy_(torch.randn_like(hidden))
    expected, _ = module._forward_triton(hidden, weights, selected)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(replayed, expected, rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize("static_decode", [False, True])
@torch.inference_mode()
def test_moe_combine_is_exactly_repeatable(static_decode):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    torch.manual_seed(17)
    module = ExpertParallelMoE(128, 256, 8, 8, True).cuda().bfloat16()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(0, 0.1)
    num_tokens = 16
    hidden = torch.randn(num_tokens, 128, device="cuda", dtype=torch.bfloat16)
    selected = torch.arange(8, device="cuda").expand(num_tokens, 8)
    weights = torch.full(
        (num_tokens, 8), 0.125, device="cuda", dtype=torch.bfloat16
    )
    forward = (
        module._forward_static_decode if static_decode else module._forward_triton
    )
    outputs = [forward(hidden, weights, selected)[0] for _ in range(8)]
    assert all(torch.equal(output, outputs[0]) for output in outputs[1:])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_route_order_combine_handles_missing_and_permuted_routes(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    torch.manual_seed(23)
    module = ExpertParallelMoE(32, 64, 8, 4, True)
    packed = torch.randn(9, 32, device=device, dtype=torch.bfloat16)
    route_rows = torch.tensor(
        [3, 0, 9, 7, 2, 9, 8, 1, 4, 9, 6, 5],
        device=device, dtype=torch.int32,
    )
    expected = torch.cat((packed, torch.zeros_like(packed[:1])))[
        route_rows.long()
    ].reshape(3, 4, 32).float().sum(dim=1).bfloat16()
    actual = module._combine_expert_routes(packed, route_rows, 3, 4)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@torch.inference_mode()
@pytest.mark.xfail(
    strict=True,
    reason="BF16 projection-before-routing parity regresses TP+EP/PP+EP output",
)
def test_grouped_down_projection_rounds_before_route_weight():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    torch.manual_seed(41)
    module = ExpertParallelMoE(32, 64, 8, 4, True)
    hidden = torch.randn(16, 32, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(8, 64, 32, device="cuda", dtype=torch.bfloat16)
    row_ids = torch.arange(16, device="cuda", dtype=torch.int32)
    expert_ids = torch.zeros(1, device="cuda", dtype=torch.int32)
    route_weights = torch.rand(16, device="cuda", dtype=torch.bfloat16)

    unweighted = module._run_grouped_gemm(
        hidden, weights, row_ids, expert_ids
    )
    weighted = module._run_grouped_gemm(
        hidden, weights, row_ids, expert_ids, route_weights
    )
    expected = unweighted * route_weights[:, None]
    torch.testing.assert_close(weighted, expected, rtol=0, atol=0)
