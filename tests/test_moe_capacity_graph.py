"""Fixed expert capacity must preserve decode results under graph replay."""

import pytest
import torch

from nanovllm.layers.moe import ExpertParallelMoE


@pytest.mark.parametrize("num_tokens", [16, 32])
@pytest.mark.parametrize("capacity_options", [
    {"expert_capacity": 1},
    {"expert_capacity_factor": 0.5},
])
@torch.inference_mode()
def test_capacity_decode_graph_replays_changed_inputs(num_tokens, capacity_options):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    torch.manual_seed(100 + num_tokens)
    module = ExpertParallelMoE(
        128, 256, 8, 2, True,
        graph_safe_decode=True, **capacity_options,
    ).cuda().bfloat16()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(0, 0.02)
    first = torch.randn(num_tokens, 128, device="cuda", dtype=torch.bfloat16)
    second = torch.randn_like(first)

    module.graph_safe_decode = False
    expected_first = module(first)
    expected_second = module(second)
    assert module.dropped_assignment_count > 0
    module.graph_safe_decode = True
    torch.testing.assert_close(module(first), expected_first, rtol=1e-2, atol=1e-3)
    torch.testing.assert_close(module(second), expected_second, rtol=1e-2, atol=1e-3)

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
