"""Numerical contract for the deterministic Q/K norm used by MoE attention."""

import pytest
import torch

from nanovllm.layers.layernorm import RMSNorm


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_deterministic_q_norm_matches_eager_and_replays(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    torch.manual_seed(37)
    norm = RMSNorm(128, deterministic_cuda=True).cuda().to(dtype)
    norm.weight.data.uniform_(0.5, 1.5)
    source = torch.randn(64, 40, 128, device="cuda", dtype=dtype)
    q = source[:, :32]

    def eager(value):
        values = value.float()
        variance = values.square().mean(dim=-1, keepdim=True)
        return (values * torch.rsqrt(variance + norm.eps)).to(dtype) * norm.weight

    expected = eager(q)
    actual = norm(q)
    tolerance = 0.004 if dtype == torch.bfloat16 else 0.0005
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=tolerance)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = norm(q)
    graph.replay()
    torch.testing.assert_close(captured, actual, rtol=0, atol=0)
    q.copy_(torch.randn_like(q))
    graph.replay()
    torch.testing.assert_close(captured, norm(q), rtol=0, atol=0)
