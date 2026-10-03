"""Contracts for optional MoE arithmetic shared by EP=1 and EP>1."""

import pytest
import torch
from transformers import Qwen3MoeConfig

from nanovllm.layers.moe import ExpertParallelMoE
from nanovllm.models.qwen3_moe import Qwen3MoeDecoderLayer


def _small_config():
    return Qwen3MoeConfig(
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        num_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=128,
    )


def test_deterministic_qk_option_applies_to_ep1_and_ep4(monkeypatch):
    monkeypatch.setenv("NANOVLLM_EXPERIMENTAL_DETERMINISTIC_QK_NORM", "1")
    for ep_size in (1, 4):
        layer = Qwen3MoeDecoderLayer(
            _small_config(), 0, 0, ep_size, "replicated", None, False, None
        )
        assert layer.self_attn.q_norm.deterministic_cuda
        assert layer.self_attn.k_norm.deterministic_cuda


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fp32_ep_combine_preserves_local_precision(monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    monkeypatch.setenv("NANOVLLM_EXPERIMENTAL_FP32_EP_REDUCE", "1")
    module = ExpertParallelMoE(16, 32, 8, 2, True, ep_rank=0, ep_size=4)
    expert_output = torch.tensor(
        [[1.0] * 16, [0.00390625] * 16], dtype=torch.bfloat16,
        device=device,
    )
    route_rows = torch.tensor([0, 1], dtype=torch.int32, device=device)
    output = module._combine_expert_routes(expert_output, route_rows, 1, 2)
    assert output.dtype == torch.float32
    torch.testing.assert_close(
        output, torch.full((1, 16), 1.00390625, device=device), rtol=0, atol=0
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("ep_size", [1, 4])
def test_fp64_route_combine_uses_same_precision_for_all_ep_sizes(
    monkeypatch, device, ep_size
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    monkeypatch.setenv("NANOVLLM_EXPERIMENTAL_FP64_MOE_COMBINE", "1")
    module = ExpertParallelMoE(
        16, 32, 8, 8, True, ep_rank=0, ep_size=ep_size
    )
    values = torch.tensor(
        [0.043701171875, 0.037109375, -0.0047607421875,
         -0.00506591796875, -4.423782229423523e-09, -0.001953125,
         0.005859375, -0.01068115234375],
        dtype=torch.bfloat16, device=device,
    ).unsqueeze(1).expand(8, 16).contiguous()
    route_rows = torch.arange(8, dtype=torch.int32, device=device)
    output = module._combine_expert_routes(values, route_rows, 1, 8)
    assert output.dtype == torch.float64
    expected = values.double().sum(dim=0, keepdim=True)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
