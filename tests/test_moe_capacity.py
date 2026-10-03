"""Deterministic expert capacity and token dropping semantics."""

import torch
import torch.nn.functional as F
import pytest

from nanovllm.layers.moe import ExpertParallelMoE


def reference_with_capacity(module, hidden, capacity):
    logits = F.linear(hidden, module.gate.weight)
    probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    weights, experts = torch.topk(probs, module.top_k, dim=-1)
    if module.norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights.to(hidden.dtype)
    counts = [0] * module.num_experts
    output = torch.zeros_like(hidden)
    for token in range(hidden.size(0)):
        for route in range(module.top_k):
            expert = int(experts[token, route])
            if counts[expert] >= capacity:
                continue
            counts[expert] += 1
            gate_up = F.linear(hidden[token], module.gate_up_proj[expert])
            gate, up = gate_up.chunk(2)
            contribution = F.linear(
                F.silu(gate) * up, module.down_proj[expert]
            )
            output[token] += weights[token, route] * contribution
    return output, sum(counts), hidden.size(0) * module.top_k - sum(counts)


def test_expert_capacity_drops_overflow_assignments():
    torch.manual_seed(31)
    module = ExpertParallelMoE(
        16, 32, 4, 2, True, expert_capacity=1
    )
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(0, 0.1)
    hidden = torch.randn(9, 16)
    expected, kept, dropped = reference_with_capacity(module, hidden, 1)
    actual = module(hidden)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert 0 < kept <= 4
    assert dropped > 0
    assert module.dispatch_assignment_count == kept
    assert module.return_assignment_count == kept
    assert module.total_assignment_count == 18
    assert module.dropped_assignment_count == dropped


def test_capacity_must_be_positive():
    try:
        ExpertParallelMoE(16, 32, 4, 2, True, expert_capacity=0)
    except ValueError as error:
        assert "capacity" in str(error)
    else:
        raise AssertionError("zero expert capacity must be rejected")


@pytest.mark.parametrize("tokens,expected_capacity", [(2, 1), (9, 5)])
def test_capacity_factor_scales_with_current_token_count(tokens, expected_capacity):
    torch.manual_seed(31)
    module = ExpertParallelMoE(
        16, 32, 4, 2, True, expert_capacity_factor=1.0
    )
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(0, 0.1)
    hidden = torch.randn(tokens, 16)
    expected, kept, dropped = reference_with_capacity(
        module, hidden, expected_capacity
    )
    actual = module(hidden)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert module.dispatch_assignment_count == kept
    assert module.dropped_assignment_count == dropped


@pytest.mark.parametrize("factor", [0.0, -1.0, float("inf"), float("nan")])
def test_capacity_factor_must_be_finite_and_positive(factor):
    with pytest.raises(ValueError, match="capacity factor"):
        ExpertParallelMoE(16, 32, 4, 2, True, expert_capacity_factor=factor)


def test_fixed_capacity_and_factor_are_mutually_exclusive():
    with pytest.raises(ValueError, match="capacity"):
        ExpertParallelMoE(
            16, 32, 4, 2, True, expert_capacity=2,
            expert_capacity_factor=1.0,
        )
