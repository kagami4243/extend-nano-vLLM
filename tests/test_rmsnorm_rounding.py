"""The fused residual RMSNorm must retain the model's input-dtype add."""

import os
import subprocess
import sys

import pytest
import torch

from nanovllm.layers.layernorm import RMSNorm


@pytest.mark.parametrize("dtype", [
    pytest.param(torch.bfloat16, marks=pytest.mark.xfail(
        strict=True, reason="BF16 residual rounding changes MoE parallel outputs"
    )),
    pytest.param(torch.float16, marks=pytest.mark.xfail(
        strict=True, reason="FP16 residual rounding changes MoE parallel outputs"
    )),
    torch.float32,
])
def test_add_rms_matches_separate_residual_add(dtype):
    torch.manual_seed(17)
    norm = RMSNorm(2048).to(dtype)
    x = torch.randn(16, 2048, dtype=dtype)
    residual = torch.randn_like(x)

    expected_residual = x + residual
    original_x = x.clone()
    original_residual = residual.clone()
    values = expected_residual.float()
    values = values * torch.rsqrt(values.pow(2).mean(-1, keepdim=True) + norm.eps)
    expected = norm.weight * values.to(dtype)

    actual, actual_residual = norm(x, residual)
    torch.testing.assert_close(x, original_x, rtol=0, atol=0)
    torch.testing.assert_close(residual, original_residual, rtol=0, atol=0)
    torch.testing.assert_close(actual_residual, expected_residual, rtol=0, atol=0)
    tolerance = 1e-6 if dtype == torch.float32 else 0
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


def test_rms_without_residual_does_not_change_input():
    torch.manual_seed(31)
    norm = RMSNorm(128)
    x = torch.randn(4, 128)
    original = x.clone()
    norm(x)
    torch.testing.assert_close(x, original, rtol=0, atol=0)


def test_eager_rms_available_when_torchdynamo_disabled():
    code = """
import torch
from nanovllm.layers.layernorm import RMSNorm
norm = RMSNorm(128)
x = torch.randn(4, 128)
torch.testing.assert_close(norm.rms_forward_eager(x), norm.rms_forward(x), rtol=0, atol=0)
"""
    environment = os.environ.copy()
    environment["TORCHDYNAMO_DISABLE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", code], env=environment,
        capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
