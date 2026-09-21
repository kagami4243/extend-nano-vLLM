import sys
import types

import pytest
import torch
from torch import nn

try:
    from nanovllm.layers.quantization import (
        FP8_FORMATS,
        _quantize_fp8,
        _fp8_linear,
        normalize_fp8_format,
    )
except ImportError as exc:
    if "flash_attn" not in str(exc):
        raise
    flash_attn = types.ModuleType("flash_attn")
    flash_attn.flash_attn_func = lambda *args, **kwargs: None
    flash_attn.flash_attn_varlen_func = lambda *args, **kwargs: None
    flash_attn.flash_attn_with_kvcache = lambda *args, **kwargs: None
    sys.modules["flash_attn"] = flash_attn
    for module_name in list(sys.modules):
        if module_name == "nanovllm" or module_name.startswith("nanovllm."):
            del sys.modules[module_name]
    from nanovllm.layers.quantization import (
        FP8_FORMATS,
        _quantize_fp8,
        _fp8_linear,
        normalize_fp8_format,
    )


def test_fp8_format_defaults_to_per_tensor():
    assert normalize_fp8_format(None) == "per_tensor"
    assert normalize_fp8_format("per_token") == "per_token"
    assert normalize_fp8_format("PER_TENSOR") == "per_tensor"
    assert FP8_FORMATS == ("per_tensor", "per_token")


def test_fp8_format_rejects_unknown_values():
    with pytest.raises(ValueError, match="per_tensor, per_token"):
        normalize_fp8_format("per_channel")


@pytest.mark.parametrize(
    ("format_name", "expected_shape"),
    [("per_tensor", (1,)), ("per_token", (3, 1))],
)
def test_fp8_linear_passes_format_specific_activation_scale(
    monkeypatch, format_name, expected_shape
):
    calls = {}

    class Module:
        weight = torch.zeros(2, 4, dtype=torch.float8_e4m3fn)
        weight_scale = torch.ones(
            (1, 4) if format_name == "per_token" else (1,), dtype=torch.float32
        )
        output_size_per_partition = 4
        fp8_activation_format = format_name

    def fake_scaled_mm(a, b, *, scale_a, scale_b, out_dtype):
        calls["scale_a"] = scale_a.detach().clone()
        calls["scale_b"] = scale_b.detach().clone()
        return torch.zeros(a.shape[0], b.shape[1], dtype=out_dtype)

    monkeypatch.setattr(torch, "_scaled_mm", fake_scaled_mm)
    output = _fp8_linear(torch.ones(3, 2, dtype=torch.bfloat16), Module())

    assert output.shape == (3, 4)
    assert tuple(calls["scale_a"].shape) == expected_shape
    expected_weight_shape = (1, 4) if format_name == "per_token" else (1,)
    assert tuple(calls["scale_b"].shape) == expected_weight_shape


def test_fp8_linear_uses_default_format_for_legacy_modules(monkeypatch):
    class Module:
        weight = torch.zeros(2, 4, dtype=torch.float8_e4m3fn)
        weight_scale = torch.ones(1, dtype=torch.float32)
        output_size_per_partition = 4

    seen = {}

    def fake_scaled_mm(a, b, *, scale_a, scale_b, out_dtype):
        seen["scale_a_shape"] = tuple(scale_a.shape)
        return torch.zeros(a.shape[0], b.shape[1], dtype=out_dtype)

    monkeypatch.setattr(torch, "_scaled_mm", fake_scaled_mm)
    _fp8_linear(torch.ones(1, 2, dtype=torch.bfloat16), Module())
    # A module without FP8 format metadata falls back to the default format,
    # which is per-tensor and therefore carries a scalar activation scale.
    assert seen["scale_a_shape"] == (1,)


@pytest.mark.parametrize(
    ("format_name", "expected_weight_scale_shape"),
    [("per_tensor", (1,)), ("per_token", (1, 4))],
)
def test_fp8_weight_scales_follow_format(format_name, expected_weight_scale_shape):
    module = nn.Module()
    module.weight = torch.ones(4, 2, dtype=torch.bfloat16)
    _quantize_fp8(module, format_name)
    assert tuple(module.weight_scale.shape) == expected_weight_scale_shape
    assert module.fp8_activation_format == format_name
