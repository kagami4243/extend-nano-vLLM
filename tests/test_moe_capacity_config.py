"""Configuration contract for token-scaled MoE expert capacity."""

from types import SimpleNamespace

import pytest

from nanovllm.config import Config


def model_config(monkeypatch, tmp_path, model_type="qwen3_moe", **kwargs):
    monkeypatch.setattr(
        "nanovllm.config.AutoConfig.from_pretrained",
        lambda _: SimpleNamespace(
            model_type=model_type,
            num_experts=8,
            num_hidden_layers=4,
            max_position_embeddings=1024,
        ),
    )
    return Config(model=str(tmp_path), **kwargs)


def test_capacity_factor_is_available_to_moe_model(monkeypatch, tmp_path):
    config = model_config(
        monkeypatch, tmp_path, enable_expert_parallel=True,
        tensor_parallel_size=2, moe_expert_capacity_factor=1.25,
    )
    assert config.moe_expert_capacity_factor == 1.25


@pytest.mark.parametrize("factor", [0, -1, float("inf"), float("nan"), True])
def test_config_rejects_invalid_capacity_factor(monkeypatch, tmp_path, factor):
    with pytest.raises(ValueError, match="capacity factor"):
        model_config(
            monkeypatch, tmp_path, enable_expert_parallel=True,
            moe_expert_capacity_factor=factor,
        )


def test_config_rejects_two_capacity_modes(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="mutually exclusive"):
        model_config(
            monkeypatch, tmp_path, enable_expert_parallel=True,
            moe_expert_capacity=2, moe_expert_capacity_factor=1.25,
        )


def test_config_rejects_capacity_factor_for_dense_model(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="qwen3_moe"):
        model_config(
            monkeypatch, tmp_path, model_type="qwen3",
            moe_expert_capacity_factor=1.25,
        )
