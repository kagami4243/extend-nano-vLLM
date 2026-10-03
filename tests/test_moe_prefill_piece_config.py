"""MoE piecewise prefill graph is an explicit, supported-shape experiment."""

from types import SimpleNamespace

import pytest

from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner


MOE_MODEL = "./models/Qwen3-30B-A3B-Base"
DENSE_MODEL = "./models/Qwen3-0.6B"


def make_config(model=MOE_MODEL, **kwargs):
    return Config(model, max_model_len=32, max_num_batched_tokens=16, **kwargs)


def test_moe_prefill_piece_can_be_enabled_for_graph_mode():
    config = make_config(
        moe_prefill_piece=True, moe_prefill_piece_capture_sizes=(64, 512)
    )
    assert config.moe_prefill_piece is True
    assert config.moe_prefill_piece_capture_sizes == (64, 512)


def test_moe_prefill_piece_remains_opt_in():
    assert make_config().moe_prefill_piece is False


def test_moe_prefill_piece_requires_explicit_capture_sizes():
    with pytest.raises(ValueError, match="capture sizes"):
        make_config(moe_prefill_piece=True)


def test_capture_sizes_without_moe_prefill_piece_are_rejected():
    with pytest.raises(ValueError, match="capture sizes"):
        make_config(moe_prefill_piece_capture_sizes=(64,))


@pytest.mark.parametrize("sizes", [(0,), (64, 64), (64, -1), ("64",)])
def test_moe_prefill_piece_rejects_invalid_capture_sizes(sizes):
    with pytest.raises(ValueError, match="capture sizes"):
        make_config(
            moe_prefill_piece=True, moe_prefill_piece_capture_sizes=sizes
        )


def test_uncaptured_moe_prefill_shape_falls_back_without_allocating_graph():
    runner = ModelRunner.__new__(ModelRunner)
    runner.config = SimpleNamespace(
        moe_prefill_piece=True, moe_prefill_piece_capture_sizes=(64,)
    )
    runner.prefill_piece_graphs = {}
    assert runner._capture_prefill_piece_graph(32, None, None) is None
    assert runner.prefill_piece_graphs == {}


@pytest.mark.parametrize("options", [
    {"model": DENSE_MODEL},
    {"enforce_eager": True},
    {"pipeline_parallel_size": 2, "enforce_eager": True},
    {"enable_expert_parallel": True, "moe_dispatch_backend": "all_to_all"},
])
def test_moe_prefill_piece_rejects_unsupported_modes(options):
    model = options.get("model", MOE_MODEL)
    config_options = {key: value for key, value in options.items() if key != "model"}
    with pytest.raises(ValueError, match="MoE prefill piece"):
        make_config(
            model, moe_prefill_piece=True,
            moe_prefill_piece_capture_sizes=(64,), **config_options,
        )
