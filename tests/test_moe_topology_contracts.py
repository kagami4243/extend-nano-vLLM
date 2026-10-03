"""EP is the flattened DP x TP axis, not an additional world dimension."""

import pytest

from nanovllm.config import Config


MODEL = "./models/Qwen3-30B-A3B-Base"


@pytest.mark.parametrize("dp,tp,pp", [(1, 1, 1), (1, 2, 2), (2, 1, 1), (2, 2, 1)])
def test_ep_is_derived_without_multiplying_world_size(dp, tp, pp):
    config = Config(
        MODEL, data_parallel_size=dp, tensor_parallel_size=tp,
        pipeline_parallel_size=pp, enable_expert_parallel=True,
        master_port=29591, enforce_eager=pp > 1,
        max_model_len=32, max_num_batched_tokens=16,
    )
    assert config.effective_expert_parallel_size == dp * tp
    assert config.model_parallel_size == pp * tp
    assert config.parallel_world_size == dp * pp * tp
    assert config.moe_global_dp == (dp > 1)


def test_matching_explicit_ep_is_only_a_topology_check():
    config = Config(
        MODEL, data_parallel_size=2, tensor_parallel_size=2,
        expert_parallel_size=4, enable_expert_parallel=True,
        master_port=29591, max_model_len=32, max_num_batched_tokens=16,
    )
    assert config.model_parallel_size == 2
    assert config.parallel_world_size == 4


@pytest.mark.parametrize("options", [
    {"tensor_parallel_size": 1, "expert_parallel_size": 4},
    {"tensor_parallel_size": 2, "expert_parallel_size": 4},
    {"data_parallel_size": 2, "tensor_parallel_size": 2,
     "expert_parallel_size": 2},
])
def test_independent_ep_sizes_are_rejected(options):
    with pytest.raises(ValueError, match="EP.*DP.*TP"):
        Config(MODEL, enable_expert_parallel=True, master_port=29591, **options)


def test_cross_tp_expert_axis_is_removed():
    with pytest.raises(ValueError, match="independent.*EP|DP.*TP"):
        Config(
            MODEL, enable_expert_parallel=True, tensor_parallel_size=2,
            expert_parallel_size=2, moe_shard_across_tp=True,
        )


def test_ep_size_requires_expert_parallel_enabled():
    with pytest.raises(ValueError, match="enable_expert_parallel"):
        Config(MODEL, expert_parallel_size=1)


def test_ep_rejects_dense_model():
    with pytest.raises((ValueError, AssertionError), match="expert|MoE|moe"):
        Config("./models/Qwen3-0.6B", enable_expert_parallel=True)


def test_dp_disabled_expert_parallel_keeps_independent_replicas():
    config = Config(MODEL, data_parallel_size=2)
    assert config.effective_expert_parallel_size == 1
    assert not config.moe_global_dp


def test_global_ep_rejects_uncoordinated_pipeline_and_migration():
    for options in ({"pipeline_parallel_size": 2, "enforce_eager": True},
                    {"moe_dynamic_placement": True}):
        with pytest.raises(ValueError, match="global DP MoE"):
            Config(
                MODEL, data_parallel_size=2, enable_expert_parallel=True,
                master_port=29591, **options,
            )


def test_engine_rejects_obsolete_global_dp_switch_before_startup():
    from nanovllm.engine.llm_engine import LLMEngine

    with pytest.raises(ValueError, match="automatically|DP.*TP"):
        LLMEngine(MODEL, moe_global_dp=False)


def test_groups_reject_world_size_for_an_extra_ep_axis(monkeypatch):
    import torch.distributed as dist
    from nanovllm.distributed.parallel_state import initialize_model_parallel

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 4)
    with pytest.raises(ValueError, match="world size"):
        initialize_model_parallel(2, enable_expert_parallel=True)
