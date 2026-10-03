"""MoE uses the measured throughput-oriented prefill scheduler by default."""

from nanovllm.config import Config


MOE_MODEL = "./models/Qwen3-30B-A3B-Base"
DENSE_MODEL = "./models/Qwen3-0.6B"


def make_config(model, **kwargs):
    return Config(
        model, max_model_len=32, max_num_batched_tokens=16,
        **kwargs,
    )


def test_moe_prefill_batching_is_on_by_default():
    config = make_config(MOE_MODEL)
    assert config.enable_prefill_batching is True


def test_moe_prefill_batching_can_be_disabled_explicitly():
    config = make_config(MOE_MODEL, enable_prefill_batching=False)
    assert config.enable_prefill_batching is False


def test_dense_prefill_default_is_preserved():
    config = make_config(DENSE_MODEL)
    assert config.enable_prefill_batching is False
