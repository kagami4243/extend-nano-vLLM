from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.qwen3_moe import Qwen3MoeForCausalLM


_MODEL_REGISTRY = {
    "qwen3": Qwen3ForCausalLM,
    "qwen3_moe": Qwen3MoeForCausalLM,
}


def get_model_class(hf_config):
    model_type = hf_config.model_type
    try:
        return _MODEL_REGISTRY[model_type]
    except KeyError as exc:
        raise ValueError(f"unsupported model type: {model_type}") from exc
