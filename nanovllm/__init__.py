"""Public API for extend-nano-vLLM.

The runtime imports CUDA extensions, so keep the top-level exports lazy. This
lets standalone layer tools inspect their arguments without importing the full
model-execution stack.
"""

__all__ = (
    "LLM",
    "SamplingParams",
    "Eagle3Config",
    "Eagle3ForCausalLM",
    "load_eagle3_model",
)


def __getattr__(name: str):
    if name == "LLM":
        from nanovllm.llm import LLM

        return LLM
    if name == "SamplingParams":
        from nanovllm.sampling_params import SamplingParams

        return SamplingParams
    if name in {"Eagle3Config", "Eagle3ForCausalLM", "load_eagle3_model"}:
        from nanovllm.models.eagle3 import (
            Eagle3Config,
            Eagle3ForCausalLM,
            load_eagle3_model,
        )

        return {
            "Eagle3Config": Eagle3Config,
            "Eagle3ForCausalLM": Eagle3ForCausalLM,
            "load_eagle3_model": load_eagle3_model,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
