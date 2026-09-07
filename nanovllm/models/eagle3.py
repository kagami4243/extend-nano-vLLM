import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import ReplicatedLinear
from nanovllm.models.qwen3 import Qwen3Attention, Qwen3MLP
from nanovllm.utils.loader import load_model


@dataclass(frozen=True)
class Eagle3Config:
    """Runtime configuration for a speculators-format EAGLE3 checkpoint."""

    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    attention_bias: bool
    target_vocab_size: int
    draft_vocab_size: int
    target_hidden_size: int
    num_aux_hidden_states: int
    norm_before_residual: bool
    torch_dtype: torch.dtype

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> "Eagle3Config":
        config_path = Path(model_path) / "config.json"
        with config_path.open() as config_file:
            raw_config = json.load(config_file)

        if raw_config.get("speculators_model_type") != "eagle3":
            raise ValueError("expected a speculators-format EAGLE3 checkpoint")

        transformer_config = raw_config.get("transformer_layer_config")
        if not isinstance(transformer_config, dict):
            raise ValueError("EAGLE3 config must contain transformer_layer_config")
        if transformer_config.get("model_type") != "llama":
            raise ValueError(
                "only Llama-style EAGLE3 transformer layers are supported"
            )

        hidden_size = transformer_config["hidden_size"]
        aux_layer_ids = raw_config.get("eagle_aux_hidden_state_layer_ids")
        num_aux_hidden_states = (
            len(aux_layer_ids) if aux_layer_ids else raw_config.get("num_aux_hidden_states", 3)
        )
        dtype_name = raw_config.get("torch_dtype", "bfloat16")
        torch_dtype = getattr(torch, dtype_name, None)
        if not isinstance(torch_dtype, torch.dtype):
            raise ValueError(f"unsupported EAGLE3 torch_dtype: {dtype_name}")

        return cls(
            hidden_size=hidden_size,
            intermediate_size=transformer_config["intermediate_size"],
            num_attention_heads=transformer_config["num_attention_heads"],
            num_key_value_heads=transformer_config["num_key_value_heads"],
            head_dim=transformer_config.get(
                "head_dim", hidden_size // transformer_config["num_attention_heads"]
            ),
            max_position_embeddings=transformer_config["max_position_embeddings"],
            rms_norm_eps=transformer_config["rms_norm_eps"],
            rope_theta=transformer_config.get("rope_theta", 10000.0),
            attention_bias=transformer_config.get("attention_bias", False),
            target_vocab_size=transformer_config["vocab_size"],
            draft_vocab_size=raw_config["draft_vocab_size"],
            target_hidden_size=raw_config.get("target_hidden_size") or hidden_size,
            num_aux_hidden_states=num_aux_hidden_states,
            norm_before_residual=raw_config.get("norm_before_residual", True),
            torch_dtype=torch_dtype,
        )


@dataclass
class Eagle3KVCache:
    # The normalized state is used for sampling. The pre-final-norm state is
    # fed back to the next EAGLE step.
    last_hidden_state: torch.Tensor | None = None
    last_feedback_hidden_state: torch.Tensor | None = None

    def clone(self) -> "Eagle3KVCache":
        return Eagle3KVCache(
            self.last_hidden_state,
            self.last_feedback_hidden_state,
        )


class Eagle3MLP(Qwen3MLP):
    def __init__(self, config: Eagle3Config) -> None:
        super().__init__(config.hidden_size, config.intermediate_size, "silu")


class Eagle3DecoderLayer(nn.Module):
    def __init__(self, config: Eagle3Config) -> None:
        super().__init__()
        self.norm_before_residual = config.norm_before_residual
        self.hidden_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        # EAGLE3 uses the regular Qwen3 projections, RoPE, and attention. Only
        # the first layer's concatenated inputs and independent KV cache differ.
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=config.attention_bias,
            rope_theta=config.rope_theta,
            qkv_input_size=2 * config.hidden_size,
            use_qk_norm=False,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.mlp = Eagle3MLP(config)

    def forward(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if self.norm_before_residual:
            hidden_states = self.hidden_norm(hidden_states)
            residual = hidden_states
        else:
            residual = hidden_states
            hidden_states = self.hidden_norm(hidden_states)
        attention_input = torch.cat(
            (self.input_layernorm(input_embeds), hidden_states), dim=-1
        )
        hidden_states = residual + self.self_attn(positions, attention_input)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(hidden_states)

    def forward_batch(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if self.norm_before_residual:
            hidden_states = self.hidden_norm(hidden_states)
            residual = hidden_states
        else:
            residual = hidden_states
            hidden_states = self.hidden_norm(hidden_states)
        attention_input = torch.cat(
            (self.input_layernorm(input_embeds), hidden_states), dim=-1
        )
        hidden_states = residual + self.self_attn(positions, attention_input)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(hidden_states)


class Eagle3ForCausalLM(nn.Module):
    """Standalone EAGLE3 speculator, without target-model or KV-cache integration."""

    def __init__(self, config: Eagle3Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.target_vocab_size, config.hidden_size
        )
        self.fc = ReplicatedLinear(
            config.target_hidden_size * config.num_aux_hidden_states,
            config.hidden_size,
            bias=False,
        )
        self.layers = nn.ModuleList([Eagle3DecoderLayer(config)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = ParallelLMHead(config.draft_vocab_size, config.hidden_size)
        self.register_buffer("d2t", torch.zeros(config.draft_vocab_size, dtype=torch.long))
        self.register_buffer("t2d", torch.zeros(config.target_vocab_size, dtype=torch.bool))
        self._loaded_weight_names: set[str] = set()

    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def load_weight(self, name: str, loaded_weight: torch.Tensor) -> bool:
        self._loaded_weight_names.add(name)
        if name in {"d2t", "t2d"}:
            self.get_buffer(name).copy_(loaded_weight)
            return True
        return False

    def validate_loaded_weights(self) -> None:
        expected = set(self.state_dict())
        for packed_name, source_names in {
            "layers.0.self_attn.qkv_proj.weight": (
                "layers.0.self_attn.q_proj.weight",
                "layers.0.self_attn.k_proj.weight",
                "layers.0.self_attn.v_proj.weight",
            ),
            "layers.0.mlp.gate_up_proj.weight": (
                "layers.0.mlp.gate_proj.weight",
                "layers.0.mlp.up_proj.weight",
            ),
        }.items():
            if packed_name in expected:
                expected.remove(packed_name)
                expected.update(source_names)
        missing = expected - self._loaded_weight_names
        unexpected = self._loaded_weight_names - expected
        if missing or unexpected:
            raise ValueError(
                f"EAGLE3 checkpoint/model mismatch: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )

    def combine_hidden_states(self, target_hidden_states: torch.Tensor) -> torch.Tensor:
        expected_size = self.config.target_hidden_size * self.config.num_aux_hidden_states
        if target_hidden_states.ndim != 2 or target_hidden_states.size(-1) != expected_size:
            raise ValueError(
                "target_hidden_states must have shape "
                f"[num_tokens, {expected_size}]"
            )
        return self.fc(target_hidden_states)

    def forward_with_cache(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.ndim != 1 or positions.ndim != 1:
            raise ValueError("EAGLE3 forward expects 1D input_ids and positions")
        if input_ids.numel() != positions.numel():
            raise ValueError("input_ids and positions must contain the same number of tokens")
        if hidden_states.ndim != 2 or hidden_states.size(-1) != self.config.hidden_size:
            raise ValueError(
                "EAGLE3 feedback hidden_states must have shape "
                f"[num_tokens, {self.config.hidden_size}]"
            )
        input_embeds = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(input_embeds, hidden_states, positions)
        return self.norm(hidden_states), hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Standalone EAGLE3 forward using target auxiliary hidden states."""
        hidden_states = self.combine_hidden_states(target_hidden_states)
        logits_hidden_states, _ = self.forward_with_cache(
            input_ids, positions, hidden_states
        )
        return logits_hidden_states

    def compute_logits(
        self, hidden_states: torch.Tensor, map_to_target_vocab: bool = False
    ) -> torch.Tensor:
        # ParallelLMHead's forward selects prefill rows through the global
        # target context. EAGLE needs logits for every supplied draft row.
        logits = F.linear(hidden_states, self.lm_head.weight)
        if not map_to_target_vocab:
            return logits
        target_indices = torch.arange(
            self.config.draft_vocab_size, device=logits.device
        ) + self.d2t
        mapped_logits = logits.new_full(
            (logits.size(0), self.config.target_vocab_size), float("-inf")
        )
        mapped_logits[:, target_indices] = logits
        return mapped_logits

    def sample_greedy(self, hidden_states: torch.Tensor) -> torch.Tensor:
        draft_token_ids = F.linear(hidden_states, self.lm_head.weight).argmax(dim=-1)
        return draft_token_ids + self.d2t[draft_token_ids]


def load_eagle3_model(
    model_path: str | Path, device: str | torch.device = "cuda"
) -> Eagle3ForCausalLM:
    """Load an EAGLE3 checkpoint for standalone forward validation or integration."""

    config = Eagle3Config.from_pretrained(model_path)
    model = Eagle3ForCausalLM(config).to(device=device, dtype=config.torch_dtype)
    load_model(model, str(model_path))
    model.validate_loaded_weights()
    return model.eval()
