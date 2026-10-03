import re
import os

import torch
from torch import nn
from transformers import Qwen3MoeConfig

from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.moe import ExpertParallelMoE
from nanovllm.models.qwen3 import Qwen3Attention, Qwen3MLP


_EXPERT_WEIGHT_PATTERN = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)
_LAYER_WEIGHT_PATTERN = re.compile(r"^model\.layers\.(\d+)\.")


class Qwen3MoeDecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3MoeConfig,
        layer_id: int,
        ep_rank: int,
        ep_size: int,
        dispatch_backend: str,
        expert_capacity: int | None,
        graph_safe_decode: bool,
        expert_owners: tuple[int, ...] | None,
        expert_capacity_factor: float | None = None,
        dynamic_placement: bool = False,
    ) -> None:
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError("Qwen3-MoE currently supports only silu")
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
            eager_qk_norm_prefill=ep_size == 1,
        )
        if (self.self_attn.use_qk_norm
                and os.environ.get("NANOVLLM_EXPERIMENTAL_DETERMINISTIC_QK_NORM") == "1"):
            self.self_attn.q_norm.deterministic_cuda = True
            self.self_attn.k_norm.deterministic_cuda = True
        uses_moe = (
            layer_id not in getattr(config, "mlp_only_layers", [])
            and config.num_experts > 0
            and (layer_id + 1) % config.decoder_sparse_step == 0
        )
        if uses_moe:
            self.mlp = ExpertParallelMoE(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                num_experts=config.num_experts,
                top_k=config.num_experts_per_tok,
                norm_topk_prob=config.norm_topk_prob,
                ep_rank=ep_rank,
                ep_size=ep_size,
                dispatch_backend=dispatch_backend,
                expert_capacity=expert_capacity,
                graph_safe_decode=graph_safe_decode,
                expert_owners=expert_owners,
                expert_capacity_factor=expert_capacity_factor,
                shared_expert_intermediate_size=getattr(
                    config, "shared_expert_intermediate_size", None
                ),
                shared_expert_tp_sharded=getattr(
                    config, "shared_expert_tp_sharded", False
                ),
                dynamic_placement=dynamic_placement,
            )
        else:
            self.mlp = Qwen3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
            )
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = (
                self.input_layernorm(hidden_states),
                hidden_states,
            )
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual
            )
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def prefill_pre(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        q, k, v = self.self_attn.forward_prefill_pre(positions, hidden_states)
        return q, k, v, residual

    def prefill_post(
        self, attention_output: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.self_attn.forward_prefill_post(attention_output)
        return self.post_attention_layernorm(hidden_states, residual)


class Qwen3MoeModel(nn.Module):

    def __init__(
        self,
        config: Qwen3MoeConfig,
        pp_rank: int = 0,
        pp_size: int = 1,
        ep_rank: int = 0,
        ep_size: int = 1,
        dispatch_backend: str = "replicated",
        expert_capacity: int | None = None,
        graph_safe_decode: bool = False,
        expert_placement: dict[str, tuple[int, ...]] | None = None,
        expert_capacity_factor: float | None = None,
        dynamic_placement: bool = False,
    ) -> None:
        super().__init__()
        layers_per_stage = config.num_hidden_layers // pp_size
        self.start_layer = pp_rank * layers_per_stage
        self.end_layer = self.start_layer + layers_per_stage
        self.is_first_stage = pp_rank == 0
        self.is_last_stage = pp_rank == pp_size - 1
        self.embed_tokens = (
            VocabParallelEmbedding(config.vocab_size, config.hidden_size)
            if self.is_first_stage
            else None
        )
        self.layers = nn.ModuleDict(
            {
                str(layer_id): Qwen3MoeDecoderLayer(
                    config, layer_id, ep_rank, ep_size,
                    dispatch_backend, expert_capacity, graph_safe_decode,
                    None if expert_placement is None else expert_placement.get(str(layer_id)),
                    expert_capacity_factor,
                    dynamic_placement,
                )
                for layer_id in range(self.start_layer, self.end_layer)
            }
        )
        self.norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if self.is_last_stage
            else None
        )

    def forward_stage(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.is_first_stage:
            assert hidden_states is None and residual is None
            hidden_states = self.embed_tokens(input_ids)
        else:
            assert hidden_states is not None and residual is not None
        for layer in self.layers.values():
            hidden_states, residual = layer(positions, hidden_states, residual)
        if self.is_last_stage:
            hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states, residual

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        assert self.is_first_stage and self.is_last_stage
        hidden_states, _ = self.forward_stage(input_ids, positions)
        return hidden_states

    def forward_prefill_piece(self, input_ids, positions, piece_runner):
        hidden_states = piece_runner("embed", self.embed_tokens, input_ids)
        residual = None
        for layer in self.layers.values():
            q, k, v, residual = piece_runner(
                "pre", layer, positions, hidden_states, residual
            )
            attention_output = piece_runner(
                "attention", layer.self_attn.attn, q, k, v
            )
            hidden_states, residual = piece_runner(
                "post", layer, attention_output, residual
            )
            kind = "moe" if isinstance(layer.mlp, ExpertParallelMoE) else "mlp"
            hidden_states = piece_runner(kind, layer.mlp, hidden_states)
        hidden_states, _ = piece_runner("norm", self.norm, hidden_states, residual)
        return hidden_states


class Qwen3MoeForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        ".mlp.gate_proj": (".mlp.gate_up_proj", 0),
        ".mlp.up_proj": (".mlp.gate_up_proj", 1),
        ".mlp.shared_expert.gate_proj": (
            ".mlp.shared_expert.gate_up_proj", 0
        ),
        ".mlp.shared_expert.up_proj": (
            ".mlp.shared_expert.gate_up_proj", 1
        ),
    }

    def __init__(
        self,
        config: Qwen3MoeConfig,
        pp_rank: int = 0,
        pp_size: int = 1,
        ep_rank: int = 0,
        ep_size: int = 1,
        dispatch_backend: str = "replicated",
        expert_capacity: int | None = None,
        graph_safe_decode: bool = False,
        expert_placement: dict[str, tuple[int, ...]] | None = None,
        expert_capacity_factor: float | None = None,
        dynamic_placement: bool = False,
    ) -> None:
        super().__init__()
        self.model = Qwen3MoeModel(
            config, pp_rank, pp_size, ep_rank, ep_size,
            dispatch_backend, expert_capacity, graph_safe_decode,
            expert_placement, expert_capacity_factor, dynamic_placement,
        )
        self.tie_word_embeddings = config.tie_word_embeddings
        self.lm_head = (
            ParallelLMHead(config.vocab_size, config.hidden_size)
            if self.model.is_last_stage
            else None
        )
        if (
            config.tie_word_embeddings
            and self.model.is_first_stage
            and self.model.is_last_stage
        ):
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def _expert_weight_target(
        self, weight_name: str
    ) -> tuple[ExpertParallelMoE, int, str] | None:
        match = _EXPERT_WEIGHT_PATTERN.match(weight_name)
        if match is None:
            return None
        layer_id, expert_id, projection = match.groups()
        if layer_id not in self.model.layers:
            return None
        layer = self.model.layers[layer_id]
        if not isinstance(layer.mlp, ExpertParallelMoE):
            return None
        return layer.mlp, int(expert_id), projection

    def should_load_weight(self, weight_name: str) -> bool:
        layer_match = _LAYER_WEIGHT_PATTERN.match(weight_name)
        if (
            layer_match is not None
            and layer_match.group(1) not in self.model.layers
        ):
            return False
        if weight_name == "model.embed_tokens.weight":
            return self.model.is_first_stage or (
                self.tie_word_embeddings and self.model.is_last_stage
            )
        if weight_name in {"model.norm.weight", "lm_head.weight"}:
            return self.model.is_last_stage
        target = self._expert_weight_target(weight_name)
        if target is None:
            return True
        moe, expert_id, _ = target
        return moe.owns_expert(expert_id)

    def load_weight(
        self,
        weight_name: str,
        loaded_weight: torch.Tensor,
    ) -> bool:
        target = self._expert_weight_target(weight_name)
        if target is None:
            return False
        moe, expert_id, projection = target
        moe.load_expert_weight(expert_id, projection, loaded_weight)
        return True

    def forward_stage(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.model.forward_stage(
            input_ids, positions, hidden_states, residual
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.lm_head is not None
        return self.lm_head(hidden_states)

    def get_expert_diagnostics(self) -> dict[str, int]:
        moe_layers = [
            layer.mlp
            for layer in self.model.layers.values()
            if isinstance(layer.mlp, ExpertParallelMoE)
        ]
        if not moe_layers:
            return {}
        first = moe_layers[0]
        return {
            "expert_start": first.expert_start,
            "expert_end": first.expert_end,
            "local_expert_ids": list(first.local_expert_ids),
            "num_local_experts": first.num_local_experts,
            "expert_parameter_bytes": sum(
                layer.expert_parameter_bytes for layer in moe_layers
            ),
            "ep_all_reduce_count": sum(
                layer.ep_all_reduce_count for layer in moe_layers
            ),
            "ep_reduce_scatter_count": sum(
                layer.ep_reduce_scatter_count for layer in moe_layers
            ),
            "ep_all_gather_count": sum(
                layer.ep_all_gather_count for layer in moe_layers
            ),
            "ep_all_to_all_count": sum(
                layer.ep_all_to_all_count for layer in moe_layers
            ),
            "ep_broadcast_count": sum(
                layer.ep_broadcast_count for layer in moe_layers
            ),
            "dispatch_assignment_count": sum(
                layer.dispatch_assignment_count for layer in moe_layers
            ),
            "return_assignment_count": sum(
                layer.return_assignment_count for layer in moe_layers
            ),
            "total_assignment_count": sum(
                layer.total_assignment_count for layer in moe_layers
            ),
            "total_dispatch_token_count": sum(
                layer.total_dispatch_token_count for layer in moe_layers
            ),
            "dropped_assignment_count": sum(
                layer.dropped_assignment_count for layer in moe_layers
            ),
        }
