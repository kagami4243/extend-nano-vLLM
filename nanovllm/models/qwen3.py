import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3Config

from nanovllm.distributed.parallel_state import get_tp_world_size
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.utils.context import get_context


class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
        qkv_input_size: int | None = None,
        use_qk_norm: bool | None = None,
        eager_qk_norm_prefill: bool = False,
    ) -> None:
        super().__init__()
        tp_size = get_tp_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias
        self.use_qk_norm = not qkv_bias if use_qk_norm is None else use_qk_norm
        self.eager_qk_norm_prefill = eager_qk_norm_prefill

        self.qkv_proj = QKVParallelLinear(
            qkv_input_size or hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def project_qkv(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project and rotate QKV for Qwen3-compatible attention variants."""
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if self.use_qk_norm:
            # Compiled BF16 Q/K norm changes MoE prefill routing on sensitive inputs.
            if (self.eager_qk_norm_prefill and get_context().is_prefill
                    and not self.q_norm.deterministic_cuda):
                q = self.q_norm.rms_forward_eager(q)
                k = self.k_norm.rms_forward_eager(k)
            else:
                q = self.q_norm(q)
                k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        return q, k, v

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        q, k, v = self.project_qkv(positions, hidden_states)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output

    def forward_prefill_pre(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the non-attention input side of a prefill attention block."""
        return self.project_qkv(positions, hidden_states)

    def forward_prefill_post(self, attention_output: torch.Tensor) -> torch.Tensor:
        """Run the non-attention output projection after eager attention."""
        return self.o_proj(attention_output.flatten(1, -1))


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
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
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
        pp_rank: int = 0,
        pp_size: int = 1,
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
        self.layers = nn.ModuleDict({
            str(layer_id): Qwen3DecoderLayer(config)
            for layer_id in range(self.start_layer, self.end_layer)
        })
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

    def forward_with_aux_hidden_states(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        aux_hidden_state_layers: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not (self.is_first_stage and self.is_last_stage):
            raise ValueError("EAGLE3 auxiliary hidden states require PP=1")
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        aux_hidden_states = []
        for layer_id, layer in enumerate(self.layers.values(), start=1):
            hidden_states, residual = layer(positions, hidden_states, residual)
            if layer_id in aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states + residual)
        if len(aux_hidden_states) != len(aux_hidden_state_layers):
            raise ValueError("requested EAGLE3 auxiliary layer is unavailable")
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states, torch.cat(aux_hidden_states, dim=-1)

    def forward_prefill_piece(self, input_ids, positions, piece_runner):
        """Prefill with eager attention and graphable surrounding operations.

        ``piece_runner`` owns CUDA-graph capture/replay. Keeping the attention
        call here makes the cache metadata and FlashAttention launch identical
        to the eager path while allowing every projection, norm, activation,
        embedding, and final norm to run from static graph buffers.
        """
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
        hidden_states, _ = piece_runner("norm", self.norm, hidden_states, residual)
        return hidden_states

    def forward_prefill_piece_with_aux(
        self, input_ids, positions, piece_runner, aux_hidden_state_layers
    ):
        hidden_states = piece_runner("embed", self.embed_tokens, input_ids)
        residual = None
        aux_hidden_states = []
        for layer_id, layer in enumerate(self.layers.values(), start=1):
            q, k, v, residual = piece_runner(
                "pre", layer, positions, hidden_states, residual
            )
            attention_output = piece_runner(
                "attention", layer.self_attn.attn, q, k, v
            )
            hidden_states, residual = piece_runner(
                "post", layer, attention_output, residual
            )
            if layer_id in aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states + residual)
        hidden_states, _ = piece_runner("norm", self.norm, hidden_states, residual)
        if len(aux_hidden_states) != len(aux_hidden_state_layers):
            raise ValueError("requested EAGLE3 auxiliary layer is unavailable")
        return hidden_states, torch.cat(aux_hidden_states, dim=-1)


class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config,
        pp_rank: int = 0,
        pp_size: int = 1,
        ep_rank: int = 0,
        ep_size: int = 1,
    ) -> None:
        super().__init__()
        if ep_rank != 0 or ep_size != 1:
            raise ValueError("dense Qwen3 does not support expert parallelism")
        self.model = Qwen3Model(config, pp_rank, pp_size)
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

    def forward_with_aux_hidden_states(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        aux_hidden_state_layers: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.forward_with_aux_hidden_states(
            input_ids, positions, aux_hidden_state_layers
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        assert self.lm_head is not None
        return self.lm_head(hidden_states)

    def compute_all_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return logits for every token during speculative verification.

        ParallelLMHead intentionally keeps only the final prefill token for the
        normal scheduler path. EAGLE3 verification needs one target logit per
        proposed token and is limited to TP=1 by SpeculativeConfig.
        """
        assert self.lm_head is not None
        assert get_tp_world_size() == 1
        return F.linear(hidden_states, self.lm_head.weight)
