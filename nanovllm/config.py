import os
import json
import math
from dataclasses import dataclass, field
from transformers import AutoConfig

from nanovllm.layers.quantization import (
    DEFAULT_FP8_FORMAT,
    normalize_fp8_format,
    normalize_quantization,
)


@dataclass
class SpeculativeConfig:
    method: str
    model: str
    num_speculative_tokens: int = 8
    aux_hidden_state_layers: tuple[int, ...] = (2, 18, 33)
    # Kept for compatibility with vLLM-style speculative configuration.
    max_model_len: int | None = None

    def __post_init__(self):
        if self.method != "eagle3":
            raise ValueError("only the eagle3 speculative method is supported")
        if not os.path.isdir(self.model):
            raise ValueError(f"speculative model path does not exist: {self.model}")
        if not 1 <= self.num_speculative_tokens <= 8:
            raise ValueError("num_speculative_tokens must be between 1 and 8")
        if len(self.aux_hidden_state_layers) != 3:
            raise ValueError("EAGLE3 requires exactly three auxiliary hidden layers")


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    enable_prefill_batching: bool | None = None
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    data_parallel_size: int = 1
    data_parallel_rank: int = 0
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    # Optional assertion of the derived DP*TP size, never a separate axis.
    expert_parallel_size: int | None = None
    enable_expert_parallel: bool = False
    moe_shard_across_tp: bool = False
    moe_dispatch_backend: str = "replicated"
    moe_expert_capacity: int | None = None
    moe_expert_capacity_factor: float | None = None
    moe_expert_placement: str | None = None
    moe_dynamic_placement: bool = False
    moe_prefill_piece: bool = False
    moe_prefill_piece_capture_sizes: tuple[int, ...] = ()
    moe_placement_by_layer: dict[str, tuple[int, ...]] = field(
        default_factory=dict, init=False, repr=False
    )
    enforce_eager: bool = False
    enable_prefix_caching: bool = True
    master_addr: str = "127.0.0.1"
    master_port: int = 0
    run_id: str = ""
    device_offset: int = 0
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    # ``fp8`` means float8_e4m3fn storage.  This teaching implementation uses
    # one static K and V scale per attention layer, matching vLLM's fallback
    # when a checkpoint does not provide calibrated KV scales.
    kv_cache_dtype: str = "auto"
    speculative_config: SpeculativeConfig | dict | None = None
    quantization: str | None = None
    fp8_format: str = DEFAULT_FP8_FORMAT

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size == 256
        assert self.data_parallel_size >= 1
        assert 0 <= self.data_parallel_rank < self.data_parallel_size
        assert 1 <= self.tensor_parallel_size <= 8
        assert 1 <= self.pipeline_parallel_size <= 8
        if self.expert_parallel_size is not None:
            if not self.enable_expert_parallel:
                raise ValueError("expert_parallel_size requires enable_expert_parallel")
            if (type(self.expert_parallel_size) is not int
                    or self.expert_parallel_size != self.data_parallel_size * self.tensor_parallel_size):
                raise ValueError("EP must equal DP * TP; independent EP axes are not supported")
        if self.moe_shard_across_tp:
            raise ValueError("independent cross-TP EP sharding is removed; EP is always DP * TP")
        if self.moe_global_dp:
            if (
                self.pipeline_parallel_size != 1
                or self.moe_dynamic_placement
            ):
                raise ValueError(
                    "global DP MoE currently requires PP=1 and no expert migration"
                )
            if self.master_port == 0:
                raise ValueError("global DP MoE requires a shared master_port")
            if self.moe_dispatch_backend == "replicated":
                self.moe_dispatch_backend = "allgather_reduce"
            if self.moe_dispatch_backend not in (
                "allgather_reduce", "allgather_reducescatter"
            ):
                raise ValueError("global DP MoE requires allgather dispatch")
        if self.moe_dispatch_backend not in (
            "replicated", "all_to_all", "all_to_all_reduce",
            "allgather_reduce", "allgather_reducescatter",
        ):
            raise ValueError("unknown MoE dispatch backend")
        if self.moe_dispatch_backend in (
            "allgather_reduce", "allgather_reducescatter"
        ) and not self.moe_global_dp:
            raise ValueError("allgather dispatch requires global DP MoE")
        if self.moe_expert_capacity is not None and self.moe_expert_capacity < 1:
            raise ValueError("MoE expert capacity must be positive")
        if self.moe_expert_capacity_factor is not None:
            factor = self.moe_expert_capacity_factor
            if (isinstance(factor, bool) or not isinstance(factor, (int, float))
                    or not math.isfinite(factor) or factor <= 0):
                raise ValueError("MoE expert capacity factor must be finite and positive")
            if self.moe_expert_capacity is not None:
                raise ValueError("MoE fixed capacity and capacity factor are mutually exclusive")
        if self.kv_cache_dtype not in ("auto", "fp8"):
            raise ValueError("kv_cache_dtype must be 'auto' or 'fp8'")
        if self.pipeline_parallel_size > 1:
            assert self.enforce_eager, (
                "the teaching PP implementation currently requires eager mode"
            )
        assert self.device_offset >= 0
        assert self.max_num_batched_tokens > 0
        self.quantization = normalize_quantization(self.quantization)
        self.fp8_format = normalize_fp8_format(self.fp8_format)
        if isinstance(self.speculative_config, dict):
            self.speculative_config = SpeculativeConfig(**self.speculative_config)
        if self.speculative_config is not None:
            if not isinstance(self.speculative_config, SpeculativeConfig):
                raise TypeError("speculative_config must be a SpeculativeConfig or dict")
            if (
                self.data_parallel_size != 1
                or self.tensor_parallel_size != 1
                or self.pipeline_parallel_size != 1
            ):
                raise ValueError(
                    "EAGLE3 speculative decoding currently requires DP=TP=PP=1"
                )
            self.enable_prefix_caching = False
            if self.kv_cache_dtype == "fp8":
                raise ValueError(
                    "FP8 KV cache and EAGLE3 are not combined in the teaching "
                    "implementation"
                )
        self.hf_config = AutoConfig.from_pretrained(self.model)
        if self.enable_prefill_batching is None:
            self.enable_prefill_batching = self.hf_config.model_type == "qwen3_moe"
        capture_sizes = self.moe_prefill_piece_capture_sizes
        if self.moe_prefill_piece:
            if (
                not isinstance(capture_sizes, (tuple, list))
                or not capture_sizes
                or any(type(size) is not int or size < 1 for size in capture_sizes)
                or len(set(capture_sizes)) != len(capture_sizes)
            ):
                raise ValueError("MoE prefill piece capture sizes must be distinct positive integers")
            self.moe_prefill_piece_capture_sizes = tuple(sorted(capture_sizes))
        elif capture_sizes:
            raise ValueError("MoE prefill piece capture sizes require moe_prefill_piece")
        if self.moe_prefill_piece and (
            self.hf_config.model_type != "qwen3_moe"
            or self.pipeline_parallel_size != 1
            or self.enforce_eager
            or self.moe_dispatch_backend not in (
                "replicated", "allgather_reduce", "allgather_reducescatter"
            )
        ):
            raise ValueError(
                "MoE prefill piece requires Qwen3-MoE, PP=1, CUDA Graph, "
                "and graph-safe dispatch"
            )
        if self.moe_dynamic_placement and (
            self.hf_config.model_type != "qwen3_moe"
            or not self.enable_expert_parallel
            or self.effective_expert_parallel_size < 2
        ):
            raise ValueError(
                "dynamic expert placement requires EP>1"
            )
        if self.moe_expert_placement is not None:
            if self.hf_config.model_type != "qwen3_moe" or not self.enable_expert_parallel:
                raise ValueError("MoE placement requires expert parallelism")
            with open(self.moe_expert_placement) as file:
                placement = json.load(file)
            if not isinstance(placement, dict) or not isinstance(placement.get("layers"), dict):
                raise ValueError("MoE placement must contain a layers object")
            placement_model = placement.get("model")
            if (not isinstance(placement_model, str)
                    or not os.path.exists(placement_model)
                    or not os.path.samefile(placement_model, self.model)):
                raise ValueError("MoE placement model does not match checkpoint")
            if placement.get("num_experts") != self.hf_config.num_experts:
                raise ValueError("MoE placement expert count does not match checkpoint")
            if placement.get("expert_parallel_size") != self.effective_expert_parallel_size:
                raise ValueError("MoE placement EP size does not match configuration")
            sparse_step = self.hf_config.decoder_sparse_step
            moe_layers = {
                str(layer) for layer in range(self.hf_config.num_hidden_layers)
                if layer not in getattr(self.hf_config, "mlp_only_layers", [])
                and (layer + 1) % sparse_step == 0
            }
            if set(placement["layers"]) != moe_layers:
                raise ValueError("MoE placement must cover every MoE layer")
            experts_per_rank = self.hf_config.num_experts // self.effective_expert_parallel_size
            for layer, owners in placement["layers"].items():
                if (not isinstance(owners, list)
                        or len(owners) != self.hf_config.num_experts
                        or any(type(owner) is not int or not 0 <= owner < self.effective_expert_parallel_size
                               for owner in owners)
                        or any(owners.count(rank) != experts_per_rank
                               for rank in range(self.effective_expert_parallel_size))):
                    raise ValueError(f"invalid MoE placement for layer {layer}")
                self.moe_placement_by_layer[layer] = tuple(owners)
        if self.quantization is not None and self.hf_config.model_type != "qwen3":
            raise ValueError(
                "the teaching quantization implementation supports dense Qwen3 only"
            )
        if self.moe_global_dp and self.hf_config.model_type != "qwen3_moe":
            raise ValueError("global DP MoE requires a Qwen3-MoE checkpoint")
        if self.hf_config.model_type == "qwen3_moe":
            if (not self.enforce_eager
                    and self.moe_dispatch_backend not in (
                        "replicated", "allgather_reduce", "allgather_reducescatter"
                    )):
                raise ValueError(
                    "MoE CUDA Graph requires graph-safe dispatch"
                )
            assert (
                self.tensor_parallel_size == 1 or self.enable_expert_parallel
            ), (
                "Qwen3-MoE with TP>1 requires enable_expert_parallel; "
                "traditional TP-sharded experts are not implemented"
            )
        if self.enable_expert_parallel:
            assert self.hf_config.model_type == "qwen3_moe", (
                "enable_expert_parallel requires a qwen3_moe checkpoint"
            )
            assert (
                self.hf_config.num_experts % self.effective_expert_parallel_size == 0
            ), "num_experts must divide evenly across EP ranks"
        if self.moe_dispatch_backend in ("all_to_all", "all_to_all_reduce") and not self.enable_expert_parallel:
            raise ValueError("all_to_all MoE dispatch requires expert parallelism")
        if self.moe_expert_capacity is not None and self.hf_config.model_type != "qwen3_moe":
            raise ValueError("MoE expert capacity requires a qwen3_moe checkpoint")
        if self.moe_expert_capacity_factor is not None and self.hf_config.model_type != "qwen3_moe":
            raise ValueError("MoE expert capacity factor requires a qwen3_moe checkpoint")
        assert (
            self.hf_config.num_hidden_layers % self.pipeline_parallel_size == 0
        ), "model layers must divide evenly across pipeline stages"
        self.max_model_len = min(
            self.max_model_len, self.hf_config.max_position_embeddings
        )

    @property
    def model_parallel_size(self) -> int:
        return self.pipeline_parallel_size * self.tensor_parallel_size

    @property
    def moe_global_dp(self) -> bool:
        return self.enable_expert_parallel and self.data_parallel_size > 1

    @property
    def effective_expert_parallel_size(self) -> int:
        if not self.enable_expert_parallel:
            return 1
        return self.data_parallel_size * self.tensor_parallel_size

    @property
    def parallel_world_size(self) -> int:
        return self.data_parallel_size * self.model_parallel_size

    def global_rank(self, model_parallel_rank: int) -> int:
        if not 0 <= model_parallel_rank < self.model_parallel_size:
            raise ValueError("model_parallel_rank is out of range")
        return (
            self.data_parallel_rank * self.model_parallel_size
            + model_parallel_rank
        )

    def device_index(self, model_parallel_rank: int) -> int:
        return self.device_offset + self.global_rank(model_parallel_rank)
