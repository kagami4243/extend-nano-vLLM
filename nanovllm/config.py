import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    data_parallel_size: int = 1
    data_parallel_rank: int = 0
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    enable_expert_parallel: bool = False
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

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size == 256
        assert self.data_parallel_size >= 1
        assert 0 <= self.data_parallel_rank < self.data_parallel_size
        assert 1 <= self.tensor_parallel_size <= 8
        assert 1 <= self.pipeline_parallel_size <= 8
        if self.pipeline_parallel_size > 1:
            assert self.enforce_eager, (
                "the teaching PP implementation currently requires eager mode"
            )
        if self.enable_expert_parallel:
            assert self.enforce_eager, (
                "the teaching EP implementation currently requires eager mode"
            )
        assert self.device_offset >= 0
        assert self.max_num_batched_tokens > 0
        self.hf_config = AutoConfig.from_pretrained(self.model)
        if self.hf_config.model_type == "qwen3_moe":
            assert self.enforce_eager, (
                "the teaching Qwen3-MoE implementation requires eager mode"
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
                self.tensor_parallel_size == 1
                or self.pipeline_parallel_size == 1
            ), "combined PP and expert parallelism are not implemented"
            assert (
                self.hf_config.num_experts % self.tensor_parallel_size == 0
            ), "num_experts must divide evenly across TP/EP ranks"
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
