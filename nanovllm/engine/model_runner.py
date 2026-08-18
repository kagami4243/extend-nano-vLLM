import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.distributed.parallel_state import (
    destroy_model_parallel,
    get_ep_rank,
    get_ep_world_size,
    get_pp_group_ranks,
    get_pp_last_rank,
    get_pp_next_rank,
    get_pp_prev_rank,
    get_pp_rank,
    get_pp_world_size,
    get_tp_group_ranks,
    get_tp_rank,
    get_tp_world_size,
    initialize_model_parallel,
)
from nanovllm.engine.sequence import Sequence
from nanovllm.models.registry import get_model_class
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.model_parallel_size
        self.rank = rank
        self.global_rank = config.global_rank(rank)
        self.device_index = config.device_index(rank)
        self.event = event
        self._exited = False
        self.shm_name = f"nanovllm_{config.run_id}"

        torch.cuda.set_device(self.device_index)
        init_method = f"tcp://{config.master_addr}:{config.master_port}"
        dist.init_process_group(
            "nccl", init_method, world_size=self.world_size, rank=rank
        )
        initialize_model_parallel(
            config.tensor_parallel_size,
            config.pipeline_parallel_size,
            config.enable_expert_parallel,
        )
        self.pp_rank = get_pp_rank()
        self.pp_size = get_pp_world_size()
        self.is_first_pp_stage = self.pp_rank == 0
        self.is_last_pp_stage = self.pp_rank == self.pp_size - 1
        self.ep_rank = get_ep_rank()
        self.ep_size = get_ep_world_size()
        self.pipeline_send_count = 0
        self.pipeline_recv_count = 0
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        model_class = get_model_class(hf_config)
        self.model = model_class(
            hf_config,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            ep_rank=self.ep_rank,
            ep_size=self.ep_size,
        )
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(
                    name=self.shm_name, create=True, size=2**20
                )
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name=self.shm_name)
                self.loop()

    def exit(self):
        if self._exited:
            return
        self._exited = True
        try:
            if self.world_size > 1:
                self.shm.close()
                dist.barrier()
                if self.rank == 0:
                    try:
                        self.shm.unlink()
                    except FileNotFoundError:
                        pass
            if not self.enforce_eager:
                del self.graphs, self.graph_pool
            torch.cuda.synchronize()
        finally:
            destroy_model_parallel()
            if dist.is_initialized():
                dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        if n + 4 > len(self.shm.buf):
            raise ValueError(
                f"shared-memory command is too large: {n + 4} > {len(self.shm.buf)}"
            )
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        warmup_len = min(
            self.config.max_num_batched_tokens, self.config.max_model_len
        )
        num_seqs = max(
            1,
            min(
                self.config.max_num_batched_tokens // warmup_len,
                self.config.max_num_seqs,
            ),
        )
        seqs = [Sequence([0] * warmup_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // get_tp_world_size()
        head_dim = getattr(
            hf_config,
            "head_dim",
            hf_config.hidden_size // hf_config.num_attention_heads,
        )
        num_local_layers = self.model.model.end_layer - self.model.model.start_layer
        block_bytes = (
            2
            * num_local_layers
            * self.block_size
            * num_kv_heads
            * head_dim
            * hf_config.torch_dtype.itemsize
        )
        local_num_blocks = (
            int(total * config.gpu_memory_utilization - used - peak + current)
            // block_bytes
        )
        num_blocks = torch.tensor(local_num_blocks, dtype=torch.int64, device="cuda")
        if self.world_size > 1:
            dist.all_reduce(num_blocks, op=dist.ReduceOp.MIN)
        config.num_kvcache_blocks = num_blocks.item()
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(
            2,
            num_local_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
        )
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
        assert layer_id == num_local_layers

    def get_diagnostics(self):
        local = {
            "rank": self.rank,
            "global_rank": self.global_rank,
            "device": torch.cuda.current_device(),
            "device_index": self.device_index,
            "dp_rank": self.config.data_parallel_rank,
            "dp_size": self.config.data_parallel_size,
            "tp_rank": get_tp_rank(),
            "tp_size": get_tp_world_size(),
            "tp_group_ranks": [
                self.config.global_rank(rank)
                for rank in get_tp_group_ranks()
            ],
            "ep_rank": self.ep_rank,
            "ep_size": self.ep_size,
            "pp_rank": self.pp_rank,
            "pp_size": self.pp_size,
            "pp_group_ranks": [
                self.config.global_rank(rank)
                for rank in get_pp_group_ranks()
            ],
            "start_layer": self.model.model.start_layer,
            "end_layer": self.model.model.end_layer,
            "num_local_layers": (
                self.model.model.end_layer - self.model.model.start_layer
            ),
            "is_first_stage": self.is_first_pp_stage,
            "is_last_stage": self.is_last_pp_stage,
            "parameter_bytes": sum(
                parameter.numel() * parameter.element_size()
                for parameter in self.model.parameters()
            ),
            "kv_cache_layers": self.kv_cache.size(1),
            "pipeline_send_count": self.pipeline_send_count,
            "pipeline_recv_count": self.pipeline_recv_count,
        }
        expert_diagnostics = getattr(self.model, "get_expert_diagnostics", None)
        if expert_diagnostics is not None:
            local.update(expert_diagnostics())
        diagnostics = [None] * self.world_size
        dist.all_gather_object(diagnostics, local)
        return diagnostics if self.rank == 0 else None

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_computed_tokens
            scheduled = seq.num_scheduled_tokens or seq.num_tokens
            end = min(start + scheduled, seq.num_tokens)
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            seqlen_q = end - start
            seqlen_k = end
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            for token_position in range(start, end):
                block_index = token_position // self.block_size
                block_offset = token_position % self.block_size
                slot_mapping.append(
                    seq.block_table[block_index] * self.block_size + block_offset
                )
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    @torch.inference_mode()
    def run_pipeline_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        temperatures: torch.Tensor | None,
        num_seqs: int,
    ) -> list[int] | None:
        hidden_shape = (input_ids.numel(), self.config.hf_config.hidden_size)
        hidden_states = residual = None
        if not self.is_first_pp_stage:
            previous_rank = get_pp_prev_rank()
            assert previous_rank is not None
            hidden_states = torch.empty(
                hidden_shape,
                dtype=self.config.hf_config.torch_dtype,
                device="cuda",
            )
            residual = torch.empty_like(hidden_states)
            dist.recv(hidden_states, src=previous_rank)
            dist.recv(residual, src=previous_rank)
            self.pipeline_recv_count += 2

        hidden_states, residual = self.model.forward_stage(
            input_ids, positions, hidden_states, residual
        )

        if not self.is_last_pp_stage:
            next_rank = get_pp_next_rank()
            assert next_rank is not None
            dist.send(hidden_states.contiguous(), dst=next_rank)
            dist.send(residual.contiguous(), dst=next_rank)
            self.pipeline_send_count += 2
            if self.rank == 0:
                token_ids = torch.empty(
                    num_seqs, dtype=torch.int64, device="cuda"
                )
                dist.recv(token_ids, src=get_pp_last_rank())
                self.pipeline_recv_count += 1
                return token_ids.tolist()
            return None

        logits = self.model.compute_logits(hidden_states)
        if get_tp_rank() != 0:
            return None
        token_ids = self.sampler(logits, temperatures)
        if self.rank != 0:
            dist.send(token_ids, dst=0)
            self.pipeline_send_count += 1
            return None
        return token_ids.tolist()

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        try:
            input_ids, positions = (
                self.prepare_prefill(seqs)
                if is_prefill
                else self.prepare_decode(seqs)
            )
            should_sample = (
                self.is_last_pp_stage
                and get_tp_rank() == 0
                and self.ep_rank == 0
            )
            temperatures = self.prepare_sample(seqs) if should_sample else None
            if self.pp_size > 1:
                return self.run_pipeline_model(
                    input_ids, positions, temperatures, len(seqs)
                )
            logits = self.run_model(input_ids, positions, is_prefill)
            return (
                self.sampler(logits, temperatures).tolist()
                if should_sample
                else None
            )
        finally:
            reset_context()

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
