import gc
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
from nanovllm.layers.quantization import quantize_model
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
from nanovllm.models.eagle3 import Eagle3KVCache, load_eagle3_model


class _BreakablePrefillCapture:
    """Capture graph segments separated by eager attention calls."""

    def __init__(self, pool=None, stream=None):
        self.pool = pool
        self.stream = stream or torch.cuda.Stream()
        self.segments = []
        self.buffers = []
        self._graph = None

    def _begin_graph(self):
        if self._graph is not None:
            return
        graph = torch.cuda.CUDAGraph()
        if self.pool is None:
            graph.capture_begin()
        else:
            graph.capture_begin(pool=self.pool)
        self._graph = graph

    def _end_graph(self):
        if self._graph is None:
            return
        self._graph.capture_end()
        if self.pool is None:
            self.pool = self._graph.pool()
        self.segments.append(self._graph.replay)
        self._graph = None

    def ensure_graph(self):
        """Start a graph only when a graphable piece is about to run."""
        self._begin_graph()

    def capture(self, fn):
        caller_stream = torch.cuda.current_stream()
        self.stream.wait_stream(caller_stream)
        with torch.cuda.stream(self.stream):
            self._begin_graph()
            try:
                output = fn()
            finally:
                self._end_graph()
        caller_stream.wait_stream(self.stream)
        return output

    def add_eager(self, fn, *args):
        self._end_graph()
        eager_output = fn(*args)
        output = torch.empty_like(eager_output)
        output.copy_(eager_output)
        self.buffers.append((args, output))

        def replay_eager():
            output.copy_(fn(*args))

        self.segments.append(replay_eager)
        return output

    def replay(self):
        for segment in self.segments:
            segment()


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
        # Keep checkpoint loading and TP shard assembly unchanged, then replace
        # only dense LinearBase weights with the selected teaching format.
        # This mirrors vLLM's process_weights_after_loading phase.
        quantize_model(
            self.model,
            config.quantization,
            fp8_format=config.fp8_format,
        )
        self.speculative_config = config.speculative_config
        self.eagle3_model = None
        # EAGLE stores only its recurrent hidden state here. Its physical KV
        # pages are bound to attention layers below and reuse each sequence's
        # block table, so scheduler rollback has one logical source of truth.
        self.eagle3_states: dict[int, Eagle3KVCache] = {}
        self.eagle_graphs = {}
        self.eagle_graph_vars = {}
        self.prefill_piece_enabled = False
        self.prefill_piece_graphs = {}
        self.prefill_piece_graph_pool = None
        self.prefill_piece_capture_stream = torch.cuda.Stream()
        if self.speculative_config is not None:
            if hf_config.model_type != "qwen3":
                raise ValueError("EAGLE3 speculative decoding currently requires Qwen3")
            self.eagle3_model = load_eagle3_model(self.speculative_config.model)
            eagle_config = self.eagle3_model.config
            if (
                eagle_config.target_hidden_size != hf_config.hidden_size
                or eagle_config.target_vocab_size != hf_config.vocab_size
            ):
                raise ValueError("EAGLE3 checkpoint is incompatible with the target Qwen3")
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
                self.prefill_piece_graphs.clear()
                self.eagle_graphs.clear()
                self.eagle_graph_vars.clear()
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
        kv_cache_torch_dtype = (
            torch.float8_e4m3fn
            if config.kv_cache_dtype == "fp8"
            else hf_config.torch_dtype
        )
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
            * kv_cache_torch_dtype.itemsize
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
            dtype=kv_cache_torch_dtype,
        )
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
        assert layer_id == num_local_layers
        if self.eagle3_model is not None:
            # Target and draft attention need separate values, but must use
            # identical logical block IDs. This lets the scheduler reserve and
            # roll back both caches by changing the one sequence block table.
            eagle_layers = sum(
                1 for module in self.eagle3_model.modules()
                if hasattr(module, "k_cache") and hasattr(module, "v_cache")
            )
            eagle_config = self.eagle3_model.config
            self.eagle_kv_cache = torch.empty(
                2,
                eagle_layers,
                config.num_kvcache_blocks,
                self.block_size,
                eagle_config.num_key_value_heads,
                eagle_config.head_dim,
                dtype=eagle_config.torch_dtype,
                device="cuda",
            )
            layer_id = 0
            for module in self.eagle3_model.modules():
                if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                    module.k_cache = self.eagle_kv_cache[0, layer_id]
                    module.v_cache = self.eagle_kv_cache[1, layer_id]
                    layer_id += 1
            assert layer_id == eagle_layers

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
            "kv_cache_dtype": str(self.kv_cache.dtype),
            "kv_cache_blocks": self.kv_cache.size(2),
            "kv_cache_bytes": self.kv_cache.numel() * self.kv_cache.element_size(),
            "pipeline_send_count": self.pipeline_send_count,
            "pipeline_recv_count": self.pipeline_recv_count,
            "fp8_format": self.config.fp8_format,
            "prefill_piece_enabled": self.prefill_piece_enabled,
            "eagle3_runtime": self.get_eagle3_runtime_capabilities(),
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
        if any(seq.block_table for seq in seqs):
            if not all(seq.block_table for seq in seqs):
                raise RuntimeError("mixed allocated and unallocated KV-cache batches")
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
        cu_seqlens_q = torch.arange(
            len(seqs) + 1, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        cu_seqlens_k = torch.zeros(
            len(seqs) + 1, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        cu_seqlens_k[1:] = torch.cumsum(context_lens, dim=0)
        block_tables = self.prepare_block_tables(seqs)
        # Use the paged varlen kernel for decode as well as prefill.  This
        # keeps the attention computation identical when speculative verify
        # submits multiple query rows.
        set_context(
            True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=1,
            max_seqlen_k=int(context_lens.max().item()),
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if (
            is_prefill
            and self.prefill_piece_enabled
            and not self.enforce_eager
            and hasattr(getattr(self.model, "model", None), "forward_prefill_piece")
        ):
            return self.model.compute_logits(self.run_prefill_piece(input_ids, positions))
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

    def prefill_piece_cache_key(self, token_count: int) -> int:
        if token_count <= 0:
            raise ValueError("prefill piece token count must be positive")
        return int(token_count)

    @staticmethod
    def _run_prefill_piece(kind, module, *inputs):
        if kind == "pre":
            return module.prefill_pre(*inputs)
        if kind == "post":
            return module.prefill_post(*inputs)
        return module(*inputs)

    def _capture_prefill_piece_graph(
        self,
        token_count: int,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        aux_hidden_state_layers=None,
    ):
        """Capture one forward as graph segments separated by eager attention."""
        variants = self.prefill_piece_graphs.setdefault(token_count, {})
        variant_key = (
            None
            if aux_hidden_state_layers is None
            else tuple(aux_hidden_state_layers)
        )
        if variant_key in variants:
            return variants[variant_key]
        if self.enforce_eager:
            return None
        model = self.model.model
        device = torch.device("cuda", self.device_index)
        static_input_ids = torch.empty(token_count, dtype=torch.int64, device=device)
        static_positions = torch.empty(token_count, dtype=torch.int64, device=device)
        static_input_ids.copy_(input_ids)
        static_positions.copy_(positions)

        def eager_piece_runner(kind, module, *inputs):
            return self._run_prefill_piece(kind, module, *inputs)

        def forward(piece_runner):
            if aux_hidden_state_layers is None:
                return model.forward_prefill_piece(
                    static_input_ids, static_positions, piece_runner
                )
            return model.forward_prefill_piece_with_aux(
                static_input_ids,
                static_positions,
                piece_runner,
                aux_hidden_state_layers,
            )

        # Compile kernels and establish allocator state before stream capture.
        forward(eager_piece_runner)
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        capture = _BreakablePrefillCapture(
            self.prefill_piece_graph_pool, self.prefill_piece_capture_stream
        )

        def capture_piece_runner(kind, module, *inputs):
            if kind == "attention":
                return capture.add_eager(module, *inputs)
            capture.ensure_graph()
            return self._run_prefill_piece(kind, module, *inputs)

        output = capture.capture(lambda: forward(capture_piece_runner))
        self.prefill_piece_graph_pool = capture.pool
        entry = {
            "capture": capture,
            "input_ids": static_input_ids,
            "positions": static_positions,
            "output": output,
        }
        variants[variant_key] = entry
        return entry

    def run_prefill_piece(
        self, input_ids: torch.Tensor, positions: torch.Tensor,
        aux_hidden_state_layers=None,
    ):
        token_count = input_ids.numel()
        entry = self._capture_prefill_piece_graph(
            self.prefill_piece_cache_key(token_count),
            input_ids,
            positions,
            aux_hidden_state_layers,
        )
        if entry is None:
            if aux_hidden_state_layers is None:
                return self.model(input_ids, positions)
            return self.model.forward_with_aux_hidden_states(
                input_ids, positions, aux_hidden_state_layers
            )
        entry["input_ids"].copy_(input_ids)
        entry["positions"].copy_(positions)
        entry["capture"].replay()
        return entry["output"]

    def get_eagle3_runtime_capabilities(self):
        return {
            "paged_kv_cache": bool(
                getattr(self, "eagle3_model", None) is not None
                and getattr(self, "eagle_kv_cache", None) is not None
            ),
            "cuda_graph": bool(getattr(self, "eagle_graphs", {})),
            "graph_batch_sizes": sorted(getattr(self, "eagle_graphs", {})),
        }

    @torch.inference_mode()
    def run_target_with_aux(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        *,
        all_token_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # EAGLE3 conditions its draft model on selected target-layer states.
        # Return logits and auxiliary states from one target pass so proposal
        # verification does not require a second target forward.
        assert self.speculative_config is not None
        if self.prefill_piece_enabled and not self.enforce_eager:
            hidden_states, aux_hidden_states = self.run_prefill_piece(
                input_ids, positions,
                self.speculative_config.aux_hidden_state_layers,
            )
        else:
            hidden_states, aux_hidden_states = self.model.forward_with_aux_hidden_states(
                input_ids,
                positions,
                self.speculative_config.aux_hidden_state_layers,
            )
        if all_token_logits:
            return self.model.compute_all_logits(hidden_states), aux_hidden_states
        return self.model.compute_logits(hidden_states), aux_hidden_states

    @torch.inference_mode()
    def initialize_eagle3_state(
        self,
        seq: Sequence,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        aux_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
    ) -> None:
        assert self.eagle3_model is not None
        state = Eagle3KVCache()
        # EAGLE3 is conditioned on the target's next token and the current
        # target auxiliary state. This is the same one-token shift vLLM uses.
        eagle_input_ids = input_ids.clone()
        if input_ids.numel() > 1:
            eagle_input_ids[:-1] = input_ids[1:]
        eagle_input_ids[-1] = next_token_ids.reshape(-1)[-1]
        combined = self.eagle3_model.combine_hidden_states(aux_hidden_states)
        draft_hidden_states, feedback_hidden_states = (
            self.eagle3_model.forward_with_cache(
                eagle_input_ids, positions, combined
            )
        )
        state.last_hidden_state = draft_hidden_states[-1:]
        state.last_feedback_hidden_state = feedback_hidden_states[-1:]
        self.eagle3_states[seq.seq_id] = state

    @torch.inference_mode()
    def verify_eagle3_batch(
        self, seqs: list[Sequence], num_input_tokens: list[int]
    ) -> tuple[list[list[int]], list[torch.Tensor]]:
        """Verify each previous target token plus its draft in one batch.

        The first row is the uncomputed tail token already present in the
        sequence; each following row checks one proposed token. Keeping all
        rows in one paged-varlen target forward preserves target-only greedy
        semantics while avoiding per-request verification launches.
        """
        if len(seqs) != len(num_input_tokens):
            raise ValueError("EAGLE3 verification metadata does not match requests")
        try:
            for seq, num_tokens in zip(seqs, num_input_tokens):
                seq.num_scheduled_tokens = num_tokens
            input_ids, positions = self.prepare_prefill(seqs)
            target_logits, aux_hidden_states = self.run_target_with_aux(
                input_ids, positions, all_token_logits=True
            )
            offsets = [0]
            for num_tokens in num_input_tokens:
                offsets.append(offsets[-1] + num_tokens)
            target_tokens = []
            target_aux = []
            for start, end in zip(offsets, offsets[1:]):
                target_tokens.append(target_logits[start:end].argmax(dim=-1).tolist())
                target_aux.append(aux_hidden_states[start:end])
            return target_tokens, target_aux
        finally:
            for seq in seqs:
                seq.num_scheduled_tokens = 0
            reset_context()

    @torch.inference_mode()
    def _set_eagle3_decode_context(self, seqs: list[Sequence], step: int) -> None:
        """Set paged-cache metadata for one EAGLE proposal step."""
        block_tables = self.prepare_block_tables(seqs)
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            # Query RoPE position is len(seq)-1+step, but the new K/V is
            # appended after the existing len(seq) cached tokens.
            position = len(seq) + step
            block_index, block_offset = divmod(position, self.block_size)
            slot_mapping.append(
                seq.block_table[block_index] * self.block_size + block_offset
            )
            # Attention writes the current K/V before invoking the cache
            # kernel, so the read length includes that newly written slot.
            context_lens.append(len(seq) + step + 1)
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        context_lens = torch.tensor(
            context_lens, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )

    def _set_eagle3_commit_context(
        self, seq: Sequence, slot_position: int, total_length: int
    ) -> None:
        """Set paged-cache metadata for one accepted EAGLE token."""
        block_index, block_offset = divmod(slot_position, self.block_size)
        set_context(
            False,
            slot_mapping=torch.tensor(
                [seq.block_table[block_index] * self.block_size + block_offset],
                dtype=torch.int32, device="cuda",
            ),
            context_lens=torch.tensor(
                [total_length], dtype=torch.int32, device="cuda"
            ),
            block_tables=self.prepare_block_tables([seq]),
        )

    def _commit_eagle3_tokens(
        self,
        seq: Sequence,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        combined: torch.Tensor,
        state: Eagle3KVCache,
        start_position: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replay the accepted/replacement tail into the draft paged cache.

        Rejected proposal pages may contain speculative K/V, so the draft
        recurrent state must be rebuilt from the target-approved tail before
        the next proposal can start from the same logical sequence.
        """
        hidden = feedback = None
        prefix_length = start_position + 1
        for index in range(input_ids.numel()):
            self._set_eagle3_commit_context(
                seq,
                prefix_length + index,
                prefix_length + index + 1,
            )
            hidden, feedback = self.eagle3_model.forward_with_cache(
                input_ids[index:index + 1],
                positions[index:index + 1],
                combined[index:index + 1],
            )
        assert hidden is not None and feedback is not None
        return hidden, feedback

    @torch.inference_mode()
    def propose_eagle3_batch(self, seqs: list[Sequence]) -> list[list[int]]:
        """Generate greedy EAGLE proposals against the draft paged KV cache.

        Proposals are batched to share draft launches. Their physical pages
        were reserved by the scheduler before this method because each draft
        step writes K/V before target verification accepts or rejects it.
        """
        assert self.eagle3_model is not None
        assert self.speculative_config is not None
        try:
            limits = [
                min(
                    self.speculative_config.num_speculative_tokens,
                    seq.max_tokens - seq.num_completion_tokens,
                )
                for seq in seqs
            ]
            proposals = [[] for _ in seqs]
            hidden = torch.cat(
                [self.eagle3_states[s.seq_id].last_hidden_state for s in seqs],
                dim=0,
            )
            feedback = torch.cat(
                [
                    self.eagle3_states[s.seq_id].last_feedback_hidden_state
                    for s in seqs
                ],
                dim=0,
            )
            next_token = self.eagle3_model.sample_greedy(hidden)
            max_steps = max(limits, default=0)
            for step in range(max_steps):
                for row, token in enumerate(next_token.tolist()):
                    if step < limits[row]:
                        proposals[row].append(token)
                if step + 1 == max_steps:
                    break
                positions = torch.tensor(
                    [len(seq) - 1 + step for seq in seqs],
                    dtype=torch.int64,
                    device="cuda",
                )
                self._set_eagle3_decode_context(seqs, step)
                if self.eagle_graphs:
                    hidden, feedback = self.run_eagle3_graph(
                        next_token, positions, feedback, seqs
                    )
                else:
                    hidden, feedback = self.eagle3_model.forward_with_cache(
                        next_token, positions, feedback
                    )
                next_token = self.eagle3_model.sample_greedy(hidden)
            return proposals
        finally:
            reset_context()

    @torch.inference_mode()
    def commit_eagle3(
        self,
        seq: Sequence,
        output_tokens: list[int],
        target_aux_hidden_states: torch.Tensor,
        start_position: int,
    ) -> None:
        if not output_tokens:
            return
        assert self.eagle3_model is not None
        if seq.is_finished:
            return
        state = self.eagle3_states[seq.seq_id]
        input_ids = torch.tensor(output_tokens, dtype=torch.int64, device="cuda")
        positions = torch.arange(
            start_position,
            start_position + len(output_tokens),
            dtype=torch.int64,
            device="cuda",
        )
        combined = self.eagle3_model.combine_hidden_states(
            # Verification rows are [h(previous), h(accepted_1), ...].
            # The output token at each position consumes the row immediately
            # before it; this prefix therefore aligns replacement with its
            # accepted predecessor rather than with the rejected draft.
            target_aux_hidden_states[:len(output_tokens)]
        )
        draft_hidden_states, feedback_hidden_states = (
            self._commit_eagle3_tokens(
                seq, input_ids, positions, combined, state, start_position
            )
        )
        state.last_hidden_state = draft_hidden_states[-1:]
        state.last_feedback_hidden_state = feedback_hidden_states[-1:]

    def release_eagle3_state(self, seq_id: int) -> None:
        """Drop recurrent draft state when the scheduler finishes a request."""
        self.eagle3_states.pop(seq_id, None)

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
            if self.speculative_config is not None:
                if not is_prefill:
                    raise RuntimeError("EAGLE3 target runner received an unsupported decode")
                context = get_context()
                logits, aux_hidden_states = self.run_target_with_aux(
                    input_ids, positions, all_token_logits=True
                )
                # The target prefill is flattened across requests. Initialize
                # each request's independent EAGLE KV/state from its own
                # contiguous rows, then sample only its final target row.
                last_rows = context.cu_seqlens_q[1:] - 1
                sampled_logits = logits[last_rows]
                for index, seq in enumerate(seqs):
                    start = int(context.cu_seqlens_q[index].item())
                    end = int(context.cu_seqlens_q[index + 1].item())
                    self.initialize_eagle3_state(
                        seq,
                        input_ids[start:end],
                        positions[start:end],
                        aux_hidden_states[start:end],
                        sampled_logits[index].argmax().view(1),
                    )
                logits = sampled_logits
            else:
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
        self.graph_bs = [size for size in (1, 2, 4, 8) if size <= max_bs]
        self.graph_bs += list(range(16, max_bs + 1, 16))
        if max_bs not in self.graph_bs:
            self.graph_bs.append(max_bs)
        self.graph_bs = sorted(set(self.graph_bs))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            # Capture a valid one-token decode path. Dynamic metadata is copied
            # into these stable buffers before every replay.
            context_lens[:bs].fill_(1)
            set_context(
                False,
                max_seqlen_q=1,
                max_seqlen_k=1,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )
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
        self.prefill_piece_enabled = bool(
            not self.enforce_eager
            and self.pp_size == 1
            and self.config.hf_config.model_type == "qwen3"
            and hasattr(self.model.model, "forward_prefill_piece")
            # Rowwise torch._scaled_mm is unsafe in the breakable prefill
            # capture on the supported Torch/CUDA stack. Decode still uses
            # the complete CUDA graph captured above.
            and not (
                self.config.quantization == "fp8"
                and self.config.fp8_format == "per_token"
            )
        )
        if self.eagle3_model is not None:
            self._capture_eagle3_graphs(max_num_blocks)

    @torch.inference_mode()
    def _capture_eagle3_graphs(self, max_num_blocks: int):
        """Capture one recurrent EAGLE step per supported batch size.

        The captured attention still reads the draft model's paged cache. Only
        metadata and recurrent inputs are copied into static graph buffers at
        replay time, so scheduler block IDs remain authoritative.
        """
        max_bs = min(self.config.max_num_seqs, 512)
        graph_bs = [size for size in (1, 2, 4, 8) if size <= max_bs]
        graph_bs += list(range(16, max_bs + 1, 16))
        if max_bs not in graph_bs:
            graph_bs.append(max_bs)
        graph_bs = sorted(set(graph_bs))
        hidden_size = self.eagle3_model.config.hidden_size
        dtype = self.eagle3_model.config.torch_dtype
        self.eagle_graph_pool = None
        for bs in reversed(graph_bs):
            input_ids = torch.zeros(bs, dtype=torch.int64, device="cuda")
            positions = torch.zeros(bs, dtype=torch.int64, device="cuda")
            feedback = torch.zeros(bs, hidden_size, dtype=dtype, device="cuda")
            slot_mapping = torch.full((bs,), -1, dtype=torch.int32, device="cuda")
            # Keep padded rows on a valid one-token attention path during
            # capture. Their slot is -1 and their outputs are discarded.
            context_lens = torch.ones(bs, dtype=torch.int32, device="cuda")
            block_tables = torch.zeros(
                bs, max_num_blocks, dtype=torch.int32, device="cuda"
            )
            set_context(
                False,
                slot_mapping=slot_mapping,
                context_lens=context_lens,
                block_tables=block_tables,
            )
            graph = torch.cuda.CUDAGraph()
            self.eagle3_model.forward_with_cache(input_ids, positions, feedback)
            with torch.cuda.graph(graph, self.eagle_graph_pool):
                outputs = self.eagle3_model.forward_with_cache(
                    input_ids, positions, feedback
                )
            if self.eagle_graph_pool is None:
                self.eagle_graph_pool = graph.pool()
            self.eagle_graphs[bs] = graph
            self.eagle_graph_vars[bs] = {
                "input_ids": input_ids,
                "positions": positions,
                "feedback": feedback,
                "slot_mapping": slot_mapping,
                "context_lens": context_lens,
                "block_tables": block_tables,
                "outputs": outputs,
            }
            reset_context()

    @torch.inference_mode()
    def run_eagle3_graph(self, input_ids, positions, feedback, seqs):
        bs = len(seqs)
        graph_bs = next(size for size in sorted(self.eagle_graphs) if size >= bs)
        graph = self.eagle_graphs[graph_bs]
        variables = self.eagle_graph_vars[graph_bs]
        variables["input_ids"].zero_()
        variables["positions"].zero_()
        variables["feedback"].zero_()
        variables["slot_mapping"].fill_(-1)
        variables["context_lens"].fill_(1)
        variables["block_tables"].zero_()
        variables["input_ids"][:bs].copy_(input_ids)
        variables["positions"][:bs].copy_(positions)
        variables["feedback"][:bs].copy_(feedback)
        context = get_context()
        variables["slot_mapping"][:bs].copy_(context.slot_mapping)
        variables["context_lens"][:bs].copy_(context.context_lens)
        variables["block_tables"][:bs, :context.block_tables.size(1)].copy_(
            context.block_tables
        )
        graph.replay()
        hidden, next_feedback = variables["outputs"]
        return hidden[:bs], next_feedback[:bs]
