import os
import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # Keep the CPU/reference path usable for documentation tests.
    _TRITON_AVAILABLE = False

from nanovllm.distributed.parallel_state import (
    get_ep_group, get_ep_group_ranks, get_moe_group, get_tp_group,
    get_tp_world_size,
)
from nanovllm.layers.linear import (
    MergedColumnParallelLinear, ReplicatedLinear, RowParallelLinear,
)
from nanovllm.utils.context import get_context


# The kernel organization below is adapted from vLLM's
# vllm/model_executor/layers/fused_moe/fused_moe.py (Apache-2.0), revision
# 1a308c449.  It is deliberately a small unquantized subset: one local EP
# shard, BF16/FP16 weights, and two grouped GEMMs for a SwiGLU expert.
#
# vLLM normally uses its CUDA moe_align_block_size op to construct these
# arrays.  nano-vLLM makes the same expert-major, BLOCK_M-aligned layout with
# PyTorch GPU tensor operations so that this implementation stays self
# contained and easy to read.
_MOE_BLOCK_M = 16
MOE_MAX_GRAPH_TOKENS_PER_RANK = 512
_PACKED_ALLTOALL_MAX_TOKENS = 32


def reduce_fp64_to_bf16(
    local_output: torch.Tensor,
    group: dist.ProcessGroup,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    flat = local_output.contiguous().view(-1)
    world_size = dist.get_world_size(group)
    if flat.dtype != torch.float64 or flat.numel() % world_size:
        raise ValueError("FP64 EP output must divide evenly across ranks")
    reduced = torch.empty(
        flat.numel() // world_size, device=flat.device, dtype=torch.float64
    )
    dist.reduce_scatter_tensor(reduced, flat, group=group)
    gathered = torch.empty(
        flat.numel(), device=flat.device, dtype=output_dtype
    )
    dist.all_gather_into_tensor(gathered, reduced.to(output_dtype), group=group)
    return gathered.view_as(local_output)


if _TRITON_AVAILABLE:

    @triton.jit
    def _grouped_moe_gemm_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        row_ids_ptr,
        expert_ids_ptr,
        num_tokens_post_padded_ptr,
        routing_weights_ptr,
        num_input_rows,
        stride_am,
        stride_ak,
        stride_be,
        stride_bn,
        stride_bk,
        stride_cm,
        stride_cn,
        N: tl.constexpr,
        K: tl.constexpr,
        APPLY_ROUTING_WEIGHT: tl.constexpr,
        ROUTE_LAYOUT: tl.constexpr,
        INPUT_TOP_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """One expert-major tile per program, as in vLLM fused_moe_kernel.

        ``row_ids`` contains input row IDs, or flat route IDs in route layout.
        Route layout writes into token-route order and skips inactive tiles
        using a device-side length. ``expert_ids`` selects each tile's expert.
        """
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        if ROUTE_LAYOUT:
            if pid_m * BLOCK_M >= tl.load(num_tokens_post_padded_ptr):
                return
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_ids = tl.load(row_ids_ptr + offs_m)
        input_rows = row_ids // INPUT_TOP_K
        row_mask = input_rows < num_input_rows
        output_rows = row_ids if ROUTE_LAYOUT else offs_m
        expert_id = tl.load(expert_ids_ptr + pid_m)

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            a = tl.load(
                a_ptr
                + input_rows[:, None] * stride_am
                + offs_k[None, :] * stride_ak,
                mask=row_mask[:, None] & (offs_k[None, :] < K),
                other=0.0,
            )
            b = tl.load(
                b_ptr
                + expert_id * stride_be
                + offs_n[None, :] * stride_bn
                + offs_k[:, None] * stride_bk,
                mask=(offs_n[None, :] < N) & (offs_k[:, None] < K),
                other=0.0,
            )
            accumulator += tl.dot(a, b)

        if APPLY_ROUTING_WEIGHT:
            routing_weight = tl.load(
                routing_weights_ptr + output_rows, mask=row_mask, other=0.0
            )
            accumulator *= routing_weight[:, None]

        tl.store(
            c_ptr + output_rows[:, None] * stride_cm + offs_n[None, :] * stride_cn,
            accumulator,
            mask=row_mask[:, None] & (offs_n[None, :] < N),
        )


    @triton.jit
    def _map_expert_routes_kernel(
        routes_ptr,
        local_map_ptr,
        mask_ptr,
        local_ids_ptr,
        num_routes,
        NUM_EXPERTS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        expert_ids = tl.load(routes_ptr + offsets, offsets < num_routes, other=-1)
        valid = (expert_ids >= 0) & (expert_ids < NUM_EXPERTS)
        safe_ids = tl.minimum(tl.maximum(expert_ids, 0), NUM_EXPERTS - 1)
        local_ids = tl.load(local_map_ptr + safe_ids)
        tl.store(mask_ptr + offsets, valid & (local_ids >= 0), offsets < num_routes)
        tl.store(
            local_ids_ptr + offsets, tl.where(valid, local_ids, -1),
            offsets < num_routes,
        )


    @triton.jit
    def _combine_expert_routes_kernel(
        expert_output_ptr,
        route_rows_ptr,
        output_ptr,
        HIDDEN: tl.constexpr,
        TOP_K: tl.constexpr,
        num_rows,
        BLOCK_H: tl.constexpr,
        USE_FP64: tl.constexpr,
    ):
        token = tl.program_id(0)
        columns = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        if USE_FP64:
            total = tl.full((BLOCK_H,), 0, tl.float64)
        else:
            total = tl.full((BLOCK_H,), 0, tl.float32)
        for route in range(TOP_K):
            row = tl.load(route_rows_ptr + token * TOP_K + route)
            values = tl.load(
                expert_output_ptr + row * HIDDEN + columns,
                mask=(row < num_rows) & (columns < HIDDEN),
                other=0,
            )
            if USE_FP64:
                total += values.to(tl.float64)
            else:
                total += values.to(tl.float32)
        tl.store(output_ptr + token * HIDDEN + columns, total, columns < HIDDEN)


class SharedMergedReplicatedLinear(ReplicatedLinear):

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        self.intermediate_size = intermediate_size
        super().__init__(hidden_size, 2 * intermediate_size, bias=False)

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor,
        loaded_shard_id: int,
    ) -> None:
        target = param.data.narrow(
            0, loaded_shard_id * self.intermediate_size,
            self.intermediate_size,
        )
        target.copy_(loaded_weight)


class SharedExpertMLP(nn.Module):

    def __init__(
        self, hidden_size: int, intermediate_size: int,
        tp_sharded: bool = False,
    ) -> None:
        super().__init__()
        self.intermediate_size = intermediate_size
        self.gate_up_proj = (
            MergedColumnParallelLinear(
                hidden_size, [intermediate_size, intermediate_size], bias=False
            ) if tp_sharded else
            SharedMergedReplicatedLinear(hidden_size, intermediate_size)
        )
        self.down_proj = (
            RowParallelLinear(intermediate_size, hidden_size, bias=False)
            if tp_sharded else
            ReplicatedLinear(intermediate_size, hidden_size, bias=False)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class ExpertParallelMoE(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        norm_topk_prob: bool,
        ep_rank: int = 0,
        ep_size: int = 1,
        dispatch_backend: str = "replicated",
        expert_capacity: int | None = None,
        graph_safe_decode: bool = False,
        expert_owners: tuple[int, ...] | list[int] | None = None,
        expert_capacity_factor: float | None = None,
        shared_expert_intermediate_size: int | None = None,
        shared_expert_tp_sharded: bool = False,
        dynamic_placement: bool = False,
    ) -> None:
        super().__init__()
        if num_experts % ep_size != 0:
            raise ValueError("num_experts must divide evenly across EP ranks")
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        if dispatch_backend not in (
            "replicated", "all_to_all", "all_to_all_reduce",
            "allgather_reduce", "allgather_reducescatter",
        ):
            raise ValueError("unknown MoE dispatch backend")
        if expert_capacity is not None and expert_capacity < 1:
            raise ValueError("expert capacity must be positive")
        if expert_capacity_factor is not None:
            if (isinstance(expert_capacity_factor, bool)
                    or not isinstance(expert_capacity_factor, (int, float))
                    or not math.isfinite(expert_capacity_factor)
                    or expert_capacity_factor <= 0):
                raise ValueError("expert capacity factor must be finite and positive")
            if expert_capacity is not None:
                raise ValueError("fixed capacity and capacity factor are mutually exclusive")
        if graph_safe_decode and dispatch_backend not in (
            "replicated", "allgather_reduce", "allgather_reducescatter"
        ):
            raise ValueError("graph-safe MoE decode requires graph-safe dispatch")
        if (shared_expert_intermediate_size is not None
                and shared_expert_intermediate_size < 1):
            raise ValueError("shared expert intermediate size must be positive")

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.dispatch_backend = dispatch_backend
        self.fp64_combine = (
            dispatch_backend == "replicated"
            and os.environ.get("NANOVLLM_EXPERIMENTAL_FP64_MOE_COMBINE") == "1"
        )
        self.fp64_staged_reduce = (
            self.fp64_combine and ep_size > 1
            and os.environ.get("NANOVLLM_EXPERIMENTAL_FP64_STAGED_REDUCE") == "1"
        )
        self.fp32_reduce = (
            ep_size > 1 and dispatch_backend == "replicated"
            and not self.fp64_combine
            and os.environ.get("NANOVLLM_EXPERIMENTAL_FP32_EP_REDUCE") == "1"
        )
        self.expert_capacity = expert_capacity
        self.expert_capacity_factor = expert_capacity_factor
        self.graph_safe_decode = graph_safe_decode
        self.dynamic_placement = dynamic_placement
        self.num_local_experts = num_experts // ep_size
        if expert_owners is None and dynamic_placement:
            expert_owners = tuple(
                expert // self.num_local_experts for expert in range(num_experts)
            )
        self.expert_owners = (
            tuple(expert_owners) if expert_owners is not None else tuple(
                expert // self.num_local_experts for expert in range(num_experts)
            )
        )
        if expert_owners is None:
            self.expert_start = ep_rank * self.num_local_experts
            self.expert_end = self.expert_start + self.num_local_experts
            self.local_expert_ids = tuple(range(self.expert_start, self.expert_end))
            self.expert_id_to_local = None
            self.register_buffer("expert_owner_tensor", None, persistent=False)
            self.register_buffer("expert_local_index_tensor", None, persistent=False)
        else:
            if (len(expert_owners) != num_experts
                    or any(type(owner) is not int or not 0 <= owner < ep_size
                           for owner in expert_owners)
                    or any(expert_owners.count(rank) != self.num_local_experts
                           for rank in range(ep_size))):
                raise ValueError("expert placement must assign equal experts per EP rank")
            self.expert_start = -1
            self.expert_end = -1
            self.local_expert_ids = tuple(
                expert for expert, owner in enumerate(expert_owners)
                if owner == ep_rank
            )
            self.expert_id_to_local = {
                expert: index for index, expert in enumerate(self.local_expert_ids)
            }
            local_indices = [
                self.expert_id_to_local.get(expert, -1)
                for expert in range(num_experts)
            ]
            self.register_buffer(
                "expert_owner_tensor",
                torch.tensor(expert_owners, dtype=torch.int64),
                persistent=False,
            )
            self.register_buffer(
                "expert_local_index_tensor",
                torch.tensor(local_indices, dtype=torch.int64),
                persistent=False,
            )

        self.gate = ReplicatedLinear(hidden_size, num_experts, bias=False)
        self.gate_up_proj = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                2 * intermediate_size,
                hidden_size,
            )
        )
        self.down_proj = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                hidden_size,
                intermediate_size,
            )
        )
        self.shared_expert = (
            SharedExpertMLP(
                hidden_size, shared_expert_intermediate_size,
                shared_expert_tp_sharded,
            )
            if shared_expert_intermediate_size is not None else None
        )
        self.shared_expert_gate = (
            ReplicatedLinear(hidden_size, 1, bias=False)
            if shared_expert_intermediate_size is not None else None
        )
        self.ep_all_reduce_count = 0
        self.ep_reduce_scatter_count = 0
        self.ep_all_gather_count = 0
        self.ep_all_to_all_count = 0
        self.ep_broadcast_count = 0
        self.dispatch_assignment_count = 0
        self.return_assignment_count = 0
        self.total_assignment_count = 0
        self.total_dispatch_token_count = 0
        self.dropped_assignment_count = 0

    @property
    def expert_parameter_bytes(self) -> int:
        return sum(
            parameter.numel() * parameter.element_size()
            for parameter in (self.gate_up_proj, self.down_proj)
        )

    def owns_expert(self, expert_id: int) -> bool:
        if self.expert_id_to_local is not None:
            return expert_id in self.expert_id_to_local
        return self.expert_start <= expert_id < self.expert_end

    @torch.no_grad()
    def relocate_experts(self, expert_owners: tuple[int, ...] | list[int]) -> int:
        """Move expert weights at a quiescent point without changing graph storage."""
        if not self.dynamic_placement:
            raise RuntimeError("runtime expert placement was not enabled")
        owners = tuple(expert_owners)
        if (len(owners) != self.num_experts
                or any(type(owner) is not int or not 0 <= owner < self.ep_size
                       for owner in owners)
                or any(owners.count(rank) != self.num_local_experts
                       for rank in range(self.ep_size))):
            raise ValueError("expert placement must assign equal experts per EP rank")
        group = get_moe_group()
        if self.ep_size > 1:
            proposals = [None] * self.ep_size
            dist.all_gather_object(proposals, owners, group=group)
            if any(proposal != owners for proposal in proposals):
                raise ValueError("all EP ranks must request the same expert placement")
        if owners == self.expert_owners:
            return 0

        old = self.expert_owners
        new_ids = tuple(
            expert for expert, owner in enumerate(owners)
            if owner == self.ep_rank
        )
        new_local = {expert: index for index, expert in enumerate(new_ids)}

        def exchange(parameter: torch.Tensor) -> torch.Tensor:
            row_size = parameter[0].numel()
            send_ids = [
                expert for dest in range(self.ep_size)
                for expert in self.local_expert_ids if owners[expert] == dest
            ]
            recv_ids = [
                expert for source in range(self.ep_size)
                for expert in range(self.num_experts)
                if old[expert] == source and owners[expert] == self.ep_rank
            ]
            send_counts = [
                sum(owners[expert] == dest for expert in self.local_expert_ids)
                for dest in range(self.ep_size)
            ]
            recv_counts = [
                sum(old[expert] == source for expert in new_ids)
                for source in range(self.ep_size)
            ]
            send = parameter[
                [self.expert_id_to_local[expert] for expert in send_ids]
            ].contiguous().view(-1)
            recv = torch.empty_like(send)
            if self.ep_size > 1:
                dist.all_to_all_single(
                    recv, send,
                    [count * row_size for count in recv_counts],
                    [count * row_size for count in send_counts],
                    group=group,
                )
            else:
                recv.copy_(send)
            reordered = torch.empty_like(parameter)
            received = recv.view(self.num_local_experts, *parameter.shape[1:])
            for row, expert in enumerate(recv_ids):
                reordered[new_local[expert]].copy_(received[row])
            return reordered

        gate_up = exchange(self.gate_up_proj)
        down = exchange(self.down_proj)
        self.gate_up_proj.copy_(gate_up)
        self.down_proj.copy_(down)
        self.expert_owner_tensor.copy_(
            torch.tensor(owners, device=self.expert_owner_tensor.device)
        )
        self.expert_local_index_tensor.copy_(
            torch.tensor(
                [new_local.get(expert, -1) for expert in range(self.num_experts)],
                device=self.expert_local_index_tensor.device,
            )
        )
        self.local_expert_ids = new_ids
        self.expert_id_to_local = new_local
        self.expert_owners = owners
        return sum(before != after for before, after in zip(old, owners))

    def _local_routes(
        self, flat_experts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.expert_id_to_local is None:
            local_mask = (flat_experts >= self.expert_start) & (
                flat_experts < self.expert_end
            )
            return local_mask, flat_experts - self.expert_start
        if _TRITON_AVAILABLE and flat_experts.is_cuda and flat_experts.numel():
            local_mask = torch.empty_like(flat_experts, dtype=torch.bool)
            local_indices = torch.empty_like(flat_experts)
            _map_expert_routes_kernel[(triton.cdiv(flat_experts.numel(), 256),)](
                flat_experts,
                self.expert_local_index_tensor,
                local_mask,
                local_indices,
                flat_experts.numel(),
                NUM_EXPERTS=self.num_experts,
                BLOCK=256,
            )
            return local_mask, local_indices
        valid = (flat_experts >= 0) & (flat_experts < self.num_experts)
        safe_experts = flat_experts.clamp(0, self.num_experts - 1)
        local_mask = valid & (
            self.expert_owner_tensor[safe_experts] == self.ep_rank
        )
        return local_mask, self.expert_local_index_tensor[safe_experts]

    def load_expert_weight(
        self,
        expert_id: int,
        projection: str,
        loaded_weight: torch.Tensor,
    ) -> None:
        if not self.owns_expert(expert_id):
            return
        local_expert_id = (
            expert_id - self.expert_start
            if self.expert_id_to_local is None
            else self.expert_id_to_local[expert_id]
        )
        if projection == "gate_proj":
            target = self.gate_up_proj[
                local_expert_id, : self.intermediate_size
            ]
        elif projection == "up_proj":
            target = self.gate_up_proj[
                local_expert_id, self.intermediate_size :
            ]
        elif projection == "down_proj":
            target = self.down_proj[local_expert_id]
        else:
            raise ValueError(f"unknown expert projection: {projection}")
        target.data.copy_(loaded_weight)

    def _forward_reference(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Original per-expert implementation, kept as a correctness fallback."""
        output = torch.zeros_like(hidden_states)
        local_assignments = 0
        for expert_id in selected_experts.unique().tolist():
            if not self.owns_expert(expert_id):
                continue
            token_indices, route_indices = torch.where(
                selected_experts == expert_id
            )
            local_assignments += token_indices.numel()
            local_expert_id = (
                expert_id - self.expert_start
                if self.expert_id_to_local is None
                else self.expert_id_to_local[expert_id]
            )
            expert_input = hidden_states[token_indices]
            gate_up = F.linear(
                expert_input, self.gate_up_proj[local_expert_id]
            )
            gate, up = gate_up.chunk(2, dim=-1)
            expert_output = F.linear(
                F.silu(gate) * up,
                self.down_proj[local_expert_id],
            )
            expert_output *= routing_weights[
                token_indices, route_indices, None
            ]
            output.index_add_(0, token_indices, expert_output)
        return output, local_assignments

    def _can_use_triton_kernel(self, hidden_states: torch.Tensor) -> bool:
        return (
            _TRITON_AVAILABLE
            and hidden_states.is_cuda
            and hidden_states.dtype in (torch.float16, torch.bfloat16)
            and self.gate_up_proj.dtype == hidden_states.dtype
            and self.down_proj.dtype == hidden_states.dtype
        )

    def _run_grouped_gemm(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        row_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor | None = None,
        *,
        num_tokens_post_padded: torch.Tensor | None = None,
        input_top_k: int = 1,
        num_output_rows: int | None = None,
    ) -> torch.Tensor:
        """Run B[E, N, K] against rows of A in the vLLM packed MoE layout."""
        packed_rows = row_ids.numel()
        output_size = b.size(1)
        output = torch.zeros(
            (packed_rows if num_output_rows is None else num_output_rows, output_size),
            device=a.device, dtype=a.dtype,
        )
        grid = lambda meta: (
            triton.cdiv(packed_rows, meta["BLOCK_M"]),
            triton.cdiv(output_size, meta["BLOCK_N"]),
        )
        _grouped_moe_gemm_kernel[grid](
            a,
            b,
            output,
            row_ids,
            expert_ids,
            num_tokens_post_padded if num_tokens_post_padded is not None else row_ids,
            routing_weights if routing_weights is not None else output,
            a.size(0),
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            b.stride(2),
            output.stride(0),
            output.stride(1),
            N=output_size,
            K=a.size(1),
            APPLY_ROUTING_WEIGHT=routing_weights is not None,
            ROUTE_LAYOUT=num_tokens_post_padded is not None,
            INPUT_TOP_K=input_top_k,
            BLOCK_M=_MOE_BLOCK_M,
            BLOCK_N=64,
            BLOCK_K=32,
            num_warps=4,
        )
        return output

    def _combine_expert_routes(
        self,
        expert_output: torch.Tensor,
        route_rows: torch.Tensor,
        num_tokens: int,
        top_k: int,
    ) -> torch.Tensor:
        output_dtype = (
            torch.float64 if self.fp64_combine else
            torch.float32 if self.fp32_reduce else expert_output.dtype
        )
        if not _TRITON_AVAILABLE or not expert_output.is_cuda:
            padded = torch.cat((
                expert_output,
                expert_output.new_zeros((1, self.hidden_size)),
            ))
            return (
                padded[route_rows.long()]
                .reshape(num_tokens, top_k, self.hidden_size)
                .to(torch.float64 if self.fp64_combine else torch.float32)
                .sum(dim=1).to(output_dtype)
            )
        output = torch.empty(
            (num_tokens, self.hidden_size), device=expert_output.device,
            dtype=output_dtype,
        )
        _combine_expert_routes_kernel[
            (num_tokens, triton.cdiv(self.hidden_size, 128))
        ](
            expert_output, route_rows, output,
            HIDDEN=self.hidden_size,
            TOP_K=top_k,
            num_rows=expert_output.size(0),
            BLOCK_H=128,
            USE_FP64=self.fp64_combine,
        )
        return output

    def _forward_triton(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Run local experts for replicated or dispatched token assignments."""
        num_tokens = hidden_states.size(0)
        flat_experts = selected_experts.reshape(-1)
        flat_tokens = torch.arange(
            num_tokens, device=hidden_states.device, dtype=torch.int32
        ).repeat_interleave(selected_experts.size(1))
        flat_weights = routing_weights.reshape(-1)
        local_mask, flat_local_experts = self._local_routes(flat_experts)
        local_assignments = int(local_mask.sum().item())
        if local_assignments == 0:
            return torch.zeros_like(hidden_states), 0

        # Sort expert-major, then pad each expert's rows to the M tile size.
        # This is the layout consumed by vLLM's fused_moe_kernel as well.
        local_experts = flat_local_experts[local_mask]
        local_tokens = flat_tokens[local_mask]
        local_weights = flat_weights[local_mask]
        if selected_experts.size(1) > 1:
            local_routes = torch.arange(
                flat_experts.numel(), device=hidden_states.device
            )[local_mask]
        order = torch.argsort(local_experts, stable=True)
        local_experts = local_experts[order]
        local_tokens = local_tokens[order]
        local_weights = local_weights[order]
        if selected_experts.size(1) > 1:
            local_routes = local_routes[order]
        expert_counts = torch.bincount(
            local_experts, minlength=self.num_local_experts
        )
        padded_counts = (
            (expert_counts + _MOE_BLOCK_M - 1) // _MOE_BLOCK_M * _MOE_BLOCK_M
        )
        packed_rows = int(padded_counts.sum().item())
        expert_starts = torch.cumsum(expert_counts, dim=0) - expert_counts
        packed_starts = torch.cumsum(padded_counts, dim=0) - padded_counts
        within_expert = torch.arange(
            local_assignments, device=hidden_states.device
        ) - expert_starts[local_experts]
        packed_positions = packed_starts[local_experts] + within_expert

        row_ids = torch.full(
            (packed_rows,), num_tokens, device=hidden_states.device, dtype=torch.int32
        )
        packed_weights = torch.zeros(
            (packed_rows,), device=hidden_states.device, dtype=hidden_states.dtype
        )
        row_ids.scatter_(0, packed_positions, local_tokens)
        packed_weights.scatter_(0, packed_positions, local_weights)
        expert_ids = torch.repeat_interleave(
            torch.arange(
                self.num_local_experts, device=hidden_states.device, dtype=torch.int32
            ),
            padded_counts // _MOE_BLOCK_M,
        )

        gate_up = self._run_grouped_gemm(
            hidden_states, self.gate_up_proj, row_ids, expert_ids
        )
        gate, up = gate_up.chunk(2, dim=-1)
        activated = F.silu(gate) * up
        packed_row_ids = torch.arange(
            packed_rows, device=hidden_states.device, dtype=torch.int32
        )
        expert_output = self._run_grouped_gemm(
            activated,
            self.down_proj,
            packed_row_ids,
            expert_ids,
            packed_weights,
        )

        valid_rows = row_ids < num_tokens
        if selected_experts.size(1) == 1:
            output = torch.zeros_like(hidden_states)
            output.index_add_(0, row_ids[valid_rows].long(), expert_output[valid_rows])
        else:
            route_rows = torch.full(
                (flat_experts.numel(),), packed_rows,
                device=hidden_states.device, dtype=torch.int32,
            )
            route_rows.scatter_(0, local_routes, packed_positions.to(torch.int32))
            output = self._combine_expert_routes(
                expert_output, route_rows, num_tokens, selected_experts.size(1)
            )
        return output, local_assignments

    @property
    def graph_token_limit(self) -> int:
        # Global dispatch removes TP duplicates; only DP expands the token batch.
        dp_size = (
            self.ep_size // get_tp_world_size()
            if self.dispatch_backend in ("allgather_reduce", "allgather_reducescatter")
            else 1
        )
        return MOE_MAX_GRAPH_TOKENS_PER_RANK * dp_size

    def _forward_static_decode(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Compact expert tasks on GPU; keep activations in token-route order."""
        num_tokens = hidden_states.size(0)
        num_routes = selected_experts.numel()
        # Only indices need expert padding; activations have num_routes rows.
        max_padded_rows = (
            num_routes + min(num_routes, self.num_local_experts) * (_MOE_BLOCK_M - 1)
        )
        num_tiles = triton.cdiv(max_padded_rows, _MOE_BLOCK_M)
        padded_rows = num_tiles * _MOE_BLOCK_M
        flat_experts = selected_experts.reshape(-1)
        flat_weights = routing_weights.reshape(-1)
        local_mask, flat_local_experts = self._local_routes(flat_experts)
        sort_keys = torch.where(local_mask, flat_local_experts, self.num_local_experts)
        counts = torch.zeros(self.num_local_experts + 1, device=hidden_states.device,
                             dtype=torch.int32)
        counts.scatter_add_(0, sort_keys, torch.ones_like(sort_keys, dtype=torch.int32))
        padded_counts = (counts[:-1] + _MOE_BLOCK_M - 1) // _MOE_BLOCK_M * _MOE_BLOCK_M
        packed_starts = torch.cumsum(padded_counts, dim=0, dtype=torch.int32) - padded_counts
        num_tokens_post_padded = padded_counts.sum(dtype=torch.int32)
        order = torch.argsort(sort_keys, stable=True)
        sorted_experts = sort_keys[order]
        positions = torch.arange(num_routes, device=hidden_states.device)
        valid = sorted_experts < self.num_local_experts
        group_start = valid & torch.cat(
            (
                torch.ones(1, dtype=torch.bool, device=hidden_states.device),
                sorted_experts[1:] != sorted_experts[:-1],
            )
        )
        group_begin = torch.cummax(
            torch.where(group_start, positions, 0), dim=0
        ).values
        within_group = positions - group_begin

        packed_positions = torch.where(
            valid,
            packed_starts[sorted_experts.clamp(max=self.num_local_experts - 1)] + within_group,
            padded_rows,
        )
        row_ids = torch.full(
            (padded_rows + 1,), num_routes,
            dtype=torch.int32, device=hidden_states.device,
        )
        row_ids.scatter_(0, packed_positions, order.to(torch.int32))
        tile_indices = torch.where(
            valid,
            packed_positions // _MOE_BLOCK_M,
            num_tiles,
        )
        tile_experts = torch.where(
            valid, sorted_experts, 0
        ).to(torch.int32)
        expert_ids = torch.zeros(
            num_tiles + 1, dtype=torch.int32, device=hidden_states.device
        )
        expert_ids.scatter_(0, tile_indices, tile_experts)

        row_ids = row_ids[:padded_rows]
        expert_ids = expert_ids[:num_tiles]
        gate_up = self._run_grouped_gemm(
            hidden_states, self.gate_up_proj, row_ids, expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            input_top_k=selected_experts.size(1), num_output_rows=num_routes,
        )
        gate, up = gate_up.chunk(2, dim=-1)
        activated = F.silu(gate) * up
        expert_output = self._run_grouped_gemm(
            activated, self.down_proj, row_ids, expert_ids, flat_weights,
            num_tokens_post_padded=num_tokens_post_padded, num_output_rows=num_routes,
        )
        route_rows = torch.arange(num_routes, device=hidden_states.device, dtype=torch.int32)
        output = self._combine_expert_routes(
            expert_output, route_rows,
            num_tokens, selected_experts.size(1),
        )
        local_assignments = (
            0 if torch.cuda.is_current_stream_capturing()
            else int(local_mask.sum().item())
        )
        return output, local_assignments

    def _route(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)
        if self.expert_capacity is not None or self.expert_capacity_factor is not None:
            capacity = (
                self.expert_capacity if self.expert_capacity is not None else
                max(1, math.ceil(
                    self.expert_capacity_factor * hidden_states.size(0)
                    * self.top_k / self.num_experts
                ))
            )
            flat_experts = selected_experts.reshape(-1)
            order = torch.argsort(flat_experts, stable=True)
            if (
                self.graph_safe_decode and hidden_states.is_cuda
                and hidden_states.size(0) <= self.graph_token_limit
            ):
                # Bincount cannot be captured; sorted group offsets have fixed shape.
                sorted_experts = flat_experts[order]
                indices = torch.arange(flat_experts.numel(), device=hidden_states.device)
                group_start = torch.cat((
                    torch.ones(1, dtype=torch.bool, device=hidden_states.device),
                    sorted_experts[1:] != sorted_experts[:-1],
                ))
                first_in_group = torch.cummax(
                    torch.where(group_start, indices, 0), dim=0
                ).values
                positions = indices - first_in_group
            else:
                counts = torch.bincount(flat_experts, minlength=self.num_experts)
                starts = torch.cumsum(counts, dim=0) - counts
                positions = torch.arange(
                    flat_experts.numel(), device=hidden_states.device
                ) - starts[flat_experts[order]]
            keep = torch.empty_like(positions, dtype=torch.bool)
            keep[order] = positions < capacity
            keep = keep.reshape_as(selected_experts)
            if (
                not hidden_states.is_cuda
                or not torch.cuda.is_current_stream_capturing()
            ):
                self.dropped_assignment_count += int((~keep).sum().item())
            routing_weights = routing_weights.masked_fill(~keep, 0)
            selected_experts = selected_experts.masked_fill(~keep, self.num_experts)
        return routing_weights, selected_experts

    def _execute_local(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        if hidden_states.size(0) == 0:
            return torch.zeros_like(hidden_states), 0
        if self._can_use_triton_kernel(hidden_states):
            if self.graph_safe_decode and hidden_states.size(0) <= self.graph_token_limit:
                return self._forward_static_decode(
                    hidden_states, routing_weights, selected_experts
                )
            return self._forward_triton(
                hidden_states, routing_weights, selected_experts
            )
        return self._forward_reference(
            hidden_states, routing_weights, selected_experts
        )

    def _forward_global_dp(
        self, hidden_states: torch.Tensor, reduce_scatter_output: bool
    ) -> torch.Tensor:
        tp_size = get_tp_world_size()
        if self.ep_size % tp_size:
            raise RuntimeError("global DP MoE group must contain complete TP replicas")
        local_tokens = hidden_states.size(0)
        counts = get_context().moe_token_counts
        dp_rank = self.ep_rank // tp_size
        if counts is not None:
            if (len(counts) != self.ep_size // tp_size
                    or any(type(count) is not int or count < 0 for count in counts)):
                raise ValueError("MoE token counts must cover every DP replica")
            if local_tokens != counts[dp_rank] and not (
                counts[dp_rank] == 0 and get_context().is_dummy and local_tokens == 1
            ):
                raise ValueError("MoE local token count does not match coordinated batch")
        padded_tokens = max(1, max(counts)) if counts is not None else local_tokens
        payload = (
            hidden_states if padded_tokens == local_tokens else
            F.pad(hidden_states, (0, 0, 0, padded_tokens - local_tokens))
        )
        gathered = torch.empty(
            (self.ep_size * padded_tokens, self.hidden_size),
            dtype=hidden_states.dtype, device=hidden_states.device,
        )
        dist.all_gather_into_tensor(gathered, payload.contiguous(),
                                    group=get_moe_group())
        self.ep_all_gather_count += 1
        replicas = gathered.view(
            self.ep_size, padded_tokens, self.hidden_size
        )[::tp_size]
        # Padding is transport-only: dummy rows must not consume expert capacity.
        all_tokens = (
            replicas.reshape(-1, self.hidden_size) if counts is None else
            torch.cat([replica[:count] for replica, count in zip(replicas, counts)])
        )
        routing_weights, selected_experts = self._route(all_tokens)
        output, local_assignments = self._execute_local(
            all_tokens, routing_weights, selected_experts
        )
        self.dispatch_assignment_count += local_assignments
        self.return_assignment_count += local_assignments
        self.total_assignment_count += selected_experts.numel()
        if counts is not None:
            output = torch.cat([
                F.pad(part, (0, 0, 0, padded_tokens - count))
                for part, count in zip(output.split(counts), counts)
            ])
        if reduce_scatter_output:
            if output.numel() % self.ep_size:
                raise RuntimeError("MoE output cannot be split across global EP ranks")
            shard = torch.empty(
                output.numel() // self.ep_size,
                dtype=output.dtype, device=output.device,
            )
            dist.reduce_scatter_tensor(
                shard, output.contiguous().view(-1), group=get_moe_group()
            )
            self.ep_reduce_scatter_count += 1
            if tp_size == 1:
                result = shard.view(padded_tokens, self.hidden_size)
            else:
                replica = torch.empty(
                    tp_size * shard.numel(), dtype=output.dtype,
                    device=output.device,
                )
                dist.all_gather_into_tensor(replica, shard, group=get_tp_group())
                self.ep_all_gather_count += 1
                result = replica.view(padded_tokens, self.hidden_size)
        else:
            dist.all_reduce(output, group=get_moe_group())
            self.ep_all_reduce_count += 1
            result = output.narrow(0, dp_rank * padded_tokens, padded_tokens)
        if counts is not None and counts[dp_rank] == 0:
            return torch.zeros_like(hidden_states)
        return result[:local_tokens]

    def _forward_all_to_all(
        self, hidden_states: torch.Tensor, reduce_output: bool = False
    ) -> torch.Tensor:
        group = get_ep_group()
        device = hidden_states.device
        send_counts = torch.zeros(self.ep_size, dtype=torch.int64, device=device)
        if self.ep_rank == 0:
            routing_weights, selected_experts = self._route(hidden_states)
            flat_experts = selected_experts.reshape(-1)
            self.total_assignment_count += flat_experts.numel()
            if reduce_output:
                valid = selected_experts < self.num_experts
                owners = (
                    selected_experts // self.num_local_experts
                    if self.expert_id_to_local is None
                    else self.expert_owner_tensor[
                        selected_experts.clamp(max=self.num_experts - 1)
                    ]
                )
                ranks = torch.arange(self.ep_size, device=device)
                packet_mask = (
                    (owners.unsqueeze(-1) == ranks) & valid.unsqueeze(-1)
                ).any(dim=1)
                packet_ids = torch.arange(
                    hidden_states.size(0) * self.ep_size, device=device
                )[packet_mask.reshape(-1)]
                packet_tokens = packet_ids // self.ep_size
                destinations = packet_ids % self.ep_size
                order = torch.argsort(destinations, stable=True)
                send_tokens = packet_tokens[order]
                destinations = destinations[order]
                owned = valid[send_tokens] & (
                    owners[send_tokens] == destinations[:, None]
                )
                send_experts = selected_experts[send_tokens].masked_fill(
                    ~owned, self.num_experts
                )
                send_weights = routing_weights[send_tokens].masked_fill(~owned, 0)
                send_routes = send_tokens
                self.total_dispatch_token_count += send_tokens.numel()
            else:
                flat_weights = routing_weights.reshape(-1, 1)
                token_indices = torch.arange(
                    hidden_states.size(0), device=device, dtype=torch.int64
                ).repeat_interleave(selected_experts.size(1))
                route_indices = torch.arange(
                    flat_experts.numel(), device=device, dtype=torch.int64
                )
                if self.expert_capacity is not None or self.expert_capacity_factor is not None:
                    valid = flat_experts < self.num_experts
                    flat_experts = flat_experts[valid]
                    flat_weights = flat_weights[valid]
                    token_indices = token_indices[valid]
                    route_indices = route_indices[valid]
                destinations = (
                    flat_experts // self.num_local_experts
                    if self.expert_id_to_local is None
                    else self.expert_owner_tensor[flat_experts]
                )
                order = torch.argsort(destinations, stable=True)
                send_tokens = token_indices[order]
                send_routes = route_indices[order]
                send_experts = flat_experts[order].contiguous()
                send_weights = flat_weights[order].contiguous()
                destinations = destinations[order]
            send_hidden = hidden_states[send_tokens].contiguous()
            send_counts = torch.bincount(
                destinations, minlength=self.ep_size
            ).to(torch.int64)
        else:
            send_tokens = torch.empty(0, dtype=torch.int64, device=device)
            send_routes = torch.empty(0, dtype=torch.int64, device=device)
            send_experts = torch.empty(
                (0, self.top_k) if reduce_output else (0,),
                dtype=torch.int64, device=device,
            )
            send_weights = hidden_states.new_empty(
                (0, self.top_k if reduce_output else 1)
            )
            send_hidden = hidden_states.new_empty((0, self.hidden_size))

        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=group)
        self.ep_all_to_all_count += 1
        send_splits = send_counts.tolist()
        recv_splits = recv_counts.tolist()
        local_count = sum(recv_splits)
        value_bytes = hidden_states.element_size()
        hidden_bytes = self.hidden_size * value_bytes
        pack_hidden = hidden_states.size(0) <= _PACKED_ALLTOALL_MAX_TOKENS
        route_width = self.top_k if reduce_output else 1
        weight_bytes = route_width * value_bytes
        expert_bytes = route_width * 4
        payload_bytes = weight_bytes + expert_bytes + (4 if reduce_output else 0)
        payload_bytes += hidden_bytes if pack_hidden else 0
        if not pack_hidden:
            recv_hidden = hidden_states.new_empty((local_count, self.hidden_size))
            dist.all_to_all_single(
                recv_hidden, send_hidden,
                output_split_sizes=recv_splits,
                input_split_sizes=send_splits,
                group=group,
            )
            self.ep_all_to_all_count += 1
        if self.ep_rank == 0:
            send_count = send_hidden.size(0)
            payload_parts = []
            if pack_hidden:
                payload_parts.append(
                    send_hidden.view(torch.uint8).reshape(send_count, hidden_bytes)
                )
            payload_parts.extend((
                send_weights.view(torch.uint8).reshape(send_count, weight_bytes),
                send_experts.to(torch.int32).view(torch.uint8).reshape(
                    send_count, expert_bytes
                ),
            ))
            if reduce_output:
                payload_parts.append(
                    send_routes.to(torch.int32).view(torch.uint8).reshape(send_count, 4)
                )
            send_payload = torch.cat(payload_parts, dim=1)
        else:
            send_payload = torch.empty(
                (0, payload_bytes), dtype=torch.uint8, device=device
            )
        recv_payload = torch.empty(
            (local_count, payload_bytes), dtype=torch.uint8, device=device
        )
        dist.all_to_all_single(
            recv_payload.view(-1), send_payload.view(-1),
            output_split_sizes=[count * payload_bytes for count in recv_splits],
            input_split_sizes=[count * payload_bytes for count in send_splits],
            group=group,
        )
        self.ep_all_to_all_count += 1
        metadata_start = hidden_bytes if pack_hidden else 0
        if pack_hidden:
            recv_hidden = (
                recv_payload[:, :hidden_bytes].contiguous()
                .view(hidden_states.dtype).reshape(local_count, self.hidden_size)
            )
        recv_weights = (
            recv_payload[:, metadata_start:metadata_start + weight_bytes].contiguous()
            .view(hidden_states.dtype).reshape(local_count, route_width)
        )
        expert_start = metadata_start + weight_bytes
        recv_experts = (
            recv_payload[:, expert_start:expert_start + expert_bytes].clone(
                memory_format=torch.contiguous_format
            )
            .view(torch.int32).reshape(local_count, route_width).to(torch.int64)
        )
        if reduce_output:
            recv_routes = (
                recv_payload[:, expert_start + expert_bytes:expert_start + expert_bytes + 4].clone(
                    memory_format=torch.contiguous_format
                )
                .view(torch.int32).reshape(local_count)
            )

        local_output, local_assignments = self._execute_local(
            recv_hidden, recv_weights, recv_experts
        )
        self.dispatch_assignment_count += local_assignments
        self.return_assignment_count += local_assignments
        if reduce_output:
            output = torch.zeros_like(hidden_states)
            output.index_copy_(0, recv_routes.long(), local_output)
            dist.all_reduce(output, group=group)
            self.ep_all_reduce_count += 1
            return output
        return_send_splits = [local_count] + [0] * (self.ep_size - 1)
        return_recv_splits = send_splits if self.ep_rank == 0 else [0] * self.ep_size
        returned = hidden_states.new_empty((sum(return_recv_splits), self.hidden_size))
        dist.all_to_all_single(
            returned, local_output, output_split_sizes=return_recv_splits,
            input_split_sizes=return_send_splits, group=group,
        )
        self.ep_all_to_all_count += 1
        if self.ep_rank == 0:
            route_rows = torch.full(
                (hidden_states.size(0) * self.top_k,), returned.size(0),
                dtype=torch.int32, device=device,
            )
            route_rows.scatter_(
                0, send_routes,
                torch.arange(returned.size(0), dtype=torch.int32, device=device),
            )
            output = self._combine_expert_routes(
                returned, route_rows, hidden_states.size(0), self.top_k
            )
        else:
            output = torch.empty_like(hidden_states)
        dist.broadcast(output, src=get_ep_group_ranks()[0], group=group)
        self.ep_broadcast_count += 1
        return output

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        if self.dispatch_backend in (
            "allgather_reduce", "allgather_reducescatter"
        ) and self.ep_size > 1:
            output = self._forward_global_dp(
                hidden_states,
                self.dispatch_backend == "allgather_reducescatter",
            )
        elif self.dispatch_backend in ("all_to_all", "all_to_all_reduce") and self.ep_size > 1:
            output = self._forward_all_to_all(
                hidden_states, self.dispatch_backend == "all_to_all_reduce"
            )
        else:
            routing_weights, selected_experts = self._route(hidden_states)
            output, local_assignments = self._execute_local(
                hidden_states, routing_weights, selected_experts
            )

            self.dispatch_assignment_count += local_assignments
            self.return_assignment_count += local_assignments
            self.total_assignment_count += selected_experts.numel()
            if self.ep_size > 1:
                if self.fp64_staged_reduce:
                    output = reduce_fp64_to_bf16(
                        output.double(), get_moe_group(), hidden_states.dtype
                    )
                    self.ep_reduce_scatter_count += 1
                    self.ep_all_gather_count += 1
                else:
                    if self.fp64_combine:
                        output = output.double()
                    elif self.fp32_reduce:
                        output = output.float()
                    dist.all_reduce(output, group=get_moe_group())
                    if self.fp64_combine or self.fp32_reduce:
                        output = output.to(hidden_states.dtype)
                    self.ep_all_reduce_count += 1
            elif self.fp64_combine:
                output = output.to(hidden_states.dtype)
        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states)
            shared_gate = torch.sigmoid(self.shared_expert_gate(hidden_states))
            output = output + shared_gate * shared_output
        return output.reshape(original_shape)
