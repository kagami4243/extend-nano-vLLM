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

from nanovllm.distributed.parallel_state import get_ep_group
from nanovllm.layers.linear import ReplicatedLinear


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


if _TRITON_AVAILABLE:

    @triton.jit
    def _grouped_moe_gemm_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        row_ids_ptr,
        expert_ids_ptr,
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
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """One expert-major tile per program, as in vLLM fused_moe_kernel.

        ``row_ids`` maps packed expert-major rows back to rows in ``a``.  A
        sentinel equal to ``num_input_rows`` represents padding and produces
        zero output.  ``expert_ids`` contains one expert id for each M tile.
        """
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_ids = tl.load(row_ids_ptr + offs_m)
        row_mask = row_ids < num_input_rows
        expert_id = tl.load(expert_ids_ptr + pid_m)

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            a = tl.load(
                a_ptr
                + row_ids[:, None] * stride_am
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
                routing_weights_ptr + offs_m, mask=row_mask, other=0.0
            )
            accumulator *= routing_weight[:, None]

        tl.store(
            c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
            accumulator,
            mask=row_mask[:, None] & (offs_n[None, :] < N),
        )


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
    ) -> None:
        super().__init__()
        if num_experts % ep_size != 0:
            raise ValueError("num_experts must divide evenly across EP ranks")
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between 1 and num_experts")

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.num_local_experts = num_experts // ep_size
        self.expert_start = ep_rank * self.num_local_experts
        self.expert_end = self.expert_start + self.num_local_experts

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
        self.ep_all_reduce_count = 0
        self.dispatch_assignment_count = 0
        self.return_assignment_count = 0
        self.total_assignment_count = 0

    @property
    def expert_parameter_bytes(self) -> int:
        return sum(
            parameter.numel() * parameter.element_size()
            for parameter in (self.gate_up_proj, self.down_proj)
        )

    def owns_expert(self, expert_id: int) -> bool:
        return self.expert_start <= expert_id < self.expert_end

    def load_expert_weight(
        self,
        expert_id: int,
        projection: str,
        loaded_weight: torch.Tensor,
    ) -> None:
        if not self.owns_expert(expert_id):
            return
        local_expert_id = expert_id - self.expert_start
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
            local_expert_id = expert_id - self.expert_start
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
    ) -> torch.Tensor:
        """Run B[E, N, K] against rows of A in the vLLM packed MoE layout."""
        packed_rows = row_ids.numel()
        output_size = b.size(1)
        output = torch.zeros(
            (packed_rows, output_size), device=a.device, dtype=a.dtype
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
            BLOCK_M=_MOE_BLOCK_M,
            BLOCK_N=64,
            BLOCK_K=32,
            num_warps=4,
        )
        return output

    def _forward_triton(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Fused-style local-expert execution without a Python expert loop.

        Tokens are still replicated on EP ranks in this teaching implementation.
        Therefore every rank selects its local routes, executes only its local
        weights, and ``forward`` all-reduces the partial result afterwards.
        """
        num_tokens = hidden_states.size(0)
        flat_experts = selected_experts.reshape(-1)
        flat_tokens = torch.arange(
            num_tokens, device=hidden_states.device, dtype=torch.int32
        ).repeat_interleave(self.top_k)
        flat_weights = routing_weights.reshape(-1)
        local_mask = (flat_experts >= self.expert_start) & (
            flat_experts < self.expert_end
        )
        local_assignments = int(local_mask.sum().item())
        if local_assignments == 0:
            return torch.zeros_like(hidden_states), 0

        # Sort expert-major, then pad each expert's rows to the M tile size.
        # This is the layout consumed by vLLM's fused_moe_kernel as well.
        local_experts = flat_experts[local_mask] - self.expert_start
        local_tokens = flat_tokens[local_mask]
        local_weights = flat_weights[local_mask]
        order = torch.argsort(local_experts, stable=True)
        local_experts = local_experts[order]
        local_tokens = local_tokens[order]
        local_weights = local_weights[order]
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

        output = torch.zeros_like(hidden_states)
        valid_rows = row_ids < num_tokens
        output.index_add_(0, row_ids[valid_rows].long(), expert_output[valid_rows])
        return output, local_assignments

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        if self._can_use_triton_kernel(hidden_states):
            output, local_assignments = self._forward_triton(
                hidden_states, routing_weights, selected_experts
            )
        else:
            output, local_assignments = self._forward_reference(
                hidden_states, routing_weights, selected_experts
            )

        self.dispatch_assignment_count += local_assignments
        self.return_assignment_count += local_assignments
        self.total_assignment_count += selected_experts.numel()
        if self.ep_size > 1:
            dist.all_reduce(output, group=get_ep_group())
            self.ep_all_reduce_count += 1
        return output.reshape(original_shape)
