import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from nanovllm.distributed.parallel_state import get_ep_group
from nanovllm.layers.linear import ReplicatedLinear


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

        self.dispatch_assignment_count += local_assignments
        self.return_assignment_count += local_assignments
        self.total_assignment_count += selected_experts.numel()
        if self.ep_size > 1:
            dist.all_reduce(output, group=get_ep_group())
            self.ep_all_reduce_count += 1
        return output.reshape(original_shape)
