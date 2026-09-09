"""GPU correctness test for the minimal vLLM-style Triton MoE path.

Run with ``python -m tests.test_moe_kernel``.  This intentionally tests the
kernel against the retained per-expert reference path rather than a second MoE
implementation, so errors in packing, expert routing, or weighted combining
are visible on a small GPU allocation.
"""

import torch
import torch.nn.functional as F

from nanovllm.layers.moe import ExpertParallelMoE


def routed_inputs(
    module: ExpertParallelMoE, hidden_states: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = module.gate(hidden_states)
    routing_weights = F.softmax(logits, dim=-1, dtype=torch.float32)
    routing_weights, selected_experts = torch.topk(
        routing_weights, module.top_k, dim=-1
    )
    if module.norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    return routing_weights.to(hidden_states.dtype), selected_experts


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("test_moe_kernel requires CUDA")
    torch.manual_seed(0)
    module = ExpertParallelMoE(
        hidden_size=128,
        intermediate_size=256,
        num_experts=8,
        top_k=2,
        norm_topk_prob=True,
    ).cuda().bfloat16()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(mean=0.0, std=0.02)
    hidden_states = torch.randn(2, 11, 128, device="cuda", dtype=torch.bfloat16)
    flattened = hidden_states.reshape(-1, 128)
    routing_weights, selected_experts = routed_inputs(module, flattened)

    assert module._can_use_triton_kernel(flattened)
    reference, reference_assignments = module._forward_reference(
        flattened, routing_weights, selected_experts
    )
    actual, kernel_assignments = module._forward_triton(
        flattened, routing_weights, selected_experts
    )
    forward_output = module(hidden_states).reshape_as(flattened)
    torch.cuda.synchronize()
    max_abs_error = (actual.float() - reference.float()).abs().max().item()
    mean_abs_error = (actual.float() - reference.float()).abs().mean().item()
    forward_max_abs_error = (
        forward_output.float() - reference.float()
    ).abs().max().item()
    assert kernel_assignments == reference_assignments == flattened.size(0) * 2
    assert max_abs_error <= 2e-3, max_abs_error
    assert forward_max_abs_error <= 2e-3, forward_max_abs_error

    # Simulate EP=2 local contributions without creating a distributed group.
    # Both ranks see the same tokens/router, as nano-vLLM's teaching EP path
    # does, and their outputs must sum to the EP=1 reference result.
    ep_outputs = []
    ep_assignments = 0
    for ep_rank in range(2):
        shard = ExpertParallelMoE(128, 256, 8, 2, True, ep_rank, 2).cuda().bfloat16()
        with torch.no_grad():
            shard.gate.weight.copy_(module.gate.weight)
            start = ep_rank * shard.num_local_experts
            end = start + shard.num_local_experts
            shard.gate_up_proj.copy_(module.gate_up_proj[start:end])
            shard.down_proj.copy_(module.down_proj[start:end])
        partial, assignments = shard._forward_triton(
            flattened, routing_weights, selected_experts
        )
        ep_outputs.append(partial)
        ep_assignments += assignments
    ep_max_abs_error = (
        ep_outputs[0].float() + ep_outputs[1].float() - reference.float()
    ).abs().max().item()
    assert ep_assignments == flattened.size(0) * 2
    assert ep_max_abs_error <= 2e-3, ep_max_abs_error
    print(
        "Triton grouped MoE passed: "
        f"max_abs={max_abs_error:.6g}, mean_abs={mean_abs_error:.6g}, "
        f"forward_max_abs={forward_max_abs_error:.6g}, "
        f"ep2_max_abs={ep_max_abs_error:.6g}, assignments={kernel_assignments}"
    )


if __name__ == "__main__":
    main()
