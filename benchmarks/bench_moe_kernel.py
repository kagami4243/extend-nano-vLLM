"""Compare the reference and grouped Triton MoE expert paths.

The defaults use Qwen3-30B-A3B's MoE dimensions (H=2048, I=768, E=128,
top-k=8), but create random weights rather than loading the 30B checkpoint.
It measures local expert execution including GPU packing/alignment, not router
or EP communication.  Run with ``python -m benchmarks.bench_moe_kernel``.
"""

import argparse
from time import perf_counter

import torch
import torch.nn.functional as F

from nanovllm.layers.moe import ExpertParallelMoE


def route(module: ExpertParallelMoE, hidden_states: torch.Tensor):
    weights = F.softmax(module.gate(hidden_states), dim=-1, dtype=torch.float32)
    weights, experts = torch.topk(weights, module.top_k, dim=-1)
    return (weights / weights.sum(dim=-1, keepdim=True)).to(hidden_states.dtype), experts


def measure(function, iterations: int) -> float:
    for _ in range(5):
        function()
    torch.cuda.synchronize()
    start = perf_counter()
    for _ in range(iterations):
        function()
    torch.cuda.synchronize()
    return (perf_counter() - start) * 1000 / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if args.tokens < 1 or args.iterations < 1:
        parser.error("tokens and iterations must be positive")

    torch.manual_seed(0)
    module = ExpertParallelMoE(2048, 768, 128, 8, True).cuda().bfloat16()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(mean=0.0, std=0.02)
    hidden_states = torch.randn(
        args.tokens, 2048, device="cuda", dtype=torch.bfloat16
    )
    routing_weights, selected_experts = route(module, hidden_states)
    reference_output, _ = module._forward_reference(
        hidden_states, routing_weights, selected_experts
    )
    triton_output, _ = module._forward_triton(
        hidden_states, routing_weights, selected_experts
    )
    torch.cuda.synchronize()
    max_abs_error = (
        triton_output.float() - reference_output.float()
    ).abs().max().item()
    if max_abs_error > 2e-2:
        raise RuntimeError(f"grouped MoE result differs from reference: {max_abs_error}")
    reference_ms = measure(
        lambda: module._forward_reference(
            hidden_states, routing_weights, selected_experts
        ),
        args.iterations,
    )
    triton_ms = measure(
        lambda: module._forward_triton(
            hidden_states, routing_weights, selected_experts
        ),
        args.iterations,
    )
    print(
        f"tokens={args.tokens}, reference_ms={reference_ms:.3f}, "
        f"triton_ms={triton_ms:.3f}, speedup={reference_ms / triton_ms:.2f}x, "
        f"max_abs_error={max_abs_error:.6g}"
    )


if __name__ == "__main__":
    main()
