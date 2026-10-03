"""Compare one replicated MoE layer with four local expert shards."""

import argparse
import json
from pathlib import Path
from types import MethodType

import torch

from nanovllm.layers.moe import ExpertParallelMoE


def metrics(actual, expected):
    difference = (actual.float() - expected.float()).abs()
    return {
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "different_elements": int(torch.count_nonzero(difference)),
        "total_elements": difference.numel(),
    }


def combine_fp32(self, expert_output, route_rows, num_tokens, top_k):
    padded = torch.cat((
        expert_output, expert_output.new_zeros((1, self.hidden_size)),
    ))
    return (
        padded[route_rows.long()]
        .reshape(num_tokens, top_k, self.hidden_size)
        .float().sum(dim=1)
    )


def combine_fp64(self, expert_output, route_rows, num_tokens, top_k):
    padded = torch.cat((
        expert_output, expert_output.new_zeros((1, self.hidden_size)),
    ))
    return (
        padded[route_rows.long()]
        .reshape(num_tokens, top_k, self.hidden_size)
        .double().sum(dim=1)
    )


def inspect_routes(expert_output, route_rows, token_row, hidden_column, top_k):
    rows = route_rows[token_row * top_k:(token_row + 1) * top_k].long()
    values = torch.zeros(top_k, dtype=expert_output.dtype,
                         device=expert_output.device)
    valid = rows < expert_output.size(0)
    values[valid] = expert_output[rows[valid], hidden_column]
    return values.float().cpu().tolist()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=768)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--checkpoint")
    parser.add_argument("--trace-input")
    parser.add_argument("--layer-id", type=int, default=1)
    parser.add_argument("--inspect-row", type=int)
    parser.add_argument("--inspect-column", type=int, default=0)
    parser.add_argument("--result-file")
    args = parser.parse_args()
    if bool(args.checkpoint) != bool(args.trace_input):
        parser.error("--checkpoint and --trace-input must be supplied together")
    torch.manual_seed(args.seed)
    full = ExpertParallelMoE(
        args.hidden_size, args.intermediate_size,
        args.experts, args.top_k, True,
    ).cuda().bfloat16()
    if args.checkpoint:
        from safetensors import safe_open

        index = json.loads(
            (Path(args.checkpoint) / "model.safetensors.index.json").read_text()
        )["weight_map"]
        prefix = f"model.layers.{args.layer_id}.mlp."
        with torch.no_grad():
            gate_name = prefix + "gate.weight"
            with safe_open(
                str(Path(args.checkpoint) / index[gate_name]),
                framework="pt", device="cpu",
            ) as weights:
                full.gate.weight.copy_(weights.get_tensor(gate_name))
            for expert in range(args.experts):
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    name = prefix + f"experts.{expert}.{projection}.weight"
                    with safe_open(
                        str(Path(args.checkpoint) / index[name]),
                        framework="pt", device="cpu",
                    ) as weights:
                        full.load_expert_weight(
                            expert, projection, weights.get_tensor(name)
                        )
        hidden = torch.load(
            args.trace_input, map_location="cpu", weights_only=True
        )["vectors"][f"{args.layer_id}.mlp_input_span"].to(
            device="cuda", dtype=torch.bfloat16
        )
    else:
        with torch.no_grad():
            for parameter in full.parameters():
                parameter.normal_(0, 0.02)
        hidden = torch.randn(
            args.tokens, args.hidden_size, device="cuda", dtype=torch.bfloat16
        )
    weights, experts = full._route(hidden)
    if args.inspect_row is not None and (
        not 0 <= args.inspect_row < hidden.size(0)
        or not 0 <= args.inspect_column < args.hidden_size
    ):
        parser.error("inspection row or column is out of range")
    inspected = []
    if args.inspect_row is not None:
        original_combine = full._combine_expert_routes

        def inspect_combine(self, expert_output, route_rows, num_tokens, top_k):
            inspected.append(inspect_routes(
                expert_output, route_rows, args.inspect_row,
                args.inspect_column, top_k,
            ))
            return original_combine(expert_output, route_rows, num_tokens, top_k)

        full._combine_expert_routes = MethodType(inspect_combine, full)
    full_output, full_routes = full._execute_local(hidden, weights, experts)
    full._combine_expert_routes = MethodType(combine_fp64, full)
    full_fp64_output, _ = full._execute_local(hidden, weights, experts)

    shards = []
    routes = []
    fp32_local_shards = []
    fp64_local_shards = []
    for rank in range(4):
        shard = ExpertParallelMoE(
            args.hidden_size, args.intermediate_size,
            args.experts, args.top_k, True,
            ep_rank=rank, ep_size=4,
        ).cuda().bfloat16()
        start = rank * shard.num_local_experts
        end = start + shard.num_local_experts
        with torch.no_grad():
            shard.gate.weight.copy_(full.gate.weight)
            shard.gate_up_proj.copy_(full.gate_up_proj[start:end])
            shard.down_proj.copy_(full.down_proj[start:end])
        if args.inspect_row is not None:
            original_combine = shard._combine_expert_routes

            def inspect_combine(self, expert_output, route_rows, num_tokens,
                                top_k, original=original_combine):
                inspected.append(inspect_routes(
                    expert_output, route_rows, args.inspect_row,
                    args.inspect_column, top_k,
                ))
                return original(expert_output, route_rows, num_tokens, top_k)

            shard._combine_expert_routes = MethodType(inspect_combine, shard)
        output, count = shard._execute_local(hidden, weights, experts)
        shards.append(output)
        routes.append(count)
        shard._combine_expert_routes = MethodType(combine_fp32, shard)
        output, _ = shard._execute_local(hidden, weights, experts)
        fp32_local_shards.append(output)
        shard._combine_expert_routes = MethodType(combine_fp64, shard)
        output, _ = shard._execute_local(hidden, weights, experts)
        fp64_local_shards.append(output)
        del shard

    fp32_sum = torch.stack(shards).float().sum(dim=0).bfloat16()
    fp32_local_sum = torch.stack(fp32_local_shards).sum(dim=0).bfloat16()
    fp64_local_sum = torch.stack(fp64_local_shards).sum(dim=0).bfloat16()
    high_parts = [part.float() for part in fp64_local_shards]
    low_parts = [
        (part - high.double()).bfloat16()
        for part, high in zip(fp64_local_shards, high_parts)
    ]
    high_sum = torch.stack(high_parts).sum(dim=0)
    low_sum_bf16 = torch.stack(low_parts).sum(dim=0)
    compensated_sum = (
        high_sum.double() + low_sum_bf16.double()
    ).bfloat16()
    bf16_sum = torch.zeros_like(full_output)
    for output in shards:
        bf16_sum += output
    result = {
        "tokens": hidden.size(0),
        "checkpoint": args.checkpoint,
        "layer_id": args.layer_id if args.checkpoint else None,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "experts": args.experts,
        "top_k": args.top_k,
        "full_routes": full_routes,
        "shard_routes": routes,
        "fp32_shard_sum_vs_full": metrics(fp32_sum, full_output),
        "bf16_shard_sum_vs_full": metrics(bf16_sum, full_output),
        "bf16_shard_sum_vs_fp32": metrics(bf16_sum, fp32_sum),
        "fp32_local_shard_sum_vs_full": metrics(fp32_local_sum, full_output),
        "fp64_local_shard_sum_vs_full": metrics(fp64_local_sum, full_output),
        "fp64_local_shard_sum_vs_full_fp64": metrics(
            fp64_local_sum, full_fp64_output.bfloat16()
        ),
        "full_fp64_vs_full": metrics(full_fp64_output.bfloat16(), full_output),
        "fp32_plus_bf16_low_vs_full_fp64": metrics(
            compensated_sum, full_fp64_output.bfloat16()
        ),
    }
    different = torch.nonzero(fp32_local_sum != full_output)
    result["fp32_local_shard_differences"] = [
        {
            "token_row": int(row),
            "hidden_column": int(column),
            "full": float(full_output[row, column]),
            "shards": float(fp32_local_sum[row, column]),
        }
        for row, column in different[:20].tolist()
    ]
    if args.inspect_row is not None:
        result["inspected_routes"] = {
            "token_row": args.inspect_row,
            "hidden_column": args.inspect_column,
            "experts": experts[args.inspect_row].cpu().tolist(),
            "weights": weights[args.inspect_row].float().cpu().tolist(),
            "full": inspected[0],
            "shards": inspected[1:],
        }
    if args.result_file:
        with open(args.result_file, "w") as file:
            json.dump(result, file, indent=2)
            file.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
