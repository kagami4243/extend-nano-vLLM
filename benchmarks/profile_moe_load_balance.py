"""Estimate EP placement imbalance from Qwen3-MoE router assignments."""

import argparse
import json
from types import MethodType

import torch

from benchmarks.bench_moe_ep import make_prompts
from nanovllm import LLM, SamplingParams
from nanovllm.layers.moe import ExpertParallelMoE


def balanced_placement(expert_counts, ranks):
    experts_per_rank = len(expert_counts) // ranks
    loads = [0] * ranks
    placed = [0] * ranks
    owners = [0] * len(expert_counts)
    for expert in sorted(
        range(len(expert_counts)), key=lambda index: expert_counts[index],
        reverse=True,
    ):
        rank = min(
            (index for index in range(ranks) if placed[index] < experts_per_rank),
            key=lambda index: loads[index],
        )
        loads[rank] += expert_counts[expert]
        placed[rank] += 1
        owners[expert] = rank
    return owners, loads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./models/Qwen3-30B-A3B-Base")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--prompt-tokens", type=int, default=16)
    parser.add_argument("--output-tokens", type=int, default=4)
    parser.add_argument("--ep", type=int, default=4)
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--result-file")
    parser.add_argument("--placement-file")
    args = parser.parse_args()
    if min(args.batch_size, args.prompt_tokens, args.output_tokens, args.ep) < 1:
        parser.error("workload and EP dimensions must be positive")

    llm = LLM(
        args.model,
        tensor_parallel_size=1,
        enable_expert_parallel=True,
        enforce_eager=True,
        enable_prefix_caching=False,
        max_model_len=args.prompt_tokens + args.output_tokens + 8,
        max_num_batched_tokens=args.batch_size * args.prompt_tokens,
        max_num_seqs=args.batch_size,
        gpu_memory_utilization=0.9,
    )
    layers = llm.model_runner.model.model.layers
    histograms = {}
    try:
        for layer_id, layer in layers.items():
            module = layer.mlp
            if not isinstance(module, ExpertParallelMoE):
                continue
            if module.num_experts % args.ep:
                raise ValueError("EP size must divide the expert count")
            histograms[layer_id] = [0] * module.num_experts
            original_route = module._route

            def record_route(self, hidden_states, *, key=layer_id,
                             route=original_route):
                weights, experts = route(hidden_states)
                counts = torch.bincount(
                    experts.reshape(-1), minlength=self.num_experts
                ).cpu().tolist()
                histograms[key] = [
                    old + count for old, count in zip(histograms[key], counts)
                ]
                return weights, experts

            module._route = MethodType(record_route, module)

        outputs = llm.generate(
            make_prompts(args.seed, args.batch_size, args.prompt_tokens),
            SamplingParams(
                temperature=0, ignore_eos=True, max_tokens=args.output_tokens
            ),
            use_tqdm=False,
        )
        expected_per_layer = (
            args.batch_size * (args.prompt_tokens + args.output_tokens - 1)
            * llm.config.hf_config.num_experts_per_tok
        )
        per_layer = []
        for layer_id, counts in histograms.items():
            if sum(counts) != expected_per_layer:
                raise RuntimeError(
                    f"layer {layer_id} recorded {sum(counts)} assignments, "
                    f"expected {expected_per_layer}"
                )
            shard_size = len(counts) // args.ep
            contiguous = [
                sum(counts[rank * shard_size:(rank + 1) * shard_size])
                for rank in range(args.ep)
            ]
            owners, balanced = balanced_placement(counts, args.ep)
            per_layer.append({
                "layer": int(layer_id),
                "counts": counts,
                "contiguous_loads": contiguous,
                "balanced_loads": balanced,
                "balanced_owners": owners,
            })
        contiguous_peak_sum = sum(
            max(layer["contiguous_loads"]) for layer in per_layer
        )
        balanced_peak_sum = sum(
            max(layer["balanced_loads"]) for layer in per_layer
        )
        result = {
            "model": args.model,
            "batch_size": args.batch_size,
            "prompt_tokens": args.prompt_tokens,
            "output_tokens": args.output_tokens,
            "ep": args.ep,
            "seed": args.seed,
            "assignment_count_per_layer": expected_per_layer,
            "layers": per_layer,
            "contiguous_peak_sum": contiguous_peak_sum,
            "balanced_peak_sum": balanced_peak_sum,
            "peak_assignment_ratio": contiguous_peak_sum / balanced_peak_sum,
            "output_token_ids": [output["token_ids"] for output in outputs],
        }
        if args.result_file:
            with open(args.result_file, "w") as file:
                json.dump(result, file, indent=2)
                file.write("\n")
        if args.placement_file:
            placement = {
                "model": args.model,
                "num_experts": llm.config.hf_config.num_experts,
                "expert_parallel_size": args.ep,
                "layers": {
                    str(layer["layer"]): layer["balanced_owners"]
                    for layer in per_layer
                },
            }
            with open(args.placement_file, "w") as file:
                json.dump(placement, file, indent=2)
                file.write("\n")
        print(json.dumps({
            key: value for key, value in result.items() if key != "layers"
        }))
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
