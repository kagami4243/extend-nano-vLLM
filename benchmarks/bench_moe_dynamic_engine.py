"""Steady-state Qwen3-MoE before/after live placement with default EP=TP."""

import argparse
import json
from statistics import median
from time import perf_counter

import torch

from benchmarks.bench_moe_ep import make_prompts
from nanovllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./models/Qwen3-30B-A3B-Base")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--prompt-tokens", type=int, default=16)
    parser.add_argument("--output-tokens", type=int, default=4)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--result-file")
    args = parser.parse_args()
    llm = LLM(
        args.model, tensor_parallel_size=2, enable_expert_parallel=True,
        moe_dynamic_placement=True, enforce_eager=False,
        enable_prefill_batching=True,
        enable_prefix_caching=False,
        max_model_len=args.prompt_tokens + args.output_tokens + 8,
        max_num_batched_tokens=args.batch_size * args.prompt_tokens,
        max_num_seqs=args.batch_size,
        gpu_memory_utilization=0.9,
    )
    sampling = SamplingParams(
        temperature=0, ignore_eos=True, max_tokens=args.output_tokens,
    )

    def generate(seed):
        return [item["token_ids"] for item in llm.generate(
            make_prompts(seed, args.batch_size, args.prompt_tokens),
            sampling, use_tqdm=False,
        )]

    def measure():
        for seed in range(3):
            generate(seed)
        samples = []
        outputs = []
        for seed in range(args.runs):
            torch.cuda.synchronize()
            start = perf_counter()
            outputs.append(generate(100 + seed))
            torch.cuda.synchronize()
            samples.append((perf_counter() - start) * 1000)
        return samples, outputs

    try:
        before_diag = llm.model_runner.call("get_diagnostics")
        before_samples, before_outputs = measure()
        mid_diag = llm.model_runner.call("get_diagnostics")
        experts = llm.config.hf_config.num_experts
        placements = {
            str(layer): [expert % 2 for expert in range(experts)]
            for layer in range(llm.config.hf_config.num_hidden_layers)
        }
        torch.cuda.synchronize()
        start = perf_counter()
        moved = llm.relocate_experts(placements)
        torch.cuda.synchronize()
        migration_ms = (perf_counter() - start) * 1000
        after_samples, after_outputs = measure()
        after_diag = llm.model_runner.call("get_diagnostics")
        before_median = median(before_samples)
        after_median = median(after_samples)
        result = {
            "model": args.model,
            "topology": "TP=2, DP=PP=1, default EP=TP=2",
            "dispatch_backend": "replicated",
            "cuda_graph_enabled": True,
            "batch_size": args.batch_size,
            "prompt_tokens": args.prompt_tokens,
            "output_tokens": args.output_tokens,
            "prefix_cache": False,
            "enable_prefill_batching": True,
            "runs": args.runs,
            "latency_ms_contiguous": before_samples,
            "latency_ms_dynamic_placement": after_samples,
            "median_ms_contiguous": before_median,
            "median_ms_dynamic_placement": after_median,
            "speedup": before_median / after_median,
            "migration_ms": migration_ms,
            "moved_experts_per_rank": [sum(rank.values()) for rank in moved],
            "outputs_match": before_outputs == after_outputs,
            "matching_sequences": sum(
                before == after
                for before_round, after_round in zip(before_outputs, after_outputs)
                for before, after in zip(before_round, after_round)
            ),
            "total_sequences": args.runs * args.batch_size,
            "outputs_contiguous": before_outputs,
            "outputs_dynamic_placement": after_outputs,
            "decode_graph_replays_contiguous": [
                after["decode_graph_replay_count"]
                - before["decode_graph_replay_count"]
                for before, after in zip(before_diag, mid_diag)
            ],
            "decode_graph_replays_dynamic_placement": [
                after["decode_graph_replay_count"]
                - before["decode_graph_replay_count"]
                for before, after in zip(mid_diag, after_diag)
            ],
        }
        print(json.dumps(result, indent=2), flush=True)
        if args.result_file:
            with open(args.result_file, "w") as output:
                json.dump(result, output, indent=2)
                output.write("\n")
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
