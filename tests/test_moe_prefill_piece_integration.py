"""Real Qwen3-MoE piecewise prefill graph smoke on up to four GPUs."""

import argparse
import json
from time import perf_counter

import torch

from nanovllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--prompt-tokens", type=int, default=16)
    args = parser.parse_args()
    if args.prompt_tokens < 1:
        parser.error("prompt tokens must be positive")
    total_prefill_tokens = 4 * args.prompt_tokens
    llm = LLM(
        "./models/Qwen3-30B-A3B-Base",
        enable_expert_parallel=args.ep > 1,
        tensor_parallel_size=args.ep,
        moe_prefill_piece=True,
        moe_prefill_piece_capture_sizes=(total_prefill_tokens,),
        max_model_len=args.prompt_tokens + 12,
        max_num_batched_tokens=total_prefill_tokens,
        max_num_seqs=4,
        enable_prefix_caching=False,
        enforce_eager=False,
        gpu_memory_utilization=0.85,
    )
    sampling = SamplingParams(temperature=0, ignore_eos=True, max_tokens=4)
    try:
        assert llm.model_runner.prefill_piece_enabled
        outputs = []
        elapsed_ms = []
        memory_before = torch.cuda.memory_reserved()
        for seed in (1, 2):
            prompts = [
                [1000 + seed * 4 + request]
                + [100 + (position + request) % 97
                   for position in range(args.prompt_tokens - 1)]
                for request in range(4)
            ]
            torch.cuda.synchronize()
            start = perf_counter()
            outputs.append(llm.generate(prompts, sampling, use_tqdm=False))
            torch.cuda.synchronize()
            elapsed_ms.append((perf_counter() - start) * 1000)
        memory_after = torch.cuda.memory_reserved()
        entry = llm.model_runner.prefill_piece_graphs[total_prefill_tokens][None]
        assert len(entry["capture"].segments) > 0
        assert llm.scheduler.stats["prefill_steps"] == 2
        assert all(len(output["token_ids"]) == 4 for run in outputs for output in run)
        diagnostics = llm.model_runner.call("get_diagnostics")
        assert len(diagnostics) == args.ep
        assert all(rank["prefill_piece_enabled"] for rank in diagnostics)
        assert all(rank["decode_graph_replay_count"] > 0 for rank in diagnostics)
        if args.prompt_tokens > 1:
            smaller_prompts = [prompt[:-1] for prompt in prompts]
            smaller_outputs = llm.generate(smaller_prompts, sampling, use_tqdm=False)
            assert all(len(output["token_ids"]) == 4 for output in smaller_outputs)
            assert llm.scheduler.stats["prefill_steps"] == 3
            assert set(llm.model_runner.prefill_piece_graphs) == {total_prefill_tokens}
        print(json.dumps({
            "ep": args.ep,
            "prompt_tokens": args.prompt_tokens,
            "prefill_segments": len(entry["capture"].segments),
            "prefill_steps": llm.scheduler.stats["prefill_steps"],
            "captured_token_counts": sorted(llm.model_runner.prefill_piece_graphs),
            "decode_graph_replays": [
                rank["decode_graph_replay_count"] for rank in diagnostics
            ],
            "first_and_replay_ms": elapsed_ms,
            "rank0_extra_reserved_bytes": memory_after - memory_before,
            "outputs": [[output["token_ids"] for output in run] for run in outputs],
        }))
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
