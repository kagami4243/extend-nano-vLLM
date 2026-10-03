"""Real-model prefill batching check; run as a module on one GPU."""

import argparse
import json

from nanovllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./models/Qwen3-0.6B")
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-tokens", type=int, default=1)
    parser.add_argument("--shared-prefix", action="store_true")
    args = parser.parse_args()
    if min(args.prompt_tokens, args.batch_size, args.output_tokens) < 1:
        parser.error("prompt tokens, batch size and output tokens must be positive")
    if args.shared_prefix and args.prompt_tokens <= 256:
        parser.error("shared-prefix test requires more than 256 prompt tokens")

    if args.shared_prefix:
        prompts = [
            [101] * 256 + [1000 + request] + [103] * (args.prompt_tokens - 257)
            for request in range(args.batch_size)
        ]
    else:
        prompts = [
            [1000 + request] + [100 + (position + request) % 97
                                for position in range(args.prompt_tokens - 1)]
            for request in range(args.batch_size)
        ]
    llm = LLM(
        args.model,
        max_model_len=args.prompt_tokens + args.output_tokens + 8,
        max_num_batched_tokens=args.prompt_tokens * args.batch_size,
        max_num_seqs=args.batch_size,
        enable_prefill_batching=True,
        enforce_eager=False,
        enable_prefix_caching=args.shared_prefix,
        gpu_memory_utilization=0.3,
    )
    sampling = SamplingParams(
        temperature=0, ignore_eos=True, max_tokens=args.output_tokens
    )
    try:
        if args.shared_prefix:
            batched = [
                output["token_ids"]
                for output in llm.generate(prompts, sampling, use_tqdm=False)
            ]
            batch_prefill_steps = llm.scheduler.stats["prefill_steps"]
            reused_in_batch = llm.cache_stats["reused_tokens"]
            separate = [
                llm.generate([prompt], sampling, use_tqdm=False)[0]["token_ids"]
                for prompt in prompts
            ]
            assert reused_in_batch == 0, reused_in_batch
        else:
            separate = [
                llm.generate([prompt], sampling, use_tqdm=False)[0]["token_ids"]
                for prompt in prompts
            ]
            before = llm.scheduler.stats["prefill_steps"]
            batched = [
                output["token_ids"]
                for output in llm.generate(prompts, sampling, use_tqdm=False)
            ]
            batch_prefill_steps = llm.scheduler.stats["prefill_steps"] - before
            reused_in_batch = None
        assert batched == separate, (separate, batched)
        assert batch_prefill_steps == 1, batch_prefill_steps
        assert llm.model_runner.prefill_piece_enabled
        print(json.dumps({
            "separate": separate,
            "batched": batched,
            "batch_prefill_steps": batch_prefill_steps,
            "reused_in_batch": reused_in_batch,
            "reused_after_batch": llm.cache_stats["reused_tokens"],
            "prefill_piece_enabled": llm.model_runner.prefill_piece_enabled,
        }))
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
