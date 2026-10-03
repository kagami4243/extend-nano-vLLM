"""Real-model checks for TODO features; run with ``python -m tests.test_feature_integration``.

Use CUDA_VISIBLE_DEVICES=1,2,3,4 to keep all comparisons on the same GPUs.
Each case runs in a fresh process so CUDA allocator and NCCL state cannot leak.
"""

import argparse
import json
import subprocess
import sys

import torch


def run_case(model: str, case: str, prompt_tokens: int, enforce_eager: bool,
             prompt_text: str | None, capture_logits: bool, max_tokens: int) -> dict:
    from nanovllm import LLM, SamplingParams

    tp = int(case[2:]) if case.startswith("tp") else 1
    is_chunk = case == "chunk"
    is_cache = case == "cache"
    prompt = prompt_text if prompt_text is not None else [100 + i % 97 for i in range(prompt_tokens)]
    llm = LLM(
        model,
        tensor_parallel_size=tp,
        enforce_eager=enforce_eager,
        enable_prefix_caching=is_cache,
        max_model_len=prompt_tokens + 64,
        max_num_batched_tokens=64 if is_chunk else prompt_tokens,
        max_num_seqs=2,
        gpu_memory_utilization=0.3,
    )
    try:
        logit_top2 = []
        if capture_logits:
            original_sampler = llm.model_runner.sampler

            def sample_with_top2(logits, temperatures):
                values, indices = logits.float().topk(2, dim=-1)
                logit_top2.append({
                    "token_ids": indices[0].tolist(),
                    "logits": values[0].tolist(),
                })
                return original_sampler(logits, temperatures)

            llm.model_runner.sampler = sample_with_top2
        sampling = SamplingParams(temperature=0, ignore_eos=True, max_tokens=max_tokens)
        first = llm.generate([prompt], sampling, use_tqdm=False)[0]["token_ids"]
        if is_cache:
            second = llm.generate([prompt], sampling, use_tqdm=False)[0]["token_ids"]
            assert second == first, (first, second)
        return {
            "case": case,
            "tokens": first,
            "logit_top2": logit_top2,
            "prefill_piece_enabled": llm.model_runner.prefill_piece_enabled,
            "cache_stats": llm.cache_stats,
            "scheduler_stats": llm.scheduler.stats.copy(),
        }
    finally:
        llm.exit()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./models/Qwen3-0.6B")
    parser.add_argument("--prompt-tokens", type=int, default=320)
    parser.add_argument("--prompt-text")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--capture-logits", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--cases", nargs="+", default=["base", "cache", "chunk", "tp2", "tp4"])
    parser.add_argument("--child-case", choices=["base", "cache", "chunk", "tp2", "tp4"])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    selected_cases = [args.child_case] if args.child_case else args.cases
    if args.max_tokens < 1:
        parser.error("max tokens must be positive")
    if args.prompt_tokens < 1 or ("cache" in selected_cases and args.prompt_tokens < 256):
        parser.error("prompt must be positive and cache case requires >=256 tokens")
    if args.child_case:
        print(json.dumps(run_case(args.model, args.child_case, args.prompt_tokens,
                                  args.enforce_eager, args.prompt_text, args.capture_logits,
                                  args.max_tokens)))
        return
    results = {}
    for case in args.cases:
        command = [sys.executable, "-m", "tests.test_feature_integration", "--model", args.model,
                   "--prompt-tokens", str(args.prompt_tokens), "--child-case", case]
        if args.enforce_eager:
            command.append("--enforce-eager")
        if args.prompt_text is not None:
            command.extend(("--prompt-text", args.prompt_text))
        if args.capture_logits:
            command.append("--capture-logits")
        command.extend(("--max-tokens", str(args.max_tokens)))
        completed = subprocess.run(
            command,
            check=False, capture_output=True, text=True, timeout=600,
        )
        if completed.returncode:
            raise RuntimeError(f"{case} failed:\n{completed.stderr}")
        results[case] = json.loads(completed.stdout.strip().splitlines()[-1])
        print(json.dumps(results[case]), flush=True)
    if "base" in results:
        for case in ("cache", "chunk", "tp2", "tp4"):
            if case in results:
                assert results[case]["tokens"] == results["base"]["tokens"], case
    for case, result in results.items():
        assert result["prefill_piece_enabled"] is not args.enforce_eager, case
    if "cache" in results:
        assert results["cache"]["cache_stats"]["reused_tokens"] >= min(256, args.prompt_tokens)
    if "chunk" in results:
        assert results["chunk"]["scheduler_stats"]["prefill_steps"] == (args.prompt_tokens + 63) // 64
        assert results["chunk"]["scheduler_stats"]["max_prefill_tokens_per_step"] <= 64
    print("feature integration passed")


if __name__ == "__main__":
    main()
