"""Measure single-request EAGLE3 TTFT and TTPO for vLLM or nano-vLLM.

The two backends run as separate processes. Model construction and a short
warm-up request are excluded from the measurement. TTPO is measured from the
first observed output token through the final output token, divided by the
number of intervening tokens.
"""

import argparse
import asyncio
import gc
import json
from dataclasses import dataclass
from statistics import median
from time import perf_counter

import torch
from transformers import AutoTokenizer


PROMPT_TOKENS = 1024
OUTPUT_TOKENS = 256
WARMUP_PROMPT_TOKENS = 16
WARMUP_OUTPUT_TOKENS = 8
NUM_RUNS = 3


@dataclass
class Latency:
    ttft_ms: float
    ttpo_ms: float
    output_tokens: int


def make_prompt_token_ids(model: str, num_tokens: int) -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    token_ids = tokenizer.encode(" benchmark", add_special_tokens=False)
    if not token_ids:
        raise RuntimeError("failed to build the benchmark prompt token")
    repeats = (num_tokens + len(token_ids) - 1) // len(token_ids)
    return (token_ids * repeats)[:num_tokens]


def summarize(latencies: list[Latency]) -> dict[str, float | int]:
    return {
        "runs": len(latencies),
        "ttft_ms_median": median(item.ttft_ms for item in latencies),
        "ttpo_ms_median": median(item.ttpo_ms for item in latencies),
        "output_tokens": latencies[0].output_tokens,
    }


def measure_nanovllm_request(llm, prompt_token_ids: list[int], max_tokens: int) -> Latency:
    from nanovllm import SamplingParams

    llm.add_request(
        prompt_token_ids,
        SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=max_tokens),
    )
    seq = llm.scheduler.waiting[-1]
    torch.cuda.synchronize()
    start = perf_counter()
    first_token_time = None
    while not llm.is_finished():
        llm.step()
        torch.cuda.synchronize()
        if first_token_time is None and seq.num_completion_tokens:
            first_token_time = perf_counter()
    end = perf_counter()
    if first_token_time is None:
        raise RuntimeError("nano-vLLM emitted no output token")
    output_tokens = seq.num_completion_tokens
    return Latency(
        ttft_ms=(first_token_time - start) * 1000,
        ttpo_ms=(end - first_token_time) * 1000 / max(output_tokens - 1, 1),
        output_tokens=output_tokens,
    )


def run_nanovllm(args) -> dict[str, float | int]:
    from nanovllm import LLM
    import nanovllm.engine.llm_engine as engine_module

    acceptance = {"drafted": 0, "accepted": 0}
    original_verify = engine_module.verify_greedy_proposals

    def count_verify(proposal_tokens, target_tokens):
        result = original_verify(proposal_tokens, target_tokens)
        acceptance["drafted"] += len(proposal_tokens)
        acceptance["accepted"] += result.accepted_count
        return result

    engine_module.verify_greedy_proposals = count_verify

    llm = LLM(
        args.target_model,
        enforce_eager=False,
        max_model_len=PROMPT_TOKENS + OUTPUT_TOKENS,
        max_num_batched_tokens=PROMPT_TOKENS + OUTPUT_TOKENS,
        max_num_seqs=1,
        gpu_memory_utilization=0.8,
        enable_prefix_caching=False,
        speculative_config={
            "method": "eagle3",
            "model": args.draft_model,
            "num_speculative_tokens": 8,
        },
    )
    try:
        measure_nanovllm_request(
            llm,
            make_prompt_token_ids(args.target_model, WARMUP_PROMPT_TOKENS),
            WARMUP_OUTPUT_TOKENS,
        )
        acceptance["drafted"] = 0
        acceptance["accepted"] = 0
        results = [
            measure_nanovllm_request(
                llm,
                make_prompt_token_ids(args.target_model, PROMPT_TOKENS),
                OUTPUT_TOKENS,
            )
            for _ in range(NUM_RUNS)
        ]
        result = summarize(results)
        result.update(acceptance_rate=(100 * acceptance["accepted"] / acceptance["drafted"]
                                       if acceptance["drafted"] else float("nan")),
                      drafted=acceptance["drafted"], accepted=acceptance["accepted"])
        return result
    finally:
        engine_module.verify_greedy_proposals = original_verify
        llm.exit()
        del llm
        gc.collect()
        torch.cuda.empty_cache()


async def measure_vllm_request(engine, prompt_token_ids: list[int], max_tokens: int, request_id: str) -> Latency:
    from vllm import SamplingParams, TokensPrompt

    sampling_params = SamplingParams(
        temperature=0.0,
        ignore_eos=True,
        max_tokens=max_tokens,
    )
    start = perf_counter()
    first_token_time = None
    output_tokens = 0
    async for output in engine.generate(
        TokensPrompt(prompt_token_ids=prompt_token_ids), sampling_params, request_id
    ):
        output_tokens = len(output.outputs[0].token_ids)
        if first_token_time is None and output_tokens:
            first_token_time = perf_counter()
    end = perf_counter()
    if first_token_time is None:
        raise RuntimeError("vLLM emitted no output token")
    return Latency(
        ttft_ms=(first_token_time - start) * 1000,
        ttpo_ms=(end - first_token_time) * 1000 / max(output_tokens - 1, 1),
        output_tokens=output_tokens,
    )


async def run_vllm_async(args) -> dict[str, float | int]:
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.config import ProfilerConfig

    engine_args = AsyncEngineArgs(
        model=args.target_model,
        tokenizer=args.target_model,
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=False,
        max_model_len=PROMPT_TOKENS + OUTPUT_TOKENS,
        max_num_batched_tokens=PROMPT_TOKENS + OUTPUT_TOKENS,
        max_num_seqs=1,
        gpu_memory_utilization=0.8,
        enable_prefix_caching=False,
        speculative_config={
            "method": "eagle3",
            "model": args.draft_model,
            "num_speculative_tokens": 8,
            "max_model_len": PROMPT_TOKENS + OUTPUT_TOKENS,
        },
        **(
            {
                "profiler_config": ProfilerConfig(
                    profiler="torch",
                    torch_profiler_dir=args.profile_dir,
                    torch_profiler_with_stack=False,
                )
            }
            if args.profile_dir
            else {}
        ),
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    try:
        await measure_vllm_request(
            engine,
            make_prompt_token_ids(args.target_model, WARMUP_PROMPT_TOKENS),
            WARMUP_OUTPUT_TOKENS,
            "warmup",
        )
        await engine.do_log_stats()
        if args.profile_dir:
            await engine.start_profile()
            try:
                return summarize(
                    [
                        await measure_vllm_request(
                            engine,
                            make_prompt_token_ids(args.target_model, PROMPT_TOKENS),
                            OUTPUT_TOKENS,
                            "profile",
                        )
                    ]
                )
            finally:
                await engine.stop_profile()
        results = [
            await measure_vllm_request(
                engine,
                make_prompt_token_ids(args.target_model, PROMPT_TOKENS),
                OUTPUT_TOKENS,
                f"measure-{index}",
            )
            for index in range(NUM_RUNS)
        ]
        # Flush vLLM's interval logger so speculative draft/accept counts are
        # emitted for this short offline benchmark.
        await engine.do_log_stats()
        return summarize(results)
    finally:
        engine.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=("vllm", "nanovllm"))
    parser.add_argument("--target-model", required=True, help="Local target model directory")
    parser.add_argument("--draft-model", required=True, help="Local EAGLE3 draft model directory")
    parser.add_argument("--profile-dir")
    args = parser.parse_args()
    if args.profile_dir and args.backend != "vllm":
        parser.error("--profile-dir is currently supported only for vllm")
    result = (
        asyncio.run(run_vllm_async(args))
        if args.backend == "vllm"
        else run_nanovllm(args)
    )
    print(json.dumps({"backend": args.backend, **result}, indent=2))


if __name__ == "__main__":
    main()
