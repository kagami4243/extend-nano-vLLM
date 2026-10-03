"""Qwen3-30B-A3B EP=1/2/4 correctness and shard diagnostics.

Run with CUDA_VISIBLE_DEVICES=1,2,3,4. Each topology uses a fresh process.
"""

import argparse
import json
import subprocess
import sys


MODEL = "./models/Qwen3-30B-A3B-Base"
CASES = ("ep1", "ep2", "ep4")


def run_case(model, case):
    from nanovllm import LLM, SamplingParams

    ep_size = int(case[-1])
    llm = LLM(
        model,
        tensor_parallel_size=ep_size,
        enable_expert_parallel=True,
        enforce_eager=True,
        enable_prefix_caching=False,
        max_model_len=32,
        max_num_batched_tokens=16,
        max_num_seqs=1,
        gpu_memory_utilization=0.9,
    )
    try:
        outputs = llm.generate(
            [[1000 + position % 13 for position in range(16)]],
            SamplingParams(temperature=0, ignore_eos=True, max_tokens=2),
            use_tqdm=False,
        )
        ranks = llm.model_runner.call("get_diagnostics")
        assert len(ranks) == ep_size
        per_rank_experts = llm.config.hf_config.num_experts // ep_size
        expected_ranges = [
            (rank * per_rank_experts, (rank + 1) * per_rank_experts)
            for rank in range(ep_size)
        ]
        actual_ranges = sorted(
            (rank["expert_start"], rank["expert_end"]) for rank in ranks
        )
        assert actual_ranges == expected_ranges
        assert all(rank["num_local_experts"] == per_rank_experts for rank in ranks)
        assert all(rank["ep_size"] == ep_size for rank in ranks)
        assert all(not rank["prefill_piece_enabled"] for rank in ranks)
        assert all(
            rank["dispatch_assignment_count"] == rank["return_assignment_count"]
            for rank in ranks
        )
        assert len({rank["total_assignment_count"] for rank in ranks}) == 1
        assert sum(rank["dispatch_assignment_count"] for rank in ranks) == (
            ranks[0]["total_assignment_count"]
        )
        if ep_size > 1:
            assert all(rank["ep_all_reduce_count"] > 0 for rank in ranks)
        return {
            "case": case,
            "tokens": outputs[0]["token_ids"],
            "expert_ranges": actual_ranges,
            "expert_parameter_bytes": [
                rank["expert_parameter_bytes"] for rank in ranks
            ],
            "total_assignments": ranks[0]["total_assignment_count"],
            "all_reduce_counts": [rank["ep_all_reduce_count"] for rank in ranks],
        }
    finally:
        llm.exit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    parser.add_argument("--child-case", choices=CASES)
    args = parser.parse_args()
    if args.child_case:
        print(json.dumps(run_case(args.model, args.child_case)))
        return
    results = {}
    for case in args.cases:
        completed = subprocess.run(
            [sys.executable, "-m", "tests.test_moe_ep_integration",
             "--model", args.model, "--child-case", case],
            capture_output=True, text=True, timeout=1800,
        )
        if completed.returncode:
            raise RuntimeError(f"{case} failed:\n{completed.stderr}")
        results[case] = json.loads(completed.stdout.strip().splitlines()[-1])
        print(json.dumps(results[case]), flush=True)
    if "ep1" in results:
        for case, result in results.items():
            assert result["tokens"] == results["ep1"]["tokens"], case
            expected_ratio = int(case[-1])
            assert results["ep1"]["expert_parameter_bytes"][0] == (
                result["expert_parameter_bytes"][0] * expected_ratio
            )
    print("MoE EP integration passed")


if __name__ == "__main__":
    main()
