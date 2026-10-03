"""Real-model graph/eager decode parity for a 32-request MoE batch.

Run with CUDA_VISIBLE_DEVICES=1,2,3,4 and
``python -m tests.test_moe_decode_graph_parity --ep 1`` or ``--ep 4``.
"""

import argparse
import json
import subprocess
import sys


MODEL = "./models/Qwen3-30B-A3B-Base"


def run_case(model, ep, eager, expert_capacity=None, capacity_factor=None):
    from nanovllm import LLM, SamplingParams
    from benchmarks.bench_moe_ep import make_prompts

    llm = LLM(
        model,
        tensor_parallel_size=ep,
        enable_expert_parallel=True,
        moe_expert_capacity=expert_capacity,
        moe_expert_capacity_factor=capacity_factor,
        enforce_eager=eager,
        enable_prefill_batching=True,
        enable_prefix_caching=False,
        max_model_len=28,
        max_num_batched_tokens=512,
        max_num_seqs=32,
        gpu_memory_utilization=0.9,
    )
    sampling = SamplingParams(temperature=0, ignore_eos=True, max_tokens=4)
    try:
        for seed in range(1, 4):
            llm.generate(
                make_prompts(seed, 32, 16), sampling, use_tqdm=False
            )
        before = {
            item["rank"]: item["decode_graph_replay_count"]
            for item in llm.model_runner.call("get_diagnostics")
        }
        outputs = llm.generate(
            make_prompts(100, 32, 16), sampling, use_tqdm=False
        )
        after = {
            item["rank"]: item["decode_graph_replay_count"]
            for item in llm.model_runner.call("get_diagnostics")
        }
        replay_counts = {
            rank: after[rank] - count for rank, count in before.items()
        }
        assert len(replay_counts) == ep
        assert all(count == (0 if eager else 3) for count in replay_counts.values())
        tokens = [output["token_ids"] for output in outputs]
        assert len(tokens) == 32 and all(len(row) == 4 for row in tokens)
        return {"mode": "eager" if eager else "graph", "ep": ep,
                "expert_capacity": expert_capacity,
                "capacity_factor": capacity_factor,
                "tokens": tokens, "replay_counts": replay_counts}
    finally:
        llm.exit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--ep", type=int, choices=(1, 4), default=1)
    parser.add_argument("--expert-capacity", type=int)
    parser.add_argument("--capacity-factor", type=float)
    parser.add_argument("--child-mode", choices=("eager", "graph"))
    args = parser.parse_args()
    if args.expert_capacity is not None and args.capacity_factor is not None:
        parser.error("fixed capacity and capacity factor are mutually exclusive")
    if args.child_mode:
        print(json.dumps(run_case(
            args.model, args.ep, args.child_mode == "eager", args.expert_capacity,
            args.capacity_factor,
        )))
        return
    results = {}
    for mode in ("eager", "graph"):
        command = [
            sys.executable, "-m", "tests.test_moe_decode_graph_parity",
            "--model", args.model, "--ep", str(args.ep), "--child-mode", mode,
        ]
        if args.expert_capacity is not None:
            command.extend(("--expert-capacity", str(args.expert_capacity)))
        if args.capacity_factor is not None:
            command.extend(("--capacity-factor", str(args.capacity_factor)))
        completed = subprocess.run(
            command,
            capture_output=True, text=True, timeout=900,
        )
        if completed.returncode:
            raise RuntimeError(f"{mode} failed:\n{completed.stderr}")
        results[mode] = json.loads(completed.stdout.strip().splitlines()[-1])
    assert results["eager"]["tokens"] == results["graph"]["tokens"]
    print(json.dumps({
        "ep": args.ep,
        "expert_capacity": args.expert_capacity,
        "capacity_factor": args.capacity_factor,
        "matched_sequences": len(results["graph"]["tokens"]),
        "graph_replays": results["graph"]["replay_counts"],
    }))


if __name__ == "__main__":
    main()
