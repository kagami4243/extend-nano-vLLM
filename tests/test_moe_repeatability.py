"""Repeat a near-tie MoE prompt across four-GPU parallel layouts."""

import argparse
import json
import subprocess
import sys

from benchmarks.bench_moe_ep import make_prompts


MODEL = "./models/Qwen3-30B-A3B-Base"
EXPECTED_OUTPUTS = [[198, 220, 7288, 220], [170, 222, 222, 170]]
CASES = {
    "ep1": (1, 1, 1, "replicated", True),
    "ep4": (4, 1, 4, "replicated", True),
    "tp2ep2": (2, 1, 2, "replicated", True),
    "pp2ep2": (2, 2, 2, "replicated", False),
    "ep4_alltoall": (4, 1, 4, "all_to_all", False),
    "ep4_alltoall_reduce": (4, 1, 4, "all_to_all_reduce", False),
}


def run_case(case, repeats, placement_file, observe_only=False):
    from nanovllm import LLM, SamplingParams

    tp, pp, ep, backend, graph = CASES[case]
    llm = LLM(
        MODEL,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        expert_parallel_size=ep,
        enable_expert_parallel=True,
        moe_dispatch_backend=backend,
        moe_expert_placement=placement_file,
        enforce_eager=not graph,
        enable_prefix_caching=False,
        max_model_len=28,
        max_num_batched_tokens=32,
        max_num_seqs=2,
        gpu_memory_utilization=0.9,
    )
    sampling = SamplingParams(temperature=0, ignore_eos=True, max_tokens=4)
    try:
        for seed in (1, 2):
            llm.generate(make_prompts(seed, 2, 16), sampling, use_tqdm=False)
        outputs = [
            [item["token_ids"] for item in llm.generate(
                make_prompts(100, 2, 16), sampling, use_tqdm=False
            )]
            for _ in range(repeats)
        ]
        assert all(output == outputs[0] for output in outputs[1:]), outputs
        if not observe_only:
            assert outputs[0] == EXPECTED_OUTPUTS, outputs[0]
        diagnostics = llm.model_runner.call("get_diagnostics")
        if graph:
            assert all(rank["decode_graph_replay_count"] > 0 for rank in diagnostics)
        return {"case": case, "outputs": outputs[0], "repeats": repeats}
    finally:
        llm.exit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", choices=CASES,
                        default=["ep1", "ep4", "ep4_alltoall"])
    parser.add_argument("--child-case", choices=CASES)
    parser.add_argument("--placement-file")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--observe-only", action="store_true")
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("repeats must be at least two")
    if args.placement_file and any(
        CASES[case][2] != 4 for case in
        ([args.child_case] if args.child_case else args.cases)
    ):
        parser.error("placement file requires EP=4")
    if args.child_case:
        print(json.dumps(run_case(
            args.child_case, args.repeats, args.placement_file,
            args.observe_only,
        )))
        return
    results = {}
    for case in args.cases:
        command = [
            sys.executable, "-m", "tests.test_moe_repeatability",
            "--child-case", case, "--repeats", str(args.repeats),
        ]
        if args.placement_file:
            command.extend(("--placement-file", args.placement_file))
        if args.observe_only:
            command.append("--observe-only")
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=1800
        )
        if completed.returncode:
            raise RuntimeError(f"{case} failed:\n{completed.stderr}")
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        results[case] = result["outputs"]
        print(json.dumps(result), flush=True)
    if "ep1" in results and not args.observe_only:
        for case, outputs in results.items():
            assert outputs == results["ep1"], case
    print("MoE per-case repeatability passed" if args.observe_only
          else "MoE repeatability passed")


if __name__ == "__main__":
    main()
