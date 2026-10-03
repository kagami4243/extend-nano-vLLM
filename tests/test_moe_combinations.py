"""Qwen3-MoE correctness for EP=TP and PP x TP on up to four GPUs."""

import argparse
import json
import subprocess
import sys


MODEL = "./models/Qwen3-30B-A3B-Base"
CASES = {
    "ep1": (1, 1, 1),
    "ep4": (4, 1, 4),
    "tp2ep2": (2, 1, 2),
    "pp2ep2": (2, 2, 2),
}


def run_case(model, case, dispatch_backend, expert_capacity, graph, placement_file,
             shard_across_tp=False):
    from nanovllm import LLM, SamplingParams

    tp, pp, ep = CASES[case]
    llm = LLM(
        model,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        expert_parallel_size=ep,
        enable_expert_parallel=True,
        moe_dispatch_backend=dispatch_backend,
        moe_expert_capacity=expert_capacity,
        moe_expert_placement=placement_file,
        moe_shard_across_tp=shard_across_tp,
        enforce_eager=not graph,
        enable_prefix_caching=False,
        max_model_len=32,
        max_num_batched_tokens=16,
        max_num_seqs=1,
        gpu_memory_utilization=0.9,
    )
    try:
        placement = None
        if placement_file:
            with open(placement_file) as file:
                placement = json.load(file)
        outputs = llm.generate(
            [[1000 + position % 13 for position in range(16)]],
            SamplingParams(temperature=0, ignore_eos=True, max_tokens=2),
            use_tqdm=False,
        )
        diagnostics = llm.model_runner.call("get_diagnostics")
        assert len(diagnostics) == tp * pp
        experts_per_rank = llm.config.hf_config.num_experts // ep
        for rank in diagnostics:
            assert (rank["tp_size"], rank["pp_size"], rank["ep_size"]) == (tp, pp, ep)
            assert len(rank["tp_group_ranks"]) == tp
            assert len(rank["pp_group_ranks"]) == pp
            assert len(rank["ep_group_ranks"]) == ep
            assert rank["num_local_experts"] == experts_per_rank
            assert rank["moe_size"] == ep
            assert rank["moe_group_ranks"] == rank["ep_group_ranks"]
            if placement:
                owners = placement["layers"]["0"]
                expected_ids = [
                    expert for expert, owner in enumerate(owners)
                    if owner == rank["ep_rank"]
                ]
                assert rank["local_expert_ids"] == expected_ids
                assert rank["expert_start"] == rank["expert_end"] == -1
            else:
                owner_rank = rank["ep_rank"]
                assert rank["expert_start"] == owner_rank * experts_per_rank
                assert rank["expert_end"] == (owner_rank + 1) * experts_per_rank
            assert rank["kv_cache_layers"] == rank["num_local_layers"]
            assert rank["dispatch_assignment_count"] == rank["return_assignment_count"]
            if graph:
                assert rank["decode_graph_replay_count"] > 0
                assert not rank["diagnostic_counters_include_graph_replays"]
            if ep > 1 and dispatch_backend == "all_to_all":
                assert rank["ep_all_to_all_count"] > 0
                assert rank["ep_all_reduce_count"] == 0
                assert rank["ep_broadcast_count"] > 0
            if ep > 1 and dispatch_backend == "all_to_all_reduce":
                assert rank["ep_all_to_all_count"] > 0
                assert rank["ep_all_reduce_count"] > 0
                assert rank["ep_broadcast_count"] == 0
            if expert_capacity is not None and rank["ep_rank"] == 0:
                assert rank["dropped_assignment_count"] > 0
            if pp > 1:
                assert rank["pipeline_send_count"] + rank["pipeline_recv_count"] > 0
        for stage in range(pp):
            assert sorted(
                expert for rank in diagnostics if rank["pp_rank"] == stage
                for expert in rank["local_expert_ids"]
            ) == list(range(llm.config.hf_config.num_experts))
        tokens = outputs[0]["token_ids"]
        if placement:
            assert tokens == [1003, 1004]
        return {"case": case, "tokens": tokens, "ranks": diagnostics}
    finally:
        llm.exit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--cases", nargs="+", choices=CASES)
    parser.add_argument("--child-case", choices=CASES)
    parser.add_argument("--dispatch-backend", choices=("replicated", "all_to_all", "all_to_all_reduce"),
                        default="replicated")
    parser.add_argument("--expert-capacity", type=int)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--placement-file")
    parser.add_argument("--shard-across-tp", action="store_true")
    args = parser.parse_args()
    if args.cases is None:
        args.cases = [
            case for case, (_, pp, _) in CASES.items()
            if not args.graph or pp == 1
        ]
    if args.graph and (args.dispatch_backend != "replicated" or args.expert_capacity):
        parser.error("graph mode requires replicated dispatch without capacity")
    if args.graph and any(CASES[case][1] > 1 for case in args.cases):
        parser.error("pipeline parallelism currently requires eager mode")
    selected_cases = [args.child_case] if args.child_case else args.cases
    if args.shard_across_tp:
        parser.error("independent EP axes are removed; EP must equal DP * TP")
    if args.placement_file and any(CASES[case][2] != 4 for case in selected_cases):
        parser.error("placement file currently describes EP=4")
    if args.child_case:
        print(json.dumps(run_case(
            args.model, args.child_case, args.dispatch_backend,
            args.expert_capacity, args.graph, args.placement_file,
            args.shard_across_tp,
        )))
        return
    results = {}
    for case in args.cases:
        command = [
            sys.executable, "-m", "tests.test_moe_combinations",
            "--model", args.model, "--child-case", case,
            "--dispatch-backend", args.dispatch_backend,
        ]
        if args.expert_capacity is not None:
            command.extend(("--expert-capacity", str(args.expert_capacity)))
        if args.graph:
            command.append("--graph")
        if args.placement_file:
            command.extend(("--placement-file", args.placement_file))
        if args.shard_across_tp:
            command.append("--shard-across-tp")
        completed = subprocess.run(
            command,
            capture_output=True, text=True, timeout=1800,
        )
        if completed.returncode:
            raise RuntimeError(f"{case} failed:\n{completed.stderr}")
        results[case] = json.loads(completed.stdout.strip().splitlines()[-1])
        print(json.dumps({"case": case, "tokens": results[case]["tokens"]}), flush=True)
    if "ep1" in results:
        for case, result in results.items():
            assert result["tokens"] == results["ep1"]["tokens"], case
    print("MoE combinations passed")


if __name__ == "__main__":
    main()
