"""Four-GPU synchronized DP x TP MoE dispatch contract."""

from argparse import Namespace
import argparse

from benchmarks.bench_moe_global_dp import MODEL, run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--helper", action="store_true")
    parser.add_argument("--padding", action="store_true")
    args = parser.parse_args()
    if args.helper or args.padding:
        from nanovllm import SamplingParams
        from nanovllm.engine.data_parallel import generate_data_parallel

        lengths = (7, 23, 11, 29) if args.padding else (16,) * 4
        prompts = [[1000 + request] + [100 + position % 97 for position in range(length - 1)]
                   for request, length in enumerate(lengths)]
        outputs, replicas = generate_data_parallel(
            MODEL, prompts,
            SamplingParams(temperature=0, ignore_eos=True, max_tokens=4),
            data_parallel_size=2 if args.padding else 4,
            tensor_parallel_size=2 if args.padding else 1,
            enable_expert_parallel=True, enforce_eager=False,
            moe_prefill_piece=True, moe_prefill_piece_capture_sizes=(16,),
            max_model_len=40 if args.padding else 28,
            max_num_batched_tokens=16, max_num_seqs=1,
            gpu_memory_utilization=0.85,
        )
        assert replicas == ([0, 1] if args.padding else [0, 1, 2, 3])
        assert len(outputs) == 4
        assert all(len(item["token_ids"]) == 4 for item in outputs)
        print("offline helper padded DP=2 TP=2 EP=4 MoE integration passed"
              if args.padding else "offline helper DP=4 TP=1 EP=4 MoE integration passed")
        return
    args = Namespace(
        model=MODEL, prompt_tokens=16, output_tokens=4,
        warmup_runs=1, runs=1, gpu_memory_utilization=0.85,
        timeout=300, disable_prefix_cache=False, moe_prefill_piece=True,
        profile_trace=None, dispatch_backend="allgather_reduce",
    )
    result = run(args)
    assert result["effective_ep"] == 4
    assert len(result["rank_results"]) == 2
    for replica in result["rank_results"]:
        assert replica["dispatch_backend"] == "allgather_reduce"
        assert replica["prefill_piece_captured_sizes"] == [32]
        assert len(replica["first_outputs"]) == 2
        assert all(len(tokens) == 4 for tokens in replica["first_outputs"])
        assert all(count > 0 for count in replica["decode_graph_replays"])
        assert len(replica["diagnostics"]) == 2
        for worker in replica["diagnostics"]:
            assert worker["ep_size"] == 4
            assert worker["moe_group_ranks"] == [0, 1, 2, 3]
            assert worker["num_local_experts"] == 32
    print("synchronized DP=2 TP=2 EP=4 MoE integration passed")


if __name__ == "__main__":
    main()
