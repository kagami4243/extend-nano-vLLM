"""Two-GPU Qwen3-MoE live expert migration with default EP=TP topology."""

import json

from nanovllm import LLM, SamplingParams


MODEL = "./models/Qwen3-30B-A3B-Base"


def main():
    llm = LLM(
        MODEL, tensor_parallel_size=2, enable_expert_parallel=True,
        moe_dynamic_placement=True, enforce_eager=False,
        enable_prefix_caching=False, max_model_len=32,
        max_num_batched_tokens=16, max_num_seqs=1,
        gpu_memory_utilization=0.9,
    )
    prompt = [1000 + position % 13 for position in range(16)]
    sampling = SamplingParams(temperature=0, ignore_eos=True, max_tokens=2)
    try:
        before = llm.generate([prompt], sampling, use_tqdm=False)[0]["token_ids"]
        prior = llm.model_runner.call("get_diagnostics")
        assert all(rank["decode_graph_replay_count"] > 0 for rank in prior)
        experts = llm.config.hf_config.num_experts
        placement = {
            str(layer): [expert % 2 for expert in range(experts)]
            for layer in range(llm.config.hf_config.num_hidden_layers)
        }
        moved = llm.relocate_experts(placement)
        assert all(all(count > 0 for count in rank.values()) for rank in moved)
        after = llm.generate([prompt], sampling, use_tqdm=False)[0]["token_ids"]
        current = llm.model_runner.call("get_diagnostics")
        assert before == after, (before, after)
        for rank in current:
            assert rank["local_expert_ids"] == [
                expert for expert in range(experts)
                if expert % 2 == rank["ep_rank"]
            ]
            assert (rank["decode_graph_replay_count"]
                    > prior[rank["rank"]]["decode_graph_replay_count"])
        print(json.dumps({"before": before, "after": after,
                          "moved_per_rank": [sum(rank.values()) for rank in moved]}))
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
