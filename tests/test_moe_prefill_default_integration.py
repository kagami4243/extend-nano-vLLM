"""Real-model smoke for the default MoE multi-request prefill path."""

import json

from nanovllm import LLM, SamplingParams


MODEL = "./models/Qwen3-30B-A3B-Base"


def main():
    prompts = [
        [1000 + request] + [100 + (position + request) % 97 for position in range(15)]
        for request in range(4)
    ]
    llm = LLM(
        MODEL,
        max_model_len=28,
        max_num_batched_tokens=64,
        max_num_seqs=4,
        enable_prefix_caching=False,
        enforce_eager=False,
        gpu_memory_utilization=0.9,
    )
    sampling = SamplingParams(temperature=0, ignore_eos=True, max_tokens=4)
    try:
        assert llm.config.enable_prefill_batching is True
        before = llm.scheduler.stats["prefill_steps"]
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        prefill_steps = llm.scheduler.stats["prefill_steps"] - before
        assert prefill_steps == 1, prefill_steps
        assert all(len(output["token_ids"]) == 4 for output in outputs)
        assert llm.model_runner.graphs
        print(json.dumps({
            "prefill_batching": llm.config.enable_prefill_batching,
            "prefill_steps": prefill_steps,
            "decode_graph_sizes": sorted(llm.model_runner.graphs),
            "outputs": [output["token_ids"] for output in outputs],
        }))
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
