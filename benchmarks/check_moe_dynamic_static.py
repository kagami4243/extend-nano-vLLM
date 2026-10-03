"""Cold-start placement output comparator for the live migration benchmark."""

import argparse
import json
import os
import tempfile

from benchmarks.bench_moe_ep import make_prompts
from nanovllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./models/Qwen3-30B-A3B-Base")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--result-file")
    args = parser.parse_args()
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(args.model)
    placement = {
        "model": args.model,
        "num_experts": hf_config.num_experts,
        "expert_parallel_size": 2,
        "layers": {
            str(layer): [expert % 2 for expert in range(hf_config.num_experts)]
            for layer in range(hf_config.num_hidden_layers)
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as file:
        json.dump(placement, file)
        placement_path = file.name
    llm = None
    try:
        llm = LLM(
            args.model, tensor_parallel_size=2, enable_expert_parallel=True,
            moe_expert_placement=placement_path, enforce_eager=False,
            enable_prefill_batching=True,
            enable_prefix_caching=False, max_model_len=28,
            max_num_batched_tokens=32, max_num_seqs=2,
            gpu_memory_utilization=0.9,
        )
        sampling = SamplingParams(temperature=0, ignore_eos=True, max_tokens=4)

        def generate(seed):
            return [item["token_ids"] for item in llm.generate(
                make_prompts(seed, 2, 16), sampling, use_tqdm=False,
            )]

        for seed in range(3):
            generate(seed)
        outputs = [generate(100 + seed) for seed in range(args.runs)]
        result = {"topology": "TP=EP=2 static target placement",
                  "outputs": outputs}
        print(json.dumps(result, indent=2), flush=True)
        if args.result_file:
            with open(args.result_file, "w") as file:
                json.dump(result, file, indent=2)
                file.write("\n")
    finally:
        if llm is not None:
            llm.exit()
        os.unlink(placement_path)


if __name__ == "__main__":
    main()
