import argparse
import gc

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.layers.linear import LinearBase


MODEL = "/data0/fwy/Codes/model/Qwen3-0.6B"
SPECULATIVE_TARGET = "/data1/model/qwen/Qwen/Qwen3-8B"
EAGLE3_MODEL = "/data0/fwy/Codes/model/Qwen3-8B-speculator.eagle3"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("quantization", choices=("none", "w4a16", "fp8"))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--speculative", action="store_true")
    args = parser.parse_args()

    quantization = None if args.quantization == "none" else args.quantization
    llm = LLM(
        SPECULATIVE_TARGET if args.speculative else args.model,
        quantization=quantization,
        enforce_eager=not args.cuda_graph,
        max_model_len=128,
        max_num_batched_tokens=128,
        max_num_seqs=4,
        gpu_memory_utilization=0.8 if args.speculative else 0.5,
        speculative_config=(
            {
                "method": "eagle3",
                "model": EAGLE3_MODEL,
                "num_speculative_tokens": 3,
            }
            if args.speculative
            else None
        ),
    )
    try:
        layers = [
            module
            for module in llm.model_runner.model.modules()
            if isinstance(module, LinearBase)
        ]
        assert layers and all(layer.quantization == quantization for layer in layers)
        output = llm.generate(
            [[1, 2, 3, 4]],
            SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=8),
            use_tqdm=False,
        )[0]
        assert len(output["token_ids"]) == 8
        parameter_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in llm.model_runner.model.parameters()
        )
        print(
            f"{args.quantization}: layers={len(layers)}, "
            f"parameter_bytes={parameter_bytes}, tokens={output['token_ids']}"
        )
    finally:
        llm.exit()
        del llm
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
