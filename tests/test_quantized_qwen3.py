import argparse
import gc

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.layers.linear import LinearBase


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("quantization", choices=("none", "w4a16", "fp8"))
    parser.add_argument("--model", help="Local model directory for non-speculative tests")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--speculative", action="store_true")
    parser.add_argument("--target-model", help="Target model directory for EAGLE3")
    parser.add_argument("--draft-model", help="EAGLE3 draft model directory")
    parser.add_argument(
        "--fp8-format", choices=("per_tensor", "per_token"), default="per_tensor"
    )
    args = parser.parse_args()
    if args.speculative:
        if not args.target_model or not args.draft_model:
            parser.error("--speculative requires --target-model and --draft-model")
    elif not args.model:
        parser.error("--model is required without --speculative")

    quantization = None if args.quantization == "none" else args.quantization
    llm = LLM(
        args.target_model if args.speculative else args.model,
        quantization=quantization,
        fp8_format=args.fp8_format,
        enforce_eager=not args.cuda_graph,
        max_model_len=128,
        max_num_batched_tokens=128,
        max_num_seqs=4,
        gpu_memory_utilization=0.8 if args.speculative else 0.5,
        speculative_config=(
            {
                "method": "eagle3",
                "model": args.draft_model,
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
        if quantization == "fp8":
            assert all(
                layer.fp8_activation_format == args.fp8_format for layer in layers
            )
        output = llm.generate(
            [[1, 2, 3, 4]],
            SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=8),
            use_tqdm=False,
        )[0]
        assert len(output["token_ids"]) == 8
        if args.speculative:
            capabilities = llm.model_runner.get_eagle3_runtime_capabilities()
            assert capabilities["paged_kv_cache"] is True
            if args.cuda_graph:
                assert capabilities["cuda_graph"] is True
                assert capabilities["graph_batch_sizes"]
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
