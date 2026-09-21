"""GPU integration check for eager versus piecewise prefill.

Run with:
  python -m tests.test_cuda_graph_piece_integration --model /path/to/Qwen3-0.6B
"""

import argparse
import gc
import json
import subprocess
import sys

import torch


def generate(
    model: str,
    enforce_eager: bool,
    quantization: str | None,
    fp8_format: str,
    kv_cache_dtype: str,
) -> tuple[list[list[int]], dict]:
    from nanovllm import LLM, SamplingParams

    llm = LLM(
        model,
        quantization=quantization,
        fp8_format=fp8_format,
        kv_cache_dtype=kv_cache_dtype,
        enforce_eager=enforce_eager,
        max_model_len=128,
        max_num_batched_tokens=128,
        max_num_seqs=2,
        # The driver constructs eager and graph engines sequentially. Keep
        # enough headroom for both KV-cache allocations after allocator
        # teardown between runs.
        gpu_memory_utilization=0.5,
        enable_prefix_caching=False,
    )
    try:
        outputs = llm.generate(
            [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]],
            SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=8),
            use_tqdm=False,
        )
        token_ids = [output["token_ids"] for output in outputs]
        runner = llm.model_runner
        evidence = {
            "prefill_piece_enabled": runner.prefill_piece_enabled,
            "prefill_piece_token_counts": sorted(runner.prefill_piece_graphs),
        }
        return token_ids, evidence
    finally:
        llm.exit()
        del llm
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local Qwen3 model directory")
    parser.add_argument("--mode", choices=("both", "eager", "graph"), default="both")
    parser.add_argument("--quantization", choices=("none", "fp8"), default="none")
    parser.add_argument(
        "--fp8-format", choices=("per_tensor", "per_token"), default="per_tensor"
    )
    parser.add_argument("--kv-cache-dtype", choices=("auto", "fp8"), default="auto")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for piecewise graph integration")

    if args.mode != "both":
        tokens, evidence = generate(
            args.model,
            args.mode == "eager",
            None if args.quantization == "none" else args.quantization,
            args.fp8_format,
            args.kv_cache_dtype,
        )
        print(json.dumps({"tokens": tokens, "evidence": evidence}))
        return

    # CUDA graph allocations are process-owned. Run eager and graph checks in
    # separate child processes so allocator state from the first engine cannot
    # starve the second engine's KV-cache allocation.
    def run_child(mode: str) -> tuple[list[list[int]], dict]:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.test_cuda_graph_piece_integration",
                "--model",
                args.model,
                "--mode",
                mode,
                "--quantization",
                args.quantization,
                "--fp8-format",
                args.fp8_format,
                "--kv-cache-dtype",
                args.kv_cache_dtype,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        return payload["tokens"], payload["evidence"]

    eager_tokens, eager_evidence = run_child("eager")
    graph_tokens, graph_evidence = run_child("graph")
    assert eager_tokens == graph_tokens, (eager_tokens, graph_tokens)
    assert eager_evidence["prefill_piece_enabled"] is False
    per_token_fp8 = (
        args.quantization == "fp8" and args.fp8_format == "per_token"
    )
    assert graph_evidence["prefill_piece_enabled"] is not per_token_fp8
    if per_token_fp8:
        assert not graph_evidence["prefill_piece_token_counts"]
    else:
        assert graph_evidence["prefill_piece_token_counts"]
    print(
        "piecewise prefill passed:",
        {"tokens": graph_tokens, "graph": graph_evidence},
    )


if __name__ == "__main__":
    main()
