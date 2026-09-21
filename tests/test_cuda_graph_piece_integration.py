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


def generate(model: str, enforce_eager: bool) -> tuple[list[int], dict]:
    from nanovllm import LLM, SamplingParams

    llm = LLM(
        model,
        enforce_eager=enforce_eager,
        max_model_len=128,
        max_num_batched_tokens=128,
        max_num_seqs=1,
        # The driver constructs eager and graph engines sequentially. Keep
        # enough headroom for both KV-cache allocations after allocator
        # teardown between runs.
        gpu_memory_utilization=0.5,
        enable_prefix_caching=False,
    )
    try:
        output = llm.generate(
            [[1, 2, 3, 4, 5, 6]],
            SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=8),
            use_tqdm=False,
        )[0]["token_ids"]
        runner = llm.model_runner
        evidence = {
            "prefill_piece_enabled": runner.prefill_piece_enabled,
            "prefill_piece_token_counts": sorted(runner.prefill_piece_graphs),
        }
        return output, evidence
    finally:
        llm.exit()
        del llm
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="/data0/fwy/Codes/model/Qwen3-0.6B"
    )
    parser.add_argument("--mode", choices=("both", "eager", "graph"), default="both")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for piecewise graph integration")

    if args.mode != "both":
        tokens, evidence = generate(args.model, args.mode == "eager")
        print(json.dumps({"tokens": tokens, "evidence": evidence}))
        return

    # CUDA graph allocations are process-owned. Run eager and graph checks in
    # separate child processes so allocator state from the first engine cannot
    # starve the second engine's KV-cache allocation.
    def run_child(mode: str) -> tuple[list[int], dict]:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.test_cuda_graph_piece_integration",
                "--model",
                args.model,
                "--mode",
                mode,
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
    assert graph_evidence["prefill_piece_enabled"] is True
    assert graph_evidence["prefill_piece_token_counts"]
    print(
        "piecewise prefill passed:",
        {"tokens": graph_tokens, "graph": graph_evidence},
    )


if __name__ == "__main__":
    main()
