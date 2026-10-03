"""Four-GPU TP/PP/DP correctness check using a real Qwen3 checkpoint.

Run with ``CUDA_VISIBLE_DEVICES=1,2,3,4 python -m tests.test_parallel_integration``.
The baseline uses graph execution; PP uses eager because Config requires it.
"""

import argparse
import json
import subprocess
import sys

import torch


CASES = ("base", "tp2", "dp2", "pp2", "tp2pp2", "dp2tp2")


def run_case(model: str, case: str) -> dict:
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.data_parallel import generate_data_parallel

    prompts = [[100 + (i + request * 17) % 97 for i in range(64)] for request in range(4)]
    sampling = SamplingParams(temperature=0, ignore_eos=True, max_tokens=4)
    common = dict(
        max_model_len=80,
        max_num_batched_tokens=64,
        max_num_seqs=4,
        gpu_memory_utilization=0.25,
        enable_prefix_caching=False,
    )
    if case in ("dp2", "dp2tp2"):
        dp_tp = 2 if case == "dp2tp2" else 1
        outputs, replica_ids = generate_data_parallel(
            model, prompts, sampling, data_parallel_size=2,
            tensor_parallel_size=dp_tp, enforce_eager=False, **common,
        )
        assert replica_ids == [0, 1]
        return {"case": case, "tokens": [item["token_ids"] for item in outputs],
                "replica_ids": replica_ids}

    pp = 2 if case in ("pp2", "tp2pp2") else 1
    tp = 2 if case in ("tp2", "tp2pp2") else 1
    llm = LLM(model, tensor_parallel_size=tp, pipeline_parallel_size=pp,
              enforce_eager=pp > 1, **common)
    try:
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        diagnostics = llm.model_runner.call("get_diagnostics")
        assert len(diagnostics) == tp * pp
        assert bool(getattr(llm.model_runner, "graphs", {})) == (pp == 1)
        for item in diagnostics:
            assert item["tp_size"] == tp and item["pp_size"] == pp
            assert item["num_local_layers"] > 0
            assert item["kv_cache_layers"] == item["num_local_layers"]
            assert len(item["tp_group_ranks"]) == tp
            assert len(item["pp_group_ranks"]) == pp
            assert item["prefill_piece_enabled"] == (pp == 1)
            if pp > 1:
                assert item["pipeline_send_count"] + item["pipeline_recv_count"] > 0
            if case == "tp2pp2":
                expected_tp = [0, 1] if item["pp_rank"] == 0 else [2, 3]
                expected_pp = [item["tp_rank"], item["tp_rank"] + 2]
                assert item["tp_group_ranks"] == expected_tp
                assert item["pp_group_ranks"] == expected_pp
        if pp > 1:
            ordered = sorted(diagnostics, key=lambda item: item["start_layer"])
            assert ordered[0]["start_layer"] == 0
            assert ordered[-1]["end_layer"] == llm.config.hf_config.num_hidden_layers
        return {"case": case, "tokens": [item["token_ids"] for item in outputs],
                "diagnostics": diagnostics}
    finally:
        llm.exit()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./models/Qwen3-0.6B")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    parser.add_argument("--child-case", choices=CASES)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.child_case:
        print(json.dumps(run_case(args.model, args.child_case)))
        return
    results = {}
    for case in args.cases:
        completed = subprocess.run(
            [sys.executable, "-m", "tests.test_parallel_integration", "--model", args.model,
             "--child-case", case],
            check=False, capture_output=True, text=True, timeout=900,
        )
        if completed.returncode:
            raise RuntimeError(f"{case} failed:\n{completed.stderr}")
        results[case] = json.loads(completed.stdout.strip().splitlines()[-1])
        print(json.dumps({"case": case, "tokens": results[case]["tokens"]}), flush=True)
    if "base" in results:
        for case, result in results.items():
            assert result["tokens"] == results["base"]["tokens"], case
    print("parallel integration passed")


if __name__ == "__main__":
    main()
