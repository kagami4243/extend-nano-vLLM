"""Strict BF16 MoE prefill output checks against recorded references."""

import argparse
import json
import subprocess
import sys


WORKLOADS = {
    "b8p512": {
        "batch_size": 8,
        "prompt_tokens": 512,
        "expected": [[126], [246], [9216], [233], [110], [228], [249], [250]],
    },
    "b16p1024": {
        "batch_size": 16,
        "prompt_tokens": 1024,
        "expected": [
            [251], [252], [253], [223], [122], [95], [95], [242],
            [249], [32926], [32926], [32926], [18535], [123], [48800], [168],
        ],
    },
    "b16p1024o8": {
        "batch_size": 16,
        "prompt_tokens": 1024,
        "output_tokens": 8,
        "expected": [
            [251, 153, 252, 155, 251, 156, 95, 251],
            [252, 155, 223, 156, 95, 225, 156, 95],
            [253, 156, 95, 254, 156, 96, 254, 156],
            [223, 189, 190, 191, 192, 193, 194, 195],
            [122, 94, 159, 96, 94, 159, 96, 95],
            [95, 222, 159, 242, 222, 160, 246, 222],
            [95, 222, 159, 242, 222, 160, 246, 222],
            [242, 222, 160, 249, 222, 160, 249, 223],
            [249, 222, 160, 249, 223, 160, 249, 224],
            [32926, 163, 63219, 166, 95, 94, 166, 95],
            [32926, 164, 63219, 166, 95, 254, 166, 96],
            [32926, 166, 95, 94, 166, 95, 95, 166],
            [18535, 166, 95, 94, 166, 95, 95, 166],
            [123, 166, 95, 123, 166, 95, 123, 166],
            [48800, 168, 57160, 238, 120, 175, 123, 123],
            [168, 57160, 237, 108, 175, 123, 123, 123],
        ],
    },
}


def run_case(model, tp, pp, ep, dispatch_backend, workload, enable_prefill_batching,
             eager_op=None, shard_across_tp=False):
    command = [
        sys.executable, "-m", "benchmarks.bench_moe_ep",
        "--backend", "nano", "--model", model,
        "--tp", str(ep if tp is None else tp), "--ep", str(ep), "--pp", str(pp),
        "--dispatch-backend", dispatch_backend,
        "--batch-size", str(workload["batch_size"]),
        "--prompt-tokens", str(workload["prompt_tokens"]),
        "--output-tokens", str(workload.get("output_tokens", 1)),
        "--warmup-runs", "1", "--runs", "1",
        "--repeat-seed", "100", "--disable-prefix-cache",
    ]
    if dispatch_backend in ("all_to_all", "all_to_all_reduce") or pp > 1:
        command.append("--enforce-eager")
    if enable_prefill_batching:
        command.append("--enable-prefill-batching")
    if eager_op is not None:
        command.extend(("--eager-op", eager_op))
    if shard_across_tp:
        command.append("--shard-across-tp")
    if dispatch_backend == "replicated" and pp == 1 and workload.get("output_tokens", 1) > 1:
        command.append("--require-graph-replay")
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=900,
    )
    if completed.returncode:
        raise RuntimeError(f"EP={ep} failed:\n{completed.stderr}")
    return json.loads(completed.stdout.strip().splitlines()[-1])["first_output"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./models/Qwen3-30B-A3B-Base")
    parser.add_argument("--tp", type=int)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--ep", nargs="+", type=int, choices=(1, 2, 4), default=(1, 4))
    parser.add_argument("--dispatch-backend", choices=("replicated", "all_to_all", "all_to_all_reduce"),
                        default="replicated")
    parser.add_argument("--workload", choices=tuple(WORKLOADS), default="b8p512")
    parser.add_argument("--enable-prefill-batching", action="store_true")
    parser.add_argument("--shard-across-tp", action="store_true")
    parser.add_argument("--compare-to-ep-only", action="store_true")
    parser.add_argument("--eager-op", choices=("qk_norm", "residual_norm", "rope", "sampler"))
    args = parser.parse_args()
    workload = WORKLOADS[args.workload]
    if args.compare_to_ep_only or args.shard_across_tp:
        parser.error("independent EP axes are removed; this DP=1 test uses EP=TP")
    if args.tp is not None and any(args.tp != ep for ep in args.ep):
        parser.error("EP must equal TP for every selected DP=1 configuration")
    outputs = {
        ep: run_case(
            args.model, args.tp, args.pp, ep, args.dispatch_backend,
            workload, args.enable_prefill_batching, args.eager_op,
            args.shard_across_tp,
        )
        for ep in args.ep
    }
    print(json.dumps({"expected": workload["expected"], "actual": outputs}), flush=True)
    for ep, actual in outputs.items():
        assert actual == workload["expected"], f"EP={ep} differs from reference: {actual}"


if __name__ == "__main__":
    main()
