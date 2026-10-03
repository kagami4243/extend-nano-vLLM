"""Exact four-rank BF16 MoE dispatch parity; run as a module on four GPUs."""

import argparse
import json
import socket
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, port, results, real_shape, checkpoint, trace_input):
    from nanovllm.distributed.parallel_state import (
        destroy_model_parallel, initialize_model_parallel,
    )
    from nanovllm.layers.moe import ExpertParallelMoE

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}",
        rank=rank, world_size=4, device_id=torch.device("cuda", rank),
    )
    initialize_model_parallel(4, 1, True)
    try:
        torch.manual_seed(31)
        hidden_size, intermediate_size, num_experts, top_k, num_tokens = (
            (2048, 768, 128, 8, 4096) if real_shape or checkpoint
            else (128, 256, 16, 4, 128)
        )
        replicated = ExpertParallelMoE(
            hidden_size, intermediate_size, num_experts, top_k, True,
            ep_rank=rank, ep_size=4,
        ).cuda().bfloat16()
        hybrid = ExpertParallelMoE(
            hidden_size, intermediate_size, num_experts, top_k, True,
            ep_rank=rank, ep_size=4,
            dispatch_backend="all_to_all_reduce",
        ).cuda().bfloat16()
        with torch.no_grad():
            if checkpoint:
                from safetensors import safe_open

                index = json.loads((Path(checkpoint) / "model.safetensors.index.json").read_text())
                prefix = "model.layers.1.mlp."
                shard = index["weight_map"][prefix + "gate.weight"]
                with safe_open(str(Path(checkpoint) / shard), framework="pt", device="cpu") as weights:
                    replicated.gate.weight.copy_(weights.get_tensor(prefix + "gate.weight"))
                    for expert in replicated.local_expert_ids:
                        for projection in ("gate_proj", "up_proj", "down_proj"):
                            name = prefix + f"experts.{expert}.{projection}.weight"
                            replicated.load_expert_weight(
                                expert, projection, weights.get_tensor(name)
                            )
            else:
                for parameter in replicated.parameters():
                    parameter.normal_(0, 0.04)
            hybrid.load_state_dict(replicated.state_dict())
            hidden = (
                torch.load(trace_input, map_location="cpu", weights_only=True)
                ["vectors"]["1.mlp_input_span"].to(device="cuda", dtype=torch.bfloat16)
                if trace_input else torch.randn(
                    num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16
                )
            )
            reference = replicated(hidden)
            actual = hybrid(hidden)
        difference = (reference.float() - actual.float()).abs()
        different_rows = torch.nonzero((difference != 0).any(dim=1)).flatten()
        results.put({
            "rank": rank,
            "different_elements": int(torch.count_nonzero(difference)),
            "different_rows": different_rows[:20].tolist(),
            "max_abs": float(difference.max()),
            "exact": torch.equal(reference, actual),
        })
    finally:
        destroy_model_parallel()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-shape", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--trace-input")
    args = parser.parse_args()
    if bool(args.checkpoint) != bool(args.trace_input):
        parser.error("checkpoint and trace input must be supplied together")
    if torch.cuda.device_count() != 4:
        raise RuntimeError("set CUDA_VISIBLE_DEVICES to exactly four GPUs")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    ctx = mp.get_context("spawn")
    results = ctx.Queue()
    processes = [
        ctx.Process(target=worker, args=(
            rank, port, results, args.real_shape, args.checkpoint, args.trace_input,
        ))
        for rank in range(4)
    ]
    try:
        for process in processes:
            process.start()
        observations = sorted(
            (results.get(timeout=120) for _ in processes),
            key=lambda item: item["rank"],
        )
        print(json.dumps(observations), flush=True)
        assert all(item["exact"] for item in observations)
    finally:
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join()
        assert all(process.exitcode == 0 for process in processes)


if __name__ == "__main__":
    main()
