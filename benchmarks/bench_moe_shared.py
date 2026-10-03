"""Graph replay latency for a real Qwen3-Next MoE layer, with shared expert.

Run with torchrun --standalone --nproc_per_node=EP and matching --ep EP.
Only model.layers.0.mlp weights are read from the checkpoint.
"""

import argparse
import json
import os
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors import safe_open

from nanovllm.distributed.parallel_state import (
    destroy_model_parallel, get_moe_rank, get_moe_world_size,
    get_tp_group, initialize_model_parallel,
)
from nanovllm.layers.moe import ExpertParallelMoE


def load_layer(module, checkpoint):
    shard = checkpoint / "model-00001-of-00041.safetensors"
    prefix = "model.layers.0.mlp."
    with safe_open(shard, framework="pt", device="cpu") as weights:
        with torch.no_grad():
            module.gate.weight.copy_(weights.get_tensor(prefix + "gate.weight"))
            for expert_id in module.local_expert_ids:
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    tensor = weights.get_tensor(
                        f"{prefix}experts.{expert_id}.{projection}.weight"
                    )
                    module.load_expert_weight(expert_id, projection, tensor)
            if module.shared_expert is not None:
                shared = module.shared_expert
                shared.gate_up_proj.weight.weight_loader(
                    shared.gate_up_proj.weight,
                    weights.get_tensor(prefix + "shared_expert.gate_proj.weight"),
                    0,
                )
                shared.gate_up_proj.weight.weight_loader(
                    shared.gate_up_proj.weight,
                    weights.get_tensor(prefix + "shared_expert.up_proj.weight"),
                    1,
                )
                shared.down_proj.weight.weight_loader(
                    shared.down_proj.weight,
                    weights.get_tensor(prefix + "shared_expert.down_proj.weight"),
                )
                module.shared_expert_gate.weight.copy_(
                    weights.get_tensor(prefix + "shared_expert_gate.weight")
                )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path(
        "./models/Qwen3-Next-80B-A3B-Instruct"
    ))
    parser.add_argument("--ep", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--tp", type=int, choices=(1, 2, 4))
    parser.add_argument("--dp", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--shard-across-tp", action="store_true")
    parser.add_argument("--shared-tp-sharded", action="store_true")
    parser.add_argument("--tokens", type=int, choices=(16, 32), default=16)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--result-file", type=Path)
    parser.add_argument("--output-file", type=Path)
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.tp is None:
        if args.ep % args.dp:
            parser.error("EP must divide evenly across DP ranks")
        args.tp = args.ep // args.dp
    if args.ep != args.dp * args.tp or args.ep != int(os.environ.get("WORLD_SIZE", "1")):
        parser.error("--ep must equal --dp * --tp and torchrun world size")
    if args.shard_across_tp:
        parser.error("independent EP axes are removed; EP is DP * TP")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    initialize_model_parallel(
        args.tp, 1, True, data_parallel_size=args.dp
    )
    try:
        config = json.loads((args.model / "config.json").read_text())
        hidden_size = config["hidden_size"]
        shared_size = config["shared_expert_intermediate_size"]
        torch.manual_seed(71)
        hidden = torch.randn(args.tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
        results = {}
        routed_output = None
        output_snapshots = {}
        for with_shared in (False, True):
            module = ExpertParallelMoE(
                hidden_size, config["moe_intermediate_size"], config["num_experts"],
                config["num_experts_per_tok"], config["norm_topk_prob"],
                ep_rank=get_moe_rank(), ep_size=get_moe_world_size(),
                dispatch_backend="allgather_reduce" if args.dp > 1 else "replicated",
                graph_safe_decode=True,
                shared_expert_intermediate_size=shared_size if with_shared else None,
                shared_expert_tp_sharded=args.shared_tp_sharded,
            ).bfloat16()
            load_layer(module, args.model)
            module.cuda()
            for _ in range(args.warmup):
                module(hidden)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                module(hidden)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = module(hidden)
            graph.replay()
            torch.cuda.synchronize()
            if with_shared:
                shared = module.shared_expert
                gate, up = F.linear(
                    hidden, shared.gate_up_proj.weight
                ).chunk(2, dim=-1)
                projected = F.linear(
                    F.silu(gate) * up, shared.down_proj.weight
                )
                if args.tp > 1 and args.shared_tp_sharded:
                    # Row-parallel down projection sums partial outputs first.
                    dist.all_reduce(projected, group=get_tp_group())
                expected_shared = torch.sigmoid(F.linear(
                    hidden, module.shared_expert_gate.weight
                )) * projected
                torch.testing.assert_close(
                    output, routed_output + expected_shared,
                    rtol=3e-2, atol=3e-3,
                )
            else:
                routed_output = output.clone()
            if rank == 0:
                output_snapshots["shared" if with_shared else "routed_only"] = (
                    output.detach().cpu().clone()
                )
            samples = []
            gpu_samples = []
            for _ in range(args.runs):
                dist.barrier()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start = perf_counter()
                start_event.record()
                graph.replay()
                end_event.record()
                end_event.synchronize()
                samples.append((perf_counter() - start) * 1000)
                gpu_samples.append(start_event.elapsed_time(end_event))
            result = {
                "wall_ms": median(samples), "gpu_ms": median(gpu_samples),
                "finite": bool(torch.isfinite(output).all().item()),
                "graph_replays": args.runs + 1,
            }
            gathered = [None] * args.ep
            dist.all_gather_object(gathered, result)
            if rank == 0:
                results["shared" if with_shared else "routed_only"] = {
                    "slowest_rank_wall_ms": max(item["wall_ms"] for item in gathered),
                    "slowest_rank_gpu_ms": max(item["gpu_ms"] for item in gathered),
                    "ranks": gathered,
                }
            del module, graph, output
            torch.cuda.empty_cache()
        if rank == 0:
            results.update({
                "model": str(args.model), "layer": 0, "dtype": "bfloat16",
                "ep": args.ep, "tp": args.tp, "dp": args.dp,
                "moe_shard_across_tp": args.shard_across_tp,
                "shared_tp_sharded": args.shared_tp_sharded,
                "moe_shards": get_moe_world_size(),
                "tokens": args.tokens, "cuda_graph": True,
                "runs": args.runs,
            })
            results["shared_overhead_ratio"] = (
                results["shared"]["slowest_rank_wall_ms"] /
                results["routed_only"]["slowest_rank_wall_ms"]
            )
            if args.result_file:
                args.result_file.write_text(json.dumps(results, indent=2) + "\n")
            if args.output_file:
                torch.save(output_snapshots, args.output_file)
            print(json.dumps(results))
    finally:
        destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
