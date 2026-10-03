"""Diagnostic layer benchmark for live EP migration; run with four GPU ranks."""

import argparse
import gc
import json
import os
from statistics import median
from time import perf_counter

import torch
import torch.distributed as dist

from nanovllm.distributed.parallel_state import (
    destroy_model_parallel, initialize_model_parallel,
)
from nanovllm.layers.moe import ExpertParallelMoE


def gathered_max(value):
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return max(values)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-file")
    parser.add_argument("--replays", type=int, default=20)
    args = parser.parse_args()
    if torch.cuda.device_count() != 4 or os.environ.get("WORLD_SIZE") != "4":
        parser.error("run with torchrun --nproc_per_node=4 on four visible GPUs")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    initialize_model_parallel(4, 1, True)
    try:
        torch.manual_seed(71)
        initial = tuple(expert // 32 for expert in range(128))
        balanced = tuple(expert % 4 for expert in range(128))
        layer = ExpertParallelMoE(
            2048, 768, 128, 8, True, ep_rank=rank, ep_size=4,
            expert_owners=initial, dynamic_placement=True,
            graph_safe_decode=True,
        ).cuda().bfloat16()
        for parameter in layer.parameters():
            parameter.normal_(0, 0.01)
        # Eight hot experts are colocated initially and spread over EP ranks.
        layer.gate.weight.fill_(-0.01)
        layer.gate.weight[:8].fill_(0.01)
        decode_input = torch.ones(16, 2048, device="cuda", dtype=torch.bfloat16)
        prefill_input = torch.ones(256, 2048, device="cuda", dtype=torch.bfloat16)
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            layer(decode_input)
        torch.cuda.current_stream().wait_stream(capture_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = layer(decode_input)

        def measure():
            graph_samples = []
            eager_samples = []
            for _ in range(5):
                dist.barrier()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.replays):
                    graph.replay()
                end.record()
                end.synchronize()
                graph_samples.append(
                    gathered_max(start.elapsed_time(end) / args.replays)
                )
                dist.barrier()
                torch.cuda.synchronize()
                wall = perf_counter()
                layer(prefill_input)
                torch.cuda.synchronize()
                eager_samples.append(gathered_max((perf_counter() - wall) * 1000))
            return median(graph_samples), median(eager_samples)

        before = measure()
        before_output = graph_output.clone()
        def migrate(owners):
            dist.barrier()
            torch.cuda.synchronize()
            wall = perf_counter()
            moved_count = layer.relocate_experts(owners)
            torch.cuda.synchronize()
            return moved_count, gathered_max((perf_counter() - wall) * 1000)

        moved, first_migration_ms = migrate(balanced)
        _, return_migration_ms = migrate(initial)
        _, repeat_migration_ms = migrate(balanced)
        after = measure()
        torch.testing.assert_close(graph_output, before_output, rtol=3e-2, atol=3e-3)
        result = {
            "shape": {"hidden": 2048, "intermediate": 768,
                      "experts": 128, "top_k": 8, "ep": 4},
            "workload": "synthetic eight hot experts; one MoE layer",
            "decode_tokens": 16,
            "decode_cuda_graph": True,
            "prefill_tokens": 256,
            "prefill_cuda_graph": False,
            "moved_experts": moved,
            "migration_ms_first": first_migration_ms,
            "migration_ms_return": return_migration_ms,
            "migration_ms_repeat": repeat_migration_ms,
            "graph_decode_ms_before_after": [before[0], after[0]],
            "eager_prefill_ms_before_after": [before[1], after[1]],
            "graph_decode_speedup": before[0] / after[0],
            "eager_prefill_speedup": before[1] / after[1],
            "graph_decode_requests_to_amortize_first_migration": (
                first_migration_ms / (before[0] - after[0])
                if before[0] > after[0] else None
            ),
            "eager_prefill_requests_to_amortize_first_migration": (
                first_migration_ms / (before[1] - after[1])
                if before[1] > after[1] else None
            ),
        }
        if rank == 0:
            print(json.dumps(result, indent=2), flush=True)
            if args.result_file:
                with open(args.result_file, "w") as output:
                    json.dump(result, output, indent=2)
                    output.write("\n")
        del graph, graph_output, layer, decode_input, prefill_input, capture_stream
        torch.cuda.synchronize()
        gc.collect()
        dist.barrier()
    finally:
        destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
