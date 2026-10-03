"""Run with torchrun --nproc_per_node=4 on four GPUs."""

import gc
import os

import torch
import torch.distributed as dist

from nanovllm.distributed.parallel_state import (
    destroy_model_parallel, initialize_model_parallel,
)
from nanovllm.layers.moe import ExpertParallelMoE


@torch.inference_mode()
def main():
    if torch.cuda.device_count() != 4 or os.environ.get("WORLD_SIZE") != "4":
        raise RuntimeError("four visible GPUs and torchrun --nproc_per_node=4 required")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    initialize_model_parallel(4, 1, True)
    try:
        torch.manual_seed(61)
        full = ExpertParallelMoE(128, 256, 8, 2, True)
        for parameter in full.parameters():
            parameter.normal_(0, 0.02)
        initial = tuple(expert // 2 for expert in range(8))
        rotated = tuple(expert % 4 for expert in range(8))
        shard = ExpertParallelMoE(
            128, 256, 8, 2, True, ep_rank=rank, ep_size=4,
            expert_owners=initial, dynamic_placement=True,
            graph_safe_decode=True,
        )
        ids = [expert for expert, owner in enumerate(initial) if owner == rank]
        shard.gate.weight.copy_(full.gate.weight)
        shard.gate_up_proj.copy_(full.gate_up_proj[ids])
        shard.down_proj.copy_(full.down_proj[ids])
        shard = shard.cuda().bfloat16()
        x = torch.randn(16, 128, device="cuda", dtype=torch.bfloat16)
        capture_input = x.clone()
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            shard(capture_input)
        torch.cuda.current_stream().wait_stream(capture_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = shard(capture_input)
        graph.replay()
        torch.cuda.synchronize()
        before = graph_output.clone()
        assert shard.relocate_experts(rotated) > 0
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(graph_output, shard(x), rtol=0, atol=0)
        torch.testing.assert_close(graph_output, before, rtol=3e-2, atol=3e-3)
        capture_input.copy_(torch.randn_like(x))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            graph_output, shard(capture_input), rtol=0, atol=0,
        )
        dist.barrier()
        if rank == 0:
            print("dynamic placement CUDA Graph replay passed", flush=True)
        del graph, graph_output, shard, capture_input, capture_stream
        torch.cuda.synchronize()
        gc.collect()
        dist.barrier()
    finally:
        destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
