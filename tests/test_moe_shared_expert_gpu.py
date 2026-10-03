"""Four-GPU DP2 x TP2 = EP4 shared expert CUDA Graph correctness."""

import json
import os
import gc

import torch
import torch.distributed as dist

from nanovllm.distributed.parallel_state import (
    destroy_model_parallel, initialize_model_parallel,
)
from nanovllm.layers.moe import ExpertParallelMoE


@torch.inference_mode()
def worker(rank):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    torch.manual_seed(79)
    full = ExpertParallelMoE(
        128, 256, 8, 2, True, shared_expert_intermediate_size=64,
    )
    with torch.no_grad():
        for parameter in full.parameters():
            parameter.normal_(0, 0.02)
    initialize_model_parallel(2, 1, True, data_parallel_size=2)
    try:
        ep_rank = rank
        shard = ExpertParallelMoE(
            128, 256, 8, 2, True, ep_rank=ep_rank, ep_size=4,
            dispatch_backend="allgather_reduce",
            graph_safe_decode=True, shared_expert_intermediate_size=64,
            shared_expert_tp_sharded=False,
        )
        with torch.no_grad():
            shard.gate.weight.copy_(full.gate.weight)
            shard.gate_up_proj.copy_(
                full.gate_up_proj[ep_rank * 2:(ep_rank + 1) * 2]
            )
            shard.down_proj.copy_(
                full.down_proj[ep_rank * 2:(ep_rank + 1) * 2]
            )
            shard.shared_expert.gate_up_proj.weight.copy_(
                full.shared_expert.gate_up_proj.weight
            )
            shard.shared_expert.down_proj.weight.copy_(
                full.shared_expert.down_proj.weight
            )
            shard.shared_expert_gate.weight.copy_(full.shared_expert_gate.weight)
        full = full.cuda().bfloat16()
        shard = shard.cuda().bfloat16()
        first = torch.randn(16, 128, device="cuda", dtype=torch.bfloat16)
        first.add_(rank // 2 * 0.01)
        second = torch.randn_like(first)
        torch.testing.assert_close(
            shard.shared_expert(first), full.shared_expert(first),
            rtol=0, atol=0,
        )
        torch.testing.assert_close(
            shard.shared_expert_gate(first), full.shared_expert_gate(first),
            rtol=0, atol=0,
        )
        shard.graph_safe_decode = False
        expected_first = shard(first)
        expected_second = shard(second)
        torch.testing.assert_close(
            expected_first, full(first), rtol=3e-2, atol=3e-3,
        )
        torch.testing.assert_close(
            expected_second, full(second), rtol=3e-2, atol=3e-3,
        )
        shard.graph_safe_decode = True
        static_input = first.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            shard(static_input)
        torch.cuda.current_stream().wait_stream(stream)
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = shard(static_input)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            output, expected_first, rtol=3e-2, atol=3e-3,
        )
        static_input.copy_(second)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            output, expected_second, rtol=3e-2, atol=3e-3,
        )
        observations = [None] * 4
        dist.all_gather_object(
            observations, {"rank": rank, "graph_replays": 2}
        )
        if rank == 0:
            print(json.dumps(observations), flush=True)
        del graph, output, shard, full
        torch.cuda.synchronize()
        gc.collect()
        dist.barrier()
    finally:
        destroy_model_parallel()
        dist.destroy_process_group()


def main():
    if torch.cuda.device_count() != 4 or os.environ.get("WORLD_SIZE") != "4":
        raise RuntimeError("run with torchrun --nproc_per_node=4 on four GPUs")
    worker(int(os.environ["LOCAL_RANK"]))


if __name__ == "__main__":
    main()
