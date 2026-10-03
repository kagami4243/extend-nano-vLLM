"""CPU process groups for EP=DP*TP within each pipeline stage."""

import multiprocessing as mp
import socket

import pytest


def worker(rank, port, dp, tp, pp, output):
    import torch.distributed as dist
    from nanovllm.distributed.parallel_state import (
        destroy_model_parallel, get_ep_group_ranks, get_ep_rank,
        get_ep_world_size, get_moe_group_ranks, get_pp_group_ranks,
        get_pp_rank, get_replica_group, get_tp_group_ranks, get_tp_rank,
        initialize_model_parallel,
    )

    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}",
        world_size=dp * tp * pp, rank=rank,
    )
    try:
        initialize_model_parallel(tp, pp, True, data_parallel_size=dp)
        output.put((
            rank, get_tp_rank(), get_pp_rank(), get_ep_rank(),
            get_ep_world_size(), get_tp_group_ranks(), get_pp_group_ranks(),
            get_ep_group_ranks(), get_moe_group_ranks(),
            dist.get_world_size(get_replica_group()),
        ))
    finally:
        destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.timeout(120)
@pytest.mark.parametrize("dp,tp,pp", [(1, 4, 1), (1, 2, 2), (2, 2, 1), (2, 1, 2)])
def test_parallel_groups(dp, tp, pp):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    context = mp.get_context("spawn")
    output = context.Queue()
    workers = [context.Process(target=worker, args=(rank, port, dp, tp, pp, output))
               for rank in range(dp * tp * pp)]
    try:
        for process in workers:
            process.start()
        actual = sorted(output.get(timeout=90) for _ in workers)
        for rank, tp_rank, pp_rank, ep_rank, ep_size, tp_group, pp_group, ep_group, moe_group, replica_size in actual:
            dp_rank = rank // (pp * tp)
            stage = (rank // tp) % pp
            tensor = rank % tp
            assert (tp_rank, pp_rank, ep_rank, ep_size) == (tensor, stage, dp_rank * tp + tensor, dp * tp)
            assert tp_group == tuple(dp_rank * pp * tp + stage * tp + t for t in range(tp))
            assert pp_group == tuple(dp_rank * pp * tp + p * tp + tensor for p in range(pp))
            assert ep_group == tuple(d * pp * tp + stage * tp + t for d in range(dp) for t in range(tp))
            assert moe_group == ep_group
            assert replica_size == pp * tp
    finally:
        for process in workers:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join()
        assert all(process.exitcode == 0 for process in workers)
