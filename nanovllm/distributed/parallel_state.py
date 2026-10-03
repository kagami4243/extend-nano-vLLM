import torch.distributed as dist


_TP_GROUP: dist.ProcessGroup | None = None
_TP_GROUP_RANKS: tuple[int, ...] = (0,)
_TP_RANK = 0
_TP_WORLD_SIZE = 1
_PP_GROUP: dist.ProcessGroup | None = None
_PP_GROUP_RANKS: tuple[int, ...] = (0,)
_PP_RANK = 0
_PP_WORLD_SIZE = 1
_EP_GROUP: dist.ProcessGroup | None = None
_EP_GROUP_RANKS: tuple[int, ...] = (0,)
_EP_RANK = 0
_EP_WORLD_SIZE = 1
_MOE_GROUP: dist.ProcessGroup | None = None
_MOE_GROUP_RANKS: tuple[int, ...] = (0,)
_MOE_RANK = 0
_MOE_WORLD_SIZE = 1
_REPLICA_GROUP: dist.ProcessGroup | None = None


def initialize_model_parallel(
    tp_size: int,
    pp_size: int = 1,
    enable_expert_parallel: bool = False,
    expert_parallel_size: int | None = None,
    moe_shard_across_tp: bool = False,
    data_parallel_size: int = 1,
) -> None:
    global _TP_GROUP, _TP_GROUP_RANKS, _TP_RANK, _TP_WORLD_SIZE
    global _PP_GROUP, _PP_GROUP_RANKS, _PP_RANK, _PP_WORLD_SIZE
    global _EP_GROUP, _EP_GROUP_RANKS, _EP_RANK, _EP_WORLD_SIZE
    global _MOE_GROUP, _MOE_GROUP_RANKS, _MOE_RANK, _MOE_WORLD_SIZE
    global _REPLICA_GROUP
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized first")
    ep_size = data_parallel_size * tp_size
    if min(tp_size, pp_size, data_parallel_size) < 1:
        raise ValueError("parallel sizes must be positive")
    if expert_parallel_size is not None and (
        not enable_expert_parallel or expert_parallel_size != ep_size
    ):
        raise ValueError("EP must equal DP * TP; independent EP axes are not supported")
    if moe_shard_across_tp:
        raise ValueError("independent cross-TP EP sharding is removed; EP is DP * TP")
    world_size = dist.get_world_size()
    if tp_size * pp_size * data_parallel_size != world_size:
        raise ValueError(
            "parallel sizes do not match world size: "
            f"dp_size={data_parallel_size}, tp_size={tp_size}, pp_size={pp_size}, "
            f"world_size={world_size}"
        )

    rank = dist.get_rank()
    _TP_WORLD_SIZE = tp_size
    _PP_WORLD_SIZE = pp_size
    _TP_RANK = rank % tp_size
    _PP_RANK = (rank // tp_size) % pp_size
    _TP_GROUP = None
    _PP_GROUP = None

    def rank_of(replica: int, stage: int, tensor: int) -> int:
        return (replica * pp_size + stage) * tp_size + tensor

    groups = {tuple(range(world_size)): dist.group.WORLD}

    def make_group(ranks):
        if ranks not in groups:
            groups[ranks] = dist.new_group(ranks=list(ranks))
        return groups[ranks]

    for replica_id in range(data_parallel_size):
        ranks = tuple(rank_of(replica_id, stage, tensor)
                      for stage in range(pp_size) for tensor in range(tp_size))
        group = make_group(ranks)
        if rank in ranks:
            _REPLICA_GROUP = group
        for stage_id in range(pp_size):
            ranks = tuple(rank_of(replica_id, stage_id, t) for t in range(tp_size))
            group = make_group(ranks) if tp_size > 1 else None
            if rank in ranks:
                _TP_GROUP_RANKS = ranks
                _TP_GROUP = group
        for tensor_id in range(tp_size):
            ranks = tuple(rank_of(replica_id, p, tensor_id) for p in range(pp_size))
            group = make_group(ranks) if pp_size > 1 else None
            if rank in ranks:
                _PP_GROUP_RANKS = ranks
                _PP_GROUP = group

    if enable_expert_parallel:
        _EP_WORLD_SIZE = ep_size
        _EP_RANK = rank // (pp_size * tp_size) * tp_size + _TP_RANK
        for stage_id in range(pp_size):
            ranks = tuple(rank_of(d, stage_id, t)
                          for d in range(data_parallel_size) for t in range(tp_size))
            group = make_group(ranks) if ep_size > 1 else None
            if rank in ranks:
                _EP_GROUP_RANKS = ranks
                _EP_GROUP = group
    else:
        _EP_WORLD_SIZE = 1
        _EP_RANK = 0
        _EP_GROUP_RANKS = (rank,)
        _EP_GROUP = None
    _MOE_GROUP = _EP_GROUP
    _MOE_GROUP_RANKS = _EP_GROUP_RANKS
    _MOE_RANK = _EP_RANK
    _MOE_WORLD_SIZE = _EP_WORLD_SIZE


def destroy_model_parallel() -> None:
    global _TP_GROUP, _TP_GROUP_RANKS, _TP_RANK, _TP_WORLD_SIZE
    global _PP_GROUP, _PP_GROUP_RANKS, _PP_RANK, _PP_WORLD_SIZE
    global _EP_GROUP, _EP_GROUP_RANKS, _EP_RANK, _EP_WORLD_SIZE
    global _MOE_GROUP, _MOE_GROUP_RANKS, _MOE_RANK, _MOE_WORLD_SIZE
    global _REPLICA_GROUP
    _TP_GROUP = None
    _TP_GROUP_RANKS = (0,)
    _TP_RANK = 0
    _TP_WORLD_SIZE = 1
    _PP_GROUP = None
    _PP_GROUP_RANKS = (0,)
    _PP_RANK = 0
    _PP_WORLD_SIZE = 1
    _EP_GROUP = None
    _EP_GROUP_RANKS = (0,)
    _EP_RANK = 0
    _EP_WORLD_SIZE = 1
    _MOE_GROUP = None
    _MOE_GROUP_RANKS = (0,)
    _MOE_RANK = 0
    _MOE_WORLD_SIZE = 1
    _REPLICA_GROUP = None


def get_replica_group() -> dist.ProcessGroup | None:
    return _REPLICA_GROUP


def get_tp_group() -> dist.ProcessGroup | None:
    if _TP_GROUP is None and _TP_WORLD_SIZE > 1:
        raise RuntimeError("model parallel is not initialized")
    return _TP_GROUP


def get_tp_rank() -> int:
    return _TP_RANK


def get_tp_world_size() -> int:
    return _TP_WORLD_SIZE


def get_tp_group_ranks() -> tuple[int, ...]:
    return _TP_GROUP_RANKS


def get_tp_src_rank() -> int:
    return _TP_GROUP_RANKS[0]


def get_pp_group() -> dist.ProcessGroup | None:
    if _PP_GROUP is None and _PP_WORLD_SIZE > 1:
        raise RuntimeError("pipeline parallel is not initialized")
    return _PP_GROUP


def get_pp_rank() -> int:
    return _PP_RANK


def get_pp_world_size() -> int:
    return _PP_WORLD_SIZE


def get_pp_group_ranks() -> tuple[int, ...]:
    return _PP_GROUP_RANKS


def get_pp_prev_rank() -> int | None:
    if _PP_RANK == 0:
        return None
    return _PP_GROUP_RANKS[_PP_RANK - 1]


def get_pp_next_rank() -> int | None:
    if _PP_RANK == _PP_WORLD_SIZE - 1:
        return None
    return _PP_GROUP_RANKS[_PP_RANK + 1]


def get_pp_last_rank() -> int:
    return _PP_GROUP_RANKS[-1]


def get_ep_group() -> dist.ProcessGroup | None:
    if _EP_GROUP is None and _EP_WORLD_SIZE > 1:
        raise RuntimeError("expert parallel is not initialized")
    return _EP_GROUP


def get_ep_rank() -> int:
    return _EP_RANK


def get_ep_world_size() -> int:
    return _EP_WORLD_SIZE


def get_ep_group_ranks() -> tuple[int, ...]:
    return _EP_GROUP_RANKS


def get_moe_group() -> dist.ProcessGroup | None:
    if _MOE_GROUP is None and _MOE_WORLD_SIZE > 1:
        raise RuntimeError("MoE parallel is not initialized")
    return _MOE_GROUP


def get_moe_rank() -> int:
    return _MOE_RANK


def get_moe_world_size() -> int:
    return _MOE_WORLD_SIZE


def get_moe_group_ranks() -> tuple[int, ...]:
    return _MOE_GROUP_RANKS
