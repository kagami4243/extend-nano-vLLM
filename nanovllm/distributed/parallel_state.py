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
_EP_RANK = 0
_EP_WORLD_SIZE = 1


def initialize_model_parallel(
    tp_size: int,
    pp_size: int = 1,
    enable_expert_parallel: bool = False,
) -> None:
    global _TP_GROUP, _TP_GROUP_RANKS, _TP_RANK, _TP_WORLD_SIZE
    global _PP_GROUP, _PP_GROUP_RANKS, _PP_RANK, _PP_WORLD_SIZE
    global _EP_GROUP, _EP_RANK, _EP_WORLD_SIZE
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized first")
    world_size = dist.get_world_size()
    if tp_size * pp_size != world_size:
        raise ValueError(
            "parallel sizes do not match world size: "
            f"tp_size={tp_size}, pp_size={pp_size}, "
            f"world_size={world_size}"
        )

    rank = dist.get_rank()
    _TP_WORLD_SIZE = tp_size
    _PP_WORLD_SIZE = pp_size
    _TP_RANK = rank % tp_size
    _PP_RANK = rank // tp_size
    _TP_GROUP = None
    _PP_GROUP = None
    if tp_size == 1:
        _TP_GROUP_RANKS = (rank,)
    else:
        for stage_id in range(pp_size):
            ranks = tuple(
                range(stage_id * tp_size, (stage_id + 1) * tp_size)
            )
            group = (
                dist.group.WORLD
                if len(ranks) == world_size
                else dist.new_group(ranks=list(ranks))
            )
            if rank in ranks:
                _TP_GROUP_RANKS = ranks
                _TP_GROUP = group
    if pp_size == 1:
        _PP_GROUP_RANKS = (rank,)
    else:
        for tensor_rank in range(tp_size):
            ranks = tuple(
                stage_id * tp_size + tensor_rank
                for stage_id in range(pp_size)
            )
            group = (
                dist.group.WORLD
                if len(ranks) == world_size
                else dist.new_group(ranks=list(ranks))
            )
            if rank in ranks:
                _PP_GROUP_RANKS = ranks
                _PP_GROUP = group
    # EP repurposes the TP ranks only inside MoE layers. Dense layers still
    # use this same group as their ordinary tensor-parallel group.
    _EP_WORLD_SIZE = tp_size if enable_expert_parallel else 1
    _EP_RANK = _TP_RANK if enable_expert_parallel else 0
    _EP_GROUP = _TP_GROUP if enable_expert_parallel else None


def destroy_model_parallel() -> None:
    global _TP_GROUP, _TP_GROUP_RANKS, _TP_RANK, _TP_WORLD_SIZE
    global _PP_GROUP, _PP_GROUP_RANKS, _PP_RANK, _PP_WORLD_SIZE
    global _EP_GROUP, _EP_RANK, _EP_WORLD_SIZE
    _TP_GROUP = None
    _TP_GROUP_RANKS = (0,)
    _TP_RANK = 0
    _TP_WORLD_SIZE = 1
    _PP_GROUP = None
    _PP_GROUP_RANKS = (0,)
    _PP_RANK = 0
    _PP_WORLD_SIZE = 1
    _EP_GROUP = None
    _EP_RANK = 0
    _EP_WORLD_SIZE = 1


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
