"""Multi-request prefill scheduling contracts."""

from types import SimpleNamespace

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


def scheduler_with_budget(token_budget=5, max_seqs=2, batching=True):
    return Scheduler(SimpleNamespace(
        max_num_seqs=max_seqs,
        max_num_batched_tokens=token_budget,
        enable_prefill_batching=batching,
        speculative_config=None,
        eos=-1,
        num_kvcache_blocks=16,
        kvcache_block_size=256,
        enable_prefix_caching=False,
    ))


def test_prefill_batches_waiting_requests_within_token_budget():
    scheduler = scheduler_with_budget()
    sampling = SamplingParams(temperature=0, ignore_eos=True, max_tokens=1)
    first = Sequence([1] * 3, sampling)
    second = Sequence([2] * 4, sampling)
    third = Sequence([3] * 3, sampling)
    for seq in (first, second, third):
        scheduler.add(seq)

    scheduled, is_prefill = scheduler.schedule()
    assert is_prefill and scheduled == [first, second]
    assert [seq.num_scheduled_tokens for seq in scheduled] == [3, 2]
    scheduler.postprocess(scheduled, [91, 92], is_prefill)
    assert first.completion_token_ids == [91]
    assert second.completion_token_ids == []

    scheduled, is_prefill = scheduler.schedule()
    assert is_prefill and scheduled == [second, third]
    assert [seq.num_scheduled_tokens for seq in scheduled] == [2, 3]
    scheduler.postprocess(scheduled, [93, 94], is_prefill)
    assert second.completion_token_ids == [93]
    assert third.completion_token_ids == [94]
    assert scheduler.is_finished()
    assert scheduler.stats["prefill_steps"] == 2
    assert scheduler.stats["max_prefill_tokens_per_step"] == 5


def test_prefill_respects_sequence_limit():
    scheduler = scheduler_with_budget(token_budget=16, max_seqs=2)
    sampling = SamplingParams(temperature=0, ignore_eos=True, max_tokens=1)
    seqs = [Sequence([token], sampling) for token in (1, 2, 3)]
    for seq in seqs:
        scheduler.add(seq)

    scheduled, is_prefill = scheduler.schedule()
    assert is_prefill and scheduled == seqs[:2]
    scheduler.postprocess(scheduled, [11, 12], is_prefill)
    scheduled, is_prefill = scheduler.schedule()
    assert is_prefill and scheduled == seqs[2:]


def test_prefill_batching_can_be_disabled():
    scheduler = scheduler_with_budget(token_budget=16, batching=False)
    sampling = SamplingParams(temperature=0, ignore_eos=True, max_tokens=1)
    first = Sequence([1, 2], sampling)
    second = Sequence([3, 4], sampling)
    scheduler.add(first)
    scheduler.add(second)

    scheduled, is_prefill = scheduler.schedule()
    assert is_prefill and scheduled == [first]
    scheduler.postprocess(scheduled, [91], is_prefill)
    scheduled, is_prefill = scheduler.schedule()
    assert is_prefill and scheduled == [second]


def test_pending_prefix_block_is_not_reused_before_kv_is_written():
    manager = BlockManager(num_blocks=8, block_size=256, enable_prefix_caching=True)
    prefix = [17] * 256
    first = Sequence(prefix + [1])
    second = Sequence(prefix + [2])
    manager.allocate(first)
    manager.allocate(second)
    assert second.num_cached_tokens == 0
    assert first.block_table[0] != second.block_table[0]

    first.num_computed_tokens = 255
    manager.mark_computed(first)
    third = Sequence(prefix + [3])
    manager.allocate(third)
    assert third.num_cached_tokens == 0

    first.num_computed_tokens = 257
    manager.mark_computed(first)
    fourth = Sequence(prefix + [4])
    manager.allocate(fourth)
    assert fourth.num_cached_tokens == 256
    assert fourth.block_table[0] == first.block_table[0]
