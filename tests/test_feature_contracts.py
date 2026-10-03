"""Small CPU contracts for the TODO's prefix cache and chunked prefill."""

from types import SimpleNamespace

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


def test_prefix_cache_reuses_only_complete_matching_blocks():
    manager = BlockManager(num_blocks=8, block_size=256)
    first = Sequence([11] * 256 + [12] * 256 + [13] * 17)
    manager.allocate(first)
    first.num_computed_tokens = len(first)
    manager.mark_computed(first)
    manager.deallocate(first)

    matching = Sequence([11] * 256 + [12] * 256 + [99] * 17)
    manager.allocate(matching)
    assert matching.num_cached_tokens == 512
    assert matching.num_computed_tokens == 512
    assert manager.stats["prefix_cache_hits"] == 2
    manager.deallocate(matching)

    changed = Sequence([11] * 256 + [22] * 256 + [13] * 17)
    manager.allocate(changed)
    assert changed.num_cached_tokens == 256
    assert changed.num_computed_tokens == 256
    manager.deallocate(changed)


def test_recycled_block_does_not_leave_a_stale_hash():
    manager = BlockManager(num_blocks=1, block_size=256)
    old = Sequence([11] * 256)
    manager.allocate(old)
    old_hash = manager.blocks[old.block_table[0]].hash
    manager.deallocate(old)

    replacement = Sequence([22] * 256)
    manager.allocate(replacement)
    manager.deallocate(replacement)
    assert old_hash not in manager.hash_to_block_id

    requested_again = Sequence([11] * 256)
    manager.allocate(requested_again)
    assert requested_again.num_cached_tokens == 0
    manager.deallocate(requested_again)


def test_long_prefill_respects_token_budget_and_finishes_once():
    config = SimpleNamespace(
        max_num_seqs=2,
        max_num_batched_tokens=128,
        speculative_config=None,
        eos=-1,
        num_kvcache_blocks=8,
        kvcache_block_size=256,
        enable_prefix_caching=False,
    )
    scheduler = Scheduler(config)
    seq = Sequence(
        [17] * 300,
        SamplingParams(temperature=0, ignore_eos=True, max_tokens=1),
    )
    scheduler.add(seq)
    chunks = []
    while not scheduler.is_finished():
        scheduled, is_prefill = scheduler.schedule()
        assert is_prefill and scheduled == [seq]
        chunks.append(seq.num_scheduled_tokens)
        scheduler.postprocess(scheduled, [42], is_prefill)
    assert chunks == [128, 128, 44]
    assert seq.completion_token_ids == [42]
    assert scheduler.stats["prefill_steps"] == 3
    assert scheduler.stats["max_prefill_tokens_per_step"] == 128
