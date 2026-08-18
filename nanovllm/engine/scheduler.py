from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
            config.enable_prefix_caching,
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.last_step_was_prefill = False
        self.stats = {
            "prefill_steps": 0,
            "max_prefill_tokens_per_step": 0,
        }

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        if self.waiting and not (self.running and self.last_step_was_prefill):
            seq = self.waiting[0]
            if not seq.block_table:
                if not self.block_manager.can_allocate(seq):
                    if not self.running:
                        raise RuntimeError("not enough KV-cache blocks for request")
                else:
                    self.block_manager.allocate(seq)
            if seq.block_table:
                remaining = seq.num_tokens - seq.num_computed_tokens
                seq.num_scheduled_tokens = min(
                    remaining, self.max_num_batched_tokens
                )
                self.last_step_was_prefill = True
                self.stats["prefill_steps"] += 1
                self.stats["max_prefill_tokens_per_step"] = max(
                    self.stats["max_prefill_tokens_per_step"],
                    seq.num_scheduled_tokens,
                )
                return [seq], True

        # decode
        scheduled_seqs = []
        num_seqs = 0
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        self.last_step_was_prefill = False
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(
        self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool
    ) -> None:
        if is_prefill:
            for seq, token_id in zip(seqs, token_ids):
                seq.num_computed_tokens += seq.num_scheduled_tokens
                seq.num_scheduled_tokens = 0
                if not seq.is_prefill_complete:
                    continue
                self.waiting.remove(seq)
                seq.status = SequenceStatus.RUNNING
                seq.append_token(token_id)
                if (
                    (not seq.ignore_eos and token_id == self.eos)
                    or seq.num_completion_tokens == seq.max_tokens
                ):
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                else:
                    self.running.append(seq)
            return

        for seq, token_id in zip(seqs, token_ids):
            seq.num_computed_tokens = len(seq)
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
