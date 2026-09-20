from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.num_speculative_tokens = (
            config.speculative_config.num_speculative_tokens
            if config.speculative_config is not None
            else 0
        )
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

    def begin_speculation(self, seq: Sequence, proposal_tokens: list[int]) -> int:
        if seq not in self.running:
            raise RuntimeError("speculative decode requires a running sequence")
        base_num_tokens = len(seq)
        # ``reserve_speculation`` has already allocated every possible draft
        # position before EAGLE writes proposal KV values.
        seq.append_tokens(proposal_tokens)
        return base_num_tokens

    def reserve_speculation(self, seq: Sequence) -> None:
        """Reserve physical blocks before EAGLE writes proposal KV values.

        Target rejection only rolls back logical sequence tokens. The draft
        attention kernel has already scattered K/V to physical slots, so the
        slots must exist before proposal and are reclaimed by
        ``postprocess_speculation`` after the accepted prefix is known.
        """
        max_tokens = min(
            seq.max_tokens + seq.num_prompt_tokens,
            len(seq) + self.num_speculative_tokens,
        )
        self.block_manager.ensure_num_blocks_for_length(seq, max_tokens)

    def postprocess_speculation(
        self,
        seq: Sequence,
        base_num_tokens: int,
        accepted_tokens: list[int],
        replacement_token: int | None,
    ) -> int:
        remaining = seq.max_tokens - (base_num_tokens - seq.num_prompt_tokens)
        committed_tokens = accepted_tokens[:max(remaining, 0)]
        reached_eos = not seq.ignore_eos and self.eos in committed_tokens
        if reached_eos:
            committed_tokens = committed_tokens[:committed_tokens.index(self.eos) + 1]
            replacement_token = None
        elif len(committed_tokens) < len(accepted_tokens):
            replacement_token = None

        seq.truncate_tokens(base_num_tokens + len(committed_tokens))
        self.block_manager.truncate(seq)
        # The target has evaluated the current tail and accepted drafts. A
        # replacement/bonus token is an output and remains uncomputed.
        seq.num_computed_tokens = base_num_tokens + len(committed_tokens)

        if replacement_token is not None and seq.num_completion_tokens < seq.max_tokens:
            # A replacement can cross a page boundary.  Finalize the old page
            # before appending, then reserve the new page so EAGLE can replay
            # the replacement KV without indexing past the block table.
            if len(seq) % self.block_manager.block_size == 0:
                self.block_manager.may_append(seq)
            seq.append_token(replacement_token)
            self.block_manager.may_append(seq)
        elif len(seq) % self.block_manager.block_size == 0:
            # No replacement was appended, so finalize the block immediately.
            self.block_manager.may_append(seq)

        if (
            (not seq.ignore_eos and seq.last_token == self.eos)
            or seq.num_completion_tokens >= seq.max_tokens
        ):
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)
            self.running.remove(seq)
        return len(committed_tokens)
