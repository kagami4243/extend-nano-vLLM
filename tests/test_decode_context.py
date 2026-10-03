"""Decode preparation must select the same attention path as graph capture."""

import pytest
import torch

from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import get_context, reset_context


@torch.inference_mode()
def test_prepare_decode_uses_decode_attention_context():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    runner = ModelRunner.__new__(ModelRunner)
    runner.block_size = 256
    runner.prepare_block_tables = lambda seqs: torch.tensor(
        [[seq.block_table[0]] for seq in seqs],
        device="cuda", dtype=torch.int32,
    )
    first = Sequence([11] * 16)
    second = Sequence([12] * 17)
    first.block_table = [3]
    second.block_table = [5]
    try:
        input_ids, positions = runner.prepare_decode([first, second])
        context = get_context()
        assert not context.is_prefill
        assert input_ids.tolist() == [11, 12]
        assert positions.tolist() == [15, 16]
        assert context.context_lens.tolist() == [16, 17]
        assert context.slot_mapping.tolist() == [3 * 256 + 15, 5 * 256 + 16]
    finally:
        reset_context()
