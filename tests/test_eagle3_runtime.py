import torch
import pytest
import sys
import types

try:
    from nanovllm.engine.model_runner import ModelRunner
except Exception as exc:  # pragma: no cover - environment gate
    if "flash_attn" not in str(exc):
        pytest.skip(f"nano-vLLM CUDA dependencies unavailable: {exc}", allow_module_level=True)
    flash_attn = types.ModuleType("flash_attn")
    flash_attn.flash_attn_func = lambda *args, **kwargs: None
    flash_attn.flash_attn_varlen_func = lambda *args, **kwargs: None
    flash_attn.flash_attn_with_kvcache = lambda *args, **kwargs: None
    sys.modules["flash_attn"] = flash_attn
    for module_name in list(sys.modules):
        if module_name == "nanovllm" or module_name.startswith("nanovllm."):
            del sys.modules[module_name]
    try:
        from nanovllm.engine.model_runner import ModelRunner
    except Exception as retry_exc:
        pytest.skip(f"nano-vLLM dependencies unavailable: {retry_exc}", allow_module_level=True)


def test_eagle3_runtime_capabilities_report_paged_cache_and_graph_state():
    runner = ModelRunner.__new__(ModelRunner)
    runner.eagle3_model = object()
    runner.eagle_kv_cache = torch.empty(2, 1, 2, 4, 1, 8)
    runner.eagle_graphs = {1: object()}

    capabilities = runner.get_eagle3_runtime_capabilities()

    assert capabilities["paged_kv_cache"] is True
    assert capabilities["cuda_graph"] is True
    assert capabilities["graph_batch_sizes"] == [1]


def test_eagle3_runtime_capabilities_are_explicit_when_disabled():
    runner = ModelRunner.__new__(ModelRunner)
    runner.eagle3_model = None
    runner.eagle_kv_cache = None
    runner.eagle_graphs = {}

    capabilities = runner.get_eagle3_runtime_capabilities()

    assert capabilities == {
        "paged_kv_cache": False,
        "cuda_graph": False,
        "graph_batch_sizes": [],
    }


def test_eagle3_graph_replay_copies_scheduler_paged_metadata():
    class FakeGraph:
        def replay(self):
            return None

    runner = ModelRunner.__new__(ModelRunner)
    runner.eagle_graphs = {1: FakeGraph()}
    runner.eagle_graph_vars = {
        1: {
            "input_ids": torch.zeros(1, dtype=torch.int64),
            "positions": torch.zeros(1, dtype=torch.int64),
            "feedback": torch.zeros(1, 2),
            "slot_mapping": torch.full((1,), -1, dtype=torch.int32),
            "context_lens": torch.ones(1, dtype=torch.int32),
            "block_tables": torch.zeros(1, 2, dtype=torch.int32),
            "outputs": (torch.ones(1, 2), torch.full((1, 2), 2.0)),
        }
    }
    from nanovllm.utils.context import reset_context, set_context

    set_context(
        False,
        slot_mapping=torch.tensor([7], dtype=torch.int32),
        context_lens=torch.tensor([9], dtype=torch.int32),
        block_tables=torch.tensor([[3, 4]], dtype=torch.int32),
    )
    try:
        seq = type("Seq", (), {})()
        hidden, feedback = runner.run_eagle3_graph(
            torch.tensor([5]), torch.tensor([8]), torch.zeros(1, 2), [seq]
        )
        variables = runner.eagle_graph_vars[1]
        assert variables["slot_mapping"].tolist() == [7]
        assert variables["context_lens"].tolist() == [9]
        assert variables["block_tables"].tolist() == [[3, 4]]
        assert hidden.tolist() == [[1.0, 1.0]]
        assert feedback.tolist() == [[2.0, 2.0]]
    finally:
        reset_context()
