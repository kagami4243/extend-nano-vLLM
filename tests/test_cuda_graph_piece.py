import torch
import pytest
import sys
import types
from collections import OrderedDict

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


def make_runner():
    runner = ModelRunner.__new__(ModelRunner)
    runner.enforce_eager = False
    runner.prefill_piece_enabled = True
    runner.prefill_piece_graphs = {}
    runner.graphs = {}
    return runner


def test_prefill_dispatches_to_piece_graph_when_enabled(monkeypatch):
    runner = make_runner()
    runner.model = object()
    input_ids = torch.tensor([1])
    positions = torch.tensor([0])
    called = []

    def piece(ids, pos):
        called.append((ids, pos))
        return torch.tensor([[3.0]])

    monkeypatch.setattr(runner, "run_prefill_piece", piece)
    monkeypatch.setattr(runner, "model", type("Model", (), {
        "model": type("Core", (), {"forward_prefill_piece": None})(),
        "compute_logits": staticmethod(lambda hidden: hidden + 1),
    })())

    output = runner.run_model(input_ids, positions, is_prefill=True)

    assert len(called) == 1
    assert torch.equal(output, torch.tensor([[4.0]]))


def test_prefill_piece_cache_is_keyed_by_token_count():
    runner = make_runner()
    assert runner.prefill_piece_cache_key(7) == 7
    assert runner.prefill_piece_cache_key(7) == runner.prefill_piece_cache_key(7)
    assert runner.prefill_piece_cache_key(8) != runner.prefill_piece_cache_key(7)


def test_eager_prefill_remains_available():
    runner = make_runner()
    runner.prefill_piece_enabled = False
    runner.model = type("Model", (), {
        "__call__": lambda self, ids, pos: ids.float().unsqueeze(-1),
        "compute_logits": staticmethod(lambda hidden: hidden),
    })()
    output = runner.run_model(torch.tensor([2]), torch.tensor([0]), is_prefill=True)
    assert output.shape == (1, 1)


def test_prefill_piece_keeps_attention_between_graph_pieces():
    from nanovllm.models.qwen3 import Qwen3Model

    calls = []

    class Attention:
        def __call__(self, q, k, v):
            calls.append("attention")
            return q

    layer = type("Layer", (), {})()
    layer.self_attn = type("SelfAttention", (), {"attn": Attention()})()
    model = Qwen3Model.__new__(Qwen3Model)
    model.embed_tokens = object()
    model.layers = OrderedDict((("0", layer),))
    model.norm = object()

    def piece_runner(kind, module, *inputs):
        calls.append(kind)
        if kind == "embed":
            return torch.ones(1, 2)
        if kind == "pre":
            tensor = torch.ones(1, 1, 2)
            return tensor, tensor, tensor, torch.ones(1, 2)
        if kind == "attention":
            return inputs[0]
        if kind == "post":
            return torch.ones(1, 2), torch.ones(1, 2)
        return torch.ones(1, 2), None

    Qwen3Model.forward_prefill_piece(
        model, torch.tensor([1]), torch.tensor([0]), piece_runner
    )
    assert calls == ["embed", "pre", "attention", "post", "norm"]


def test_attention_piece_shape_is_head_major_before_output_projection():
    class FakeAttention:
        num_heads = 16
        head_dim = 64

    attention = FakeAttention()
    attention_shape = (128, attention.num_heads, attention.head_dim)
    assert attention_shape == (128, 16, 64)
    assert attention_shape[1] * attention_shape[2] == 1024
