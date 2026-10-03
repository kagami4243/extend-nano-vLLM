"""Local config-only checkpoints for topology tests; no weights or downloads."""

import pytest


CONFIG_TEST_MODULES = {
    "test_moe_dynamic_placement",
    "test_moe_placement",
    "test_moe_prefill_batching_default",
    "test_moe_prefill_piece_config",
    "test_moe_topology_contracts",
    "test_moe_vllm_dp_topology",
}


@pytest.fixture(autouse=True)
def config_only_checkpoints(request, tmp_path, monkeypatch):
    module = request.module
    if module.__name__.rsplit(".", 1)[-1] not in CONFIG_TEST_MODULES:
        return

    from transformers import Qwen3Config, Qwen3MoeConfig

    moe = tmp_path / "moe"
    dense = tmp_path / "dense"
    Qwen3MoeConfig(num_experts=128, num_hidden_layers=48).save_pretrained(moe)
    Qwen3Config().save_pretrained(dense)
    paths = {
        "./models/Qwen3-30B-A3B-Base": str(moe),
        "./models/Qwen3-0.6B": str(dense),
    }
    for name, path in (("MODEL", moe), ("MOE_MODEL", moe), ("DENSE_MODEL", dense)):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, str(path))

    # Defaults and parametrized values retain the strings from collection time.
    original_config = module.Config

    def make_config(model, *args, **kwargs):
        return original_config(paths.get(model, model), *args, **kwargs)

    monkeypatch.setattr(module, "Config", make_config)
