"""The offline DP helper must coordinate ranks participating in global EP."""

import queue
from types import SimpleNamespace

import pytest

from nanovllm.engine.data_parallel import generate_data_parallel
from nanovllm.sampling_params import SamplingParams


@pytest.mark.parametrize("prompts,sampling,options,match", [
    ([[1], [2], [3]], SamplingParams(ignore_eos=True), {}, "balanced"),
    (["a", "b"], SamplingParams(ignore_eos=True), {}, "token IDs"),
    ([[1], [2]], SamplingParams(), {}, "ignore_eos"),
    ([[1], [2]], [SamplingParams(ignore_eos=True, max_tokens=2),
                  SamplingParams(ignore_eos=True, max_tokens=3)], {}, "max_tokens"),
])
def test_unsafe_global_ep_requests_fail_before_starting_workers(monkeypatch, prompts, sampling, options, match):
    def no_workers(*args):
        raise AssertionError("invalid global EP input must fail before spawning")

    monkeypatch.setattr("nanovllm.engine.data_parallel.mp.get_context", no_workers)
    with pytest.raises(ValueError, match=match):
        generate_data_parallel(
            "unused", prompts, sampling, data_parallel_size=2,
            enable_expert_parallel=True, **options,
        )


@pytest.mark.parametrize("expert_parallel,expected_ports", [(True, [29591, 29591]),
                                                            (False, [29591, 29592])])
@pytest.mark.parametrize("prefix_caching", [None, True, False])
def test_helper_uses_shared_ep_port_and_preserves_dense_dp_routing(monkeypatch, expert_parallel, expected_ports, prefix_caching):
    calls = []

    class FakeLLM:
        def __init__(self, model, **kwargs):
            calls.append(kwargs)

        def generate(self, prompts, sampling, **kwargs):
            return [{"token_ids": prompt} for prompt in prompts]

        def exit(self):
            pass

    class LocalProcess:
        exitcode = 0

        def __init__(self, target, args):
            self.target, self.args = target, args

        def start(self):
            self.target(*self.args)

        def join(self, **kwargs):
            pass

        def is_alive(self):
            return False

    monkeypatch.setattr("nanovllm.LLM", FakeLLM)
    monkeypatch.setattr(
        "nanovllm.engine.data_parallel.mp.get_context",
        lambda _: SimpleNamespace(Queue=queue.Queue, Process=LocalProcess),
    )
    prompts = [[1], [3, 4], [5, 6, 7], [7, 8, 9, 10]]
    options = {} if prefix_caching is None else {"enable_prefix_caching": prefix_caching}
    outputs, ranks = generate_data_parallel(
        "unused", prompts, SamplingParams(ignore_eos=True),
        data_parallel_size=2, enable_expert_parallel=expert_parallel,
        master_port=29591, run_id="test_dp",
        **options,
    )
    assert [item["token_ids"] for item in outputs] == prompts
    assert ranks == [0, 1]
    assert [item["master_port"] for item in calls] == expected_ports
    assert [item["run_id"] for item in calls] == ["test_dp_dp0", "test_dp_dp1"]
    if prefix_caching is None:
        assert all("enable_prefix_caching" not in item for item in calls)
    else:
        assert all(item["enable_prefix_caching"] is prefix_caching for item in calls)
