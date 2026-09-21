from benchmarks.bench_fp8_qwen3 import (
    benchmark_kernel_names,
    validate_result,
    vllm_kernel_selection_report,
)


def test_fp8_benchmark_names_the_kernel_under_test():
    names = benchmark_kernel_names()
    assert "nano_vllm_torch_scaled_mm" in names
    assert all(isinstance(name, str) and name for name in names)


def test_fp8_benchmark_records_vllm_dispatch_policy():
    report = vllm_kernel_selection_report()
    assert set(report) == {
        "available", "kernels", "activation_policy", "cuda_priority",
        "ada89_expected",
    }
    assert isinstance(report["kernels"], list)


def test_fp8_benchmark_result_requires_kernel_and_latency():
    validate_result({
        "kernel": "nano_vllm_torch_scaled_mm",
        "fp8_format": "per_token",
        "mean_ms": 1.0,
    })

    try:
        validate_result({"kernel": "", "fp8_format": "per_token", "mean_ms": 1.0})
    except ValueError:
        pass
    else:
        raise AssertionError("empty kernel name must be rejected")


def test_fp8_benchmark_rejects_unknown_format():
    try:
        validate_result({
            "kernel": "nano_vllm_torch_scaled_mm",
            "fp8_format": "block",
            "mean_ms": 1.0,
        })
    except ValueError:
        pass
    else:
        raise AssertionError("unknown FP8 format must be rejected")
