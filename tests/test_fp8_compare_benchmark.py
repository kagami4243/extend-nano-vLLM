from benchmarks.bench_fp8_compare import (
    summarize_result,
    validate_backend,
    validate_result,
)


def test_compare_benchmark_result_schema():
    result = summarize_result(
        backend="vllm",
        quantization="fp8",
        batch_size=1,
        prompt_tokens=128,
        decode_tokens=8,
        elapsed_ms=100.0,
    )
    validate_result(result)
    assert result["backend"] == "vllm"
    assert result["quantization"] == "fp8"
    assert result["fp8_format"] == "per_token"
    assert result["elapsed_ms"] == 100.0


def test_compare_benchmark_rejects_unknown_backend():
    try:
        validate_result({"backend": "unknown", "quantization": "fp8"})
    except ValueError:
        pass
    else:
        raise AssertionError("unknown benchmark backend must be rejected")


def test_vllm_backend_only_accepts_fp8():
    validate_backend("vllm", "fp8")
    try:
        validate_backend("vllm", "none")
    except ValueError:
        pass
    else:
        raise AssertionError("vllm comparison must use fp8")

    try:
        validate_backend("vllm", "fp8", "per_tensor")
    except ValueError:
        pass
    else:
        raise AssertionError("vllm comparison must use its per-token path")
