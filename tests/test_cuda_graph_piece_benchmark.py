from benchmarks.bench_cuda_graph_piece import summarize, validate_result


def test_piece_benchmark_summary_uses_median_ttft():
    result = summarize("piece", [3.0, 1.0, 2.0], 128)
    assert result == {
        "mode": "piece",
        "runs": 3,
        "prompt_tokens": 128,
        "ttft_ms_median": 2.0,
    }


def test_piece_benchmark_result_schema():
    validate_result({
        "mode": "eager",
        "runs": 1,
        "prompt_tokens": 16,
        "ttft_ms_median": 1.0,
    })
