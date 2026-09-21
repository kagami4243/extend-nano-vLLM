"""GPU correctness test for FP8 paged KV cache kernels.

Run with ``python -m tests.test_fp8_kv_cache``.  The reference deliberately
reads the stored FP8 values back into FP32, so it checks page addressing and
attention math independently from FP8 quantization error versus BF16 cache.
"""

import torch

from nanovllm.layers.attention import (
    fp8_paged_decode_attention,
    gather_dequant_kvcache,
    store_kvcache,
)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("test_fp8_kv_cache requires CUDA")
    torch.manual_seed(0)
    length, block_size, num_kv_heads, num_heads, head_dim = 300, 256, 2, 4, 128
    key = torch.randn(
        length, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16
    ) * 0.1
    value = torch.randn_like(key) * 0.1
    k_cache = torch.empty(
        2,
        block_size,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )
    v_cache = torch.empty_like(k_cache)
    block_table = torch.tensor([[1, 0]], dtype=torch.int32, device="cuda")
    positions = torch.arange(length, device="cuda")
    slot_mapping = (
        block_table[0, positions // block_size] * block_size
        + positions % block_size
    ).to(torch.int32)
    scale = torch.ones(1, device="cuda", dtype=torch.float32)
    store_kvcache(key, value, k_cache, v_cache, slot_mapping, scale, scale)

    gathered_key = gather_dequant_kvcache(
        k_cache, block_table[0], length, scale, torch.bfloat16
    )
    gathered_value = gather_dequant_kvcache(
        v_cache, block_table[0], length, scale, torch.bfloat16
    )
    quantized_key = k_cache[block_table[0]].reshape(-1, num_kv_heads, head_dim)[
        :length
    ].bfloat16()
    quantized_value = v_cache[block_table[0]].reshape(
        -1, num_kv_heads, head_dim
    )[:length].bfloat16()
    assert torch.equal(gathered_key, quantized_key)
    assert torch.equal(gathered_value, quantized_value)

    query = torch.randn(
        1, num_heads, head_dim, device="cuda", dtype=torch.bfloat16
    ) * 0.1
    actual = fp8_paged_decode_attention(
        query,
        k_cache,
        v_cache,
        torch.tensor([length], dtype=torch.int32, device="cuda"),
        block_table,
        head_dim**-0.5,
        scale,
        scale,
    )
    expanded_key = quantized_key.repeat_interleave(num_heads // num_kv_heads, dim=1)
    expanded_value = quantized_value.repeat_interleave(
        num_heads // num_kv_heads, dim=1
    )
    scores = torch.einsum("hd,lhd->hl", query[0].float(), expanded_key.float())
    weights = torch.softmax(scores * head_dim**-0.5, dim=-1)
    reference = torch.einsum("hl,lhd->hd", weights, expanded_value.float())
    torch.cuda.synchronize()
    max_abs_error = (actual[0].float() - reference).abs().max().item()
    assert max_abs_error <= 2e-3, max_abs_error

    # The same graph must read the updated device-side context length. This is
    # the condition required by model decode graphs reused across requests.
    graph_context_len = torch.ones(1, dtype=torch.int32, device="cuda")
    fp8_paged_decode_attention(
        query,
        k_cache,
        v_cache,
        graph_context_len,
        block_table,
        head_dim**-0.5,
        scale,
        scale,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = fp8_paged_decode_attention(
            query,
            k_cache,
            v_cache,
            graph_context_len,
            block_table,
            head_dim**-0.5,
            scale,
            scale,
        )
    graph_errors = []
    for graph_length in (113, length):
        graph_context_len.fill_(graph_length)
        graph.replay()
        graph_key = quantized_key[:graph_length].repeat_interleave(
            num_heads // num_kv_heads, dim=1
        )
        graph_value = quantized_value[:graph_length].repeat_interleave(
            num_heads // num_kv_heads, dim=1
        )
        graph_scores = torch.einsum(
            "hd,lhd->hl", query[0].float(), graph_key.float()
        )
        graph_weights = torch.softmax(graph_scores * head_dim**-0.5, dim=-1)
        graph_reference = torch.einsum(
            "hl,lhd->hd", graph_weights, graph_value.float()
        )
        torch.cuda.synchronize()
        graph_errors.append(
            (graph_output[0].float() - graph_reference).abs().max().item()
        )
    assert max(graph_errors) <= 2e-3, graph_errors
    print(
        f"FP8 KV cache passed: max_abs={max_abs_error:.6g}, "
        f"graph_max_abs={max(graph_errors):.6g}"
    )


if __name__ == "__main__":
    main()
