import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import (
    flash_attn_func,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    k_scale_ptr,
    v_scale_ptr,
    D: tl.constexpr,
    FP8_KV_CACHE: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    if FP8_KV_CACHE:
        # Same per-tensor scale convention as vLLM's
        # triton_reshape_and_cache_flash: cache stores round(x / scale),
        # attention reconstructs x by multiplying the scale when reading.
        key /= tl.load(k_scale_ptr)
        value /= tl.load(v_scale_ptr)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


@triton.jit
def gather_dequant_kvcache_kernel(
    cache_ptr,
    output_ptr,
    block_table_ptr,
    scale_ptr,
    num_tokens,
    cache_stride_block,
    cache_stride_token,
    cache_stride_head,
    output_stride_token,
    output_stride_head,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Gather one paged FP8 cache into temporary BF16/FP16 contiguous KV."""
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    if token_idx >= num_tokens:
        return
    offsets = tl.arange(0, HEAD_DIM)
    block_idx = tl.load(block_table_ptr + token_idx // BLOCK_SIZE).to(tl.int64)
    block_offset = token_idx % BLOCK_SIZE
    values = tl.load(
        cache_ptr
        + block_idx * cache_stride_block
        + block_offset * cache_stride_token
        + head_idx * cache_stride_head
        + offsets
    )
    tl.store(
        output_ptr
        + token_idx * output_stride_token
        + head_idx * output_stride_head
        + offsets,
        values.to(tl.float32) * tl.load(scale_ptr),
    )


@triton.jit
def fp8_paged_decode_attention_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    output_ptr,
    context_lens_ptr,
    block_tables_ptr,
    k_scale_ptr,
    v_scale_ptr,
    q_stride_token,
    q_stride_head,
    output_stride_token,
    output_stride_head,
    k_stride_block,
    k_stride_token,
    k_stride_head,
    v_stride_block,
    v_stride_token,
    v_stride_head,
    block_table_stride,
    softmax_scale,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS_PER_KV: tl.constexpr,
    MAX_CONTEXT_LEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Minimal FP8 paged decode attention, adapted from vLLM Triton attention.

    One program computes one (sequence, query-head) row and uses online
    softmax while reading physical pages through the block table.  vLLM's
    unified kernel generalizes this to prefill, cascade, sliding-window and
    several scale modes; this teaching subset is intentionally decode-only.
    """
    token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)
    kv_head_idx = query_head_idx // NUM_QUERY_HEADS_PER_KV
    offsets_d = tl.arange(0, HEAD_DIM)
    q = tl.load(
        q_ptr
        + token_idx * q_stride_token
        + query_head_idx * q_stride_head
        + offsets_d
    ).to(tl.float32)
    context_len = tl.load(context_lens_ptr + token_idx)

    running_max = tl.full([1], -float("inf"), tl.float32)
    running_sum = tl.full([1], 0.0, tl.float32)
    accumulator = tl.zeros([HEAD_DIM], tl.float32)
    offsets_n = tl.arange(0, BLOCK_N)
    for start in range(0, MAX_CONTEXT_LEN, BLOCK_N):
        positions = start + offsets_n
        valid = positions < context_len
        physical_blocks = tl.load(
            block_tables_ptr
            + token_idx * block_table_stride
            + positions // BLOCK_SIZE,
            mask=valid,
            other=0,
        ).to(tl.int64)
        offsets_kv = (
            physical_blocks[:, None] * k_stride_block
            + (positions % BLOCK_SIZE)[:, None] * k_stride_token
            + kv_head_idx * k_stride_head
            + offsets_d[None, :]
        )
        key = tl.load(k_cache_ptr + offsets_kv, mask=valid[:, None], other=0.0)
        scores = tl.sum(q[None, :] * key.to(tl.float32), axis=1)
        scores = scores * (softmax_scale * tl.load(k_scale_ptr))
        scores = tl.where(valid, scores, -float("inf"))
        tile_max = tl.max(scores, axis=0)
        next_max = tl.maximum(running_max, tile_max)
        weights = tl.exp(scores - next_max)
        previous_scale = tl.exp(running_max - next_max)
        offsets_v = (
            physical_blocks[:, None] * v_stride_block
            + (positions % BLOCK_SIZE)[:, None] * v_stride_token
            + kv_head_idx * v_stride_head
            + offsets_d[None, :]
        )
        value = tl.load(v_cache_ptr + offsets_v, mask=valid[:, None], other=0.0)
        accumulator = accumulator * previous_scale + tl.sum(
            value.to(tl.float32) * weights[:, None], axis=0
        )
        running_sum = running_sum * previous_scale + tl.sum(weights, axis=0)
        running_max = next_max

    tl.store(
        output_ptr
        + token_idx * output_stride_token
        + query_head_idx * output_stride_head
        + offsets_d,
        accumulator / running_sum * tl.load(v_scale_ptr),
    )


def _is_fp8_kv_cache(k_cache: torch.Tensor, v_cache: torch.Tensor) -> bool:
    return k_cache.dtype == v_cache.dtype == torch.float8_e4m3fn


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    fp8_kv_cache = _is_fp8_kv_cache(k_cache, v_cache)
    store_kvcache_kernel[(N,)](
        key,
        key.stride(0),
        value,
        value.stride(0),
        k_cache,
        v_cache,
        slot_mapping,
        k_scale,
        v_scale,
        D,
        FP8_KV_CACHE=fp8_kv_cache,
    )


def gather_dequant_kvcache(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    num_tokens: int,
    scale: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Materialize a single request's FP8 paged cache for FA2 prefill."""
    num_kv_heads, head_dim = cache.shape[-2:]
    output = torch.empty(
        (num_tokens, num_kv_heads, head_dim), device=cache.device, dtype=output_dtype
    )
    gather_dequant_kvcache_kernel[(num_tokens, num_kv_heads)](
        cache,
        output,
        block_table,
        scale,
        num_tokens,
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        output.stride(0),
        output.stride(1),
        BLOCK_SIZE=cache.shape[1],
        HEAD_DIM=head_dim,
    )
    return output


def fp8_paged_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    scale: float,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    max_context_len: int,
) -> torch.Tensor:
    """FP8 paged decode wrapper for Qwen's 128-wide attention heads."""
    num_tokens, num_heads, head_dim = q.shape
    if head_dim != 128:
        raise ValueError("the teaching FP8 paged kernel currently requires head_dim=128")
    if num_heads % k_cache.size(2) != 0:
        raise ValueError("num query heads must divide evenly by KV heads")
    output = torch.empty_like(q)
    fp8_paged_decode_attention_kernel[(num_tokens, num_heads)](
        q,
        k_cache,
        v_cache,
        output,
        context_lens,
        block_tables,
        k_scale,
        v_scale,
        q.stride(0),
        q.stride(1),
        output.stride(0),
        output.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        block_tables.stride(0),
        scale,
        BLOCK_SIZE=k_cache.shape[1],
        HEAD_DIM=head_dim,
        NUM_QUERY_HEADS_PER_KV=num_heads // k_cache.size(2),
        MAX_CONTEXT_LEN=max_context_len,
        BLOCK_N=64,
        num_warps=4,
    )
    return output


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        # vLLM uses checkpoint-provided scales when available; otherwise its
        # default is 1.0.  Keep that deterministic fallback for this minimal
        # implementation rather than calibrating from one arbitrary request.
        self.register_buffer("k_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("v_scale", torch.tensor(1.0, dtype=torch.float32))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        # Standalone model forwards (before the runner binds a cache) still
        # need ordinary causal attention, e.g. checkpoint validation.
        if not k_cache.numel() or not v_cache.numel():
            return flash_attn_func(
                q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
                softmax_scale=self.scale, causal=True,
            ).squeeze(0)
        if context.slot_mapping is None or context.block_tables is None:
            raise RuntimeError(
                "paged KV attention requires slot mapping and block tables"
            )
        # Serving forwards always write through the logical-to-physical mapping
        # before FlashAttention reads the layer-local paged cache.
        store_kvcache(
            k, v, k_cache, v_cache, context.slot_mapping, self.k_scale, self.v_scale
        )
        if _is_fp8_kv_cache(k_cache, v_cache):
            # vLLM's FlashAttention FP8 path requires its FA3 fork on Hopper.
            # On this SM89/FA2 setup use the vLLM Triton-attention algorithm
            # for one-token decode, where direct paged FP8 reads matter most.
            if context.max_seqlen_q == 1 and context.context_lens is not None:
                return fp8_paged_decode_attention(
                    q,
                    k_cache,
                    v_cache,
                    context.context_lens,
                    context.block_tables,
                    self.scale,
                    self.k_scale,
                    self.v_scale,
                    context.max_seqlen_k,
                )
            # The scheduler currently issues one prefill request at a time.
            # FA2 cannot read scaled FP8 pages, so gather/dequantize only this
            # layer's active pages into a short-lived contiguous buffer.
            if context.block_tables.size(0) != 1:
                raise NotImplementedError(
                    "FP8 KV prefill currently supports one request per step"
                )
            k = gather_dequant_kvcache(
                k_cache,
                context.block_tables[0],
                context.max_seqlen_k,
                self.k_scale,
                q.dtype,
            )
            v = gather_dequant_kvcache(
                v_cache,
                context.block_tables[0],
                context.max_seqlen_k,
                self.v_scale,
                q.dtype,
            )
            return flash_attn_varlen_func(
                q,
                k,
                v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
            )
        if context.is_prefill:
            o = flash_attn_varlen_func(
                q, k_cache, v_cache,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.block_tables,
            )
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
