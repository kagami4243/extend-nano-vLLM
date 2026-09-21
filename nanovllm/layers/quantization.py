import torch
from torch import nn
import triton
import triton.language as tl

# The W4A16 kernel below is adapted from vLLM's
# vllm/model_executor/kernels/linear/mixed_precision/triton_w4a16.py
# (Apache-2.0).  This teaching version retains vLLM's GPTQ sequential INT4
# packing and fused dequant-GEMM loop, but supports only symmetric group-wise
# weights (zero-point 8), so it does not need qzeros/g_idx or vLLM's kernel
# selection framework.
#
# SPDX-License-Identifier: Apache-2.0


SUPPORTED_QUANTIZATIONS = ("w4a16", "fp8")
FP8_FORMATS = ("per_tensor", "per_token")
# Per-tensor improves Qwen3-8B prefill on the current Torch scaled_mm kernel.
# Per-token remains opt-in for its finer activation and weight scales.
DEFAULT_FP8_FORMAT = "per_tensor"
W4A16_GROUP_SIZE = 128


def normalize_quantization(quantization: str | None) -> str | None:
    if quantization is None:
        return None
    if not isinstance(quantization, str):
        raise TypeError("quantization must be a string or None")
    quantization = quantization.lower()
    if quantization not in SUPPORTED_QUANTIZATIONS:
        supported = ", ".join(SUPPORTED_QUANTIZATIONS)
        raise ValueError(
            f"unsupported quantization {quantization!r}; supported: {supported}"
        )
    return quantization


def normalize_fp8_format(fp8_format: str | None) -> str:
    """Normalize activation scale granularity for online FP8 quantization."""
    if fp8_format is None:
        return DEFAULT_FP8_FORMAT
    if not isinstance(fp8_format, str):
        raise TypeError("fp8_format must be a string or None")
    fp8_format = fp8_format.lower()
    if fp8_format not in FP8_FORMATS:
        supported = ", ".join(FP8_FORMATS)
        raise ValueError(
            f"unsupported fp8_format {fp8_format!r}; supported: {supported}"
        )
    return fp8_format


@triton.jit
def _w4a16_gemm_kernel(
    x_ptr,
    packed_weight_ptr,
    scale_ptr,
    output_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_ym,
    stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """vLLM-derived GPTQ-sequential W4A16 GEMM with FP32 accumulation."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )

        # GPTQ sequential packing: [K, N // 8] int32, with eight adjacent
        # output channels at bit offsets [0, 4, ..., 28].  This is the
        # layout consumed by vLLM's TritonW4A16LinearKernel.
        packed_n = pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)
        packed = tl.load(
            packed_weight_ptr
            + offs_k[:, None] * stride_wk + packed_n[None, :] * stride_wn,
            mask=(offs_k[:, None] < K) & (packed_n[None, :] < N // 8),
            other=0,
        )
        weight = tl.interleave(packed, packed)
        weight = tl.interleave(weight, weight)
        weight = tl.interleave(weight, weight)
        shifts = (offs_n % 8) * 4
        weight = (weight >> shifts[None, :]) & 0xF
        # Our symmetric signed INT4 [-8, 7] is stored as vLLM uint4b8:
        # q_uint4 = q_signed + 8.
        weight = weight - 8
        scale = tl.load(
            scale_ptr
            + (offs_k[:, None] // GROUP_SIZE) * N
            + offs_n[None, :],
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        weight = weight.to(x.dtype) * scale
        accumulator += tl.dot(x, weight)

    output_offsets = (
        output_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    )
    tl.store(
        output_offsets,
        accumulator,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def _replace_weight(module: nn.Module, weight: torch.Tensor) -> None:
    module.weight = nn.Parameter(weight, requires_grad=False)


@torch.inference_mode()
def _quantize_w4a16(module: nn.Module, group_size: int) -> None:
    weight = module.weight.detach()
    output_size, input_size = weight.shape
    if output_size % 8:
        raise ValueError("W4A16 requires output features divisible by 8")
    if input_size % group_size:
        raise ValueError(
            f"W4A16 input features ({input_size}) must be divisible by group size "
            f"({group_size})"
        )

    grouped = weight.float().view(output_size, input_size // group_size, group_size)
    scales = grouped.abs().amax(dim=-1).clamp_min(1e-12) / 7.0
    quantized = torch.round(grouped / scales.unsqueeze(-1)).clamp(-8, 7)
    # Match vLLM's uint4b8 symmetric convention: shift signed values into
    # [0, 15], transpose to [K, N], then pack eight N values per int32.
    quantized = (quantized.to(torch.int16).view(output_size, input_size) + 8)
    quantized = quantized.t().contiguous().to(torch.int32)
    shifts = torch.arange(8, device=weight.device, dtype=torch.int32) * 4
    packed = torch.sum(
        quantized.view(input_size, output_size // 8, 8) << shifts,
        dim=-1,
        dtype=torch.int32,
    )

    _replace_weight(module, packed.contiguous())
    module.register_buffer("weight_scale", scales.t().to(weight.dtype).contiguous())
    module.quantization = "w4a16"
    module.quant_group_size = group_size
    module.input_size_per_partition = input_size
    module.output_size_per_partition = output_size


@torch.inference_mode()
def _quantize_fp8(module: nn.Module, fp8_format: str | None = None) -> None:
    fp8_format = normalize_fp8_format(fp8_format)
    if not hasattr(torch, "_scaled_mm"):
        raise RuntimeError("FP8 quantization requires torch._scaled_mm")
    if module.weight.device.type == "cuda":
        major, minor = torch.cuda.get_device_capability(module.weight.device)
        if (major, minor) < (8, 9):
            raise RuntimeError("FP8 scaled GEMM requires CUDA capability 8.9+")

    weight = module.weight.detach()
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    if fp8_format == "per_token":
        # Rowwise quantization of the source [N, K] weight becomes a
        # [1, N] channel scale after the GEMM operand is transposed to [K, N].
        scale = (
            weight.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
            / fp8_max
        )
    else:
        scale = weight.float().abs().amax().clamp_min(1e-12) / fp8_max
    quantized = (weight.float() / scale).clamp(-fp8_max, fp8_max)
    # torch._scaled_mm expects B as [K, N]. Keep this transposed view
    # column-major, matching vLLM's online FP8 post-load conversion.
    _replace_weight(module, quantized.to(torch.float8_e4m3fn).t())
    weight_scale = (
        scale.float().reshape(1)
        if fp8_format == "per_tensor"
        else scale.float().t().contiguous()
    )
    module.register_buffer("weight_scale", weight_scale)
    module.quantization = "fp8"
    module.fp8_activation_format = fp8_format
    module.input_size_per_partition = weight.shape[1]
    module.output_size_per_partition = weight.shape[0]


@torch.inference_mode()
def quantize_model(
    model: nn.Module,
    quantization: str | None,
    *,
    w4a16_group_size: int = W4A16_GROUP_SIZE,
    fp8_format: str | None = None,
) -> None:
    """Quantize loaded LinearBase weights in place.

    Loading remains unchanged and therefore preserves TP shard semantics.
    Quantization happens once after all packed QKV/MLP shards are assembled.
    """
    quantization = normalize_quantization(quantization)
    if quantization is None:
        return
    if quantization == "fp8":
        fp8_format = normalize_fp8_format(fp8_format)

    from nanovllm.layers.linear import LinearBase

    for module in model.modules():
        if not isinstance(module, LinearBase):
            continue
        if quantization == "w4a16":
            _quantize_w4a16(module, w4a16_group_size)
        else:
            _quantize_fp8(module, fp8_format)


def _w4a16_linear(x: torch.Tensor, module: nn.Module) -> torch.Tensor:
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("W4A16 activations must be float16 or bfloat16")
    original_shape = x.shape
    x_2d = x.reshape(-1, original_shape[-1]).contiguous()
    output = torch.empty(
        (x_2d.shape[0], module.output_size_per_partition),
        dtype=x.dtype,
        device=x.device,
    )
    # Same CUDA-side tile policy as vLLM's Triton W4A16 wrapper. The source
    # kernel is tuned primarily for ROCm; this remains a correctness/teaching
    # kernel on NVIDIA, not a replacement for vLLM's Marlin path.
    block_m = 32 if x_2d.shape[0] <= 32 else (64 if x_2d.shape[0] <= 64 else 128)
    block_n = 64 if x_2d.shape[0] <= 64 else 128
    block_k = 32
    grid = (
        triton.cdiv(x_2d.shape[0], block_m),
        triton.cdiv(module.output_size_per_partition, block_n),
    )
    _w4a16_gemm_kernel[grid](
        x_2d,
        module.weight,
        module.weight_scale,
        output,
        x_2d.shape[0],
        module.output_size_per_partition,
        module.input_size_per_partition,
        module.quant_group_size,
        x_2d.stride(0),
        x_2d.stride(1),
        module.weight.stride(0),
        module.weight.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return output.view(*original_shape[:-1], module.output_size_per_partition)


def _fp8_linear(x: torch.Tensor, module: nn.Module) -> torch.Tensor:
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("FP8 scaled GEMM output must be float16 or bfloat16")
    original_shape = x.shape
    x_2d = x.reshape(-1, original_shape[-1]).contiguous()
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    fp8_format = normalize_fp8_format(
        getattr(module, "fp8_activation_format", DEFAULT_FP8_FORMAT)
    )
    if fp8_format == "per_token":
        input_scale = (
            x_2d.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
            / fp8_max
        )
    else:
        input_scale = x_2d.float().abs().amax().clamp_min(1e-12) / fp8_max
    x_fp8 = (x_2d.float() / input_scale).clamp(-fp8_max, fp8_max)
    x_fp8 = x_fp8.to(torch.float8_e4m3fn)
    scale_a = input_scale.float()
    if fp8_format == "per_tensor":
        scale_a = scale_a.reshape(1)
    try:
        output = torch._scaled_mm(
            x_fp8,
            module.weight,
            scale_a=scale_a,
            scale_b=module.weight_scale,
            out_dtype=x.dtype,
        )
    except RuntimeError as scaled_mm_error:
        if fp8_format != "per_token":
            raise
        # Some CUDA/Torch combinations expose scaled_mm but do not implement
        # rowwise scale_a. Keep per-token semantics correct on those builds;
        # a native rowwise kernel can be selected without changing the API.
        try:
            output = torch.mm(
                x_fp8.float() * scale_a,
                module.weight.float() * module.weight_scale,
            ).to(x.dtype)
        except Exception:
            raise scaled_mm_error
    if isinstance(output, tuple):
        output = output[0]
    return output.view(*original_shape[:-1], module.output_size_per_partition)


def apply_quantized_linear(
    x: torch.Tensor,
    module: nn.Module,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    if module.quantization == "w4a16":
        output = _w4a16_linear(x, module)
    elif module.quantization == "fp8":
        output = _fp8_linear(x, module)
    else:
        raise RuntimeError(f"invalid linear quantization: {module.quantization}")
    if bias is not None:
        output = output + bias
    return output
