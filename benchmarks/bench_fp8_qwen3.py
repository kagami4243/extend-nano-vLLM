"""Benchmark the FP8 GEMM kernel used by nano-vLLM and optional vLLM baselines."""

import argparse
import importlib.util
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch


def benchmark_kernel_names() -> list[str]:
    names = ["nano_vllm_torch_scaled_mm"]
    names.extend(f"vllm_{name}" for name in vllm_kernel_selection_report()["kernels"])
    return names


def vllm_kernel_selection_report() -> dict:
    """Describe vLLM's source-level FP8 dispatch without importing CUDA ops."""
    spec = importlib.util.find_spec("vllm")
    if spec is not None and spec.submodule_search_locations:
        root = Path(next(iter(spec.submodule_search_locations)))
    else:
        sibling = Path(__file__).resolve().parents[2] / "vllm" / "vllm"
        if not sibling.exists():
            return {
                "available": False,
                "kernels": [],
                "activation_policy": "unknown",
                "cuda_priority": [],
                "ada89_expected": None,
            }
        root = sibling
    init_path = root / "model_executor" / "kernels" / "linear" / "__init__.py"
    fp8_path = root / "model_executor" / "layers" / "quantization" / "online" / "fp8.py"
    text = ""
    if init_path.exists():
        text += init_path.read_text()
    if fp8_path.exists():
        text += fp8_path.read_text()
    kernels = [
        name for name in (
            "CutlassFP8ScaledMMLinearKernel",
            "PerTensorTorchFP8ScaledMMLinearKernel",
            "MarlinFP8ScaledMMLinearKernel",
        ) if name in text
    ]
    policy = (
        "per-token activation when cutlass_fp8_supported"
        if "cutlass_fp8_supported" in text else "unverified"
    )
    priority = [
        name for name in (
            "MarlinFP8ScaledMMLinearKernel",
            "FlashInferFP8ScaledMMLinearKernel",
            "CutlassFP8ScaledMMLinearKernel",
            "PerTensorTorchFP8ScaledMMLinearKernel",
        ) if name in text
    ]
    return {
        "available": True,
        "kernels": kernels,
        "activation_policy": policy,
        "cuda_priority": priority,
        # On SM89 Marlin is disabled by default (unless a test override is
        # set) and FlashInfer requires SM100; CUTLASS is therefore the source
        # expected candidate when its custom op is built.
        "ada89_expected": (
            "CutlassFP8ScaledMMLinearKernel"
            if "CutlassFP8ScaledMMLinearKernel" in text
            else None
        ),
    }


def validate_result(result: dict) -> None:
    if not result.get("kernel"):
        raise ValueError("benchmark result must identify the kernel")
    if result.get("fp8_format") not in {"per_tensor", "per_token"}:
        raise ValueError("benchmark result must identify per_tensor or per_token")
    if float(result.get("mean_ms", 0.0)) <= 0:
        raise ValueError("benchmark result must contain positive mean_ms")


def measure(
    rows: int,
    hidden: int,
    out_features: int,
    warmup: int,
    runs: int,
    fp8_format: str = "per_token",
):
    from nanovllm.layers.quantization import _fp8_linear

    if fp8_format not in {"per_tensor", "per_token"}:
        raise ValueError("fp8_format must be per_tensor or per_token")

    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    module = type("Fp8Module", (), {})()
    weight = torch.randn(out_features, hidden, device="cuda", dtype=torch.bfloat16)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    if fp8_format == "per_token":
        scale = weight.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / fp8_max
    else:
        scale = weight.float().abs().amax().clamp_min(1e-12) / fp8_max
    # Keep B column-major, matching _quantize_fp8 and the layout accepted by
    # CUDA CUBLASLt for small M dimensions.
    module.weight = (weight / scale).to(torch.float8_e4m3fn).t()
    module.weight_scale = (
        scale.float().t().contiguous()
        if fp8_format == "per_token"
        else scale.float().reshape(1)
    )
    module.fp8_activation_format = fp8_format
    module.output_size_per_partition = out_features

    for _ in range(warmup):
        _fp8_linear(x, module)
    torch.cuda.synchronize()
    samples = []
    for _ in range(runs):
        start = perf_counter()
        _fp8_linear(x, module)
        torch.cuda.synchronize()
        samples.append((perf_counter() - start) * 1000)
    result = {
        "kernel": "nano_vllm_torch_scaled_mm",
        "fp8_format": fp8_format,
        "rows": rows,
        "hidden": hidden,
        "out_features": out_features,
        "mean_ms": mean(samples),
        "median_ms": sorted(samples)[len(samples) // 2],
        "device": torch.cuda.get_device_name(),
        "capability": "%d.%d" % torch.cuda.get_device_capability(),
        "torch": torch.__version__,
    }
    validate_result(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--out-features", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument(
        "--fp8-format", choices=("per_tensor", "per_token"), default="per_token"
    )
    args = parser.parse_args()
    print(json.dumps(measure(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
