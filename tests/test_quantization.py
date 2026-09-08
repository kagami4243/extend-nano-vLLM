import argparse

import torch
import torch.nn.functional as F

from nanovllm.layers.linear import ReplicatedLinear
from nanovllm.layers.quantization import quantize_model


def run_linear_test(quantization: str) -> None:
    torch.manual_seed(0)
    dtype = torch.bfloat16
    layer = ReplicatedLinear(256, 128, bias=True).cuda().to(dtype)
    with torch.no_grad():
        layer.weight.normal_(mean=0.0, std=0.02)
        layer.bias.normal_(mean=0.0, std=0.01)
    x = torch.randn(7, 256, device="cuda", dtype=dtype)
    reference = F.linear(x, layer.weight, layer.bias)
    original_weight_bytes = layer.weight.numel() * layer.weight.element_size()

    quantize_model(layer, quantization)
    output = layer(x)
    cosine = F.cosine_similarity(
        output.float().flatten(), reference.float().flatten(), dim=0
    ).item()
    quantized_weight_bytes = (
        layer.weight.numel() * layer.weight.element_size()
        + layer.weight_scale.numel() * layer.weight_scale.element_size()
    )

    # W4A16 follows vLLM's GPTQ sequential format: each int32 packs eight
    # uint4b8 values along the output dimension.
    expected_dtype = (
        torch.int32 if quantization == "w4a16" else torch.float8_e4m3fn
    )
    assert layer.weight.dtype == expected_dtype
    assert output.dtype == dtype
    assert quantized_weight_bytes < original_weight_bytes
    threshold = 0.990 if quantization == "w4a16" else 0.999
    assert cosine >= threshold, (quantization, cosine)
    print(
        f"{quantization}: cosine={cosine:.6f}, "
        f"weight_bytes={original_weight_bytes}->{quantized_weight_bytes}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "quantization", choices=("w4a16", "fp8", "all"), default="all", nargs="?"
    )
    args = parser.parse_args()
    methods = ("w4a16", "fp8") if args.quantization == "all" else (args.quantization,)
    for method in methods:
        run_linear_test(method)


if __name__ == "__main__":
    main()
