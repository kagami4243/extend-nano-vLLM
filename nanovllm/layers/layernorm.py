import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _deterministic_rms_kernel(
    x_ptr, weight_ptr, output_ptr,
    input_token_stride, input_head_stride, input_dim_stride,
    output_token_stride, output_head_stride, output_dim_stride,
    weight_stride,
    HEADS: tl.constexpr, HIDDEN: tl.constexpr, EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // HEADS
    head = row % HEADS
    columns = tl.arange(0, BLOCK)
    values = tl.load(
        x_ptr + token * input_token_stride + head * input_head_stride
        + columns * input_dim_stride,
        mask=columns < HIDDEN, other=0,
    ).to(tl.float32)
    variance = tl.sum(values * values, 0) / HIDDEN
    normalized = (values * tl.rsqrt(variance + EPS)).to(x_ptr.dtype.element_ty)
    weight = tl.load(
        weight_ptr + columns * weight_stride,
        mask=columns < HIDDEN, other=0,
    ).to(tl.float32)
    result = normalized.to(tl.float32) * weight
    tl.store(
        output_ptr + token * output_token_stride + head * output_head_stride
        + columns * output_dim_stride,
        result, mask=columns < HIDDEN,
    )


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        deterministic_cuda: bool = False,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.deterministic_cuda = deterministic_cuda
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def rms_forward_deterministic(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim not in (2, 3) or x.shape[-1] != self.weight.numel():
            raise ValueError("deterministic RMSNorm expects [tokens, (heads,) hidden]")
        tokens = x.shape[0]
        heads = x.shape[1] if x.ndim == 3 else 1
        input_head_stride = x.stride(1) if x.ndim == 3 else 0
        output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
        output_head_stride = output.stride(1) if x.ndim == 3 else 0
        _deterministic_rms_kernel[(tokens * heads,)](
            x, self.weight, output,
            x.stride(0), input_head_stride, x.stride(-1),
            output.stride(0), output_head_stride, output.stride(-1),
            self.weight.stride(0),
            HEADS=heads, HIDDEN=x.shape[-1], EPS=self.eps,
            BLOCK=triton.next_power_of_2(x.shape[-1]),
        )
        return output

    def rms_forward_eager(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        # float() aliases FP32 input, while the normalization below is in-place.
        if orig_dtype == torch.float32:
            x = x.clone()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.rms_forward_eager(x)

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float()
        if orig_dtype == torch.float32:
            x = x.clone()
        x.add_(residual.float())
        # Keep the pre-norm residual separate from the in-place FP32 work buffer.
        residual = x.clone() if orig_dtype == torch.float32 else x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            if self.deterministic_cuda and x.is_cuda:
                return self.rms_forward_deterministic(x)
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
