# 最小量化实现：W4A16 与 FP8 W8A8

本文先解释量化解决什么问题，再说明 nano-vLLM 中两种教学实现的代码路径、
使用方式和边界。当前实现刻意只覆盖 dense Qwen3 的 Linear 层，不试图复刻 vLLM
完整的 checkpoint 格式、kernel 选择和硬件兼容矩阵。

## 1. 量化在做什么

浮点张量可以用一个低精度张量 `q` 和 scale `s` 近似表示：

```text
q = round(x / s)
x ≈ q * s
```

量化主要带来两类收益：

1. **减少模型权重显存和显存带宽。** Decode 阶段经常受读取权重的带宽限制，较小的
   权重可能提高吞吐。
2. **使用低精度 Tensor Core。** 如果 activation 和 weight 都进入低精度 GEMM，硬件
   可以提供更高的理论计算吞吐。

低 bit 数同时会引入舍入和截断误差。scale 粒度越细通常精度越好，但 scale 数量、
读取开销和 kernel 复杂度也会增加。

## 2. 两种实现的区别

| 模式 | 权重存储 | Linear 输入 | GEMM 乘法 | Linear 输出 | 主要目的 |
|---|---|---|---|---|---|
| `w4a16` | packed signed INT4 | FP16/BF16 | FP16/BF16 | FP16/BF16 | 减少权重容量和带宽 |
| `fp8` | FP8 E4M3 | 动态量化为 FP8 E4M3 | FP8 × FP8 | FP16/BF16 | 减少权重并使用 FP8 GEMM |

这里的 A16/A8 指 activation 精度。标准 FP8 推理并不表示模型中的每一个操作都用
FP8：RMSNorm、RoPE、attention、激活函数、residual、KV cache 和输出仍保持模型的
FP16/BF16 dtype。FP8 只用于 Linear GEMM 的输入和权重，scaled GEMM 再输出
FP16/BF16。这样才能保留数值范围并匹配常见 FP8 推理实现。

## 3. 参考 vLLM 的部分

本实现参考了本地 vLLM 的三层结构：

- `vllm/model_executor/layers/quantization/online/fp8.py`：先正常加载高精度
  checkpoint，再在 `process_weights_after_loading` 阶段转换权重。
- `vllm/model_executor/kernels/linear/scaled_mm/pytorch.py`：动态量化 activation，
  用 `torch._scaled_mm` 执行 FP8 scaled GEMM。
- `vllm/model_executor/kernels/linear/mixed_precision/triton_w4a16.py`：INT4 packed
  weight 在 Triton GEMM tile 内反量化，activation 保持 FP16/BF16。nano-vLLM 的
  W4A16 packing 和 kernel loop 是对此文件的 Apache-2.0 适配移植；本次参考的本地
  vLLM revision 是 `1a308c449`。

vLLM 还支持 GPTQ、AWQ、Marlin、CUTLASS、block/channel/token-wise scales、预量化
checkpoint 和多种硬件后端。本仓库没有复制这些重型基础设施。

## 4. W4A16 的具体实现

`w4a16` 使用简单的 symmetric round-to-nearest（RTN）在线量化：

- 每个 output channel 沿 input/K 维每 128 个元素共享一个 scale。
- signed INT4 范围为 `[-8, 7]`，加 8 后按 vLLM 的 `uint4b8` convention 存储。
- 连续 8 个 output channel 的值打包进一个 `int32`；packed weight 布局为
  `[K, N / 8]`，scale 布局为 `[K / 128, N]`。
- Triton kernel 使用 vLLM 的 GPTQ sequential unpack 次序，在当前 tile 内计算
  `(q_uint4 - 8) * scale`，随后执行 FP16/BF16 dot 并以 FP32 累加；不会在每次
  forward 创建完整 FP16/BF16 weight。

不计 scale 时，每个权重由 BF16/FP16 的 2 bytes 降为 0.5 byte。group scale 会带来
少量额外空间。该路径的意义首先是展示 weight-only quantization 的布局和执行过程；
它保留了 vLLM Triton W4A16 的公开 packing/unpack 算法，但移除了 `qzeros`、`g_idx`
和 kernel selector，只支持 symmetric RTN。vLLM 的该 Triton source 主要面向 ROCm
MI300；本仓库在 CUDA 上验证正确性，但它不是 vLLM 在 NVIDIA 上的生产性能路径
（Marlin/CUTLASS），不能期待同等性能。

## 5. FP8 W8A8 的具体实现

`fp8` 使用 E4M3 和 per-tensor symmetric scale：

- checkpoint 加载后，为每个 Linear weight 计算一次静态 scale，并转换为
  `torch.float8_e4m3fn`。
- 每次 forward 根据当前完整 activation tensor 的绝对最大值动态计算一个 scale，
  再转换为 E4M3。
- `torch._scaled_mm(A_fp8, W_fp8, scale_a, scale_b)` 真正接收两个 FP8 operand，
  结果转换回输入的 FP16/BF16 dtype。
- 当前要求 NVIDIA compute capability 8.9 或更高。

相对于 BF16/FP16，FP8 weight 从每元素 2 bytes 降为 1 byte。动态 per-tensor scale
代码最少、兼容性清楚，但精度通常不如 vLLM 在支持时选择的 dynamic per-token
activation scale。

## 6. 代码改动与调用路径

- `nanovllm/config.py`：增加 `quantization=None | "w4a16" | "fp8"`，并限制到
  dense Qwen3。
- `nanovllm/engine/model_runner.py`：checkpoint 和 TP shard 完成后调用
  `quantize_model`。这样不会改变原有 packed QKV/MLP loader。
- `nanovllm/layers/linear.py`：保留 TP Linear 类，只增加统一的量化 dispatch；
  RowParallelLinear 仍在 GEMM 后执行原来的 all-reduce。
- `nanovllm/layers/quantization.py`：权重转换、W4A16 Triton kernel、FP8 动态量化和
  scaled GEMM。
- `tests/test_quantization.py`：小矩阵数值、dtype 和存储测试。
- `tests/test_quantized_qwen3.py`：Qwen3-0.6B eager/CUDA Graph 以及 Qwen3-8B + EAGLE3
  组合 smoke test。
- `benchmarks/bench_quantization.py`：同一 workload 下的 BF16/W4A16/FP8 实际
  prefill、固定 batch decode 对比。

只量化继承 `LinearBase` 的 attention/MLP Linear。Embedding、LM head、KV cache 和
EAGLE3 draft model 不量化。EAGLE3 场景中只有 target Qwen3 使用所选量化模式。

## 7. 使用方法

W4A16：

```python
from nanovllm import LLM

llm = LLM(model_path, quantization="w4a16")
```

FP8 W8A8：

```python
llm = LLM(model_path, quantization="fp8")
```

可以与当前 EAGLE3 target 配置组合：

```python
llm = LLM(
    target_model_path,
    quantization="fp8",
    speculative_config={
        "method": "eagle3",
        "model": eagle3_model_path,
        "num_speculative_tokens": 3,
    },
)
```

## 8. 数值与功能验证

在 RTX 5880 Ada、Torch 2.11 环境中：

- 小矩阵 W4A16 与 BF16 reference 的 cosine similarity 为 `0.993376`，weight +
  scale 从 `65536` bytes 降到 `16896` bytes。
- 小矩阵 FP8 与 BF16 reference 的 cosine similarity 为 `0.999290`，weight + scale
  从 `65536` bytes 降到 `32772` bytes。
- Qwen3-0.6B 的 eager 和 CUDA Graph 路径均完成 8-token smoke test，两种模式各
  覆盖 112 个 Linear。
- Qwen3-8B target + EAGLE3 draft 均完成 W4A16、FP8 的 speculative smoke test，
  两种模式各覆盖 target 的 144 个 Linear。

运行命令：

```bash
conda run -n nanovllm python -m tests.test_quantization all
conda run -n nanovllm python -m tests.test_quantized_qwen3 w4a16 --model /path/to/Qwen3-0.6B --cuda-graph
conda run -n nanovllm python -m tests.test_quantized_qwen3 fp8 --model /path/to/Qwen3-0.6B --cuda-graph
conda run -n nanovllm python -m tests.test_quantized_qwen3 w4a16 --speculative --target-model /path/to/Qwen3-8B --draft-model /path/to/Qwen3-8B-eagle3
conda run -n nanovllm python -m tests.test_quantized_qwen3 fp8 --speculative --target-model /path/to/Qwen3-8B --draft-model /path/to/Qwen3-8B-eagle3
```

## 9. 实际推理性能

所有数据在 RTX 5880 Ada（SM 8.9）、Torch 2.11、eager、无 prefix cache 下测得。
每个模式独立进程加载，模型初始化中的 warmup/编译不计时。当前 scheduler 每步只
admit 一个 prefill sequence，因此表中的 prefill 是 16 个真实单序列 prefill step 的
总 token/总时间；decode 在所有请求已进入 running 后，以固定 16 路 batch 测量 64 步。

命令：

```bash
conda run -n nanovllm python -m benchmarks.bench_quantization none --model /path/to/Qwen3-0.6B
conda run -n nanovllm python -m benchmarks.bench_quantization w4a16 --model /path/to/Qwen3-8B
conda run -n nanovllm python -m benchmarks.bench_quantization fp8 --model /path/to/Qwen3-8B
```

`batch=16`、每请求 `prompt=512`、测量 `decode=64` 的实际生成结果：

| 模型 | 模式 | Prefill tok/s | 相对 BF16 | 16 路 Decode tok/s | 相对 BF16 |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | BF16 | 8,497 | — | 800 | — |
| Qwen3-0.6B | W4A16 | 7,967 | -6.2% | 451 | -43.6% |
| Qwen3-0.6B | FP8 | 6,503 | -23.5% | 439 | -45.1% |
| Qwen3-8B | BF16 | 5,089 | — | 606 | — |
| Qwen3-8B | W4A16 | 2,876 | -43.5% | 368 | -39.3% |
| Qwen3-8B | FP8 | 5,911 | +16.2% | 341 | -43.8% |

再使用对 quantized GEMM 更有利的单请求长 prefill（`prompt=4096`、`decode=1`）测量
Qwen3-8B：BF16 为 4,056 tok/s，W4A16 为 3,927 tok/s（-3.2%），FP8 为 4,639 tok/s
（+14.4%）。

结论：本实现的 FP8 确实在 8B 的长/较大 prefill GEMM 上加速；小模型或小矩阵 decode
中，动态 `amax`、FP32 cast 和 FP8 cast 的成本超过收益。W4A16 在所有实测条件下都没有
超过 BF16；它减少了权重存储，但当前 software unpack + 解量化后 BF16 dot 不使用 NVIDIA
INT4 Tensor Core。若目标是 NVIDIA W4A16 加速，应实现/接入 vLLM 的 Marlin-compatible
pre-quantized GPTQ/AWQ path，而不是把这个教学 kernel 当作生产 kernel。

## 10. 当前限制

1. 只支持从普通 FP16/BF16 dense Qwen3 checkpoint 在线转换，不读取 GPTQ/AWQ 或
   已序列化 FP8 checkpoint。模型已经在 CUDA 上完成加载后才在线量化，因此转换和
   packing 都在 GPU 上执行；启动时仍需先容纳高精度模型和转换临时张量的峰值显存。
2. W4A16 只有固定 group size 128 和 RTN，没有 calibration、AWQ/GPTQ error
   compensation，也没有经过性能调优。
3. FP8 只有 per-tensor weight/per-tensor dynamic activation scale，不支持 block-wise、
   per-channel、per-token 或静态 calibration scale。
4. 不支持 Qwen3-MoE、量化 embedding/LM head、量化 KV cache，也不量化 EAGLE3
   draft model。
5. 数值 smoke test 不等于模型质量评测。实际使用前应在目标任务上比较 perplexity、
   token agreement 或任务指标，并单独测量显存和吞吐。
