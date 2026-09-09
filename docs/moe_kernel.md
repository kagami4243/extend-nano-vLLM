# MoE grouped kernel（教学版）

`ExpertParallelMoE` 在 CUDA BF16/FP16 推理时使用两个 Triton grouped GEMM：

1. 按本 rank 的 expert id 排序 top-k 路由，并将每个 expert 的 token 行补齐到 16；
2. 一次 kernel launch 计算所有本地 expert 的 `gate_up`；
3. GPU 上执行 `SiLU(gate) * up`，第二次 grouped GEMM 计算 `down`，同时乘路由权重；
4. 用一次 `index_add` 合并同一 token 的 top-k 贡献；EP 模式再执行原有的
   `all_reduce(SUM)`。

这替换了过去的 `selected_experts.unique().tolist()`、逐 expert `torch.where` 和两次
`F.linear`。路由、权重布局和 EP 行为没有改变；CPU、非 BF16/FP16 权重或没有 Triton
时自动保留原始 Python reference path。

## 与 vLLM 的关系

实现的分块 grouped-GEMM 布局来自本机 vLLM revision `1a308c449` 的
`vllm/model_executor/layers/fused_moe/fused_moe.py`（Apache-2.0）。完整 vLLM 会在
CUDA 上通过 `moe_align_block_size` 自定义 op 构造排序和 block table，并按硬件选择
Triton、FlashInfer CUTLASS/TRT-LLM、AITER 等后端；量化 MoE 还有额外后端。

nano-vLLM 刻意只保留 unquantized Triton 路径：排序/16 行对齐由 GPU PyTorch tensor
操作完成，因此无需引入 vLLM 编译扩展或额外 GitHub 依赖。试验过 FlashInfer 的
CUTLASS 后端，但它在本机 SM89 首次运行需要 JIT 编译，启动成本不适合作为本仓库默认
教学路径。因而没有下载 DeepGEMM 或其他第三方仓库，也没有创建 `third_party/`。

## 限制

- 这是 **grouped** GEMM，不是完整生产级单 kernel fused MoE：中间 SwiGLU、排序和
  combine 仍是独立 GPU 操作。
- EP 仍复制 token/router，再 all-reduce 局部 expert 输出；尚未实现 all-to-all dispatch/
  combine。
- 只支持 BF16/FP16 unquantized expert weights；量化 MoE、shared experts、capacity 与
  dynamic expert placement 尚未覆盖。

## 验证与基准

```bash
conda run -n nanovllm python -m tests.test_moe_kernel
conda run -n nanovllm python -m benchmarks.bench_moe_kernel --tokens 256
```

基准默认采用 Qwen3-30B-A3B 的 MoE 形状（H=2048、I=768、128 experts、top-8），但只
分配随机专家权重；它报告本地 expert 执行（含排序/对齐）的时间，不代表端到端生成速度。

在本机 RTX 5880 Ada 上，2026-09-08 的 `--tokens 256 --iterations 20` 结果为：原始
逐 expert Python 路径 `20.844 ms`，grouped Triton 路径 `3.541 ms`，即 `5.89x`；同次
运行相对 reference 的最大绝对误差为 `0.00585938`。该数字不包含 router、EP
`all_reduce`、attention 或采样；短 decode batch 的每 expert padding 和排序成本更显著，
不能据此推断端到端同等加速。
