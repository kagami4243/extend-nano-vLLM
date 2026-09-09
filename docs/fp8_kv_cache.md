# FP8 KV cache（单卡教学版）

设置 `kv_cache_dtype="fp8"` 后，KV page 以 `torch.float8_e4m3fn` 保存，而模型计算仍为
BF16/FP16。每个 KV 元素由 1 byte 取代 BF16 的 2 bytes，因此相同显存预算下 KV cache
可容纳约两倍 token；模型权重、activation 和临时 prefill buffer 不变。

```python
llm = LLM("/path/to/Qwen3-0.6B", kv_cache_dtype="fp8")
```

## 实现与 vLLM 对照

实现位于 `nanovllm/layers/attention.py`，采用与 vLLM 相同的 **FP8 E4M3 + 每层 K/V
per-tensor scale** 约定：写入时保存 `round(x / scale)`，读取时乘回 `scale`。没有来自
checkpoint 的 calibrated scale 时，vLLM 的默认值和本实现一样是 `1.0`。

- `store_kvcache_kernel` 对应 vLLM
  `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py` 的
  `reshape_and_cache_kernel_flash`；它按 `slot_mapping` 写入物理 page，并在 store 前量化。
- `fp8_paged_decode_attention_kernel` 的 online softmax、block-table 间接寻址和 FP8
  dequant 思路来自 vLLM
  `vllm/v1/attention/ops/triton_unified_attention.py`。这里仅保留 Qwen 的 decode 子集：
  `head_dim=128`、无 sliding window、无 ALiBi、FP8 per-tensor scale。
- FlashAttention 2.8.3 的公开 API 不接受 K/V descale 参数；因此 fp8 prefill 以
  `gather_dequant_kvcache_kernel` 将当前 request 的 page 临时反量化为 BF16/FP16，再调用
  现有 FlashAttention。持久 cache 仍是 FP8。

vLLM 在 Hopper 上可借其定制 FlashAttention 3 直接读 FP8 cache；当前 RTX 5880 Ada
(SM89) 不具备该路径。因此本项目的 decode 使用 Triton paged kernel，prefill 使用临时
dequant fallback。这是硬件/API 限制下的正确性优先实现，而不是端到端加速功能。

## 限制

- 自动启用 eager：现有 CUDA Graph capture 没有 FP8 prefill metadata，不能安全复用。
- 单 request prefill；当前 scheduler 本来也一次只提交一个 prefill request。
- 不与 EAGLE3、FP8 KV calibrated checkpoint scale、per-token/per-head scale、量化 MoE
  组合；不支持 `head_dim != 128`。
- FP8 可能改变 logits 或 greedy token。必须用目标模型和实际 prompt 做质量验证。

## 验证

```bash
conda run -n nanovllm python -m tests.test_fp8_kv_cache
```

该测试检查非连续物理 page 的 FP8 store/gather，以及 Triton decode 输出相对同一量化
cache 的 FP32 reference。2026-09-09 在 RTX 5880 Ada 上最大绝对误差为 `5.82e-5`。
Qwen3-0.6B 的 4-token greedy smoke test 与 BF16 cache 均输出
`[264, 10950, 17847, 13]`；这不是对所有 prompt 的质量保证。

同机、同一 Qwen3-0.6B、`gpu_memory_utilization=0.2`、block size 256 的 cache
allocation 实测为：BF16 `287` pages（`8,426,356,736` bytes），FP8 `568` pages
（`8,338,276,352` bytes），容量比为 `1.98x`。两次分配的总字节数接近同一个显存预算，
因此 page 数增长来自 element size 从 2 bytes 降至 1 byte。
