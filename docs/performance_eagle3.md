# Qwen3-8B + EAGLE3 性能

## 2026-08-21：Batch 1，Prompt 1024，Response 256，CUDA Graph

测试在 `nanovllm` conda 环境中完成（Torch 2.11.0+cu129、Transformers 5.13.0），
硬件为 NVIDIA RTX 5880 Ada。两侧使用相同的 Qwen3-8B target model 和
EAGLE3 checkpoint，采用 greedy 解码，`k=8`、`temperature=0`、
`ignore_eos=True`、关闭 prefix cache，并设置 `enforce_eager=False`。
Prompt 长度为 1024，要求生成 256 个 token。一次 warm-up 请求不计入延迟，
正式测量 3 次并取中位数。

| Backend | TTFT (ms) | TTPO (ms/token) | Drafted | Accepted | Acceptance | Decode 吞吐 (token/s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nano-vLLM | 115.81 | 4.636 | 825 | 663 | 80.36% | 215.7 |
| vLLM | 114.21 | 5.661 | 1,311 | 600 | 45.76% | 176.6 |

vLLM 的 acceptance 由 warm-up 后的 `SpecDecoding metrics` 日志得到，nano-vLLM
在 benchmark driver 中统计相同的 greedy proposal/target 比较。此次测试中，
nano-vLLM 的 TTPO 快约 1.22 倍，TTFT 基本相同（差异 1.4%）。主要原因是
acceptance 不同：vLLM 每个 256-token 请求约需要 55 个 speculative round，
nano-vLLM 约需要 35 个，因此 vLLM 执行了更多 draft 和 verification，即使其
单个 CUDA Graph kernel 更快。

Profiler self-time（一次正式请求；CPU 时间包含 host 侧等待 CUDA 的时间）：

| Backend | CPU self-time | CUDA self-time | CPU 占比 | GPU 占比 |
| --- | ---: | ---: | ---: | ---: |
| nano-vLLM | 1.335 s | 1.053 s | 55.9% | 44.1% |
| vLLM | 1.494 s | 1.554 s | 49.0% | 51.0% |

nano-vLLM 的同步阶段计时显示，主要耗时为 target verification（`71.5%`），
其次是 EAGLE proposal（`17.5%`）、prefill（`7.1%`）和 commit（`3.9%`）。
这些比例针对一次生成请求，应结合 acceptance 一起解读，因为 acceptance 越低，
需要执行的 verification round 越多。

## 2026-08-21：CUDA Graph，Batch 1，长 Prompt

硬件：NVIDIA RTX 5880 Ada Generation（48,140 MiB）。两侧使用相同的 Python
环境（`torch 2.11.0+cu129`、`vLLM 0.23.1rc1.dev775+g1a308c449`）、
Qwen3-8B target checkpoint 和 Qwen3-8B-speculator.eagle3 draft checkpoint。

配置：

| 设置 | 值 |
| --- | --- |
| Batch size | 1 |
| Prompt / response | 4,096 / 1,024 tokens |
| Draft method / k | EAGLE3 greedy / 8 |
| Sampling | temperature 0，`ignore_eos=True` |
| Prefix cache | 关闭 |
| CUDA Graph | 开启（`enforce_eager=False`） |
| 测试次数 | 1 次 warm-up，之后 3 次正式测试，报告中位数 |

TTFT 是从提交请求到观察到第一个输出 token 的时间。TTPO 是第一个输出 token
之后的耗时除以剩余 1,023 个输出 token。初始化、编译和 graph capture 不计入
测量。

| Backend | TTFT (ms) | TTPO (ms/token) | Decode 吞吐 (token/s) | 端到端 (ms) |
| --- | ---: | ---: | ---: | ---: |
| nano-vLLM | 526.12 | 13.006 | 76.89 | 13,839.91 |
| vLLM | 468.36 | 8.667 | 115.38 | 9,334.53 |
| nano / vLLM | 1.12x | 1.50x | 0.67x | 1.48x |

原始测试结果：

| Backend | TTFT (ms) | TTPO (ms/token) | 端到端 (ms) |
| --- | ---: | ---: | ---: |
| nano-vLLM | 518.88 | 13.027 | 13,845.13 |
| nano-vLLM | 526.12 | 13.002 | 13,827.58 |
| nano-vLLM | 534.85 | 13.006 | 13,839.91 |
| vLLM | 464.81 | 8.662 | 9,326.54 |
| vLLM | 468.36 | 8.667 | 9,334.53 |
| vLLM | 470.78 | 8.668 | 9,338.31 |

两侧 CUDA Graph 覆盖范围并不相同。nano-vLLM 捕获了常规 target decode graph
和 EAGLE batch-1 draft graph（`eagle_graph_sizes=[1]`），但 EAGLE target
verification 仍然 eager 调用 `forward_with_aux_hidden_states()`。vLLM 则报告了
`FULL_AND_PIECEWISE` target graph，以及 EAGLE prefill 和 decode graph。因此，
该测试比较的是当前两个系统的端到端实现，并不是完全相同的 graph 覆盖范围。

nano-vLLM 测得的 draft acceptance rate 为 21.03%。该值明显低于预期的 EAGLE
acceptance rate，原因是当前 fused draft kernel 改变了 BF16 reduction order，
同时 multi-token target verification 与 sequential target decode 在数值上并不
完全一致。该版本 vLLM client-side benchmark 没有记录 acceptance rate。

这次测试还暴露了一个跨越 256-token KV block boundary 的 scheduler bug：在普通
decode 路径之外追加 accepted 或 replacement token 时，没有完成尾部完整 block
的 finalize。现在 `Scheduler.postprocess_speculation()` 会在追加 replacement
token 前完成完整 block 的 finalize；没有追加 replacement 时也会立即 finalize。
