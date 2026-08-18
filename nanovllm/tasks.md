# Nano-vLLM 实现任务

单卡测试默认使用物理 GPU 1；本次 2026-08-18 验收使用 Qwen3-0.6B，并在 GPU 1、2、3、4
上完成 TP=4、TP=2+PP=2 和 DP=2+TP=2。自动 pytest 启动器仍会在目标 GPU 有计算任务时
跳过；手工长上下文/组合测试通过 `NANOVLLM_GPU_MEMORY_UTILIZATION` 和显式端口运行，
仅在剩余显存足够时继续。

## P0. 测试基础设施与可复现性

**当前状态：已完成最小版本。** 已实现唯一端口/run ID/共享内存名、幂等退出、
greedy sampling、GPU 空闲门禁和独立测试子进程。

**对应测试：** `tests/test_p0_infrastructure.py`

### 任务：使 GPU 测试隔离、可重复、可诊断

**涉及模块：** `config.py`、`engine/llm_engine.py`、
`engine/model_runner.py`，新增 `tests/`。

**实现路线：**

1. 在 `Config` 增加 `master_addr`、`master_port`、`run_id`。未传入时由
   `LLMEngine` 在启动子进程前统一分配空闲端口和 UUID，子进程不得自行生成。
2. 使用这些配置构造 NCCL `init_method`，替换 `ModelRunner` 中固定的
   `tcp://localhost:2333`。
3. 共享内存名改为 `nanovllm-{run_id}`。将 `close`、`unlink`、`barrier`、
   `destroy_process_group` 置于 `try/finally`，保证任一 worker 失败后可回收。
4. 新增分布式测试启动器：每个 case 在独立子进程运行，设置超时，收集每个
   rank 的 stderr；任一子进程非零退出即失败，避免 CUDA/NCCL 状态污染下一个 case。
5. 在 `Sampler` 支持 greedy 分支：`temperature == 0` 直接 `argmax`，保留
   `temperature > 0` 的随机采样路径。分布式正确性测试只使用 greedy。
6. 提供 Qwen3-0.6B 的固定 prompt、token IDs、随机种子和 Transformers
   reference fixture，reference 应输出 next-token logits 与 greedy token IDs。
7. 每个 rank 输出结构化诊断：global rank、各并行 rank、device、已分配/峰值
   显存和 process-group 成员。性能指标只记录，不替代正确性断言。

**验收：**

- 两次不同 `run_id` 的 TP=2 顺序运行均可在 GPU 2、3 正常退出。
- 人为使某个 worker 抛异常后，没有残留 `/dev/shm` 条目或存活子进程。
- 同一 prompt 重复运行的 greedy 输出完全一致。
- 测试结果包含所有目标 GPU 的 rank 与显存记录。

## P1. 固化并验证张量并行（TP）

**当前状态：TP=1/2/4 已完成并通过。** Qwen3-0.6B 的 TP=2、TP=4 greedy
token IDs 均与 TP=1 baseline 完全一致；TP=4 已在 GPU 1、2、3、4 实际运行。

**对应测试：** `tests/test_p1_tensor_parallel.py`

### 任务：用显式并行上下文替代隐式 world size

**涉及模块：** 新增 `distributed/parallel_state.py`，以及 `config.py`、
`engine/llm_engine.py`、`engine/model_runner.py`、`layers/` 下 TP 层。

**实现路线：**

1. 定义拓扑对象，包含 `dp_size`、`pp_size`、`tp_size`、`ep_size`、`cp_size`。
   第一版要求 `dp=pp=ep=cp=1`，并保留 `tensor_parallel_size` 作为 `tp_size` 的
   兼容别名。
2. 校验所有并行度乘积等于 world size；创建 TP group，在 parallel state 中暴露
   `tp_rank`、`tp_world_size` 及封装后的 collective。
3. 修改 `linear.py`、`embed_head.py`、`qwen3.py` 与 KV cache 分配，使其使用
   TP group，而非默认 world 的 `get_rank`、`all_reduce`、`gather`。
4. 保留当前 Q/K/V、MLP column/row、vocab 的切分规则；加载权重前校验维度可整除，
   异常信息必须指出不可整除的维度。
5. 在 debug 模式统计 TP all-reduce/gather 次数，证明所有 rank 实际参与计算而非
   仅被启动。

**模型与拓扑：**

- Qwen3-0.6B，TP=1/2/4：单测和冒烟。16 个 attention heads 与 8 个 KV heads
  均可被 4 整除。
- Qwen3-8B，TP=1/2/4：日常集成测试。
- Qwen3-32B，TP=4：四卡 dense 端到端测试。其 64 层、64 个 attention heads、
  8 个 KV heads 都适合四卡切分。

**验收：**

- 比较固定 prompt 下 TP=1 和 TP=2/4 的 next-token logits（BF16 容差），并要求
  greedy token IDs 完全相同。
- 所有 rank 均报告 TP group 成员关系，且 collective 计数非零。
- Qwen3-8B TP=4 时每卡权重显存约为切分前的四分之一；Qwen3-32B TP=4 必须能
  在四卡加载并完成生成。
- 退出后无 worker、process group、端口或共享内存残留。

## P2. Prefix Cache 与 Chunked Prefill

**当前状态：已完成教学版并通过长上下文验收。** Prefix cache 支持命中统计和陈旧 hash
清理；chunked prefill 每步推进一个 sequence，并按 `max_num_batched_tokens` 切分。暂未
恢复多请求 prefill batching。

**对应测试：** `tests/test_p2_prefix_chunked_prefill.py`

### 任务：将已有的 prefix cache 机制做成可观测、可回归的功能

**涉及模块：** `engine/block_manager.py`、`engine/sequence.py`、
`engine/scheduler.py`、`engine/model_runner.py`，新增 scheduler 测试。

**实现路线：**

1. 为 `BlockManager` 增加统计：完整 block hit/miss、复用 token 数、淘汰数、
   存活引用数；每批请求完成后由 engine 导出。
2. 保持现有链式 hash 设计，但 hash 命中后仍校验 token 序列；块被复用前删除旧的
   `hash_to_block_id` 映射，避免命中已被覆盖的 block。
3. 将 sequence 状态拆为 `num_computed_tokens` 和 `num_cached_tokens`：前者在每次
   部分 prefill 后推进，后者只统计可复用的完整 block。
4. 将 scheduler 的 prefill 从“整条请求准入”改为“按 token 预算准入”。当剩余
   prompt 超过 `max_num_batched_tokens` 时，仅调度可容纳的前缀。
5. 扩展 `prepare_prefill`，只处理本轮已调度的 token 范围，并在对应 KV slot 写入；
   未完成 prefill 的 sequence 在后续轮次继续运行，不能重复分配 block。
6. 增加公平性：每轮为已有 decode 请求保留预算，避免长 prefill 独占调度器。

**验收：**

- Prefix cache：Qwen3-0.6B 提交两个共享 1024 token 前缀、suffix 不同的请求；第二个
  请求命中 4 个 block，复用 1024 tokens，开启/关闭 cache 的输出一致。
- Chunked prefill：Qwen3-0.6B 使用 8192-token prompt 和
  `max_num_batched_tokens=1024`，共 8 个 prefill step；与单次 8192-token prefill
  的 token IDs `[192, 193, 194, 195]` 一致。
- TP=4 集成：Qwen3-0.6B 输出与 TP=1 baseline 一致。Qwen3-8B TP=4 仍未测试。

## P3. 流水线并行（PP）与数据并行（DP）

**当前状态：PP=2、DP=2 及 2+2 组合均已完成教学版。** DP 使用两个独立进程、副本 scheduler
和 KV cache，按 prompt 下标轮询分发并恢复原输出顺序。PP 将 Qwen3-0.6B 的 28 层
拆为 14/14，在 stage 间传递 `hidden_states + residual`，由末 stage 采样后将 token
返回 rank 0。TP=2+PP=2 和 DP=2+TP=2 均已在四卡实测；仍仅支持 eager、单 microbatch，
不做流水重叠。

**对应测试：** `tests/test_p3_pipeline_data_parallel.py`。DP=2 与 PP=2 均实际运行。

### 任务：先实现 PP，再通过统一拓扑组合 DP

**涉及模块：** `distributed/parallel_state.py`、`config.py`、`models/qwen3.py`、
`engine/model_runner.py`、`engine/llm_engine.py`、`engine/scheduler.py`。

**PP 实现路线：**

1. 模型构造接收 `pp_rank` 与 `pp_size`。rank 0 持有 embedding，末 stage 持有
   final norm/LM head，中间 stage 只构造所属的连续 layer 区间。
2. loader 只读取本 stage 的 tensor；不能先构造完整模型再删除层，否则 PP 显存验收
   没有意义。
3. 相邻 stage 使用 point-to-point 传递 activation。定义 prefill/decode 元数据、
   hidden states、positions、完成与 shutdown 信号的消息协议。
4. TP collective 仅在同一个 PP stage 的 TP group 内执行。先实现单 microbatch
   正确路径，正确后再实现流水化 microbatch 调度。

**DP 实现路线：**

1. 每个 DP group 是完整副本，拥有独立 scheduler、KV cache、PP/TP groups。
2. 新增 coordinator，初版轮询将 request 分配到 replica leader，并按 `seq_id`
   合并输出。
3. 第一版不跨 DP replica 共享 block/KV cache；跨副本 prefix cache 是独立优化项。

**验收：**

- Qwen3-32B，PP=4、TP=1：每 stage 恰有 16/64 层，greedy 输出等于 TP=1 reference。
- Qwen3-32B，PP=2、TP=2：检查 group membership，证明 PP 和 TP group 不混用。
- Qwen3-32B，DP=2、TP=2：并发发送两批请求，输出等于 TP=2 baseline，两个 replica
  leader 均收到请求。
- warmup 后报告单 replica 与双 replica 吞吐及其比值；吞吐不是机器无关的正确性门槛。

### 暂未完成：PP-aware 调度与流水填充

**当前限制：** `LLMEngine.step()` 调用 `ModelRunner.run()` 后同步等待末 stage
返回 token，随后立即执行 scheduler `postprocess`。因此一个 batch 必须完整经过所有
PP stage 后，下一 batch 才能进入 stage 0；当前只有 layer partition，不存在多个
in-flight microbatch 的流水重叠。

**目标：** 对 `PP=P`，允许最多 `P` 个独立微批同时在流水线中推进。stage 是固定的
模型层分区；scheduler step 是一次微批提交。稳态时，stage 0 处理新微批，后续 stage
同时处理更早提交的微批。单个 autoregressive request 在其 sampled token 返回前不能
再次 decode，但 scheduler 可以提交其他可运行 request 来填充流水线。

**涉及模块：** `engine/sequence.py`、`engine/scheduler.py`、
`engine/llm_engine.py`、`engine/model_runner.py`，以及必要的控制消息和测试辅助代码。

**实现路线：**

1. 为 sequence 增加 `IN_FLIGHT` 状态、generation/batch ID、
   `next_decode_eligible_step` 和未结算输出计数；scheduler 将已提交 sequence 从可调度
   队列移入 in-flight 集合，结果结算前不可再次调度或释放其 KV blocks。
2. 定义 `PPWorkItem`/`BatchMetadata`：至少包括 `batch_id`、seq IDs、prefill/decode
   标记、positions、slot mapping、context lengths、block tables 和采样参数。中间 stage
   除 hidden states 外仍需要这些 attention/KV 元数据。
3. 将 `LLMEngine` 改为 submit/poll 两阶段：每次 tick 先收割
   `(batch_id, token_ids)` 并调用 `postprocess`，再在 in-flight 容量内提交新微批，
   不再同步等待单个 `run()` 完成。`is_finished()` 还必须检查 in-flight batches。
4. 将 `run_pipeline_model()` 的阻塞 `dist.send/recv` 改为每个 stage 的常驻 worker loop，
   使用带 batch ID 的 `isend/irecv` 传递 activation；每 stage 维护至少 `PP` 组
   activation/ring buffers、通信 handles 和 CUDA events，防止在接收方使用前覆盖 buffer。
5. 末 stage 将 `(batch_id, token_ids)` 异步回传 coordinator。第一版可仅回传 rank 0，
   由 coordinator 在下次提交时向各 stage 提供更新后的元数据；若各 stage 持有请求状态，
   则改为广播 sampled tokens。
6. 为 block manager 引入 in-flight reservation 或引用计数。禁止 preempt/复用仍被
   pipeline batch 使用的 block；先完成 eager、PP=2，再扩展 PP>2、chunked prefill、
   prefix cache、CUDA graph 和 speculative decoding。

**验收：**

- PP=2、至少两个独立 request：trace 显示 stage 0 处理 batch B 时 stage 1 正在处理
  batch A；输出与同步 PP/TP=1 reference 一致。
- PP=3：填充和排空阶段无死锁，所有 batch ID 恰好结算一次，结果按 `seq_id` 恢复顺序。
- 请求在 in-flight 期间不可重复 decode、不可 preempt；KV block 不被过早释放或复用。
- worker 异常或 shutdown 后，所有未完成 P2P work 被回收，其他 stage 不永久阻塞。

### 已完成教学版：TP+PP 组合并行

**当前状态：** 已移除 TP 与 PP 的互斥限制，使用 `DP × PP × TP` rank 布局，TP 为最
内层维度。当前只验收同步单 microbatch 的 TP=2、PP=2 教学路径。

**目标拓扑：** 使用 `DP x PP x TP` rank 布局，TP 为最内层维度。对于 `PP=3, TP=2`：

```text
rank:      0      1      2      3      4      5
(pp,tp): (0,0)  (0,1)  (1,0)  (1,1)  (2,0)  (2,1)
TP groups: [0,1], [2,3], [4,5]
PP groups: [0,2,4], [1,3,5]
```

**实现路线：**

1. 在 `distributed/parallel_state.py` 根据 rank 网格显式创建 `dist.new_group()`：每个
   `(dp, pp)` 坐标一个 TP group，每个 `(dp, tp)` 坐标一个 PP group；保存 group、
   group ranks、局部 `tp_rank`/`pp_rank` 和 `src/dst` helpers，不能再以 `WORLD` 或
   `rank +/- 1` 替代。
2. 移除 `config.py` 中 `TP>1 && PP>1` 的拒绝条件，但保留 `TP * PP * EP == world_size`
   校验；第一阶段仍禁止 EP 与 TP/PP 的组合。
3. 将 TP layer 的 all-reduce/gather 限制在本 stage 的 TP group。权重加载、embedding、
   LM head、attention QKV 和 MLP 的 shard 规则保持 TP 语义，但每个 PP stage 仅加载自己
   的连续 layers。
4. 将 PP activation P2P 改为同一 `tp_rank` 的相邻 PP rank：例如 `(pp=0,tp=1)` 的
   rank 1 必须向 `(pp=1,tp=1)` 的 rank 3 发送，而不是向 rank 2 发送。
5. 明确 activation 的分片语义。第一版可在每个 TP rank 间传递完整 replicated
   hidden states；优化版可让每个 TP rank 向对应下一 PP rank 发送本地 slice，并在接收
   stage 的 TP group 内 all-gather 重建。residual 在 sequence parallel 等模式下可能是
   分片的，不能无条件 all-gather。
6. sampling/logits 只由最后 PP stage 的 TP rank 0 聚合/采样，再将 token 返回
   coordinator 或广播给本 PP replica 的其他 stage；所有 TP ranks 必须参与前向所需的
   collective，避免 collective 次序不一致导致死锁。
7. KV cache 容量按每个 PP stage 的本地层数和每个 TP rank 的 KV heads 计算；跨所有
   TP/PP ranks 做 MIN reduce 以得到所有 stage 都能容纳的 block 数。

**验收：**

- `PP=2, TP=2` 的 groups 分别为 TP `[0,1]`、`[2,3]`，PP `[0,2]`、`[1,3]`。
- Qwen3-0.6B 在 GPU 1、2、3、4 上的输出为 `[12095, 13, 576, 6722]`，与单卡
  baseline 一致。
- 每 rank 仅加载自己的 PP layer 区间及其 TP shard；所有 stage 的参数字节总和符合
  “按 PP 切层、每层按 TP 分片”的预期。
- Qwen3-0.6B 的 greedy logits/token IDs 与 `TP=1, PP=1` reference 一致；再覆盖
  PP=3、TP=2（模型结构可整除时）。
- 组合测试确认 TP collective、PP P2P 和采样回传顺序可完成；尚未覆盖 PP-aware 调度
  或长时间多请求压力。

### 已完成教学版：DP+TP 组合并行

DP wrapper 为每个 replica 启动独立的 TP=2 engine。四卡布局中 replica 0 使用 ranks
`[0,1]`，replica 1 使用 ranks `[2,3]`；请求按下标轮询分配，最后恢复原顺序。

**验收：** Qwen3-0.6B 的 4 个固定 prompt 输出与单卡 batch baseline 完全一致，两个
replica 均成功返回。跨 replica 的 prefix/KV cache 共享、负载均衡和故障恢复仍未实现。

## P4. MoE 与专家并行（EP）

**当前状态：已完成教学版 MoE 与 EP=1/2。** EP 没有独立 size：启用
`enable_expert_parallel` 时复用 TP ranks，因此 Qwen3-MoE 的非 expert 部分仍按 TP
切分，而每个 rank 持有完整的本地 expert 子集。使用
`/data1/model/qwen/Qwen/Qwen3-30B-A3B-Base`：48 个 MoE 层、128 experts、
top-8。EP=2 时 GPU 2、3 各持有连续的 64 个 experts，attention、router、
embedding 与 LM head 保持复制。

**对应测试：** `tests/test_p4_moe_expert_parallel.py`。包含 router 数学单测、packed
expert 权重加载单测，以及 GPU 1 上 EP=1、GPU 2/3 上 EP=2 的真实模型生成测试。

### 任务：建立模型注册表，实现 Qwen3-MoE 与 EP dispatch

**涉及模块：** 新增 `models/registry.py`、`models/qwen3_moe.py`、`layers/moe.py`，
以及 `utils/loader.py`、`distributed/parallel_state.py`、`engine/model_runner.py`。

**实现路线：**

1. 注册表按 `AutoConfig.model_type` 选择 `qwen3` 或 `qwen3_moe`。MoE 模型复用已有
   Qwen3 attention、RoPE、RMSNorm、KV cache 与 LM head，只替换 decoder 的 MLP。
2. `ExpertParallelMoE` 使用 `[local_experts, 2 * intermediate, hidden]` 的
   `gate_up_proj` 和 `[local_experts, hidden, intermediate]` 的 `down_proj`。
   loader 从 `experts.E.{gate,up,down}_proj.weight` 逐 tensor 写入对应 slice；权重跨
   safetensors shard 时不需要特殊处理。
3. router 在 FP32 中对 128 个 logits 做 softmax，再选择 top-8；当
   `norm_topk_prob=true` 时重新归一化，最后转回 hidden-state dtype。每个 expert
   计算 `down(SiLU(gate(x)) * up(x))`，按 routing weight 加权后 `index_add`。
4. EP=1 持有全部 experts。启用 EP 且 TP=2 时，EP group 复用 TP group，rank 0 持有
   `[0, 64)`，rank 1 持有 `[64, 128)`；attention、embedding 与 LM head 继续按 TP
   分片。非本地 expert 在读取 tensor 前跳过，避免复制完整专家权重。
5. 当前教学版在所有 EP rank 复制 token 和 router 计算，每卡只计算本地 expert
   contribution，每个 MoE 层固定执行一次 `all_reduce(SUM)` 得到完整输出。即使本卡
   没有命中 expert，也必须用全零输出参与 collective，避免死锁。
6. 诊断输出 expert-ID 区间、本地 expert 数、expert 参数字节数、local dispatch/
   return assignment 数和 EP all-reduce 次数。第一版没有 capacity 和 token 丢弃。

**模型与拓扑：** 当前真实测试使用 Qwen3-30B-A3B-Base。总权重约 56.9 GiB，
其中 expert 权重 54 GiB；EP=2 后每卡约 29.9 GiB 模型权重。EP=4 可自然扩展为每卡
32 experts，但尚未加入四卡回归。

**验收：**

- EP=1 能正确加载并生成两个 greedy tokens。
- EP=2 的 expert 区间为 `[0, 64)`、`[64, 128)`，每卡 expert 参数恰为 EP=1 的一半。
- 每个 rank 的 local dispatch 等于 local return，两个 rank 的 dispatch 总和等于
  完整 top-k assignment 数，且每个 rank 的 all-reduce 计数非零。
- EP=2 greedy token IDs 必须与 EP=1 完全一致。

**暂未完成：** 当前不是生产级 all-to-all token dispatch，没有融合/分组 GEMM，短
batch 会产生较多小 GEMM；也不支持 EP 与 TP/PP 组合。后续路线是先用两次
all-to-all 替换复制 token 路径，再增加 EP=4 和 Transformers 固定 logits/token
reference。当前实现只能称为“专家权重切分 + 输出归并”的教学 EP。

## P5. Speculative Decoding

**当前状态：仅完成 greedy proposal verification 单元。** 只有 Qwen3-0.6B 时可以让
draft 和 target 使用同一模型，但只能得到退化的 100% 接受，无法验证自然 mismatch
或加速。完整测试需要 Qwen3-8B target、Qwen3-0.6B draft，以及两套独立 KV cache
和 rollback。

**对应测试：** `tests/test_p5_speculative_decoding.py`。Verification 单元实际通过；
完整 GPU engine case 明确跳过。

### 任务：实现带可回滚 KV cache 的 target verification

**涉及模块：** `config.py`、`engine/sequence.py`、`engine/scheduler.py`、
`engine/model_runner.py`，新增 draft model 与 verification 辅助模块。

**实现路线：**

1. 增加显式 target/draft model 配置；引擎启动前校验 tokenizer vocab 与 special-token
   IDs 完全相同。
2. 扩展 `Sequence`，保存 proposal tokens 与 KV checkpoint：block-table 长度、最后
   block token 数、KV 写入位置。
3. draft 每个 decode sequence 最多 proposal `k` 个 token；target 批量验证 proposal
   加一个 bonus token。
4. greedy 模式接受最长的 target 匹配前缀。首次 mismatch 时只保留已接受 KV 条目，
   将 draft/target 状态恢复至 checkpoint，再继续普通 decode。
5. 输出 proposed、accepted、rejected、target-forward-token 计数；只有 greedy 语义
   完全正确后才实现概率采样 speculative decoding。

**模型与拓扑：** target 为 Qwen3-8B，draft 为 Qwen3-0.6B，先使用 TP=1；单卡正确后
再为 target 增加 TP=2。

**验收：**

- 启动时验证 tokenizer IDs 一致；不兼容模型对必须被拒绝。
- 多个 prompt、多种 proposal 长度下，speculative greedy token IDs 与 target-only
  greedy 完全相同。
- 人为制造 mismatch 后，cache/block-table 状态可恢复，后续输出仍等于 target-only。
- 报告 acceptance rate、每个输出 token 的 target forward 数与相对 target-only 吞吐。

## P6. 上下文并行（CP）

**当前状态：暂缓。** Qwen3-0.6B 可用于 CP=2 smoke，但当前 attention 和 worker 将
default world 当作 TP；在独立 CP group、全局 position 切分和跨 rank causal
attention 完成前，不能把 TP=2 的结果标记为 CP 通过。原计划 TP=2、CP=2 需要四卡；
两卡教学版应先实现 TP=1、CP=2、单 sequence prefill。

**对应测试：** `tests/test_p6_context_parallel.py`，当前带上述原因明确跳过。

### 任务：先支持长上下文 prefill 的 CP，再考虑 decode CP

**涉及模块：** `distributed/parallel_state.py`、`layers/attention.py`、
`engine/model_runner.py`、`engine/sequence.py`，新增长上下文测试。

**实现路线：**

1. 明确范围：CP 第一版只在 prefill 切分 sequence positions 和 KV cache；普通 decode
   仍在 TP group 中运行，不能宣称已经支持完整 decode CP。
2. 创建与 TP 正交的 CP group，将 sequence 按连续 token 区间切分，为每个 rank 指定
   position offset 与本地 KV cache 范围。
3. 选择并实现分布式 causal attention 算法：ring attention，或 blockwise all-gather
   加数值稳定的 distributed softmax。生产路径不得 all-gather 全量 KV cache。
4. 扩展 block table 与 slot mapping，提供 global-to-local block 转换；断言 rank 不能
   写入其他 CP rank 的 KV 范围。
5. 对短 sequence 提供 fallback，因为其 CP 通信成本可能高于节省的显存；阈值作为可调
   参数并在文档中说明。

**模型与拓扑：** Qwen3-8B，TP=2、CP=2，使用 16K/32K synthetic-token prompt；
Qwen3-0.6B 只用于更小的功能冒烟。

**验收：**

- 比较同一长 prompt 下 TP=2、CP=1 与 TP=2、CP=2 的 logits/token IDs。
- 验证每个 CP rank 仅拥有本地 KV 范围，且跨分区 prompt 实际产生跨 rank attention 通信。
- 与 TP=2 baseline 一同报告每卡峰值显存和 prefill 吞吐，不设机器相关的绝对加速阈值。
