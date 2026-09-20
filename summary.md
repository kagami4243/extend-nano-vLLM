# LLMEngine 与 Scheduler 执行流程

本文总结 `nanovllm/engine/llm_engine.py` 中的 `LLMEngine.step()`，以及
`nanovllm/engine/scheduler.py` 中的 `Scheduler.schedule()`。

## 总览

一次普通推理 step 是一次完整的调度事务：选择本轮请求、执行模型、提交结果并更新
请求与 KV cache 状态。

```text
Scheduler.schedule()
  -> (seqs, is_prefill)
ModelRunner.call("run", seqs, is_prefill)
  -> token_ids
Scheduler.postprocess(seqs, token_ids, is_prefill)
  -> 更新 sequence 状态、KV block 与队列
```

`step()` 最后收集本轮完成请求的 `(seq_id, completion_token_ids)`，并返回供
`generate()` 汇总。[`LLMEngine.step()`](nanovllm/engine/llm_engine.py) 是引擎中将
调度、模型执行和状态提交连接起来的关键边界。

## LLMEngine.step()

### 普通 prefill / decode 路径

1. 调用 `scheduler.schedule()`，取得本轮执行的 `seqs` 和阶段标记 `is_prefill`。
2. 调用 `model_runner.call("run", seqs, is_prefill)`。
   - 单卡时这是本地方法调用。
   - 模型并行时，rank 0 会先通过共享内存通知其他 `ModelRunner` worker 执行相同命令，
     以保持各 rank 同步。
3. `ModelRunner.run()` 根据阶段准备不同的输入与 paged KV metadata：
   `input_ids`、positions、slot mapping、context lengths 和 block tables；随后运行模型，
   并在应采样的 rank 产生每个 sequence 的下一个 token。
4. 调用 `scheduler.postprocess()` 提交结果：追加 token、推进已计算 token 数、迁移队列，
   并在 EOS 或达到 `max_tokens` 时释放 KV block。

### EAGLE3 speculative decode 路径

启用 EAGLE3 后，prefill 仍使用普通 `run()`；但 decode 阶段改为：

```text
reserve_speculation
  -> propose_eagle3_batch
  -> verify_eagle3_batch
  -> verify_greedy_proposals
  -> postprocess_speculation
  -> commit_eagle3 或 release_eagle3_state
```

原因是 draft model 会先写入 proposal 的 KV，target model 再一次性验证 proposal。调度器
必须提前保留足够的物理 KV block，并在 target 拒绝部分 proposal 后回滚逻辑 token 与多余
block。

## Scheduler.schedule()

`Scheduler` 维护两个队列：

```text
waiting: 等待 prefill 或被抢占、等待重新 prefill 的 sequence
running: 已完成 prefill、等待下一轮 decode 的 sequence
```

`schedule()` 返回 `(seqs, is_prefill)`：`seqs` 是本轮模型执行的请求，`is_prefill`
决定 `ModelRunner` 选择 prefill 还是 decode 输入组织方式。

### Prefill 调度

当前实现每轮最多选择一个 `waiting` sequence 做 prefill：

1. 取 `waiting[0]`。
2. 首次执行时，若没有 `block_table`，通过 `BlockManager.allocate(seq)` 为 sequence 的当前
   token 长度建立 logical block table。
3. 分配按 `ceil(num_tokens / block_size)` 个 block 计算，不等同于本轮实际计算的 token 数。
   完整 prefix block 可由 prefix cache 复用；未命中的 block 使用空闲物理 KV page。
4. 本轮实际 prefill token 数为：

   ```python
   min(seq.num_tokens - seq.num_computed_tokens, max_num_batched_tokens)
   ```

   因此长 prompt 会以多个 chunked prefill step 完成，即 block table 可以已分配完成，
   实际 forward 仍按 token budget 分段执行。
5. prefill 完成后，`postprocess()` 将 sequence 从 `waiting` 转移到 `running`，并追加
   prefill 产生的首个生成 token。

### Decode 调度

decode 阶段从 `running` 队列依次取 sequence，最多调度 `max_num_seqs` 个：

1. 每个被选中的 sequence 本轮只计算一个 token。
2. 如果写入该 token 会跨入新的 KV block，先检查空闲 block 是否足够，并在需要时扩展
   block table。
3. 选中的 sequence 会按原顺序重新放回 `running`，等待 `postprocess()` 根据本轮输出更新
   或移除它们。
4. 若 KV block 不足，调度器会抢占一个 running sequence：释放其 block table，将其放到
   `waiting` 队首；该 sequence 之后需要重新 prefill。

### Prefill 与 decode 的公平性

当前策略不是“只要有 waiting request 就持续优先 prefill”。当 `waiting` 和 `running`
都非空时，`last_step_was_prefill` 使两类 step 交替：

```text
prefill 一个 waiting sequence
decode 一批 running sequences
prefill 一个 waiting sequence
decode 一批 running sequences
...
```

这避免连续 prefill 使已开始生成的请求饿死。它也意味着当前实现尚未将多个 prefill
request 合并为同一次 forward；一次 prefill step 只含一个 sequence，而 decode step 才会
形成多 sequence batch。

## ModelRunner

`ModelRunner` 负责在每个 model-parallel rank 上加载本地模型分片、分配该 rank 的 KV
cache、组织本轮模型输入，并执行 eager forward 或 CUDA Graph replay。普通执行主线为：

```text
allocate_kv_cache（初始化时一次）
  -> prepare_prefill 或 prepare_decode（每个 step）
  -> run_model
  -> sampler（仅最后 PP stage、TP rank 0、EP rank 0）
```

### KV cache 分配

`allocate_kv_cache()` 按物理 KV block/page 估算显存，而不是先按单 token 分配。单个 block
的字节数为：

```text
2 (K/V)
× num_local_layers
× block_size
× num_kv_heads_per_tp_rank
× head_dim
× kv_cache_dtype.itemsize
```

再根据 `gpu_memory_utilization` 对应的可用显存计算 `num_kvcache_blocks`；模型并行时通过
`all_reduce(MIN)` 使所有 rank 使用相同的 block 数。最终每个 rank 持有：

```text
[2, num_local_layers, num_kvcache_blocks, block_size,
 num_kv_heads_per_tp_rank, head_dim]
```

的连续 `torch.Tensor`，并将每层 attention 的 `k_cache`、`v_cache` 绑定到该 tensor 的
对应 layer slice。

这里要区分物理存储和 sequence 的逻辑地址：底层 KV cache tensor 与每个物理 page 内的
token 均连续；但一个 sequence 的 logical block 可以映射到不连续的 physical page，原因
包括空闲 page 分配、prefix cache 复用和 preemption/re-prefill。

```text
sequence logical block:  0   1   2
physical page id:       17   4  39
```

`block_table` 保存上述 logical-block 到 physical-page 的映射，主要供 attention 读取历史
KV。`slot_mapping` 则保存本轮每个输入 token 的实际写入 slot：

```text
physical_page = block_table[token_position // block_size]
slot_mapping  = physical_page * block_size + token_position % block_size
```

因此，`block_table` 解决 page 级间接寻址，`slot_mapping` 指定本次 K/V scatter write 的
扁平物理位置。

### 输入准备

`prepare_prefill()` 将本轮 token 拼接为一维 `input_ids` 和 `positions`，并构造：

```text
cu_seqlens_q / cu_seqlens_k
max_seqlen_q / max_seqlen_k
slot_mapping / block_tables
```

这些 metadata 供 FlashAttention varlen paged path 使用。长 prompt 的本轮范围由
`num_computed_tokens` 与 `num_scheduled_tokens` 决定，因此只会准备当前 chunk。

`prepare_decode()` 为每个 sequence 准备一个 `seq.last_token`，同时构造 `context_lens`、
`slot_mapping`、`block_tables`。它也生成 `cu_seqlens_q/k`，使 eager decode 可以使用
paged-varlen attention；此时每个 query length 为 1。

### Eager 与 CUDA Graph

`run_model()` 的选择条件为：

```text
prefill，或 enforce_eager=True，或 decode batch size > 512
  -> eager forward
否则
  -> CUDA Graph replay
```

所以 `capture_cudagraph()` 只捕获普通 decode，不捕获 prefill。prefill 的总 token 数、
各 sequence 长度、`cu_seqlens` 与最大长度随请求变化，无法直接使用当前固定 shape 的 graph
缓冲区。

初始化时，`capture_cudagraph()` 为一组固定 decode batch size 捕获 graph：

```text
1, 2, 4, 8, 16, 32, ... , max_bs（最大不超过 512）
```

实际 decode batch 为 `bs` 时，选择最小的 `captured_bs >= bs` 的 graph。真实的
`input_ids`、positions、slot mapping、context lengths 和 block tables 写入固定 buffer 的
前 `bs` 行，剩余行作为 padding；replay 后只取前 `bs` 行输出。

capture 时 context 使用 `is_prefill=False`，因此 graph 内固化的是
`flash_attn_with_kvcache` decode attention 分支。与之不同，普通 eager decode 的
`prepare_decode()` 当前设置 `is_prefill=True`，会走 `flash_attn_varlen_func` 的
paged-varlen 路径。这是当前实现中 eager decode 和 graph decode 的 attention 路径差异。

## BlockManager 与 Prefix Cache

`BlockManager` 管理物理 KV page 的分配、释放、共享与 prefix cache 索引。它维护：

```text
blocks:            所有物理 Block 元数据
free_block_ids:    当前可分配的 physical page id
used_block_ids:    当前被引用的 physical page id
hash_to_block_id:  完整 token prefix hash -> physical page id
```

### Block 元数据

每个 `Block` 除了 `block_id` 之外还保存：

- `token_ids`：该 logical block 的 token 列表；用于在 hash 命中后再次确认内容，避免仅依赖
  hash 判断。
- `hash`：截至该完整 block 末尾的整段 token prefix 标识。
- `ref_count`：当前有多少 active sequence 的 `block_table` 引用该 physical page。

多个 sequence 的 prompt 有相同完整前缀时，它们可以引用同一个 physical KV page。任一
sequence 结束或被抢占时，`deallocate()` 会将其所有 block 的 `ref_count` 减一；只有计数
归零时，page 才回到 `free_block_ids`。

### 链式 prefix hash

`compute_hash(token_ids, prefix)` 对 token 的字节表示计算 xxHash；若有前序 hash，则先将
该 hash 写入 hash 输入。因此第 `i` 个完整 block 的 hash 可表示为：

```text
h_i = hash(h_(i-1), token_ids_of_block_i)
```

它代表的不是“当前 256 个 token 是否相同”，而是“从 sequence 开头到当前 block 末尾的 token
前缀是否相同”。例如：

```text
prefix A + block X
prefix B + block X
```

即使 `block X` 的 token 相同，只要 `prefix A != prefix B`，其链式 hash 就不同，不能复用
KV page。Transformer 的后续层会让该 block 内 token 的 hidden state、K/V 依赖此前上下文；
本实现又将各层的 KV page 作为整体复用，因此完整 token 前缀必须一致。

只有完整 block 会得到可缓存 hash。最后一个不足 `block_size` 的尾 block 的 hash 为 `-1`，
因为后续生成仍会写入该 page，其内容尚不稳定，不能安全作为 prefix cache 项。

### allocate()：最长连续 prefix 复用

首次为 sequence 建立 `block_table` 时，`allocate(seq)` 从第一个 logical block 开始处理：

```text
计算当前完整 block 的 chained hash
  -> 在 hash_to_block_id 查询候选 physical page
  -> 校验保存的 hash 与 token_ids
  -> 命中：复用 page，ref_count += 1，增加 num_cached_tokens
  -> 未命中：从 free_block_ids 分配新 page
  -> 将当前 block 写入 block_table，并登记完整 block 的 hash
```

`cache_miss` 表示当前 sequence 尚未建立到该位置的连续缓存前缀：一旦发生 miss，后续 block
不再作为该 sequence 的已计算 prefix 复用，并会分配新 page。这保证
`num_cached_tokens` 只表示从开头连续可跳过 forward 的 token 数。

新分配的完整 block 仍会登记到 `hash_to_block_id`，因此它们可作为未来 sequence 的 prefix
cache 命中项；当前请求只是不能把 miss 之后的 block 视为已经计算完成。

### Decode 扩展与 speculative rollback

decode 时，`may_append()` 在 sequence 首次写入新 block 时分配 physical page；当一个尾 page
填满时，计算它相对于前一完整 prefix 的 chained hash 并登记到 prefix cache。

EAGLE3 speculative decode 则用 `ensure_num_blocks_for_length()` 预留 proposal 可能写入的所有
page。target verification 后，`truncate()` 释放被拒绝 proposal 对应的尾部 page，使 block
table 与最终提交的 token 长度重新一致。

## Sequence

`Sequence` 是一个轻量的请求状态结构，不包含复杂算法；它将 token、调度进度和逻辑 KV
地址放在一起，供 `Scheduler`、`BlockManager` 和 `ModelRunner` 使用。

关键状态包括：

```text
token_ids                 prompt 与已生成 token
status                    WAITING / RUNNING / FINISHED
block_table               logical block -> physical KV page
num_computed_tokens       已完成 forward 且已写入 KV 的 token 数
num_scheduled_tokens      当前 prefill step 临时调度的 token 数
num_cached_tokens         prefix cache 连续命中的可跳过 token 数
```

需要保持的核心不变量是 `num_tokens == len(token_ids)`。此外，`num_computed_tokens` 可以小于
`num_tokens`：通常最新生成的 token 已追加到 `token_ids`，但要到下一次 decode forward 才会
写入 KV cache。`append_token()`、`truncate_tokens()` 负责同步维护这些派生状态；不应直接
修改 `token_ids`。

## Tensor Parallel Linear、MLP 与 Attention

这一节按矩阵乘法理解 Tensor Parallel（TP）。PyTorch `F.linear(x, weight)` 的 weight
存储形状为 `[out_features, in_features]`，但对于按行组织的 activation，它实际计算：

```text
y = x @ W^T

x: [tokens, in_features]
W: [out_features, in_features]      # PyTorch storage
W^T: [in_features, out_features]    # 数学乘法中的右侧矩阵
y: [tokens, out_features]
```

因此下面的 Column/Row 指的是数学矩阵 `W^T` 的切分方向；从 PyTorch `weight` 的存储视角
看，方向正好相反。

### 两种 TP Linear

**ColumnParallelLinear** 将 `W^T` 按输出维度（列）切分；等价于将 PyTorch `weight` 按
`dim=0`、即输出维度切分：

```text
W^T = [W_0^T | W_1^T | ... | W_(p-1)^T]
y_r = x @ W_r^T

W_r: [out_features / p, in_features]
y_r: [tokens, out_features / p]
```

每个 rank 的 `y_r` 是完整输出向量中正确的一段 channel shard，不是需要求和的部分贡献。
若要恢复完整 `y`，理论上需要 `concat(y_0, ..., y_(p-1))`；但 Transformer 中后续算子也可
按同一维度并行，因此通常不立即 all-gather。

**RowParallelLinear** 将 `W^T` 按输入维度（行）切分；等价于将 PyTorch `weight` 按
`dim=1`、即输入维度切分：

```text
x = [x_0 | x_1 | ... | x_(p-1)]
W^T = [W_0^T; W_1^T; ...; W_(p-1)^T]
partial_y_r = x_r @ W_r^T
y = sum_r(partial_y_r)

x_r:         [tokens, in_features / p]
W_r:         [out_features, in_features / p]
partial_y_r: [tokens, out_features]
```

所以 RowParallelLinear 必须在各 TP rank 上执行 `all_reduce(SUM)`。它假设输入 `x_r` 已由
上游 ColumnParallelLinear 或 local attention 产生，不负责自己切分输入。Row parallel 的 bias
只由 TP rank 0 加入，再参与 all-reduce，避免 bias 被重复相加。

### MLP 的 TP 切分

Qwen3 MLP 的抽象计算为：

```text
gate = x @ W_gate^T
up   = x @ W_up^T
z    = SiLU(gate) ⊙ up
y    = z @ W_down^T
```

其中 `x, y` 的 hidden dimension 为 `H`，`gate, up, z` 的 intermediate dimension 为 `I`。
当前实现的并行链路是：

```text
x [H]，每个 rank 都持有完整 x
  -> MergedColumnParallelLinear(gate_up_proj)
     W_gate / W_up 沿 I 切分
  -> gate_r, up_r [I / p]
  -> local SiLU(gate_r) * up_r
  -> z_r [I / p]
  -> RowParallelLinear(down_proj)
     W_down 沿输入 I 切分
  -> all_reduce(SUM)
  -> y [H]，每个 rank 都重新得到完整输出
```

`MergedColumnParallelLinear` 只是把 gate 与 up 的两次 ColumnParallelLinear 合并为一次
投影和加载布局；TP 原理不变。

### Attention 的 TP 切分

Attention 的抽象计算为：

```text
Q = x @ W_Q^T
K = x @ W_K^T
V = x @ W_V^T
A = Attention(Q, K, V)
y = Concat(A_1, ..., A_num_heads) @ W_O^T
```

当前实现按 attention head 切分：

```text
x [H]，每个 rank 都持有完整 x
  -> QKVParallelLinear
     Q/K/V projection 沿输出维度切分
     每个 rank 持有 num_heads / p 个 Q head，和 num_kv_heads / p 个 KV head
  -> local paged attention
     每个 rank 只读写本地 head 对应的 KV cache shard
  -> local attention output [num_heads / p, head_dim]
  -> RowParallelLinear(o_proj)
     W_O 沿其输入 head-concatenation 维度切分
  -> all_reduce(SUM)
  -> y [H]，每个 rank 都重新得到完整 attention output
```

因此，MLP 的 intermediate feature shard 与 Attention 的 head shard 都在本地完成大部分
计算；每个 Transformer 子层的末尾通过 RowParallelLinear 的 all-reduce 回到复制的 hidden
state，便于 residual connection 和下一子层继续执行。

### Activation 与 RoPE

`SiLU` 等 activation 是逐元素计算；MLP 中的 `SiLU(gate_r) * up_r` 也只使用同一 rank 的
local intermediate shard，因此不需要 TP 通信。

RoPE 不是每个标量完全独立：它按同一 attention head 内的两个通道组成的 pair 旋转，并依赖
token position。但当前 TP 按完整 head 切分，每个 rank 都持有本地 head 的完整 `head_dim`，
所以 RoPE 同样可以本地执行，无需 TP collective。若未来实现沿 `head_dim`
切分，则必须保证每个旋转 pair 不会跨 rank，或额外处理跨 rank 数据。

## Attention 与 Paged KV Cache

`Attention` 负责将当前 forward 产生的 K/V 写入 layer-local paged cache，并用 Q 对历史 KV
执行 causal attention。Attention 数学没有因为 paged cache 改变；变化的是 K/V 的物理地址
不再按 sequence 连续排列。

### 两套地址 metadata

服务请求的 attention forward 先调用：

```text
store_kvcache(k, v, k_cache, v_cache, slot_mapping, ...)
```

`slot_mapping` 描述本轮每个输入 token 的写入地址：

```text
token position
  -> slot_mapping
  -> physical page + offset
  -> paged K/V write
```

读取历史 K/V 时，attention backend 使用 `block_table`：

```text
logical token position
  -> logical block index = position // block_size
  -> physical page = block_table[sequence, logical block index]
  -> paged K/V read at [physical page, position % block_size]
```

因此，paged attention 不只是多一个逻辑地址转换：写入由 `slot_mapping` 驱动，读取由
`block_table` 和 sequence length 驱动，二者都避免为每个 sequence 长期 materialize 连续 KV。

### BF16/FP16 cache 路径

若 attention layer 尚未绑定 runner 分配的 KV cache，例如独立模型 forward，则直接调用普通
连续 K/V 的 `flash_attn_func(q, k, v, causal=True)`。

正常 serving 路径下，当前 cache 是 BF16/FP16 时：

```text
store_kvcache
  -> prefill / eager decode: flash_attn_varlen_func(..., block_table=...)
  -> CUDA Graph decode: flash_attn_with_kvcache(..., block_table=...)
```

FlashAttention 在 kernel 内通过 `block_table` 间接读取物理 KV page，不需要先将该 sequence 的
完整历史 cache gather 成连续 tensor。`cache_seqlens` 或 varlen 的 `cu_seqlens_k` 限定每个
sequence 实际可读取的 token 范围。

### FP8 KV cache

FP8 cache 使用 `torch.float8_e4m3fn` 保存 K/V。写入时每个元素按 layer 的 K/V scale 缩放后
cast/store；读取时 attention 计算补回对应 scale。当前实现的 scale 是每个 attention layer
各一个 K 标量与 V 标量，不是 per-token 或 per-head scale。

Hopper（SM90+）上的生产 backend 可使用支持 FP8 KV scale 的 FlashAttention 3 路径；当前项目
面向的 FlashAttention 2 / SM89 环境无法通过公开 FA2 API 直接将 scaled FP8 paged KV 传入
attention，因此采用分阶段 fallback。

**FP8 decode** 在 `max_seqlen_q == 1` 时调用自定义
`fp8_paged_decode_attention()`：

```text
store quantized K/V
  -> Triton FP8 paged decode attention
  -> direct paged read + scale compensation + online softmax
```

该 Triton kernel 的 grid 为：

```text
(num_query_token_rows, num_query_heads)
```

decode 中第一维通常对应当前调度的 sequence 数。一个 Triton program 处理一个
`(sequence/query-token row, query head)` 输出，在内部按 `BLOCK_N=64` 分块遍历上下文；可近似
理解为一个 program 对应一个 CTA，但具体 CUDA CTA/warp 映射由 Triton 编译与 launch 配置决定。

**FP8 prefill** 不使用自定义 Triton attention：

```text
store quantized K/V
  -> Triton gather_dequant_kvcache（按 block_table 读取并临时反量化）
  -> 连续 BF16/FP16 K/V
  -> flash_attn_varlen_func
```

因此当前项目的 FP8 decode 可以直接从 paged FP8 cache 计算 attention，而 FP8 prefill 是
正确性优先的临时 materialization fallback。更完整的生产 FP8 paged-attention 路径需要 GPU、
FlashAttention backend 和 scale 格式共同支持。

## Vocab Parallel Embedding 与 LM Head

词表 embedding 的权重 `E` 形状为 `[V, H]`，其中 `V` 是 vocab size，`H` 是 hidden size。它的
显存可能很大，但计算只是按 token id 取对应的一行。`VocabParallelEmbedding` 因此沿 vocab 行维度
切分，而不是沿 hidden dimension 切分：TP world size 为 `p` 时，rank `r` 仅保存
`E_r = E[r * V/p : (r + 1) * V/p]`，权重显存降低到约原来的 `1/p`。

```text
token id t
  -> 若 t 属于本 rank 的 vocab range：查 E_r[t - vocab_start]
  -> 否则：lookup 后以 mask 置为 0
  -> all_reduce(SUM)
  -> E[t] [H]，每个 rank 都得到完整 token embedding
```

非本地 token 在当前实现中会先被映射为本地索引 0 以满足 `F.embedding` 的索引范围，再由 mask 清零，
所以不会将错误 embedding 加入最终结果。对于任意 token，恰好一个 rank 保留非零结果，故
`all_reduce(SUM)` 正好恢复原始 lookup 值。

这确实缓解了单卡超大 embedding tensor 的权重显存压力；但输出 activation `[num_tokens, H]` 仍会
在每个 rank 完整存在，并且每次 embedding forward 都有一次 TP all-reduce。因此它用通信换取权重
显存，通常适合词表很大或单卡显存紧张的模型。

`ParallelLMHead` 同样沿 vocab 行切分其权重，但数学方向不同：

```text
local_logits_r = hidden_states @ E_r^T       # [tokens, V / p]
logits = Concat(local_logits_0, ..., local_logits_(p-1))
```

这里每个 rank 产生的是不同 vocab 区间的 logits，不能相加；当前实现使用 `dist.gather` 将所有局部
logits 收到 TP rank 0，再沿最后一维拼接为完整 `[tokens, V]` logits。对于 prefill，`ParallelLMHead`
只选择各 sequence 的最后一个 token，以避免为不会采样的中间 token materialize 全词表 logits。权重
tied 时，`lm_head.weight` 与 `embed_tokens.weight` 复用同一份本地 shard，因此不会额外复制 embedding
权重。

## MoE：路由、Expert-major 打包与 Grouped GEMM

`ExpertParallelMoE` 的输入先展平为 `hidden_states [S, H]`。router 是 replicated 的线性层：

```text
router_logits = X @ W_router^T                       # [S, E]
routing_weights = softmax(router_logits)             # [S, E]
routing_weights, selected_experts = topk(..., K)     # 均为 [S, K]
```

`selected_experts[s, k]` 表示 token `s` 的第 `k` 个 route 所选 expert，`routing_weights[s, k]`
是该 route 的门控权重。`norm_topk_prob=True` 时，选出的 K 个权重还会重新归一化。随后
`_forward_triton` 将这两个 `[S, K]` tensor 展平为 token-major 的 `[S*K]` route 列表。

### EP local route 筛选

EP rank `r` 只持有连续 expert 范围 `[expert_start, expert_end)` 的 `gate_up_proj` 和 `down_proj`。
`local_mask` 从所有 top-k route 中筛选应由该 rank 执行的 assignment，得到：

```text
local_experts: 本地 expert id [T]
local_tokens:  原 token id [T]
local_weights: 对应 routing weight [T]
```

同一 token 可以选择多个 expert，所以 `local_tokens` 可以为 `[0, 0, 1, ...]`。reference 路径在
Python 中逐 expert 循环，但每个 expert 内部仍是一个 batch GEMM；它的主要问题是多个小 GEMM 和
kernel launch 开销，而非逐 token 计算。

### Expert-major、BLOCK_M 对齐的 packed layout

`torch.argsort(local_experts, stable=True)` 将 assignment 排成 expert-major 顺序，使同一 expert 的
token 聚集。令 `count_e` 是 local expert `e` 的 route 数，当前实现以 `BLOCK_M = 16` 对齐：

```text
padded_count_e = ceil(count_e / BLOCK_M) * BLOCK_M
expert_starts[e]: 未填充的 expert-major route 数组中，expert e 的起点
packed_starts[e]: 对齐后 packed 数组中，expert e 的起点
```

`within_expert` 给出一个有效 route 在所属 expert 内的偏移，
`packed_position = packed_starts[local_expert] + within_expert` 给出其最终 packed 行号。
`row_ids[packed_position]` 写入原始 token id，`packed_weights[packed_position]` 写入该 route 的
gate weight。未写入的 padding 行将 `row_ids` 设为 sentinel `S`，权重为 0。

`expert_ids` 的长度是 `sum_e padded_count_e / BLOCK_M`；每项对应一个 `BLOCK_M` 行 tile 所属的
local expert，而不是每个 token 一个 expert id。kernel 以 `row_ids < S` 屏蔽 sentinel 行，因此
padding 不会从 `hidden_states` 读入有效数据，也不会写入最终结果。

### 两次 Grouped GEMM 与 route 聚合

每个 local route 的 SwiGLU expert 计算为：

```text
gate_up = X[row_id] @ W_gate_up[expert_id]^T
gate, up = Split(gate_up)
activated = SiLU(gate) * up
route_output = activated @ W_down[expert_id]^T * routing_weight
output[token_id] += route_output
```

`_run_grouped_gemm` 共调用两次：第一次从原始 `hidden_states` 通过 packed `row_ids` 读取 token；
第二次的输入已经是 packed `activated`，因此传入连续的 `packed_row_ids = [0, 1, ...]`。routing
weight 融合在第二次 GEMM 的输出乘法中。最后 `index_add_` 对相同原始 token id 的 K 个 route 输出
求和。

### Triton kernel 与当前 EP 模型

`_grouped_moe_gemm_kernel` 的计算核心仍是 tiled GEMM：

```text
A[row_ids, :] @ B[expert_ids, :, :]^T
```

每个 program 处理一个 `(BLOCK_M rows, BLOCK_N output columns)` tile。它比普通 GEMM 多出的关键
索引是：由 `row_ids` 间接读取 A 的实际 token 行，由 `expert_ids` 选择 B 的 expert 权重；随后在
K 维按 `BLOCK_K` 循环执行 `tl.dot` 累加。第二次调用可选择在 store 前乘 `routing_weights`。

这里的 EP 是教学用的本地 expert shard 模型：每个 rank 仍持有完整 `hidden_states`，仅计算属于本地
expert 的 routes，最后通过 `all_reduce(SUM)` 汇总局部输出。它没有生产 MoE 常用的 token
`all_to_all` dispatch/return；后者能避免在各 rank 复制全部 token activation，但需要更复杂的负载
均衡与通信处理。

## 并行实现现状与 vLLM 对照

当前项目的并行配置入口是 `nanovllm/config.py`，进程启动和模型执行入口是
`nanovllm/engine/llm_engine.py`、`nanovllm/engine/model_runner.py`，通信组定义集中在
`nanovllm/distributed/parallel_state.py`。项目支持 DP、TP、PP 和 EP，但实现目标是便于学习，
各并行方式的功能范围和生产级 vLLM 有明显差异。

### 当前项目的功能

**TP（Tensor Parallel）**

- `nanovllm/layers/linear.py` 实现 `ColumnParallelLinear`、`RowParallelLinear` 和
  `QKVParallelLinear`。
- Column parallel 沿输出维度切分，Row parallel 沿输入维度切分并执行 TP `all_reduce`。
- `nanovllm/layers/embed_head.py` 实现 vocab-parallel embedding 和 LM head。
- Qwen3 的 attention 按 Q/KV head 切分，MLP 按 intermediate dimension 切分。
- 主要限制是 hidden size、attention heads、KV heads 等维度需要满足 TP 整除条件，通信也主要是
  直接的 PyTorch collective，没有复杂的异步通信或通信后端选择。

**PP（Pipeline Parallel）**

- `nanovllm/models/qwen3.py` 和 `nanovllm/models/qwen3_moe.py` 按 decoder layer 平均划分 stage。
- 首 stage 负责 embedding，末 stage 负责 final norm、logits 和 sampling。
- `nanovllm/engine/model_runner.py::run_pipeline_model` 使用 `dist.send`/`dist.recv` 传递
  hidden states 和 residual。
- 当前 PP 要求 `enforce_eager=True`，是基础的同步 stage 传递，没有实现完整的 microbatch 1F1B、
  pipeline bubble 优化或通信计算 overlap。

**EP（Expert Parallel）**

- `nanovllm/distributed/parallel_state.py` 当前直接复用 TP group 作为 EP group。
- `nanovllm/layers/moe.py` 每个 rank 保存一部分 experts，筛选本地 routes 后使用 Triton grouped
  GEMM 计算。
- 所有 rank 仍保留完整 hidden states，MoE 输出最后使用 EP `all_reduce(SUM)` 汇总。
- 这是本地 expert shard 的教学实现，没有 token `all_to_all` dispatch/return；因此会复制 activation，
  也无法像生产 MoE 那样只向目标 expert 所在 rank 发送 token。
- `nanovllm/config.py` 和 `nanovllm/models/qwen3_moe.py` 当前禁止 EP 与 PP 组合；EP 也要求 eager
  模式，Qwen3-MoE 的 expert 数量需要能够均匀分配。

**DP（Data Parallel）**

- `nanovllm/engine/data_parallel.py::generate_data_parallel` 启动多个独立 LLM replica，并将
  prompts 分配给不同 replica。
- 每个 replica 有独立的 scheduler、KV cache 和模型并行进程组；replica 之间不共享请求调度状态。
- 当前 DP 更接近离线 batch 的多副本吞吐扩展，没有统一的 DP coordinator、在线负载均衡、DP rank
  间 batch padding 同步或 MoE DP/EP 联动。

### vLLM 如何解决这些限制

以下源码位置基于本地 vLLM commit `1a308c449`。

**统一并行拓扑**

- `vllm/config/parallel.py::ParallelConfig` 统一管理 TP、PP、DP、EP、PCP 等参数。
- `vllm/distributed/parallel_state.py::initialize_model_parallel` 按
  `ExternalDP x DP x PP x PCP x TP` 构造通信组。
- vLLM 分别建立 TP、PP、DP、EP、prefill context parallel 和 decode context parallel group，
  不把 EP 简单等同于 TP。
- EP group 通常覆盖同一 PP stage 内的 DP/PCP/TP rank，使 TP、PP、EP 可以组合使用。

**TP**

- `vllm/model_executor/layers/linear.py` 和
  `vllm/model_executor/layers/vocab_parallel_embedding.py` 提供切分线性层和 vocab-parallel
  embedding。
- `vllm/distributed/communication_op.py` 对 all-reduce、all-gather、reduce-scatter 等操作做统一
  封装，并可使用 NCCL、custom all-reduce、CUDA communicator 等后端。
- 因此数学上的 Column/Row parallel 与当前项目类似，但 vLLM 增加了异步通信、内存复用和硬件相关
  collective 优化。

**PP**

- 模型通过 `get_pp_group().is_first_rank` 和 `is_last_rank` 判断 stage 边界，例如
  `vllm/model_executor/models/qwen3.py` 和 `llama.py`。
- `vllm/v1/worker/gpu_worker.py` 使用 `irecv_tensor_dict` 和 `isend_tensor_dict` 异步传递
  intermediate tensors，并可与 TP group 的 tensor gather 配合。
- vLLM 的 worker、runner 和 scheduler 共同处理 microbatch、CUDA Graph、异步发送和接收，因此 PP
  不局限于 nano-vLLM 的同步 eager stage forwarding。

**EP 与 MoE dispatch**

- `vllm/model_executor/layers/fused_moe/layer.py` 构造 router、expert 参数、expert placement
  和 fused MoE runner。
- `vllm/model_executor/layers/fused_moe/all2all_utils.py` 选择 MoE 的 prepare/finalize 路径。
- `vllm/model_executor/layers/fused_moe/prepare_finalize/` 负责 token dispatch 和 combine；
  `naive_dp_ep.py` 提供较简单的 AllGather/ReduceScatter 路径。
- `vllm/distributed/device_communicators/all2all.py` 提供 all-to-all manager。
- 更高性能的路径位于 `deepep_ht.py`、`deepep_ll.py`、`deepep_v2.py`，可使用 DeepEP 的 high
  throughput/low latency kernel；此外还支持 FlashInfer、MoRI、NIXL 等后端。

完整的 vLLM MoE 流程是：

```text
router top-k
  -> prepare/permute
  -> token all-to-all dispatch
  -> local expert grouped GEMM
  -> all-to-all combine/unpermute
  -> routing-weight reduction
```

这与当前项目的：

```text
local route filter
  -> local grouped GEMM
  -> EP all_reduce
```

不同。前者减少了每个 rank 保存和计算的 activation，代价是需要更复杂的 token 重排、通信 buffer、
expert placement、负载均衡和 CUDA Graph 兼容处理。

**DP**

- `vllm/v1/engine/coordinator.py` 管理多个 DP engine。
- `vllm/v1/engine/core_client.py` 提供 `DPAsyncMPClient` 和 `DPLBAsyncMPClient`，支持内部或外部
  load balancing。
- `vllm/v1/worker/dp_utils.py::coordinate_batch_across_dp` 使用 DP group 的 collective 同步
  各 rank 的 token 数、microbatch 选择和 CUDA Graph mode；必要时将各 rank padding 到相同 token 数。
- 因此 vLLM 保留了 DP replica 的独立 scheduler，但增加了请求路由、统一执行条件和 MoE 场景所需的
  DP/EP 协作，而不是只在 Python 层轮询 prompts。

总体而言，nano-vLLM 当前已经具备学习并行 Transformer 的基本骨架；最大的生产差距集中在 PP 的
异步调度、DP 的统一请求协调，以及 EP 的 token all-to-all dispatch/combine，而不是基础的矩阵
切分公式。

## Benchmark 结果

### vLLM：FP8 与 BF16

测试使用相同的模型和 token 输入：16 个请求，每个 prompt 512 tokens，每个请求生成 64
tokens，共生成 1024 tokens。vLLM 从同一个 BF16 checkpoint 启用 `quantization="fp8"`，并使用
`enforce_eager=True`。

| 模型 | BF16 | FP8 | FP8/BF16 |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B | 1266.95 tok/s | 1083.16 tok/s | 0.85x |
| Qwen3-8B | 397.63 tok/s | 656.44 tok/s | 1.65x |

Qwen3-8B 上 FP8 加速约 65.1%；Qwen3-0.6B 由于矩阵规模较小，FP8 kernel 的额外开销超过
收益，反而慢约 14.5%。vLLM 日志显示 Qwen3-8B 选择了
`CutlassFP8ScaledMMLinearKernel`。该测试是离线 batch 端到端吞吐测试，与本项目按 prefill 和
decode 分开统计的 benchmark 不完全等价。

### Qwen3-8B：EAGLE3 与普通 decode

使用 `benchmarks/bench_spec_decode_eagle3.py`，nano-vLLM 单卡运行，prompt 为 1024 tokens，
完整生成 4096 tokens，batch size 为 1，每种模式运行 1 次。总生成时间按
`TTFT + TPOT × (4096 - 1)` 计算。

| 模式 | TTFT | TPOT | 总生成时间 | 吞吐 |
| --- | ---: | ---: | ---: | ---: |
| 不使用 EAGLE3 | 98.15 ms | 18.959 ms/token | 77.737 s | 52.69 tok/s |
| 使用 EAGLE3 | 121.07 ms | 4.859 ms/token | 20.019 s | 204.61 tok/s |

EAGLE3 的接受率为 98.54%，总生成加速约 **3.88x**，decode TPOT 加速约 **3.90x**，总耗时
降低约 74.25%。EAGLE3 的 TTFT 略高，但在 4096-token 长 decode 中影响很小。

该 benchmark 新增了 `--disable-eagle`、`--prompt-tokens`、`--output-tokens`、
`--num-runs` 和 `--result-file` 参数，因此可以在相同输入条件下分别测量普通 decode 与
EAGLE3 decode。
