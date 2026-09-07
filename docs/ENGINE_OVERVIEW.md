# nano-vllm Engine 模块总览

## 项目简介

**nano-vllm** 是 vLLM 推理框架的轻量级实现，提供了一个完整的 LLM 推理引擎。整个引擎采用**模块化设计**，各部分职责明确，易于理解和扩展。

---

## 文件结构

```
nanovllm/engine/
├── sequence.py          # 请求序列抽象
├── block_manager.py     # KV-Cache内存管理
├── scheduler.py         # 请求调度引擎
├── model_runner.py      # 模型推理执行
└── llm_engine.py        # 高层用户接口
```

---

## 模块关系图

```
┌─────────────────────────────────────────┐
│          LLMEngine (user API)            │
│         生成接口 generate()               │
└──────────────┬──────────────────────────┘
               │
        ┌──────┴──────┐
        │             │
        ▼             ▼
┌──────────────┐  ┌──────────────────┐
│ Scheduler    │  │ ModelRunner      │
│ (调度逻辑)   │  │ (推理执行)        │
└──────────────┘  └──────────────────┘
        │                  │
        └────────┬─────────┘
                 │
        ┌────────▼──────────┐
        │  BlockManager      │
        │  (内存管理)        │
        └────────────────────┘
        
        ┌────────────────────┐
        │   Sequence         │
        │  (请求抽象)        │
        └────────────────────┘
```

---

## 各模块详细说明

### 1. Sequence（序列管理）

**文件**: `sequence.py`

**核心职责**：
- 抽象单个推理请求为 `Sequence` 对象
- 追踪序列的生命周期状态（WAITING → RUNNING → FINISHED）
- 管理序列的 token IDs 和 KV-Cache 映射

**关键数据**：
```python
Sequence
  ├─ seq_id: 唯一ID
  ├─ status: WAITING/RUNNING/FINISHED
  ├─ token_ids: 积累的所有token
  ├─ num_cached_tokens: 已缓存的token数
  ├─ block_table: KV-Cache块ID映射
  ├─ temperature: 采样温度
  └─ max_tokens: 最大生成长度
```

**示例**：
```
用户输入: "What is AI?"
         ↓ tokenize
    Sequence([101, 2054, 2003, ...])
         ↓ prefill
    KV-Cache分配，block_table=[0,1,2]
         ↓ decode (逐token)
    append_token(2222) → token_ids=[..., 2222]
    append_token(4605) → token_ids=[..., 2222, 4605]
         ↓
    is_finished → FINISHED → deallocate
```

---

### 2. BlockManager（内存管理）

**文件**: `block_manager.py`

**核心职责**：
- 管理 KV-Cache 的分配和释放
- 实现块重用机制，支持多序列共享
- 提供哈希缓存快速查找相同块

**关键机制**：

#### a) 块分割
```
序列token: [101, 2054, 2003, 4605, 3010, ...]
                              ↓
          Block 0        Block 1       Block 2
      [256 tokens]   [256 tokens]  [remaining]
```

#### b) 块重用（共享）
```
请求1: [开始] [中文] [生成] [很] [快]
请求2: [开始] [中文] [性能] [很] [好]

Block 0: [开始] [中文]  ← ref_count=2 (两个请求共享)
Block 1a: [生成] ...    ← ref_count=1 (请求1独占)
Block 1b: [性能] ...    ← ref_count=1 (请求2独占)

优势：节省 ~256 tokens 的显存
```

#### c) 引用计数
```
Block被n个序列使用 → ref_count = n
序列1完成   → ref_count -= 1
序列2完成   → ref_count -= 1
ref_count = 0 → 块回收，加入free_block_ids
```

**关键方法**：
| 方法 | 时机 | 作用 |
|------|------|------|
| `allocate()` | Prefill | 分配块（支持共享） |
| `can_append()` | Decode | 检查是否可追加 |
| `may_append()` | Decode | 追加一个token |
| `deallocate()` | 完成 | 释放所有块 |

---

### 3. Scheduler（调度引擎）

**文件**: `scheduler.py`

**核心职责**：
- 实现两阶段推理调度（Prefill + Decode）
- 管理请求的队列和状态转换
- 处理 GPU 资源竞争和抢占

**关键流程**：

#### Prefill 阶段
```
约束：
├─ 最大并发序列: max_num_seqs (通常4个)
├─ 最大token数: max_num_batched_tokens (通常512个)
└─ KV-Cache足够: block_manager.can_allocate()

流程：
waiting: [A, B, C, D, ...]
    ↓ schedule()
    ├─ A: 512 tokens + 512 > limit → skip
    ├─ B: 256 tokens → OK → allocate + running
    ├─ C: 128 tokens + 256 > limit → skip
    │
    └─ scheduled_seqs = [B], is_prefill = True
```

#### Decode 阶段
```
内存竞争处理：

running: [A, B, C]  (已有KV-Cache)
new_blocks: 仅剩1个

流程：
    ├─ 取A → can_append(A)? NO
    │   ├─ preempt(C) → 释放C的块
    │   └─ can_append(A)? YES → scheduled
    │
    ├─ 取B → can_append(B)? YES → scheduled
    │
    └─ scheduled_seqs = [A, B], is_prefill = False

状态转换：
A, B: running → decode → running
C: running → preempt → waiting (高优先级重新调度)
```

**抢占策略**：
```
目标: 确保高优先级请求获得GPU资源

优先级顺序：
新请求(waiting) > Prefill中的请求 > Decode中的请求

实现:
├─ Decode时，if 内存不足
├─ 抢占 running.pop() (最后进入的)
└─ 重新加入 waiting.appendleft() (队首，最高优先级)
```

---

### 4. ModelRunner（推理执行）

**文件**: `model_runner.py`

**核心职责**：
- 加载和初始化 LLM 模型
- 自动计算和分配 KV-Cache 内存
- 执行 Prefill 和 Decode 推理
- 支持多卡张量并行（Tensor Parallelism）
- CUDA 图捕获优化

**关键流程**：

#### Prefill
```
输入：多个序列的完整提示词
处理：
  ├─ 合并所有序列的token
  ├─ 计算注意力（所有token相互交互）
  └─ 生成完整序列的KV-Cache

特点：计算密集，一次性处理完整序列
```

#### Decode
```
输入：每个序列的最后一个token
处理：
  ├─ 对每个序列的最后token计算query
  ├─ 与缓存的KV交互
  └─ 生成1个新token

特点：内存密集，逐token生成
```

#### CUDA 图优化
```
Prefill: 直接执行 (batch_size变化)
Decode:  使用预捕获的CUDA图 (加速)

预捕获batch_size: [1, 2, 4, 8, 16, ..., max]

性能提升：~20-30% decode吞吐量
```

#### 多卡推理
```
rank=0 (主进程)          rank>0 (从进程)
    │                        │
    ├─ write_shm()          └─ loop()
    │   (任务)                  │
    ├─ run()               read_shm()
    │   (执行)                  │
    └─ 同步                execute call()
                               │
    通过NCCL保证同步
```

---

### 5. LLMEngine（用户接口）

**文件**: `llm_engine.py`

**核心职责**：
- 提供简洁的用户 API (`generate()`)
- 编排整个推理流程
- 管理多进程生命周期
- 统计推理吞吐量

**关键接口**：
```python
# 初始化
llm = LLM("model_path", tensor_parallel_size=1)

# 添加请求
llm.add_request(prompt, sampling_params)

# 生成结果
results = llm.generate(
    prompts=["What is AI?"],
    sampling_params=SamplingParams(max_tokens=100)
)
# 返回: [{"text": "...", "token_ids": [...]}]
```

---

## 完整推理流程（端到端）

```
┌──────────────────────────────────────────────────┐
│  用户调用: llm.generate([prompts], sampling_params)  │
└────────────────┬─────────────────────────────────┘
                 │
        ┌────────▼──────────┐
        │ add_request()     │
        │ for each prompt   │
        └────────┬──────────┘
                 │
      Prompt → Tokenize → Sequence
                 │
      ┌─────────▼──────────┐
      │  waiting queue     │
      └─────────┬──────────┘
                 │
    ┌────────────▼────────────┐
    │  推理循环: step()        │
    │                         │
    │  ┌────────────────────┐ │
    │  │ Scheduler.schedule │ │
    │  └────┬───────────────┘ │
    │       │                 │
    │   ┌───▼────────────────┐│
    │   │ Prefill 阶段       ││
    │   │ ├─ waiting→running ││
    │   │ ├─ allocate        ││
    │   │ └─ 处理完整提示词   ││
    │   └────┬───────────────┘│
    │        │                │
    │   ┌───▼────────────────┐│
    │   │ Decode 阶段        ││
    │   │ ├─ 逐token生成     ││
    │   │ ├─ may_append      ││
    │   │ └─ 内存竞争/抢占    ││
    │   └────┬───────────────┘│
    │        │                │
    │  ┌─────▼───────────────┐│
    │  │ ModelRunner.run()   ││
    │  │ ├─ prepare_prefill ││
    │  │ ├─ prepare_decode  ││
    │  │ ├─ forward()       ││
    │  │ └─ sampler()       ││
    │  └────┬────────────────┘│
    │       │                 │
    │  ┌────▼──────────────┐ │
    │  │ Scheduler.postprocess │
    │  │ ├─ append_token   │ │
    │  │ ├─ 检查完成       │ │
    │  │ └─ deallocate     │ │
    │  └────┬──────────────┘ │
    │       │                │
    │  is_finished? ────────┐ │
    │       │               │ │
    │     ✓ 否              │ │
    │       │               │ │
    │     继续循环           │ │
    │       └────────────────┘ │
    │                          │
    │     ✓ 是，推理完成        │
    └──────────┬──────────────┘
               │
        ┌──────▼──────────┐
        │ 结果整理        │
        │ ├─ 排序         │
        │ ├─ decode       │
        │ └─ 返回         │
        └──────┬──────────┘
               │
        返回给用户
```

---

## 数据结构关系

```
Request
  ↓ (tokenize)
token_ids
  ↓ (pack)
Sequence
  ├─ token_ids
  ├─ block_table (BlockManager管理)
  └─ status (Scheduler维护)

推理步骤：
1. Scheduler.schedule() → 选择要执行的Sequence列表
2. ModelRunner.run(sequences) → 执行推理
3. Scheduler.postprocess() → 更新Sequence状态
4. 重复直到is_finished()
```

---

## 性能关键指标

### 1. 吞吐量（Throughput）

```
Prefill: tokens/秒
  = sum(input_tokens_per_step) / elapsed_time

Decode: tokens/秒  
  = num_sequences / elapsed_time
  
例：
Prefill: 512 tokens / 0.5s = 1024 tok/s
Decode:  4 sequences / 0.1s = 40 tok/s
```

### 2. 延迟（Latency）

```
End-to-End延迟 = Prefill时间 + Decode时间

Prefill时间 = input_tokens / prefill_throughput
Decode时间 = max_tokens / decode_throughput
```

### 3. 显存利用（Memory Utilization）

```
GPU显存 = 模型参数 + 激活值 + KV-Cache

KV-Cache优化：
├─ 块分割 (固定大小)
├─ 块重用 (多序列共享)
└─ 块回收 (及时释放)

利用率 = 实际使用显存 / 总显存
```

---

## 核心优化技术总结

### 1. 块重用（Block Sharing）
```
多个请求的相同前缀共享KV-Cache
节省显存，特别是在批处理场景
```

### 2. CUDA图捕获（CUDA Graph）
```
Decode: 预录制计算图，重放执行
避免频繁的kernel launch和同步
```

### 3. 抢占调度（Preemption）
```
内存不足时，抢占低优先级请求
确保高优先级请求完成
```

### 4. 两阶段调度（Prefill + Decode）
```
分离计算密集和内存密集任务
充分利用GPU计算和内存带宽
```

### 5. 张量并行（Tensor Parallelism）
```
多卡分布式推理
支持大模型推理
```

---

## 代码阅读建议

**推荐阅读顺序**：

1. **sequence.py** - 理解基本数据抽象
2. **block_manager.py** - 掌握内存管理机制
3. **scheduler.py** - 理解调度策略
4. **model_runner.py** - 了解推理执行
5. **llm_engine.py** - 综合理解整个流程

**学习重点**：

```
└─ 数据流向
   ├─ token_ids → Sequence
   ├─ Sequence → BlockManager (allocate/deallocate)
   ├─ Scheduler.schedule() → 选择要执行的序列
   ├─ ModelRunner.run() → 执行推理
   └─ Scheduler.postprocess() → 更新状态

└─ 状态转换
   ├─ WAITING → RUNNING (allocate)
   ├─ RUNNING → FINISHED (postprocess)
   └─ RUNNING → WAITING (preempt)

└─ 内存管理
   ├─ free_block_ids (空闲块)
   ├─ used_block_ids (使用中)
   ├─ ref_count (引用计数)
   └─ hash_to_block_id (块共享查找)
```

---

## 常见问题

### Q1: 为什么要分Prefill和Decode两个阶段？
```
Prefill: 计算密集，大量token一起处理
         → 充分利用GPU并行度
         → 高吞吐量
         
Decode: 内存密集，每步生成1个token
        → 需要读取全部KV-Cache
        → 低延迟优先
        
分离两阶段可以分别优化
```

### Q2: 块重用是如何实现的？
```
通过哈希值快速识别相同块：
├─ compute_hash(token_ids, prefix)
├─ hash_to_block_id 字典映射
└─ 快速查找 O(1)

示例：
多个请求 [开始] [中国] [生成] ...
都通过 allocate() 时
哈希碰撞 → 发现块可共享
→ 只分配1个块，ref_count=2
```

### Q3: 抢占如何保证公平性？
```
优先级：
新请求(waiting) > Prefill > Decode最后进入的

当Decode内存不足时：
preempt(running.pop())  // 抢占最后进入的
waiting.appendleft()    // 加入队首，最高优先级

确保新请求最终能被调度
```

### Q4: 多卡如何协调？
```
SharedMemory + Event + NCCL

rank=0 (主):        rank>0 (从):
  ├─ write_shm()      ├─ loop()
  ├─ run()            ├─ read_shm()
  └─ set event()      └─ execute call()

NCCL collective ops 保证同步
```

---

## 扩展方向

1. **支持更多优化**：Flash-Attention、Paged Attention等
2. **动态批处理**：根据内存动态调整batch size
3. **请求优先级**：支持用户定义的优先级
4. **多模型推理**：同时加载多个模型
5. **长文本处理**：支持超长序列输入

---

## 参考

相关论文和项目：
- vLLM: Efficient Memory Management for Large Language Model Serving
- Flash-Attention: Fast and Memory-Efficient Exact Attention
- Tensor Parallelism: A Simple Method for Distributed Large Language Model Training
