# scheduler.py - 请求调度模块

## 文件作用

该模块实现了 **LLM推理的核心调度逻辑**，负责：
1. 管理请求的**生命周期**（WAITING → RUNNING → FINISHED）
2. 实现**两阶段调度**（Prefill + Decode）
3. 处理GPU资源竞争和**抢占机制**
4. 与BlockManager协作管理KV-Cache内存

---

## 核心概念

### 请求队列状态

```
waiting: 等待调度的请求队列
    ↓
running: 正在推理的请求队列
    ↓
finished: 已完成的请求（被移除）
```

### 两阶段推理

| 阶段 | 说明 | 特点 |
|------|------|------|
| **Prefill** | 处理整个提示词 | 计算密集，一次性处理完整序列 |
| **Decode** | 逐token生成 | 内存密集，每步生成1个token |

---

## Scheduler 类

### 初始化

```python
def __init__(self, config: Config):
    self.max_num_seqs = 4              # 最大并发序列数
    self.max_num_batched_tokens = 512  # 最大批处理token数
    self.eos = config.eos              # EOS token ID
    self.block_manager = BlockManager(...)  # KV-Cache管理器
    self.waiting = deque()             # 等待队列
    self.running = deque()             # 运行队列
```

---

## 核心方法

### 1. add - 添加请求

```python
def add(self, seq: Sequence):
    """将新请求加入等待队列"""
    self.waiting.append(seq)
```

**状态转换**：
```
新请求 → waiting队列 → 等待scheduler.schedule()调度
```

---

### 2. schedule - 主调度逻辑

**整体结构**：
```python
def schedule() -> (list[Sequence], bool):
    """
    两阶段调度：
    
    返回值：
    - list[Sequence]: 本轮要执行的序列
    - bool: 是否为Prefill阶段（True=Prefill, False=Decode）
    """
```

---

#### 2.1 Prefill 阶段

```python
# prefill
scheduled_seqs = []
num_seqs = 0
num_batched_tokens = 0

while self.waiting and num_seqs < self.max_num_seqs:
    seq = self.waiting[0]  # 查看队首（FIFO）
    
    # 约束检查
    if num_batched_tokens + len(seq) > self.max_num_batched_tokens:
        break  # Token数超限
    if not self.block_manager.can_allocate(seq):
        break  # KV-Cache不足
    
    # 执行分配
    num_seqs += 1
    self.block_manager.allocate(seq)
    num_batched_tokens += len(seq) - seq.num_cached_tokens
    seq.status = SequenceStatus.RUNNING
    self.waiting.popleft()
    self.running.append(seq)
    scheduled_seqs.append(seq)

if scheduled_seqs:
    return scheduled_seqs, True  # 返回True表示Prefill
```

**Prefill约束**：
| 约束 | 限制 | 说明 |
|------|------|------|
| 并发数 | `num_seqs < max_num_seqs` | 最多同时处理4个序列 |
| Token数 | `num_batched_tokens < max_num_batched_tokens` | 最多512个token |
| 内存 | `block_manager.can_allocate()` | KV-Cache足够 |

**工作流程**：
```
waiting: [A(1000), B(800), C(600), ...]
                    ↓ 每次迭代

iter1: A(1000 tokens)
  ├─ token约束: 0 + 1000 < 512? NO → break
  
iter2: 换一个时间步...
  ├─ A(512 tokens) - [cached: 488]
  ├─ B(512 tokens)
  ├─ token约束: 24 + 512 < 512? NO → break
  
✓ scheduled_seqs = [A, B]  (Prefill)
```

---

#### 2.2 Decode 阶段

当Prefill为空时，执行Decode阶段。

```python
# decode
while self.running and num_seqs < self.max_num_seqs:
    seq = self.running.popleft()  # 取出运行中的序列
    
    while not self.block_manager.can_append(seq):
        # KV-Cache不足，需要抢占
        if self.running:
            self.preempt(self.running.pop())  # 抢占其他序列
        else:
            self.preempt(seq)  # 无法抢占，抢占自己
            break
    else:
        # KV-Cache充足，可以执行decode
        num_seqs += 1
        self.block_manager.may_append(seq)
        scheduled_seqs.append(seq)

self.running.extendleft(reversed(scheduled_seqs))
return scheduled_seqs, False  # 返回False表示Decode
```

**Decode 内存竞争处理**：

```
running: [A, B, C]
KV-Cache: 仅剩2个block

迭代1: 取A
  ├─ can_append(A)? NO (需要新block)
  ├─ running: [B, C]
  ├─ preempt(C) (抢占最后一个)
  │   ├─ C.status = WAITING
  │   ├─ 释放C的blocks
  │   └─ C回到waiting队列
  ├─ can_append(A)? YES (现在有block了)
  └─ scheduled_seqs = [A]

迭代2: 取B
  ├─ can_append(B)? YES
  └─ scheduled_seqs = [A, B]

返回: [A, B]  (Decode)
```

---

### 3. preempt - 抢占机制

```python
def preempt(self, seq: Sequence):
    """
    抢占一个序列，释放其资源重新等待
    
    流程：
    1. 状态改为WAITING
    2. 释放所有KV-Cache块
    3. 加入等待队列队首（优先级高）
    """
    seq.status = SequenceStatus.WAITING
    self.block_manager.deallocate(seq)
    self.waiting.appendleft(seq)  # 队首，优先被重新调度
```

**抢占策略**：
```
Decode阶段，GPU内存紧张时：

优先级：
高 ←── 调度顺序 ── 低
新请求(waiting) > Prefill中的请求 > Decode中的请求

抢占规则：
1. 优先抢占Decode中最后加入的请求（recency）
2. 被抢占的请求回到waiting队首
3. 确保新请求和Prefill请求优先级
```

---

### 4. postprocess - 后处理

```python
def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
    """
    处理本轮推理的结果
    
    流程：
    1. 追加生成的token
    2. 检查是否完成（EOS or max_tokens）
    3. 移除已完成序列
    """
    for seq, token_id in zip(seqs, token_ids):
        seq.append_token(token_id)
        
        # 完成条件
        if (not seq.ignore_eos and token_id == self.eos) or \
           seq.num_completion_tokens == seq.max_tokens:
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)
            self.running.remove(seq)
```

**完成条件**：
| 条件 | 说明 |
|------|------|
| 生成EOS token | 正常完成生成 |
| 达到max_tokens | 强制截断 |

---

## 完整工作流程

```
┌─────────────────────────────────────────────┐
│           新请求到达                         │
├─────────────────────────────────────────────┤
│  add(seq) → seq加入waiting队列               │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│        schedule() - Prefill阶段             │
├─────────────────────────────────────────────┤
│  从waiting队列选择序列                      │
│  ├─ 检查token数约束                        │
│  ├─ 检查内存约束                           │
│  └─ block_manager.allocate()               │
│  状态: waiting → running                    │
│  返回: (seqs, is_prefill=True)             │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│        ModelRunner.run() - 模型推理         │
├─────────────────────────────────────────────┤
│  处理整个提示词（Prefill）                  │
│  返回生成的token IDs                        │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│     postprocess() - 结果处理                 │
├─────────────────────────────────────────────┤
│  seq.append_token(token_id)                 │
│  检查是否完成                                │
│  是 → deallocate + 移除                      │
│  否 → 保留在running中                       │
└──────────────┬──────────────────────────────┘
               │
        ┌──────┴──────┐
        │             │
        ▼             ▼
   还有新请求    还有待处理请求
        │             │
        ▼             ▼
  Prefill阶段   Decode阶段（schedule）
                      │
              ├─────────┼─────────┐
              ▼         ▼         ▼
          can_append? 内存足? 抢占?
             |           |       |
           YES         YES      处理
             │           │
             └─────┬─────┘
                   ▼
           ModelRunner.run()
           （decode：生成1个token）
                   │
                   ▼
              postprocess()
                   │
          是否完成？
          ├─ 是 → 输出结果
          └─ 否 → 继续decode
```

---

## 调度策略分析

### Prefill优先级

```
采用FIFO策略，先到先执行
约束：
- 最大并发: max_num_seqs（通常4个）
- 最大token: max_num_batched_tokens（通常512）
- 内存充足: can_allocate()
```

### Decode优先级

```
采用FIFO + 抢占策略
优先级：
1. 继续decode的序列（已在running中）
2. 内存不足时，抢占最后进入的序列
3. 被抢占序列回到waiting队首，优先重新调度
```

---

## 性能特征

| 操作 | 复杂度 | 备注 |
|------|--------|------|
| add | O(1) | deque append |
| schedule | O(max_num_seqs * T) | T为token检查开销 |
| preempt | O(num_blocks) | deallocate遍历 |
| postprocess | O(num_seqs) | 遍历本轮序列 |

---

## 与其他模块的集成

```
LLMEngine
    │
    ├─ add_request() → Scheduler.add()
    ├─ step()
    │   ├─ schedule() ─→ Scheduler
    │   ├─ run() ──────→ ModelRunner
    │   └─ postprocess() ─→ Scheduler
    │
BlockManager ← Scheduler
    ├─ can_allocate()
    ├─ allocate()
    ├─ can_append()
    ├─ may_append()
    └─ deallocate()
```
