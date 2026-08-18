# LLMEngine 推理循环详解

这段代码实现了 LLM 推理引擎的**核心执行逻辑**，包括单步推理 (`step`)、完成检查 (`is_finished`) 和完整生成流程 (`generate`)。

---

## 方法一：`step()` - 单步推理

```python
def step(self):
    seqs, is_prefill = self.scheduler.schedule()
    token_ids = self.model_runner.call("run", seqs, is_prefill)
    self.scheduler.postprocess(seqs, token_ids)
    outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
    num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
    return outputs, num_tokens
```

### 逐行解析

#### 第 1 行：调度

```python
seqs, is_prefill = self.scheduler.schedule()
```

**作用**：从调度器获取本轮要处理的序列

**返回值**：
| 变量 | 类型 | 含义 |
|------|------|------|
| `seqs` | `list[Sequence]` | 本轮要处理的序列列表 |
| `is_prefill` | `bool` | 是否是 Prefill 阶段 |

**详解**：
```
调度器决定：
  ├─ 如果有新请求在等待 → Prefill 阶段
  │   └─ 处理完整的提示词
  │
  └─ 如果没有新请求 → Decode 阶段
      └─ 为每个序列生成下一个 token
```

---

#### 第 2 行：模型推理

```python
token_ids = self.model_runner.call("run", seqs, is_prefill)
```

**作用**：调用模型执行推理，生成 token

**参数解释**：
```python
self.model_runner.call("run", seqs, is_prefill)
#                       ↑      ↑      ↑
#                   方法名   序列  是否Prefill

# 等价于在 ModelRunner 中调用：
# self.model_runner.run(seqs, is_prefill)
```

**返回值**：
```python
token_ids = [12345, 6789, 1011]  # 每个序列生成的 token ID
# 长度 = len(seqs)
```

**为什么用 `call()` 而不是直接调用？**
```
多卡推理时：
  call() 会通过共享内存通知所有工作进程
  确保所有 GPU 同步执行相同的推理任务
  
单卡推理时：
  call() 直接调用方法，没有额外开销
```

---

#### 第 3 行：后处理

```python
self.scheduler.postprocess(seqs, token_ids)
```

**作用**：处理推理结果，更新序列状态

**内部操作**：
```python
def postprocess(self, seqs, token_ids):
    for seq, token_id in zip(seqs, token_ids):
        # 1. 将新 token 追加到序列
        seq.append_token(token_id)
        
        # 2. 检查是否完成
        if token_id == self.eos or seq.num_completion_tokens == seq.max_tokens:
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)  # 释放 KV-Cache
            self.running.remove(seq)            # 从运行队列移除
```

---

#### 第 4 行：收集完成的序列

```python
outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
```

**作用**：筛选出已完成的序列，收集其结果

**列表推导式分解**：
```python
outputs = []
for seq in seqs:
    if seq.is_finished:  # 只处理已完成的序列
        outputs.append((seq.seq_id, seq.completion_token_ids))
        #               ↑              ↑
        #           序列唯一ID    生成的所有token

# 示例输出：
# [(0, [12345, 6789, 1011, 2]),  # 序列0完成，EOS=2
#  (2, [3456, 7890, 2])]          # 序列2完成
```

---

#### 第 5 行：统计 token 数量

```python
num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
```

**作用**：计算本轮处理的 token 数量（用于吞吐量统计）

**逻辑分解**：
```python
if is_prefill:
    # Prefill 阶段：统计所有输入 token 数
    num_tokens = sum(len(seq) for seq in seqs)
    # 例如：3个序列，长度分别为 100, 200, 150
    # num_tokens = 100 + 200 + 150 = 450
else:
    # Decode 阶段：返回负数表示序列数
    num_tokens = -len(seqs)
    # 例如：3个序列
    # num_tokens = -3
```

**为什么 Decode 用负数？**
```
用于区分两个阶段：
  num_tokens > 0  → Prefill 阶段
  num_tokens < 0  → Decode 阶段
  
后续计算吞吐量时：
  Prefill: tokens / time
  Decode:  -num_tokens / time = 序列数 / time
```

---

#### 第 6 行：返回结果

```python
return outputs, num_tokens
```

**返回值**：
| 变量 | 类型 | 含义 |
|------|------|------|
| `outputs` | `list[tuple]` | 已完成序列的 (seq_id, token_ids) 列表 |
| `num_tokens` | `int` | 处理的 token 数（正=Prefill，负=Decode） |

---

## 方法二：`is_finished()` - 检查完成

```python
def is_finished(self):
    return self.scheduler.is_finished()
```

**作用**：检查所有请求是否都已完成

**内部逻辑**：
```python
# scheduler.py 中
def is_finished(self):
    return not self.waiting and not self.running
    #           ↑                    ↑
    #      等待队列为空          运行队列为空
```

**返回值**：
- `True`：所有请求都已完成
- `False`：还有请求在处理中

---

## 方法三：`generate()` - 完整生成流程

这是**用户调用的主入口**。

```python
def generate(
    self,
    prompts: list[str] | list[list[int]],
    sampling_params: SamplingParams | list[SamplingParams],
    use_tqdm: bool = True,
) -> list[str]:
```

### 参数说明

| 参数 | 类型 | 含义 |
|------|------|------|
| `prompts` | `list[str]` 或 `list[list[int]]` | 提示词列表（文本或token ID） |
| `sampling_params` | `SamplingParams` 或其列表 | 采样参数 |
| `use_tqdm` | `bool` | 是否显示进度条 |

### 逐行解析

#### 初始化进度条

```python
if use_tqdm:
    pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
```

**作用**：创建进度条，总数为请求数量

**显示效果**：
```
Generating:  50%|██████████          | 5/10 [00:02<00:02, Prefill: 1024tok/s, Decode: 40tok/s]
```

---

#### 标准化采样参数

```python
if not isinstance(sampling_params, list):
    sampling_params = [sampling_params] * len(prompts)
```

**作用**：确保每个 prompt 都有对应的采样参数

**示例**：
```python
# 输入
prompts = ["Hello", "World", "AI"]
sampling_params = SamplingParams(temperature=0.7)

# 转换后
sampling_params = [
    SamplingParams(temperature=0.7),
    SamplingParams(temperature=0.7),
    SamplingParams(temperature=0.7),
]
```

---

#### 添加所有请求

```python
for prompt, sp in zip(prompts, sampling_params):
    self.add_request(prompt, sp)
```

**作用**：将所有请求加入调度器的等待队列

**内部流程**：
```python
def add_request(self, prompt, sampling_params):
    if isinstance(prompt, str):
        prompt = self.tokenizer.encode(prompt)  # 文本 → token IDs
    seq = Sequence(prompt, sampling_params)     # 创建序列对象
    self.scheduler.add(seq)                     # 加入等待队列
```

---

#### 初始化变量

```python
outputs = {}
prefill_throughput = decode_throughput = 0.
```

| 变量 | 类型 | 含义 |
|------|------|------|
| `outputs` | `dict` | 存储结果的字典 {seq_id: token_ids} |
| `prefill_throughput` | `float` | Prefill 吞吐量 (tokens/s) |
| `decode_throughput` | `float` | Decode 吞吐量 (tokens/s) |

---

#### 主循环

```python
while not self.is_finished():
    t = perf_counter()
    output, num_tokens = self.step()
```

**作用**：不断执行推理步骤，直到所有请求完成

**流程**：
```
while 还有未完成的请求:
    ├─ 记录开始时间
    ├─ 执行一步推理 (step)
    ├─ 计算吞吐量
    ├─ 更新进度条
    └─ 收集完成的结果
```

---

#### 计算吞吐量

```python
if use_tqdm:
    if num_tokens > 0:
        prefill_throughput = num_tokens / (perf_counter() - t)
    else:
        decode_throughput = -num_tokens / (perf_counter() - t)
    pbar.set_postfix({
        "Prefill": f"{int(prefill_throughput)}tok/s",
        "Decode": f"{int(decode_throughput)}tok/s",
    })
```

**计算公式**：
```
Prefill 吞吐量 = 处理的 token 数 / 耗时
Decode 吞吐量 = 处理的序列数 / 耗时

示例：
  Prefill: 512 tokens / 0.5s = 1024 tok/s
  Decode:  4 sequences / 0.1s = 40 tok/s
```

**进度条显示**：
```
Generating: 50%|██████| 5/10 [Prefill: 1024tok/s, Decode: 40tok/s]
```

---

#### 收集完成的结果

```python
for seq_id, token_ids in output:
    outputs[seq_id] = token_ids
    if use_tqdm:
        pbar.update(1)
```

**作用**：将完成的序列结果保存到字典中

**为什么用字典？**
```
请求完成的顺序可能与添加顺序不同：
  添加顺序：[0, 1, 2, 3, 4]
  完成顺序：[2, 0, 4, 1, 3]  # 短的先完成

用字典按 seq_id 存储，最后按顺序取出
```

---

#### 整理输出

```python
outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
```

**作用**：按 seq_id 顺序排列结果

**示例**：
```python
# 字典（无序）
outputs = {2: [1,2,3], 0: [4,5,6], 1: [7,8,9]}

# 排序后（有序列表）
outputs = [[4,5,6], [7,8,9], [1,2,3]]
#            ↑        ↑        ↑
#          seq_id=0  seq_id=1  seq_id=2
```

---

#### 解码为文本

```python
outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
```

**作用**：将 token IDs 转换为人类可读的文本

**示例**：
```python
# 输入
token_ids = [101, 2054, 2003, 2023, 1029]

# 输出
{
    "text": "What is this?",
    "token_ids": [101, 2054, 2003, 2023, 1029]
}
```

---

#### 关闭进度条

```python
if use_tqdm:
    pbar.close()
return outputs
```

**作用**：清理进度条资源，返回最终结果

---

## 完整流程图

```
generate(prompts, sampling_params)
        │
        ├─ 初始化进度条
        │
        ├─ 标准化 sampling_params
        │
        ├─ for prompt in prompts:
        │       add_request(prompt, sp)
        │           └─ tokenize → Sequence → scheduler.waiting
        │
        ├─ while not is_finished():
        │   │
        │   ├─ step()
        │   │   │
        │   │   ├─ scheduler.schedule()
        │   │   │   ├─ Prefill: 从 waiting 取序列
        │   │   │   └─ Decode: 从 running 取序列
        │   │   │
        │   │   ├─ model_runner.call("run", seqs, is_prefill)
        │   │   │   ├─ 准备输入数据
        │   │   │   ├─ 执行模型推理
        │   │   │   └─ 采样生成 token
        │   │   │
        │   │   ├─ scheduler.postprocess(seqs, token_ids)
        │   │   │   ├─ 追加 token 到序列
        │   │   │   └─ 检查是否完成
        │   │   │
        │   │   └─ 返回 (outputs, num_tokens)
        │   │
        │   ├─ 计算吞吐量
        │   │
        │   ├─ 更新进度条
        │   │
        │   └─ 收集完成的结果
        │
        ├─ 按 seq_id 排序结果
        │
        ├─ 解码为文本
        │
        └─ 返回 outputs
```

---

## 数据流向

```
用户输入                          内部处理                         用户输出
─────────                        ──────────                       ──────────
prompts                          tokenize                         outputs
["Hello"]    ──────────────────► [101, 2054] ────────────┐
["World"]    ──────────────────► [102, 3010] ──────┐     │
                                                    │     │
                                    ┌───────────────┴─────┴───────────────┐
                                    │                                     │
                                    │         Scheduler                   │
                                    │    ┌─────────────────────┐          │
                                    │    │ waiting: [seq0, seq1]│         │
                                    │    │ running: []          │         │
                                    │    └─────────────────────┘          │
                                    │              │                      │
                                    │              ▼ schedule()           │
                                    │    ┌─────────────────────┐          │
                                    │    │ waiting: []          │         │
                                    │    │ running: [seq0, seq1]│         │
                                    │    └─────────────────────┘          │
                                    │              │                      │
                                    │              ▼ model_runner.run()   │
                                    │    ┌─────────────────────┐          │
                                    │    │ 生成 token:         │          │
                                    │    │ seq0: [101,2054,555]│          │
                                    │    │ seq1: [102,3010,666]│          │
                                    │    └─────────────────────┘          │
                                    │              │                      │
                                    │              ▼ postprocess()        │
                                    │    ┌─────────────────────┐          │
                                    │    │ seq0: FINISHED      │          │
                                    │    │ seq1: RUNNING       │          │
                                    │    └─────────────────────┘          │
                                    │              │                      │
                                    │              ▼ decode()             │
                                    │    ┌─────────────────────┐          │
                                    │    │ seq0: "Hello world" │──────────┼──► outputs[0]
                                    │    │ seq1: "World peace" │──────────┼──► outputs[1]
                                    │    └─────────────────────┘          │
                                    │                                     │
                                    └─────────────────────────────────────┘
```

---

## 关键设计点

### 1. 两阶段推理

```
Prefill 阶段：
  ├─ 处理完整的提示词
  ├─ 计算密集型
  └─ 一次性处理所有输入 token

Decode 阶段：
  ├─ 逐 token 生成
  ├─ 内存密集型
  └─ 每步只处理 1 个 token/序列
```

### 2. 批处理

```
多个请求同时处理：
  ├─ 共享计算资源
  ├─ 提高 GPU 利用率
  └─ 减少单请求延迟
```

### 3. 异步完成

```
请求可能以不同顺序完成：
  ├─ 短请求先完成
  ├─ 长请求后完成
  └─ 用字典收集，最后排序
```

### 4. 吞吐量统计

```
实时显示性能指标：
  ├─ Prefill: tokens/s（输入处理速度）
  └─ Decode: tokens/s（生成速度）
```

---

## 使用示例

```python
from nanovllm import LLM
from nanovllm.sampling_params import SamplingParams

# 初始化引擎
llm = LLM("Qwen3-0.6B")

# 设置采样参数
params = SamplingParams(temperature=0.7, max_tokens=100)

# 批量生成
results = llm.generate(
    prompts=["What is AI?", "Explain machine learning."],
    sampling_params=params,
    use_tqdm=True
)

# 输出结果
for r in results:
    print(r["text"])
```

**输出**：
```
Generating: 100%|██████████| 2/2 [00:05<00:00, Prefill: 1024tok/s, Decode: 40tok/s]

AI is a branch of computer science...
Machine learning is a subset of AI that...
```

---

## 总结表

| 方法 | 作用 | 调用频率 |
|------|------|---------|
| `step()` | 执行一步推理 | 每次循环调用 |
| `is_finished()` | 检查是否完成 | 每次循环检查 |
| `generate()` | 完整生成流程 | 用户调用一次 |

| 变量 | 含义 | 用途 |
|------|------|------|
| `seqs` | 本轮处理的序列 | 传递给模型 |
| `is_prefill` | 是否 Prefill 阶段 | 选择推理模式 |
| `token_ids` | 生成的 token | 追加到序列 |
| `outputs` | 完成的结果 | 返回给用户 |
| `num_tokens` | token 数量 | 计算吞吐量 |

---

## 核心理解

🎯 **`generate()` 是用户的主入口**：
1. 接收提示词列表和采样参数
2. 将请求加入调度队列
3. 循环调用 `step()` 执行推理
4. 收集完成的结果
5. 解码为文本返回

🎯 **`step()` 是推理的核心**：
1. 调度器决定处理哪些序列
2. 模型执行推理生成 token
3. 后处理更新序列状态
4. 返回完成的结果和统计信息

🎯 **设计亮点**：
- 批处理提高效率
- 两阶段调度优化资源
- 异步完成支持不同长度请求
- 实时吞吐量统计

---

**完成时间**：2026-03-09
