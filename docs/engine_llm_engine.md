# llm_engine.py - 高层推理引擎接口

## 文件作用

该模块提供了 **LLM推理的用户级API**，是整个nano-vllm的最外层接口。它：
1. 封装了底层的调度、推理、采样逻辑
2. 提供简洁的 `generate()` 接口供用户使用
3. 管理多卡推理中的进程生命周期
4. 统计推理吞吐量

---

## LLMEngine 类

### 初始化

```python
def __init__(self, model, **kwargs):
    """
    初始化推理引擎
    
    参数：
    - model: 模型路径或名称（HuggingFace格式）
    - **kwargs: 配置参数（传入Config）
    
    步骤：
    1. 构建Config对象
    2. 启动多进程（张量并行）
    3. 初始化ModelRunner
    4. 加载tokenizer
    5. 创建Scheduler
    """
```

**初始化细节**：

```python
# 1. 配置提取
config_fields = {field.name for field in fields(Config)}
config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
config = Config(model, **config_kwargs)

# 2. 多进程创建（张量并行）
ctx = mp.get_context("spawn")
for i in range(1, config.tensor_parallel_size):
    event = ctx.Event()  # 进程间同步事件
    process = ctx.Process(target=ModelRunner, args=(config, i, event))
    process.start()
    self.ps.append(process)
    self.events.append(event)

# 3. rank=0进程的ModelRunner
self.model_runner = ModelRunner(config, 0, self.events)

# 4. Tokenizer加载
self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
config.eos = self.tokenizer.eos_token_id

# 5. 调度器
self.scheduler = Scheduler(config)

# 6. 注册清理函数
atexit.register(self.exit)
```

---

## 核心方法

### 1. add_request - 添加请求

```python
def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
    """
    添加推理请求到引擎
    
    参数：
    - prompt: 文本提示词或token ID列表
    - sampling_params: 采样参数（温度、最大长度等）
    
    流程：
    1. 将文本转换为token ID
    2. 创建Sequence对象
    3. 加入scheduler的等待队列
    """
```

**代码流程**：

```python
if isinstance(prompt, str):
    prompt = self.tokenizer.encode(prompt)
seq = Sequence(prompt, sampling_params)
self.scheduler.add(seq)
```

---

### 2. step - 推理步骤

执行一轮调度-推理-采样的完整流程。

```python
def step(self):
    """
    执行一个推理步骤
    
    返回：
    - outputs: [(seq_id, completion_token_ids), ...]
    - num_tokens: 本轮处理的token数
    
    流程：
    1. 调度（scheduler.schedule）
    2. 推理（model_runner.run）
    3. 后处理（scheduler.postprocess）
    """
```

**详细流程**：

```python
# 1. 调度阶段
seqs, is_prefill = self.scheduler.schedule()
# 返回本轮要执行的序列和阶段标识

# 2. 推理阶段
token_ids = self.model_runner.call("run", seqs, is_prefill)
# 执行forward，生成token

# 3. 后处理阶段
self.scheduler.postprocess(seqs, token_ids)
# 追加token，检查完成条件

# 4. 收集完成的序列
outputs = [(seq.seq_id, seq.completion_token_ids) 
           for seq in seqs if seq.is_finished]

# 5. 统计token数
num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
# Prefill: 统计输入token数（正数）
# Decode: 统计输出token数（负数表示decode）

return outputs, num_tokens
```

**Token统计设计**：

```
Prefill: num_tokens > 0
    sum(len(seq) for seq in seqs) = sum of all input tokens

Decode: num_tokens < 0
    -len(seqs) = -(number of sequences)
    
示例：
Prefill: 512个token → 512
Decode: 4个序列 → -4

用途：计算吞吐量 = tokens / time
```

---

### 3. is_finished - 检查完成

```python
def is_finished(self):
    """检查所有请求是否完成"""
    return self.scheduler.is_finished()
```

**检查条件**：
```python
# scheduler中
not self.waiting and not self.running
# waiting队列和running队列都为空
```

---

### 4. generate - 主生成接口

用户主要调用这个方法。

```python
def generate(
    self,
    prompts: list[str] | list[list[int]],
    sampling_params: SamplingParams | list[SamplingParams],
    use_tqdm: bool = True,
) -> list[str]:
    """
    生成多个请求的结果
    
    参数：
    - prompts: 列表，每个元素是一个提示词
    - sampling_params: 采样参数（可共享或单独指定）
    - use_tqdm: 是否显示进度条
    
    返回：
    - list[str]: 每个请求的生成结果
    """
```

**工作流程**：

```python
# 1. 初始化进度条
if use_tqdm:
    pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)

# 2. 标准化sampling_params
if not isinstance(sampling_params, list):
    sampling_params = [sampling_params] * len(prompts)

# 3. 添加所有请求
for prompt, sp in zip(prompts, sampling_params):
    self.add_request(prompt, sp)

# 4. 推理循环
outputs = {}
prefill_throughput = decode_throughput = 0.
while not self.is_finished():
    t = perf_counter()
    output, num_tokens = self.step()
    
    # 计算吞吐量
    if use_tqdm:
        if num_tokens > 0:
            prefill_throughput = num_tokens / (perf_counter() - t)
        else:
            decode_throughput = -num_tokens / (perf_counter() - t)
        pbar.set_postfix({
            "Prefill": f"{int(prefill_throughput)}tok/s",
            "Decode": f"{int(decode_throughput)}tok/s",
        })
    
    # 收集结果
    for seq_id, token_ids in output:
        outputs[seq_id] = token_ids
        if use_tqdm:
            pbar.update(1)

# 5. 结果整理
outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
outputs = [{"text": self.tokenizer.decode(token_ids), 
            "token_ids": token_ids} 
           for token_ids in outputs]

if use_tqdm:
    pbar.close()

return outputs
```

---

### 5. exit - 清理资源

```python
def exit(self):
    """
    清理所有资源
    
    流程：
    1. 通知ModelRunner退出
    2. 等待所有进程结束
    3. 释放GPU资源
    """
```

**清理步骤**：

```python
# 1. 向rank=0 ModelRunner发送exit信号
self.model_runner.call("exit")

# 2. 删除ModelRunner
del self.model_runner

# 3. 等待所有子进程结束
for p in self.ps:
    p.join()
```

**自动触发**：
```python
# 在__init__中注册
atexit.register(self.exit)
# 程序结束时自动调用
```

---

## 完整推理流程

```
用户调用
    │
    ▼
generate(prompts, sampling_params)
    │
    ├─ add_request() ─→ 所有requests进入waiting队列
    │
    └─ while not is_finished():
        │
        ├─ step()
        │   │
        │   ├─ scheduler.schedule()
        │   │   │
        │   │   ├─ Prefill阶段（如有新请求）
        │   │   │   └─ waiting → running
        │   │   │
        │   │   └─ Decode阶段（如无新请求）
        │   │       └─ running继续生成
        │   │
        │   ├─ model_runner.run(seqs, is_prefill)
        │   │   │
        │   │   ├─ prepare_prefill/prepare_decode
        │   │   ├─ run_model()
        │   │   └─ sampler()
        │   │
        │   ├─ scheduler.postprocess(seqs, token_ids)
        │   │   │
        │   │   ├─ append_token()
        │   │   ├─ 检查是否完成
        │   │   └─ 完成 → deallocate + 移除
        │   │
        │   └─ 收集完成的序列
        │
        └─ 更新进度条
    
    ▼
结果整理（排序、解码）
    │
    ▼
返回 list[dict]
```

---

## 数据流向

```
用户输入
    │
    ▼ add_request()
Prompt (str) → Tokenize → token_ids → Sequence → waiting
    │
    ├─ Prefill循环
    │   ├─ scheduler.schedule()
    │   │   └─ waiting → running (allocate)
    │   │
    │   ├─ model_runner.run(prefill=True)
    │   │   └─ forward(all tokens)
    │   │
    │   └─ scheduler.postprocess()
    │       └─ append_token
    │
    ├─ Decode循环（每步生成1个token）
    │   ├─ scheduler.schedule()
    │   │   └─ running: select, may_append
    │   │
    │   ├─ model_runner.run(prefill=False)
    │   │   └─ forward(last token)
    │   │
    │   └─ scheduler.postprocess()
    │       ├─ append_token
    │       └─ is_finished? → deallocate
    │
    ▼
outputs dict: {seq_id: token_ids}
    │
    ▼
Detokenize → text
    │
    ▼
返回给用户
```

---

## 多卡推理架构

```
LLMEngine (rank=0)
    │
    ├─ ModelRunner(rank=0)
    │   └─ run()
    │       └─ call("run", seqs, is_prefill)
    │           ├─ write_shm()  [发送任务]
    │           ├─ 执行推理
    │           └─ 等待从进程完成
    │
    ├─ Process(ModelRunner, rank=1)
    │   └─ loop()
    │       └─ read_shm()  [接收任务]
    │           └─ call()   [执行]
    │
    ├─ Process(ModelRunner, rank=2)
    │   └─ loop()
    │
    └─ Process(ModelRunner, rank=N)
        └─ loop()

同步机制：
    │
    ├─ SharedMemory (IPC)
    │   └─ 传递方法名和参数
    │
    ├─ Event (同步)
    │   └─ rank=0触发，从进程等待
    │
    └─ dist.barrier() (NCCL)
        └─ 确保所有进程同步完成
```

---

## 吞吐量统计

```python
# Prefill吞吐量（tokens/秒）
prefill_throughput = num_tokens / elapsed_time
# 例：512 tokens / 0.5s = 1024 tok/s

# Decode吞吐量（tokens/秒）
decode_throughput = -num_tokens / elapsed_time  # 负数转正
# 例：4 sequences / 0.1s = 40 tok/s

# 显示格式
"Prefill: 1024tok/s, Decode: 40tok/s"
```

---

## 使用示例

```python
from nanovllm import LLM
from nanovllm.sampling_params import SamplingParams

# 1. 初始化引擎
llm = LLM("Qwen3-0.6B", tensor_parallel_size=1)

# 2. 定义采样参数
sampling_params = SamplingParams(temperature=0.7, max_tokens=100)

# 3. 单个请求
response = llm.generate(
    "Hello, world!",
    sampling_params
)
print(response[0]["text"])

# 4. 批量请求
responses = llm.generate(
    ["What is AI?", "Explain machine learning"],
    sampling_params
)
for resp in responses:
    print(resp["text"])
```

---

## 性能特征

| 操作 | 复杂度 | 备注 |
|------|--------|------|
| add_request | O(prompt_len) | tokenize |
| step | O(batch_size) | 调度 + 推理 |
| generate | O(总token数) | 推理循环 |
| exit | O(1) | 清理 |

---

## 与其他模块的关系

```
LLMEngine
    │
    ├─ Scheduler
    │   ├─ schedule() ← 调度逻辑
    │   └─ postprocess() ← 后处理
    │
    ├─ ModelRunner
    │   ├─ run() ← 推理执行
    │   └─ call() ← 多卡通信
    │
    ├─ BlockManager (via Scheduler)
    │   └─ 内存管理
    │
    ├─ Sequence (via Scheduler)
    │   └─ 请求抽象
    │
    └─ Tokenizer (HuggingFace)
        └─ 文本-token互转
```
