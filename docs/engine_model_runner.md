# model_runner.py - 模型推理执行引擎

## 文件作用

该模块实现了 **LLM 模型推理的执行层**，负责：
1. **加载和初始化模型**（Qwen3ForCausalLM）
2. **KV-Cache内存分配和管理**
3. **数据预处理**（Prefill和Decode阶段）
4. **CUDA图捕获**优化和执行
5. **多卡张量并行**（Tensor Parallelism）支持
6. **采样和生成**

---

## 核心概念

### 推理流程

```
ModelRunner
    ├─ Prefill（处理完整提示词）
    │   ├─ 计算query和key的注意力
    │   └─ 更新所有KV-Cache块
    │
    └─ Decode（逐token生成）
        ├─ 计算当前token的query
        ├─ 读取缓存的key和value
        └─ 生成1个新token
```

### 多进程架构

```
rank=0 (主进程)
    ├─ 加载模型、初始化参数
    ├─ 执行推理
    └─ 通过SharedMemory与其他进程通信

rank>0 (从进程)
    ├─ 接收推理任务
    └─ 执行张量并行计算
```

---

## ModelRunner 类

### 初始化流程

```python
def __init__(self, config: Config, rank: int, event: Event | list[Event]):
    """
    初始化推理引擎
    
    步骤：
    1. 分布式初始化（NCCL）
    2. 模型加载
    3. KV-Cache预分配
    4. CUDA图捕获（可选）
    """
```

**初始化细节**：

```python
# 1. 分布式设置
dist.init_process_group("nccl", ...)
torch.cuda.set_device(rank)

# 2. 模型加载
self.model = Qwen3ForCausalLM(hf_config)
load_model(self.model, config.model)

# 3. 采样器
self.sampler = Sampler()

# 4. 预热模型
self.warmup_model()

# 5. KV-Cache分配
self.allocate_kv_cache()

# 6. 图捕获（Decode优化）
if not self.enforce_eager:
    self.capture_cudagraph()
```

---

## 核心方法

### 1. allocate_kv_cache - KV-Cache分配

```python
def allocate_kv_cache(self):
    """
    根据GPU内存自动计算KV-Cache块数
    
    流程：
    1. 查询GPU当前内存使用
    2. 计算单个块的内存大小
    3. 分配尽可能多的块（保留headroom）
    4. 关联到模型的各层
    """
```

**计算逻辑**：

```python
# GPU总内存
total = torch.cuda.mem_get_info()[1]

# 已使用的内存
used = total - free
peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

# 单个块的大小（字节）
block_bytes = 2 * num_layers * block_size * num_kv_heads * head_dim * dtype_size

# 可分配块数
num_blocks = int(total * gpu_memory_utilization - used - peak + current) // block_bytes
```

**内存分配**：

```
GPU总内存 (16GB)
    │
    ├─ 模型参数 (3GB)
    ├─ 激活值 (2GB)
    ├─ KV-Cache (8GB) ← allocate_kv_cache分配
    └─ 显存余量 (3GB)
```

**关键代码**：

```python
# 创建KV-Cache张量
self.kv_cache = torch.empty(
    2,  # K和V
    hf_config.num_hidden_layers,
    config.num_kvcache_blocks,
    self.block_size,
    num_kv_heads,
    head_dim
)

# 将KV-Cache分配给各层
layer_id = 0
for module in self.model.modules():
    if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
        module.k_cache = self.kv_cache[0, layer_id]
        module.v_cache = self.kv_cache[1, layer_id]
        layer_id += 1
```

---

### 2. prepare_prefill - Prefill数据预处理

处理完整提示词的注意力计算数据。

```python
def prepare_prefill(self, seqs: list[Sequence]):
    """
    准备Prefill阶段所需的数据
    
    返回：
    - input_ids: 未缓存的token IDs
    - positions: token位置编码
    """
```

**关键输出**：

| 输出 | 说明 | 形状 |
|------|------|------|
| `input_ids` | 本轮要处理的token IDs | [num_tokens] |
| `positions` | 每个token的位置索引 | [num_tokens] |
| `cu_seqlens_q` | query的累积序列长度 | [num_seqs+1] |
| `cu_seqlens_k` | key的累积序列长度 | [num_seqs+1] |
| `slot_mapping` | token到KV-Cache块的映射 | [num_tokens] |
| `block_tables` | 序列的block表（如有缓存） | [num_seqs, max_blocks] |

**工作流程**：

```
多序列Prefill示例：
seq1: [101, 2054, 2003, 4605, 3010]
seq2: [101, 2054, 2003, 2222]
seq3已有缓存: [101, 2054]，新token: [2003, 4605]

处理：
1. seq1: 输入5个token，都是新的
2. seq2: 输入4个token，都是新的  
3. seq3: 输入2个token（缓存），计算2个新token

input_ids = [101, 2054, 2003, 4605, 3010, 101, 2054, 2003, 2222, 2003, 4605]
positions = [0, 1, 2, 3, 4, 0, 1, 2, 3, 2, 3]

cu_seqlens_q = [0, 5, 9, 11]  # query长度
cu_seqlens_k = [0, 5, 9, 13]  # key长度（包括缓存）
```

**前缀缓存处理**：

```python
if cu_seqlens_k[-1] > cu_seqlens_q[-1]:  # 有缓存
    block_tables = self.prepare_block_tables(seqs)
```

---

### 3. prepare_decode - Decode数据预处理

处理逐token生成阶段。

```python
def prepare_decode(self, seqs: list[Sequence]):
    """
    准备Decode阶段所需的数据
    
    返回：
    - input_ids: 最后生成的token
    - positions: 当前token位置
    - slot_mapping: token在KV-Cache中的位置
    - context_lens: 每个序列的上下文长度
    - block_tables: KV-Cache块映射
    """
```

**工作流程**：

```
Decode第n步：
seq1: [101, 2054, 2003, 4605, 3010, 2222]
         └─ prompt ─┘  └─────── decode ──────┘
                                        ↑
                                    当前token
seq2: [101, 2054, 2003, 2222, 4605]
                          ↑
                      当前token

输出：
input_ids = [2222, 4605]        # 两个序列的最后一个token
positions = [5, 4]              # 在完整序列中的位置
context_lens = [6, 5]           # 上下文长度（用于注意力）
slot_mapping = [block*256+slot1, block*256+slot2]  # KV-Cache位置
```

---

### 4. prepare_sample - 采样参数

```python
def prepare_sample(self, seqs: list[Sequence]):
    """
    准备采样所需的参数
    
    返回：
    - temperatures: 每个序列的温度参数
    """
```

---

### 5. run_model - 模型推理执行

```python
@torch.inference_mode()
def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, 
              is_prefill: bool):
    """
    执行模型推理
    
    策略：
    - Prefill或强制eager: 直接执行
    - Decode且batch_size小: 使用CUDA图加速
    """
```

**执行路径**：

```python
if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
    # 路径1: 直接执行（不使用图）
    return self.model.compute_logits(self.model(input_ids, positions))
else:
    # 路径2: 使用CUDA图加速（Decode优化）
    graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
    # 更新图中的输入
    graph_vars["input_ids"][:bs] = input_ids
    graph_vars["positions"][:bs] = positions
    # 重放图
    graph.replay()
    return self.model.compute_logits(graph_vars["outputs"][:bs])
```

**CUDA图优化**：
- 预先捕获多个不同batch size的计算图
- Decode时直接重放，避免CPU-GPU通信开销

---

### 6. capture_cudagraph - CUDA图捕获

```python
@torch.inference_mode()
def capture_cudagraph(self):
    """
    为不同batch size预先捕获CUDA计算图
    
    batch_size: [1, 2, 4, 8, 16, 32, ..., max_bs]
    """
```

**工作流程**：

```python
for bs in reversed(self.graph_bs):
    graph = torch.cuda.CUDAGraph()
    
    # 1. 预热（准备GPU状态）
    outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
    
    # 2. 捕获（记录计算图）
    with torch.cuda.graph(graph, self.graph_pool):
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
    
    # 3. 保存
    self.graphs[bs] = graph
```

**性能提升**：
```
直接执行: 需要多次kernel launch
使用图: kernel launch已预先录制，只需重放
提升: ~20-30% Decode吞吐量
```

---

### 7. run - 主推理接口

```python
def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
    """
    执行一轮推理
    
    流程：
    1. 预处理数据（prepare_prefill或prepare_decode）
    2. 模型推理（run_model）
    3. 采样生成token（sampler）
    4. 返回生成的token IDs
    """
```

**返回值**：
```python
# rank=0返回token IDs
token_ids = [120, 345, 234, ...]  # 新生成的token

# rank>0返回None（从进程无需返回）
token_ids = None
```

---

### 8. 多进程通信

```python
def write_shm(self, method_name, *args):
    """向SharedMemory写入方法和参数"""
    data = pickle.dumps([method_name, *args])
    n = len(data)
    self.shm.buf[0:4] = n.to_bytes(4, "little")
    self.shm.buf[4:n+4] = data
    for event in self.event:
        event.set()  # 触发从进程

def read_shm(self):
    """从SharedMemory读取方法和参数"""
    self.event.wait()
    n = int.from_bytes(self.shm.buf[0:4], "little")
    method_name, *args = pickle.loads(self.shm.buf[4:n+4])
    self.event.clear()
    return method_name, args
```

**多卡执行流程**：

```
rank=0 (主进程)
    │
    ├─ run(seqs, is_prefill)
    │   │
    │   ├─ 写入SharedMemory
    │   ├─ 触发event
    │   └─ 执行推理
    │
rank=1 (从进程)
    │
    ├─ loop()
    │   │
    │   ├─ 等待event
    │   ├─ 读取SharedMemory
    │   └─ 执行同步计算
    │
rank=2, rank=3, ...
    └─ 同理
```

---

## 完整推理流程

```
┌────────────────────────────────┐
│   ModelRunner.__init__()        │
├────────────────────────────────┤
│ 1. 分布式初始化                 │
│ 2. 模型加载                     │
│ 3. KV-Cache预分配              │
│ 4. CUDA图捕获                  │
└────────────┬────────────────────┘
             │
             ▼
┌────────────────────────────────┐
│    run(seqs, is_prefill=True)  │ ← Prefill
├────────────────────────────────┤
│ prepare_prefill()               │
│   ├─ 整理input_ids              │
│   ├─ 计算positions              │
│   └─ 生成slot_mapping           │
│                                 │
│ run_model()                     │
│   └─ 执行forward，得logits      │
│                                 │
│ sampler()                       │
│   └─ 采样生成token              │
│                                 │
│ 返回: [token_id1, token_id2]   │
└────────────┬────────────────────┘
             │
             ▼
┌────────────────────────────────┐
│    run(seqs, is_prefill=False) │ ← Decode
├────────────────────────────────┤
│ prepare_decode()                │
│   ├─ 取最后一个token            │
│   ├─ 计算当前位置                │
│   └─ 查询KV-Cache映射           │
│                                 │
│ run_model()（可用CUDA图加速）   │
│   └─ 执行forward，得logits      │
│                                 │
│ sampler()                       │
│   └─ 采样生成1个新token         │
│                                 │
│ 返回: [token_id1]              │
└────────────┬────────────────────┘
             │
             ▼
┌────────────────────────────────┐
│    exit()                       │
├────────────────────────────────┤
│ 清理SharedMemory                │
│ 销毁分布式进程组                 │
│ 同步GPU                        │
└────────────────────────────────┘
```

---

## 关键优化技术

### 1. KV-Cache 块管理

```
自动计算块数（不需手动配置）
充分利用GPU显存
支持BlockManager的块重用机制
```

### 2. CUDA 图捕获

```
Prefill: 直接执行
Decode: 使用预捕获的CUDA图
优势: 减少CPU-GPU通信，提升吞吐量
```

### 3. 张量并行

```
多卡分布式推理
通过SharedMemory和NCCL协调
支持大模型推理
```

### 4. 前缀缓存

```
block_tables在Prefill中被使用
支持提示词缓存复用
```

---

## 性能特征

| 操作 | 复杂度 | 备注 |
|------|--------|------|
| allocate_kv_cache | O(1) | 常数时间 |
| prepare_prefill | O(num_tokens) | 线性扫描 |
| prepare_decode | O(batch_size) | 小常数 |
| run_model | O(seq_len * hidden) | 模型复杂度 |
| CUDA图重放 | O(1) | 预录制 |

---

## 与其他模块的协作

```
LLMEngine
    │
    └─ step()
        │
        ├─ Scheduler.schedule() → [seqs, is_prefill]
        │
        ├─ ModelRunner.run(seqs, is_prefill)
        │   ├─ prepare_prefill/prepare_decode
        │   ├─ run_model
        │   └─ 返回token_ids
        │
        └─ Scheduler.postprocess(seqs, token_ids)
```
