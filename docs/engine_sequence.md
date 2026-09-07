# sequence.py - 序列管理模块

## 文件作用

该模块定义了 **`Sequence`** 类和 **`SequenceStatus`** 枚举，用于管理 LLM 推理中的**单个请求序列**。每个用户请求都被抽象为一个 `Sequence` 对象，跟踪其生成过程中的状态、token、KV-Cache等信息。

---

## 核心类

### SequenceStatus（枚举）

定义序列的三种生命周期状态：

```python
class SequenceStatus(Enum):
    WAITING = auto()   # 等待调度
    RUNNING = auto()   # 正在推理
    FINISHED = auto()  # 已完成生成
```

---

### Sequence（序列类）

管理单个推理序列的完整生命周期。

#### 初始化属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `seq_id` | int | 唯一序列ID（自增计数器） |
| `status` | SequenceStatus | 序列当前状态 |
| `token_ids` | list[int] | 累积生成的所有token ID |
| `last_token` | int | 最后一个生成的token |
| `num_tokens` | int | 当前总token数 |
| `num_prompt_tokens` | int | 提示词token数 |
| `num_cached_tokens` | int | KV-Cache中已缓存的token数 |
| `block_table` | list[int] | KV-Cache物理block ID映射表 |
| `temperature` | float | 采样温度参数 |
| `max_tokens` | int | 最多生成token数 |
| `ignore_eos` | bool | 是否忽略EOS token |

#### 关键属性和方法

```python
@property
def num_completion_tokens(self) -> int:
    """生成token数 = 总token数 - 提示词token数"""
    return self.num_tokens - self.num_prompt_tokens

@property
def num_cached_blocks(self) -> int:
    """已缓存的block数量"""
    return self.num_cached_tokens // self.block_size

@property
def num_blocks(self) -> int:
    """总block数量（包括不完整block）"""
    return (self.num_tokens + self.block_size - 1) // self.block_size

@property
def last_block_num_tokens(self) -> int:
    """最后一个block中的token数"""
    return self.num_tokens - (self.num_blocks - 1) * self.block_size

def block(self, i: int) -> list[int]:
    """获取第i个block的token IDs"""
    return self.token_ids[i*self.block_size: (i+1)*self.block_size]

def append_token(self, token_id: int):
    """追加新生成的token"""
    self.token_ids.append(token_id)
    self.last_token = token_id
    self.num_tokens += 1
```

---

## 工作流程

```
初始化（Prefill阶段）
    ↓
WAITING → RUNNING（调度）
    ↓
Decode循环（逐token生成）
    ↓
append_token（追加新token）
    ↓
FINISHED（生成完成或达到max_tokens）
```

---

## 关键设计特点

### 1. **Block分块管理**
- `block_size = 256`：固定大小，便于KV-Cache内存管理
- 通过 `block_table` 映射虚拟block到物理block

### 2. **序列计数器**
```python
counter = count()  # 全局自增计数器
```
- 每个序列有唯一ID，便于追踪请求

### 3. **Pickle序列化支持**
```python
def __getstate__(self):
    """支持多进程通信中的状态保存"""
```
- 在分布式推理中，序列可在进程间传递

### 4. **Prompt vs Completion分离**
- `num_prompt_tokens`：原始输入
- `completion_token_ids`：新生成的部分
- 便于追踪生成过程

---

## 与其他模块的关系

```
Sequence
    ↓
Scheduler（调度）→ BlockManager（KV-Cache管理）
    ↓
ModelRunner（模型推理）
```

---

## 使用场景示例

```python
# 1. 创建序列
prompt = [101, 2054, 2003]  # "What is..."
seq = Sequence(prompt, sampling_params)

# 2. 推理过程中追加token
seq.append_token(7592)  # 生成新token

# 3. 检查状态
if seq.num_completion_tokens >= seq.max_tokens:
    seq.status = SequenceStatus.FINISHED

# 4. 获取结果
result = seq.completion_token_ids
```

---

## 性能特征

- **内存占用**：O(n)，其中n为序列长度
- **查询复杂度**：O(1)（属性访问）
- **block操作**：O(1)（block_size固定）
