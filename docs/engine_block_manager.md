# block_manager.py - KV-Cache块管理模块

## 文件作用

该模块实现了 **KV-Cache 内存管理**，采用 **块（Block）** 粒度进行管理。通过引入**块重用机制**和**智能哈希缓存**，实现了多个序列之间的 KV-Cache 共享，大幅提升显存利用效率。

---

## 核心概念

### Block（块）

KV-Cache 的基本管理单位，固定大小为 256 个 token。

```python
class Block:
    block_id: int       # 全局唯一块ID
    ref_count: int      # 引用计数（共享机制）
    hash: int          # 块内容哈希值（用于去重）
    token_ids: list    # 块对应的token IDs
```

**设计意义**：
- 固定大小便于内存对齐和高效GPU操作
- 引用计数支持多序列共享同一块
- 哈希值用于快速定位相同内容的块

---

## BlockManager 类

### 初始化

```python
def __init__(self, num_blocks: int, block_size: int):
    self.block_size = 256              # 固定块大小
    self.blocks: list[Block]           # 所有块的数组
    self.hash_to_block_id: dict        # 哈希值→块ID映射
    self.free_block_ids: deque         # 空闲块队列
    self.used_block_ids: set           # 已使用块集合
```

### 内存状态

```
总块数 = num_blocks
    ├─ free_block_ids（空闲块）
    └─ used_block_ids（已使用块）
```

---

## 核心方法

### 1. compute_hash - 哈希计算

```python
@classmethod
def compute_hash(cls, token_ids: list[int], prefix: int = -1) -> int:
    """
    计算token序列的哈希值
    
    参数：
    - token_ids: 当前块的token IDs
    - prefix: 前一个块的哈希值（支持链式哈希）
    
    返回：xxhash64哈希值（用于快速比对）
    """
```

**例子**：
```python
# 块0: [101, 2054, 2003, ...]  → hash0
# 块1: [4605, 3010, ...]       → hash1 = compute_hash([...], prefix=hash0)
```

---

### 2. allocate - Prefill阶段分配

处理提示词阶段的KV-Cache分配，支持**块重用**。

```python
def allocate(self, seq: Sequence):
    """
    为序列分配KV-Cache块
    
    流程：
    1. 按block_size分割序列token
    2. 对每个块计算哈希值
    3. 查询哈希表是否存在相同块
       ├─ 存在（cache hit）: 增加ref_count
       └─ 不存在（cache miss）: 分配新块
    """
```

**工作细节**：

| 情况 | 处理 | ref_count |
|------|------|----------|
| 哈希命中 + 块存在 | 增加引用 | +1 |
| 哈希命中 + 块释放 | 重新分配 | reset |
| 哈希未命中 | 分配新块 | 初始化 |

**例子**：
```
多个请求的提示词前缀相同：
请求1: [开始] [中国] [北京] [天气] ...
请求2: [开始] [中国] [北京] [中文] ...

块0: [开始] [中国] [北京]  ← 共享此块（ref_count=2）
块1: [天气] ...            ← 请求1独占
块2: [中文] ...            ← 请求2独占
```

---

### 3. can_append - 解码前检查

```python
def can_append(self, seq: Sequence) -> bool:
    """
    检查是否有足够空间为序列追加新token
    
    返回：
    - True: 有1个空闲块
    - False: 无空闲块（需要抢占）
    """
```

**逻辑**：
```python
# 只有当序列需要新块时才返回True
需要新块 = (len(seq) % block_size == 1)
```

---

### 4. may_append - Decode阶段追加

处理每个解码步骤中的KV-Cache更新。

```python
def may_append(self, seq: Sequence):
    """
    为序列追加一个新token的KV-Cache
    
    三种情况：
    1. 跨块边界（len % block_size == 1）
       → 分配新块
    2. 块满（len % block_size == 0）
       → 更新块哈希和映射
    3. 块未满（其他）
       → 等待下一个token
    """
```

**例子**：
```
block_size = 4
序列长度演化：

len=3: [A, B, C, _]           → 块未满，等待
len=4: [A, B, C, D]           → 块满，计算哈希
len=5: [A, B, C, D], [E, _, _, _]  → 新块需要分配
```

---

### 5. deallocate - 内存释放

```python
def deallocate(self, seq: Sequence):
    """
    释放序列占用的所有块
    
    流程：
    1. 遍历seq.block_table的所有块
    2. 减少ref_count
    3. ref_count==0时，归还给free_block_ids
    """
```

**引用计数逻辑**：
```
块A被两个序列共享（ref_count=2）
序列1完成 → deallocate() → ref_count=1 → 块仍存留
序列2完成 → deallocate() → ref_count=0 → 块回收
```

---

## 工作流程

```
┌─────────────────────────────────────────┐
│          Prefill（提示词处理）           │
├─────────────────────────────────────────┤
│  allocate(seq)                          │
│  ├─ 分块                                 │
│  ├─ 计算哈希                             │
│  └─ 执行共享（如果哈希匹配）              │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│        Decode（逐token生成）             │
├──────────────────────────────────────────┤
│  while not finished:                     │
│    ├─ can_append(seq)? → 需要新块？      │
│    ├─ may_append(seq)  → 追加token KV   │
│    └─ append_token()   → 生成新token    │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│        完成（释放内存）                   │
├──────────────────────────────────────────┤
│  deallocate(seq)                        │
│  ├─ 遍历block_table                      │
│  ├─ 减少ref_count                       │
│  └─ ref_count==0 → 回收块                │
└──────────────────────────────────────────┘
```

---

## 关键设计亮点

### 1. **块重用机制**

通过哈希值快速识别相同内容的块，支持多序列共享。

```
优势：
✓ 减少GPU显存占用
✓ 特别对于批量请求的相同前缀收益大
✓ 支持动态共享和释放
```

### 2. **引用计数**

每个块维护引用计数，支持安全的共享和释放。

```python
块被n个序列使用 → ref_count = n
序列1释放 → ref_count -= 1
...
最后一个序列释放 → ref_count = 0 → 块回收
```

### 3. **两阶段KV-Cache管理**

- **Prefill**：大批量分配，支持共享
- **Decode**：逐token追加，增量式更新

---

## 内存使用示例

```python
# 配置
num_blocks = 1000          # 1000个块
block_size = 256           # 每块256个token
总容量 = 1000 * 256 = 256K tokens

# Prefill阶段
请求1: 提示词1000个token
  └─ 占用 4个块（256*4=1024）

请求2: 提示词800个token（前256个与请求1相同）
  └─ 占用 3个块（块0共享，块1-2独占）
  └─ 实际增加显存 = 2个块（512 tokens）

# 内存效率提升
不共享: 1000 + 800 = 1800 tokens
有共享: 1024 + 512 = 1536 tokens
节省: (1800-1536)/1800 = 14.7%
```

---

## 与Scheduler的协作

```python
scheduler.schedule():
    ├─ block_manager.can_allocate(seq)      # Prefill检查
    ├─ block_manager.allocate(seq)          # 分配块
    └─ block_manager.can_append(seq)        # Decode检查
    
    若can_append失败：
    └─ preempt(seq) → deallocate() → 释放块
```

---

## 性能特征

| 操作 | 复杂度 | 说明 |
|------|--------|------|
| 分配块 | O(num_blocks/block_size) | 线性遍历序列的块 |
| 哈希查询 | O(1) | 哈希表查找 |
| 释放块 | O(len(block_table)) | 遍历块表 |
| 引用计数 | O(1) | 简单加减 |
