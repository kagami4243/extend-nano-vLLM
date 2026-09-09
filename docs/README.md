# nano-vllm Engine 模块文档索引

## 📚 文档列表

### 总体架构
- **[ENGINE_OVERVIEW.md](ENGINE_OVERVIEW.md)** - 引擎模块总体架构和流程介绍

### 各模块详细文档

#### 0️⃣ 量化专题
- **[quantization.md](quantization.md)**
  - W4A16 weight-only 与 FP8 W8A8 的区别
  - 量化布局、执行路径、测试和当前限制

#### 0️⃣.1 MoE grouped kernel
- **[moe_kernel.md](moe_kernel.md)**
  - vLLM-style Triton grouped GEMM 的最小实现
  - EP 语义、验证方式与当前限制

#### 0️⃣.2 FP8 KV cache
- **[fp8_kv_cache.md](fp8_kv_cache.md)**
  - 单卡 FP8 page 存储与 Triton paged decode attention
  - vLLM 参考实现、显存收益与硬件限制

#### 1️⃣ 序列管理模块
- **[engine_sequence.md](engine_sequence.md)**
  - `Sequence` 类定义
  - `SequenceStatus` 状态管理
  - 序列生命周期（WAITING → RUNNING → FINISHED）
  - Block分块机制
  - 关键属性和方法

#### 2️⃣ 内存管理模块  
- **[engine_block_manager.md](engine_block_manager.md)**
  - `Block` 块结构
  - `BlockManager` 内存管理
  - KV-Cache 块分割和分配
  - **块重用机制**（多序列共享）
  - 引用计数管理
  - 四大核心方法：`allocate`, `can_append`, `may_append`, `deallocate`

#### 3️⃣ 调度引擎模块
- **[engine_scheduler.md](engine_scheduler.md)**
  - `Scheduler` 调度器
  - **两阶段调度**：Prefill + Decode
  - 请求队列管理（waiting/running）
  - **抢占机制**（内存竞争处理）
  - 调度约束和策略
  - 完整推理流程

#### 4️⃣ 推理执行模块
- **[engine_model_runner.md](engine_model_runner.md)**
  - `ModelRunner` 推理引擎
  - 模型初始化和加载
  - **KV-Cache自动分配**
  - Prefill 数据预处理
  - Decode 数据预处理
  - **CUDA图捕获优化**
  - **多卡张量并行**
  - 完整推理管道

#### 5️⃣ 用户接口模块
- **[engine_llm_engine.md](engine_llm_engine.md)**
  - `LLMEngine` 高层接口
  - `generate()` 方法（用户主要调用）
  - 推理循环实现
  - 吞吐量统计
  - 多进程生命周期管理
  - 完整端到端流程

---

## 🔍 快速导航

### 按场景查找

**想了解整个推理流程？**
→ 从 [ENGINE_OVERVIEW.md](ENGINE_OVERVIEW.md) 开始

**想深入某个模块？**
- 序列如何表示 → [engine_sequence.md](engine_sequence.md)
- 内存如何管理 → [engine_block_manager.md](engine_block_manager.md)
- 请求如何调度 → [engine_scheduler.md](engine_scheduler.md)
- 模型如何推理 → [engine_model_runner.md](engine_model_runner.md)
- 如何使用API → [engine_llm_engine.md](engine_llm_engine.md)

**想理解核心优化？**
- 块重用机制 → [engine_block_manager.md](engine_block_manager.md) 的 "块重用机制"
- 抢占调度 → [engine_scheduler.md](engine_scheduler.md) 的 "抢占机制"
- CUDA图优化 → [engine_model_runner.py](engine_model_runner.md) 的 "capture_cudagraph"
- 两阶段调度 → [engine_scheduler.md](engine_scheduler.md) 的 "两阶段推理"

**想学习代码实现？**
→ 按推荐阅读顺序：Sequence → BlockManager → Scheduler → ModelRunner → LLMEngine

---

## 📊 模块关系图

```
┌────────────────────────────────────────┐
│     LLMEngine (engine_llm_engine.md)   │
│        用户调用 generate()              │
└────────────┬───────────────────────────┘
             │
        ┌────┴────┐
        │          │
        ▼          ▼
┌──────────────┐  ┌──────────────────┐
│ Scheduler    │  │ ModelRunner      │
│(scheduler.md)│  │(model_runner.md) │
└──────────────┘  └──────────────────┘
        │                  │
        └────────┬─────────┘
                 │
        ┌────────▼──────────┐
        │  BlockManager      │
        │(block_manager.md)  │
        └────────────────────┘
        
        ┌────────────────────┐
        │   Sequence         │
        │(sequence.md)       │
        └────────────────────┘
```

---

## 🚀 学习路径

### 初级：理解概念
1. 阅读 [ENGINE_OVERVIEW.md](ENGINE_OVERVIEW.md)
2. 了解 [engine_sequence.md](engine_sequence.md)
3. 理解基本推理流程

### 中级：掌握各模块
1. 深入 [engine_block_manager.md](engine_block_manager.md) - 内存管理
2. 学习 [engine_scheduler.md](engine_scheduler.md) - 调度逻辑
3. 理解 [engine_model_runner.md](engine_model_runner.md) - 推理执行

### 高级：精通优化技术
1. 块重用和引用计数
2. 抢占和调度策略
3. CUDA图捕获和张量并行
4. 完整推理流程中的性能瓶颈

---

## 💡 关键概念速查

| 概念 | 在哪里 | 说明 |
|------|--------|------|
| Sequence | sequence.md | 请求的抽象表示 |
| Block | block_manager.md | KV-Cache的分割单元 |
| 块重用 | block_manager.md | 多请求共享相同块 |
| 引用计数 | block_manager.md | 安全释放共享块 |
| Prefill | scheduler.md | 处理完整提示词 |
| Decode | scheduler.md | 逐token生成 |
| 抢占 | scheduler.md | 内存竞争时释放低优先序列 |
| CUDA图 | model_runner.md | 预录制计算图加速 |
| 张量并行 | model_runner.md | 多卡分布式推理 |
| 吞吐量 | llm_engine.md | tokens/秒 |

---

## 🔗 文件关联

### Sequence.py 相关：
- 使用场景：BlockManager, Scheduler, ModelRunner
- 数据流向：text → tokenize → Sequence
- 生命周期：WAITING → RUNNING → FINISHED

### BlockManager.py 相关：
- 被调用者：Scheduler (allocate, deallocate, can_append, may_append)
- 关键数据：free_block_ids, used_block_ids, hash_to_block_id
- 性能优化：块重用、引用计数

### Scheduler.py 相关：
- 依赖：BlockManager, Sequence
- 主要任务：schedule(), preempt(), postprocess()
- 策略：FIFO + 抢占

### ModelRunner.py 相关：
- 依赖：Sequence, Config
- 执行：Prefill和Decode
- 优化：CUDA图、张量并行

### LLMEngine.py 相关：
- 集成：所有模块
- 接口：generate(), add_request(), step()
- 用户调用入口

---

## 📝 阅读建议

### 快速概览（15分钟）
1. 读 ENGINE_OVERVIEW.md 的"完整推理流程"部分
2. 看模块关系图
3. 浏览各模块的核心方法

### 深度学习（1-2小时）
1. 按推荐顺序读5个模块文档
2. 关注每个模块的"工作流程"部分
3. 理解数据如何在模块间流转

### 代码实现（1小时以上）
1. 对照文档读源代码
2. 追踪一个请求的完整生命周期
3. 理解性能优化的实现细节

---

## ❓ 常见问题

**Q: 应该先学哪个模块？**
A: Sequence! 它是基础数据结构，后续所有模块都围绕它。

**Q: BlockManager的块重用真的那么重要吗？**
A: 是的！在批处理相同前缀的请求时，可节省20-40%的显存。

**Q: 为什么要分Prefill和Decode两个阶段？**
A: 因为它们的计算特性不同：
- Prefill：计算密集，需要全并行
- Decode：内存密集，低延迟优先

**Q: CUDA图是什么，为什么需要？**
A: 预先录制GPU计算图，避免频繁的kernel launch和CPU-GPU同步，Decode性能提升20-30%。

---

## 📚 相关资源

### 论文
- vLLM: Efficient Memory Management for Large Language Model Serving
- Paged Attention for LLM Serving
- Flash Attention

### 项目
- [vLLM官方](https://github.com/vllm-project/vllm)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)

### 更多文档
- PyTorch 官方文档
- CUDA 编程指南
- Distributed Training 最佳实践

---

## 📄 文档版本

| 版本 | 日期 | 说明 |
|------|------|------|
| 1.0 | 2025-03-07 | 初始版本，包含5个模块 |

---

祝你学习愉快！有问题欢迎提issue或讨论 😊
