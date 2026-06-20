# vLLM — 高吞吐 LLM 推理引擎

> **预览**：VSCode `Cmd+Shift+V`，KaTeX 渲染。

> 读完这一章你应该能回答：
> 1. vLLM 是什么，解决什么问题？
> 2. **PagedAttention** 是什么？为什么它把吞吐提了 10×？
> 3. Continuous batching 又是什么？
> 4. 三个核心 API 怎么用？
> 5. 为什么 RL 训练（特别是 GRPO）必须用 vLLM？
> 6. 在 vaEvas 项目里，vLLM 用在哪两个地方？

## 1. vLLM 是什么 + 解决什么问题

**vLLM** = "virtual large language model"，加州伯克利 Sky Lab 出品的**开源 LLM 推理引擎**。

它解决的核心问题是 **朴素 LLM 推理（HuggingFace `model.generate()`）的性能瓶颈**：

| 朴素推理的问题 | 现象 |
|---|---|
| KV cache 内存碎片化 | 60–80% 显存浪费 |
| 不能高效 batch 不同长度的请求 | 长请求拖死整个 batch |
| 没法在线服务多个并发请求 | 一个一个串行处理 |
| 长 context 严重拖慢 | 注意力是 $O(n^2)$ |

vLLM 通过 **PagedAttention** + **Continuous Batching** 两个核心创新，把吞吐量提升 **10–24×**（来自 vLLM 论文的实测）。

### 数字对比（A100 80G, Llama-7B）

| 引擎 | 吞吐 (tokens/s) | 内存效率 | 并发请求数 |
|---|---|---|---|
| HuggingFace `model.generate()` | ~1000 | 低 | ~4 |
| FasterTransformer | ~3000 | 中 | ~20 |
| **vLLM** | **~24000** | **高** | **~40** |

10× 不是夸张，是真实数字。这就是为什么 **vLLM 几乎是所有生产 LLM 服务的默认选择**。

## 2. PagedAttention — 把 KV cache 当虚拟内存管理

这是 vLLM 最重要的创新。理解了它，你就理解了 vLLM 大半。

### 2.1 什么是 KV cache，为什么需要

LLM 自回归生成时，每生成下一个 token，attention 都要回看**前面所有 token 的 key 和 value**。

如果每生成一个 token 都重新算所有历史 token 的 K/V → 越往后越慢，每步要算 $O(n)$ 次注意力点积，总复杂度 $O(n^2)$。

解决：把每个 token 的 K/V **缓存在 GPU 显存**里，下次直接读。这就是 KV cache。

```
生成第 1 个 token: 算 K_1, V_1 → 存进 cache
生成第 2 个 token: 读 K_1, V_1, 算 K_2, V_2 → 存
生成第 3 个 token: 读 K_1..K_2, V_1..V_2, 算 K_3, V_3 → 存
...
```

KV cache 让生成复杂度从 $O(n^2)$ 降到 $O(n)$。**所有 LLM 推理引擎都有 KV cache**。

### 2.2 朴素 KV cache 的问题：内存碎片

最直观的实现：给每个请求**预分配一块连续显存**，大小是 `max_seq_length`（比如 2048）。

但实际请求往往**远没用满**：

```
请求 A (max=2048, 实际生成 200 tokens):
[ 用了 200 ████ | 浪费 1848 ░░░░░░░░░░░░░░░░░ ]

请求 B (max=2048, 实际生成 1500 tokens):
[ 用了 1500 █████████████ | 浪费 548 ░░░░░ ]

请求 C (max=2048, 实际生成 50 tokens):
[ 用了 50 █ | 浪费 1998 ░░░░░░░░░░░░░░░░░░ ]
```

→ **大量"内部碎片"**，平均 60-80% 显存浪费。能并发的请求数被严重压制。

### 2.3 PagedAttention 的解法：参考操作系统

参考操作系统怎么解决进程内存碎片化：
- 把物理内存分成**固定大小的页 (page)**
- 进程拿到的是**逻辑上连续、物理上不连续**的内存
- 通过**页表**做地址映射

PagedAttention 干同一件事：
- 把 KV cache 分成**固定大小的 block**（默认每个 block 存 16 个 token 的 K/V）
- 每个请求拿到的是一组**物理上分散的 block**
- 通过 **block table** 索引

```
物理 GPU 显存（按 block 编号）:
[Block 0] [Block 1] [Block 2] [Block 3] [Block 4] [Block 5] ...

请求 A 的 block table: [0, 2, 5, 7]   ← 4 个 block = 64 tokens
请求 B 的 block table: [1, 3, 4, 6, 8, 9, ...]
请求 C 的 block table: [10]            ← 1 个 block = 16 tokens
```

attention 计算时，按 block table 查实际物理地址，跨 block 拼接。

### 2.4 PagedAttention 带来什么

| 指标 | 朴素 | PagedAttention |
|---|---|---|
| 内存碎片 | 60-80% | **< 4%** |
| 并发请求数 | 4-8 | **40+** |
| 前缀共享 | ❌ 无 | ✅ 多请求共享同一段 prompt 的 block |

**前缀共享 (prefix caching)** 特别值得讲：如果两个请求的 prompt 前 N 个 token 完全一样，它们的 block table **前 N/16 个条目可以指向同一组物理 block** —— 这部分 K/V 算一次就行。

### 2.5 你的实务直觉

不用懂 PagedAttention 的源码实现，但要建立这些直觉：

- ✅ `max_model_len` 设大一些**不浪费**显存，vLLM 只为实际用到的 token 分配 block
- ✅ 多个请求 prompt 相同时（比如 GRPO 的 N=8 个 completion），vLLM **自动共享** prompt 部分的 block
- ✅ 相同 GPU 显存，vLLM 能跑 **5-10× 的并发请求数**

## 3. Continuous Batching — 另一个核心

### 3.1 朴素 batching 的问题

静态 batch：等 N 个请求都来，凑一个 batch，一起进 forward。问题：

```
请求 1: ┃▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶▶ 生成 30 个 token
请求 2: ┃▶▶ 生成 2 个 token (但还是要等到请求 1 完)
请求 3: ┃▶▶▶▶▶▶▶▶▶▶ 生成 10 个 token
请求 4: ┃▶▶▶▶ 生成 4 个 token

时间: ──────────────────────────────────►
                         ↑ 请求 2/3/4 早就完了，但还在占 GPU slot
```

最长的请求没完成，整个 batch 占着 GPU，**短请求生成完了也不能 evict**。GPU 严重空闲。

### 3.2 Continuous Batching 的做法

**每生成一个 token，就重新看一下 batch**：
- 有请求完成？立刻 evict（把它从 batch 里移除）
- 有新请求等着？立刻 admit（把它加入 batch）

batch 内容**每步都在变**。GPU 几乎不空闲。

```
时间步 t=1:  Batch = [1, 2, 3, 4]
时间步 t=3:  请求 2 完成 → evict, 新请求 5 进 batch → Batch = [1, 3, 4, 5]
时间步 t=5:  请求 4 完成 → evict, 新请求 6 进 batch → Batch = [1, 3, 5, 6]
...
```

### 3.3 PagedAttention + Continuous Batching = 暴力组合

- **PagedAttention** → KV cache 不浪费 → 单卡能塞更多请求的 K/V
- **Continuous Batching** → 计算单元不浪费 → GPU 时刻满载

两者结合 → 极致吞吐。

## 4. 三个核心 API

vLLM 的 Python API 就这三个核心 class/function。掌握这三个，离线推理你就会了。

```python
from vllm import LLM, SamplingParams

# 1. LLM(): 加载模型，一次性
llm = LLM(
    model="Qwen/Qwen2.5-Coder-7B-Instruct",   # HF 路径或本地路径
    dtype="bfloat16",                          # 标准精度
    tensor_parallel_size=1,                    # 单卡; 多卡时填 2 / 4 / 8
    max_model_len=8192,                        # 最大支持的序列长度
    gpu_memory_utilization=0.9,                # 用 90% GPU 显存
)

# 2. SamplingParams: 采样参数
params = SamplingParams(
    temperature=0.7,                           # 0=确定性，>0=随机
    top_p=0.95,
    max_tokens=2048,                           # 单 completion 最多生成多少 token
    n=8,                                       # 同一 prompt 生成 N 个 completion (GRPO 关键)
    stop=["<|im_end|>"],                       # 遇到这些 token 就停
)

# 3. generate(): 批量推理
prompts = [
    "Write a Verilog-A comparator model.",
    "Generate a sample-and-hold testbench.",
    ...,
]
outputs = llm.generate(prompts, params)

# outputs 是 List[RequestOutput], 长度等于 prompts 数量
for output in outputs:
    print(f"Prompt: {output.prompt[:50]}...")
    # output.outputs 是 List[CompletionOutput], 长度等于 SamplingParams.n
    for i, completion in enumerate(output.outputs):
        print(f"  [{i}] {completion.text[:100]}...")
        print(f"      finish_reason: {completion.finish_reason}")  # stop / length / ...
```

**就这三个 API**。vLLM 的所有功能都建在它们之上。

## 5. 三种典型使用模式

### 5.1 模式 A：离线批量推理（eval 阶段）

最简单的模式。直接调 `generate()`。

```python
# Phase 4 eval 的典型代码
from vllm import LLM, SamplingParams

llm = LLM(model="path/to/sft-or-grpo-ckpt", dtype="bfloat16")
params = SamplingParams(temperature=0.0, max_tokens=2048)   # eval 用确定性输出

eval_prompts = load_eval_prompts("train/data/eval/")
outputs = llm.generate(eval_prompts, params)

results = []
for output, prompt in zip(outputs, eval_prompts):
    va_code = parse_answer(output.outputs[0].text)
    if openvaf_compile(va_code):
        if evas_simulate(va_code, tb):
            results.append("correct")
        else:
            results.append("compile_only")
    else:
        results.append("fail")
```

**省的时间**：50 条 eval，HuggingFace generate 要半小时，vLLM 大约 3 分钟。

### 5.2 模式 B：OpenAI 兼容的 HTTP server（部署阶段）

启动一个 web server，提供 OpenAI 风格的 REST API。

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --dtype bfloat16 \
    --port 8000 \
    --tensor-parallel-size 2
```

启动后，客户端可以用 OpenAI SDK 直接调：

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-Coder-7B-Instruct",
    messages=[
        {"role": "system", "content": "You are a Verilog-A engineer."},
        {"role": "user", "content": "Generate a comparator."},
    ],
    temperature=0.7,
)
print(response.choices[0].message.content)
```

**适用场景**：训练完模型分享给同事用 / 接入 ARIS 等下游工具 / 长期跑评测服务。

### 5.3 模式 C：嵌入式 RL rollout（GRPO 阶段）

verl 这种 RL 框架不调 HTTP server，而是把 vLLM **直接嵌入训练进程**：

```python
# verl 内部大致这样用 vLLM (简化伪代码)
class GRPOTrainer:
    def __init__(self):
        self.policy_model = ...                          # 训练用的 transformer
        self.rollout_engine = LLM(model="...", ...)      # vLLM 推理引擎

    def step(self, prompts):
        # ① rollout: vLLM 快速生成 N 个 completion / prompt
        outputs = self.rollout_engine.generate(prompts, params_n8)

        # ② compute reward
        rewards = [reward_fn(o) for o in outputs]

        # ③ update policy (用 policy_model 算 advantage, 梯度更新)
        loss = compute_grpo_loss(outputs, rewards, ...)
        loss.backward()

        # ④ ★ 同步 weights: policy_model → rollout_engine
        sync_weights(self.policy_model, self.rollout_engine)
```

**关键是第 ④ 步**：训练完一步后，新的权重必须同步给 vLLM rollout 引擎 —— 否则下一步 rollout 用的还是旧模型。这是 verl/slime/OpenRLHF 等框架的核心工程难点。

## 6. 为什么 RL 训练 **必须** 用 vLLM

GRPO 每一步训练流程：

```
for prompt in batch:
    Sample N=8 completions      ← rollout 阶段 (瓶颈!)
    Compute rewards
    Compute advantage
    Update policy
```

**rollout 是吞吐瓶颈**。具体算一下：

| 假设 | 数值 |
|---|---|
| Group size | $N = 8$ |
| 平均 completion 长度 | 1500 tokens |
| 每个 prompt 要生成的总 tokens | $8 \times 1500 = 12000$ |

| 引擎 | 单 prompt rollout 时间 | 200 步 × batch 16 总时间 |
|---|---|---|
| HF generate | 12000 / 1000 ≈ **12 秒** | 200 × 16 × 12 = **10.7 小时** |
| vLLM | 12000 / 24000 ≈ **0.5 秒** | 200 × 16 × 0.5 = **27 分钟** |

→ **用 vLLM 训练 27 分钟；不用 vLLM 训练 10 小时**。

这就是为什么 verl / slime / OpenRLHF 全部把 vLLM 当**核心组件**。

## 7. 在 vaEvas 项目中的具体使用

按 Phase 划分：

| 阶段 | 用 vLLM 干什么 | 备注 |
|---|---|---|
| Phase 2 SFT | ❌ 不用 | SFT 是训练，不需要快速推理 |
| Phase 3 GRPO | ✅ 通过 verl 间接用 | rollout 引擎 |
| Phase 4 Eval | ✅ 直接用 | 跑 held-out 评测 |
| 后续部署 | ✅ OpenAI server 模式 | 给同事提供 API |

**关键时间投资**：在 Phase 3 之前必须把 vLLM 的 §4 三个 API + §5 三种模式吃透。不然到 GRPO 阶段配置 verl 时会一头雾水。

## 8. 常见坑

| 现象 | 原因 | 解决 |
|---|---|---|
| OOM 启动 | `gpu_memory_utilization=0.9` 太激进 | 降到 0.8 或 0.7 |
| OOM 推理中 | `max_model_len` 太大 | 按真实需求设（不要默认 32k）|
| 模型加载失败 | vLLM 不支持该架构 | 查 [vLLM 支持模型列表](https://docs.vllm.com.cn/en/latest/models/supported_models.html)；Qwen2.5 / Llama / Mistral / DeepSeek 主流都支持 |
| 多卡报错 | `tensor_parallel_size` 和实际 GPU 数不匹配 | 设成实际 GPU 数（A100×2 → 2）|
| 生成乱码 | tokenizer / model 配置不一致 | 用同一个 `model_path` 加载 tokenizer，不要混 |
| 量化模型加载慢 | AWQ / GPTQ 模型 | 加 `quantization="awq"` 参数 |
| GRPO rollout 同步慢 | 训练完每步都要同步 weights | verl 内部用 NCCL，配好就行；自己实现要注意 |
| Speculative decoding 报错 | draft model 配置 | 不熟先关 |

## 9. 学习路径（按 vaEvas Phase 推荐）

### 现在（SFT 学习阶段，可以慢慢来）

**最小可用学习量（90 分钟）**：
1. **30 分钟** — 读 [vLLM Quickstart](https://docs.vllm.com.cn/en/latest/getting_started/quickstart.html)，跟着把"三个核心 API"用一遍
2. **30 分钟** — 跑 vLLM 自带的 [offline_inference example](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/offline_inference.py)
3. **30 分钟** — 启一个 OpenAI server，用 curl 或 openai SDK 调用一次

### Phase 3 GRPO 开始之前（必须完成）

4. 读 [PagedAttention 论文](https://arxiv.org/abs/2309.06180)（30 分钟，看核心思路即可）或对应的技术博文
5. 看 verl 配置里 `rollout.name: vllm` 部分长什么样（30 分钟，§ verl 笔记 §5）

### Phase 4 Eval 时（实务）

6. 自己写一个 vLLM-based eval 脚本（参考本文 §5.1）；处理：批量 prompt → 生成 → 解析 `<answer>` → EVAS 验证

## 10. 本章 take-aways

1. **vLLM 是业界标准的高吞吐 LLM 推理引擎**，比 HF generate 快 10–24×。
2. **核心创新**：**PagedAttention**（KV cache 当虚拟内存，消除碎片 60–80% → < 4%） + **Continuous Batching**（动态进出 batch，GPU 不空闲）。
3. **三个核心 API**：`LLM()` 加载，`SamplingParams()` 配置，`generate()` 批量生成 —— 离线推理你就这点东西要学。
4. **三种使用模式**：离线推理（eval）/ OpenAI server（部署）/ 嵌入式（RL rollout via verl）。
5. **GRPO 必用 vLLM**：rollout 是 RL 训练的瓶颈，不用 vLLM 训练时间 ×10。
6. 在 vaEvas 项目里，vLLM 用在 **Phase 4 eval（直接调）** 和 **Phase 3 GRPO（verl 间接调）**。

## 11. 下一步

读完本章后:
- 进 [`verl.md`](./verl.md) → 学**怎么把 vLLM 接进 GRPO 训练**
- 回 [`../sft/02b_real_world_reference.md`](../sft/02b_real_world_reference.md) → SFT 学习继续
- 回 [`../REFERENCES.md`](../REFERENCES.md) → 整体路线图
