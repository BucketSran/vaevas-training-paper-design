# 02 — Data and Format

> **预览**：VSCode `Cmd+Shift+V`，KaTeX 渲染。
>
> **本章公式不多**，主要是工程细节。如果你来自 [`01_foundations.md`](./01_foundations.md)，那一章告诉你"SFT 在最大化什么"；本章告诉你**"从 `.jsonl` 数据文件，到 GPU 上一个 batch 的张量，中间发生了什么"**。

> 读完这一章，你应该能回答：
> 1. 数据从 `.jsonl` 到 GPU 张量经历了哪些步骤？
> 2. **prompt mask 是什么？为什么 prompt 也算 loss 会废？**
> 3. Qwen2.5 的 chat template 长什么样？写错了会怎样？
> 4. Verilog-A 代码经过 tokenizer 会变成什么？哪些"坑"？
> 5. `padding` 和 `packing` 怎么选？长度怎么定？
> 6. 我们的 `<think>/<answer>` trajectory 怎么塞进 chat template？

---

## 1. 数据从 `.jsonl` 到 GPU 张量：完整流水线

先看一遍**全流程**，后面每节再展开。

```
你写的 .jsonl 一行                    < —— 人类视角，干净的 JSON
        │
        │  pipelines/pack_sft.py
        ▼
{"prompt": "...", "completion": "..."}
        │
        │  apply_chat_template()       <—— Qwen 期望的特殊格式
        ▼
"<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>"
        │
        │  tokenizer.encode()           <—— 文本 → token ID
        ▼
[151644, 8948, 198, ...,151645, 198, ...]       <—— input_ids
        │
        │  build labels                 <—— 关键！prompt 部分设成 -100
        ▼
labels   = [-100, -100, -100, ..., 151645, 198, 6885, ...]   <—— 同长度，prompt 处为 -100
        │
        │  pad to batch / pack samples  <—— 装进同一 batch
        ▼
input_ids: tensor[B, L]
labels:    tensor[B, L]
attn_mask: tensor[B, L]
        │
        │  model.forward(input_ids, labels=labels)
        ▼
loss (scalar)                                <—— 自动跳过 labels = -100 的位置
```

四个关键变换：
1. **Chat template**：原始 `(prompt, completion)` → 带特殊 token 的字符串
2. **Tokenization**：字符串 → token ID 序列
3. **Label 构造 + masking**：prompt 部分的 label 标成 `-100`（告诉 loss "这部分不算"）
4. **Pad / pack 到 batch**：把不同长度的样本装进同一个 tensor

第 2 至第 4 步**任何一个写错都会让 SFT 失败**。下面逐个展开。

---

## 2. Chat Template — Qwen2.5 期望什么样的输入

### 2.1 什么是 chat template

预训练时模型见的是**互联网文本流**。指令微调（instruct）模型则被训练去理解**"对话"结构** —— 系统消息 + 用户消息 + 助手回复。

为了让模型知道"我现在是 system / user / assistant"，每家模型厂商在指令微调时**约定了一套特殊 token 来标记角色边界**。这就是 chat template。

**重点**：chat template **不是可选项**。如果你用 Qwen2.5-Coder-7B**-Instruct** 做 SFT，但忘了套 chat template，模型在推理时会困惑（因为推理时套了 template，训练时没套，分布不一致）。

### 2.2 Qwen2.5 的 chat template 具体长什么样

一个完整的 Qwen 对话：

```
<|im_start|>system
You are an expert Verilog-A engineer.<|im_end|>
<|im_start|>user
Generate a Verilog-A model for a sample-and-hold circuit.<|im_end|>
<|im_start|>assistant
<think>
<port>
inputs: vin (electrical), clk (logic)
outputs: vout (electrical)
</port>
<behavior>
sample vin on clk rising edge, hold otherwise
</behavior>
</think>
<answer>
`include "disciplines.vams"
module sample_hold(clk, vin, vout);
  ...
endmodule
</answer><|im_end|>
```

关键边界 token：
| Token | 角色 |
|---|---|
| `<|im_start|>` | "instant message start" —— 一个角色发言开始 |
| `<|im_end|>` | 发言结束 |
| `system` / `user` / `assistant` | 普通字符（不是特殊 token），但放在 `<|im_start|>` 后表示角色 |
| `\n` | 普通换行（不是特殊 token） |

### 2.3 实际代码

`transformers` 库已经把这套规则内建进 tokenizer 的 `apply_chat_template` 方法：

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

messages = [
    {"role": "system", "content": "You are an expert Verilog-A engineer."},
    {"role": "user", "content": "Generate a sample-and-hold model."},
    {"role": "assistant", "content": "<think>...</think><answer>...</answer>"}
]

text = tok.apply_chat_template(
    messages,
    tokenize=False,           # 先看文本（debug 用）
    add_generation_prompt=False  # 训练时 False；推理时 True 会在末尾加 assistant 开头
)
print(text)
# 输出就是上面 2.2 那个字符串
```

**强烈建议** 写好 `pack_sft.py` 之后**手工跑一次 `apply_chat_template` 看输出**，确认 special token 都在正确位置。这一步省 5 分钟，能避免 SFT 跑 3 小时回来发现训练目标完全错位。

### 2.4 写错 chat template 的后果

| 错误 | 症状 |
|---|---|
| 不套 chat template | 训练时模型见的格式 vs 推理时不一致，推理输出乱码或拒答 |
| 角色顺序乱（system 放最后）| 模型混淆指令和回答，可能开始"扮演用户" |
| `<|im_end|>` 漏了 | 模型不知道何时停，生成可能无限延伸 |
| 把 `<|im_start|>` 当普通字符 token 化 | 切碎成 `<`、`|`、`im_start`、`|`、`>` 五个 token，模型完全识别不出来 |

最后一条最隐蔽 —— 见 §3。

---

## 3. Special Tokens：必须被 tokenizer "看见"

### 3.1 什么是 special token

普通 token：`"hello"` → 可能切成 `["he", "llo"]` 之类，由 BPE 算法决定。
特殊 token：`<|im_start|>` → tokenizer 必须把它整体当**一个 token**，分配一个唯一 ID（比如 `151644`），而**不是**按字符串拆分。

**为什么必须整体**：模型在嵌入层是按 token ID 学的。如果 `<|im_start|>` 被拆成 5 个普通字符 token，模型看到的是"小于 + 竖线 + im_start + 竖线 + 大于" —— 跟训练时它学的"那一个表示角色开始的特殊符号"完全不是一个嵌入向量。

### 3.2 Qwen2.5 已有的特殊 token

```python
tok.special_tokens_map
# {'eos_token': '<|im_end|>',
#  'pad_token': '<|endoftext|>',
#  'additional_special_tokens': ['<|im_start|>', '<|im_end|>', ...]}
```

关键几个：
| Special token | 含义 |
|---|---|
| `<|im_start|>` (ID `151644`) | 角色发言开始 |
| `<|im_end|>` (ID `151645`) | 角色发言结束，**也兼做 `eos`** |
| `<|endoftext|>` | 整段文本终止；Qwen 借它当 `pad_token` |

### 3.3 你自己加的 special token（重要！）

我们的 trajectory 用了 `<think>`, `</think>`, `<port>`, `</port>`, `<device>`, `</device>`, `<connection>`, `</connection>`, `<answer>`, `</answer>`。

**默认情况下这些会被 BPE 拆碎**。验证一下：

```python
tok.tokenize("<think>")
# 可能输出: ['<', 'think', '>']  ← 拆成 3 个普通 token
# 这是灾难
```

如果不显式声明这些是特殊 token，模型每次见到 `<think>` 都得"从 3 个无关 token 重新组装语义" —— 既慢又学不稳。

**修复方法**：在 SFT 开始前把它们加入 tokenizer：

```python
new_special_tokens = ["<think>", "</think>",
                      "<port>",  "</port>",
                      "<device>", "</device>",
                      "<connection>", "</connection>",
                      "<answer>", "</answer>"]

num_added = tok.add_special_tokens({
    "additional_special_tokens": new_special_tokens
})
print(f"Added {num_added} tokens, new vocab size: {len(tok)}")

# IMPORTANT: tokenizer 词表变大了，模型的 embedding 也得跟着扩
model.resize_token_embeddings(len(tok))
```

加完之后：

```python
tok.tokenize("<think>")
# ['<think>']  ← 整体一个 token，正确
```

**这一步不做，SFT 会"训得动但效果差很多"**，因为模型需要花精力把拆碎的 tag 重新组装成语义单元。

---

## 4. Prompt Masking — 最重要的工程细节（全章核心）

### 4.1 问题：默认情况下，loss 会算所有 token

`transformers` 的 causal LM loss 函数（基本上就是 `CrossEntropyLoss` 的封装）默认对**每个位置**都算 loss：

$$
L = -\frac{1}{N} \sum_{t=1}^{N} \log P_\theta(\text{token}_t \mid \text{token}_{<t})
$$

其中 $N$ 是序列总长度，**包括 prompt + completion**。

**这就是问题所在**：如果你不告诉 loss 函数"前 $|x|$ 个 token 是 prompt，不要算 loss"，模型会被训练去**生成 prompt 本身** —— 也就是学会"复读用户的输入"。这显然不是我们想要的。

### 4.2 解决方案：`-100` 占位

PyTorch 的 `CrossEntropyLoss` 有个魔法值 `ignore_index=-100`。**任何 label 为 -100 的位置，loss 函数都会跳过**。

所以**正确的 label 构造**长这样：

```
input_ids:  [<|im_start|>, system, \n, ..., <|im_end|>, <|im_start|>, user, \n, ..., <|im_end|>,
             <|im_start|>, assistant, \n, <think>, ..., </think>, <answer>, ..., </answer>, <|im_end|>]
                   ↑                                                              ↑                                       ↑
                  这部分是 prompt (包括 system + user + assistant 标签头)         才开始算 loss              到这里结束

labels:     [-100, -100, -100, ..., -100, -100, -100, -100, ..., -100,
             -100, -100, -100, <think>, ..., </think>, <answer>, ..., </answer>, <|im_end|>]
              ↑                                                              ↑
            prompt 部分全是 -100                                  从 completion 开始用真实 token ID
```

数学上等价于：

$$
L = -\frac{1}{|y|} \sum_{t \in \text{completion positions}} \log P_\theta(\text{token}_t \mid \text{tokens}_{<t})
$$

回到 [`01_foundations.md`](./01_foundations.md) 公式 1 的形式 —— **prompt $x$ 不在求和里**。

### 4.3 实务实现

`trl.SFTTrainer` 默认会自动做这件事，**前提是** 你用 `formatting_func` 或者按"chat 格式"的数据：

```python
# trl 的标准做法
from trl import SFTTrainer

def formatting_func(example):
    # 返回完整对话字符串
    return tok.apply_chat_template(example["messages"], tokenize=False)

trainer = SFTTrainer(
    model=model,
    tokenizer=tok,
    train_dataset=dataset,
    formatting_func=formatting_func,
    data_collator=DataCollatorForCompletionOnlyLM(
        response_template="<|im_start|>assistant\n",  # 标记 completion 开始
        tokenizer=tok
    )
)
```

关键是 `DataCollatorForCompletionOnlyLM` —— 它会扫描每条样本，找到 `response_template` 第一次出现的位置，**把它之前的所有 label 改成 -100**。

### 4.4 不做 prompt masking 会怎样

| 症状 | 解释 |
|---|---|
| 训练 loss 异常低 | 大部分 loss 来自易学的 prompt 复读，掩盖了真正难学的 completion |
| 推理时模型先重复用户问题，再回答 | 训练让它学会了 prompt 的概率分布，自然会先输出 prompt |
| eval 上看似正常但生成质量不对 | loss 数字好看，实际任务能力没学到 |
| 多 epoch 后过拟合特别严重 | prompt 被反复"背"了 N 遍 |

**这个 bug 太常见了**，每个做 SFT 的人都至少踩过一次。**先验证你的 collator 正确，再开始训练**。

### 4.5 验证 collator 是否正确（必做 smoke test）

```python
# Smoke test: 拿一条数据，跑一次 collator，肉眼检查
sample = dataset[0]
formatted = formatting_func(sample)
tokens = tok(formatted, return_tensors="pt")
collated = data_collator([{"input_ids": tokens["input_ids"][0]}])

# 把 labels 跟 input_ids 并排打印
for i, (inp, lbl) in enumerate(zip(collated["input_ids"][0], collated["labels"][0])):
    text = tok.decode([inp.item()])
    label_text = "<MASKED>" if lbl == -100 else tok.decode([lbl.item()])
    print(f"{i:4} | {text:20} | label: {label_text}")
```

**正确的输出应该是**：
- 从 `<|im_start|>` 一直到 `<|im_start|>assistant\n` 这部分 label 全是 `<MASKED>`
- 之后是真实 token，与 input_ids 一致

如果不对，去检查 `response_template` 的 token 化结果是否和 `formatting_func` 输出里的 assistant 标签匹配。

---

## 5. Tokenization for Code — Verilog-A 的特殊情况

### 5.1 BPE 简介（30 秒版）

BPE (Byte Pair Encoding) 是 GPT/Qwen 用的 tokenizer 算法。原理：从字符级开始，反复把**最常见的相邻 pair 合并成一个 token**，直到达到目标词表大小。

结果：
- 常见英文词：1 个 token（如 `"the"`, `"and"`）
- 常见英文短语：1 个 token（`" function"`, `" return"`）
- 罕见词：多 token（`" disciplines.vams" → [" disciplines", ".", "v", "ams"]`）
- 代码符号：通常单 token（`"{"`, `"}"`, `";"`）

### 5.2 看看 Qwen 怎么切 Verilog-A

```python
sample = """`include "disciplines.vams"
module sample_hold(clk, vin, vout);
  input clk, vin;
  output vout;
  electrical vin, vout;
  parameter real gain = 1.0;
endmodule"""

tokens = tok.tokenize(sample)
print(tokens)
# 大致输出（具体取决于 Qwen2.5-Coder tokenizer 实际行为）：
# ['`include', ' "', 'discipl', 'ines', '.v', 'ams', '"\n',
#  'module', ' sample', '_hold', '(', 'clk', ',', ' vin', ',', ' vout', ');',
#  '\n  input clk, vin;', ...]
```

观察几个**对 SFT 重要的现象**：

1. **代码关键词通常是单 token**：`module`, `input`, `output`, `electrical`, `parameter`, `endmodule` 一般都是 1 个 token —— Qwen Coder 词表里见过。
2. **标识符按 underscore/驼峰切**：`sample_hold` → `["sample", "_hold"]` 或 `["sample_hold"]` 取决于词频。
3. **数字常被拆**：`1.0` 可能切成 `["1", ".", "0"]`，对模型的数值理解不太友好。
4. **缩进 + 换行被压缩**：`"\n  input"` 可能是 1 个 token（包含换行 + 空格）。
5. **`` ` `` (反引号) 可能和后面词粘**：``"`include"`` 可能整体 1 个 token。

### 5.3 验证 tokenization 的几条规则

每次写 SFT 前**必做**：

```python
# 1. 你的 trajectory 标签是否被正确识别为 special token
for tag in ["<think>", "</think>", "<port>", "</port>", "<answer>", "</answer>"]:
    ids = tok.encode(tag, add_special_tokens=False)
    print(f"{tag:15} → {ids}  (len={len(ids)})")
    # 期望: len = 1，即只占 1 个 token

# 2. 一条完整样本的 token 长度分布
lengths = [len(tok.encode(format_sample(s))) for s in dataset]
import statistics
print(f"min={min(lengths)}, median={statistics.median(lengths)}, "
      f"p95={sorted(lengths)[int(0.95 * len(lengths))]}, max={max(lengths)}")
```

第 2 条决定 §6 的 `max_seq_length` 怎么选。

### 5.4 Verilog-A 特殊"坑"

| 现象 | 后果 | 缓解 |
|---|---|---|
| `` `include`` 反引号语法 Qwen 没见过 | 模型可能在生成时改用 `#include`（C 风格）| SFT 数据多见几次就纠正 |
| 双反斜杠注释 `// ...` 在很多语言通用 | 还好 | 无需特殊处理 |
| `@(initial_step)` / `@(timer(...))` 这种 Verilog-A 特有的 event control | 训练数据少时模型容易乱用 | 收集尽量多种用法 |
| 长字符串 `"some_signal_name_that_is_very_long"` 被切多 token | 占用 context | 命名风格保持一致 |

---

## 6. Length Budgeting — `max_seq_length` 怎么选

### 6.1 长度的两端权衡

| `max_seq_length` 太小 | `max_seq_length` 太大 |
|---|---|
| Truncation 砍掉 trajectory 或 answer 末尾 | 显存占用线性甚至平方增长 |
| 训练时模型见不到完整 completion | 大部分位置是 padding，浪费计算 |
| 推理时模型不会生成长答案 | batch size 被迫缩小 |

### 6.2 怎么决定

```python
# 用真实样本测算
lengths = [len(tok.encode(format_sample(s))) for s in dataset]
import numpy as np

print(f"p50: {np.percentile(lengths, 50):.0f}")
print(f"p90: {np.percentile(lengths, 90):.0f}")
print(f"p95: {np.percentile(lengths, 95):.0f}")
print(f"p99: {np.percentile(lengths, 99):.0f}")
print(f"max: {max(lengths)}")
```

**经验法则**：取 `p95` 或 `p99` 向上 round 到 256 的倍数。例如：
- p95 = 3400 → `max_seq_length = 3584` 或 `4096`
- p99 = 5600 → `max_seq_length = 6144` 或 `8192`

**永远不要选 max**：万一有一条 7 万 token 的怪异样本，会让你 max 选到不可承受的值。**那 1% 的离群样本直接丢掉或截断**。

### 6.3 Verilog-A 任务的预估

| 任务家族 | 典型 prompt | 典型 completion | 估计总长 |
|---|---|---|---|
| `spec-to-va` | 200-500 | 600-2000（含 trajectory） | 800-2500 |
| `tb-generation` | 300-600 | 500-1500 | 800-2100 |
| `end-to-end` | 300-700 | 1200-3500（VA + TB）| 1500-4200 |
| `bugfix` | 500-1500（含 broken code）| 800-2000 | 1300-3500 |

**建议默认 `max_seq_length = 4096`**，对 95% 的样本足够；少数 e2e 长样本截断或剔除。

---

## 7. Padding vs Packing — Batch 怎么装

### 7.1 朴素 padding（默认做法）

把一个 batch 内所有样本**补到同样长度**（用 `pad_token`）：

```
batch[0]: [t1, t2, t3, t4, t5, PAD, PAD, PAD]   labels: [-100, ..., -100, c1, c2, -100, ...]
batch[1]: [t1, t2, t3, t4, t5, t6, t7, t8]      labels: [-100, ..., -100, c1, c2, c3]
batch[2]: [t1, t2, t3, PAD, PAD, PAD, PAD, PAD]
```

`attention_mask` 标出哪些位置是有效 token、哪些是 padding（让 attention 忽略 padding）。

**问题**：如果样本长度差异大（200 vs 4000），padding 浪费严重。极端情况 90% 计算花在 padding 上。

### 7.2 Sample packing

把多条短样本**拼成一条接近 `max_seq_length` 的长样本**：

```
[sample_A (1200 tok)] [sample_B (1800 tok)] [sample_C (900 tok)] [PAD (96 tok)]
```

每条样本之间用 `<|endoftext|>` 之类的 token 隔开（让模型知道边界）。

**优势**：消除 90% 的 padding，吞吐量大幅提升（实测 1.5-3x）。
**注意**：
- attention 必须**只在样本内部**做（不能跨样本看），需要 packing-aware attention（FlashAttention 2 支持）
- `labels` 在样本间的边界 token 处也要置 -100

### 7.3 vaBench 任务该用哪个？

| 数据规模 | 选择 |
|---|---|
| 几百条，长度差异大 | **Packing**（节省宝贵的训练时间） |
| 几千条，长度均匀 | 都行，padding 简单 |
| 长样本占多数（e2e 多） | Packing 收益小，padding 也 OK |

**默认建议**：用 packing。`trl.SFTTrainer` 加 `packing=True` 即可，剩下的它处理。

---

## 8. 我们的 `<think>/<answer>` 怎么落到 chat template

把第 1 章+本章学到的都用上。最终训练样本的拼装：

```python
def build_training_example(spec_text: str, trajectory: dict, answer_code: str) -> dict:
    """
    返回符合 chat-format 的样本，喂给 trl.SFTTrainer + formatting_func。
    """
    # Step 1: 构造 assistant 的完整 completion
    completion_text = (
        "<think>\n"
        f"  <port>\n{format_ports(trajectory['ports'])}\n  </port>\n"
        f"  <behavior>\n{format_behavior(trajectory['behavior'])}\n  </behavior>\n"
        "</think>\n"
        "<answer>\n"
        f"{answer_code}\n"
        "</answer>"
    )

    # Step 2: 组装 messages
    messages = [
        {"role": "system",
         "content": "You are an expert Verilog-A engineer. "
                    "Output a structured reasoning trace in <think>...</think>, "
                    "then the final code in <answer>...</answer>."},
        {"role": "user",
         "content": spec_text},
        {"role": "assistant",
         "content": completion_text}
    ]

    return {"messages": messages}
```

然后 `formatting_func`：

```python
def formatting_func(example):
    return tok.apply_chat_template(example["messages"], tokenize=False)
```

`DataCollatorForCompletionOnlyLM` 用 `response_template="<|im_start|>assistant\n"` 把 prompt 部分自动 mask 掉。

**Mask 的结果**：
- system + user + `<|im_start|>assistant\n` → 全 -100，不算 loss
- `<think>...</think><answer>...</answer><|im_end|>` → 真实 token，算 loss

这正好对应 [`01_foundations.md`](./01_foundations.md) 公式 1 的语义。

---

## 9. 数据质量检查清单（pack_sft.py 输出前必跑）

每一条都用 5 个随机样本手工验证：

- [ ] **Tokenization**：所有 `<think>`、`<port>` 等 tag 是单 token（不是被拆成 `<`, `think`, `>`）
- [ ] **Chat template**：`<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>` 顺序对，特殊 token 整齐
- [ ] **Prompt mask**：用 collator 跑一遍，肉眼看 prompt 部分 labels 是 -100、completion 部分是真实 token
- [ ] **长度**：p95 / p99 都在 `max_seq_length` 以内；少数超长样本剔除或截断
- [ ] **格式一致**：所有样本的 trajectory 结构相同（`<port>` 在前，`<behavior>` 在后，etc.）
- [ ] **答案验证**：抽 5 条样本的 `<answer>` 跑过 OpenVAF + EVAS，确认能编译+仿真
- [ ] **分布**：4 个 task family 至少各有 30% / 30% / 25% / 15% 之类的覆盖；电路类别多样
- [ ] **去重**：fuzzy 去重，避免同一个 spec 用不同措辞出现多次（人为膨胀数据量）

**这 8 条任一失败，don't train。** 训完才发现数据有问题，浪费 8 GPU-小时只是金钱损失，重写 reward 和 trajectory 才是真痛苦。

---

## 10. 本章核心 take-aways

1. **数据 pipeline = 4 个变换**：chat template → tokenize → label masking → pad/pack。每一步都关键。
2. **Prompt masking 是 SFT 工程最容易翻车的地方**。`-100` 占位 + `DataCollatorForCompletionOnlyLM` 是标准做法。**collator 正确性必须 smoke test**。
3. **Chat template 不是可选**。Qwen2.5-Instruct 必须套，否则训练分布 ≠ 推理分布。
4. **自定义 tag (`<think>`, `<answer>` 等) 必须显式 `add_special_tokens`**，否则被 BPE 拆碎、模型学得慢学得烂。
5. **`max_seq_length` 选 p95-p99**，不要选 max。
6. **Packing 比 padding 高效 1.5-3x**，对长度差异大的数据集尤其重要。
7. **数据质量检查清单的 8 条必须全过**才开训。

---

## 11. 下一章入口

按你选的路径 B，下一章是 **`06_practical_recipe.md` 实战配方**（先有完整 picture），然后进 GRPO 学习线。

如果你想先回顾哪个细节：
- 回 [`01_foundations.md`](./01_foundations.md) §6 "token-level vs sequence-level" 看为什么"loss 低但生成废"
- 回 [`01a_formulas_walkthrough.md`](./01a_formulas_walkthrough.md) 公式 3 看"one-hot cross-entropy 只取决于正确答案概率"

如果某个工程细节本章没讲透（比如 FlashAttention 2 packing、bf16 数值稳定性、特殊 token resize embedding 的注意事项），告诉我具体哪里，我加专题。
