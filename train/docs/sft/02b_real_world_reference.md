# 02b — 真实工程参考：RTL-Coder

> 配套 [`02_data_and_format.md`](./02_data_and_format.md)。本文做三件事：
> 1. **交代 02 章信息来源**（chat template、`.jsonl` 等不是我编的）
> 2. **介绍 RTL-Coder 这个真实电路领域 SFT 工程**
> 3. **把 RTL-Coder 的代码逐段拆开**，对照 02 章概念，再讲怎么改写成 vaEvas 用得上的范式

RTL-Coder 是外部参考仓库；本设计文档不要求 vendored 第三方源码。

---

## 1. 02 章信息出处（透明溯源）

### 1.1 Chat template

| 内容 | 一级来源 |
|---|---|
| Qwen2.5 chat template 具体 token (`<\|im_start\|>` 等) | `tokenizer_config.json` 里 `chat_template` 字段（HF 仓库 [`Qwen/Qwen2.5-Coder-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/blob/main/tokenizer_config.json)，这是**机器可执行的事实标准**） |
| 设计动机 | Qwen2.5 Technical Report ([arXiv 2412.15115](https://arxiv.org/abs/2412.15115)) §"Post-training" |
| 实务调用 | HF `transformers` 文档 ["Templates for chat models"](https://huggingface.co/docs/transformers/chat_templating) |

**为什么 `tokenizer_config.json` 是决定性来源**：你写 `tok.apply_chat_template(messages)`，HF 库就是在内部读取 `chat_template` 字段（一个 Jinja2 模板字符串）然后渲染。**这个字段说什么，事实就是什么**。如果哪天 Qwen 改了，你下载新版 tokenizer 就跟着变。

### 1.2 为什么是 `.jsonl`

**ML 训练数据约定俗成是 JSONL** —— JSON Lines，每行一个独立 JSON 对象。原因：

| 比较 | JSON | JSONL |
|---|---|---|
| 结构 | 整文件一个大列表 | 每行独立 |
| 必须全部加载 | 是 | **否，可流式读** |
| 一条坏掉 | 整文件挂 | 只丢那一行 |
| 大数据集 | ❌ 内存吃不消 | ✅ |
| 多进程读 | 难 | **天然**支持 |

HF `datasets`, `trl`, DeepSpeed, vLLM 全部默认 jsonl。

**RTL-Coder 的"`.json` 实际是 jsonl"**：文件名叫 `Resyn27k.json` 但 26,532 条记录其实每行一个 JSON，直接 `json.load()` 会报 `Extra data: line 2 column 1`。这是 ML 圈一个常见命名陷阱 —— 永远以**第一行能否单独解析为 JSON** 来判断格式。

实操：

```python
# 安全的读法（无论 .json 还是 .jsonl 都能处理）
import json
with open(path) as f:
    first_line = f.readline()
    f.seek(0)
    try:
        json.loads(first_line)
        is_jsonl = True
    except json.JSONDecodeError:
        is_jsonl = False

if is_jsonl:
    data = [json.loads(l) for l in f if l.strip()]
else:
    data = json.load(f)
```

### 1.3 prompt masking & `-100` 占位

| 内容 | 一级来源 |
|---|---|
| `ignore_index=-100` 这个魔法值 | PyTorch [`nn.CrossEntropyLoss`](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) 文档：`ignore_index` 默认 `-100` |
| HF Trainer 的 loss 调用链 | `transformers.models.<arch>.modeling_<arch>.<Model>ForCausalLM.forward` 里对 `labels` 计算 `CrossEntropyLoss` |
| `DataCollatorForCompletionOnlyLM` | TRL 库源码 [`trl/trainer/utils.py`](https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py) |

---

## 2. RTL-Coder 是什么 / 为什么选它

**RTL-Coder**: 香港科技大学 zhiyao 组开源，**Verilog/RTL 代码生成模型** + SFT 训练管线。

- 论文：[arXiv 2312.08617](https://arxiv.org/abs/2312.08617)
- 仓库：[hkust-zhiyao/RTL-Coder](https://github.com/hkust-zhiyao/RTL-Coder)
- 数据集：Resyn27k（27k 条 instruction-response 对）

**为什么这个选它做教学参考**：
1. **训练代码完整自包含**（不像 VeriReason 把 SFT 推给 LlamaFactory，看不到细节）
2. **手写 `SupervisedDataset` + 手动 prompt masking** —— 把 `trl.SFTTrainer` 隐藏的步骤**公开展示**，最适合学习
3. **三个递进版本**：`mle.py`（baseline SFT）→ `mle_scoring.py`（加 scoring）→ `mle_scoring_grad_split.py`（梯度分流），是 SFT → RL 思路过渡的好教材
4. **电路领域**（虽然是 Verilog 不是 Verilog-A，但格式和我们要做的同构）
5. **HF Trainer + DeepSpeed ZeRO-2 标准栈**，技术债少

**和我们的差距**：
- 它训练 base 模型，不是 Instruct → **不用 chat template**
- 它用 fp16 + DeepSpeed ZeRO-2 + optimizer CPU offload（2023 年代风格）；我们用 bf16 + ZeRO-3（A100 友好）
- 它任务是 Verilog；我们要的是 Verilog-A（behavioral 模型 + EVAS 验证）

---

## 3. RTL-Coder 数据集真实长什么样

### 3.1 顶层结构

```python
{
  "Instruction": "<text spec>",   # ← 我们 prompt 的对应物
  "Response": [
    "<verilog code>"               # ← 我们 completion 的对应物
  ]                                # 注意：是 list，但只有 1 个元素
}
```

`Response` 是 list 这件事看起来怪 —— 应该是为了**未来支持多个 reference answer**（一个 prompt 对应多个对的代码）。RTL-Coder 实际只用了 `Response[-1]`，等价于"取最后一个"。

### 3.2 真实条目示例（来自 `Resyn27k.json` entry 0）

```json
{
  "Instruction": "\nYou are tasked with designing a module for a simple calculator that can perform basic arithmetic operations. The module should have two inputs, `a` and `b`, and four outputs, `add`, `sub`, `mul`, and `div`. The module should be able to perform the following operations:\n- `add`: add `a` and `b` and output the result\n- `sub`: subtract `b` from `a` and output the result\n- `mul`: multiply `a` and `b`...",
  "Response": [
    "module calculator(\n    input logic [31:0] a,\n    input logic [31:0] b,\n    input logic reset_n,\n    output logic [31:0] add,\n    output logic [31:0] sub,\n    output logic [31:0] mul,\n    output logic [31:0] div\n);\n\n    always @(a, b, reset_n) begin\n        if(!reset_n) begin\n            add <= 0;\n  ..."
  ]
}
```

**对照我们要做的 Verilog-A**：

| RTL-Coder 字段 | 我们对应字段 | 备注 |
|---|---|---|
| `Instruction`（自然语言 spec）| `prompt`（spec.md 文本）| 几乎一对一映射 |
| `Response[0]`（Verilog 代码）| `completion`（`<think>...</think><answer>...VA code...</answer>`）| 我们要加 **trajectory 包装** |
| 无 | trajectory step ground truth | 我们额外要存的 step 标注 |

**对照结论**：RTL-Coder 是"无 reasoning 的 baseline"；我们要做的相当于"加上 reasoning trajectory 的版本"。**数据 schema 直接借鉴它，再扩展两个字段**：

```jsonl
{"Instruction": "...", "Response": ["<think>...</think><answer>...</answer>"], "trajectory_meta": {...}, "task_family": "spec-to-va"}
```

---

## 4. `mle.py` 代码逐段解析（与 02 章概念对照）

`train/references/RTL-Coder/train/mle.py` 共 217 行。把它拆成 5 个模块讲。

### 4.1 数据加载（line 117-139）`SupervisedDataset`

```python
class SupervisedDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        list_data_dict = [json.loads(l) for l in open(data_path, "r")]
        #                ^^^^^^^^^^^^^ ← 一行一行读，这就是 JSONL 流式读取

        sources = [example['Instruction'] + '\n' for example in list_data_dict]
        targets = [f"{example['Response'][-1]}{tokenizer.eos_token}"
                   for example in list_data_dict]
        #          ^^^^^^^^^^^^^^^^^^^^^^^^^^^ ← 末尾加 eos_token，告诉模型"答案结束"

        data_dict = preprocess(sources, targets, tokenizer)
        ...
```

**对照 02 章**：
- 这里**没有 chat template** —— 仅 `Instruction + "\n" + Response + eos`。原因是它微调 base 模型，没有 chat 概念。
- `eos_token` 是 02 章 §3 讲过的特殊 token，作用是"句子终止"。

### 4.2 Prompt masking（line 99-111）`preprocess`

**这一节是整个 mle.py 最关键的 12 行**，把 02 章 §4 的 prompt masking **手动**演示了：

```python
def preprocess(sources, targets, tokenizer):
    # 1. 把 source 和 target 拼接，作为最终输入序列
    examples = [s + t for s, t in zip(sources, targets)]

    # 2. 分别 tokenize "完整序列" 和 "只 source 部分"
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer)
        for strings in (examples, sources)
    ]

    input_ids = examples_tokenized["input_ids"]   # ← 完整序列的 token IDs
    labels = copy.deepcopy(input_ids)             # ← 初始 labels = input_ids 的拷贝

    # 3. 把 source 部分的 labels 改成 IGNORE_INDEX (-100)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
        #     ^^^^^^^^^^^^^^^^^^^^^^^^^^ ← 这一行就是 prompt masking 的实现

    return dict(input_ids=input_ids, labels=labels)
```

**逐行解读**：

| 步骤 | 等价于 02 章哪里 |
|---|---|
| `examples = source + target` | 02 章 §4.2 "input_ids 是 prompt + completion 拼接" |
| 分别 tokenize 完整版和 source 版 | 是为了**算出 source 占多少个 token**（`source_len`），后面才知道前几个 token 是 prompt |
| `labels = copy.deepcopy(input_ids)` | 默认情况下 labels 等于 input_ids（每个位置算 loss）|
| `label[:source_len] = IGNORE_INDEX` | **核心**：前 `source_len` 个位置的 label 改成 `-100` |
| 后续 `Trainer` 计算 loss 时遇到 `-100` 自动跳过 | 02 章 §4.2 "PyTorch 的 `CrossEntropyLoss` 有 `ignore_index=-100`" |

**这就是 02 章 §4.3 那个 `DataCollatorForCompletionOnlyLM` 内部实际做的事**。只不过 trl 给你包了一层，RTL-Coder 直接手写。**学习角度上，RTL-Coder 的版本更清楚 —— 你能看到每一步**。

### 4.3 数据 collator（line 142-158）`DataCollatorForSupervisedDataset`

```python
class DataCollatorForSupervisedDataset:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels")
        )
        # pad input_ids 用 pad_token_id，pad labels 用 IGNORE_INDEX (-100)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        # attention_mask 通过 input_ids 是否等于 pad_token 推导
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
```

**对照 02 章 §7**：这是**朴素 padding**，不是 packing。简单透明，但效率不如 packing。

**关键细节**：padding labels 时用 `IGNORE_INDEX = -100`，不是 `pad_token_id`。所以 padding 位置**不会算 loss**。这一步如果用错（用 `pad_token_id` 当 padding label）→ 模型会被训练去预测 padding，相当于学"答完后无限输出 pad token"，推理时会乱套。

### 4.4 训练主流程（line 172-213）`train()`

```python
def train():
    # 1. 解析命令行参数
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 2. 加载模型，fp16
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.float16,
    )
    model.gradient_checkpointing_enable()  # ← 用计算换显存

    # 3. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",   # ← 注意是 left，不是 right！
        use_fast=False,
    )

    # 4. 处理 pad_token 缺失（base 模型有时没有）
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )

    # 5. 构造数据 module、Trainer，开训
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
```

**几个细节值得讲**：

| 决定 | 解释 |
|---|---|
| `torch_dtype=torch.float16` | 2023 年代选 fp16；**A100 上现在主流是 bf16**（更稳） |
| `gradient_checkpointing_enable()` | 减少 50% 激活显存，代价是反向传播多算一次。**大模型必开** |
| `padding_side="left"` | **不常见**！SFT 通常是 `right`。Left padding 主要为推理服务。RTL-Coder 这么写可能是为了和推理保持一致，但对 SFT 来说**应该改 right**（详见 §6） |
| `use_fast=False` | 强制用 Python tokenizer，而不是 Rust 快版本。原因可能是 base 模型的 fast tokenizer 当时还不稳。**今天用 fast 版本即可** |

### 4.5 特殊 token 处理（line 52-72）`smart_tokenizer_and_embedding_resize`

```python
def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))   # ← 关键！

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        # 新 token 的 embedding 用"所有老 token embedding 的平均"初始化
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
```

**对照 02 章 §3.3**：我们 02 章说"自定义 `<think>` 等 tag 必须 `add_special_tokens` 然后 `resize_token_embeddings`"。RTL-Coder 这段就是**完整实现**。

**一个比 02 章我讲得多的细节**：新加 token 的 embedding 怎么初始化？
- ❌ **错误做法**：让它随机初始化（默认 PyTorch 行为）→ 新 token 的语义和老 token 完全不连贯，模型要花很多步才能学到
- ✅ **正确做法（RTL-Coder 这里）**：新 token 的 embedding 用"所有老 token embedding 的平均"作为初值 → 新 token 一开始的语义是个"平均 token"，模型从这个起点学起更快

**这个 trick 来自 [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py)**，几乎已成 SFT 业内默认。我会在 02 章里补这个细节。

### 4.6 DeepSpeed 配置（`ds_stage_2.json`）

```json
{
    "fp16": { "enabled": "auto", ... },
    "optimizer": {
        "type": "AdamW",
        "params": { "lr": "auto", "betas": "auto", ... }
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",      ← 把 Adam 状态卸到 CPU 内存，省 GPU 显存
            "pin_memory": true
        },
        ...
    },
    ...
}
```

**这是 ZeRO-2 + optimizer CPU offload 的标准配置**。
- ZeRO-2：把 optimizer state（Adam 的 m / v 张量）和梯度切片到所有 GPU
- offload_optimizer.cpu：再把 optimizer state 推到 CPU 内存 → GPU 显存大幅释放
- 代价：每步训练有 CPU↔GPU 通信开销，**速度慢 30-50%**，但能让 7B 在小显存 GPU 上跑

**我们的预算（2×A100 80G）够，建议改 ZeRO-3 不 offload**（速度优先）。

---

## 5. 从 RTL-Coder 到 vaEvas 的适配清单

如果我们要"照 RTL-Coder 的骨架改一版 Verilog-A SFT"，要改 8 个地方：

| 改动 | RTL-Coder | vaEvas 版 | 原因 |
|---|---|---|---|
| 基座模型 | CodeLlama-7B / Mistral | **Qwen2.5-Coder-7B-Instruct** | 我们要 Instruct 起步 |
| 数据格式 | `{"Instruction", "Response": [...]}` | `{"messages": [...]}` (chat 格式) | 用 chat template |
| Source 构造 | `Instruction + "\n"` | `apply_chat_template(messages)` | Qwen instruct 必须套 |
| Prompt mask 方法 | 手动 `label[:source_len] = -100` | `DataCollatorForCompletionOnlyLM` 自动 | 用 trl 高级封装 |
| 自定义 tag | 无 | `<think>` / `<port>` / `<answer>` 等加为 special token | 我们用 trajectory |
| 精度 | `torch.float16` | `torch.bfloat16` | A100 上更稳 |
| DeepSpeed stage | ZeRO-2 + CPU offload | ZeRO-3，不 offload | 我们显存够 |
| Padding side | `"left"` | `"right"` | SFT 标准 |
| Packing | 关 | 开 (`packing=True` in trl) | 长度差异大时省时间 |
| 数据 schema | Instruction + Response | + trajectory 字段 + task_family + provenance | 我们的 reward 需要 |

**核心训练逻辑（计算 loss、更新参数）和 RTL-Coder 完全一致**。改动都在**输入数据格式 + 工具栈选型**层。

---

## 6. 你应该看的 RTL-Coder 文件（按重要度）

我建议按这个顺序通读一遍：

| # | 文件 | 行数 | 学什么 | 时长 |
|---|---|---|---|---|
| 1 | `train/mle.py` | 217 | 整个 SFT 训练流程的最小完整版 | 30 分钟 |
| 2 | `train/ds_stage_2.json` | 40 | DeepSpeed 配置长什么样 | 10 分钟 |
| 3 | `requirements.txt` | 7 | 依赖版本 | 5 分钟 |
| 4 | `dataset/Resyn27k.json` 的前 5 行 | - | 真实 SFT 数据长什么样 | 10 分钟 |
| 5 | `README.md` | 17919 字节 | 整体使用说明 | 30 分钟 |
| 6 | `train/mle_scoring.py` | 11887 字节 | **进阶版**：加 scoring（多个 response 选最优）| 1 小时 |
| 7 | `train/mle_scoring_grad_split.py` | 15034 字节 | **更进阶**：梯度分流（SFT → RL 的过渡）| 1 小时 |

第 6、7 两个文件是 SFT → RL 思路的桥梁，对我们之后学 GRPO 有帮助。**但建议先把 baseline `mle.py` 完全吃透再去看**。

`data_generation/` 和 `benchmark_inference/` 暂时不用看 —— 那是怎么造数据集和评测的，跟训练流程独立。

---

## 7. 把 RTL-Coder 当 Reference 的纪律

`train/references/RTL-Coder/` 是**只读参考**。**不要**在里面改任何东西。**不要**直接运行它的训练（数据下载慢、配置和我们 GPU 不匹配）。

我们的工作流：
1. **读** RTL-Coder 代码，理解每一步在做什么
2. **复制思想**到我们自己的 `train/train_sft/train.py`
3. **对比对照**：哪些保留、哪些升级（按 §5 表格）

`.gitignore` 也建议把 `train/references/RTL-Coder/` 排除 —— 它是外部仓库，不应该进我们的 git。

---

## 8. 本章 take-aways

1. **02 章 chat template 来源**：HF tokenizer 的 `tokenizer_config.json::chat_template` Jinja2 模板，是机器可执行的事实标准。
2. **`.jsonl` 是 ML 数据约定**：流式读、坏一行不影响整体。`.json` 扩展名有时实际是 jsonl，检查第一行。
3. **RTL-Coder 是最适合学习的 SFT 工程参考**：手写 prompt masking 把 02 章 §4 那个抽象概念**变成 12 行可执行代码**。
4. **它和"现代最佳实践"的差距点恰好是教学价值**：没用 chat template / 用 fp16 / 用 ZeRO-2+offload / 用左 padding。我们改成现代版（chat template / bf16 / ZeRO-3 / 右 padding / packing）即可。
5. **核心训练逻辑同构**：不管用哪一代工具栈，SFT 本质都是"次 token 预测 + prompt masking + Adam 更新"。RTL-Coder 帮你看穿这件事。
6. **数据 schema 直接借鉴**：`Instruction + Response[...]` 是 1:1 对应我们的 `prompt + completion`。我们只需要加 trajectory 字段。

---

## 9. 下一步

按你选的路径 B：

1. ✅ SFT 02 数据格式
2. ✅ SFT 02b RTL-Coder 真实工程参考（**当前**）
3. ⬜ **SFT 06 实战配方** —— 把 RTL-Coder 的骨架 + 02 章的概念 + 我们的需求，整合成一份具体的 yaml + Python skeleton
4. ⬜ GRPO 01 foundations
5. ⬜ GRPO 02 reward 形态

**建议**：先去 upstream RTL-Coder 的 `train/mle.py` 自己花 20 分钟通读一遍，对照本文 §4 的逐行注释。然后告诉我：
- 哪些行还看不懂？
- 哪个概念希望更深？
- 是否准备好继续 SFT 06 实战配方？
