# 07 — 参数原理字典

> **预览**：VSCode `Cmd+Shift+V`，KaTeX 渲染。
>
> **定位**：[`06_practical_recipe.md`](./06_practical_recipe.md) 告诉你**用什么值**；本文告诉你**每个参数是什么、为什么**。当 06 里出现 `lora_rank: 16`、`learning_rate: 1e-4`、`gradient_accumulation_steps: 8` 这种值，你想搞懂背后机制，来这里查。

> **怎么读**：不用从头读到尾。当成参考字典，按你正在配置的参数查节即可。每节结构：
> 1. **是什么** — 一句话定义
> 2. **数学/机制** — 精确解释
> 3. **影响** — 太高/太低各自的后果
> 4. **怎么选** — rule of thumb
> 5. **我们的值** — 指回 06 章

## 章节索引

| 类别 | 包含参数 |
|---|---|
| [§1 优化器和学习率](#1-优化器和学习率) | optimizer / learning_rate / lr_scheduler / warmup / weight_decay |
| [§2 Batch 相关](#2-batch-相关) | per_device_batch_size / gradient_accumulation_steps / effective batch |
| [§3 序列长度](#3-序列长度) | cutoff_len / max_seq_length / truncation / padding |
| [§4 数值精度](#4-数值精度) | fp32 / fp16 / bf16 / mixed precision |
| [§5 LoRA 参数](#5-lora-参数) | rank / alpha / dropout / target |
| [§6 训练长度](#6-训练长度) | epochs / max_steps |
| [§7 评估和保存](#7-评估和保存) | eval_strategy / eval_steps / save_steps / val_size |
| [§8 分布式训练](#8-分布式训练) | DDP / ZeRO 1/2/3 / FSDP / gradient checkpointing |
| [§9 其他](#9-其他) | logging_steps / report_to / seed |
| [§10 决策速查表](#10-决策速查表) | 按场景查参数 |

---

## 1. 优化器和学习率

### 1.1 `learning_rate` (学习率)

**是什么**：每次梯度更新时，参数沿梯度反方向移动多大一步。

**数学**：参数更新规则（SGD）：
$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta L(\theta_t)$$

其中 $\eta$ 就是 learning rate。$\nabla L$ 是 loss 对参数的梯度。

**直观理解**：把训练想象成爬山（找谷底）。LR 是你每步迈多大：
- LR 太大 → 一步迈过谷底，在两侧反复横跳，永远到不了底
- LR 太小 → 步子太小，要走非常多步才到底，且容易卡在小坑里
- LR 合适 → 平稳地下到谷底

**影响**：

| LR 太高 | LR 太低 |
|---|---|
| Loss NaN / 爆炸 | Loss 几乎不动 |
| Loss 剧烈震荡 | 训练极慢 |
| 模型遗忘基座能力 | 没学到 task |

**怎么选**：

| 情况 | 推荐 LR |
|---|---|
| 全参数 SFT (LLM) | $1\times10^{-5}$ 到 $5\times10^{-5}$ |
| LoRA SFT | $1\times10^{-4}$ 到 $3\times10^{-4}$ |
| RL (GRPO) | $1\times10^{-6}$ 到 $1\times10^{-5}$ |
| QLoRA (4bit) | $2\times10^{-4}$ 到 $5\times10^{-4}$ |

**为什么 LoRA 比 Full 高 5-10×**：LoRA 只动 0.5% 参数，每个被更新的参数都要承担更大的"信息量"，需要更激进的步长。Full SFT 动 100% 参数，每个参数承担少，小步长稳。

**我们的值**：06 章 §5.1 —— LoRA 用 `1.0e-4`，Full 用 `2e-5`。

### 1.2 优化器：`optim: adamw_torch`

**是什么**：决定**怎么用梯度更新参数**的算法。AdamW 是 LLM SFT 事实标准。

**对比**：

| 优化器 | 特点 | 何时用 |
|---|---|---|
| SGD | 最简单，只用梯度 | 小模型、有 momentum 时 |
| Adam | 每个参数自适应 LR + momentum | 已被 AdamW 取代 |
| **AdamW** | Adam + 正确的 weight decay | **LLM 默认** |
| Lion | 新，省显存 | 实验性 |

**AdamW 关键内部状态**：
- $m$ (一阶动量)：梯度的指数移动平均
- $v$ (二阶动量)：梯度平方的指数移动平均
- 更新：$\theta_{t+1} = \theta_t - \eta \cdot \frac{m_t}{\sqrt{v_t} + \epsilon}$

**显存代价**：AdamW 每个参数额外存 $m, v$ 两个 fp32 张量 → 优化器状态 ≈ **8 字节/参数**（fp32）。7B 模型 → 56 GB 优化器状态。

→ 这就是为什么 Full SFT 显存爆，LoRA 显存友好（只有 LoRA 矩阵参数有优化器状态，~0.5% × 56GB ≈ 280 MB）。

**我们的值**：默认 `adamw_torch`（LF 默认）。不用改。

### 1.3 学习率调度 `lr_scheduler_type`

**是什么**：LR 不是固定值，而是**随训练步数变化**的函数。常见类型：

| Scheduler | 形状 | 适合 |
|---|---|---|
| `constant` | LR 不变 | 极简实验 |
| `linear` | 线性从 LR 降到 0 | 一般 |
| `cosine` | 余弦曲线降到 0 | **LLM SFT 标准** |
| `cosine_with_restarts` | cosine 多次重启 | 大数据集 |
| `polynomial` | 多项式降 | 少见 |

**为什么 cosine 最好**：
- 前期 LR 大，快速逼近最优
- 后期 LR 小，精修细节
- 比 linear 在末期更"温和"，不至于太快归零

**Cosine 数学**：

$$\eta_t = \eta_{\max} \cdot \frac{1 + \cos(\pi \cdot t / T)}{2}$$

其中 $T$ 是总训练步数。$t=0$ 时 $\eta = \eta_{\max}$，$t=T$ 时 $\eta = 0$。

**我们的值**：06 章 §5.5 —— `cosine`。

### 1.4 Warmup：`warmup_ratio` / `warmup_steps`

**是什么**：训练**最初几步**让 LR 从 0 线性涨到目标 LR。

**为什么需要**：
- 训练刚开始，模型参数还没"找到方向"，梯度方向噪声大
- 直接用大 LR 会把模型推飞（loss NaN）
- Warmup 给模型一个"热身"窗口，让 BN/LayerNorm 统计稳定下来

**机制示意**：

```
LR
 │
1e-4 ┤        ╱──────────────╲
     │       ╱                ╲
     │      ╱                  ╲___ (cosine decay)
     │     ╱                       ╲
   0 ┤────╱                         ╲___ → 0
     └──────────────────────────────────► steps
     0   warmup            steady       total
```

**两种指定方式**：

| 选项 | 含义 |
|---|---|
| `warmup_ratio: 0.1` | 前 10% 总步数做 warmup |
| `warmup_steps: 50` | 前 50 步做 warmup（固定数）|

**怎么选**：

| 总训练步数 | warmup_ratio 推荐 |
|---|---|
| < 100 | 0.1 (~10 步)|
| 100-1000 | 0.05-0.1 |
| > 1000 | 0.03-0.05 (固定 50-100 步够了) |

**我们的值**：06 章 §5.5 —— `warmup_ratio: 0.1`。

### 1.5 Weight Decay：`weight_decay`

**是什么**：每步更新时**额外**把参数往 0 拉一点。一种正则化。

**数学**（AdamW 中）：
$$\theta_{t+1} = \theta_t - \eta \cdot (\hat{m}_t / \sqrt{\hat{v}_t} + \lambda \theta_t)$$

其中 $\lambda$ 就是 weight decay。

**直观**：防止参数变得太大（过拟合的征兆）。每步施加一个微小的"参数收缩"。

**典型值**：

| 任务 | weight_decay |
|---|---|
| LLM SFT | **0.01-0.1** |
| 不用 | 0 |

**我们的值**：默认 `0.01`（LF 默认；不在 06 章显式列）。

---

## 2. Batch 相关

### 2.1 `per_device_train_batch_size`

**是什么**：每张 GPU **每次 forward** 处理多少 sample。

**显存关系**：
$$\text{显存} \approx \text{model} + \text{optimizer} + B \times L \times \text{activation\_per\_token}$$

其中 $B$ 是 per_device_batch_size，$L$ 是 sequence length。**B 翻倍，activation 显存翻倍**。

**为什么默认 = 1**：cutoff_len=4096 时，单个 sample 的 activation 已经占 8-15 GB。设 $B=2$ 大概率 OOM。

**何时改大**：cutoff_len 短（< 2048）或显存大（A100 80GB + LoRA）时可以试 $B=2$ 或 $B=4$。

**我们的值**：`1`。

### 2.2 `gradient_accumulation_steps`

**是什么**：**累积 N 次梯度才更新一次参数**。等价于把 micro batch 拼成"虚拟大 batch"。

**机制**：

```
正常训练（grad_accum=1）：
  forward → backward → update → forward → backward → update → ...

grad_accum=4：
  forward → backward → (累积梯度，不 update)
  forward → backward → (累积)
  forward → backward → (累积)
  forward → backward → update (4 步梯度求和后才更新)
```

**等价于**：用 $B \times 4$ 的大 batch 做训练，但 **显存不变**（每次还是 $B$ 个 sample）。

**为什么需要**：受显存限制 `per_device_batch_size` 只能 1，但 effective batch 太小（如 1）会让梯度噪声极大、训练不稳定。Grad accum 是**用时间换 effective batch**。

**我们的值**：06 章 §5.2 —— `8`。

### 2.3 Effective Batch Size

**公式**：
$$\text{Effective batch} = \text{per\_device\_batch} \times \text{grad\_accum} \times \text{n\_gpus}$$

| 场景 | 计算 | 结果 |
|---|---|---|
| 单卡 smoke | $1 \times 4 \times 1$ | 4 |
| 单卡 pilot | $1 \times 8 \times 1$ | 8 |
| 8 卡 production | $1 \times 8 \times 8$ | 64 |

**为什么 Effective Batch 重要**：
- 太小（< 4）→ 梯度噪声大，loss 抖动剧烈
- 太大（> 256）→ 训练效率高但泛化可能变差（"large batch generalization gap"）
- LLM SFT 甜点：**16-128**

**Batch Size 和 LR 的关系**：经验法则 —— effective batch 翻倍，LR 也翻倍（linear scaling rule）。但 LLM SFT 上不绝对，cosine + warmup 通常能自适应一些。

**我们的值**：单卡 8，多卡按 GPU 数线性 scale。LR 不变。

---

## 3. 序列长度

### 3.1 `cutoff_len` / `max_seq_length`

**是什么**：模型一次最多处理多少 token。超过的部分**截断**。

**显存关系**：
$$\text{Attention 显存} = O(B \times L^2 \times d)$$

其中 $L$ 是 seq length。**L 翻倍，attention 显存 4 倍**。FlashAttention 把这个降到 $O(B \times L \times d)$，但仍然线性涨。

**怎么选**：用真实数据测**长度分布**，取 p95-p99（[02 章 §6](./02_data_and_format.md#6-length-budgeting--max_seq_length-怎么选)）。

| 任务 | 推荐 cutoff_len |
|---|---|
| 通用对话 | 2048-4096 |
| 代码生成 | 4096-8192 |
| 长文档 / 多轮 | 8192-32768 |

**我们的值**：06 章 §5.4 —— `4096`。覆盖 vaBench 95% 样本。

### 3.2 Truncation 策略

当样本超过 `cutoff_len`，要砍掉一部分：

| 策略 | 砍哪里 | 适合 |
|---|---|---|
| `truncation_side: right` | 末尾（最常见）| 通用 |
| `truncation_side: left` | 开头 | 关注最近上下文（对话）|

LLaMA-Factory 默认 right。对我们 SFT 没影响（completion 在末尾，被截掉的是 prompt 开头）。

### 3.3 Padding

**是什么**：batch 内不同长度的样本要补齐成一样长。

| 策略 | 补什么 | 注意 |
|---|---|---|
| Right padding | 右边补 pad_token | SFT 默认 |
| Left padding | 左边补 | 推理常用 |

**Padding 影响 loss**：pad 位置的 label 必须设为 `-100`（不算 loss）。LF 自动处理。

**Padding 浪费**：如果 batch 内长度差异大，浪费严重。**Sample packing**（[02 章 §7](./02_data_and_format.md#7-padding-vs-packing--batch-怎么装)）能解决，LF 加 `packing: true` 启用。

---

## 4. 数值精度

### 4.1 fp32 / fp16 / bf16 对比

| 格式 | 总 bit | 指数 bit | 尾数 bit | 数值范围 | 精度 |
|---|---|---|---|---|---|
| **fp32** | 32 | 8 | 23 | $\pm 3.4 \times 10^{38}$ | 高 |
| **fp16** | 16 | 5 | 10 | $\pm 6.5 \times 10^{4}$ | 低 |
| **bf16** | 16 | 8 | 7 | $\pm 3.4 \times 10^{38}$ | 中 |

**关键观察**：bf16 的**指数 bit 和 fp32 一样**（8 bit），所以数值范围一样大。fp16 范围只有 fp32 的 $10^{-34}$ 倍 → **训 LLM 时容易 overflow 或 underflow**。

→ **A100 / H100 上 bf16 是默认且推荐**。fp16 是 2017 年代选项。

### 4.2 Mixed Precision

**是什么**：模型参数 / forward / backward 用低精度（bf16），但**权重更新用 fp32**。

```
Forward:    bf16 weights → bf16 activations  (省显存)
Backward:   bf16 gradients                     (省显存)
Update:     梯度 cast 到 fp32, Adam 状态 fp32   (数值稳定)
```

**显存收益**：activation 和 gradient 显存减半。Weight 本身不变（Adam state 仍是 fp32）。

**LLaMA-Factory 启用**：`bf16: true` 就这么简单。

**我们的值**：06 章 §5.6 —— `bf16: true`。

---

## 5. LoRA 参数

### 5.1 LoRA 是什么（前置知识）

LoRA = Low-Rank Adaptation。**不更新原模型权重 $W$，而是学一个低秩增量 $\Delta W$**：

$$W' = W + \Delta W = W + \frac{\alpha}{r} \cdot B A$$

其中：
- $W \in \mathbb{R}^{d \times d}$：原权重（冻结）
- $A \in \mathbb{R}^{r \times d}$：LoRA 矩阵 A（可训）
- $B \in \mathbb{R}^{d \times r}$：LoRA 矩阵 B（可训）
- $r \ll d$：rank（"低"秩）
- $\alpha$：scaling 系数

**参数量对比**：
- 原 $W$：$d^2$
- LoRA $(A, B)$：$2dr$

当 $r=16, d=4096$：原 $W$ 16M，LoRA 0.13M。**减少 99.2%**。

### 5.2 `lora_rank` (r)

**是什么**：LoRA 低秩矩阵的秩。控制可训参数量和"表达能力"。

**$r$ 的意义**：可以理解为"$\Delta W$ 这个矩阵能表达多少种不同方向的变化"：
- $r=1$：只能在 1 个方向上调整 $W$
- $r=8$：8 个方向
- $r=16$：16 个方向（标准）
- $r=64$：64 个方向（接近 full）

**影响**：

| r 太低 (1-4) | r 合适 (8-32) | r 太高 (>64) |
|---|---|---|
| Underfit, loss 降不动 | 学到任务，泛化好 | 接近 full 效果，显存翻倍但收益小 |

**怎么选**：

| 任务复杂度 | 推荐 r |
|---|---|
| 简单指令调整 | 4-8 |
| 标准 SFT | **16** |
| 复杂多任务 | 32-64 |
| 接近 full | 64-128 |

**经验**：99% 的 LoRA 工作选 $r=16$。除非发现 underfit（loss 不降），先 $r=16$。

**我们的值**：06 章 §3 —— `16`。

### 5.3 `lora_alpha` (α)

**是什么**：LoRA 输出的 scaling 系数。

**机制**：实际生效的 $\Delta W$ 是 $\frac{\alpha}{r} \cdot B A$。

- $\alpha = r$ 时，scaling = 1（LoRA 论文原作）
- $\alpha = 2r$ 时，scaling = 2（实务常用）
- $\alpha = 4r$ 时，scaling = 4（激进）

**为什么 $\alpha/r$ 比 $\alpha$ 本身重要**：单独调 $\alpha$ 没意义，重要的是 $\alpha/r$ 这个**有效学习率倍率**。

**直觉**：$\alpha/r$ 大 = LoRA 增量被放大 = 学得激进。

| α/r 比值 | 行为 |
|---|---|
| 0.5 | 保守，LoRA 影响小 |
| 1 | 标准 |
| **2** | 激进，常用 |
| 4+ | 易过拟合 |

**怎么选**：$\alpha = 2r$ 是业界事实标准（Stanford Alpaca、LLaMA-Factory 默认等）。

**我们的值**：06 章 §3 —— `lora_alpha: 32`（= $2 \times 16$）。

### 5.4 `lora_dropout`

**是什么**：在 LoRA 输入上应用 dropout。**仅 LoRA 层有，原 frozen weights 不变**。

**机制**：训练时随机把一些 LoRA 输入置 0（按 dropout 概率），推理时关闭。

**作用**：防止 LoRA 过拟合（数据少时尤其有用）。

**典型值**：

| dropout | 适合 |
|---|---|
| 0 | 数据多 / 不担心过拟合 |
| **0.05-0.1** | 标准 SFT |
| 0.2+ | 数据极少 |

**我们的值**：06 章 §3 —— `0.05`。

### 5.5 `lora_target`

**是什么**：哪些 linear 层上加 LoRA。

**常见选项**：

| 设置 | 加在哪 | 参数量 | 效果 |
|---|---|---|---|
| `q_proj, v_proj` | attention 的 Q 和 V | 最少 | LoRA 论文原配 |
| `q_proj, k_proj, v_proj, o_proj` | attention 全部 | 中 | 标准 |
| **`all`** | attention + MLP 所有 linear | 最多 | **最好** |

**为什么 `all` 最好**：MLP 是 Transformer 里参数量最大的部分（占 2/3），不在 MLP 上加 LoRA 等于浪费 90% 模型容量没被微调。

**显存代价**：`all` 比 `q_proj, v_proj` 多 5-10x LoRA 参数，但绝对值仍很小（百 MB 级），相对显存影响 < 5%。

**我们的值**：06 章 §3 —— `all`。

---

## 6. 训练长度

### 6.1 `num_train_epochs`

**是什么**：完整遍历数据集多少遍。

**总训练步数公式**：
$$\text{total\_steps} = \frac{\text{n\_samples} \times \text{epochs}}{\text{effective\_batch}}$$

**怎么选**：

| 数据 N | 推荐 epochs (LoRA) | 推荐 epochs (Full) |
|---|---|---|
| < 100 | 5-10 | 5 |
| 100-500 | 3-5 | 2-3 |
| 500-2000 | 2-3 | 2 |
| > 2000 | 1-2 | 1 |

**为什么数据多反而 epochs 少**：每 epoch 模型看一遍所有数据；数据多时**第 1 epoch 已经看了大量样本**，第 2、3 epoch 边际收益变小，且容易过拟合。

**Circuit-Think 表 1**：full SFT 在 1000 sample × 1 epoch (~125 步) 就开始 overfit。强证据。

**我们的值**：06 章 §5.3 —— LoRA 3 epochs，Full 2 epochs。

### 6.2 `max_steps`

**是什么**：直接指定最大训练步数。覆盖 `num_train_epochs`。

**何时用**：
- 想固定训练时间（不管数据多少都训 1000 步）
- 数据动态变化时

**LLaMA-Factory 默认**：不设（用 `num_train_epochs`）。我们也不用。

---

## 7. 评估和保存

### 7.1 `eval_strategy` / `eval_steps`

**是什么**：训练过程中**多久跑一次 eval**。

| `eval_strategy` | 何时 eval |
|---|---|
| `no` | 不 eval |
| `steps` | 每 N 步 (N = `eval_steps`) |
| `epoch` | 每 epoch 结束 |

**作用**：监控**过拟合**。如果 train loss 在降但 eval loss 在升 → 该 early stop 了。

**频率怎么选**：

| 总步数 | 推荐 eval_steps |
|---|---|
| < 100 | 20 |
| 100-500 | 50 |
| 500-2000 | 100 |
| > 2000 | 200-500 |

**别太频繁**：eval 也要时间，每 10 步 eval 一次会拖慢训练 30%+。

**我们的值**：06 章 §5.7 —— `eval_strategy: steps, eval_steps: 100`。

### 7.2 `save_strategy` / `save_steps`

**是什么**：多久保存一个 checkpoint。

**和 eval_steps 通常对齐**（每 save 时也 eval，方便选 best ckpt）。

**`save_total_limit`**：最多保留几个 ckpt（避免磁盘爆）。LF 默认无限，自己加：

```yaml
save_total_limit: 3  # 只留最新 3 个 ckpt
```

**我们的值**：06 章 §5.7 —— `save_steps: 100`。建议加 `save_total_limit: 3`。

### 7.3 `val_size`

**是什么**：从训练集自动切多少比例作 validation。

**怎么选**：

| 总数据 N | 推荐 val_size |
|---|---|
| < 100 | 0.2 (确保 val 有 20+ 条) |
| 100-500 | 0.1 |
| 500-2000 | 0.05-0.1 |
| > 2000 | 0.02-0.05 |

**为什么数据多 val_size 小**：val 数据脱离训练浪费，固定 100-500 条做 val 通常够。

**我们的值**：06 章 §5.7 —— `0.1`（数据少时）。

---

## 8. 分布式训练

### 8.1 DDP (Distributed Data Parallel)

**是什么**：把同一个模型**复制**到每张 GPU，每张卡跑不同的 sample，最后**梯度同步**。

```
GPU 0: model copy  +  batch[0:8]   → grad_0
GPU 1: model copy  +  batch[8:16]  → grad_1
...
                                       ↓
                       梯度求平均 → 同步给所有 GPU → 各自更新
```

**显存特点**：每张 GPU 都有完整 model + optimizer state → **显存不减**。
**适合**：模型放得下单卡时（LoRA、小模型），用 DDP 简单。

**启用**：LLaMA-Factory 默认。

### 8.2 ZeRO (Zero Redundancy Optimizer)

DeepSpeed 提供。把**优化器状态 / 梯度 / 模型参数**分布到不同 GPU，消除冗余。

| 阶段 | 切片的内容 | 显存收益 | 通信代价 |
|---|---|---|---|
| ZeRO-1 | 只切 optimizer state | 4× 减少 | 低 |
| **ZeRO-2** | 切 optimizer state + gradients | 8× 减少 | 中 |
| **ZeRO-3** | 切 optimizer state + gradients + **model params** | 任意 GPU 数线性减 | 高 |

**ZeRO-1/2 vs ZeRO-3**：
- ZeRO-2：每张 GPU 仍持有完整 model param（forward 不通信，快）
- ZeRO-3：model param 也分片（forward/backward 需要 all-gather，慢但显存极省）

**何时用哪个**：

| 场景 | 推荐 |
|---|---|
| LoRA + 单卡或 2-4 卡 | **DDP** |
| Full SFT 7B + 8×A100 | **ZeRO-2** + optimizer CPU offload |
| Full SFT 30B+ | **ZeRO-3** |
| 多节点训练 | ZeRO-3 + FSDP |

**LLaMA-Factory 启用 ZeRO**：加配置：

```yaml
deepspeed: examples/deepspeed/ds_z2_config.json
```

（LF 自带几个模板 config）

**我们的值**：当前 LoRA 用 DDP，未来切 Full 时启用 ZeRO-2。

### 8.3 FSDP (Fully Sharded Data Parallel)

**是什么**：PyTorch 原生的 ZeRO-3 等价物。比 DeepSpeed 更原生集成 PyTorch。

**何时选 FSDP vs DeepSpeed**：差不多。社区惯例 LF 默认 DeepSpeed。

### 8.4 `gradient_checkpointing`

**是什么**：训练时**不保存所有 activation**，用到时**重新算一遍**。

**机制**：
- 正常 backward：所有 activation 都存着，用时直接读 → 内存 $O(L)$
- Checkpointing：只存关键点的 activation，其他用时重算 → 内存 $O(\sqrt{L})$ 但计算多 1.5×

**收益**：activation 显存大幅降低（50-80%），代价是 **训练慢 20-30%**。

**何时开**：大模型 / 长序列 / 显存紧张时。LoRA 7B 一般不用，Full SFT 7B 强烈建议开。

**LLaMA-Factory 启用**：

```yaml
gradient_checkpointing: true
```

LF 默认对大模型自动开。

---

## 9. 其他

### 9.1 `logging_steps`

每多少步打印一次 loss。

| `logging_steps` | 适合 |
|---|---|
| 1-5 | 调试 |
| **10** | 正常 |
| 50+ | 长训练，避免日志太多 |

**我们的值**：`10`。

### 9.2 `report_to`

实验日志输出到哪。

| 选项 | 效果 |
|---|---|
| `none` | 只本地 console |
| `tensorboard` | 本地 tensorboard 文件 |
| **`wandb`** | Weights & Biases 云端（推荐）|
| `mlflow` | MLflow |

**Wandb 优势**：
- 多次实验自动对比
- 实时曲线
- 团队共享
- 免费（学术）

**配置**：先 `wandb login` 一次，然后 YAML 加：

```yaml
report_to: wandb
run_name: vaevas-sft-run-001
```

**我们的值**：06 章 §8 —— 建议 `wandb`。

### 9.3 `seed`

随机种子。控制：
- 数据 shuffle 顺序
- 参数初始化（虽然基座是 pre-trained，但 LoRA 矩阵随机初始化）
- DropOut 随机

**固定 seed 让实验可复现**。

```yaml
seed: 42
```

**我们的值**：06 章没显式列，建议加 `seed: 42`。

### 9.4 `dataloader_num_workers`

DataLoader 用多少进程异步加载数据。

| 值 | 适合 |
|---|---|
| 0 | 单进程（debug）|
| **4** | 标准 |
| 8-16 | 大数据集 + 复杂预处理 |

**我们的值**：06 章 §8 —— `4`。

---

## 10. 决策速查表

按"我现在该改什么参数"的场景查：

### 训练 OOM 了

| 试 | 效果 | 代价 |
|---|---|---|
| 降 `cutoff_len` | 显存大幅降 | 长样本截断 |
| `per_device_batch_size = 1` | 显存降 | 慢 |
| 开 `gradient_checkpointing: true` | 显存降 50-80% | 慢 30% |
| 用 LoRA（如果在跑 full） | 显存降 6× | 效果略低 |
| 开 ZeRO-2 / ZeRO-3 | 显存任意降 | 通信慢 |
| `bf16: true`（如果还没开）| 显存降 50% | 几乎无代价 |

### Loss 不降

| 试 | 效果 |
|---|---|
| LR 提高 (×3) | 大概率解决 |
| 数据检查（tokenize 后内容对吗）| 排查 root cause |
| `lora_rank` 提高（如果 LoRA）| underfit 时有用 |
| `lora_target: all`（如果不是）| underfit 时有用 |

### Loss NaN

| 试 | 效果 |
|---|---|
| LR 降低 (÷10) | 大概率解决 |
| 关 `bf16`（试 fp16 → fp32）| 罕见但能救 |
| 检查数据有没有极端长 sample | 攻击源 |
| 加大 `warmup_ratio` (0.1 → 0.2) | 前期更稳 |

### 训得动但效果差

| 试 | 效果 |
|---|---|
| 数据加量 / 提质 | 治本 |
| Epochs 加 | 微改善 |
| 切 Full SFT (从 LoRA) | 效果上限提高 |
| `lora_rank` 提高（LoRA）| 表达力强 |

### 过拟合（train loss 降 / eval loss 升）

| 试 | 效果 |
|---|---|
| Epochs 减 | 直接解 |
| 数据加量 | 治本 |
| `lora_dropout` 提高（0.05 → 0.1）| 防过拟合 |
| `weight_decay` 提高（0.01 → 0.1）| 防过拟合 |
| Early stop（用 best ckpt）| 工程解 |

---

## 11. 各参数我们的最终选值表

把 06 章所有参数 + 本文解释合到一张表：

| 参数 | 值 | 来自 | 理由 |
|---|---|---|---|
| `model_name_or_path` | `Qwen2.5-Coder-7B-Instruct-vaevas` | 06 §2 | 代码强 + 加了 trajectory token |
| `finetuning_type` | `lora` | 06 §3 | 起步快，显存友好 |
| `lora_rank` | `16` | 06 §3 / 本文 §5.2 | 业界甜点 |
| `lora_alpha` | `32` | 06 §3 / 本文 §5.3 | $= 2r$ 激进学习 |
| `lora_dropout` | `0.05` | 06 §3 / 本文 §5.4 | 标准防过拟合 |
| `lora_target` | `all` | 06 §3 / 本文 §5.5 | 含 MLP，效果最好 |
| `learning_rate` | `1e-4` (LoRA) | 06 §5.1 / 本文 §1.1 | LoRA 标准 |
| `lr_scheduler_type` | `cosine` | 06 §5.5 / 本文 §1.3 | LLM SFT 标准 |
| `warmup_ratio` | `0.1` | 06 §5.5 / 本文 §1.4 | 防前期不稳 |
| `weight_decay` | `0.01` | LF 默认 / 本文 §1.5 | 微正则 |
| `per_device_train_batch_size` | `1` | 06 §5.2 / 本文 §2.1 | cutoff 4096 显存限 |
| `gradient_accumulation_steps` | `8` | 06 §5.2 / 本文 §2.2 | effective batch 8-64 |
| `cutoff_len` | `4096` | 06 §5.4 / 本文 §3.1 | p95 覆盖率 |
| `bf16` | `true` | 06 §5.6 / 本文 §4.2 | A100 标准 |
| `num_train_epochs` | `3` (LoRA) | 06 §5.3 / 本文 §6.1 | LoRA 多看几遍 |
| `eval_strategy` | `steps` | 06 §5.7 / 本文 §7.1 | 监控过拟合 |
| `eval_steps` | `100` | 06 §5.7 / 本文 §7.1 | 数据少要密 |
| `save_steps` | `100` | 06 §5.7 / 本文 §7.2 | 同上 |
| `save_total_limit` | `3` (建议加) | 本文 §7.2 | 防磁盘爆 |
| `val_size` | `0.1` | 06 §5.7 / 本文 §7.3 | 数据少要大 |
| `report_to` | `wandb` (建议) | 06 §8 / 本文 §9.2 | 实验追踪 |
| `seed` | `42` (建议加) | 本文 §9.3 | 可复现 |
| `dataloader_num_workers` | `4` | 本文 §9.4 | 标准 |
| `logging_steps` | `10` | 06 §8 / 本文 §9.1 | 标准 |

---

## 12. 怎么用这份文档

### 场景 A：看 06 章某个参数想知道是什么

→ 来本文对应章节查"是什么 / 数学 / 影响"

### 场景 B：训练遇到问题

→ §10 决策速查表 → 找症状对应的"试什么"

### 场景 C：要改一个参数但不知道改多少

→ 本文对应章节查"怎么选"小节

### 场景 D：理解为什么 06 这么选

→ 本文对应章节 + §11 总表

---

## 13. take-aways

1. **每个超参背后都有数学/机制**。理解机制比记住数字更重要 —— 数字会随任务变，机制不变。
2. **LR 是最重要的超参**。先试默认值，效果不对优先调它。
3. **Effective batch = per_device × grad_accum × n_gpus**。显存限 per_device，时间限 grad_accum，钱限 n_gpus。
4. **LoRA 的灵魂是 $\alpha/r$**，不是 $\alpha$ 或 $r$ 单独。
5. **bf16 不是 fp16**。LLM 训练用 bf16。
6. **过拟合的标志是 eval loss 升，不是 train loss 高**。
7. **改一个参数前先去 §10 看决策表**，别盲调。

---

## 14. 下一步

读完本章后：
- 回 [`06_practical_recipe.md`](./06_practical_recipe.md) 看每个值的来源，现在应该更清楚
- 看 [`../tools/llamafactory.md`](../tools/llamafactory.md) §4 你远端的实际 YAML，每个字段对照本文哪一节
- 如果某个参数你想要更深的数学推导（比如 cosine 衰减公式、AdamW 完整公式）告诉我，我加专题

参数原理 + 配方 + 工具栈 → SFT 学习线现在真的齐了。准备好就进 GRPO。
