# 06 — 实战配方：vaEvas SFT 完整设计

> **预览**：VSCode `Cmd+Shift+V`，KaTeX 渲染。
>
> **本章定位**：把 [01 foundations](./01_foundations.md) / [02 data and format](./02_data_and_format.md) / [02b RTL-Coder 参考](./02b_real_world_reference.md) / [tools/llamafactory.md](../tools/llamafactory.md) 全部综合成一份**可以直接跑**的 vaEvas SFT 配方。每个超参都有出处和理由。
>
> **参数看不懂？** → 配套文档 [`07_parameter_primer.md`](./07_parameter_primer.md)，每个参数的**是什么 / 数学 / 影响 / 怎么选**逐项解释。

> 读完这一章你应该能回答：
> 1. 我应该选什么基座、什么微调方法、什么超参？
> 2. 为什么是这些值（不是其他）？
> 3. 三档训练（smoke / pilot / production）该用什么配置？
> 4. 哪些参数随数据规模变？哪些不变？
> 5. 训练完怎么验收？
> 6. 第一次没训好该改什么？

---

## 1. 配方一页纸 (TL;DR)

```
┌─────────────────────────────────────────────────────────────┐
│  vaEvas SFT 默认配方                                          │
├─────────────────────────────────────────────────────────────┤
│  基座     : Qwen2.5-Coder-7B-Instruct                        │
│  方法     : LoRA  (rank=16, alpha=32, dropout=0.05, all)     │
│            (论文对比阶段切 full)                              │
│  数据     : <300 → smoke; 300-1k → pilot; >1k → production   │
│  Epochs   : 3 (LoRA) / 2 (full)                              │
│  LR       : 1e-4 (LoRA) / 2e-5 (full)                        │
│  Effective: per_device=1 × grad_accum=8 × n_gpus             │
│  Cutoff   : 4096                                              │
│  bf16     : true                                              │
│  Template : qwen                                              │
│  Save     : 每 50-100 步                                      │
│  Eval     : 每 50-100 步, val 自动切 10%                      │
│  GPU      : 1-2 卡 smoke; 4-8 卡 production                  │
│  Stop     : loss 不再下降 / val 开始升 / 达到 epoch 上限       │
│  验收     : compile≥60% / sim≥40% / 比 base 高 ≥15pp         │
└─────────────────────────────────────────────────────────────┘
```

下面逐项展开理由。

---

## 2. 基座模型选择

### 决策：`Qwen/Qwen2.5-Coder-7B-Instruct`

| 候选 | 取舍 |
|---|---|
| ✅ **Qwen2.5-Coder-7B-Instruct** | 代码专精；7B 单卡放得下；中文+英文+多语种代码；预训练见过 Verilog/Verilog-A 相关内容 |
| Qwen2.5-Coder-32B | 效果上限高，但 8×A100 才勉强全参；目前 overkill |
| Qwen2.5-7B（非 Coder）| 通用更强但代码弱，浪费 |
| DeepSeek-Coder-V2-Lite | 也可，但社区生态比 Qwen 稍弱 |
| Qwen2.5-Coder-7B（非 Instruct）| base 模型，需要从零教 chat template，工作量大 |

**关键观察**：选 Instruct 不是 base。理由：
- Instruct 已经懂 chat template (`<|im_start|>...<|im_end|>`)，省一轮 SFT
- 已经有指令遵循能力，从我们的 `instruction → output` 任务更容易迁移
- LLaMA-Factory `template: qwen` 直接对接 Instruct

**何时改**：
- 数据 < 100 → 用更小模型（如 3B）防过拟合
- 数据 > 5k + 8 卡空闲 → 上 14B / 32B
- 任务变 multilingual → 维持 Qwen

---

## 3. 微调方法选择：LoRA 还是 Full？

### 决策：**起步 LoRA，论文对比阶段 Full**

| 维度 | LoRA r=16 | Full SFT |
|---|---|---|
| 可训参数量 | ~0.5% | 100% |
| 单卡训练显存 | ~14 GB | ~84 GB |
| 训练速度 (8 卡) | 快 5-10× | 慢 |
| 效果上限 | 95% of full | 100% |
| 适合场景 | 快速 iter / 多实验 | 最优效果 / 论文 |
| Checkpoint 大小 | ~100 MB | ~14 GB |
| 切换基座成本 | 低 | 高 |

### 为什么先 LoRA？

数据 pipeline 在第一次跑通前，你会迭代很多次（数据格式、trajectory schema、reward 设计）。每次都全参 SFT 太贵：
- 时间：4 小时 vs 30 分钟
- 显存：8 卡 全占 vs 2 卡够
- 存储：每个 ckpt 14 GB vs 100 MB

**等流程定型，选定最佳 trajectory + reward 后**，再做一次全参 SFT 当 final 模型。这也是 Circuit-Think 实际做的（论文里用的是 full SFT，但应该有大量 LoRA 实验铺垫）。

### LoRA 关键超参

```yaml
finetuning_type: lora
lora_rank: 16        # r=8/16/32 都常见，16 是甜点
lora_alpha: 32       # 通常 = 2 × rank
lora_dropout: 0.05   # 防过拟合
lora_target: all     # 所有 linear 层（attn + mlp）
```

#### `rank` 选 16 的依据
- $r=8$：参数太少，复杂任务学不会
- $r=16$：标准选择，~99% 的 SFT 工作选这个
- $r=32$：更接近 full 效果，但显存翻倍
- $r=64+$：边际收益急剧下降

**Verilog-A 是结构化代码生成，r=16 足够**。如果发现 underfit (loss 训不下去) 再升 r=32。

#### `alpha` 选 32 的依据

LoRA 论文 (Hu et al. 2021) 推荐 $\alpha = r$，但实务上很多工作用 $\alpha = 2r$ 因为这样**学习更激进**：

$$
\text{LoRA update} = \alpha / r \times \Delta W
$$

$\alpha/r = 32/16 = 2$ 意味着 LoRA 增量被放大 2 倍，加速学习。

#### `lora_target: all` 的依据
- `q_proj, v_proj only` (原论文)：参数最少，效果可接受
- `q_proj, k_proj, v_proj, o_proj`（attention all）：标准
- **`all`**（attention + MLP 所有 linear）：最广，效果最好，显存代价小

7B 模型上 `all` 不会显存爆，建议直接 all。

---

## 4. 数据规模规划（三档）

### 决策：分三档跑，每档目标明确

| 档位 | 数据量 | 用途 | 训练时间 (LoRA, 单卡) | 期望 compile rate |
|---|---|---|---|---|
| **Smoke** | 5-20 | 验证流水线没 bug | 5-10 分钟 | 不关心 |
| **Pilot** | 100-300 | 看 baseline 效果 | 20-60 分钟 | ≥30% |
| **Production** | 500-2000 | 正式 SFT，去 GRPO | 1-3 小时 | ≥60% |

### 为什么这么分

#### Smoke (5-20 条)
**目标只有一个**：**确认从 jsonl → tokenize → 训练 → save → inference 整条链没断**。

数据量小到几乎一定过拟合（loss 会降到 0.1 以下）。**不要看 eval，只看 loop 跑通**：
- 没报错 ✓
- 至少存了 1 个 ckpt ✓
- inference 能生成可读输出 ✓

→ 通过即可，最快 5 分钟搞定。

#### Pilot (100-300 条)
**目标**：看到**有意义的 loss 曲线 + 真实的 compile rate**。

这一档你会发现一堆问题：
- 某些 task family 数据稀缺（比如 bugfix 才 5 条）
- trajectory 标注不一致
- 长样本 truncate 后语义残缺
- LR / warmup 不合适

**Pilot 是真正调参的阶段**。期望 compile rate 30-50%，sim rate 10-30%。

#### Production (500-2000 条)
正式 SFT。期望 compile ≥60%, sim ≥40%，**比 base model 高 ≥15pp**。

如果 production 数字达不到，说明 pilot 阶段调参没到位 —— 不要直接堆数据，回头改 reward/trajectory。

---

## 5. 关键超参逐项设计

### 5.1 Learning Rate

| 方法 | LR | 出处 |
|---|---|---|
| LoRA | **1e-4** | Hu et al. 2021 + 业界标准 |
| Full SFT | **2e-5** | DeepSeek-R1 / Circuit-Think |

**为什么 LoRA 比 Full 高 5×**：LoRA 只动很小一部分参数，需要更大步长才能把这小部分推到位。Full 动全部参数，需要小步长防止把预训练知识擦掉（catastrophic forgetting，见 [01 foundations §4.2](./01_foundations.md#42-训太狠会擦掉基座能力)）。

### 5.2 Effective Batch Size

$$
\text{Effective batch} = \text{per\_device\_batch} \times \text{grad\_accum} \times \text{n\_gpus}
$$

| 配置 | 数字 | 总有效 batch |
|---|---|---|
| Smoke (单卡) | 1 × 4 × 1 | **4** |
| Pilot (单卡) | 1 × 8 × 1 | **8** |
| Production (8 卡) | 1 × 8 × 8 | **64** |

#### 为什么 `per_device_batch = 1`

`cutoff_len = 4096`, bf16 下一个 sample 大约占 8-12 GB activation 显存。LoRA 模型 + Adam state + activation 总和接近 80GB 上限。**per_device_batch = 2 多半 OOM**。

#### 为什么 `grad_accum = 8`

为了让 effective batch 足够大（>=8）以稳定梯度。如果只 `grad_accum=1`，每步 1 个 sample 的梯度噪声太大。

#### 多卡时 effective batch 会变大

8 卡训练时 effective = 64，**梯度方差小**，**应该相应调大 LR**：
- 一种规则：linear scaling, LR × $\sqrt{8}$ = LR × 2.83
- 但 LLaMA-Factory 一般不需要手动 scale，cosine + warmup 会自适应

→ **多卡时 LR 不变即可**（如果发现 loss 不动再考虑放大）。

### 5.3 Epochs

| 方法 | Epochs | 理由 |
|---|---|---|
| LoRA | **3** | 参数少，需多看几遍 |
| Full SFT | **2** | Circuit-Think 用 2；第 3 epoch 开始过拟合 |

参考 Circuit-Think Table 1：full SFT 在 step 200（约 1 epoch on 1000 samples）就开始过拟合，loss 反弹。**不要训太多 epoch，宁愿 underfit 也别 overfit**。

#### 数据量小怎么办

| 数据 N | 推荐 epochs |
|---|---|
| < 100 | 5（小数据多看几遍）|
| 100-500 | 3-4 |
| 500-2000 | 2-3 |
| > 2000 | 1-2 |

### 5.4 Cutoff Length

```yaml
cutoff_len: 4096
```

来源：[02 章 §6](./02_data_and_format.md#6-length-budgeting--max_seq_length-怎么选) 的 p95-p99 法则。

#### vaBench 预估

| Task family | 典型总长 | 95th percentile 估计 |
|---|---|---|
| spec-to-va | 800-2500 | ~2800 |
| tb-generation | 800-2100 | ~2400 |
| end-to-end | 1500-4200 | ~4000 |
| bugfix | 1300-3500 | ~3200 |

→ **4096 覆盖所有 family 的 95%**。少数 end-to-end 超长样本会被截断，可接受。

**调小到 2048 的代价**：end-to-end family 大半被截，模型生成断头。**调大到 8192**：显存 ×2，需要 per_device_batch 切回 1 + grad_accum 加倍。

### 5.5 LR Scheduler & Warmup

```yaml
lr_scheduler_type: cosine
warmup_ratio: 0.1
```

| 选项 | 含义 |
|---|---|
| `lr_scheduler_type: cosine` | LR 从初始值按 cosine 曲线降到 0 |
| `warmup_ratio: 0.1` | 前 10% step 从 0 线性涨到初始 LR |

**Cosine + warmup** 是 LLM SFT 标准配方。理由：
- Warmup 防止前期 LR 太高把模型推飞
- Cosine 比 linear 衰减温和，最后阶段精修

#### 数字感

假设 production = 1000 samples × 3 epoch / 8 grad_accum = ~375 step。
- Warmup: 37 步 (LR: 0 → 1e-4)
- Steady: 步 37-188 左右 LR 在 1e-4 附近
- Decay: 步 188-375 LR 缓慢降到 ~0

### 5.6 Precision

```yaml
bf16: true
```

- **bf16** (brain float 16): A100 原生支持, 数值范围和 fp32 一样, 精度低于 fp32
- **fp16**: 2017 年代选项, 数值范围窄 (容易 overflow), A100 也支持但不推荐
- **fp32**: 显存 ×2, 训练慢, 现代 LLM 训练几乎不用

**A100 上 bf16 是默认且唯一推荐**。Circuit-Think / DeepSeek / 所有现代 SFT 都用 bf16。

### 5.7 Save / Eval 频率

```yaml
save_steps: 100      # 默认 500, 我们改 100
eval_steps: 100      # 默认 500, 我们改 100
val_size: 0.1        # 默认 0.05, 数据少时改 0.1
```

#### 为什么改 100

总步数估算：
- Pilot (300 sample × 3 ep / 8 batch) = ~113 step
- Production (1000 × 3 / 8) = ~375 step

如果 `save_steps=500`，pilot 整训完都不会 save。改 100：
- Pilot 会有 1 个 ckpt
- Production 会有 3-4 个 ckpt
- 能 "early-stop" 到最佳 ckpt

#### 为什么 val_size 0.1

数据 <500 时，5% 太少（25 条以下，noise 太大）。10% 给 50-200 条 val，曲线相对稳。

### 5.8 GPU 选择和并行

| 阶段 | 用几张 / 哪几张 |
|---|---|
| Smoke | **1 张空闲卡**（GPU 0 或 1）|
| Pilot | **1-2 张**（GPU 0,1）|
| Production | **8 张（全开）** |

#### 看 GPU 占用情况

```bash
ssh jinzhihong@... 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader'
```

避开高 util 的卡（别人在训）。**8 张 A100 共用，礼貌看一眼 nvidia-smi 再启动**。

#### 多卡启动方式

```bash
# 单卡（默认）
bash scripts/train_lora_sft.sh

# 2 卡 DDP
CUDA_VISIBLE_DEVICES=0,1 FORCE_TORCHRUN=1 bash scripts/train_lora_sft.sh

# 8 卡 DDP（production）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FORCE_TORCHRUN=1 bash scripts/train_lora_sft.sh
```

LLaMA-Factory 自动用 DDP（不是 ZeRO，因为 LoRA 没必要）。

---

## 6. Trajectory tag 的特殊 token 处理

我们的 trajectory 格式（参考 [`../04_trajectory_format.md`](../04_trajectory_format.md)）包含：
```
<think>, </think>, <port>, </port>, <behavior>, </behavior>,
<testbench>, </testbench>, <answer>, </answer>,
+ bugfix 的 <symptom>, <root_cause>, <fix>
```

### 关键问题：LLaMA-Factory 不会自动加这些 special token

LF 用的是 Qwen 原生 tokenizer，**只认识 `<|im_start|>` 这种系统级 token**。我们的 `<think>` 等会被 BPE 拆碎：

```python
tok.tokenize("<think>")
# 默认: ['<', 'think', '>']  ← 3 个 token，灾难（见 02 章 §3.3）
```

### 解决方案：训练前预处理 tokenizer

LF 当前没有内置的"add special tokens"配置选项。**最干净的做法**：

```python
# 在 train/pipelines/preprocess_tokenizer.py 里:
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "/data/jinzhihong/vaEVAS/models/base/Qwen2.5-Coder-7B-Instruct"
NEW_TOKENS = [
    "<think>", "</think>",
    "<port>", "</port>",
    "<behavior>", "</behavior>",
    "<testbench>", "</testbench>",
    "<answer>", "</answer>",
    "<symptom>", "</symptom>",
    "<root_cause>", "</root_cause>",
    "<fix>", "</fix>",
]

tok = AutoTokenizer.from_pretrained(MODEL)
n_added = tok.add_special_tokens({"additional_special_tokens": NEW_TOKENS})
print(f"Added {n_added} tokens, new vocab size: {len(tok)}")

# resize embeddings (用现有 embedding 平均初始化, 见 02b §4.5)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="bfloat16")
model.resize_token_embeddings(len(tok))
old_size = model.get_input_embeddings().weight.shape[0] - n_added
input_avg = model.get_input_embeddings().weight[:old_size].mean(dim=0, keepdim=True)
output_avg = model.get_output_embeddings().weight[:old_size].mean(dim=0, keepdim=True)
model.get_input_embeddings().weight.data[-n_added:] = input_avg
model.get_output_embeddings().weight.data[-n_added:] = output_avg

# 保存到新路径（不污染原模型）
NEW_PATH = "/data/jinzhihong/vaEVAS/models/base/Qwen2.5-Coder-7B-Instruct-vaevas"
tok.save_pretrained(NEW_PATH)
model.save_pretrained(NEW_PATH)
print(f"Saved to {NEW_PATH}")
```

然后改 YAML 指向新模型：

```yaml
model_name_or_path: /data/jinzhihong/vaEVAS/models/base/Qwen2.5-Coder-7B-Instruct-vaevas
```

### 验证 tag 已成 special token

```python
tok.tokenize("<think>")
# 应该输出 ['<think>']
```

**这一步必须做**。漏了的后果：trajectory 永远学不会，loss 看起来在降但生成的 `<think>` 标签是拼接出来的字符。

---

## 7. 训练监控：要看哪些曲线

### 7.1 训练 loss

健康的 SFT loss 曲线大致这样：

```
loss
 │
3.0 ┤●
    │ ●●
2.0 ┤   ●●●●
    │       ●●●●●●●
1.0 ┤              ●●●●●●●●●●●●●●●●●●
0.5 ┤                                ●●●●●●●●●
0.3 ┤
    └──────────────────────────────────────────► steps
    0    37       100        200       300   375
    warmup           steady               decay
```

| 现象 | 健康吗 |
|---|---|
| 前 10 步 loss 抖动 | ✅ 正常（warmup 在加速 LR）|
| 第 100 步 loss 降到 1.0 以下 | ✅ 正常 |
| Loss 突然 NaN | ❌ LR 太高 / bf16 不稳，立刻停 |
| Loss 完全不动 | ❌ LR 太低 / 数据格式错 |
| Loss 降到 0.05 以下 | ⚠️ 多半过拟合（数据太小 / epoch 太多）|

### 7.2 Eval loss

每 `eval_steps` 跑一次。健康的：和训练 loss 平行下降，差 0.1-0.3 nat。

不健康的：
- Eval loss 上升 → **过拟合**，回滚到最近一个 ckpt
- Eval loss > training loss × 2 → **严重过拟合**

### 7.3 LR 曲线

LLaMA-Factory 默认输出 LR 曲线。应该是：0 → 1e-4（warmup）→ 1e-4（steady）→ 0（cosine decay）。

如果你看到 LR 抖动或异常 → 配置错了。

### 7.4 学习曲线之外的"必看"

`train.log` 或 wandb 之外，**必须自己跑生成检查**：

```bash
bash scripts/chat_lora.sh
# 输入: "Generate a simple voltage divider Verilog-A model."
# 看输出: 有 <think>...</think><answer>...</answer> 结构吗?
#         编译能过吗 (复制到 EVAS)?
```

**loss 看着对但生成是垃圾** —— 太常见了。每个 save 出来的 ckpt 都要这么跑一次。

---

## 8. 完整 YAML 配置（直接复制可用）

基于你远端已有的 `vaevas_lora_sft.yaml`，按本章设计调整：

```yaml
### model
model_name_or_path: /data/jinzhihong/vaEVAS/models/base/Qwen2.5-Coder-7B-Instruct-vaevas
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target: all

### dataset
dataset_dir: /data/jinzhihong/vaEVAS/data/llamafactory
dataset: vaevas_sft
template: qwen
cutoff_len: 4096
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: /data/jinzhihong/vaEVAS/outputs/qwen2.5-coder-7b/lora/sft-run-001
logging_steps: 10
save_steps: 100                # 改了：500 → 100
plot_loss: true
overwrite_output_dir: false
save_only_model: false
report_to: wandb               # 改了：none → wandb (实验追踪)

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null

### eval
val_size: 0.1                  # 改了：0.05 → 0.1 (数据少时)
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 100                # 改了：500 → 100
```

**和你原配置的 4 个改动**：
1. `model_name_or_path` 指向加了 special tokens 的新版本
2. `save_steps` / `eval_steps`: 500 → 100
3. `val_size`: 0.05 → 0.1
4. `report_to`: none → wandb（更好的实验追踪）

---

## 9. 三档训练配方

### 9.1 Smoke (单卡, 5-10 min, 验流水线)

**临时 YAML 改动**（覆盖默认）：

```yaml
num_train_epochs: 1
save_steps: 20
eval_steps: 20
logging_steps: 5
```

```bash
# 启动（单卡）
ssh jinzhihong@117.148.167.107
cd /data/jinzhihong/vaEVAS
CUDA_VISIBLE_DEVICES=0 bash scripts/train_lora_sft.sh
```

**验收**：
- ✓ 没报错
- ✓ 至少 1 个 ckpt 在 `outputs/.../sft-run-001/checkpoint-20/`
- ✓ `bash scripts/chat_lora.sh` 能生成可读输出

### 9.2 Pilot (1-2 卡, 30-60 min, baseline 效果)

用默认 YAML，启用 2 卡：

```bash
CUDA_VISIBLE_DEVICES=0,1 FORCE_TORCHRUN=1 bash scripts/train_lora_sft.sh
```

**验收**：
- ✓ train loss 降到 1.0 以下
- ✓ eval loss 跟着降，差距 < 0.3
- ✓ 生成结果格式正确（有 `<think>` 和 `<answer>` 标签）
- ✓ 抽 5 条 chat 生成跑 OpenVAF：compile rate ≥ 30%

### 9.3 Production (8 卡, 1-3 hr, 正式 baseline)

用默认 YAML，全 GPU 开：

```bash
# 先看 GPU 占用
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
# 确认 8 张都比较空，再启
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FORCE_TORCHRUN=1 bash scripts/train_lora_sft.sh
```

**验收**（最严格）：
- ✓ train + eval loss 健康
- ✓ 在我们 `train/data/eval/` 上跑：
  - compile rate ≥ 60%
  - sim rate ≥ 40%
  - 比 base model（无 SFT）高 ≥15pp
- ✓ ckpt 保留至少 3 个（best + last + 中间）

---

## 10. 训练后处理：merge + chat 验证

### 10.1 Merge LoRA

LoRA 训出来是 adapter，合并成完整模型方便后续 GRPO / vLLM 使用：

```bash
bash scripts/merge_lora.sh
```

产物：`/data/jinzhihong/vaEVAS/models/qwen2.5-coder-7b-vaevas-sft/`，14 GB 完整 HF 模型。

### 10.2 chat 测试

```bash
bash scripts/chat_lora.sh
# 输入你设计的几个 spec
```

不只是看"能生成"，看：
- 格式对吗（XML 标签完整）
- trajectory 合理吗（port → behavior 顺序）
- 代码能编译吗（复制到 EVAS 跑）

### 10.3 用 vLLM 跑批量 eval

```python
from vllm import LLM, SamplingParams

llm = LLM(model="/data/jinzhihong/vaEVAS/models/qwen2.5-coder-7b-vaevas-sft", dtype="bfloat16")
params = SamplingParams(temperature=0.0, max_tokens=2048)

eval_prompts = load_eval_prompts("train/data/eval/")
outputs = llm.generate(eval_prompts, params)

# 跑 EVAS 验证，统计 compile / sim / correct
```

→ 这就是 [Phase 4 eval](../06_eval_protocol.md) 的入口。

---

## 11. 训练效果不好怎么改

按"先简单后复杂"顺序排查：

### 11.1 第一步：看 loss 曲线

| Loss 现象 | 大概率原因 | 改什么 |
|---|---|---|
| 完全不降 | LR 太低 / 数据格式错 | LR ×3；检查 tokenize 后内容 |
| 立刻 NaN | LR 太高 / bf16 不稳 | LR ÷10；试 fp16 |
| 降到 0.05 以下 | 严重过拟合 | epoch 降到 1-2；数据增多 |
| 抖动剧烈 | batch 太小 | grad_accum ×2 |

### 11.2 第二步：看生成质量

| 生成现象 | 原因 | 改什么 |
|---|---|---|
| 格式标签缺失 | special token 没加 | 跑 `preprocess_tokenizer.py` |
| 标签拆碎成 `<`, `think`, `>` | 同上 | 同上 |
| 编译过但仿真不过 | 模型学了语法没学语义 | 数据里多放完整 simulate-pass 例子 |
| 生成截断（中途停） | cutoff 太短 / max_tokens 太短 | 调大 |
| 生成重复 / 死循环 | 训练数据有重复 / temperature=0 时正常 | 去重 / chat 时设 temperature=0.7 |

### 11.3 第三步：看 eval 指标分布

按 task family 拆开看：

```
spec-to-va:    compile 70%, sim 50% ✓
tb-generation: compile 65%, sim 45% ✓
end-to-end:    compile 30%, sim 10% ❌ (长度问题?)
bugfix:        compile 80%, sim 60% ✓
```

如果某个 family 明显落后，针对它的数据**加量 ×3**，再 SFT 一轮。

### 11.4 第四步：考虑切 full SFT

LoRA 调到天花板（compile ~60%, sim ~30%）后还要更高 → 切 `finetuning_type: full`：
- LR 改 2e-5
- 单卡显存爆，必须 8 卡 + DeepSpeed ZeRO-3
- 训练时间 ×5-10

LLaMA-Factory 已经支持 ZeRO-3，加一行配置即可。

---

## 12. 本章 take-aways

1. **配方 = 基座 + 方法 + 超参 + 数据规模 + 监控指标**。每一项都有理由，每一项都可改。
2. **起步 LoRA r=16 / α=32 / target=all**，论文阶段切 full。LR 分别 1e-4 / 2e-5。
3. **三档训练**：Smoke 验流水线 / Pilot 看效果 / Production 上正式。
4. **`save_steps` `eval_steps` 100，不是默认 500**。否则小数据训完一次 ckpt 都没。
5. **Trajectory tag 必须预处理成 special token**，否则白训。新模型存到 `Qwen2.5-Coder-7B-Instruct-vaevas`。
6. **看 loss 不够，必须看生成质量**。loss=0.5 但 compile=10% 的 ckpt 不可用。
7. **8 卡是 production 用，不是默认**。Smoke / Pilot 单 / 双卡就够，礼貌让别人也能跑。

---

## 13. 完整 action plan（从现在到第一个可用 ckpt）

```
[依赖项]
  □ LF 安装完成 (setup_llamafactory.sh 已跑) ← 当前在跑
  □ 基座模型已下载 ✓ (已有 Qwen2.5-Coder-7B-Instruct)

[一次性预处理]
  □ 写 train/pipelines/preprocess_tokenizer.py
  □ 跑出 Qwen2.5-Coder-7B-Instruct-vaevas (加了 trajectory special tokens)
  □ 改 vaevas_lora_sft.yaml 指向新模型

[数据准备]
  □ Smoke 数据: 手工写 5-10 条 .jsonl
  □ Pilot 数据: 用 pipelines/synthesize.py 出 100-300 条
  □ Production 数据: 扩到 500-2000 条

[训练]
  □ Smoke run (5-10 min, 单卡, 验流水线)
  □ Pilot run (30-60 min, 2 卡, 看 baseline)
  □ Production run (1-3 hr, 8 卡, 正式)

[验收]
  □ Merge LoRA → 独立 HF 模型
  □ vLLM batch eval on held-out
  □ 跨 task family 看分布
  □ 决定是否切 full SFT 再来一轮

[交付]
  □ Best ckpt 路径
  □ Eval report (compile / sim / correct, 按 family 分)
  □ Wandb / log 链接
  □ → 准备进入 GRPO 阶段
```

---

## 14. 下一步

读完本章后：

**立刻可做**（不依赖 LF 安装完成）：
1. 写 `train/pipelines/preprocess_tokenizer.py`（§6 那段代码）
2. 设计 5-10 条 smoke 数据，确认 jsonl 格式

**LF 安装完成后**：
3. 复制本章 §8 的 YAML 到远端覆盖默认 config
4. 跑 Smoke run

**Smoke 通过后**：
5. 进数据 pipeline（[`../05_data_pipeline.md`](../05_data_pipeline.md)）正式做数据
6. 跑 Pilot → Production

**Production 出 baseline 后**：
7. SFT 学习线告一段落
8. 进 GRPO 学习线（[`../tools/verl.md`](../tools/verl.md)）

整个 SFT 章节系列就到这里。后面是 GRPO + reward 设计了。
