# 08 — Smoke 验证计划 + 后续训练路线

> **预览**：VSCode `Cmd+Shift+V`，KaTeX 渲染。
>
> **定位**：[`06_practical_recipe.md`](./06_practical_recipe.md) 讲了**该用什么参数**，[`07_parameter_primer.md`](./07_parameter_primer.md) 讲了**参数原理**，本章讲**怎么从"环境装好"一步步验证到"可以正式训练"**，以及之后怎么 scale 到 pilot / production。

> 读完这一章你应该能回答：
> 1. SFT 环境装好后，怎么用最小代价验证它能跑？
> 2. Smoke 通过后，下一步该做什么？
> 3. Pilot → Production 怎么逐步扩规模？
> 4. 哪个阶段做哪种验证？哪些可以延后？

---

## 1. 当前远端环境 snapshot（2026-06-04 验证通过）

| 项 | 状态 |
|---|---|
| Host | huaxiyun085 (117.148.167.107) |
| Conda env | `/data/jinzhihong/envs/llamafactory` (Python 3.11.15) |
| LLaMA-Factory | 0.9.6.dev0 (装在上面 env) |
| torch | 2.6.0+cu124 ✅ |
| CUDA driver | 550.54.15 ✅ 支持 CUDA 12.4 |
| GPU | 8× A100 80GB |
| 基座模型 | `models/base/Qwen2.5-Coder-7B-Instruct` (15 GB, 完整) ✅ |
| LF 内置 chat template | `qwen` 模板可用 |
| Trajectory special tokens | ❌ **还未注册** |
| 训练数据 | ❌ **还未上传任何 .jsonl** |

→ 我们现在卡在 2 件事：**(a) trajectory token 没加进 tokenizer**，**(b) 没数据**。

---

## 2. Smoke 测试的总体思路

**不要直接跑完整 SFT 然后发现哪里炸了**。按"信号最小化"原则，**逐层放大验证范围**，每层只验一类问题：

```
Tier 0 (env)        →  Tier 1 (model)      →  Tier 2 (tokenizer)   →  Tier 3 (mini-SFT)  →  Tier 4 (e2e)
   5 min               5 min                    5 min                 10-15 min              15 min
   ↓                   ↓                        ↓                     ↓                      ↓
 装好?              模型能 load + 推?         trajectory 标签        训练 loop 跑通?        merge + chat 通?
                                              是 1 个 token?         loss 在降?              EVAS 编译过吗?
```

**每一层失败都有明确的修复路径**。这比"跑完才发现"省 10 倍时间。

---

## 3. Tier 0 — 环境健康检查（5 分钟）

### 目标
确认所有库都能 import，CUDA 可用，没装错版本。

### 步骤

```bash
ssh jinzhihong@117.148.167.107
source /data/home/maxiaokang/miniconda3/etc/profile.d/conda.sh
conda activate /data/jinzhihong/envs/llamafactory

# 1. 验证 CLI
llamafactory-cli version

# 2. 验证关键 import 和 CUDA
python << 'EOF'
import torch
import transformers
import peft
import datasets
import trl
import accelerate
import llamafactory

print(f"✅ torch          : {torch.__version__}")
print(f"✅ cuda available : {torch.cuda.is_available()}")
print(f"✅ gpu count      : {torch.cuda.device_count()}")
print(f"✅ transformers   : {transformers.__version__}")
print(f"✅ peft           : {peft.__version__}")
print(f"✅ trl            : {trl.__version__}")
print(f"✅ accelerate     : {accelerate.__version__}")
print(f"✅ llamafactory   : {llamafactory.__version__}")

# 简单 GPU 矩阵运算
x = torch.randn(1000, 1000, device='cuda')
y = x @ x
print(f"✅ GPU matmul ok  : {y.shape}, dtype={y.dtype}")
EOF
```

### 验收
- 所有 `import` 不报错
- `cuda available: True`
- `gpu count: 8`
- 矩阵运算成功，不报 CUDA error

### 失败常见原因
| 现象 | 修 |
|---|---|
| `ModuleNotFoundError: trl` | env 没激活；重 `conda activate` |
| `cuda available: False` | torch 装错版本（cu13 vs cu124）；按 06 章 §11 流程重装 |
| `CUDA error: out of memory` | 别人在用同一张 GPU 跑大任务，`nvidia-smi` 看，换张卡 |

---

## 4. Tier 1 — 基座模型可加载 + 可推理（5 分钟）

### 目标
确认 `Qwen2.5-Coder-7B-Instruct` 文件完整、能 load 到 GPU、能 tokenize、能生成。

### 步骤

```bash
python << 'EOF'
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL = "/data/jinzhihong/vaEVAS/models/base/Qwen2.5-Coder-7B-Instruct"

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
print(f"  vocab size: {len(tok)}")
print(f"  special tokens: {tok.special_tokens_map}")

print("\nLoading model (bf16)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    trust_remote_code=True,
)
print(f"  model device: {next(model.parameters()).device}")
print(f"  model dtype : {next(model.parameters()).dtype}")
print(f"  param count : {sum(p.numel() for p in model.parameters())/1e9:.1f}B")

# 跑一次 chat template
msgs = [{"role": "user", "content": "Write a one-line Verilog-A module declaration."}]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
print(f"\nChat template output:\n{prompt}")

# 生成 30 token
inputs = tok(prompt, return_tensors="pt").to("cuda:0")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=30, do_sample=False)
gen = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
print(f"\nGenerated:\n{gen}")
EOF
```

### 验收
- 模型加载成功，约 10-20 秒
- `param count: 7.6B`（Qwen2.5-Coder-7B 实际大小）
- chat template 输出包含 `<|im_start|>user`...`<|im_end|>` 等标记
- 生成的文本能看懂（Verilog 关键词出现就行，不要求完美）

### 失败常见原因
| 现象 | 修 |
|---|---|
| `FileNotFoundError` | 模型路径错；`ls models/base/Qwen2.5-Coder-7B-Instruct` 确认 |
| OOM on load | GPU 上有别人占用大显存；换 `device_map="cuda:1"` 等 |
| `trust_remote_code` 报错 | 加 `trust_remote_code=True` |
| 生成全是同一 token | model 加载时 dtype 错；确认 `bfloat16` |

---

## 5. Tier 2 — Trajectory token 注册 + embedding resize（5 分钟）

### 目标
把 `<think>`, `<port>`, `<answer>` 等 trajectory 标签注册为 special token，并把这些 token 的 embedding 用现有 token 的平均初始化，避免随机初始化让模型从 0 学。**这一步训练前必须做**。

### 步骤：写并运行预处理脚本

在远端创建 `train/pipelines/preprocess_tokenizer.py`：

```bash
ssh jinzhihong@117.148.167.107
mkdir -p /data/jinzhihong/vaEVAS/pipelines

cat > /data/jinzhihong/vaEVAS/pipelines/preprocess_tokenizer.py << 'EOF'
"""
将 trajectory 标签注册为 special tokens，resize embedding，存到新路径。
参考: 06_practical_recipe.md §6, 02b_real_world_reference.md §4.5
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

SRC_MODEL  = "/data/jinzhihong/vaEVAS/models/base/Qwen2.5-Coder-7B-Instruct"
DST_MODEL  = "/data/jinzhihong/vaEVAS/models/base/Qwen2.5-Coder-7B-Instruct-vaevas"

NEW_TOKENS = [
    "<think>",     "</think>",
    "<port>",      "</port>",
    "<behavior>",  "</behavior>",
    "<testbench>", "</testbench>",
    "<answer>",    "</answer>",
    "<symptom>",   "</symptom>",
    "<root_cause>","</root_cause>",
    "<fix>",       "</fix>",
]

print(f"Loading tokenizer from {SRC_MODEL}")
tok = AutoTokenizer.from_pretrained(SRC_MODEL, trust_remote_code=True)
print(f"  Original vocab size: {len(tok)}")

# 检查这些 token 当前的切分
print("\nBefore registration:")
for t in NEW_TOKENS[:4]:
    ids = tok.encode(t, add_special_tokens=False)
    print(f"  {t!r:18} -> {ids} (len={len(ids)})")

n_added = tok.add_special_tokens({"additional_special_tokens": NEW_TOKENS})
print(f"\n✅ Added {n_added} special tokens, new vocab size: {len(tok)}")

print("\nAfter registration:")
for t in NEW_TOKENS[:4]:
    ids = tok.encode(t, add_special_tokens=False)
    print(f"  {t!r:18} -> {ids} (len={len(ids)})  {'✅' if len(ids)==1 else '❌'}")

print("\nLoading model (bf16)...")
model = AutoModelForCausalLM.from_pretrained(
    SRC_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True,
    device_map="cuda:0",
)

print("Resizing embeddings...")
old_size = model.get_input_embeddings().weight.shape[0]
model.resize_token_embeddings(len(tok))
new_size = model.get_input_embeddings().weight.shape[0]
print(f"  embedding: {old_size} -> {new_size}")

if n_added > 0:
    input_emb = model.get_input_embeddings().weight.data
    output_emb = model.get_output_embeddings().weight.data

    in_avg = input_emb[:old_size].mean(dim=0, keepdim=True)
    out_avg = output_emb[:old_size].mean(dim=0, keepdim=True)

    input_emb[-n_added:] = in_avg
    output_emb[-n_added:] = out_avg
    print(f"✅ Initialized {n_added} new token embeddings with mean of existing")

os.makedirs(DST_MODEL, exist_ok=True)
print(f"\nSaving to {DST_MODEL}")
tok.save_pretrained(DST_MODEL)
model.save_pretrained(DST_MODEL, safe_serialization=True)
print("✅ Done")
EOF

# 跑脚本
cd /data/jinzhihong/vaEVAS
python pipelines/preprocess_tokenizer.py
```

### 验收
- 输出 "After registration" 部分，所有 trajectory 标签 `len=1`（✅ 而不是 ❌）
- `embedding: 151936 -> 151952` 之类，大了 16
- 新模型在 `models/base/Qwen2.5-Coder-7B-Instruct-vaevas/` 约 15 GB

### 失败常见原因
| 现象 | 修 |
|---|---|
| `len=3` (`<` + `think` + `>`) | `add_special_tokens` 没调用成功；检查 NEW_TOKENS 拼写 |
| OOM 保存时 | 加 `low_cpu_mem_usage=True` |
| 写不进 DST_MODEL | 检查路径权限 |

---

## 6. Tier 3 — 5 条数据跑通完整 SFT loop（10-15 分钟）

### 目标
用最少数据让 `llamafactory-cli train` 跑通：tokenize → forward → backward → 优化 → save checkpoint。**不关心效果，只关心整条 pipeline 没报错**。

### 步骤

#### 6.1 准备 5 条 smoke 数据

```bash
ssh jinzhihong@117.148.167.107
cat > /data/jinzhihong/vaEVAS/data/llamafactory/vaevas_sft.jsonl << 'EOF'
{"instruction":"Generate a Verilog-A model for an inverter (simple logical NOT).","input":"","output":"<think><port>in (electrical input), out (electrical output)</port><behavior>out = ~in, modeled with V contribution and step transition</behavior></think><answer>`include \"disciplines.vams\"\nmodule inverter(in, out);\n  input in; output out;\n  electrical in, out;\n  analog V(out) <+ (V(in) > 0.5 ? 0.0 : 1.0);\nendmodule</answer>","system":"You are an expert Verilog-A engineer. Output structured reasoning then code."}
{"instruction":"Generate a Verilog-A model for a voltage buffer (unity gain).","input":"","output":"<think><port>vin (electrical input), vout (electrical output)</port><behavior>vout follows vin directly</behavior></think><answer>`include \"disciplines.vams\"\nmodule buffer(vin, vout);\n  input vin; output vout;\n  electrical vin, vout;\n  analog V(vout) <+ V(vin);\nendmodule</answer>","system":"You are an expert Verilog-A engineer. Output structured reasoning then code."}
{"instruction":"Generate a Verilog-A model for a voltage divider with ratio 0.5.","input":"","output":"<think><port>vin (electrical input), vout (electrical output)</port><behavior>vout = 0.5 * vin</behavior></think><answer>`include \"disciplines.vams\"\nmodule divider(vin, vout);\n  input vin; output vout;\n  electrical vin, vout;\n  analog V(vout) <+ 0.5 * V(vin);\nendmodule</answer>","system":"You are an expert Verilog-A engineer. Output structured reasoning then code."}
{"instruction":"Generate a Verilog-A model for a simple comparator (output 1 if v1 > v2 else 0).","input":"","output":"<think><port>v1, v2 (electrical inputs), out (electrical output)</port><behavior>compare v1 and v2, threshold at 0</behavior></think><answer>`include \"disciplines.vams\"\nmodule comp(v1, v2, out);\n  input v1, v2; output out;\n  electrical v1, v2, out;\n  analog V(out) <+ (V(v1) > V(v2) ? 1.0 : 0.0);\nendmodule</answer>","system":"You are an expert Verilog-A engineer. Output structured reasoning then code."}
{"instruction":"Generate a Verilog-A model for a scaled voltage amplifier (gain=2).","input":"","output":"<think><port>vin (electrical input), vout (electrical output)</port><behavior>vout = 2 * vin</behavior></think><answer>`include \"disciplines.vams\"\nmodule amp(vin, vout);\n  input vin; output vout;\n  electrical vin, vout;\n  analog V(vout) <+ 2.0 * V(vin);\nendmodule</answer>","system":"You are an expert Verilog-A engineer. Output structured reasoning then code."}
EOF

wc -l /data/jinzhihong/vaEVAS/data/llamafactory/vaevas_sft.jsonl
# 期望: 5
```

> 这 5 条**都很简单**，主要为了走通流程。effective 训练效果不重要。

#### 6.2 临时改 YAML 做 smoke

```bash
cat > /data/jinzhihong/vaEVAS/configs/llamafactory/vaevas_lora_sft_smoke.yaml << 'EOF'
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
cutoff_len: 2048              # smoke 缩短
overwrite_cache: true
preprocessing_num_workers: 4
dataloader_num_workers: 2

### output
output_dir: /data/jinzhihong/vaEVAS/outputs/qwen2.5-coder-7b/lora/smoke
logging_steps: 1               # 每步打印
save_steps: 5                  # 5 步存一次
save_total_limit: 2
plot_loss: true
overwrite_output_dir: true     # smoke 可覆盖
report_to: none

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 2  # smoke 减小
learning_rate: 1.0e-4
num_train_epochs: 3.0           # 5 条数据 × 3 epoch ÷ 2 = 7-8 步
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### eval
val_size: 0.2                   # 5 条数据切 1 条做 val
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 5
EOF
```

#### 6.3 启动 smoke training

```bash
cd /data/jinzhihong/vaEVAS

# 看 GPU 占用，挑空闲的
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

# 启动（假设 GPU 0 空闲）
CUDA_VISIBLE_DEVICES=0 CONFIG=configs/llamafactory/vaevas_lora_sft_smoke.yaml \
    bash scripts/train_lora_sft.sh 2>&1 | tee logs/smoke_$(date +%Y%m%d_%H%M%S).log
```

### 验收
- 没有报错
- 看到 loss 输出（应该从 ~2-3 降到 < 1）
- 至少存了 1 个 checkpoint（`outputs/qwen2.5-coder-7b/lora/smoke/checkpoint-5/` 或类似）
- 总训练时间 < 5 分钟

### 失败常见原因
| 现象 | 修 |
|---|---|
| OOM | `cutoff_len: 2048` 还不够小，降到 `1024` |
| Loss NaN | LR 太高（smoke 用 5 条数据可能不稳）；改 `1e-5` |
| "Dataset not found" | `dataset_info.json` 的 `file_name` 没对上；检查 |
| Tokenize 失败 | 数据 jsonl 格式错；用 `python -c "import json; [json.loads(l) for l in open('vaevas_sft.jsonl')]"` 验证 |

---

## 7. Tier 4 — Merge + chat 验证（10-15 分钟）

### 目标
把 smoke 训出来的 LoRA adapter 合并到基座，跑一次 chat 看输出格式对不对。

### 步骤

#### 7.1 改 merge 配置指向 smoke ckpt

```bash
sed -i.bak \
  -e 's|adapter_name_or_path: .*|adapter_name_or_path: /data/jinzhihong/vaEVAS/outputs/qwen2.5-coder-7b/lora/smoke|' \
  -e 's|export_dir: .*|export_dir: /data/jinzhihong/vaEVAS/models/qwen2.5-coder-7b-vaevas-smoke|' \
  /data/jinzhihong/vaEVAS/configs/llamafactory/vaevas_merge_lora.yaml

# 改基座模型路径指向带 trajectory token 的版本
sed -i \
  -e 's|model_name_or_path: .*|model_name_or_path: /data/jinzhihong/vaEVAS/models/base/Qwen2.5-Coder-7B-Instruct-vaevas|' \
  /data/jinzhihong/vaEVAS/configs/llamafactory/vaevas_merge_lora.yaml

# 跑 merge
cd /data/jinzhihong/vaEVAS
bash scripts/merge_lora.sh
```

#### 7.2 改 chat 配置 + 跑

```bash
sed -i.bak \
  -e 's|adapter_name_or_path: .*|adapter_name_or_path: /data/jinzhihong/vaEVAS/outputs/qwen2.5-coder-7b/lora/smoke|' \
  -e 's|model_name_or_path: .*|model_name_or_path: /data/jinzhihong/vaEVAS/models/base/Qwen2.5-Coder-7B-Instruct-vaevas|' \
  /data/jinzhihong/vaEVAS/configs/llamafactory/vaevas_inference_lora.yaml

CUDA_VISIBLE_DEVICES=0 bash scripts/chat_lora.sh
# 进入交互式 chat
# 输入: Generate a Verilog-A model for a 3x voltage amplifier.
# 看输出
```

### 验收
- merge 完成，`models/qwen2.5-coder-7b-vaevas-smoke/` 存在（~14 GB）
- chat 能交互，无 error
- 模型输出**包含 `<think>` 和 `<answer>` 标签**（说明 trajectory token 被识别）
- 生成的 Verilog-A 代码至少**语法上像 VA**（有 `module`, `analog`, `endmodule` 等关键字）
- **不要求**编译过 OpenVAF —— 5 条数据训不出来正确的代码

### 失败常见原因
| 现象 | 修 |
|---|---|
| 输出无 `<think>` `<answer>` 标签 | trajectory token 没正确加；回 Tier 2 |
| 输出乱码 | template 错；config 里 `template: qwen` |
| Chat 启动报 adapter not found | `adapter_name_or_path` 路径错 |

---

## 8. Smoke 全通后的下一步：进 Pilot 阶段

### 8.1 Pilot 的目标

| 维度 | Smoke | Pilot |
|---|---|---|
| 数据量 | 5 | 100-300 |
| 用途 | 验流水线 | **看真实 baseline** |
| 训练时间 | 5 分钟 | 30-60 分钟 |
| 期望效果 | 不关心 | compile rate ≥ 30% |
| 用的 YAML | smoke 版 | **正式版 (06 §8)** |

### 8.2 Pilot 阻塞项：数据从哪来

5 条手写 smoke 数据简单。**100-300 条需要 pipeline**。可选源头：

| 源 | 量 | 难度 | 推荐 |
|---|---|---|---|
| 手写 | 10-30 | 中 | smoke 用，pilot 不够 |
| EVAS bundled examples | ~30 | 低 | ✅ 直接用 |
| `veriloga-skills/` reference | ~20 | 低 | ✅ 直接用 |
| `behavioral-veriloga-eval/tasks/` 历史 | ~100 | **必须做污染审计** | ⚠️ 见 [`SCOPE_BOUNDARY.md`](../../SCOPE_BOUNDARY.md) |
| 教材抠例子 | 50+ | 中（手动 + EVAS 验证）| Pilot 后期 |
| LLM 合成（Claude/GPT-4 + EVAS verify）| 上百 | 高（写 pipeline）| Production |

**Pilot 速通建议**：直接合并前 3 个免审计源头 → 大约 60-80 条 → 凑数到 100 用手写补 → 这是 pilot 数据的最快出处。

### 8.3 Pilot 训练 checklist

```
[数据]
  □ 100-300 EVAS-verified <prompt, completion> 对
  □ 按 trajectory schema 转 jsonl (Alpaca format)
  □ 数据 → vaevas_sft.jsonl

[配置]
  □ 用 06 §8 的正式 YAML
  □ output_dir 改成 .../pilot-run-001 之类
  □ enable wandb (report_to: wandb)

[训练]
  □ 选 1-2 张空闲 GPU (避开别人占的卡)
  □ CUDA_VISIBLE_DEVICES=0,1 FORCE_TORCHRUN=1 bash scripts/train_lora_sft.sh
  □ 看 wandb 实时 loss 曲线
  □ 跑 30-60 分钟

[验收]
  □ train + eval loss 下降，曲线平行
  □ 最后 ckpt 跑 chat：输出格式正确
  □ 抽 10 条 chat 输出跑 OpenVAF → compile rate ≥ 30%
  □ 抽 5 条能编译的跑 EVAS → sim rate > 0%
```

### 8.4 Pilot 阶段会发现的常见问题

| 现象 | 大概率原因 |
|---|---|
| Loss 降但 chat 输出格式飘 | trajectory schema 不统一 / 数据噪声 |
| 某 task family compile rate 远低 | 该 family 数据太少 |
| Eval loss 早早就升 | 数据太少 epoch 太多，把 epoch 降到 2 |
| 编译过但仿真不过 | 数据里语法对但语义错的样本多 |

→ **Pilot 是真正的调参阶段**。修完上面问题再上 Production。

---

## 9. Production 阶段：什么时候 / 怎么做

### 9.1 进入 Production 的条件

**只有满足以下全部，才进 Production**（避免烧 GPU 时间）：

- ✅ Pilot 训完 compile rate ≥ 50%
- ✅ Trajectory schema 稳定（不再每周改）
- ✅ 数据 pipeline 能持续出新数据
- ✅ Held-out eval set 已 freeze（见 [`05_data_pipeline.md`](../05_data_pipeline.md)）

### 9.2 Production 训练

```
[数据]
  □ 500-2000 EVAS-verified 配对
  □ 数据按 4 task family 平衡（至少每个家族 50+）
  □ 已做污染审计 (SCOPE_BOUNDARY.md)
  □ Held-out eval set 50-100 条已分出

[配置]
  □ 06 §8 默认 YAML
  □ 8 卡训练
  □ wandb run_name 标清: vaevas-sft-prod-NNN

[启动]
  □ nvidia-smi 看 8 卡都空闲（或大部分空）
  □ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FORCE_TORCHRUN=1 bash scripts/train_lora_sft.sh

[验收]
  □ compile rate ≥ 60% on held-out eval
  □ sim rate ≥ 40% on held-out eval
  □ 比 base model 高 ≥ 15pp
  □ 每个 task family compile rate ≥ 40%
  □ Eval 报告生成（参考 06 §10.3 vLLM batch eval）
```

### 9.3 Production 后

如果 Production LoRA 表现达标 → 进 GRPO 阶段（[`../tools/verl.md`](../tools/verl.md)）。

如果不达标 → 按 [06 §11](./06_practical_recipe.md#11-训练效果不好怎么改) 排查；考虑切 Full SFT 再做一轮。

---

## 10. 完整时间表（理想情况）

| 阶段 | 工作 | 时间预估 | 现在做不做 |
|---|---|---|---|
| **Tier 0-1** | 环境 + 模型加载验证 | 10 分钟 | **可立刻做** |
| **Tier 2** | trajectory token 预处理 | 5 分钟 | **可立刻做** |
| **Tier 3-4** | 5 条数据 smoke + merge + chat | 30 分钟 | **可立刻做** |
| --- | --- | --- | --- |
| 数据 pipeline | 写 synthesize.py + verify_evas.py | 1-2 周 | 阻塞项 |
| **Pilot** | 100-300 条 + 训练 + eval | 半天 | 数据齐后 |
| --- | --- | --- | --- |
| 数据扩量 | 到 500-2000 条 | 1-2 周 | 阻塞项 |
| **Production** | 8 卡正式训练 + eval | 1 天 | Pilot 通过后 |
| --- | --- | --- | --- |
| GRPO 阶段 | verl 集成 + reward + RL 训 | 2-4 周 | Production 后 |

→ **Tier 0-4 是阻塞所有后续的唯一前置**。今天就能完成。

---

## 11. 我们现在应该按什么顺序做

**建议立刻按这个顺序走**：

```
1. Tier 0 环境验证       (5 min)
2. Tier 1 模型加载验证   (5 min)
3. Tier 2 trajectory token 注册  (5 min)
4. Tier 3 5 条数据 smoke (10-15 min)
5. Tier 4 merge + chat   (10-15 min)
                               ↓ 通过
6. 接着做数据 pipeline       (1-2 weeks)
7. Pilot                     (半天)
                               ↓ 通过
8. Production                (1 天)
                               ↓ 通过
9. 进 GRPO                   (2-4 weeks)
```

**Tier 0-4 全套 ~30-50 分钟**。今天就能让你拿到"完整流程跑通"的确定性，剩下都是数据规模的问题。

---

## 12. take-aways

1. **远端环境已经 ready**（torch 2.6+cu124 + 8×A100），可以立刻开始 smoke。
2. **Smoke 分 4 层**：env → model → tokenizer → mini-SFT → merge+chat。每层只验一类问题。
3. **trajectory token 预处理是必做前置**（Tier 2），否则训练等于白训。
4. **5 条数据足够 smoke**，不要求效果，只验证 pipeline 不报错。
5. **Pilot 是真正调参阶段**（100-300 条 + 30-60 分钟），决定后续 Production 怎么调。
6. **数据是 Pilot/Production 的真正阻塞**，不是训练代码。
7. **Tier 0-4 全套 30-50 分钟今天可以做完**。

---

## 13. 下一步

按本章 §11 顺序，**立刻可以做 Tier 0**。

如果你想：
- **我帮你跑** Tier 0-4：告诉我，我远端依次执行，每层报告结果
- **你自己跑**：跟着本章 §3-§7 复制命令即可
- **先把 Tier 2 (trajectory token 预处理) 跑了**：那是最关键的一次性步骤，后面所有训练都依赖
- **先继续写学习笔记**（GRPO 等），smoke 稍后做：也可以

我建议**至少把 Tier 2 跑了**。trajectory token 注册是一次性的，做完之后你的训练才有意义。剩下 Tier 3-4 可以等数据齐了一起做。
