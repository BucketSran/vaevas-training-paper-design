# LLaMA-Factory — SFT 一站式微调框架

> **预览**：VSCode `Cmd+Shift+V`，KaTeX 渲染。
>
> **本章独特之处**：不是讲抽象用法，而是基于**你已经在远端搭好的真实项目** `/data/jinzhihong/vaEVAS/`（117.148.167.107，8× A100 80GB）逐字段拆解。

> 读完这一章你应该能回答：
> 1. LLaMA-Factory 是什么，做什么的？
> 2. 远端 `vaEVAS/` 的目录结构每一层是什么？
> 3. `vaevas_lora_sft.yaml` 每个字段的含义和影响？
> 4. `dataset_info.json` 干什么的？
> 5. 4 个 bash 脚本各自干什么？
> 6. 现在到能跑通第一次训练，还差哪几步？

---

## 1. LLaMA-Factory 是什么

**LLaMA-Factory** = 北航 hiyouga 团队开源的**通用 LLM 微调框架**，把 SFT / DPO / PPO / 量化 / 推理服务**用 YAML 配置 + 一个 CLI 命令**就能跑起来。

### 核心定位

| 它替代什么 | 它的角色 |
|---|---|
| 自己写 `train.py` + 一堆 trl/accelerate 配置 | YAML 一个，命令一行 |
| 手动管 LoRA adapter / 量化 / 模板 | 全部封装 |
| 写 inference 服务 | CLI 自带 chat / API server |

### 它适合谁

✅ **生产 SFT** —— 配置简单，少踩坑
✅ **不想写训练脚本** —— hiyouga 已经把脚手架封装好
✅ **要尝试多种方法** —— SFT / DPO / PPO / KTO / 量化 都用同一个 YAML 结构
❌ **想理解每一步内部发生什么** —— 那应该看 RTL-Coder `mle.py`（[02b](../sft/02b_real_world_reference.md)）
❌ **要做复杂 GRPO with vLLM rollout** —— 那是 verl 的事（[verl.md](./verl.md)）

### 我们的策略

按 [`../REFERENCES.md` 框架选型景观](../REFERENCES.md#框架选型景观-framework-landscape)：

- **学概念** → RTL-Coder `mle.py` 逐行读（已完成）
- **生产 SFT** → LLaMA-Factory（**就是我们远端环境**）
- **生产 GRPO** → verl

---

## 2. 远端 vaEVAS 项目结构

你在 `117.148.167.107:/data/jinzhihong/vaEVAS/` 搭好的目录：

```
vaEVAS/
├── README.md                       ← 项目说明（已写好）
├── configs/llamafactory/           ← 所有 YAML 配置
│   ├── vaevas_lora_sft.yaml        ← ★ SFT 训练配置（本章核心）
│   ├── vaevas_merge_lora.yaml      ← LoRA 合并配置
│   └── vaevas_inference_lora.yaml  ← 推理配置
├── data/
│   ├── raw/                        ← 原始数据 dump
│   ├── processed/                  ← 中间处理产物
│   └── llamafactory/               ← ★ LF 训练直接读这里
│       ├── dataset_info.json       ← ★ 数据集注册（已配）
│       └── vaevas_sft.jsonl        ← ❌ 训练数据本体（还没上传）
├── models/
│   └── base/                       ← 基座模型存放
│       └── Qwen2.5-Coder-7B-Instruct/  ← ❌ 还没下载
├── outputs/                        ← LoRA checkpoints / 训练产物
│   └── qwen2.5-coder-7b/lora/sft/  ← 训完会在这
├── logs/                           ← 运行日志
└── scripts/                        ← 入口脚本
    ├── setup_llamafactory.sh       ← 一次性安装
    ├── train_lora_sft.sh           ← 跑训练
    ├── merge_lora.sh               ← 合并 LoRA 权重到基座
    └── chat_lora.sh                ← 和训完的模型对话测试
```

**这是 LLaMA-Factory 推荐的标准目录布局**。后续所有改动都建议遵守这套布局。

### 关联但不在仓库里的两个路径

| 路径 | 内容 |
|---|---|
| `/data/jinzhihong/LlamaFactory/` | LLaMA-Factory 仓库本体（由 `setup_llamafactory.sh` 克隆）|
| `/data/jinzhihong/envs/llamafactory/` | 隔离的 conda 环境 |

为什么分开？项目工程目录 (`vaEVAS/`) 和工具仓库 (`LlamaFactory/`) 分离的好处：
- 工具升级不污染你的实验配置
- 多个项目可共用同一份 LF 安装

---

## 3. `dataset_info.json` 详解

LLaMA-Factory 不直接读 `.jsonl`，它要求你**先注册数据集**。

### 你的注册内容

```json
{
  "vaevas_sft": {
    "file_name": "vaevas_sft.jsonl",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output",
      "system": "system",
      "history": "history"
    }
  }
}
```

### 逐字段解读

| 字段 | 含义 |
|---|---|
| `"vaevas_sft"` (key) | 数据集**别名**。训练 config 里写 `dataset: vaevas_sft` 就是引用这个 |
| `file_name` | jsonl 文件名（在 `dataset_dir` 目录里） |
| `columns.prompt` | jsonl 里**哪一列**当 prompt → 这里映射到 `instruction` |
| `columns.query` | 哪一列当 query → 映射到 `input` |
| `columns.response` | 哪一列当 response → 映射到 `output` |
| `columns.system` | 系统提示词 → 映射到 `system` |
| `columns.history` | 多轮对话历史 → 映射到 `history` |

### Alpaca 格式（你 README 里描述的）

```jsonl
{"instruction":"user task","input":"","output":"assistant answer","system":"optional system prompt"}
{"instruction":"...","input":"","output":"...","history":[["old user","old assistant"]]}
```

字段映射对应关系：

| jsonl 字段 | LF 内部用途 | chat template 后的位置 |
|---|---|---|
| `instruction` | prompt 主体 | user message |
| `input` | prompt 补充（往往为空）| 拼到 instruction 后面 |
| `output` | 目标 completion | assistant message |
| `system` | 系统提示 | system message |
| `history` | 多轮上下文 | 前面几轮 user/assistant pairs |

### 我们 Verilog-A 任务的数据应该长什么样

```jsonl
{"instruction":"Generate a Verilog-A model for a sample-and-hold circuit. Inputs: clk (logic), vin (electrical). Output: vout (electrical). On clk rising edge, sample vin; hold otherwise.","input":"","output":"<think>\n  <port>vin (in, electrical), clk (in, electrical), vout (out, electrical)</port>\n  <behavior>sample vin on clk rising edge via @cross; hold via V contribution</behavior>\n</think>\n<answer>`include \"disciplines.vams\"\nmodule sh(clk, vin, vout);\n  ...\nendmodule</answer>","system":"You are an expert Verilog-A engineer."}
```

key points:
- `instruction` = spec
- `output` = `<think>` trajectory + `<answer>` Verilog-A code
- `system` = role 设定
- 多轮（bugfix 任务）可以用 `history`

→ 这就是我们 `pipelines/pack_sft.py` 最终要产出的格式。

---

## 4. `vaevas_lora_sft.yaml` 逐字段拆解（★ 本章核心）

这是远端实际使用的训练配置。我把每个字段标注成 **what / why**，并指出哪些**应该重点关注**。

### Section 1: model

```yaml
### model
model_name_or_path: /data/jinzhihong/vaEVAS/models/base/Qwen2.5-Coder-7B-Instruct
trust_remote_code: true
```

| 字段 | 含义 | 注释 |
|---|---|---|
| `model_name_or_path` | 基座模型路径（HF ID 或本地）| ❌ **当前不存在**，需要下载（详见 §7）|
| `trust_remote_code` | 信任仓库里的自定义代码 | Qwen 等需要，标 true |

### Section 2: method

```yaml
### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target: all
```

| 字段 | 含义 | 设计意图 |
|---|---|---|
| `stage` | 训练阶段：`sft` / `dpo` / `ppo` / `kto` 等 | SFT |
| `do_train` | 是否真的训练（vs 只 eval）| true |
| `finetuning_type` | 微调方法：`lora` / `freeze` / `full` | ★ **LoRA**（不是全参） |
| `lora_rank` | LoRA 矩阵的秩，控制可训参数量 | r=16 是 LoRA 标准 |
| `lora_alpha` | scaling 系数，通常 = 2 × rank | $\alpha = 32, r = 16$ |
| `lora_dropout` | LoRA 上的 dropout | 0.05 防过拟合 |
| `lora_target` | 哪些层加 LoRA：`all` / 具体层名 | all = 所有 linear 层 |

#### LoRA vs 全参 SFT（重要的设计选择）

| | LoRA | 全参 SFT (Circuit-Think 用的) |
|---|---|---|
| 训练显存 | 14 GB | 84 GB |
| 训练参数量 | ~0.5% (rank=16) | 100% |
| 训练时长 | 短 | 长 |
| 效果上限 | 略低 | 高 |
| 适合 | 快速 iter / 显存紧 | 最优效果 |

**你当前选 LoRA 是合理的**——先快速跑通流程、看到 baseline 效果。**正式做 SFT 论文比较时再切全参**（把 `finetuning_type: lora` 改成 `full`）。

### Section 3: dataset

```yaml
### dataset
dataset_dir: /data/jinzhihong/vaEVAS/data/llamafactory
dataset: vaevas_sft
template: qwen
cutoff_len: 4096
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4
```

| 字段 | 含义 | 注释 |
|---|---|---|
| `dataset_dir` | 数据所在目录 | LF 会去这找 `dataset_info.json` |
| `dataset` | 数据集名（在 dataset_info.json 里注册过的）| `vaevas_sft` |
| `template` | **chat 模板** | ★ Qwen 必须 `qwen` |
| `cutoff_len` | max sequence length，超长截断 | 4096，跟我 02 章建议一致 |
| `overwrite_cache` | 重训时刷掉 cache | true（调试期间）|
| `preprocessing_num_workers` | tokenize 用多少进程 | 16（够快）|
| `dataloader_num_workers` | dataloader 多少进程 | 4 |

#### `template: qwen` 是什么意思

LLaMA-Factory 内置了几十种模板（参考 [LF 模板列表](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/data/template.py)）。`qwen` 模板会自动套：

```
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{instruction}{input}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>
```

→ 完全对应我 [02 章 §2](../sft/02_data_and_format.md#2-chat-template--qwen25-期望什么样的输入) 讲的 Qwen chat template。**LF 帮你自动套**，你不用手写。

### Section 4: output

```yaml
### output
output_dir: /data/jinzhihong/vaEVAS/outputs/qwen2.5-coder-7b/lora/sft
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: false
save_only_model: false
report_to: none
```

| 字段 | 含义 | 注释 |
|---|---|---|
| `output_dir` | 训完的 LoRA adapter 放哪 | 你已经规划好了 |
| `logging_steps` | 每 N 步打印一次 loss | 10 步比较密 |
| `save_steps` | 每 N 步保存一次 checkpoint | 500 |
| `plot_loss` | 自动画 loss 曲线 | ✅ 训完会有 `training_loss.png` |
| `overwrite_output_dir` | 是否覆盖已有 output_dir | false（防止误覆盖）|
| `save_only_model` | 只存 model 不存 optimizer | false（保留 optimizer 可 resume）|
| `report_to` | 日志报告到哪：`wandb` / `tensorboard` / `none` | none，本地看 |

**建议**：开 `wandb` 后续看实验曲线更方便。先 none 也行。

### Section 5: train

```yaml
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
```

| 字段 | 含义 | 设计意图 |
|---|---|---|
| `per_device_train_batch_size` | 每卡每步 micro batch | 1（cutoff 4096，显存吃紧）|
| `gradient_accumulation_steps` | 累积多少 micro batch 才更新一次 | 8 |
| 实际 effective batch | $1 \times 8 = 8$（单卡）；多卡 × 卡数 | |
| `learning_rate` | 学习率 | 1e-4（LoRA 标准；全参 SFT 用 1e-5）|
| `num_train_epochs` | 训几个 epoch | 3.0（LoRA 可接受 3，全参 SFT 建议 2）|
| `lr_scheduler_type` | LR 调度策略 | cosine 标准 |
| `warmup_ratio` | warmup 占比 | 10% |
| `bf16` | 用 bfloat16 | ✅ A100 标准 |
| `ddp_timeout` | DDP 超时（秒）| 18000 万秒 ≈ 5 年（实际就是不超时）|
| `resume_from_checkpoint` | 断点续训路径 | null（从头训）|

#### 一组关键数字（Verilog-A 任务）

| | 数据 N=500 | N=2000 |
|---|---|---|
| 每 epoch step 数 | 500/8 = 63 | 2000/8 = 250 |
| 总 step (3 ep) | 188 | 750 |
| 总训练时间 (8 卡)* | ~10 分钟 | ~40 分钟 |
| save_steps=500 会保几个 ckpt | 0（不到 500 步）| 1 |

\* 估值，按 LoRA 在 A100 上每 step ~3 秒计

**问题**：当前 `save_steps=500` 对小数据不友好（可能整个训练都没 save 过一次）。建议改 `save_steps=100` 或 `save_steps=50`。

### Section 6: eval

```yaml
### eval
val_size: 0.05
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 500
```

| 字段 | 含义 | 注释 |
|---|---|---|
| `val_size` | 从训练集自动切多少作 val | 5%；如果数据量小（<500），考虑提到 10% |
| `eval_strategy` | 何时跑 eval | `steps` = 每 N 步 |
| `eval_steps` | 每 500 步跑一次 eval | 同 `save_steps` 问题，建议调小到 100 |

⚠️ **重要**：LF 的 `val_size` 是从训练集**随机切**，不是从我们前面规划的 held-out eval。我们规划的 held-out（`train/data/eval/`）和 LF val 是两件事：
- LF val：训练过程中监控 overfitting，用同分布数据
- 我们的 held-out：训完后判定真实泛化能力，用 OOD 数据

**两者都要**。

---

## 5. 四个 bash 脚本逐行解读

### 5.1 `setup_llamafactory.sh` —— 一次性安装

```bash
#!/usr/bin/env bash
set -euo pipefail                  # 严格模式：失败立刻退出

# 路径默认值（可被环境变量覆盖）
CONDA_SH="${CONDA_SH:-/data/home/maxiaokang/miniconda3/etc/profile.d/conda.sh}"
CONDA_EXE="${CONDA_EXE:-/data/home/maxiaokang/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-/data/jinzhihong/envs/llamafactory}"
LLAMA_FACTORY_ROOT="${LLAMA_FACTORY_ROOT:-/data/jinzhihong/LlamaFactory}"

# 检查 conda 存在
[ -f "$CONDA_SH" ] || { echo "conda profile script not found"; exit 1; }
[ -x "$CONDA_EXE" ] || { echo "conda executable not found"; exit 1; }

source "$CONDA_SH"

# 创建 env (Python 3.11)
[ -d "$CONDA_ENV" ] || "$CONDA_EXE" create -p "$CONDA_ENV" python=3.11 -y

conda activate "$CONDA_ENV"

# 克隆 LF 仓库
[ -d "$LLAMA_FACTORY_ROOT/.git" ] || \
  git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git "$LLAMA_FACTORY_ROOT"

# 安装 LF + metrics 依赖
cd "$LLAMA_FACTORY_ROOT"
python -m pip install -U pip
python -m pip install -e .                            # 可编辑模式安装
python -m pip install -r requirements/metrics.txt

llamafactory-cli version || true                      # 检查能否调用 CLI
```

**当你跑这个**：创建 conda env、克隆 LF、装好 → 远端就有 `llamafactory-cli` 命令了。

### 5.2 `train_lora_sft.sh` —— 跑训练

```bash
#!/usr/bin/env bash
set -euo pipefail

# 路径解析（PROJECT_ROOT = 脚本所在父目录）
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SH="${CONDA_SH:-/data/home/maxiaokang/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-/data/jinzhihong/envs/llamafactory}"
LLAMA_FACTORY_ROOT="${LLAMA_FACTORY_ROOT:-/data/jinzhihong/LlamaFactory}"
CONFIG="${CONFIG:-$PROJECT_ROOT/configs/llamafactory/vaevas_lora_sft.yaml}"

# 激活 conda env
[ -f "$CONDA_SH" ] && { source "$CONDA_SH"; conda activate "$CONDA_ENV"; }

# 检查 CLI 存在
command -v llamafactory-cli >/dev/null || { echo "Run setup_llamafactory.sh first"; exit 1; }

# 检查 LF 仓库
[ -d "$LLAMA_FACTORY_ROOT" ] || { echo "LF root not found"; exit 1; }

# 默认用 GPU 6（单卡）；可通过环境变量覆盖
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"

cd "$LLAMA_FACTORY_ROOT"
llamafactory-cli train "$CONFIG" "$@"
```

**核心一行**：`llamafactory-cli train <CONFIG>`。

**几种调用方式**：

```bash
# 默认单卡 (GPU 6)
bash scripts/train_lora_sft.sh

# 指定其他单卡
CUDA_VISIBLE_DEVICES=3 bash scripts/train_lora_sft.sh

# 两卡 DDP
CUDA_VISIBLE_DEVICES=6,7 FORCE_TORCHRUN=1 bash scripts/train_lora_sft.sh

# 八卡 DDP (你的全部资源)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FORCE_TORCHRUN=1 bash scripts/train_lora_sft.sh
```

### 5.3 `merge_lora.sh` —— LoRA 合并到基座

LoRA 训完产出的是 **adapter**（增量权重 Δ），不是完整模型。要合并成可直接 load 的完整模型：

```bash
llamafactory-cli export "$CONFIG"
```

合并配置 `vaevas_merge_lora.yaml`：

```yaml
model_name_or_path: /data/jinzhihong/vaEVAS/models/base/Qwen2.5-Coder-7B-Instruct
adapter_name_or_path: /data/jinzhihong/vaEVAS/outputs/qwen2.5-coder-7b/lora/sft
template: qwen
trust_remote_code: true

export_dir: /data/jinzhihong/vaEVAS/models/qwen2.5-coder-7b-vaevas-sft
export_size: 5        # 每个 shard 5GB
export_device: cpu    # 用 CPU 合并，不占 GPU
export_legacy_format: false
```

合并后在 `models/qwen2.5-coder-7b-vaevas-sft/` 有一个**独立可用的 HF 模型**，能用 vLLM / OpenVAF eval / 部署。

### 5.4 `chat_lora.sh` —— 和训完的模型对话

```bash
llamafactory-cli chat "$CONFIG"
```

会弹出一个交互式对话窗口，可以手动测试模型生成效果。inference 配置 `vaevas_inference_lora.yaml`：

```yaml
model_name_or_path: /data/jinzhihong/vaEVAS/models/base/Qwen2.5-Coder-7B-Instruct
adapter_name_or_path: /data/jinzhihong/vaEVAS/outputs/qwen2.5-coder-7b/lora/sft
template: qwen
infer_backend: huggingface     # 也可以改 vllm
trust_remote_code: true
```

**进阶**：把 `infer_backend: huggingface` 改成 `vllm` 就用 vLLM 推理（见 [vllm.md](./vllm.md)）。LoRA 模式下 vLLM 比 HF 快很多。

---

## 6. 我对当前配置的评估 + 改进建议

### 配置整体评估：B+（适合快速 iter，但要正式比较前需要升级）

| 维度 | 当前 | 建议 |
|---|---|---|
| 微调方式 | LoRA r=16 | ✅ 起步 OK；论文阶段切 `full` |
| 默认 GPU 用量 | 单卡 | 数据量上来后 8 卡全开 |
| save / eval 频率 | 500 步 | ⚠️ 小数据集（<2000）改 50-100 |
| 数据集 val | 自动切 5% | ⚠️ 数据 <500 时改 10-15% |
| 日志监控 | `report_to: none` | 建议开 `wandb` |
| `lora_target: all` | all linear 层 | ✅ 标准做法 |
| `cutoff_len: 4096` | ✅ 和我们设计一致 | |
| `bf16: true` | ✅ A100 标准 | |

### 推荐改动（按优先级）

**P0（必须）**：
- `save_steps: 500` → `save_steps: 100` （数据量小，否则训完一次 ckpt 都没存）
- `eval_steps: 500` → `eval_steps: 100`

**P1（强烈建议）**：
- `report_to: none` → `report_to: wandb`（实验曲线管理）
- `val_size: 0.05` → `val_size: 0.1`（数据 <500 时）

**P2（数据规模上来后）**：
- 切多卡：`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FORCE_TORCHRUN=1`
- 提 effective batch：grad_accum 从 8 提到 4，但卡数 ×8 → effective batch = 1 × 4 × 8 = 32

**P3（实验对比时）**：
- 切全参 SFT：`finetuning_type: lora` → `full`，`learning_rate: 1e-4` → `2e-5`

---

## 7. 从现在到能跑通第一次训练的 action plan

按依赖顺序排：

### Step 1 — 远端环境 setup（一次性，约 10-20 分钟）

```bash
ssh jinzhihong@117.148.167.107
cd /data/jinzhihong/vaEVAS
bash scripts/setup_llamafactory.sh
# 等待 conda env 创建 + LF clone + pip install
# 完成后应该能跑: llamafactory-cli version
```

### Step 2 — 下载基座模型（约 15-30 分钟，看网速）

```bash
cd /data/jinzhihong/vaEVAS/models/base/
# 方法 A: HuggingFace CLI
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct \
    --local-dir Qwen2.5-Coder-7B-Instruct \
    --local-dir-use-symlinks False

# 方法 B: 国内镜像 (modelscope, 快)
pip install modelscope
python -c "from modelscope import snapshot_download; \
    snapshot_download('Qwen/Qwen2.5-Coder-7B-Instruct', \
    cache_dir='/data/jinzhihong/vaEVAS/models/base')"
```

### Step 3 — 准备最小可用 SFT 数据（依赖我们的数据 pipeline）

这一步是阻塞项。需要先有 EVAS-verified 的 `<spec, va>` 配对（参考 [`../05_data_pipeline.md`](../05_data_pipeline.md)）。

最小 smoke test 数据可以这样：

```bash
# 先做一个 5-10 条的微型数据集验证流程
cat > /data/jinzhihong/vaEVAS/data/llamafactory/vaevas_sft.jsonl << 'EOF'
{"instruction":"Generate a Verilog-A model for a simple inverter.","input":"","output":"<think>...</think><answer>`include \"disciplines.vams\"\nmodule inverter(in, out);\n  ...\nendmodule</answer>","system":"You are a Verilog-A expert."}
...更多条目...
EOF
```

**这一步暂时不用搞 300+ 条，先用 5-10 条跑通流程**。

### Step 4 — Smoke test（5-10 分钟）

把训练配置临时改成超小规模，确认 pipeline 跑通：

```yaml
# 临时改这几个：
num_train_epochs: 1
save_steps: 20
eval_steps: 20
logging_steps: 5
```

```bash
# 单卡跑（最快）
bash scripts/train_lora_sft.sh
# 看到 loss 在下降 → 成功
```

### Step 5 — 正式训练（30 分钟到几小时，看数据量）

数据扩到 300-2000 条后，恢复正式参数，开多卡：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FORCE_TORCHRUN=1 \
    bash scripts/train_lora_sft.sh
```

### Step 6 — 合并 + chat 测试

```bash
bash scripts/merge_lora.sh
bash scripts/chat_lora.sh
```

---

## 8. 常见坑

| 现象 | 原因 | 解决 |
|---|---|---|
| `llamafactory-cli: command not found` | env 没激活或没装 | `conda activate /data/jinzhihong/envs/llamafactory` |
| OOM 启动 | cutoff_len 太大 / batch 太大 | cutoff_len 降到 2048 / batch=1 |
| 训练 loss 不降 | LR 太低 / 数据格式错 | 检查 LR 1e-4；用 LF 的 `--print_param_status` 看可训参数 |
| Loss 突然变 NaN | LR 太高 | 降一半试 |
| 模型加载报 `trust_remote_code` | Qwen 自定义代码 | YAML 里设 `trust_remote_code: true`（已设）|
| 多卡跑不起来 | 没设 `FORCE_TORCHRUN=1` | 加上这个环境变量 |
| Loss 看着对但 chat 输出乱码 | template 错 | 检查 `template: qwen`（已设）|
| LoRA merge 后模型变大很多 | 正常，merge 后是完整模型 | export_dir 大约 15GB 正常 |

---

## 9. 学习路径

### 现在（数据准备阶段）

1. **30 分钟** — 读本章 §1-§5（弄懂你已经搭的环境）
2. **30 分钟** — 跟着 §7 走一遍 Step 1-2（远端 setup + 下基座模型）
3. **跳过** Step 3-6（等数据 pipeline 准备好）

### 数据 pipeline 出 100+ 条后

4. 用 50-100 条做 smoke test（§7 Step 4）
5. 看 loss 曲线、看 val 指标、跑 chat 测试
6. 调整 config (P0/P1 改动)

### 数据扩到 500+ 条后

7. 正式训练（§7 Step 5）
8. Merge + 在我们的 held-out eval set 上跑（[`../06_eval_protocol.md`](../06_eval_protocol.md)）

---

## 10. take-aways

1. **LLaMA-Factory 把 SFT 训练封装到一个 YAML + 一行 CLI**。我们用它替代手写 train.py（在生产阶段）。
2. **你远端环境（8× A100 80G）已经按 LF 标准结构搭好了脚手架**，但 4 个 gap 还要补：LF 仓库未克隆 / conda env 未建 / 基座模型未下 / 数据未上传。
3. **LoRA 是当前微调方式**：r=16, alpha=32, target=all。比全参 SFT 显存省 5-10×，效果略低。先 LoRA 验证流程，定型后切全参。
4. **`dataset_info.json` 的注册机制**：jsonl 字段 → LF 内部命名的映射。我们的数据要按 Alpaca 格式（instruction/input/output/system）。
5. **template: qwen 自动套 chat 模板**，我们 02 章手写的 `<|im_start|>...<|im_end|>` LF 全代劳。
6. **建议的改动**：save_steps/eval_steps 改 100，开 wandb，数据量上来切多卡。
7. **action plan 6 步**：setup → 下模型 → 小数据 smoke test → 正式训 → merge → chat。前 2 步现在就能做。

---

## 11. 下一步

读完本章后：

**立刻可做**（不依赖数据）：
1. SSH 进远端跑 `bash scripts/setup_llamafactory.sh`（一次性安装 LF）
2. 下载 Qwen2.5-Coder-7B-Instruct 到 `models/base/`

**接下来等数据**：
3. 完成数据 pipeline（[`../05_data_pipeline.md`](../05_data_pipeline.md)）
4. 准备 5-10 条 smoke test 数据
5. 跑第一次训练

**长期**：
- 切回 [`../sft/02b_real_world_reference.md`](../sft/02b_real_world_reference.md) 对比 LF 和 RTL-Coder 的差异
- 进 [`vllm.md`](./vllm.md) 准备 GRPO 阶段
- 进 [`verl.md`](./verl.md) 设计 GRPO config

---

## 附：你的远端环境快照（snapshot of 2026-06-04）

| 项目 | 值 |
|---|---|
| Host | huaxiyun085 (117.148.167.107) |
| OS | Ubuntu 22.04.5 LTS |
| GPU | 8× NVIDIA A100-SXM4-80GB |
| Python | 3.10.12 |
| 项目根 | `/data/jinzhihong/vaEVAS/` |
| LF 期望路径 | `/data/jinzhihong/LlamaFactory/` (❌ 未克隆)|
| conda env 期望路径 | `/data/jinzhihong/envs/llamafactory/` (❌ 未创建)|
| 基座模型期望路径 | `/data/jinzhihong/vaEVAS/models/base/Qwen2.5-Coder-7B-Instruct/` (❌ 未下载)|
| 数据期望路径 | `/data/jinzhihong/vaEVAS/data/llamafactory/vaevas_sft.jsonl` (❌ 未上传)|

下次有大变动时（比如装好 LF 或上传数据），更新这个 snapshot。
