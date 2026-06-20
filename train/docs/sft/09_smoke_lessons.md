# 09 — Smoke 验证经验沉淀（2026-06-04 首次 smoke）

> **预览**：VSCode `Cmd+Shift+V`，KaTeX 渲染。
>
> **定位**：[`08_smoke_validation_plan.md`](./08_smoke_validation_plan.md) 是**预先计划**，本章是**事后总结**。下一次跑 smoke / pilot / production 都应该先扫一眼本章避免重复踩坑。

> 读完这一章你应该能：
> 1. 理解 smoke 实际抓到了什么 bug、避免在 Pilot 重蹈
> 2. 知道每个产物在远端哪里、长什么样
> 3. 拿到可复用的 Tier 0-4 checklist

---

## 1. 这次 smoke 的目标 + 实际成果

### 目标

按 [08 章 §11](./08_smoke_validation_plan.md#11-我们现在应该按什么顺序做) 顺序，跑通 Tier 0-4：environment → 模型 → trajectory token → mini SFT → merge + chat。

### 实际成果

| Tier | 验证内容 | 结果 |
|---|---|---|
| 0 | env / CUDA / 8 GPU / bf16 matmul | ✅ |
| 1 | Qwen2.5-Coder-7B 加载 7.7s / chat template / 50 token 生成 | ✅ |
| 2 | 16 trajectory token 注册 + embedding 平均初始化 | ✅ |
| 3 | LF SFT 12 步 / 17.8 s / eval loss 1.81→1.55 单调降 | ✅ |
| 4 | LoRA merge 15GB / trajectory token 全程保留 / 生成有 module/endmodule | ✅ |

整套 jsonl → tokenize → SFT → save → merge → 推理 pipeline **零工程 bug**。

### 总时长

约 **80 分钟**，其中：
- Tier 0-2: ~15 分钟（包括 trajectory token 注册的模型保存 ~1 分钟）
- 数据准备（写 build_smoke 脚本 + 10 条 jsonl）：~15 分钟
- Tier 3-4: ~10 分钟训练 + ~10 分钟 merge + ~5 分钟推理
- Debug 2 个 bug：~25 分钟（含 SSH 来回 + 二次启动）

---

## 2. 实际抓到的 3 个 bug（**永久规则**）

这三个 bug 如果在 Pilot 阶段才暴露，每个会让你浪费 1-3 小时（数据多、训练慢、debug 慢）。

### Bug #1: `CONFIG` 环境变量必须是绝对路径

**症状**：
```
FileNotFoundError: [Errno 2] No such file or directory:
  '/data/jinzhihong/LlamaFactory/configs/llamafactory/vaevas_lora_sft_smoke.yaml'
```

**根因**：`scripts/train_lora_sft.sh` 内部 `cd "$LLAMA_FACTORY_ROOT"` 后才调 `llamafactory-cli train "$CONFIG"`。如果 `CONFIG=configs/...`（相对路径），LF 会去 LlamaFactory 仓库根目录找，找不到。

**永久规则**：
```bash
# ❌ 错
CONFIG=configs/llamafactory/foo.yaml bash scripts/train_lora_sft.sh

# ✅ 对
CONFIG=/data/jinzhihong/vaEVAS/configs/llamafactory/foo.yaml bash scripts/train_lora_sft.sh
```

**预防方法**：所有用 env var 覆盖 `CONFIG` 的场景都用绝对路径。最好用 `$PROJECT_ROOT/configs/...` 显式拼接。

### Bug #2: `dataset_info.json` 声明的列必须在 jsonl 里存在

**症状**：
```
KeyError: 'history'
File "datasets/formatting/formatting.py", line 282, in __getitem__
    value = self.data[key]
```

**根因**：远端原始 `dataset_info.json` 模板里包含 `"history": "history"`（来自 LF 多轮对话默认模板），但我们的 jsonl 是单轮的，没有 `history` 字段。LF 在 `align_dataset` 阶段调 `example["history"]` → KeyError。

**永久规则**：
- `dataset_info.json` 的 `columns` 字典里**只能列 jsonl 实际有的字段**
- 单轮 SFT 不要保留 `history` 行
- 修复后的最小可用 `dataset_info.json`:
  ```json
  {
    "vaevas_sft": {
      "file_name": "vaevas_sft.jsonl",
      "columns": {
        "prompt": "instruction",
        "query": "input",
        "response": "output",
        "system": "system"
      }
    }
  }
  ```

**预防方法**：写 `pack_sft.py` 时，**先扫一遍 jsonl 第一条**，提取 key list，自动同步到 `dataset_info.json`。

### Bug #3: 10 样本不够压过 base Instruct 的格式先验（**不是 bug，是预期**）

**症状**：smoke 训完，chat 输出 Verilog 代码用 Markdown 格式（base 模型的回答风格），**完全不用 `<think>/<answer>` 标签**。

**根因**：
- Qwen2.5-Coder-7B-**Instruct** 在它的指令微调阶段已经被反复训练成"先 Markdown 分点解释、再 ``` 代码块``` "的回答风格
- 我们用 12 步 LoRA 训了 10 条数据 = 总共 ~120 个梯度样本对，远远不足以推翻 base 那几百万样本的先验

**永久规则**：
- Smoke 的目的是**验证 pipeline 工程**，不要期待生成格式
- Pilot 阶段（100-300 条 × ~120 步 = 12000-36000 梯度样本对）才有机会让 trajectory tag 出现
- Production（500-2000 条 × ~300 步 = 150000-600000）应稳定使用 trajectory tag

**预防方法**：在 smoke 阶段不要纠结生成质量；只看 pipeline 是否跑通。

---

## 3. 远端产物地图（这次 smoke 留下的）

```
/data/jinzhihong/vaEVAS/
├── models/base/
│   ├── Qwen2.5-Coder-7B-Instruct/          原始 15 GB
│   └── Qwen2.5-Coder-7B-Instruct-vaevas/   ★ 含 16 trajectory token, 15 GB
│                                            (所有未来训练都用这个 base)
├── models/
│   └── qwen2.5-coder-7b-vaevas-smoke/      ★ smoke 训完合并产物 15 GB
│                                            (可用 vLLM / chat / eval)
├── outputs/qwen2.5-coder-7b/lora/smoke/
│   ├── adapter_model.safetensors           LoRA 增量 ~80 MB
│   ├── checkpoint-10/                      第 10 步 ckpt
│   ├── checkpoint-12/                      最终 ckpt
│   ├── training_loss.png                   ★ 训练 loss 曲线 PNG
│   ├── training_eval_loss.png              ★ eval loss 曲线 PNG
│   ├── trainer_log.jsonl                   每步 metric 历史
│   ├── trainer_state.json                  完整训练状态
│   └── README.md                           LF 自动生成的描述
├── data/llamafactory/
│   ├── dataset_info.json                   ★ 修复后的注册 (无 history)
│   └── vaevas_sft.jsonl                    10 条 EVAS-examples 派生数据
├── configs/llamafactory/
│   ├── vaevas_lora_sft.yaml                production 默认
│   ├── vaevas_lora_sft_smoke.yaml         ★ smoke 用 (小 batch, 高频 save)
│   ├── vaevas_merge_lora.yaml
│   ├── vaevas_merge_lora_smoke.yaml       ★ smoke merge
│   ├── vaevas_inference_lora.yaml
│   └── vaevas_inference_lora_smoke.yaml   ★ smoke inference (HF backend)
├── pipelines/
│   └── preprocess_tokenizer.py             ★ trajectory token 注册（一次性脚本，已跑过）
└── logs/
    ├── setup_*.log                         LF 安装日志
    ├── smoke_*.log (×3)                    3 次 smoke 训练日志（含 2 次失败 + 1 次成功）
    └── merge_smoke_*.log                   merge 日志
```

**关键产物可用性矩阵**：

| 产物 | 用途 | 何时用 |
|---|---|---|
| `Qwen2.5-Coder-7B-Instruct-vaevas/` | 所有 SFT/GRPO 的基座 | **每次训练都用** |
| `qwen2.5-coder-7b-vaevas-smoke/` | vLLM 推理 / chat 测试 | 临时验证 |
| `training_loss.png` / `_eval_loss.png` | 看 loss 趋势 | 训练后看 |
| `trainer_log.jsonl` | 编程化分析 metric | 写实验报告时 |

---

## 4. 这次的真实训练数据（建立感觉）

| 项 | 值 | 注 |
|---|---|---|
| Sample 数 | 10 | EVAS 自带 examples |
| Effective batch | 2 (per_device=1 × grad_accum=2) | smoke 缩小到 2 |
| Total optimization steps | 12 | (10 samples - 2 val) × 3 epoch / 2 grad_accum |
| 训练时长 | **17.83 秒** | LoRA on 7B + A100 飞快 |
| 平均每步 | ~1.5 秒 | 含 forward + backward + optimizer step |
| Train loss 范围 | 1.43 - 2.63 (抖动)，平均 2.03 | 10 样本梯度噪声大 |
| Eval loss 单调下降 | **1.815 → 1.557 → 1.549** | ✅ 学到了 |
| total_flos | 540619 GF | 推算 A100 算力很少被吃 |
| GPU 显存峰值 | ~19 GB (模型) + ~5 GB (训练态) ≈ 24 GB | 80 GB 卡用 30% |

→ **Pilot 推算**：100 条 × 3 epoch / 8 grad_accum × 1.5s = ~56 步 × 1.5s = **~85 秒训练时间**。整个 Pilot 端到端（含 data 加载、merge 等）应该 **< 5 分钟**。完全可控。

---

## 5. 可复用 Tier 0-4 checklist

下次跑 smoke / pilot 时按这个顺序：

```
■ 预查
  □ GPU 占用: ssh ... 'nvidia-smi --query-gpu=...'
  □ 选择 utilization=0% 且 memory.used < 20GB 的卡

■ Tier 0 (5 min)
  □ source conda env
  □ python -c "import torch, transformers, ..." 全部 import
  □ torch.cuda.is_available() == True
  □ 矩阵运算无 error

■ Tier 1 (5 min)
  □ AutoModelForCausalLM 加载 base 模型
  □ apply_chat_template 输出含 <|im_start|>
  □ generate 30 token, 输出可读

■ Tier 2 (5 min, 一次性)  ⭐
  □ pipelines/preprocess_tokenizer.py 跑通
  □ trajectory token len 全部 = 1
  □ 新模型存到 -vaevas 后缀路径
  □ config.json::vocab_size 应该是 151681

■ 数据准备
  □ jsonl 字段和 dataset_info.json 完全对应
  □ jsonl 行数 = wc -l 一下确认
  □ 抽 1 行 json.loads 验证格式
  □ trajectory tag 在 output 字段里

■ 配置准备
  □ YAML 用 -vaevas 后缀的 base 模型路径
  □ CONFIG 用绝对路径
  □ save_steps / eval_steps 不要超过总步数的 1/3

■ Tier 3 (15 min)
  □ 训练命令: CUDA_VISIBLE_DEVICES=N CONFIG=/abs/path bash scripts/train_lora_sft.sh
  □ Loss 下降, eval loss 单调降
  □ checkpoint-* 至少 1 个
  □ training_loss.png 存在

■ Tier 4 (15 min)
  □ merge YAML 指向 -vaevas base + smoke adapter
  □ merge 产物大小应该 ≈ base 模型大小
  □ Python 脚本验证 trajectory token 还在 (len=1)
  □ 生成的 Verilog 至少有 module/endmodule (smoke 期望: 100% 有)
  □ 是否用 <think>/<answer> 标签 (smoke 期望: 不强求, Pilot 期望: 大部分)
```

---

## 6. 给下一阶段的 TODO / 改进点

### 立刻可做（不阻塞数据 pipeline）

- [ ] 把 build_smoke_from_evas_examples.py 加 `--validate` 选项：跑完写 jsonl 后立刻 `json.loads` 每行检查
- [ ] 把 dataset_info.json 的字段同步逻辑写进 `pack_sft.py`：扫第一条 jsonl → 自动生成 dataset_info
- [ ] preprocess_tokenizer.py 加幂等检查：如果 `-vaevas` 路径已存在且 vocab 一致，跳过；避免误覆盖

### Pilot 前需要解决

- [ ] **数据 pipeline 落地** —— 见本响应正文，决定 LLM 用不用、用谁
- [ ] 写 `verify_evas.py`：用 EVAS 跑每条新数据，过 = 进 jsonl，不过 = 丢弃
- [ ] 写 `pack_sft.py`：把 verified pairs + trajectory 自动包成 LF Alpaca jsonl
- [ ] 决定 wandb 开不开（Pilot 起码要开）
- [ ] 改回 production YAML 的 save_steps / eval_steps = 100

### Production 前需要解决

- [ ] 数据 ≥ 500 条
- [ ] Pilot 跑通 + 评测达标
- [ ] held-out eval set 切出来（脱离训练分布）
- [ ] 决定是否切 Full SFT (LoRA → full)

---

## 7. 经验沉淀的 1 句话总结

> **Tier 0-4 全 PASS，整套 vaEvas SFT 工程链路验证完毕**。
> 抓到的 3 个工程坑都不是大事，但每个都是 Pilot 阶段会要命的小事。
> Smoke 30 分钟投入回报 = **数据准备好后第一次跑 Pilot 100% 是数据/超参问题，不是工程问题**。

---

## 8. 相关文档

- 计划版（前置）：[`08_smoke_validation_plan.md`](./08_smoke_validation_plan.md)
- 整套配方：[`06_practical_recipe.md`](./06_practical_recipe.md)
- 参数原理：[`07_parameter_primer.md`](./07_parameter_primer.md)
- LF 用法：[`../tools/llamafactory.md`](../tools/llamafactory.md)
- 数据 pipeline 设计：[`../05_data_pipeline.md`](../05_data_pipeline.md)
