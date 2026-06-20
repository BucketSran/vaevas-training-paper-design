# SFT — Deep Dive

`docs/01_sft_principles.md` 是一页快速速查。**真正想搞懂的请从这里读。**

> **预览说明**：本目录所有公式用 LaTeX。请用 VSCode 打开 `.md` 文件后按 `Cmd+Shift+V` 开 preview，`markdown-all-in-one` 扩展会自动渲染 KaTeX。GitHub web 预览也原生支持。

## 推荐阅读顺序

每章独立成篇，但有依赖关系。按这个顺序读理解最顺：

1. **[01_foundations.md](./01_foundations.md)** — 数学目标、模仿学习视角、SFT 能教什么不能教什么。所有后续章节的基础。
   - **配套**：[`01a_formulas_walkthrough.md`](./01a_formulas_walkthrough.md) — §1 公式的逐符号解释 + 小数据具体计算。看不懂公式时翻它。
2. **[02_data_and_format.md](./02_data_and_format.md)** — chat template、特殊 token、prompt 掩码、tokenization 对代码任务的影响。
   - **配套**：[`02b_real_world_reference.md`](./02b_real_world_reference.md) — 用真实开源工程 RTL-Coder 把 02 章的抽象概念**逐行映射到可读代码**，并交代 02 章信息来源。
3. **03_training_loop.md**（计划章节）— 优化器、学习率调度、batch 构造、累积、分布式。
4. **04_memory_and_scale.md**（计划章节）— 显存预算 (Adam 状态 / 梯度 / 激活)、ZeRO/FSDP、LoRA 取舍、7B 模型在目标 GPU 设置上的实际数字。
5. **05_diagnostics.md**（计划章节）— 训练曲线读法、健康/警告/停止信号、常见 bug 与诊断步骤。
6. **[06_practical_recipe.md](./06_practical_recipe.md)** ✅ — Qwen2.5-Coder-7B on Verilog-A 完整配方，综合前面所有章节。**结合远端真实环境给出可直接跑的 YAML + 三档训练 recipe**。
7. **[07_parameter_primer.md](./07_parameter_primer.md)** ✅ — **参数原理字典**。06 章告诉你用什么值，本章告诉你每个值是什么、为什么。当成查阅手册用。
8. **[08_smoke_validation_plan.md](./08_smoke_validation_plan.md)** ✅ — **从"环境装好"到"可以正式训练"的 5 层 smoke 验证 + Pilot/Production 路线规划**。今天就能跑 Tier 0-4。
9. **[09_smoke_lessons.md](./09_smoke_lessons.md)** ✅ — **首次 smoke 实战经验沉淀**（2026-06-04）。3 个真实抓到的 bug + 永久规则 + 远端产物地图 + 可复用 checklist。下次 smoke / Pilot 前先扫一眼。

## 什么时候读哪一章

- **现在**（数据还没好）→ 1、2、3、4 反复读，重点 1 和 2。
- **数据 pipeline 跑起来后** → 5、6。
- **第一次训练崩了** → 立刻回到 5，然后回到 3 或 4。

## 与其他文档的关系

- 这里讲 **SFT 是什么、为什么**。
- `../02_grpo_principles.md` 讲 GRPO（SFT 之后才用）。
- `../03_reward_design.md` 讲 reward（GRPO 的灵魂）。
- `../05_data_pipeline.md` 讲数据流水线 + 上下游约束。
- 训练脚本本身在 `train_sft/README.md`。

## 写作风格约定

- 每章独立、可单读
- 数学放在框里、推导步骤完整
- 每个建议附"为什么"，不只是"怎么做"
- 优先讲"非显然但重要"，跳过"教科书都讲过"
