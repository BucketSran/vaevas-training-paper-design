# GRPO — Deep Dive

> **预览**：VSCode `Cmd+Shift+V`，KaTeX 渲染。

> 这一层放 **GRPO（Group Relative Policy Optimization）方法**的概念性学习材料。
> 实际跑训练用的工具栈在 [`../tools/verl.md`](../tools/verl.md)（verl 框架）和 [`../tools/vllm.md`](../tools/vllm.md)（rollout 引擎）。

## 前置阅读（必须先看的 SFT 章节）

GRPO 不能跳过 SFT 直接学。先确保你看完了：

- [`../sft/01_foundations.md`](../sft/01_foundations.md) §7 "SFT → RL 为什么是这个顺序" —— 解释为什么 RL 需要 SFT 冷启动
- [`../sft/01a_formulas_walkthrough.md`](../sft/01a_formulas_walkthrough.md) 公式 7 (GRPO advantage 数值例) —— 让你对 GRPO 数学形式有第一印象
- [`../02_grpo_principles.md`](../02_grpo_principles.md) —— 一页快速速查（高层）

## 推荐阅读顺序

| 章节 | 学什么 | 状态 |
|---|---|---|
| **[00_environment_prep.md](./00_environment_prep.md)** | GRPO 训练栈 (verl + vLLM + EVAS verifier) 环境准备计划 + 4 步 checklist。 | ✅ |
| **[01_foundations.md](./01_foundations.md)** | RL 基础 (state/action/reward) → policy gradient → PPO → **GRPO 推导**。本系列的根。 | ✅ |
| **[01a_formulas_walkthrough.md](./01a_formulas_walkthrough.md)** | GRPO 完整 loss 公式逐符号解释 + 小数据具体计算。从 group advantage 到 PPO clip 到 KL 项一项项算。 | ✅ |
| **[02_reward_shapes_behavior.md](./02_reward_shapes_behavior.md)** | reward 形态如何决定 policy 行为：dense vs sparse / continuous vs binary / reward hacking 模式 / Circuit-Think 的多级 reward 分析 | ✅ |
| **03_training_loop.md** | rollout → score → advantage → 加权对数似然 → update。和 SFT 训练循环的对比。 | 计划章节 |
| **04_stability.md** | KL coef β 调参 / entropy collapse 检测 / reward 跑飞救援 / mode collapse | 计划章节 |
| **05_diagnostics.md** | reward 曲线 / KL 曲线 / entropy 曲线读法 / 5 大死法 | 计划章节 |
| **06_practical_recipe.md** | Qwen2.5-Coder-7B-SFT + EVAS-grounded reward 的 verl 完整配方 | 计划章节 |

## 工具栈学习（强烈建议先看）

| 工具 | 什么时候看 |
|---|---|
| [`../tools/vllm.md`](../tools/vllm.md) | **学 01 之前** —— 知道 rollout engine 是什么 |
| [`../tools/verl.md`](../tools/verl.md) | 学 03/06 之前 —— 知道生产框架长什么样 |

## 与 SFT 学习线的差异

| | SFT (01-09) | GRPO (01-06) |
|---|---|---|
| 训练目标 | 模仿数据分布 | 最大化 reward |
| 数据 | (instruction, completion) | 只要 instruction (RL 自己采 completion) |
| 反向梯度来源 | cross-entropy loss | 加权 log-likelihood by advantage |
| 显存 | 1 模型 + Adam | 3-4 模型 (policy/ref/rollout/critic) |
| 训练框架 | LLaMA-Factory ✅ | verl |
| 推理框架 | (不直接用) | **vLLM 必须** |
| Reward 设计 | (没有 reward) | **核心** —— 决定一切 |

## 总体节奏建议

1. **现在** (路径 B 当前位置) → 读 01 + 01a 建立 RL/GRPO 直觉
2. **数据 pipeline 在准备时** → 读 02 (reward 设计) —— 因为 reward 决定数据需要什么 trajectory
3. **GRPO 训练实操前** → 读 03/04/05/06 + 重读 tools/verl.md

## 写作风格

和 sft/ 系列一致：
- 每章独立成篇，可单读
- 数学放在框里、推导步骤完整
- 每个建议附"为什么"，不只是"怎么做"
- 优先讲"非显然但重要"，跳过"教科书都讲过"
