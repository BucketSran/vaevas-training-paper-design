# 01 — GRPO Foundations: 从 RL 到 GRPO 的完整推导

> **预览**：VSCode `Cmd+Shift+V`，KaTeX 渲染。
>
> **公式看不懂？** → 配套 [`01a_formulas_walkthrough.md`](./01a_formulas_walkthrough.md)，每个公式拆到符号级 + 小数据具体算到底。

> 读完这一章，你应该能回答：
> 1. RL 解决的是 SFT 解决不了的什么问题？
> 2. Policy Gradient 是什么？为什么需要 baseline？
> 3. PPO 比 vanilla policy gradient 强在哪？
> 4. GRPO 比 PPO 简化了什么？为什么 LLM RL 都用 GRPO 不用 PPO？
> 5. 为什么 GRPO 必须先 SFT？(用数学回答, 不是"loss 爆炸"这种含糊话)

---

## 1. 从 SFT 到 RL：到底变了什么

[SFT 01 §5](../sft/01_foundations.md#5-sft-能教什么-vs-不能教什么) 列了 SFT 能教什么 / 不能教什么。**RL 是为解决 SFT 教不了的两件事而生**：

| SFT 能教 | SFT 不能教 | RL 怎么补 |
|---|---|---|
| 输出格式、风格 | "正确性" —— 比如代码是否能编译/仿真 | reward 直接定义"正确" |
| 任务条件化 | 在多个对答案中**选最好的** | reward 排序 = 隐式偏好 |
| 模仿数据分布 | 探索数据里没有的更好答案 | sampling + reward 引导探索 |

**核心区别**：

| | SFT | RL |
|---|---|---|
| 信号来源 | 标准答案 (one-hot label) | 标量 reward |
| 学习方式 | "模仿这个 token" | "增加得高 reward 的整段输出的概率" |
| 数据需要 | (input, output) 对 | 只要 input；output 自己生成 |
| 信号粒度 | 每个 token 都有 label | 整段输出才有一个 reward |

→ **RL 直接对齐"任务目标"，不依赖人写出"完美答案"** —— 只要你能定义"好"（reward function），RL 就能让模型朝那个方向走。

## 2. RL 基础三要素：State / Action / Reward

经典 RL 框架：

```
              ┌──────────┐
   state →    │  Agent   │  → action
              └──────────┘
                   ↑
                   │ reward
              ┌──────────┐
              │Environment│
              └──────────┘
```

对应到 LLM RL：

| RL 概念 | LLM RL 里是什么 |
|---|---|
| **State** $s$ | prompt + 已生成的部分 token（i.e., $(x, y_{<t})$）|
| **Action** $a$ | 下一个 token $y_t$ |
| **Policy** $\pi_\theta(a \mid s)$ | LLM 在给定上下文下输出下一个 token 的概率分布 |
| **Trajectory / Rollout** | 完整生成 $(y_1, y_2, \dots, y_T)$，给定 prompt $x$ |
| **Reward** $R$ | 整段 completion 的得分（compile pass = 1, fail = 0；或者更细的多级 reward）|

**关键观察**：在 LLM 里 reward 是 **trajectory-level**（一整段输出才有一个分），不是 token-level（每个 token 一个分）。这就是后面"advantage 怎么分摊到 token"的问题来源。

## 3. RL 的优化目标

我们要找最优 policy $\pi^*_\theta$ 让**期望 reward 最大**：

$$
\theta^* = \arg\max_\theta \; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot \mid x)} \big[\, R(x, y) \,\big]
$$

拆解：
- $x \sim \mathcal{D}$：prompt 从数据集采样
- $y \sim \pi_\theta(\cdot \mid x)$：completion 由模型生成（关键：不是数据集里给的）
- $R(x, y)$：reward function 给这对打分
- 外层取期望：对所有可能的 $(x, y)$ 求平均

**注意**和 SFT 目标的差别：

| | 信号 | 期望对象 |
|---|---|---|
| SFT | $\log P_\theta(y \mid x)$ | $(x, y) \sim \mathcal{D}$ (固定的对) |
| RL | $R(x, y)$ | $x \sim \mathcal{D}, y \sim \pi_\theta$ (动态采样) |

→ RL 的训练数据是**模型自己生成的**，每一步 $\pi_\theta$ 更新后下一步采样分布也变 —— 这是 RL 比 SFT 不稳的根源。

## 4. Policy Gradient：怎么对这个目标求导

目标 $J(\theta) = \mathbb{E}_{y \sim \pi_\theta}[R(y)]$（省略 $x$ 简化记号）。要梯度上升，需要 $\nabla_\theta J$。

直接对 $\mathbb{E}$ 求导有个问题：**期望本身依赖 $\theta$**（因为采样分布是 $\pi_\theta$）。不能简单换序。

经典推导用 **log-derivative trick**：

$$
\nabla_\theta J(\theta) = \nabla_\theta \int \pi_\theta(y) R(y)\, dy = \int \nabla_\theta \pi_\theta(y) \cdot R(y)\, dy
$$

利用 $\nabla_\theta \pi_\theta(y) = \pi_\theta(y) \cdot \nabla_\theta \log \pi_\theta(y)$：

$$
\nabla_\theta J(\theta) = \int \pi_\theta(y) \cdot \nabla_\theta \log \pi_\theta(y) \cdot R(y)\, dy = \mathbb{E}_{y \sim \pi_\theta}\!\big[\, R(y) \cdot \nabla_\theta \log \pi_\theta(y) \,\big]
$$

→ **REINFORCE 算法的核心公式**。

### 4.1 这个公式的直觉

$$
\boxed{\nabla_\theta J = \mathbb{E}_y\!\big[ R(y) \cdot \nabla_\theta \log \pi_\theta(y) \big]}
$$

读法：**"想让 $J$ 上升 → 增加得高 reward 的 $y$ 的对数概率"**。

具体操作（一个 step）：
1. 用当前 $\pi_\theta$ 采样 $y$
2. 算 reward $R(y)$
3. 算梯度 $\nabla_\theta \log \pi_\theta(y)$（这就是 SFT loss 的梯度，按 $y$ 反向传播）
4. **用 $R(y)$ 加权梯度**，再做梯度上升

**SFT 是 $\log P_\theta$ 直接对所有 (x, y) 求和；RL 是 $R \cdot \log P_\theta$ 按 reward 加权**。SFT 等价于 "$R \equiv 1$" 的 RL —— 所有 y 一样重要。RL 让"好的 y"权重大、"差的 y"权重小（甚至负）。

### 4.2 这个公式的问题：高方差

朴素 REINFORCE 训练 RL 几乎不能用，因为**方差太高**：

- $R(y)$ 通常是 0 或 1（稀疏）或者数值很大（密集但绝对值大）
- 模型采到一个高 reward 的 $y$ 时，梯度被乘以一个大的正数 → 一步乱跑
- 模型采到一个低 reward 的 $y$ 时，梯度被乘以小数 / 0 / 负数 → 抵消之前的进步

→ **训练曲线像癫痫**。

## 5. 解决方差问题：引入 Baseline

把 reward 减去一个 baseline $b$ 不改变梯度的**期望**，但能**大幅减小方差**。

数学：

$$
\nabla_\theta J = \mathbb{E}_y\!\big[ (R(y) - b) \cdot \nabla_\theta \log \pi_\theta(y) \big]
$$

证明 baseline 不影响期望：

$$
\mathbb{E}_y\!\big[ b \cdot \nabla_\theta \log \pi_\theta(y) \big] = b \cdot \nabla_\theta \int \pi_\theta(y)\, dy = b \cdot \nabla_\theta(1) = 0
$$

(因为 $\int \pi = 1$ 总成立。)

### 5.1 Baseline 怎么选

经典选择：**$b = V^\pi(s)$**，即"在当前 state 下的预期 reward"。

直觉：
- $R(y) > V(s)$：这个 $y$ **比预期好**，强化它
- $R(y) < V(s)$：这个 $y$ **比预期差**，弱化它

定义 **Advantage**：

$$
A(s, y) = R(y) - V^\pi(s)
$$

→ Advantage = "实际 reward 比平均高多少"。Policy gradient 改写：

$$
\nabla_\theta J = \mathbb{E}\!\big[ A(s, y) \cdot \nabla_\theta \log \pi_\theta(y \mid s) \big]
$$

### 5.2 但 $V^\pi$ 怎么估？

PPO 等"actor-critic"方法的做法：**另训一个网络 $V_\phi$ 来预测 $V^\pi$**（叫 critic）。

代价：
- 多一个模型 = 显存 ×2
- Critic 自己也要训（梯度+损失）
- Critic 不准时，advantage 也偏

→ GRPO 后面会用一个**更简单的 baseline 替代 critic**（§7）。

## 6. PPO：处理"采样和更新分布不同"的问题

REINFORCE 还有个问题：**采样用的是旧 policy $\pi_{old}$，更新对象是新 policy $\pi_\theta$**。如果两者差太远，估计就不准。

PPO 通过**重要性采样**和**clip** 限制这个差距。

### 6.1 重要性采样

如果想用 $\pi_{old}$ 采样估 $\pi_\theta$ 下的期望：

$$
\mathbb{E}_{y \sim \pi_\theta}[f(y)] = \mathbb{E}_{y \sim \pi_{old}}\!\Big[\, \frac{\pi_\theta(y)}{\pi_{old}(y)} \cdot f(y) \,\Big]
$$

那个比值 $r_\theta(y) = \pi_\theta(y) / \pi_{old}(y)$ 是**重要性权重 / probability ratio**。

代入 policy gradient：

$$
\nabla_\theta J = \mathbb{E}_{y \sim \pi_{old}}\!\big[ r_\theta(y) \cdot A(s, y) \cdot \nabla_\theta \log \pi_\theta(y \mid s) \big]
$$

但 $r_\theta$ 可能很大（如果 $\pi_\theta$ 给了某个 $y$ 远高于 $\pi_{old}$ 的概率），导致更新过激。

### 6.2 PPO 的 clip

PPO 的 loss 形式（最大化）：

$$
L^{\text{PPO}}(\theta) = \mathbb{E}_y\!\Big[\, \min\!\big(\, r_\theta \cdot A,\;\; \text{clip}(r_\theta,\, 1-\epsilon,\, 1+\epsilon) \cdot A \,\big) \,\Big]
$$

含义：
- 如果 advantage 是正的 → 想让 $r_\theta$ 涨；但如果涨过 $1+\epsilon$，clip 住，**不再奖励超过这个界限**
- 如果 advantage 是负的 → 想让 $r_\theta$ 降；但如果降到 $1-\epsilon$ 以下，clip 住，**不再惩罚**

**直觉**：**给 policy update 一个安全围栏**。新 policy 偏离旧 policy 太多时，loss 不再给信号，防止训飞。

### 6.3 PPO 的完整 loss

实际还要加 KL 项防止偏离参考模型（通常是 SFT 模型）太远：

$$
L^{\text{PPO+KL}}(\theta) = L^{\text{PPO}}(\theta) - \beta \cdot \mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})
$$

其中 $\pi_{\text{ref}}$ 是冻结的参考 policy（通常是 SFT checkpoint）。

→ **两层防御**：clip 防止单步偏离 $\pi_{old}$ 太多；KL 项防止累积偏离 $\pi_{\text{ref}}$ 太多。

### 6.4 PPO 的痛点

| 痛点 | 来源 |
|---|---|
| 需要 critic 网络 | $V^\pi$ 估计要单独训 |
| 显存 = 2× model 大小 | actor + critic |
| Critic 调参难 | critic 自己的 loss 也得监控 |
| Critic 拖累整体 | 早期 critic 不准 → advantage 噪 → 训练不稳 |

→ 这些是 GRPO 想避开的。

## 7. GRPO：用 Group Baseline 替代 Critic

DeepSeekMath (Shao et al. 2024) 提出 GRPO，核心一句话：

> **不要 critic。对每个 prompt 采 N 个 completion，用这 N 个的平均 reward 当 baseline。**

### 7.1 GRPO 的 advantage 公式

给定 prompt $x$，采样 $N$ 个 completion $\{o_1, \dots, o_N\}$，得到各自的 reward $\{R_1, \dots, R_N\}$。每个 completion 的 **Group-normalized Advantage**：

$$
\boxed{\;\hat{A}_i = \frac{R_i - \mathrm{mean}(\{R_1, \dots, R_N\})}{\mathrm{std}(\{R_1, \dots, R_N\})}\;}
$$

读法：
- **分子** $R_i - \mathrm{mean}$：去掉组内平均 reward（baseline）
- **分母** $\mathrm{std}$：按组内 reward 散度归一化（让数值落在合理范围）

### 7.2 GRPO 的完整 loss

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}\!\left[\,
\frac{1}{N} \sum_{i=1}^N
\min\!\Big( r_\theta(o_i) \cdot \hat{A}_i,\;
\text{clip}(r_\theta(o_i),\, 1-\epsilon,\, 1+\epsilon) \cdot \hat{A}_i \Big)
\;-\; \beta \cdot \mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})
\,\right]
$$

和 PPO 完全同形，**唯一区别是 $\hat{A}_i$ 怎么算**。

### 7.3 GRPO 替 PPO 省掉了什么

| 组件 | PPO | GRPO |
|---|---|---|
| Actor (policy) | ✅ | ✅ |
| Reference (KL) | ✅ | ✅ |
| **Critic ($V_\phi$)** | ❌ **不要了** | ✅ 需要 |
| 显存 (7B) | 2× = ~28 GB | 1× = ~14 GB |
| Critic loss 调参 | 需要 | 不需要 |
| 训练复杂度 | 高 | 中 |

→ GRPO 把"PPO 里最难调的 critic"换成"廉价的组内平均"，**显存省一半，调参少一类**。

### 7.4 为什么 group baseline 有效

直觉：**同一个 prompt 下，N 个 sample 在统计上是"相互参考"的**。

- 高于组平均的 sample → "在这个 prompt 下，你比同组其他人做得好" → 强化
- 低于组平均的 sample → 弱化

**关键好处**：自动**适应任务难度**。
- 简单 prompt 下 N 个 sample 都得 0.9 → 大家都好 → advantage 接近 0 → 不学（已经够好了）
- 难 prompt 下 N 个 sample 散布 0.1-0.8 → advantage 区分度大 → 学得激进

→ 这就是 group baseline 替代 $V^\pi$ 的根本原理。$V^\pi$ 是"长期平均",  group mean 是"当下当地平均"。后者其实**更适合 LLM 的稀疏 reward 场景**。

## 8. 为什么 GRPO 必须先 SFT：数学回答

[SFT 01 §7](../sft/01_foundations.md#7-sft--rl为什么是这个顺序不是反过来) 给过结论："SFT 是 RL 的入场券"。这里用 GRPO 公式精确说明为什么。

考察 advantage 在两种极端情况下：

### 情况 A：基座模型 5% compile rate

8 个 sample 几乎全 $R = 0$，偶尔有 1 个 $R = 1$：

$$
\{R_i\} = \{0, 0, 0, 0, 0, 0, 1, 0\}
$$

- $\mathrm{mean} = 1/8 = 0.125$
- $\mathrm{std} \approx 0.33$
- 第 7 个 sample: $\hat{A}_7 = (1 - 0.125) / 0.33 = 2.65$ —— **过大**
- 其他 sample: $\hat{A}_i = -0.125 / 0.33 = -0.38$

第 7 个的梯度被放大 2.65 倍 → 一步乱跑 → policy 偏离 reference 很远 → 后续 KL 暴增 → 训练崩。

### 情况 B：所有 sample 都失败

$$
\{R_i\} = \{0, 0, 0, 0, 0, 0, 0, 0\}
$$

- $\mathrm{mean} = 0$
- $\mathrm{std} = 0$
- $\hat{A}_i = 0 / 0 = \mathrm{NaN}$ —— **数学不定义**

实现上通常会 skip 这个 group（loss 不更新）。但如果 100% 的 prompt 都这样 → policy **永远不更新** → 训练完全卡死。

### 情况 C：SFT 后的模型，30-50% compile rate

$$
\{R_i\} = \{0, 1, 1, 0, 1, 0, 0, 1\}
$$

- $\mathrm{mean} = 0.5$
- $\mathrm{std} = 0.5$
- 成功的: $\hat{A} = 0.5 / 0.5 = 1$
- 失败的: $\hat{A} = -0.5 / 0.5 = -1$

**优势恰好在 ±1 量级，advantage 信号干净，policy update 稳**。

→ 所以 SFT 的角色是：把基础成功率从 ~5% 推到 ~30-50%，让 GRPO 的 group baseline 有方差。**这不是"经验法则"，是数学要求**。

## 9. GRPO 在 LLM 上有效的额外原因

除了"省 critic"，GRPO 还有几个 LLM-特有的优势：

### 9.1 Reward 模型可以不可微

PPO 的 critic 要从 reward 反向传播，**reward 必须可微**（或者要 reward model 是个神经网络）。

GRPO 不反传 reward，只用 reward 的标量值。**reward 可以是任何东西**：
- ✅ 程序的执行结果（compile pass / fail）
- ✅ 外部 verifier（EVAS 仿真）
- ✅ 字符串匹配
- ✅ 复杂的多级 reward (Circuit-Think 那种)

→ 这就是为什么 vaEvas 任务非常适合 GRPO：reward 来自 EVAS 验证，不是网络。

### 9.2 Group baseline 天然 prompt-adaptive

LLM 的 reward 分布在不同 prompt 上差别巨大：
- "写个 inverter" → 容易，几乎都过
- "写个 PLL" → 难，大部分失败

PPO 用全局 $V^\pi$ 估计无法兼顾两者。GRPO 每个 prompt **独立** group baseline，天然处理。

### 9.3 与 vLLM rollout 高度协同

GRPO 的核心瓶颈是 rollout（每 prompt 采 N 个 completion）。vLLM 的 PagedAttention 在多 sample 同 prompt 时有**prefix caching**（多个 sample 共享 prompt 的 KV cache）→ 内存效率 ×N，时间也快很多。

→ 这是为什么 verl / slime 等 GRPO 框架的 rollout 引擎几乎都是 vLLM。详见 [`../tools/vllm.md`](../tools/vllm.md) §2.4。

## 10. 常见误区

### "GRPO 是新的 RL 算法"
错。它就是 **PPO 把 critic 换成 group baseline**。PPO 的 clip / KL / 重要性采样全保留。

### "Reward 越复杂越好"
错。Reward 是 GRPO 的灵魂，但**复杂 reward = 多个 reward hack 入口**。先用最简单的 reward (compile pass = 1) 跑通，再分层加。

### "更多 sample N 总是好的"
不绝对。N 越大，advantage 估计越准，但显存和时间线性涨。Circuit-Think 用 $N = 8$，业界主流 8-16。**先试 8**。

### "SFT 越久，GRPO 起点越好"
错到极致。SFT 训太久 → 模式坍缩、过拟合 → group 内全部 sample 答案一致 → std 趋 0 → advantage 爆炸。**SFT 训到 compile rate ~50% 就停，给 GRPO 留探索空间**。

### "GRPO 可以替代 SFT"
看任务。Math reasoning 这种 base model 也偶尔能蒙对的任务 → GRPO 可以 "from scratch"。但 vaEvas 这种 base model 几乎 0% compile 的任务，**SFT 必不可少**。

## 11. 本章核心 take-aways

1. **RL 解决 SFT 教不了的两件事**：正确性 + 多答案选优。Reward 直接对齐任务目标。
2. **Policy Gradient = "把 reward 当权重，加权 log-likelihood"**。$\nabla J = \mathbb{E}[R \cdot \nabla \log \pi]$
3. **Baseline 不改变期望但减方差**。$\hat{A} = R - b$ 是 RL 工程的灵魂。
4. **PPO = REINFORCE + 重要性采样 + clip + KL**。安全围栏防止训飞。代价：必须 critic 估 $V^\pi$。
5. **GRPO = PPO 但 baseline 换成 group mean**。省掉 critic，显存 ×0.5，调参少一类。
6. **GRPO 的 advantage 数学逼着你先 SFT**：基座模型 reward 全 0 → std=0 → advantage NaN/爆炸 → 训不起来。
7. **GRPO 天然适合 LLM**：reward 可以是任意 verifier 输出 / group baseline 自适应 prompt 难度 / 与 vLLM rollout 协同。

---

## 12. 下一步

读完概念后：

- **想看公式逐项算到底** → [`01a_formulas_walkthrough.md`](./01a_formulas_walkthrough.md)
- **想看 reward 怎么设计** → [`02_reward_shapes_behavior.md`](./02_reward_shapes_behavior.md)（下一章）
- **想看 verl 框架怎么用** → [`../tools/verl.md`](../tools/verl.md)
- **想看 vLLM 怎么做 rollout** → [`../tools/vllm.md`](../tools/vllm.md)

---

## 13. 关键参考文献

- **DeepSeekMath**（Shao et al. 2024, arXiv 2402.03300）— GRPO 原始论文
- **DeepSeek-R1**（Guo et al. 2025, arXiv 2501.12948）— SFT → GRPO 在 LLM reasoning 上的标志性应用
- **PPO 原文**（Schulman et al. 2017, arXiv 1707.06347）— PPO 的提出
- **Circuit-Think**（Jiang et al. 2026, AAAI）— GRPO 在电路领域的应用 + 多级 reward 设计

详见 [`../REFERENCES.md`](../REFERENCES.md)。
