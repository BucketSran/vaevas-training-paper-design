# 01a — GRPO 公式逐项解读 + 实例计算

> **预览**：VSCode `Cmd+Shift+V`，KaTeX 渲染。
>
> **怎么用**：陪读 [`01_foundations.md`](./01_foundations.md)。每个公式拆到符号级 + 小数据具体算。

> **约定**：$\log$ 都指自然对数 $\ln$。所有数值四舍五入到 3 位小数。

## 符号速查表

| 符号 | 含义 |
|---|---|
| $\theta$ | policy 模型可学参数（policy 是被训练的 actor） |
| $\pi_\theta(y \mid x)$ | 模型在 prompt $x$ 下生成 completion $y$ 的概率 |
| $\pi_{\text{old}}$ | 上一次更新前的 policy（重要性采样用） |
| $\pi_{\text{ref}}$ | 参考 policy（通常是 SFT checkpoint，冻结，KL 项用） |
| $x$ | prompt（一条训练数据的输入）|
| $y, o$ | completion（模型生成的整段回答）|
| $R(y)$ | reward function 对 $y$ 的标量打分 |
| $V^\pi(s)$ | state $s$ 下 policy $\pi$ 的预期 reward（critic 估的）|
| $A(s, y)$ | advantage = $R(y) - V^\pi(s)$，"比平均好多少" |
| $\hat{A}$ | advantage 的**估计量**（不是真值）|
| $b$ | baseline（一个标量，期望意义上不改变梯度但减方差）|
| $N$ | group size（GRPO 中每个 prompt 采的 sample 数）|
| $r_\theta(y)$ | probability ratio $= \pi_\theta(y) / \pi_{\text{old}}(y)$ |
| $\epsilon$ | PPO clip 范围（一般 0.2）|
| $\beta$ | KL 项系数（一般 0.04）|
| $\mathrm{KL}(p \| q)$ | $p$ 关于 $q$ 的 KL 散度（[SFT 01a §5](../sft/01a_formulas_walkthrough.md#公式-5kl-散度)）|

---

## 公式 1：RL 优化目标

$$
\theta^* = \arg\max_\theta \; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot \mid x)} \big[\, R(x, y) \,\big]
$$

### 通俗解释

> 调整模型参数 $\theta$，让"模型按当前 policy 生成的 $y$"的 reward **平均最大**。

### 符号详解

| 符号 | 含义 |
|---|---|
| $\theta^*$ | 最优参数（我们要找的）|
| $\arg\max_\theta$ | "让后面表达式取最大值的那个 $\theta$" |
| $\mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta}$ | 对 prompt $x$ **和** 模型采样的 $y$ 求期望 |
| $R(x, y)$ | reward function，对 (prompt, completion) 打分 |

### 实例计算

设训练集 $\mathcal{D}$ 只有 1 个 prompt：

$$x = \text{"生成一个 inverter"}$$

模型当前 $\pi_\theta$ 在这个 prompt 下生成 4 种可能 completion（简化为 4 个）：

| $y$ | $\pi_\theta(y \mid x)$ | $R(y)$ |
|---|---|---|
| $y_1$ = 正确的 inverter 代码 | 0.5 | 1.0 |
| $y_2$ = 编译过但仿真错 | 0.3 | 0.5 |
| $y_3$ = 语法错的 | 0.15 | 0.0 |
| $y_4$ = 完全乱码 | 0.05 | 0.0 |

当前目标值：

$$
\mathbb{E}_{y \sim \pi_\theta}[R(y)] = 0.5 \times 1.0 + 0.3 \times 0.5 + 0.15 \times 0 + 0.05 \times 0 = \mathbf{0.65}
$$

如果训练后 $\pi_\theta$ 变成 $(0.8, 0.2, 0, 0)$（更偏向正确）：

$$
\mathbb{E}[R] = 0.8 \times 1.0 + 0.2 \times 0.5 = \mathbf{0.90}
$$

→ 训练目标就是把这个数从 0.65 推高到 0.90 / 1.0。

---

## 公式 2：Policy Gradient（REINFORCE）

$$
\nabla_\theta J(\theta) = \mathbb{E}_{y \sim \pi_\theta}\!\big[\, R(y) \cdot \nabla_\theta \log \pi_\theta(y) \,\big]
$$

### 通俗解释

> $J$ 对 $\theta$ 的梯度 = **reward 加权的 log-likelihood 梯度**。
> 训练时：增加 reward 高的 $y$ 的概率，减少 reward 低的 $y$ 的概率。

### 符号详解

| 符号 | 含义 |
|---|---|
| $\nabla_\theta J$ | 目标 $J$ 对参数 $\theta$ 的梯度 |
| $\nabla_\theta \log \pi_\theta(y)$ | log-likelihood 对参数的梯度 —— **跟 SFT loss 的梯度同形**！|
| $R(y) \cdot$ | 用 $R(y)$ 作为这条样本的"权重" |

### 实例计算

继续公式 1 的设定。假设当前 $\pi_\theta = (0.5, 0.3, 0.15, 0.05)$。我们采样 4 次（每个 $y$ 都采到一次，按概率权重）。

**逐项算 $R(y_i) \cdot \nabla \log \pi_\theta(y_i)$**：

我用一个简化的"参数变化"代表 $\nabla \log \pi$（实际是张量；这里只看符号方向）：

| $y$ | $R$ | $\nabla \log \pi$（方向）| $R \cdot \nabla \log \pi$ |
|---|---|---|---|
| $y_1$ | 1.0 | "增加 $y_1$ 概率的方向" | **+1.0 × 那个方向** = 强烈推动 $y_1$ |
| $y_2$ | 0.5 | "增加 $y_2$ 概率的方向" | +0.5 × 那个方向 = 中等推动 |
| $y_3$ | 0.0 | "增加 $y_3$ 概率的方向" | **0 × 任何方向 = 0** = 不推动 |
| $y_4$ | 0.0 | "增加 $y_4$ 概率的方向" | 0 |

**梯度上升后**：$\pi_\theta$ 朝 $y_1$（高 reward）方向偏，远离 $y_3, y_4$。

### 这个公式的"问题"在哪

如果实际上 reward 是 $\{0, 0, 0, 100\}$（高度稀疏）：
- $y_4$ 的 reward = 100 → 梯度乘以 100 → **一步把模型推到只生成 $y_4$**
- $y_4$ 如果不是真正最优（只是这次运气好被采到了），整个 policy 就跑偏

→ **方差太高**。引入 baseline 解决（公式 3）。

---

## 公式 3：Baseline 减少方差

$$
\nabla_\theta J = \mathbb{E}_y\!\big[\, (R(y) - b) \cdot \nabla_\theta \log \pi_\theta(y) \,\big]
$$

### 通俗解释

> Reward 先减去一个 baseline $b$ 再加权。**baseline 不改变期望，但大幅减小梯度方差**。

### 关键证明（为什么 baseline 不改变期望）

$$
\mathbb{E}_y\!\big[ b \cdot \nabla_\theta \log \pi_\theta(y) \big] = b \cdot \mathbb{E}_y\!\big[ \nabla_\theta \log \pi_\theta(y) \big] = b \cdot \nabla_\theta \mathbb{E}_y[1] = b \cdot 0 = 0
$$

因为 $\mathbb{E}_y[1] = \int \pi_\theta(y)\, dy = 1$ 始终成立 → 对 $\theta$ 求导是 0。

### 实例：baseline 如何减方差

假设我们采 5 个 $y$：

| $y_i$ | $R_i$ |
|---|---|
| $y_1$ | 0.9 |
| $y_2$ | 0.6 |
| $y_3$ | 0.5 |
| $y_4$ | 0.4 |
| $y_5$ | 0.1 |

**不用 baseline**：梯度系数是 reward 本身 $\{0.9, 0.6, 0.5, 0.4, 0.1\}$。
- 都是正数，所有样本的梯度都往"增加自己概率"方向
- **政策**只能学到 "$y_1$ 最重要"，但 $y_5$ 也被强化了

**用 baseline $b = \mathrm{mean}(R) = 0.5$**：
- 系数变成 $\{+0.4, +0.1, 0, -0.1, -0.4\}$
- $y_1$ 强化，$y_2$ 弱强化，$y_3$ 不动，$y_4$ 弱抑制，$y_5$ 强抑制
- **方差明显减小，信号方向明确**

**计算方差**：
- 无 baseline 的系数方差 $\approx \mathrm{Var}\{0.9, 0.6, 0.5, 0.4, 0.1\} = 0.080$
- 用 baseline 的系数方差 $= \mathrm{Var}\{0.4, 0.1, 0, -0.1, -0.4\} = 0.080$

诶，这里方差没变？因为 mean baseline 是"中心化"。**实际方差减小的是更高阶量**（梯度估计的 second moment）：

- 无 baseline: $\mathbb{E}[R^2] = \mathrm{Var}(R) + \mathrm{mean}(R)^2 = 0.080 + 0.25 = 0.33$
- 用 baseline: $\mathbb{E}[(R-b)^2] = 0.080$

→ **second moment 减小 4 倍**，对应梯度估计的方差减小 4 倍。

---

## 公式 4：GRPO 的 Group-Normalized Advantage

$$
\hat{A}_i = \frac{R_i - \mathrm{mean}(\{R_1, \dots, R_N\})}{\mathrm{std}(\{R_1, \dots, R_N\})}
$$

### 通俗解释

> 给定 prompt，采 $N$ 个 completion。每个 completion 的 advantage = (它的 reward − 组内平均) ÷ 组内标准差。

### 实例计算（详细版）

prompt $x = $ "生成一个 inverter"。group size $N = 4$。采 4 个 completion：

| $i$ | $o_i$ (简述) | $R_i$ |
|---|---|---|
| 1 | 正确 inverter | 1.0 |
| 2 | 编译过、仿真不对 | 0.5 |
| 3 | 编译过、行为 OK 但格式有问题 | 0.7 |
| 4 | 语法错 | 0.0 |

**第 1 步：算 mean**

$$
\mathrm{mean} = \frac{1.0 + 0.5 + 0.7 + 0.0}{4} = \frac{2.2}{4} = 0.55
$$

**第 2 步：算 std**

差的平方：
| $i$ | $R_i - 0.55$ | $(R_i - 0.55)^2$ |
|---|---|---|
| 1 | $+0.45$ | $0.2025$ |
| 2 | $-0.05$ | $0.0025$ |
| 3 | $+0.15$ | $0.0225$ |
| 4 | $-0.55$ | $0.3025$ |
| 和 | | $0.5300$ |

$$
\mathrm{std} = \sqrt{0.53 / 4} = \sqrt{0.1325} \approx 0.364
$$

**第 3 步：算 $\hat{A}_i$**

| $i$ | $R_i - 0.55$ | $\hat{A}_i = (R_i - 0.55) / 0.364$ |
|---|---|---|
| 1 | $+0.45$ | $\mathbf{+1.236}$ ← 强化 |
| 2 | $-0.05$ | $\mathbf{-0.137}$ ← 微抑制 |
| 3 | $+0.15$ | $\mathbf{+0.412}$ ← 弱强化 |
| 4 | $-0.55$ | $\mathbf{-1.511}$ ← 强抑制 |

### 解读

- $\hat{A}_1 = +1.24$: $o_1$ 在这组里**显著超平均**，policy 应该增加生成它的概率
- $\hat{A}_2 = -0.14$: $o_2$ 接近平均，几乎不调
- $\hat{A}_3 = +0.41$: $o_3$ 略好于平均，弱强化
- $\hat{A}_4 = -1.51$: $o_4$ **显著低于平均**，强抑制

**验证：$\sum_i \hat{A}_i \approx 0$**（归一化的副产品）：

$$1.236 - 0.137 + 0.412 - 1.511 = 0.000 \;\; \checkmark$$

正负 advantage 平衡，符合"减 baseline"的数学要求。

### 极端情况：退化到 NaN（已在 [SFT 01a 公式 7](../sft/01a_formulas_walkthrough.md#公式-7grpo-advantageRL-阶段会用-先放着熟悉一下) 讲过）

| 情况 | $\{R_i\}$ | $\mathrm{mean}$ | $\mathrm{std}$ | $\hat{A}$ |
|---|---|---|---|---|
| 全失败 | $\{0, 0, 0, 0\}$ | 0 | 0 | **NaN** |
| 几乎相等 | $\{0.30, 0.30, 0.31, 0.29\}$ | 0.30 | 0.007 | $\hat{A}_i \in \pm 10+$ **爆炸** |
| 健康 | $\{0.9, 0.6, 0.3, 0.2\}$ | 0.5 | 0.274 | $\hat{A}_i \in \pm 1.5$ **干净** |

→ 这数学上**要求 SFT 必须在 GRPO 之前**：把 reward 分布从全 0/全 1 推到中等，让 std 有合理值。

---

## 公式 5：Probability Ratio（重要性采样）

$$
r_\theta(y) = \frac{\pi_\theta(y \mid x)}{\pi_{\text{old}}(y \mid x)}
$$

### 通俗解释

> 当前 policy 给 $y$ 的概率 ÷ 旧 policy 给 $y$ 的概率。
> $r = 1$ 意味着新旧 policy 一样；$r > 1$ 意味着新 policy 更喜欢 $y$。

### 为什么需要它

GRPO 训练**先采样后多步更新**：
1. 用 $\pi_{\text{old}}$ 采 $N$ 个 sample（这一步固定）
2. 用这些 sample 算 advantage（固定）
3. 优化 $\theta$ 多步 → $\pi_\theta$ 偏离 $\pi_{\text{old}}$
4. 每一步用 importance weight $r_\theta$ 修正"采样分布和当前 policy 不一样"

### 实例计算

prompt $x$，某个 sample $y$。在 token level，$\pi(y \mid x) = \prod_t \pi(y_t \mid x, y_{<t})$，是各 token 概率连乘。简化为整段：

| 时刻 | $\pi_{\text{old}}(y \mid x)$ | $\pi_\theta(y \mid x)$ | $r_\theta(y)$ |
|---|---|---|---|
| t=0（刚开始） | 0.0010 | 0.0010 | $1.0$ |
| 第 5 步更新后 | 0.0010 | 0.0015 | $1.5$ |
| 第 10 步更新后 | 0.0010 | 0.0030 | **$3.0$** ← 太大 |

**$r > 1.5$ 已经很激进**，$r > 3$ 通常意味着 policy 已经偏离 $\pi_{\text{old}}$ 太多，**这时候用旧样本估梯度不再可信**。

→ 解决方案：**clip**（公式 6）。

---

## 公式 6：PPO Clipped Loss（也就是 GRPO 主项）

$$
L^{\text{clip}}_i = \min\!\big(\, r_\theta(o_i) \cdot \hat{A}_i,\;\; \mathrm{clip}(r_\theta(o_i),\, 1-\epsilon,\, 1+\epsilon) \cdot \hat{A}_i \,\big)
$$

### 通俗解释

> 想最大化 $r_\theta \cdot \hat{A}$，但**给 $r_\theta$ 装个"围栏"**：超过 $(1-\epsilon, 1+\epsilon)$ 范围就不再奖励 / 惩罚。

### 符号详解

| 符号 | 含义 |
|---|---|
| $\epsilon$ | clip 范围，默认 0.2 |
| $\mathrm{clip}(x, a, b)$ | 把 $x$ 裁到 $[a, b]$: $\max(a, \min(x, b))$ |
| $\min(\cdot, \cdot)$ | 取两项里**较小**的（保守的）|

### 为什么取 min

四种情况分析（$\epsilon = 0.2$，所以裁到 $[0.8, 1.2]$）：

| 情况 | $\hat{A}$ | $r_\theta$ | 没 clip 项 $r \cdot \hat{A}$ | clip 项 $\mathrm{clip}(r) \cdot \hat{A}$ | $\min$ |
|---|---|---|---|---|---|
| A | $+1$ | 1.5 | $1.5$ | $1.2$ | **1.2** ← clip 起作用 |
| B | $+1$ | 0.5 | $0.5$ | $0.8$ | **0.5** ← 不 clip |
| C | $-1$ | 1.5 | $-1.5$ | $-1.2$ | **-1.5** ← 不 clip |
| D | $-1$ | 0.5 | $-0.5$ | $-0.8$ | **-0.8** ← clip 起作用 |

**直觉**：
- $\hat{A} > 0$（好 sample）：想让 $r$ 涨，但**涨过 $1+\epsilon$ 就不再给奖励**（case A）
- $\hat{A} < 0$（差 sample）：想让 $r$ 降，但**降到 $1-\epsilon$ 以下就不再给惩罚**（case D）
- 反向（case B, C）：clip 不限制，保留信号

→ **效果**：policy 不会**激进地远离 $\pi_{\text{old}}$**，但**保留向后退（修正错误）的自由**。

### 实例计算

设 $N = 4$, $\epsilon = 0.2$, $\hat{A} = \{+1.24, -0.14, +0.41, -1.51\}$（来自公式 4），假设当前 $r_\theta = \{1.3, 1.0, 0.7, 0.9\}$：

| $i$ | $\hat{A}_i$ | $r_\theta$ | $\mathrm{clip}(r, 0.8, 1.2)$ | $r \cdot \hat{A}$ | $\mathrm{clip}(r) \cdot \hat{A}$ | $\min$ = $L^{\text{clip}}_i$ |
|---|---|---|---|---|---|---|
| 1 | $+1.24$ | 1.3 | 1.2 | $1.612$ | $1.488$ | **1.488** (clipped) |
| 2 | $-0.14$ | 1.0 | 1.0 | $-0.14$ | $-0.14$ | $-0.14$ |
| 3 | $+0.41$ | 0.7 | 0.8 | $0.287$ | $0.328$ | **0.287** (not clipped) |
| 4 | $-1.51$ | 0.9 | 0.9 | $-1.359$ | $-1.359$ | $-1.359$ |

总平均 $\frac{1}{N}\sum L^{\text{clip}}_i = (1.488 - 0.14 + 0.287 - 1.359)/4 = 0.069$

→ 这就是 GRPO 主 loss 的一个 step 的值。

---

## 公式 7：KL 项（防偏离参考模型）

$$
L^{\text{KL}} = \beta \cdot \mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})
$$

### 通俗解释

> 加一个"拉绳子"项：**当前 policy 偏离参考 policy 越远，惩罚越大**。

### 符号详解

| 符号 | 含义 |
|---|---|
| $\beta$ | KL 系数，默认 0.04 |
| $\mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})$ | $\pi_\theta$ 关于 $\pi_{\text{ref}}$ 的 KL 散度（见 [SFT 01a 公式 5](../sft/01a_formulas_walkthrough.md#公式-5kl-散度)）|
| $\pi_{\text{ref}}$ | 参考 policy，**通常是 SFT checkpoint**（冻结） |

### 为什么用 SFT 而不是 base 当参考

- $\mathrm{KL}(\pi_\theta \| \pi_{\text{base}})$：把 policy 拉回 base → **擦除 SFT 学到的格式 / 风格**。糟糕。
- $\mathrm{KL}(\pi_\theta \| \pi_{\text{SFT}})$：把 policy 拉回 SFT → **保留 SFT 学到的能力，只用 RL 微调**。正确。

### 实例：$\beta$ 影响有多大

假设 KL 散度本身 = 5 nat（policy 偏离 ref 中等程度）。

| $\beta$ | $\beta \cdot \mathrm{KL}$ | 占 loss 比重 | 行为 |
|---|---|---|---|
| 0 | 0 | 0 | 没约束，policy 自由乱跑 |
| 0.01 | 0.05 | 小 | 弱约束 |
| **0.04** | 0.2 | 中 | **DeepSeekMath / Circuit-Think 默认** |
| 0.1 | 0.5 | 大 | 强约束 |
| 1.0 | 5.0 | 主导 | 几乎不学 RL 信号 |

→ **$\beta$ 调太大** → 模型几乎不更新；**$\beta$ 调太小** → policy 偏离 ref 太远 → 推理时输出乱码。**0.04 是甜点**，对应"轻度软约束"。

---

## 公式 8：GRPO 完整 Loss

把所有项合起来：

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}\!\left[\,
\underbrace{\frac{1}{N} \sum_{i=1}^N L^{\text{clip}}_i}_{\text{Clipped PG (公式 6)}}
\;-\;
\underbrace{\beta \cdot \mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})}_{\text{KL 项 (公式 7)}}
\,\right]
$$

### 这一个 step 的端到端流程

总结公式 1-8，一步训练做的事：

```
1. (Rollout 阶段)
   ├─ 给定 prompt x
   ├─ 用 π_θ (= π_old) 采样 N=8 个 completion {o_1, ..., o_8}
   └─ 用 reward function 算 {R_1, ..., R_8}

2. (Advantage 阶段, 公式 4)
   ├─ mean(R), std(R)
   └─ Â_i = (R_i - mean) / std

3. (Loss 阶段, 公式 6 + 7)
   ├─ 对每个 o_i 算 r_θ(o_i) = π_θ(o_i) / π_old(o_i)
   ├─ L_clip_i = min(r·Â, clip(r,1-ε,1+ε)·Â)
   ├─ KL_term = β · KL(π_θ || π_ref)
   └─ J = mean(L_clip) - KL_term

4. (Optimization 阶段)
   ├─ 反向传播 ∇J
   ├─ Adam 更新 θ
   └─ 同步新 θ 给 vLLM rollout engine

5. (重复)
```

### 数值示例（合并前面所有数字）

| 项 | 值 |
|---|---|
| $\hat{A}_i$ (4 个) | $\{+1.24, -0.14, +0.41, -1.51\}$ |
| $r_\theta(o_i)$ | $\{1.3, 1.0, 0.7, 0.9\}$ |
| $\mathrm{mean}(L^{\text{clip}})$ | $0.069$ (公式 6 算出来的) |
| 假设 $\mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})$ | $5.0$ nat |
| $\beta$ | $0.04$ |
| $\beta \cdot \mathrm{KL}$ | $0.20$ |
| **$\mathcal{J} = 0.069 - 0.20 = \mathbf{-0.131}$** | 负值（loss 是要 maximize 的目标）|

→ 这一步 GRPO 的目标值 = $-0.131$。梯度上升把它往正向推。

---

## 公式 9：实务 — 怎么把 trajectory-level reward 分摊到 token-level

GRPO 的 reward 是**整段 completion 一个标量**，但梯度计算是 **token-level**（每个 token 都要算 $\nabla \log \pi(y_t \mid \dots)$）。怎么"分配"reward？

**最简单的做法（GRPO 默认）**：**整段 completion 的所有 token 共享同一个 advantage**。

即：

$$
L_i = \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min(r_\theta(t) \cdot \hat{A}_i, \, \mathrm{clip}(r_\theta(t)) \cdot \hat{A}_i)
$$

其中 $r_\theta(t) = \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{old}}(y_t \mid x, y_{<t})}$ 是**token-level** ratio。

### 实例

$o_i$ 有 100 个 token，整体 reward = 0.7，$\hat{A}_i = +1.24$。**所有 100 个 token 都被乘以 +1.24**。第 50 个 token 和第 99 个 token 的"贡献"一样。

### 这合理吗

**对短 completion 不错；对长 completion 有点粗暴**。最后几个 token 决定 reward（最后才看 compile 结果），但所有 token 都被同等强化/弱化。

→ 这就是"credit assignment problem"。GRPO 不解决，依赖 SFT 让 policy 已经 "大致" 知道怎么生成结构化输出。如果生成完全乱，GRPO 也救不了。

### 替代方案（Circuit-Think 用的）

Circuit-Think 引入**多级 reward**，给不同步骤独立打分，把"长 trajectory 共享一个 reward" → 拆成"每个子步骤独立 reward + 加权和"。这样 advantage 信号在 token 上分布更精确。

详见 [`02_reward_shapes_behavior.md`](./02_reward_shapes_behavior.md)（下一章）。

---

## 最后：一个 cheat sheet 串起所有公式

```
公式 1 (RL 目标)              ← 最终目标：最大化 E[R]
       │
       │ ∇_θ
       ▼
公式 2 (Policy Gradient REINFORCE) ← R · ∇log π，朴素但方差大
       │
       │ 加 baseline
       ▼
公式 3 (Baseline)             ← 减 b 不改期望但减方差
       │
       │ b = mean(R) of group N
       ▼
公式 4 (GRPO Advantage Â)     ← group baseline + std 归一化
       │
       │ + 重要性采样 + clip
       ▼
公式 5+6 (PPO Clipped Loss)   ← 安全围栏，防止偏离 π_old 太多
       │
       │ + KL 项
       ▼
公式 7 (KL Term)              ← 防止累积偏离 π_ref 太多
       │
       │ 合体
       ▼
公式 8 (GRPO 完整 Loss)       ← 实际训练的目标
```

**核心一句**：**GRPO = "用 group baseline 减方差" + "用 clip 限单步偏离" + "用 KL 限累积偏离" 三件事的合体**，外加一个 reward function 你自己定义。

---

## 下一步

读完本章后，再读：

- [`01_foundations.md`](./01_foundations.md) §8 重看"为什么必须先 SFT"的数学（现在应该一眼就懂了）
- [`02_reward_shapes_behavior.md`](./02_reward_shapes_behavior.md) —— 公式都过了，下一步是 reward 怎么设计
- [`../tools/verl.md`](../tools/verl.md) §5 —— 看实际 verl YAML，把这些公式映射到配置字段
- [`../sft/01a_formulas_walkthrough.md`](../sft/01a_formulas_walkthrough.md) 公式 7 —— 之前的"打底版"，现在应该觉得"原来如此"

如果某个公式还想要更深（比如 importance sampling 的方差分析、KL 的 unbiased estimator），告诉我具体哪一节，我加专题。
