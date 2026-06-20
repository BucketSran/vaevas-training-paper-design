# 01a — 公式逐行解读 + 实例计算

> **怎么用这份文档**：和 [`01_foundations.md`](./01_foundations.md) 配套读。`01` 讲概念叙事，本文件把里面每个公式拆开，给你**符号清单 + 小数据具体计算**。
>
> **预览**：VSCode `Cmd+Shift+V`，已装 `markdown-all-in-one` 自动渲 KaTeX。
>
> **约定**：本文 $\log$ 都指自然对数 $\ln$（以 $e \approx 2.718$ 为底）。所有数值四舍五入到 3 位小数。

## 符号速查表（先放这里，后面用到都来这查）

| 符号 | 念法 / 含义 |
|---|---|
| $\theta$ | "西塔" theta，模型的可学习参数（transformer 的几十亿个权重） |
| $P_\theta(\cdot)$ | 由参数 $\theta$ 决定的概率分布 |
| $\mathcal{D}$ | "花体 D"，训练数据集 |
| $\sim$ | "采样自"，$x \sim \mathcal{D}$ = "$x$ 从 $\mathcal{D}$ 里抽出来" |
| $\mathbb{E}_{x \sim \mathcal{D}}[f(x)]$ | "$f(x)$ 在 $\mathcal{D}$ 上的期望"，实务上 = 训练集每条算 $f$ 再求平均 |
| $\sum_{t=1}^N$ | 对 $t$ 从 1 到 $N$ 求和 |
| $\prod_{i=1}^N$ | 对 $i$ 从 1 到 $N$ **求积**（不是求和） |
| $y_t$ | 序列 $y$ 的第 $t$ 个 token |
| $y_{<t}$ | 序列 $y$ 的第 $t$ 个之前的所有 token，即 $y_1, y_2, \dots, y_{t-1}$ |
| $\mid$ | "条件于"，$P(A \mid B)$ = "在 $B$ 已知的前提下 $A$ 的概率" |
| $\arg\max_\theta f(\theta)$ | "让 $f(\theta)$ 取最大值的那个 $\theta$"（不是最大值本身） |
| $V$ | 词表（vocabulary），模型能输出的所有 token 集合 |
| $v \in V$ | 词表里的一个具体 token |
| $\delta_a$ | "delta on $a$"，one-hot 分布：$a$ 处为 1，其他为 0 |
| $H(p, q)$ | $p$ 和 $q$ 的 cross-entropy |
| $H(p)$ | $p$ 自身的 entropy（= $H(p, p)$） |
| $\mathrm{KL}(p \| q)$ | $p$ 关于 $q$ 的 KL 散度 |
| $\hat{A}$ | "A hat"，估计量（不是真值） |

---

## 公式 1：SFT 总损失

$$
L(\theta) \;=\; -\, \mathbb{E}_{(x, y) \sim \mathcal{D}} \left[\; \sum_{t=1}^{|y|} \log P_\theta\!\left(y_t \mid x,\, y_{<t}\right) \;\right]
$$

### 通俗一句话

> 对每一对 (输入, 目标答案)，让模型**一个 token 一个 token**地预测 target，错得越离谱（给正确 token 的概率越低）loss 越大。最后**对所有训练样本求平均**。

### 符号详解

| 符号 | 含义 |
|---|---|
| $L(\theta)$ | 总损失，是参数 $\theta$ 的函数。训练目标是**最小化** $L$。 |
| $-$ (开头的负号) | 让"概率高 → loss 小"。$\log$ 概率本身是负的（概率 ≤ 1），加负号变正。 |
| $\mathbb{E}_{(x,y) \sim \mathcal{D}}$ | 对训练集所有样本对求平均。 |
| $\sum_{t=1}^{|y|}$ | 对 target $y$ 的每个 token（从第 1 个到第 $|y|$ 个）求和。$\lvert y \rvert$ 是 $y$ 的长度。 |
| $P_\theta(y_t \mid x, y_{<t})$ | 模型在看到 prompt $x$ 和 target 的前 $t-1$ 个 token 后，**预测下一个 token 等于 $y_t$ 的概率**。 |
| $\log$ | 自然对数。 |

### 实例计算

设词表 $V = \{\texttt{the}, \texttt{cat}, \texttt{sat}, \texttt{.}\}$，训练集只有 **1 条**样本：

- $x = $ "complete:"
- $y = $ `the cat sat .` （4 个 token：$y_1=\texttt{the}$, $y_2=\texttt{cat}$, $y_3=\texttt{sat}$, $y_4=\texttt{.}$）

模型在每个位置给出的预测概率（假设的）：

| $t$ | 上下文（模型看到的） | 真实 $y_t$ | 模型给 $y_t$ 的概率 | $\log P_\theta(y_t \mid \dots)$ |
|---|---|---|---|---|
| 1 | `complete:` | the | 0.5 | $\log 0.5 = -0.693$ |
| 2 | `complete: the` | cat | 0.6 | $\log 0.6 = -0.511$ |
| 3 | `complete: the cat` | sat | 0.4 | $\log 0.4 = -0.916$ |
| 4 | `complete: the cat sat` | . | 0.9 | $\log 0.9 = -0.105$ |

加总：

$$
\sum_{t=1}^{4} \log P_\theta(y_t \mid \dots) \;=\; -0.693 - 0.511 - 0.916 - 0.105 \;=\; -2.225
$$

加负号取期望（只有 1 条样本，期望就等于自己）：

$$
L \;=\; -(-2.225) \;=\; 2.225
$$

### 这告诉你什么

- 这条样本的总 loss 是 **2.225 nat**。每 token 平均 surprise $= 2.225 / 4 = 0.556$ nat。
- 如果模型完美预测（4 个概率全是 1）：$L = -\log 1 \times 4 = 0$。
- 如果模型给某个真实 token 的概率是 0：$\log 0 = -\infty$，loss 会爆炸。**这是为什么训练里偶尔出 NaN —— 模型把某个真实 token 的概率推到了机器精度的下界**。

---

## 公式 2：Cross-entropy（一般形式）

$$
H(p, q) \;=\; -\sum_{v} p(v) \log q(v)
$$

### 通俗一句话

> 真实分布是 $p$，但你用 $q$ 来近似。$H(p, q)$ 就是**用 $q$ 描述 $p$ 时，平均每个样本要"多花掉"的信息代价**（单位 nat）。

### 符号详解

| 符号 | 含义 |
|---|---|
| $p, q$ | 两个概率分布，定义在同一个事件空间（同一个词表 $V$） |
| $v$ | 事件空间里的一个具体事件（词表里的一个具体 token） |
| $p(v)$ | 真实分布给事件 $v$ 的概率 |
| $q(v)$ | 你的预测分布给事件 $v$ 的概率 |
| $\sum_v$ | 对事件空间里所有事件求和 |
| 前面的 $-$ | 让结果非负（$\log$ 概率本身是负的） |

### 实例计算

词表 $V = \{A, B, C\}$。真实分布：

$$p = (0.5,\, 0.3,\, 0.2)$$

考察两个预测分布 $q_1, q_2$：

| 预测 | A | B | C |
|---|---|---|---|
| $q_1$（完美） | 0.5 | 0.3 | 0.2 |
| $q_2$（很糟） | 0.1 | 0.1 | 0.8 |

**算 $H(p, q_1)$**：

$$
H(p, q_1) = -[0.5 \log 0.5 + 0.3 \log 0.3 + 0.2 \log 0.2]
$$

| 项 | $p(v)$ | $\log q_1(v)$ | $p(v) \log q_1(v)$ |
|---|---|---|---|
| A | 0.5 | $\log 0.5 = -0.693$ | $-0.347$ |
| B | 0.3 | $\log 0.3 = -1.204$ | $-0.361$ |
| C | 0.2 | $\log 0.2 = -1.609$ | $-0.322$ |
| **和** | | | $-1.030$ |

$$H(p, q_1) = -(-1.030) = 1.030 \text{ nat}$$

**算 $H(p, q_2)$**：

| 项 | $p(v)$ | $\log q_2(v)$ | $p(v) \log q_2(v)$ |
|---|---|---|---|
| A | 0.5 | $\log 0.1 = -2.303$ | $-1.151$ |
| B | 0.3 | $\log 0.1 = -2.303$ | $-0.691$ |
| C | 0.2 | $\log 0.8 = -0.223$ | $-0.045$ |
| **和** | | | $-1.887$ |

$$H(p, q_2) = -(-1.887) = 1.887 \text{ nat}$$

### 这告诉你什么

- $H(p, q_2) > H(p, q_1)$：$q_2$ 偏离 $p$ 更远，cross-entropy 更大。
- $H(p, q_1) = 1.030$ 是**下界**：当 $q = p$ 时，cross-entropy 取到最小值，这个最小值就是 $p$ 自身的 entropy $H(p)$。
- 训练神经网络（包括 SFT）的本质就是**调整 $\theta$ 让模型预测 $q = P_\theta$ 越来越接近真实 $p$**，等价于把 cross-entropy 往 $H(p)$ 这个下界压。

---

## 公式 3：Cross-entropy + one-hot 真实分布（SFT 的特例）

$$
H(\delta_{y_t},\, P_\theta) \;=\; -\sum_{v \in V} \delta_{y_t}(v) \log P_\theta(v \mid x, y_{<t}) \;=\; -\log P_\theta(y_t \mid x, y_{<t})
$$

### 通俗一句话

> 当真实分布是 one-hot（即"标准答案只有一个"），cross-entropy 的求和里只有"正确答案那一项"不为零，化简就是 **$-\log$(模型给正确答案的概率)**。

### 符号详解

| 符号 | 含义 |
|---|---|
| $\delta_{y_t}$ | one-hot 分布，$y_t$ 处为 1，其他 token 处为 0 |
| $\delta_{y_t}(v)$ | 在词 $v$ 上的取值（要么 1 要么 0） |
| 其他 | 同公式 2 |

### 实例计算

接公式 2 的设定。词表 $V = \{A, B, C\}$。真实下一个 token 是 $A$，所以：

$$\delta_A = (1, 0, 0)$$

模型预测 $P_\theta = (0.7, 0.2, 0.1)$。

按 cross-entropy 完整公式：

| 项 | $\delta_A(v)$ | $\log P_\theta(v)$ | $\delta_A(v) \log P_\theta(v)$ |
|---|---|---|---|
| A | **1** | $\log 0.7 = -0.357$ | $-0.357$ |
| B | 0 | $\log 0.2 = -1.609$ | **$0$**（因为乘了 0） |
| C | 0 | $\log 0.1 = -2.303$ | **$0$**（因为乘了 0） |
| **和** | | | $-0.357$ |

$$H(\delta_A, P_\theta) = -(-0.357) = 0.357 \text{ nat}$$

**关键观察**：B 和 C 那两项**完全不贡献**，因为 $\delta_A$ 在那两个位置是 0。所以整个 cross-entropy **只取决于"模型给正确答案 A 的概率"**：

$$H(\delta_A, P_\theta) = -\log P_\theta(A) = -\log 0.7$$

**对比不同预测**：

| 模型预测 $P_\theta$ | 模型给 A 的概率 | $-\log P_\theta(A)$ |
|---|---|---|
| $(0.95, 0.03, 0.02)$ | 0.95 | $0.051$（自信对，loss 小）|
| $(0.7, 0.2, 0.1)$ | 0.7 | $0.357$（基线）|
| $(0.5, 0.3, 0.2)$ | 0.5 | $0.693$ |
| $(0.1, 0.5, 0.4)$ | 0.1 | $2.303$（认为是 B，loss 大）|
| $(0.001, 0.999, 0)$ | 0.001 | $6.908$（强烈相信不是 A，loss 极大）|

### 这告诉你什么

- **SFT loss 里每一项 $-\log P_\theta(y_t \mid \dots)$ 就是这个东西**。公式 1 里你看到的 4 个 $\log$ 值，每一个就是一个 cross-entropy（真实是 one-hot 的特例）。
- 模型对正确答案越自信 → loss 越小。给错答案高概率 → loss 急剧上升。
- **真实下一个 token 已知** → one-hot 分布 → cross-entropy 化简成"$-\log$ 正确概率" → 计算上简化为一次查表操作（不需要遍历整个词表）。这就是为什么 GPU 训 LLM 这么快。

---

## 公式 4：MLE（最大似然估计）

$$
\theta_{\text{MLE}} \;=\; \arg\max_\theta \prod_{i=1}^n P_\theta(x_i)
\quad\Longleftrightarrow\quad
\arg\max_\theta \sum_{i=1}^n \log P_\theta(x_i)
$$

### 通俗一句话

> 我观察到了 $n$ 个数据样本，模型有参数 $\theta$。MLE 说：**选那个让"我观察到的所有数据"概率最大的 $\theta$**。

### 符号详解

| 符号 | 含义 |
|---|---|
| $\theta_{\text{MLE}}$ | MLE 估计出的参数值 |
| $\arg\max_\theta$ | "让后面的表达式取最大值的那个 $\theta$"（**不是最大值本身**）|
| $\prod_{i=1}^n$ | 对 $n$ 个样本**求积**（每个样本的概率连乘）|
| $\sum_{i=1}^n$ | **求和**（取 log 之后连乘变连加，数值上更稳定）|
| $P_\theta(x_i)$ | 模型在参数 $\theta$ 下给第 $i$ 个样本的概率 |

**为什么要取 log**：直接连乘 $n$ 个概率（每个都 ≤ 1）→ 数值急速趋近 0 → 计算机精度爆掉。取 log 把连乘变连加，数值稳定。**因为 $\log$ 单调递增，$\arg\max$ 的结果不变**。

### 实例计算（硬币）

扔硬币 10 次，结果：H, H, H, T, H, T, H, H, T, H（**7 正 3 反**）。

模型：伯努利分布 $P_\theta(\text{H}) = \theta$，$P_\theta(\text{T}) = 1 - \theta$。$\theta$ 就是我们要估的"正面概率"。

写出似然函数（10 次独立扔的概率连乘）：

$$L(\theta) = \theta \cdot \theta \cdot \theta \cdot (1-\theta) \cdot \theta \cdot (1-\theta) \cdot \theta \cdot \theta \cdot (1-\theta) \cdot \theta = \theta^7 (1-\theta)^3$$

取 log：

$$\log L(\theta) = 7 \log \theta + 3 \log(1-\theta)$$

为找 argmax，对 $\theta$ 求导，设为 0：

$$\frac{d}{d\theta} [7 \log \theta + 3 \log(1-\theta)] = \frac{7}{\theta} - \frac{3}{1-\theta} = 0$$

整理：

$$7(1-\theta) = 3\theta \;\Rightarrow\; 7 = 10\theta \;\Rightarrow\; \boxed{\theta_{\text{MLE}} = 0.7}$$

**数值验证**（试几个 $\theta$，看哪个 $\log L$ 最大）：

| $\theta$ | $\log L(\theta) = 7 \log \theta + 3 \log(1-\theta)$ |
|---|---|
| 0.5 | $7 \log 0.5 + 3 \log 0.5 = 10 \times (-0.693) = -6.93$ |
| 0.6 | $7 \log 0.6 + 3 \log 0.4 = -3.58 + -2.75 = -6.33$ |
| **0.7** | $7 \log 0.7 + 3 \log 0.3 = -2.50 + -3.61 = \mathbf{-6.11}$ ← 最大 |
| 0.8 | $7 \log 0.8 + 3 \log 0.2 = -1.56 + -4.83 = -6.39$ |
| 0.9 | $7 \log 0.9 + 3 \log 0.1 = -0.74 + -6.91 = -7.65$ |

最大值在 $\theta = 0.7$ 处 ✓。

### 这告诉你什么

- **MLE 在很多简单情形下就是"用频率估概率"**。7/10 = 0.7，直觉和数学一致。
- 在 SFT 里，"数据"是 $(x, y)$ 对，"模型"是 transformer，MLE 目标是最大化 $\sum_i \log P_\theta(y_i \mid x_i)$ —— **就是公式 1 的负数**（最小化 loss = 最大化 log-likelihood）。
- ML 训练 80% 的 loss 形式都是 MLE 变种（pretraining / SFT / VAE 主项 / normalizing flow 等）。

---

## 公式 5：KL 散度

$$
\mathrm{KL}(p \,\|\, q) \;=\; \sum_v p(v) \log \frac{p(v)}{q(v)}
$$

### 通俗一句话

> KL 散度衡量"用 $q$ 描述 $p$" 比 "用 $p$ 自己描述 $p$"**多花了多少信息代价**。$p = q$ 时为 0；偏差越大、KL 越大。

### 符号详解

| 符号 | 含义 |
|---|---|
| $\mathrm{KL}(p \,\|\, q)$ | "$p$ 关于 $q$ 的 KL 散度"。注意 $p$ 在前、$q$ 在后**不能交换**。 |
| $\log \frac{p(v)}{q(v)}$ | 比值的 log。$p > q$ 时为正，$p < q$ 时为负。|
| 没有前面的负号 | 因为 $p(v) \log(p/q)$ 本身在求和后总是非负 |

### 实例计算

接公式 2：$p = (0.5, 0.3, 0.2)$，$q = (0.1, 0.1, 0.8)$。

**直接按 KL 公式算**：

| 项 | $p(v)$ | $q(v)$ | $\frac{p(v)}{q(v)}$ | $\log \frac{p}{q}$ | $p(v) \log \frac{p}{q}$ |
|---|---|---|---|---|---|
| A | 0.5 | 0.1 | 5.0 | $\log 5 = 1.609$ | $0.805$ |
| B | 0.3 | 0.1 | 3.0 | $\log 3 = 1.099$ | $0.330$ |
| C | 0.2 | 0.8 | 0.25 | $\log 0.25 = -1.386$ | $-0.277$ |
| **和** | | | | | $\mathbf{0.857}$ |

$$\mathrm{KL}(p \,\|\, q) = 0.857 \text{ nat}$$

**验证非对称性 —— 算 $\mathrm{KL}(q \,\|\, p)$**（注意 $p$ 和 $q$ 角色互换）：

| 项 | $q(v)$ | $p(v)$ | $\frac{q(v)}{p(v)}$ | $\log \frac{q}{p}$ | $q(v) \log \frac{q}{p}$ |
|---|---|---|---|---|---|
| A | 0.1 | 0.5 | 0.2 | $-1.609$ | $-0.161$ |
| B | 0.1 | 0.3 | 0.333 | $-1.099$ | $-0.110$ |
| C | 0.8 | 0.2 | 4.0 | $1.386$ | $1.109$ |
| **和** | | | | | $\mathbf{0.838}$ |

$$\mathrm{KL}(q \,\|\, p) = 0.838 \neq 0.857 = \mathrm{KL}(p \,\|\, q) \;\;\checkmark \text{ 非对称}$$

### 这告诉你什么

- KL 值是个**非负数**。$p = q$ 时取 0；偏差越大、KL 越大。
- $\mathrm{KL}(p \,\|\, q) \neq \mathrm{KL}(q \,\|\, p)$，要看清楚谁在前谁在后。
- 在 RL 里，"防止模型偏离参考模型太远"的 KL 项写的是 $\mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$ —— **新策略相对参考策略**的 KL，不能写反。

---

## 公式 6：KL 散度 = cross-entropy − entropy（最关键的桥梁等式）

$$
\mathrm{KL}(p \,\|\, q) \;=\; H(p, q) - H(p)
$$

其中 $H(p) = -\sum_v p(v) \log p(v)$ 是 $p$ 自身的 entropy（也等于 $H(p, p)$）。

### 通俗一句话

> KL 散度 = 用 $q$ 描述 $p$ 的代价 **减去** 用 $p$ 自己描述 $p$ 的最小代价。这个等式让"最小化 KL"和"最小化 cross-entropy"变成**同一件事**（差一个不影响优化的常数）。

### 符号详解

| 符号 | 含义 |
|---|---|
| $H(p, q)$ | cross-entropy（公式 2）|
| $H(p)$ | $p$ 自己的 entropy，$= -\sum_v p(v) \log p(v) = H(p, p)$ |
| $-$ | 减号 |

### 实例计算

接公式 2 和 5：$p = (0.5, 0.3, 0.2)$，$q = (0.1, 0.1, 0.8)$。

**先算 $H(p)$**（即 $H(p, p)$，注意 $q = p$ 是特例）：

| 项 | $p(v)$ | $\log p(v)$ | $p(v) \log p(v)$ |
|---|---|---|---|
| A | 0.5 | $-0.693$ | $-0.347$ |
| B | 0.3 | $-1.204$ | $-0.361$ |
| C | 0.2 | $-1.609$ | $-0.322$ |
| **和** | | | $-1.030$ |

$$H(p) = -(-1.030) = 1.030 \text{ nat}$$

注意：这正好等于公式 2 里算的 $H(p, q_1)$，因为 $q_1 = p$。

**算 $H(p, q)$**（公式 2 已经算过）：

$$H(p, q) = 1.887$$

**两者相减**：

$$\mathrm{KL}(p \,\|\, q) = H(p, q) - H(p) = 1.887 - 1.030 = 0.857$$

**对照公式 5 直接算的结果**：$\mathrm{KL}(p \,\|\, q) = 0.857$ ✓ **完全一致**。

### 这告诉你什么

这个等式连接了 §2.1 里"MLE 视角"和"behavior cloning 视角"：

- **MLE 视角**说：最小化 $-\sum \log P_\theta(y \mid x)$（等价于最大化对数似然）
- **Behavior cloning 视角**说：最小化 $\mathrm{KL}(\pi_T \,\|\, \pi_\theta) = H(\pi_T, \pi_\theta) - H(\pi_T)$

但 $H(\pi_T)$ **不依赖 $\theta$**（教师分布是固定的），所以对 $\theta$ 求最优时它是常数，可以丢掉：

$$
\arg\min_\theta \mathrm{KL}(\pi_T \,\|\, \pi_\theta) \;=\; \arg\min_\theta H(\pi_T, \pi_\theta) \;=\; \arg\max_\theta \log P_\theta(y \mid x)
$$

**最小化 KL = 最小化 cross-entropy = 最大化 log-likelihood**。三者在优化上等价。

这就是为什么不管你用哪一种语言看 SFT，得到的训练目标都一样。

---

## 公式 7：GRPO advantage（RL 阶段会用，先放着熟悉一下）

$$
\hat{A}_i \;=\; \frac{R_i - \mathrm{mean}(\{R_1, \dots, R_N\})}{\mathrm{std}(\{R_1, \dots, R_N\})}
$$

### 通俗一句话

> 同一个 prompt 我采 $N$ 个 completion，每个有自己的 reward。**每个 completion 的"优势"= 它的 reward 比组平均高多少，按组标准差归一化**。

### 符号详解

| 符号 | 含义 |
|---|---|
| $\hat{A}_i$ | 第 $i$ 个 completion 的归一化优势（advantage）|
| 帽子 $\hat{}$ | "估计量"，提醒这不是真值 |
| $R_i$ | 第 $i$ 个 completion 的 reward |
| $\mathrm{mean}(\{R_i\}) = \frac{1}{N}\sum_i R_i$ | 组内 reward 平均 |
| $\mathrm{std}(\{R_i\}) = \sqrt{\frac{1}{N}\sum_i (R_i - \mathrm{mean})^2}$ | 组内 reward 标准差 |
| $N$ | group size（Circuit-Think 用 8）|

### 实例计算

设 $N = 4$，对同一个 prompt 采了 4 个 completion，reward 如下：

$$R_1 = 0.9,\; R_2 = 0.6,\; R_3 = 0.3,\; R_4 = 0.2$$

**算 mean**：

$$\mathrm{mean} = \frac{0.9 + 0.6 + 0.3 + 0.2}{4} = \frac{2.0}{4} = 0.5$$

**算 std**（先算每个偏差的平方）：

| $i$ | $R_i$ | $R_i - \mathrm{mean}$ | $(R_i - \mathrm{mean})^2$ |
|---|---|---|---|
| 1 | 0.9 | $+0.4$ | $0.16$ |
| 2 | 0.6 | $+0.1$ | $0.01$ |
| 3 | 0.3 | $-0.2$ | $0.04$ |
| 4 | 0.2 | $-0.3$ | $0.09$ |
| **和** | | | $0.30$ |

$$\mathrm{std} = \sqrt{\frac{0.30}{4}} = \sqrt{0.075} \approx 0.274$$

**算每个 $\hat{A}_i$**：

| $i$ | $R_i$ | $R_i - 0.5$ | $\hat{A}_i = (R_i - 0.5) / 0.274$ |
|---|---|---|---|
| 1 | 0.9 | $+0.4$ | $\mathbf{+1.46}$ ← 强化 |
| 2 | 0.6 | $+0.1$ | $\mathbf{+0.36}$ ← 小幅强化 |
| 3 | 0.3 | $-0.2$ | $\mathbf{-0.73}$ ← 抑制 |
| 4 | 0.2 | $-0.3$ | $\mathbf{-1.09}$ ← 强抑制 |

### 这告诉你什么

- **$\hat{A}_i > 0$（高于平均）→ 模型会被推动生成更多这种 completion**。
- **$\hat{A}_i < 0$（低于平均）→ 模型会被推动远离这种 completion**。
- 数学上，GRPO loss 大致是 $\sum_i \hat{A}_i \cdot \log \pi_\theta(o_i \mid x)$（再加 PPO clip 和 KL 项），等价于"按 advantage 加权的对数似然"。

**两种退化情况（解释为什么 SFT 必须在 RL 之前）**：

1. **全部失败**：$R_1 = R_2 = R_3 = R_4 = 0$
   $\mathrm{mean} = 0$, $\mathrm{std} = 0$
   $\hat{A}_i = 0 / 0 = \mathrm{NaN}$
   → **数学未定义，训不动**。

2. **几乎相等**：$R_1, R_2, R_3, R_4 = 0.30, 0.30, 0.31, 0.29$
   $\mathrm{std} \approx 0.007$（极小）
   $\hat{A}_i$ 被放大成 $\pm 10$ 量级的怪异大值
   → 梯度极不稳定，**训得起来但训得很烂**。

→ 这就是 §1 最后那条 take-away "**SFT 是 RL 的入场券**"的数学根据：SFT 把基础成功率提到中等水平（不全 0、不全 1、有方差），advantage 才能算出有意义的数值。

---

## 最后：一个 cheat sheet 把 6+1 个公式串起来

```
公式 1 (SFT loss)        ← 实际训练时算的目标
    └── 每一项 -log P(y_t | ...)
            ↓
公式 3 (one-hot CE)      ← 因为真实是 one-hot，CE 化简成 -log P(正确)
            ↓
公式 2 (一般 CE)         ← 这就是 CE 的特例
            ↓
公式 6 (CE = H + KL)     ← 桥梁
            ↓
公式 5 (KL 散度)         ← 用"分布距离"的语言重写
            ↓
公式 4 (MLE)             ← 三者数学等价的另一种表述

公式 7 (GRPO advantage)  ← 后面 RL 阶段用，这里先了解
```

**核心一句**：SFT loss 的每一项是 cross-entropy，整体是负 log-likelihood，最大化它就是 MLE，等价于最小化 student-teacher KL —— **同一件事的四种说法**。

## 下一步

读完这一章 + 这份公式陪读，你应该能：

- 看到 $L(\theta) = -\mathbb{E}[\sum \log P_\theta]$ → 知道每个符号、知道怎么算
- 看到 cross-entropy / KL / MLE → 不再是符号 soup，能解释每一个
- 看到 GRPO advantage 公式 → 提前知道为什么 SFT 是 RL 的前提

接下来如果你想继续：
- **进 §2 第 2 章** [`02_data_and_format.md`](./02_data_and_format.md) → 学数据格式、chat template、prompt mask（公式不多，但工程细节决定 SFT 成败）
- **回 [`01_foundations.md`](./01_foundations.md)** → 现在看 §2.1 / §2.2 / §2.3 应该全通了

如果某个公式还想要更深的展开（比如想看导数推导、或想知道为什么某个等式对），告诉我具体哪一行，我再加。
