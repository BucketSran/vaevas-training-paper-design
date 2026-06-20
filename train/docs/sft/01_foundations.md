# 01 — Foundations: What SFT Actually Does

> **预览说明**：本文公式用 LaTeX。请用 VSCode `Cmd+Shift+V` 打开 preview（已装 `markdown-all-in-one` 扩展，自动渲染 KaTeX）。
>
> **公式看不懂？** → 配套文档 [`01a_formulas_walkthrough.md`](./01a_formulas_walkthrough.md) 把本章每个公式拆到**符号级别**，并用**小数据具体算到底**。

> 读完这一章，你应该能回答：
> 1. SFT 在最大化什么？
> 2. 为什么"学到了"，本质是什么？
> 3. 数据为什么比超参重要 10 倍？
> 4. 为什么 SFT 之后还需要 RL？
> 5. 为什么"loss 越低越好"是错的？

## 1. 目标函数：一行公式

SFT 优化的目标，**就这一行**：

$$
L(\theta) \;=\; -\, \mathbb{E}_{(x, y) \sim \mathcal{D}} \left[\; \sum_{t=1}^{|y|} \log P_\theta\!\left(y_t \mid x,\, y_{<t}\right) \;\right]
$$

拆开来：

- $(x, y)$ 是 prompt + target completion 对
- $\mathcal{D}$ 是训练集分布
- $y_t$ 是 target 的第 $t$ 个 token
- $y_{<t}$ 是前面所有 token（causal mask）
- $P_\theta(y_t \mid x,\, y_{<t})$ 是模型在给定上下文下对 token $y_t$ 的预测概率
- 外层求和：对 target 的**每一个 token**累积 negative log-likelihood
- 外层期望：对训练数据**取平均**

**关键观察**：

1. **prompt $x$ 不在求和里**。我们只对 target 的 token 算 loss。**prompt 的 token 是条件，不是目标**。这就是为什么 "prompt masking" 这件事重要（见第 2 章）。
2. 每一项都是 cross-entropy，所以这就是 next-token prediction 的 CE loss，**和 pretraining 同一个 loss**。
3. 唯一区别是 $\mathcal{D}$ —— pretraining 的 $\mathcal{D}$ 是互联网，SFT 的 $\mathcal{D}$ 是你精心挑选的 $(x, y)$ 对。

**所以 SFT 的本质是：换一个数据分布继续做 next-token prediction。** 这件事很重要，重要到值得停下来再读一遍。

### 1.1 为什么"每一项都是 cross-entropy"，又为什么"和 pretraining 同一个 loss"

上面"关键观察"第 2 条压得很紧，展开讲三件事。

**(a) 什么是 cross-entropy**

Cross-entropy 衡量两个概率分布的"距离"。对离散分布 $p, q$：

$$
H(p, q) \;=\; -\sum_{v} p(v) \log q(v)
$$

直觉：如果真实分布是 $p$，但你用 $q$ 去描述它，那 $H(p, q)$ 是你**平均每个样本要多付出的信息代价**（单位 nat 或 bit）。$q$ 越接近 $p$，$H(p,q)$ 越小；$q = p$ 时 $H$ 取到下界 $H(p, p) = $ entropy of $p$。

**(b) 为什么 $-\log P_\theta(y_t \mid x, y_{<t})$ 就是 cross-entropy**

在 SFT 训练里，"真实分布"是一个 **one-hot 分布**：第 $t$ 步的真实下一个 token 已知就是 $y_t$，所以词表 $V$ 上的真实分布是

$$
\delta_{y_t}(v) \;=\; \begin{cases} 1, & v = y_t \\ 0, & v \neq y_t \end{cases}
$$

模型预测的分布是 $P_\theta(\cdot \mid x, y_{<t})$ —— 也就是 transformer 最后一层 softmax 输出的整个词表上的概率分布。两者的 cross-entropy：

$$
H(\delta_{y_t},\, P_\theta) \;=\; -\sum_{v \in V} \delta_{y_t}(v) \log P_\theta(v \mid x, y_{<t})
$$

因为 $\delta_{y_t}$ 只在 $v = y_t$ 这一点为 1，其他全为 0，**求和里只有一项幸存**：

$$
H(\delta_{y_t},\, P_\theta) \;=\; -\log P_\theta(y_t \mid x, y_{<t})
$$

所以 SFT loss 公式里 **每一个 $-\log P_\theta(y_t \mid x, y_{<t})$ 项**，本质都是"one-hot ground-truth 分布"和"模型预测分布"之间的 cross-entropy。整段 target 的 loss 就是 $|y|$ 个 cross-entropy 项之和。

**(c) 为什么和 pretraining 是同一个 loss**

GPT 风格预训练用的是 causal LM 损失：

$$
L_{\text{pretrain}}(\theta) \;=\; -\, \mathbb{E}_{w \sim \text{corpus}} \left[\; \sum_{t=1}^{|w|} \log P_\theta(w_t \mid w_{<t}) \;\right]
$$

对比 SFT loss：

$$
L_{\text{SFT}}(\theta) \;=\; -\, \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[\; \sum_{t=1}^{|y|} \log P_\theta(y_t \mid x, y_{<t}) \;\right]
$$

**数学结构完全一样**。区别只有两个：

1. **求和范围**：pretraining 对整段文本 $w$ 的所有 token 都算 loss；SFT 只对 target $y$ 的 token 算（prompt $x$ 不算 —— 见 第 2 章的 prompt masking）。
2. **数据分布**：pretraining 是大规模 corpus（互联网）；SFT 是精挑细选的 $(x, y)$ 配对。

如果把 SFT 的 $(x, y)$ 拼成一段连续文本，再加一个 mask 告诉 loss "前 $|x|$ 个 token 不计入"，就**完全等价于"用过滤后的小数据继续预训练"**。SFT 不是一种"新算法"，它是 pretraining 在不同数据上的延续。

**这件事的三个实用后果**

1. **不需要新架构 / 新代码**。SFT 复用预训练时的所有前向、反向、采样实现，只在 loss 计算时加一个 prompt-mask（实务上就是把 prompt 部分的 `labels` 设成 `-100`，HuggingFace 的 cross-entropy 自动跳过）。
2. **Catastrophic forgetting 有了机理解释**。SFT 和预训练在**同一条 loss landscape** 上跑，optimizer 把参数从"预训练最优"推向"SFT 数据最优"。**朝新最优移动的过程中必然偏离老最优** —— 通用能力的丢失不是 bug，是数学结果。
3. **SFT 的能力上限 ≈ 同模型继续预训练的能力上限**。靠 SFT 你只能"重新分配"模型在预训练时已经获得的概率质量，**无法突破预训练时设定的能力天花板**。要突破，必须换 loss（RLHF / DPO / GRPO）或者扩规模（更大模型 / 更多预训练数据）。

→ 这一段是后面所有章节的根。读完应该建立一个稳定直觉：**SFT 是"换数据继续做同一件事"，不是"做一件新事"**。

## 2. 两种解读，决定不同的直觉

### 2.1 先决概念：MLE 与 KL 散度

后面 §2.2 和 §2.3 是用两种不同的统计语言看同一个 SFT loss。在那之前先把这两套语言本身讲清楚 —— 之后看 §2.2、§2.3 就只是"换皮"。

#### MLE（最大似然估计，Maximum Likelihood Estimation）

**问题设定**：你手上有一批数据样本 $\{x_1, x_2, \dots, x_n\}$，假设它们独立同分布（i.i.d.）来自某个未知分布。你建立了一个由参数 $\theta$ 控制的概率模型 $P_\theta$（比如一个 transformer）。问：**$\theta$ 取什么值最合理？**

**MLE 的回答**：取那个让"观察到当前数据"的概率最大的 $\theta$：

$$
\theta_{\text{MLE}} \;=\; \arg\max_\theta \prod_{i=1}^n P_\theta(x_i)
$$

数值上为避免连乘下溢，通常取 log：

$$
\theta_{\text{MLE}} \;=\; \arg\max_\theta \sum_{i=1}^n \log P_\theta(x_i)
$$

**直觉（硬币例子）**：扔硬币 10 次出现 7 次正面，正面概率 $p$ 该估计多少？MLE 说 $p = 0.7$ —— 因为这个 $p$ 让"恰好观察到 7 正 3 反"的概率最大。一句话总结：**"哪个参数让我观察到的事情最不奇怪，就选哪个参数"**。

**在 SFT 里**：把 $(x, y)$ 对当成数据样本，模型 $P_\theta(y \mid x)$ 是参数化条件分布。最大化整个训练集的 log-likelihood：

$$
\theta^* \;=\; \arg\max_\theta \; \mathbb{E}_{(x,y) \sim \mathcal{D}} \log P_\theta(y \mid x)
$$

这就是 §2.2 要讲的"MLE 视角"。

**为什么 ML 训练经常长这样**：MLE 是统计学最经典的参数估计原则。几乎所有"最大化对数似然"或者"最小化负对数似然"的训练目标 —— 包括 pretraining、SFT、VAE 的主项、normalizing flows 等 —— 都是 MLE 的变种。理解了 MLE，你就理解了 ML 中 80% 的 loss 公式背后的统一逻辑。

#### KL 散度（Kullback–Leibler divergence）

**问题设定**：有两个概率分布 $p, q$，定义在同一个事件空间。怎么**量化它们的"差距"**？

**KL 散度的定义**（离散情形）：

$$
\mathrm{KL}(p \,\|\, q) \;=\; \sum_v p(v) \log \frac{p(v)}{q(v)}
$$

（连续情形把求和换成积分。）

**直觉**：如果真实分布是 $p$，但你用 $q$ 去描述它（编码、预测、采样），那 $\mathrm{KL}(p \,\|\, q)$ 是你**平均每个样本多花掉的信息代价**（单位 nat）。$p = q$ 时 $\mathrm{KL} = 0$；差距越大、$\mathrm{KL}$ 越大。

**三个关键性质**：

1. **非负**：$\mathrm{KL}(p \,\|\, q) \geq 0$，等号当且仅当 $p = q$（几乎处处）。
2. **非对称**：$\mathrm{KL}(p \,\|\, q) \neq \mathrm{KL}(q \,\|\, p)$。这不是 bug，是 feature —— "真实是 $p$ 你猜成 $q$" 和 "真实是 $q$ 你猜成 $p$" 是两件不同的事，分别对应 **mode covering** 与 **mode seeking** 两种 fit 行为。
3. **不是距离**：不满足三角不等式，所以叫"散度"不叫"距离"。

**和 cross-entropy 的核心关系**（最重要的一个等式）：

$$
\mathrm{KL}(p \,\|\, q) \;=\; H(p, q) - H(p)
$$

其中 $H(p, q)$ 是 cross-entropy，$H(p)$ 是 $p$ 自身的 entropy。所以 **"KL 散度 = cross-entropy 减一个不依赖 $\theta$ 的常数"**。

→ 这正是 §2.3 里"最小化 student-teacher KL 等价于 MLE 目标（常数项忽略）"的来源。展开看：

$$
\arg\min_\theta \mathrm{KL}(\pi_T \,\|\, \pi_\theta) \;=\; \arg\min_\theta \big[\, H(\pi_T, \pi_\theta) - H(\pi_T) \,\big] \;=\; \arg\min_\theta H(\pi_T, \pi_\theta)
$$

因为 $H(\pi_T)$ 跟 $\theta$ 无关，求最优时常数项可忽略。所以 **最小化 KL = 最小化 cross-entropy = 最大化对数似然**，三者在优化层面是等价的。

**为什么 RL 论文里 KL 项到处都是**：KL 散度是约束"模型新策略 $\pi_\theta$ 不能离参考策略 $\pi_{\text{ref}}$ 太远"的标准工具。GRPO / PPO / DPO 都在 loss 里显式加一个 $\beta \cdot \mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$ 项，作用都是"拉一根绳子防止 policy 在 RL 优化时跑飞"。在 SFT 这里 KL 是**分析目标的语言**；在 RL 那里 KL 是 **loss 里的显式正则项** —— 同一个数学对象，扮演不同角色。

#### 三者关系总览

| 概念 | 公式 | 在 SFT 里扮演什么 |
|---|---|---|
| Cross-entropy $H(p, q)$ | $-\sum_v p(v) \log q(v)$ | 训练 loss 的**每一项**就是它 |
| MLE 目标 | $\arg\max_\theta \mathbb{E} \log P_\theta$ | SFT 全局**优化目标** |
| KL 散度 $\mathrm{KL}(p \,\|\, q)$ | $\sum_v p \log (p/q) = H(p,q) - H(p)$ | 用"分布逼近"的语言重写 MLE，**和 MLE 等价（差常数）** |

理解这个三角等价之后，§2.2 和 §2.3 就只是"把同一件事换两种语言讲"，不会再有"为什么 KL 突然冒出来"的困惑。

### 2.2 MLE 视角：最大似然估计

把上式写成：

$$
\theta^{*} \;=\; \arg\max_{\theta} \; \mathbb{E}_{(x,y) \sim \mathcal{D}} \; \log P_\theta(y \mid x)
$$

$P_\theta(y \mid x)$ 是模型对整个序列 $y$ 的概率（用 chain rule 拆成 token-level 就是上式）。

**这是在做什么？** 在调整 $\theta$，让模型对训练数据中**实际出现过的 $(x, y)$ 配对**赋予更高概率。

直觉：**"教科书例题做多了考试就会"**。

### 2.3 Behavior Cloning 视角：模仿学习

把数据看成一个 "teacher policy" $\pi_T$ 产生的轨迹：
- teacher 看到 $x$，决定输出 $y$
- 我们想让 student $\pi_\theta$ 模仿 teacher 的行为

$$
L \;=\; \mathbb{E}_{x \sim \mathcal{X}} \; \mathrm{KL}\!\big(\, \pi_T(\cdot \mid x) \,\|\, \pi_\theta(\cdot \mid x) \,\big)
$$

最小化 student 和 teacher 的 KL 散度 → 等价于上面的 MLE 目标（常数项忽略）。

**这是在做什么？** 让模型的**条件分布**逼近 teacher 在数据里展示的条件分布。

直觉：**"看徒弟学着师父怎么做"**。

### 2.4 两种视角的差别决定你怎么思考问题

| 思考问题 | MLE 视角 | Behavior cloning 视角 |
|---|---|---|
| 数据少时怎么办？ | "MLE 容易过拟合" | "示范不够，徒弟学不全" |
| 数据有噪声怎么办？ | "估计有偏" | "师父示范错了，徒弟也学错" |
| 推理时输出和训练差很多 | "测试分布漂移" | "徒弟没见过的场景，自由发挥" |
| 一个 prompt 多个对的答案 | "多模态分布，MLE 学平均" | "几个师父示范不一样，徒弟困惑" |

**强烈建议平时主要用 behavior cloning 视角**，因为它更直观地解释**为什么数据质量、数据分布、数据多样性是决定性的**。

## 3. 教师分布：为什么数据是 SFT 的灵魂

SFT 学到的能力**完全由训练数据的条件分布决定**。改超参不会让能力凭空多出来。

具体来说，SFT 后的模型 $\pi_\theta(y \mid x)$ 会收敛到：

$$
\begin{aligned}
\pi_\theta(y \mid x) &\;\approx\; \pi_{\text{data}}(y \mid x) && \text{当 } x \in \text{training prompts} \\
\pi_\theta(y \mid x) &\;\approx\; \pi_{\text{base}}(y \mid x) && \text{当 } x \text{ 远离 training prompts}
\end{aligned}
$$

中间地带是插值。

**这有四个推论**：

**推论 1：数据里没有的能力，SFT 教不出来。**
你的训练集没有 "comparator 写法"，模型 SFT 后不会突然会写 comparator。RL 也救不了 —— RL 只能优化已有的行为分布，不能凭空创造。

**推论 2：数据里的错误会被学到。**
如果你的训练集里 30% 的 Verilog-A 编译失败，SFT 后的模型也会有相当一部分输出编译失败。Garbage in, garbage out **比一般情况下更严格** —— 因为 SFT loss 不区分"对的 token"和"错的 token"。

**推论 3：数据分布的偏差会放大。**
训练集 60% 是简单运放，30% 是比较器，10% 是 SAR ADC？SFT 后模型在 SAR ADC 上的表现会比这个比例还要差，因为简单情况的 loss 更低（模型更自信）所以梯度更小，难情况学得慢。

**推论 4：质量 $\gg$ 数量。**
1000 条全部 EVAS 验证通过的数据 $\gg$ 10000 条混着噪声的数据。Circuit-Think 用 1000 条 SFT，DeepSeek 用更少。**这不是"数据效率高"，是"高质量数据下学得稳"**。

→ 这就是为什么我们的 Phase 1 把"EVAS 验证"作为强制门槛。**没经过 EVAS 验证的数据不进训练集，没有例外**。

## 4. 基座模型是一个超强的先验

SFT 不是从零开始学。它是在 `Qwen2.5-Coder-7B` 这个已经会写代码的模型上**做微调**。

这意味着：

### 4.1 SFT 只需要"重新分配概率"，不需要"学新东西"

基座模型已经知道：
- Verilog-A 的语法
- 模块声明 `module ... endmodule` 的结构
- 常见 EDA 习惯（虽然不一定准确）

SFT 要做的，是把基座模型**已有的概率分布**重新调整。比如：
- 让"模块名是 `comparator` 时，端口顺序为 $(\text{vin}_1, \text{vin}_2, \text{vout})$"这个条件概率从 5% 提到 80%
- 让"输出前先写 `<think>` 标签"这个条件概率从 0.001% 提到 99%

这是为什么 SFT 几千个样本就能改变模型行为：**我们不是在教新东西，我们在重新分配已有的概率**。

### 4.2 训太狠会"擦掉"基座能力

但这把双刃剑也有反面。如果 SFT 把模型推得太远（loss 降太低、epoch 太多、数据太狭窄），**模型会忘掉基座的通用能力**。常见症状：
- SFT 后模型只会输出 Verilog-A，问它写 Python 它也输出 Verilog-A
- 输出格式僵化，遇到没见过的 spec 就崩
- 一般推理能力下降

这叫 **catastrophic forgetting**。SFT 默认只跑 2 个 epoch、用相对低的 LR（$10^{-5}$ 量级），**就是为了不擦得太狠**。

### 4.3 选基座 = 选起点 + 选最大能力上限

- 基座对你的任务越友好（已经见过类似数据）→ SFT 越省力
- 基座越大 → 最大能力上限越高
- 基座代码能力越强 → 你的 Verilog-A 任务起点越高

我们选 **Qwen2.5-Coder-7B** 因为：
- Coder 系列预训练里见过大量代码（含 Verilog / SystemVerilog / VHDL，虽然 Verilog-A 不一定多）
- 7B 在 $2 \times$ A100 上做全参 SFT 还吃得消
- 同尺寸下代码任务上常常 SOTA 或接近 SOTA

## 5. SFT 能教什么 vs 不能教什么

| 能 | 不能（得靠 RL 或别的） |
|---|---|
| 输出格式（XML 标签、JSON 结构） | 输出正确的语义内容 |
| 风格（简洁 vs 详细、注释 vs 不注释） | 内容的"对错" |
| 任务条件化（看到 "spec → " 就生成 VA） | 多个对答案中挑最好 |
| 词表偏好（用 `electrical` 而不是 `wire`） | 推理多步骤后的全局正确性 |
| 已有能力的"激活" | 全新事实知识的注入 |

**最关键的两条**：

1. **SFT 不能教"正确性"**。它只能教"看起来像训练集"。如果训练集里的代码都对，模型倾向输出对的；如果训练集有错，模型也学错。**SFT 没有"自己验证"的机制**。
2. **SFT 不能教"在多个对答案中选最好"**。一个 prompt 在数据里多次出现，对应不同 target，SFT 会学到平均分布 —— 这是 mode covering，输出会变成不像任何一个 target 的"中间品"。

这两点直接动机了 **为什么 SFT 之后要做 RL**（见第 7 节）。

## 6. Token-level 训练 vs Sequence-level 评估：经典缝隙

SFT 在 token-level 算 loss。但我们真正关心的指标是 **序列级别** 的：
- 整个 `.va` 文件是否编译？
- 整个仿真是否跑通？
- 输出是否符合规范？

这两个粒度**不对齐**。具体后果：

### 6.1 Loss 很低不代表生成质量高

模型可能在 99% 的 token 上预测得很准（loss $\approx 0.1$），但在关键的 1% 上错了（一个语法错误的字符），整个 .va 就不编译。**生成质量的损失函数和训练 loss 不是一个东西**。

### 6.2 Exposure bias

训练时：模型每一步看到的都是 ground-truth 的前缀（teacher forcing）。
推理时：模型每一步看到的是自己生成的前缀。

如果自己生成的前缀里早早出现一个错误，后续生成就在"错误的条件"下进行，错误会**滚雪球**。这个 train-test gap 叫 **exposure bias**。

减缓办法（后续章节会展开）：
- 训练时偶尔用模型自己的 token（scheduled sampling）—— 工程复杂，效果有限
- **RL 训练**直接解决：RL 训练时模型本来就在用自己的前缀，没有 mismatch ← 这是更强的理由偏好 RL

### 6.3 直接后果：评估必须看序列级别

不要只看 loss 曲线判断 SFT 是否成功。必须跑完整生成 + EVAS 验证。loss $= 0.5$ 但 compile rate $= 20\%$ 的 checkpoint，是不可用的，哪怕 loss 数字好看。

## 7. SFT → RL：为什么是这个顺序，不是反过来

经常被问：能不能直接对基座做 RL，跳过 SFT？

**理论上可以，实际上几乎一定训不动**，原因纯粹是 reward 信号问题。考虑 GRPO 的优势归一化公式：

$$
\hat{A}_i \;=\; \frac{R_i - \mathrm{mean}(\{R_1, \dots, R_N\})}{\mathrm{std}(\{R_1, \dots, R_N\})}
$$

基座模型生成 `.va` 的 compile rate 大约 5–15%。假设 group 大小 $N = 8$：

- 大概率 8 个样本里全部 $R_i = 0$ → $\mathrm{std} = 0$ → $\hat{A}_i$ 未定义（NaN）
- 即使偶尔有一两个非零 → $\mathrm{std}$ 极小 → $\hat{A}_i$ 数值爆炸 → 训不稳

SFT 的作用是把基座模型推到"中等成功率"的区间。Circuit-Think 论文里 SFT 后 compile/格式正确率大约 30–40%，这时候 group 里既有成功也有失败，**reward 有方差**，GRPO 才能学。

**所以 SFT 的角色是**：
1. 教会格式
2. 把基础成功率提到 RL 可用的水平（$\geq 20\%$）
3. 提供一个 reference policy 让 KL 项有意义

→ 一句话：**SFT 是 RL 的入场券**。

## 8. 常见误区

### "Loss 越低越好"

错。Loss 是 token-level 的代理，不是任务目标。SFT 训到 loss $= 0.05$ 但 compile $= 20\%$ 的 checkpoint **不如** loss $= 0.4$ 但 compile $= 55\%$ 的 checkpoint。

**正确的判据**：在 held-out 上的任务级指标（compile / sim / correct）。Loss 只是辅助。

### "训得越久越好"

错。SFT 数据小时（几百到几千），通常**第 1–2 个 epoch 就接近最优**，第 3 个 epoch 开始过拟合。Circuit-Think 表 1 里 SFT 200 步反而比 50 步差，就是这个现象。

### "数据越多越好"

不全错，但常被滥用。在 SFT 的小数据 regime，10000 条平庸数据 < 1000 条精品。

**何时多更好**：数据质量已经控制住，模型还远未饱和。
**何时少更好**：数据有噪声或不一致。

### "SFT 学的是知识"

误导。SFT 主要学的是**条件分布的形状**（什么样的 $x$ 该映射到什么样的 $y$），不是"事实记忆"。教模型一个新的 SPICE 关键字几乎可以靠 SFT 做到；教模型一个新的电路原理几乎不行 —— 后者需要预训练规模或者 RAG。

### "SFT 越精细越好（包括做 step-by-step trajectory）"

半对。结构化 trajectory 确实有用（这就是 Circuit-Think 的关键），但它的价值**不只来自 SFT**，更多来自 RL 阶段能对 trajectory 做 step-wise reward。**只 SFT trajectory 但不在 RL 用它，价值有限**。

## 9. 本章核心 take-aways

1. **SFT = 换个数据分布继续做 next-token prediction**。loss 和 pretraining 一样。
2. **行为克隆视角更有用**。让你直觉到为什么数据质量决定一切。
3. **数据是 SFT 的灵魂**：没在数据里的能力，SFT 不会创造；数据里的错，SFT 会忠实学到。
4. **基座是先验**：SFT 在重新分配已有概率，不是从零学。训太狠会擦掉基座能力。
5. **SFT 能教格式 / 风格 / 任务条件化；不能教正确性 / 多答案选优**。
6. **token-level 训练 vs sequence-level 评估有缝隙**。loss 低不等于生成好。判 SFT 必须看任务指标。
7. **SFT 之前 RL 训不动**，因为 reward 信号稀疏到无法构成 advantage。

## 10. 下一章的入口

读完这一章之后，最自然的下一个问题是：**"那数据具体怎么准备？格式怎么定？为什么 prompt 要 mask？"**

→ 进 `02_data_and_format.md`。

如果你想先跳过数据，看训练机制：
→ 进 `03_training_loop.md`。

如果你想看显存预算（决定能不能跑得动）：
→ 进 `04_memory_and_scale.md`。
