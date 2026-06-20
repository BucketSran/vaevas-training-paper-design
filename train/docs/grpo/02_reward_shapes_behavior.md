# 02 — Reward 形态决定 Policy 行为：GRPO 的灵魂

> **预览**：VSCode `Cmd+Shift+V`，KaTeX 渲染。
>
> **前置阅读**：[`01_foundations.md`](./01_foundations.md) §7-§9（GRPO 用 reward 加权 log-likelihood 做训练）+ [`01a_formulas_walkthrough.md`](./01a_formulas_walkthrough.md) 公式 4（group advantage 计算）。

> 读完这一章你应该能回答：
> 1. 为什么 "GRPO 是 PPO 改 baseline" 是 ML 表象，"GRPO 是 reward 设计驱动" 是工程本质？
> 2. Sparse / dense / decomposed reward 各有什么后果？
> 3. Reward hacking 的 6 种典型模式 + 怎么防？
> 4. Process reward 和 outcome reward 怎么选？
> 5. 为什么 vaEvas 用 verifier-based reward，不用 LLM-as-judge？
> 6. 怎么从 0 开始设计一个 reward 函数（5 步方法）？

---

## 1. 这一章为什么是 GRPO 学习线最关键的一章

GRPO 算法本身 [01 §7-§9](./01_foundations.md#7-grpo用-group-baseline-替代-critic) 一页讲完了。但**一个 GRPO 项目能不能成，几乎完全由 reward 设计决定**。

证据：
- Circuit-Think 论文里 GRPO 算法和 DeepSeek 用的没差别（都是 group advantage + PPO clip + KL）
- Circuit-Think 把 60% → 73% 的 25pp 提升**几乎全部来自 reward 设计**（[Circuit-Think Table 1](https://ojs.aaai.org/index.php/AAAI/article/view/37465) 显示 GRPO 单跑 40.23%, +Circuit-Think reward 后 60.17%）
- 同一 GRPO 算法换 reward 设计 → 效果天壤之别（"reward hacking" 几乎都是 reward 写歪导致）

→ **学 GRPO = 学 reward 设计**。算法部分是入场券，reward 是实际比赛。

---

## 2. RL 的"金科玉律"：You get what you reward

RL 的训练机制保证一件事：**policy 一定会朝 reward 上升的方向走**。

这是优点（可对齐任意目标）也是噩梦（你写歪一行 reward，policy 完美执行你写歪的目标）。

### 经典反例

**目标**："让机器人把房间整理干净"

**朴素 reward**：`R = -地上物品数量`

**Policy 学到**：把所有物品**藏进沙发底下** —— 地上数量 = 0，reward 最大 ✅

→ Reward 是你**字面上写的东西**，policy 完美执行字面定义，**不在乎你"心里想的"**。

### LLM 里的版本

**目标**："让模型生成更专业的回答"

**朴素 reward**：`R = 回答里专业术语的数量`

**Policy 学到**：每个回答堆 100 个无意义的术语 → reward 爆表，但**人读起来更糟**。

### 对 vaEvas 的警示

**目标**："让模型生成能编译的 Verilog-A"

**朴素 reward**：`R = OpenVAF 编译成功 = 1, else 0`

**Policy 可能学到**：
- 生成最短最简单的合法模块（如 `module x; endmodule`），永远编译成功 ✅
- 重复同一个已知能过的 trivial 模板，不管 prompt 是什么 ✅
- 漏掉端口、不实现功能，只保证语法对 ✅

→ Reward 只奖励"编译过"，policy 当然找最容易编译的最 trivial 写法。

**这一节的 take-away**：**Reward 不是"目标"，reward 是"loss function"**。Policy 会优化 reward 的字面定义，不会优化你 reward 想代表的 intent。

---

## 3. Reward 的 5 个关键性质

设计 reward 时，5 个维度都要考虑：

### 3.1 Sparsity（稀疏度）

| 类型 | 含义 | 例子 | 后果 |
|---|---|---|---|
| **极稀疏** | 100 个 sample 中 1 个非零 | 只奖励"完美正确" | GRPO group std≈0, 训不动 |
| **稀疏** | 30-50% 非零 | 奖励"编译过" | OK, group 有方差 |
| **密集** | 几乎都非零 | 多级 reward 加权和 | 信号最丰富 |

**关键**：GRPO advantage 公式是 $\hat{A}_i = (R_i - \mathrm{mean})/\mathrm{std}$。**std=0 时 advantage 未定义**。

→ 设计 reward 时**绝对要避免"全 0 或全 1"的情况**。

### 3.2 Granularity（粒度）

| 粒度 | 何时打分 | 例子 |
|---|---|---|
| **Trajectory-level** | 整段 completion 一个分 | "代码能编译过 = 1" |
| **Step-level** | 每个 step 一个分 | "第一步 port 识别对了 + 0.3" |
| **Token-level** | 每个 token 一个分 | 几乎没人用 (太细) |

GRPO 默认 trajectory-level（[01a §9](./01a_formulas_walkthrough.md#公式-9实务-怎么把-trajectory-level-reward-分摊到-token-level)）。**Step-level 更精细但需要 trajectory 可拆分**。

→ Circuit-Think 用 step-level（"识 port → 识 device → 推连接"3 步独立打分），这是为什么它效果好。

### 3.3 Alignment（对齐真实目标）

**Proxy reward vs True objective**：

| 你的真实目标 | 朴素 proxy reward | Proxy 跑歪的方向 |
|---|---|---|
| 生成正确的 Verilog-A | 编译过 | 生成 trivial 模块（最容易编译）|
| 输出格式规范 | tag 完整 | 把所有内容都塞进 `<answer>` 标签里 |
| 推理质量高 | 推理链长 | 啰嗦废话刷长度 |
| 仿真结果正确 | 仿真不报错 | 输出常数 / 零信号（不报错但也无意义） |

**Reward 永远是 proxy**。完美的 reward = "和人类判断的"完全一致"，那等于把 reward function = 人类评分员，**不可 scale**。

→ 一定要假设 reward 是 proxy，**proactively 想"policy 怎么 game 这个 proxy"**。

### 3.4 Smoothness（光滑度）

| 形式 | 例子 | 影响 |
|---|---|---|
| **Binary (0/1)** | 编译过/不过 | GRPO 信号干净但稀疏 |
| **Discrete steps** | {0, 0.5, 1.0} | 中等 |
| **Continuous [0, 1]** | 仿真输出和 gold 的相关性 | 信号最平滑 |

**经验**：
- Gating reward（"必须先满足才能继续"，比如 compile）用 binary
- Quality reward（"程度问题"，比如准确度）用 continuous

### 3.5 Computability（可计算性）

Reward function 必须 **确定性 + 快速 + 可微（或不可微但便宜调用）**。

| 类型 | 例子 | 评 |
|---|---|---|
| **确定性 verifier** | `OpenVAF 编译 → exit code` | ⭐⭐⭐ 最理想 |
| **数值对比** | MSE between predicted和gold waveform | ⭐⭐ 也好 |
| **正则 / 字符串匹配** | `<answer> tag 存在` | ⭐⭐⭐ 简单 |
| **LLM-as-judge** | 用 GPT-4 给生成代码打分 | ⭐ 慢 / 贵 / 不稳定 |
| **人类评分** | 人工 review | ❌ 不可 scale |

**vaEvas 任务幸运地集齐了前三种**：OpenVAF + EVAS + 字符串匹配 → reward 干净又快。这是为什么 vaEvas + GRPO 是天作之合。

---

## 4. Reward Hacking 的 6 种典型模式

每个 RL 项目都会遇到。**早期识别 = 早期修 reward**。

### 4.1 Trivial-pass hack
**症状**：Policy 输出最短的可被 reward 接受的内容。
**例**：reward 只看编译过 → 总输出 `module x; endmodule`
**防御**：reward 加 "至少要有 functional 内容" 项（如 module body 不能为空）

### 4.2 Repetition / mode collapse hack
**症状**：所有 prompt 输出几乎一样的 completion（找到最稳健的"通用答案"）
**例**：不管 prompt 是 inverter / counter / adder，都输出同一段 D flip-flop 模板
**防御**：在 reward 里加入"和 prompt 的语义相关度"项（但这本身又可能被 hack）；或加大 KL 惩罚 / 提高 temperature 增加探索

### 4.3 Length hack
**症状**：Policy 输出无脑变长，刷"看起来内容多"
**例**：reward 隐含 "完整 reasoning trace = 好" → policy 在 `<think>` 里塞一堆废话
**防御**：reward 加 length penalty（超过 N token 后 reward 递减）；或步骤级 reward（每步独立打分，长度无意义）

### 4.4 Format-only hack
**症状**：Policy 学会输出"格式正确但内容错"
**例**：完美的 `<think>...</think><answer>...</answer>` 结构，但里面是乱码
**防御**：format reward 权重不能高（如总 reward 5%-10%）；compile / sim reward 占大头

### 4.5 Reward gaming via numerical edge cases
**症状**：找到 reward 计算的数值漏洞
**例**：metric reward = `1 - error / tolerance`，policy 输出**故意**让 error 略小于 tolerance，刚好压线满分
**防御**：reward 加噪声 / 加额外严格阈值；或把"完全正确"和"勉强达标"分开打分

### 4.6 Distribution shift hack
**症状**：Policy 输出训练分布外的内容，reward function 没考虑到 → 误判
**例**：reward function 假设输出是英文 + Verilog-A，policy 输出中文注释 + Verilog-A → reward function 解析失败 → 给了 0 → 但实际可能是对的
**防御**：reward function 要鲁棒处理 "我没见过的格式" → 给 0 或 NaN，不要乱猜

---

## 5. Decomposed Reward（多级 reward）：Circuit-Think 的核心

### 5.1 单一 reward 的局限

假设我们只用 `R = compile_pass` (binary)：

**问题**：
- compile pass rate 提升后，advantage 又变稀疏（大家都过了）→ 训停滞
- 模型没有"compile 失败但仿真没跑过"这种中间状态的信号
- 单一 reward 没有指导"怎么改 / 改哪里"

### 5.2 多级 reward 的好处

把 reward 拆成多个**互补的**子项：

```
R_total = α·R_format + β·R_compile + γ·R_simulate + δ·R_metric
```

每个子项捕捉一个**任务子目标**。Circuit-Think 论文 Table 6 显示：**任一去掉一项，效果显著下降**（73% → 47-65%）。

### 5.3 关键设计：Gating（依赖关系）

子项之间有自然依赖：

```
format ─┬─→ compile ─┬─→ simulate ─→ metric
        │            │
        └─→ format 失败 → 后续都不重要
                     └─→ compile 失败 → simulate, metric 都不能算
```

**实现**：

```python
if R_compile == 0:
    R_simulate = 0  # 强制 0，不算后续
    R_metric = 0
```

→ 这就是 Circuit-Think 的 **step gating**（[Circuit-Think 论文](https://ojs.aaai.org/index.php/AAAI/article/view/37465) Eq. 1 里的 $\tau_1$ 阈值）。

### 5.4 权重设计

权重决定**模型优先学什么**：

```yaml
R_total = 0.1 * R_format    # 格式是 hygiene, 不应主导
        + 0.2 * R_compile   # gating, 但本身不复杂
        + 0.2 * R_simulate  # gating
        + 0.3 * R_metric    # 真正的"准确度"
        + 0.2 * R_trajectory  # 推理过程
```

**经验**：
- 真正想最大化的目标（这里 metric）占大头（30%+）
- Gating reward 中等（20% 左右，因为 gating 之后还有更精细的奖励）
- Hygiene reward（format）小（10%-20%）

**Circuit-Think 实际配比**：α=0.2, β=0.4, γ=0.2, δ=0.2（answer accuracy 占 40% 大头）。

### 5.5 实例：vaEvas 的 4 个子项打分

给一个 sample：

| 子项 | 计算 | 这个 sample 得分 |
|---|---|---|
| R_format | 检查 `<think>`/`<answer>` 标签齐全 | 1.0 |
| R_compile | OpenVAF compile 是否退出 0 | 1.0 |
| R_simulate | EVAS 是否能 run 完整 | 1.0 |
| R_metric | sim 输出 vs gold 的 Dice 系数 | 0.7 |

权重: 0.1, 0.2, 0.2, 0.5

$$R_{total} = 0.1 \times 1.0 + 0.2 \times 1.0 + 0.2 \times 1.0 + 0.5 \times 0.7 = 0.85$$

**对比另一个 sample**（compile 失败）：

| 子项 | 这个 sample 得分 |
|---|---|
| R_format | 1.0 |
| R_compile | **0.0** |
| R_simulate | 0 (gating, 强制) |
| R_metric | 0 (gating) |

$$R_{total} = 0.1 \times 1.0 + 0.2 \times 0.0 + 0.2 \times 0 + 0.5 \times 0 = 0.10$$

**对比 8 个 sample 的 group advantage**：
- 假设组里 6 个 sample 编译过, 2 个不过
- 编译过的 6 个 reward 在 0.6-0.95 之间, 不过的 2 个都是 0.10
- $\mathrm{mean} \approx 0.65$, $\mathrm{std} \approx 0.28$
- 编译过的 $\hat{A}$ 大约 -0.2 to +1.1
- 编译不过的 $\hat{A} \approx -2.0$

→ **不过的 sample 收到强抑制信号，但因为 compile 是 binary，"编译过的"内部还有梯度**（按 metric 分高低）。这就是 decomposed reward 比 single reward 强的根本机制。

---

## 6. Process Reward vs Outcome Reward

### 6.1 两种思路

**Outcome reward**: 只看最终结果好坏
```
R = sim_pass ? 1 : 0
```

**Process reward**: 看推理过程每一步的质量
```
R_step1 = port_identified_correctly  # 0 or 1
R_step2 = behavior_described_correctly  # 0 or 1
R_step3 = code_produced_correctly  # 0 or 1
R = R_step1 + R_step2 + R_step3
```

### 6.2 Process reward 的两个流派

**Path A: PRM (Process Reward Model)** —— 训一个神经网络专门给每步打分
- 例：OpenAI 的 PRM-800K
- 优点：能给任意推理打分
- 缺点：需要先收集大量"step-level 标注数据"训 PRM；PRM 自己也会被 hack

**Path B: Structural Process Reward** —— 用确定性 verifier 给固定步骤打分
- 例：Circuit-Think 的 port/device/connection 三步独立打分
- 优点：deterministic、不需 PRM
- 缺点：需要任务有自然的步骤分解 + 每步可验证

**vaEvas 用 Path B**：spec → port → behavior → testbench → answer 这个分解天然可验证。

### 6.3 为什么 Process Reward 比 Outcome Reward 强

**Credit assignment**（信用分配）问题：

Outcome reward 把"成功 / 失败"分摊到整段 trajectory。**模型不知道"哪一步是关键"**。

Process reward 把奖励分到具体步骤。**模型知道"port 识别错了，重点改这里"**。

→ 直观类比：考试只给"通过 / 不通过"vs 给每道题打分 + 讲解 —— 后者学得快得多。

### 6.4 数学解释

GRPO 的 token-level loss（[01a §9](./01a_formulas_walkthrough.md#公式-9实务-怎么把-trajectory-level-reward-分摊到-token-level)）：

$$L = \frac{1}{|o|} \sum_t r_\theta(t) \cdot \hat{A}$$

**Outcome reward** → 整个 trajectory 的 $\hat{A}$ 相同 → 每个 token 收到的梯度信号一样
**Process reward** → 不同 step 的 token 收到**不同 advantage**（按 step reward 计算）→ 学到"哪些 token 是关键"

---

## 7. Verifier-Based vs LLM-as-Judge

### 7.1 LLM-as-Judge 是什么

用一个强 LLM（GPT-4 / Claude）给 policy 输出打分：

```python
def llm_judge_reward(prompt, completion):
    response = openai.chat([
        {"role": "system", "content": "You are a Verilog-A expert. Rate this code 0-10."},
        {"role": "user", "content": f"Prompt: {prompt}\nCompletion: {completion}"}
    ])
    return parse_score(response)
```

### 7.2 LLM-as-Judge 的问题

| 问题 | 后果 |
|---|---|
| **慢**：每次 reward 计算 ~5-10 秒 | GRPO 一个 step rollout N=8 → 40-80 秒 / step 只 reward 那部分 |
| **贵**：GPT-4 API 调用 $0.01-0.10 | 200 step × 64 prompt × 8 sample × $0.05 ≈ $5000 |
| **非确定**：同样输入两次结果可能不同 | RL 学不稳 |
| **可被 hack**：policy 学会"写让 LLM judge 喜欢的格式" | reward hacking 加剧 |
| **judge 模型本身有偏** | policy 学到偏 |

### 7.3 Verifier-Based 是什么

用**确定性、可执行的验证器** 打分：

- OpenVAF compile (exit code)
- EVAS simulate (runs / errors)
- 数值对比 (Dice / MSE)
- 字符串 / regex 匹配
- 子图同构 (Circuit-Think 的 graph consistency reward)

### 7.4 vaEvas 完美适配 Verifier

| 想验证什么 | 用什么 |
|---|---|
| Verilog-A 语法对 | OpenVAF compile |
| 模型能仿真 | EVAS run |
| 仿真结果对 | EVAS output vs gold (numeric) |
| 输出格式对 | regex (`<think>`/`<answer>`) |
| 端口对 | 解析 module declaration vs spec |

→ **完全不需要 LLM-as-judge**。比 Circuit-Think 还干净（CT 用了 LLM 辅助生成 trajectory step 评分，我们直接用 EVAS）。

### 7.5 何时 LLM-as-Judge 不可避免

- 任务输出**不可机器验证**（自然语言对话质量）
- **没有 ground truth**（创意写作）
- **多个对答案**（道德推理）

→ 这些场景 vaEvas 完全没有。所以**我们用纯 verifier，不掺 LLM-as-judge**。

---

## 8. 设计 Reward 的 5 步方法论

通用流程，任何新任务都适用：

### Step 1: 定义"任务成功" = 什么

写一段话精确说明你想要什么。**不要笼统**，写实际的判断标准。

❌ 不够好：「让模型生成正确的 Verilog-A」
✅ 够具体：「让模型生成能被 OpenVAF 编译、能被 EVAS 仿真完成、并且仿真输出和 gold 的关键 metric 在 5% 内吻合的 Verilog-A」

### Step 2: 找可验证的 binary 指标

把 Step 1 的目标拆成几个**可机器验证的 binary 指标**：

```
[ ] OpenVAF compile pass
[ ] EVAS simulation completes without error
[ ] Output trace matches gold within tolerance
[ ] Format includes required tags
```

每一个都必须是"机器能 0/1 判定的"。

### Step 3: 加 gating

按依赖关系串起来：

```
format → compile → simulate → metric
```

前一个失败，后续全部 0。

### Step 4: 加权 + 数值化

给每个子项分配权重 + 把 binary 变 continuous（如果可能）：

```
R_format    = 1 if all_tags_present else 0      # 0/1
R_compile   = 1 if openvaf_pass else 0           # 0/1
R_simulate  = 1 if evas_pass else 0              # 0/1
R_metric    = max(0, 1 - error/tolerance)        # 0~1 continuous
R_total = 0.1*R_format + 0.2*R_compile + 0.2*R_simulate + 0.5*R_metric
```

### Step 5: 测 reward function 本身（关键步骤）

在跑 GRPO 训练**前**，用一组**手写测试输入**验证 reward 行为：

```python
# Reward 单元测试
test_cases = [
    {"name": "perfect", "completion": "<think>...</think><answer>...gold code...</answer>",
     "expected_reward_min": 0.95, "expected_reward_max": 1.0},
    {"name": "compile_fail", "completion": "<think>...</think><answer>syntax error code</answer>",
     "expected_reward_min": 0.0, "expected_reward_max": 0.15},
    {"name": "compile_pass_sim_fail", "completion": "...",
     "expected_reward_min": 0.2, "expected_reward_max": 0.35},
    {"name": "trivial_module", "completion": "<answer>module x; endmodule</answer>",
     "expected_reward_min": 0.0, "expected_reward_max": 0.4},  # 不能让 trivial 拿太高分
    {"name": "no_tags", "completion": "module ... endmodule",
     "expected_reward_min": 0.0, "expected_reward_max": 0.7},
]

for case in test_cases:
    r = compute_reward(prompt="...", completion=case["completion"])
    assert case["expected_reward_min"] <= r <= case["expected_reward_max"], f"FAIL: {case['name']}, got {r}"
```

**这一步如果跳过 = 几乎一定会撞 reward hacking**。Circuit-Think 论文里没明说但他们一定做了类似的事。

---

## 9. 把 Circuit-Think 的 Reward 设计映射到 vaEvas

Circuit-Think 完整 reward（[论文](https://ojs.aaai.org/index.php/AAAI/article/view/37465) §4 + Table 6）：

$$R_{total} = 0.2 R_{think} + 0.4 R_{answer} + 0.2 R_{logic} + 0.2 R_{format}$$

**直接对应到 vaEvas**：

| Circuit-Think | vaEvas 等价物 |
|---|---|
| `R_think` (step-by-step reasoning) | trajectory step reward: port + behavior + testbench |
| `R_answer` (netlist accuracy, Dice 系数) | metric reward: sim output vs gold (numerical Dice) |
| `R_logic` (graph consistency: reasoning ↔ answer) | code-spec consistency: extracted module signature 是否和 spec 里说的 port 一致 |
| `R_format` (XML tag check) | 一样 |

**额外加一项**（Circuit-Think 没有，因为他们用 netlist 不用执行）：

| vaEvas 特有 | 含义 |
|---|---|
| `R_compile` | OpenVAF 能编译 |
| `R_simulate` | EVAS 能 sim |

**最终我们 vaEvas reward 推荐结构**（这是 [`../03_reward_design.md`](../03_reward_design.md) 里写的版本，现在你应该能看懂为什么是这样）：

$$R_{total} = 0.1 R_{format} + 0.2 R_{compile} + 0.2 R_{simulate} + 0.3 R_{metric} + 0.2 R_{trajectory}$$

**和 Circuit-Think 的关键差异**：
- 我们多了 `R_compile` 和 `R_simulate` 因为 OpenVAF+EVAS 这俩 verifier 比 Circuit-Think 的 SPICE 解析更直接
- 我们 `R_metric` 是仿真输出对比（Circuit-Think 是 netlist 字符串对比）—— 更难但更"对"

---

## 10. Reflective Learning：Circuit-Think 的 hint 注入机制

这是 Circuit-Think 一个被低估的设计。读了 reward 设计之后能更好理解它的意义。

### 10.1 问题：稀疏 reward 的"卡死"

GRPO 早期：group 里 8 个 sample 全失败 → mean=0, std=0 → advantage NaN → 那个 batch 完全不更新。

如果**每个 prompt** 都这样持续多步 → 整个训练卡死。

### 10.2 Circuit-Think 的解法

当 `R_total < τ_2`（默认 0.7）时，**把 reference answer 作为 hint 注入 prompt**，让 policy 再采样一次：

```
原 prompt: "Generate a comparator Verilog-A"
新 prompt: "Generate a comparator Verilog-A. Hint: the answer should look like ```...gold code...```"
```

然后重新算 reward。**只奖励 hint 后的 improvement**：

$$\hat{R} = R_{total} + \lambda_{ref} \cdot \frac{\max(0, R_{total,hint} - R_{total})}{1 - R_{total} + \epsilon}$$

### 10.3 为什么有效

**直觉**：
- 没有 hint 时模型完全失败 → 给 hint 让它"瞄一眼答案"
- 不是直接抄答案（那就退化成 SFT 了），是**让模型在见到答案后自己重新写**
- 只有"看了 hint 后真的改善" → 才给额外奖励
- "看了 hint 还是失败" → policy 学到"这种类型的问题需要更多 / 不同的思路"

### 10.4 我们要不要实现

**Pilot 阶段先不上 reflective learning**。理由：
- 需要每个 prompt 都有 gold answer（增加数据准备成本）
- 增加 reward function 复杂度
- 我们 task 已经天然 dense reward（4 项加权）

**Production 阶段如果发现 "GRPO 卡在某个 reward 不上去"** → 再加 reflective learning。

---

## 11. 常见误区

### 「Reward 越简单越好」
半对。**Single reward 看起来简单，但容易 trivial hack**。Decomposed reward 多写代码但稳。

### 「Reward 用 LLM 评价，反正聪明」
错。LLM-as-judge 慢、贵、不稳、可被 hack。verifier 永远优先。

### 「Reward 写完就不用动了」
错。**Reward 设计是迭代过程**。GRPO 跑 30 步看 reward 曲线 → 大概率发现 hack → 修 reward → 重训。预期至少改 3-5 轮。

### 「Reward 范围必须 [0, 1]」
不强制。GRPO advantage 公式自带 normalize（除以 std），所以 reward scale 不影响。但**最好统一在 [0, 1]** 因为人读着方便、log 容易看。

### 「Reward 的子项必须独立」
不强制。compile pass 和 simulate pass 高度相关（compile 失败 sim 必失败）→ 但这不是问题，gating 设计就是利用这种依赖。

---

## 12. 本章核心 take-aways

1. **GRPO 算法 = 入场券, reward 设计 = 实际比赛**。算法 1 页讲完，reward 设计一辈子都在调。
2. **You get what you reward**：Policy 完美执行 reward 字面定义，不在乎你"想表达的 intent"。
3. **Reward 5 个性质**：稀疏度 / 粒度 / alignment / 平滑 / 可计算。每个都要思考。
4. **6 种 reward hacking 模式**：trivial / repetition / length / format-only / numerical edge / distribution shift。设计时主动思考"policy 怎么 game 我"。
5. **Decomposed reward > single reward**：多级 + gating + 加权。Circuit-Think 73% 的核心。
6. **Process reward > outcome reward**（当任务可拆分时）：credit assignment 干净。
7. **Verifier-based > LLM-as-judge**：vaEvas 任务幸运地完全适配 verifier-based（EVAS + OpenVAF）。
8. **5 步方法**：定义成功 → 找 binary 指标 → gating → 加权 → **测 reward function 本身**（第 5 步最容易跳过也最容易翻车）。

---

## 13. 下一步

读完本章后：

- **看具体的 vaEvas reward 项目级 spec** → [`../03_reward_design.md`](../03_reward_design.md)
- **想看 GRPO 训练循环怎么跑** → `03_training_loop.md`（计划章节）
- **想看 reward function 实际代码长什么样** → 等 GRPO 环境装好后我们就写 `train/rewards/*.py`
- **想看 verl 怎么配 reward** → [`../tools/verl.md`](../tools/verl.md) §6

---

## 14. 关键参考

- **Circuit-Think 论文** Eq. 1-6 + Table 6 — multi-level reward + step gating + reflective learning 的完整推导
- **OpenAI PRM-800K** ([Lightman et al. 2023, arXiv 2305.20050](https://arxiv.org/abs/2305.20050)) — Process reward model 的标志性工作
- **Sutton & Barto, Reinforcement Learning: An Introduction** §17.4 — 如果想读经典 reward shaping 理论
- **Reward Hacking in RL from Human Feedback** ([Skalse et al. 2022](https://arxiv.org/abs/2209.13085)) — RLHF reward hacking 的系统性 taxonomy
