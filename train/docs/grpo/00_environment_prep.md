# 00 — GRPO 训练环境准备计划

> **预览**：VSCode `Cmd+Shift+V`。
>
> **定位**：[`01_foundations.md`](./01_foundations.md) 讲 GRPO **原理**；本章讲实际跑 GRPO **前**需要的环境准备。和 SFT 的环境准备类似，但因为多了 RL 框架 + rollout 引擎 + EVAS verifier，需要分几个阶段。

> 读完这一章你应该能：
> 1. 知道 GRPO 训练栈需要哪些组件
> 2. 知道哪些已经装好了、哪些还要装
> 3. 知道 EVAS verifier 怎么变成"reward 服务"
> 4. 有一个分 4 步的环境准备 checklist

---

## 1. GRPO 训练栈：必备组件清单

```
┌─────────────────────────────────────────────────────┐
│  RL 训练框架: verl                                    │
│  ├─ 协调 actor / ref / rollout / reward 4 个角色      │
│  ├─ 用 Ray 做多进程协调                              │
│  └─ 用 vLLM 做 rollout                              │
├─────────────────────────────────────────────────────┤
│  Rollout 引擎: vLLM                                  │
│  ├─ 每 step 给一批 prompt 采 N=8 个 completion        │
│  ├─ PagedAttention + prefix caching                  │
│  └─ 必须支持权重从 verl 训练进程同步                  │
├─────────────────────────────────────────────────────┤
│  Reward 函数: 我们自己写                             │
│  ├─ 调 OpenVAF 编译                                  │
│  ├─ 调 EVAS 仿真                                     │
│  ├─ 做数值/结构对比 (vs gold)                        │
│  └─ 返回 0-1 标量                                    │
├─────────────────────────────────────────────────────┤
│  基础设施                                            │
│  ├─ Conda env (已有, LF 装好的)                      │
│  ├─ 基座模型 = SFT checkpoint (Pilot 后才有)         │
│  ├─ RL prompts (从数据 pipeline 来)                   │
│  └─ wandb (实验追踪, 可选)                           │
└─────────────────────────────────────────────────────┘
```

---

## 2. 已经装好的（继承自 SFT 阶段）

| 组件 | 状态 | 路径 / 版本 |
|---|---|---|
| Conda env | ✅ | `/data/jinzhihong/envs/llamafactory/` |
| Python | ✅ | 3.11.15 |
| torch | ✅ | 2.6.0+cu124 |
| transformers | ✅ | 5.6.0 |
| peft | ✅ | 0.18.1 |
| trl | ✅ | 0.24.0 |
| accelerate | ✅ | 1.11.0 |
| datasets | ✅ | 4.0.0 |
| modelscope | ✅ | 1.37.1 |
| llamafactory | ✅ | 0.9.6.dev0 |
| CUDA driver | ✅ | 550.54.15 |
| GPU | ✅ | 8 × A100 80GB |

**LF env 已经够用作 GRPO 训练基础**。我们要在这个 env 上补几个东西。

---

## 3. 需要额外装的（实测时按顺序）

### 3.1 vLLM（rollout 引擎）

**为什么需要**：[verl 必须用 vLLM 做 rollout](../tools/verl.md#5-verl-的-yaml-配置长什么样)。LF 的 `requirements/metrics.txt` 没有装 vLLM。

**怎么装**：

```bash
ssh jinzhihong@117.148.167.107
source /data/home/maxiaokang/miniconda3/etc/profile.d/conda.sh
conda activate /data/jinzhihong/envs/llamafactory

# 验证当前有没有
pip show vllm 2>/dev/null && echo "Already have vLLM" || echo "Need to install"

# 没有的话装（vLLM 自带 PyPI wheel, cu124 编译版）
pip install vllm
# 这会带来: vllm + cu124 库 + flash-attn 等
```

**预期**：vLLM 0.6.x 或 0.7.x，下载量 ~2-3 GB。

**风险**：
- vLLM 装好后**可能 downgrade torch**（vLLM 自带 torch dependency）。检查 `pip show vllm` 看 install 输出。
- 如果 vLLM 试图装 torch 2.5.x → 用 `pip install vllm --no-deps` 然后手动补 vllm 真正需要的依赖
- 装完一定要重跑 `torch.cuda.is_available()` 确认 CUDA 还能用

### 3.2 verl（GRPO 训练框架）

**怎么装**：

```bash
# verl 通过 pip 装可能不带最新功能，建议 git clone + editable
cd /data/jinzhihong
git clone https://github.com/volcengine/verl.git
cd verl
pip install -e .

# 或者直接 PyPI:
# pip install verl
```

**预期**：~200 MB 仓库 + 装 Ray 等依赖。

**会拉来的依赖**：
- ray (multi-process orchestrator)
- omegaconf / hydra (config 系统)
- 其他小依赖

**风险**：
- verl API 在快速演进，建议 git checkout 一个稳定 tag（比如 v0.5.x）
- 第一次跑会发现各种 Ray 配置警告，无伤大雅

### 3.3 flash-attention（如果还没装）

**为什么需要**：vLLM 和 verl 训练都用 flash-attn 加速。

```bash
pip show flash-attn 2>/dev/null || pip install flash-attn --no-build-isolation
```

如果 vLLM 已经装过，flash-attn 大概率也已经有了。

### 3.4 wandb（实验追踪，可选但强烈推荐）

```bash
pip install wandb
wandb login  # 一次性，需要 API key
```

GRPO 训练时盯多个曲线（reward / KL / entropy / advantage），wandb 的多曲线对比比本地 PNG 方便 10×。

---

## 4. Reward 函数所需基础设施

GRPO 的灵魂是 reward function。我们 reward 要调用 EVAS + OpenVAF。这部分**不依赖 verl 安装**，但要单独设计。

### 4.1 EVAS 调用方式选择

| 方式 | 优点 | 缺点 |
|---|---|---|
| **Python 进程内直接调用** | 简单，无 IPC | 训练进程和 EVAS 互相影响（崩了一起死） |
| **Subprocess 启动 `evas` 命令** | 隔离好 | 每次启动开销，串行 |
| **EVAS-as-a-Service (HTTP)** | 隔离 + 并行 | 多写一层 |

→ **Pilot 阶段建议 subprocess 起步**，简单可靠。Production 时如果 reward 计算成为瓶颈再上 HTTP 服务。

### 4.2 EVAS 在远端的可用性 (TODO 实测)

需要确认：
- [ ] 远端是否装了 EVAS (`pip show evas-sim`)
- [ ] OpenVAF 是否在 PATH (`which openvaf`)
- [ ] Spectre 是否可用 (我们 GRPO 阶段先用 EVAS 不用 Spectre)

**如果 EVAS 没装**：

```bash
pip install evas-sim
# 验证: evas --version
```

### 4.3 Reward function 架构 (待 GRPO 阶段实现)

预计的 reward 函数签名（对接 verl）：

```python
# train/rewards/vaevas_reward.py
def compute_reward(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    """
    For each (prompt, completion):
      1. Parse <answer>...</answer> to extract VA code
      2. Run OpenVAF compile
      3. If pass, run EVAS sim
      4. If pass, compare numerical output to gold
      5. Return weighted total reward in [0, 1]
    """
    ...
```

具体逐项 reward 见 [`02_reward_shapes_behavior.md`](./02_reward_shapes_behavior.md)（下一章）和 [`../03_reward_design.md`](../03_reward_design.md)。

---

## 5. 显存预算 — GRPO 多模型同时活

[verl 笔记 §2](../tools/verl.md#2-rlhf-训练里有几个角色模型) 算过粗略预算。我们的具体场景：

| 角色 | 占用（7B bf16）|
|---|---|
| Policy (actor) | 14 GB 参数 + 56 GB optimizer (full) **或** 14 GB + 1 GB (LoRA) |
| Reference (frozen) | 14 GB |
| Rollout (vLLM) | 14 GB + KV cache (~5-20 GB) |
| **总计 (Full SFT actor)** | ~110 GB |
| **总计 (LoRA actor)** | ~50 GB |

**单 A100 80GB 装不下 Full SFT 4-model 配置**，必须：
- 用 **LoRA actor** (省 55 GB)
- 或者 **ZeRO-2/3 + offload** (省到 50-70 GB)
- 或者 **跨卡 colocation** (详见 verl 笔记 §3)

**我们的选择**：**LoRA actor + vLLM 占第二张卡**。即：

```
GPU 0:  Policy (LoRA) + Reference + critic placeholder ≈ 30 GB
GPU 1:  vLLM rollout engine ≈ 30 GB
GPU 2-7: 空闲 (或者别人的 workload)
```

→ **GRPO 训练只需要 2 张 GPU**（如果 LoRA + 适度 batch）。8 卡 production 可以做更大 batch。

---

## 6. 数据准备：GRPO 不需要 `(prompt, completion)` 对

**关键差异**：

| | SFT 数据格式 | GRPO 数据格式 |
|---|---|---|
| 一条样本 | `{instruction, output}` | `{prompt}` (只要输入！) |
| 为什么 | 模型学着模仿 output | 模型自己采 completion，靠 reward 学 |
| 量需要 | 100-2000 | 100-1000 prompts 就够 (每个采 N=8) |

具体 jsonl 格式：

```jsonl
{"prompt": "Generate a Verilog-A model for a clock divider with ratio 4."}
{"prompt": "Generate a Verilog-A model for a 2-input XOR gate."}
{"prompt": "..."}
```

**好消息**：GRPO 训练**不需要再准备 output**！SFT 阶段做过的 `.va` gold answer 这里不是必须的（reward function 也不需要 ground truth，只需要 EVAS verifier）。

→ 这意味着我们的 GRPO 数据准备**比 SFT 简单**，只需要"100-300 条 prompt"，不需要正确答案。可以直接复用 SFT 的 instruction 字段，丢掉 output 字段。

---

## 7. 完整 GRPO 环境准备 Checklist

```
■ 阶段 A: 基础（依赖 LF env，可立刻做）
  □ ssh 检查当前 env，列已装包
  □ pip install vllm (可能 ~10 min)
  □ 验证 vllm 能 import + CUDA OK
  □ python -c "from vllm import LLM; print('ok')"

■ 阶段 B: verl 安装（B 依赖 A）
  □ git clone verl 到 /data/jinzhihong/verl
  □ pip install -e .
  □ verl --help 能跑
  □ 跑 verl 自带 GSM8K toy example (可选, 1 小时)

■ 阶段 C: EVAS 验证基础设施
  □ pip install evas-sim
  □ which openvaf (确认在 PATH)
  □ 跑一次 evas simulate 验证可用
  □ 写一个 micro-test script: 给定 .va + .scs, 返回 pass/fail

■ 阶段 D: Reward function 实现 (依赖 SFT 完成)
  □ 写 train/rewards/format_reward.py (检查 <think>/<answer> 标签)
  □ 写 train/rewards/compile_reward.py (调 OpenVAF, 返回 0/1)
  □ 写 train/rewards/sim_reward.py (调 EVAS, 返回 0/1)
  □ 写 train/rewards/metric_reward.py (对比 gold, 返回 0-1)
  □ 写 train/rewards/vaevas_reward.py (主函数, 聚合)
  □ 单元测试: 给一组手写 completion, 验证 reward 合理

■ 阶段 E: GRPO smoke (依赖 Pilot SFT checkpoint)
  □ 写 train_rl/grpo_config.yaml
  □ 准备 5-10 条 RL prompts (从 SFT 数据 instruction 字段抽)
  □ 跑 GRPO 一个完整 step
  □ 验证 reward 曲线非 NaN
  □ 验证 KL 收敛在合理范围
```

---

## 8. 时间预算

| 阶段 | 预估时间 | 阻塞项 |
|---|---|---|
| A. vLLM 装 | 10-15 min | 无 |
| B. verl 装 | 30 min | A |
| C. EVAS 验证 | 30 min | 无 |
| D. Reward 实现 | 1-2 天 | C, SFT pilot 数据 |
| E. GRPO smoke | 半天 | A, B, D, **SFT pilot checkpoint** |

**阶段 A、B、C 可以现在做**（不依赖数据 pipeline）。**D、E 必须等 SFT Pilot 出 checkpoint 才能跑**。

→ **建议现在做 A、B、C**（约 1.5 小时），把 GRPO 基础设施备好。等数据 pipeline + SFT Pilot 完成时直接进 D、E。

---

## 9. 风险 + 缓解

| 风险 | 概率 | 缓解 |
|---|---|---|
| vLLM 装完 downgrade torch / 破坏 LF | 中 | 装前 `pip freeze > backup.txt`，破了能 rollback |
| verl 版本和 vLLM 不兼容 | 中 | 装稳定 tag (v0.5+) |
| EVAS 在 reward function 里**进程 hang** | 高 | subprocess.run 强制 timeout |
| 多角色模型 OOM | 中 | LoRA actor + ZeRO-2 offload |
| RL prompts 数据**和 SFT training prompts 重叠** | 高 | 单独审计 (类似 SCOPE_BOUNDARY) |
| GRPO 训完 chat 输出乱码 | 中 | KL coef β 调高（0.04 → 0.1）|

最关键的：**第 5 行**。RL prompts 和 SFT training data 都是 prompt-only，**很容易**写重叠的 spec。这次需要明确：

- **SFT prompts** 用于训练**模仿** —— 看到 instruction 就生成 trajectory + code
- **RL prompts** 用于**强化探索** —— 模型自己生成多次，按 reward 排序

→ 如果两套数据 prompt 集合**完全相同**，那 RL 阶段相当于"对 SFT 训练数据再训一遍"，**RL 效果会被严重低估**（因为 SFT 已经背了答案）。

**正确做法**：把数据分成 train_sft / train_rl / eval 三块，**互不重叠**。

---

## 10. take-aways

1. **GRPO 训练栈 = verl + vLLM + reward function**。在 LF env 上补两个 pip install，再写 reward 函数。
2. **vLLM 是 rollout 必备**，必须先装。
3. **verl 装好后跑 GSM8K toy example 验证**，再上 vaEvas 任务。
4. **EVAS 作为 reward verifier 必须 subprocess 调用并 timeout**，不能进程内直接调。
5. **Reward function 是 GRPO 的灵魂**，按 [Circuit-Think 风格的多级 reward](../03_reward_design.md)实现。
6. **数据**：GRPO 只需要 prompt（不需要 gold output）；prompt 集合**必须**和 SFT 训练数据隔开。
7. **阶段 A/B/C 现在就能做**（vLLM + verl + EVAS 安装）；D/E 等 SFT Pilot 结束。

---

## 11. 下一步

读完这一章后：

- **想立刻装环境** → 跑本章 §7 的阶段 A/B/C，~1.5 小时
- **想继续学 reward** → [`02_reward_shapes_behavior.md`](./02_reward_shapes_behavior.md)（reward 设计深度展开）
- **想看 verl 配置** → [`../tools/verl.md`](../tools/verl.md) §5
- **想看 vLLM 内部** → [`../tools/vllm.md`](../tools/vllm.md)
