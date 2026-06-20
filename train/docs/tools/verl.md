# verl — LLM 强化学习训练框架

> **预览**：VSCode `Cmd+Shift+V`，KaTeX 渲染。
>
> **前置阅读**：[`vllm.md`](./vllm.md) 必读。verl 把 vLLM 当 rollout 引擎，不懂 vLLM 看不懂 verl。

> 读完这一章你应该能回答：
> 1. verl 是什么，为什么需要它（trl 不够吗）？
> 2. RLHF 训练里有哪几个角色模型？它们怎么协作？
> 3. **Train-Inference Colocation** 是什么？为什么是 verl 的核心？
> 4. verl YAML 配置长什么样，关键字段是哪几个？
> 5. reward function 怎么写？
> 6. 在 vaEvas 项目里，verl 怎么进 Phase 3 流水线？

## 1. verl 是什么 + 解决什么问题

**verl** = "**V**olcano **E**ngine **R**einforcement **L**earning"，字节跳动出品的**开源 LLM 强化学习训练框架**。

它解决的是**工程复杂度**问题，不是算法问题。算法上 verl 用的还是 PPO / GRPO / DAPO 这些已有算法。verl 的价值在于把 RL 训练里**几个最痛苦的工程问题**封装好：

| 痛点 | verl 怎么处理 |
|---|---|
| 多模型同时在显存（policy + ref + reward + critic）→ 显存爆炸 | FSDP + offload + colocation |
| Rollout 慢 → 训练时间几小时到一天 | 集成 vLLM 做 rollout |
| 训练-推理代码栈分离 → 调试痛苦 | 统一抽象，权重同步自动 |
| 配置复杂 | YAML 化 |
| 多节点 scale | Ray 一等公民 |

**写 RL 训练代码的人都踩过这些坑**。verl 不是新算法，是新工程实现。

## 2. RLHF 训练里有几个角色模型

做 GRPO 训练时，同时要持有 **3–4 个模型**在显存里：

| 角色 | 在干什么 | 参数会更新吗？ | 显存占用（7B bf16）|
|---|---|---|---|
| **Policy (Actor)** | 主模型，被训练的对象 | ✅ 是 | 14 GB（参数） + 56 GB（Adam state，全参） + 14 GB（梯度） = **~84 GB** |
| **Reference (Ref)** | 冻结的 SFT 模型，用来算 KL 项 | ❌ 否 | 14 GB（只要参数） |
| **Rollout Engine** | 推理引擎（vLLM），生成 completions | ❌ 否（权重从 Policy 同步） | 14 GB + KV cache |
| **Reward Model** | 评分模型（如果用 model-based reward）| ❌ 否 | 视模型大小 |
| **Critic** | value function 估计（PPO 才用，**GRPO 不用** ✓）| ✅ 是 | ~84 GB |

GRPO 至少 3 个角色（Policy / Ref / Rollout）。PPO 多一个 Critic 是 4 个。

### 显存预算冲击

光 Policy 一个就要 84 GB。Reference 再来 14 GB。Rollout 再来 14 GB + KV cache。**总共超过 110 GB**。

我们的 2×A100 80G = 160 GB 总显存。**正好够，但每一项都得抠**。这就是为什么 verl 提供了一整套显存优化机制（offload / colocation / ZeRO）。

## 3. Train-Inference Colocation — verl 的灵魂

### 3.1 朴素方案：训练推理分卡

一组 GPU 训练 policy，另一组 GPU 跑 vLLM rollout：

```
GPU 0,1: 训练 (policy + ref)     ← rollout 时空闲
GPU 2,3: vLLM rollout            ← 训练时空闲
```

**优点**：简单，逻辑清晰。
**缺点**：GPU 利用率低 —— 训练时 rollout GPU 闲，rollout 时训练 GPU 闲。

### 3.2 verl 方案：同卡复用（colocation）

**同一组 GPU 在不同时间复用**：

```
时间步 t (训练):
  GPU 0,1: policy（满载） + ref（offload 到 CPU）

时间步 t.rollout:
  GPU 0,1: 训练态卸载 → vLLM 加载新权重 → 满载 rollout

时间步 t+1 (训练):
  GPU 0,1: vLLM 卸载 → 训练态加载 → 满载训 policy
```

每个 GPU 时间片**轮流**做训练和推理。**这就是 colocation**。

### 3.3 Colocation 难在哪

不是"切换一下"那么简单。挑战：

1. **权重同步**：训练完一步，新 policy 参数必须**快速**同步给 vLLM 实例（否则 rollout 用的是旧策略 = 数据浪费）。verl 用 NCCL 直接 GPU-GPU 传，几百毫秒内完成。
2. **优化器状态保留**：训练态切到 rollout 时，Adam 状态不能丢；得用 ZeRO offload 到 CPU 或别的 GPU。
3. **显存峰值管理**：训练态 + rollout 态不能同时全活。要精细调度。

verl 把这些都封装在 **Hybrid Engine** 抽象里，你写 YAML 配置就行。

## 4. 与 trl.GRPOTrainer 的对比

我们前期学习用 trl，生产训练切 verl。两者侧重不同：

| 维度 | `trl.GRPOTrainer` | **verl** |
|---|---|---|
| Rollout 引擎 | HF `generate()`（慢）| **vLLM**（快 10×） |
| 多 GPU | `accelerate` / DeepSpeed | 原生 Ray + Hybrid Engine |
| 多节点 | 麻烦 | **一等公民** |
| 配置方式 | Python 代码 | YAML / Hydra |
| 学习曲线 | 平缓 | 陡峭 |
| 适合规模 | 单节点 7B–13B | 单节点 → 多节点 70B+ |
| 文档成熟度 | HF 标准（稳定）| 快速演进（可能 break）|
| Reward 函数集成 | `reward_funcs=[fn]` 列表 | `compute_reward()` |
| 调试 | 直接（单进程）| 难（multi-process）|

**我们的策略**：
- **学概念**用 trl.GRPOTrainer（透明、单进程、容易跟踪）
- **生产训练**用 verl（快、可 scale）

## 5. verl 的 YAML 配置长什么样

verl 一个 GRPO experiment 的核心配置（精简版，真实文件更长）：

```yaml
# grpo_config.yaml
algorithm:
  algorithm: grpo                       # 算法选 grpo（也可 ppo / dapo）
  use_kl_in_reward: false               # KL 不进 reward, 进 loss

actor_rollout_ref:
  model:
    path: /path/to/sft/checkpoint       # ★ SFT 训完的 checkpoint
    tokenizer_path: same as model.path

  actor:
    optim:
      lr: 1e-6                          # ★ GRPO 标准学习率
      lr_scheduler: cosine
      warmup_steps: 10
    ppo_mini_batch_size: 16             # 每个 mini-batch 16 个 sample
    ppo_micro_batch_size: 4             # 每张卡每次实际处理 4 个
    use_kl_loss: true                   # ★ 开 KL 项
    kl_loss_coef: 0.04                  # ★ KL 系数 β
    clip_ratio: 0.2                     # PPO clip ε

  rollout:
    name: vllm                          # ★★ 用 vLLM
    n: 8                                # ★★ group size, Circuit-Think 用 8
    temperature: 1.0
    top_p: 0.95
    max_new_tokens: 2048
    gpu_memory_utilization: 0.5         # rollout 用一半显存（训练要留另一半）

  ref:
    fsdp_config:
      param_offload: true               # ★ ref model 参数 offload 到 CPU
      grad_offload: true                # ref 不算梯度，offload 不影响

data:
  train_files: /path/to/train.parquet   # RL 训练 prompts（不需要 completions）
  val_files: /path/to/val.parquet
  prompt_key: prompt
  max_prompt_length: 1024
  max_response_length: 2048

reward_model:
  reward_manager: naive                 # 用我们自己写的 reward function

trainer:
  total_epochs: 1
  total_training_steps: 200             # ★ Circuit-Think 用 200
  save_freq: 50
  test_freq: 20
  project_name: vaevas-grpo
  experiment_name: run-001
  logger: ["wandb"]
  n_gpus_per_node: 2                    # ★ 2× A100
  nnodes: 1
```

打 ★ 的是**关键字段**，必须懂。打 ★★ 的是核心决策。

### 配置爆炸：注意

verl 实际的 config 比上面长 5 倍。完整的 [`ppo_trainer.yaml` 模板](https://github.com/volcengine/verl/blob/main/verl/trainer/config/ppo_trainer.yaml) 有 200+ 行。大部分有合理 default，**只需要改上面这些**就行。

## 6. 怎么写 reward function

verl 的 reward function 接受**一个 batch 的 prompts 和 completions**，返回每个 completion 的标量奖励：

```python
# vaevas_reward.py
def compute_reward(prompts, completions, **kwargs):
    """
    prompts:     List[str], 长度 B*N (B 个 prompt × N 个 completion)
    completions: List[str], 长度 B*N

    Returns: List[float], 每个 completion 的总 reward
    """
    rewards = []
    for prompt, comp in zip(prompts, completions):
        va_code = parse_answer(comp)  # 从 <answer>...</answer> 抠出代码

        # 多级 reward (映射到我们的 docs/03_reward_design.md)
        r_format    = check_format(comp)                 # 0 or 1
        r_compile   = openvaf_compile(va_code)           # 0 or 1
        r_simulate  = evas_simulate(va_code, tb=None)    # 0 or 1
        r_metric    = compare_with_gold(va_code, prompt) # 0~1

        # gating: 编译不过, 后面不用算
        if r_compile == 0:
            r_simulate = 0
            r_metric = 0

        # 加权求和
        total = (
            0.1 * r_format +
            0.2 * r_compile +
            0.2 * r_simulate +
            0.5 * r_metric
        )
        rewards.append(total)

    return rewards
```

→ 这就是我们 **Phase 3 的核心编程任务**。reward function 写得好不好，决定 GRPO 训出来效果。

verl 怎么调用这个 function？config 里指：

```yaml
reward_model:
  reward_manager: naive
  custom_reward_function:
    path: /path/to/vaevas_reward.py
    name: compute_reward
```

## 7. 在 vaEvas 项目中的具体使用

Phase 3 GRPO 训练的完整 pipeline：

```
Phase 2 结束
    ↓
SFT checkpoint at  models/sft-run-N/
    ↓
Phase 3 开始
    ↓
1. 写 train/rewards/vaevas_reward.py
   - format / compile / simulate / metric 各一个函数
   - 主函数 compute_reward(prompts, completions)
   - 集成 EVAS, OpenVAF

2. 写 train/train_rl/grpo_config.yaml
   - 基于 verl 模板
   - 填 SFT checkpoint 路径 / RL prompts 路径 / reward function 路径

3. 准备 RL prompts: train/data/rl/prompts.jsonl
   - 每行 {"prompt": "..."}（不需要 completion, RL 自己采）

4. 启动训练:
   bash train/train_rl/launch.sh
   # 内部: python -m verl.trainer.main_ppo --config grpo_config.yaml
    ↓
verl 内部干的事:
   ① 加载 SFT model 作 actor
   ② 复制一份冻结作 reference
   ③ 启动 vLLM rollout engine, 同步初始权重
   ④ 进入训练循环:
      for step in range(200):
          - vLLM 生成 N=8 completions / prompt
          - 调 compute_reward() 算分
          - 算 advantage (group-normalized)
          - PPO clip + KL 项更新 actor
          - 同步新权重到 vLLM
          - log reward / KL / entropy 到 wandb
    ↓
Phase 3 结束
    ↓
GRPO checkpoint at models/grpo-run-N/
```

## 8. 常见坑

| 现象 | 原因 | 解决 |
|---|---|---|
| OOM 启动 | 多模型同时挤显存 | 调小 `rollout.gpu_memory_utilization`（0.5 → 0.4）；开 `ref.param_offload: true` |
| Rollout 慢 | vLLM 没生效 | 检查 `rollout.name: vllm`；看日志是否显示 "Using vLLM" |
| Reward 不动 | reward function bug | **单独跑 reward function** 给几个手写 completion 看返回值是不是合理 |
| 训练 NaN | LR 太高 / KL 太小 | LR 1e-6 起步；β 0.04 起步；不要乱调 |
| KL 暴涨 | 模型偏离 ref 太远 | 加大 KL coef β；降 LR |
| Entropy 崩塌 | 模型收敛到单一输出（reward hacking）| 调试 reward function 找 shortcut；用 sample inspection |
| Multi-node 起不来 | Ray 启动失败 | 看 verl Ray 部署文档；先单节点跑通再上多节点 |
| Checkpoint 不能加载 | verl 的 checkpoint 格式特殊（HF 不直接兼容）| 用 verl 提供的 [convert script](https://github.com/volcengine/verl/blob/main/scripts/model_merger.py) 转回 HF 格式 |
| 训练完不知道哪个 checkpoint 最好 | 没设 eval | `test_freq: 20` 让 verl 每 20 步在 val set 上跑一次 |

## 9. 学习路径

### 现在（SFT 学习阶段，可以先跳过 verl）

verl 不是现在必须学的。但建议**扫一眼**：
- **30 分钟** — 读 [verl 官方 GRPO 教程](https://verl.org.cn/en/latest/algo/grpo.html) 的概念部分
- **30 分钟** — 翻一下 verl GitHub 上的 [example config](https://github.com/volcengine/verl/tree/main/examples/grpo_trainer)

### Phase 3 GRPO 开始之前（必须完成）

1. **2 小时** — 跑通 verl 官方的 [GSM8K GRPO toy example](https://verl.org.cn/en/latest/start/quickstart.html)
2. **1 小时** — 看 verl 一个真实的 reward function（比如 `examples/reward_func/`）
3. **2 小时** — 看一遍 `verl/trainer/ppo/ray_trainer.py` 源码（不用全懂，知道架构即可）

### Phase 3 GRPO 训练时（边做边学）

4. 写 vaEvas reward function（参考本文 §6）
5. 写 vaEvas YAML config（参考本文 §5）
6. 第一次跑训练，debug 各种坑（参考本文 §8）

### 备用知识（如果遇到具体问题）

- 多卡 / 多节点：[verl 部署指南](https://verl.org.cn/en/latest/start/install.html)
- Hybrid Engine 内部：[verl architecture doc](https://verl.org.cn/en/latest/design/architecture.html)
- Reward 调试技巧：verl GitHub issues 里搜 "reward debugging"

## 10. 本章 take-aways

1. **verl 是字节出品的开源 LLM RL 训练框架**，专门为 PPO/GRPO/DAPO 这类后训练算法的**工程实现**而生。
2. **RLHF 至少 3 个角色模型**同时活在显存里：Policy（训练）+ Reference（KL）+ Rollout（生成）。GRPO 不用 Critic，省一个。
3. **Train-Inference Colocation** 是 verl 的灵魂 —— 同 GPU 时间复用，不浪费算力。代价是工程复杂。
4. **配置是 YAML**，关键字段就 10 个左右（lr / KL / group size / vLLM 配置 / SFT checkpoint 路径）。
5. **核心编程任务是写 reward function** —— 这一段决定了 GRPO 训得好不好。
6. 我们 Phase 3 GRPO 训练用 verl，前期学概念用 trl.GRPOTrainer。

## 11. 下一步

读完本章后：
- 回 [`vllm.md`](./vllm.md) 强化对 rollout 引擎的理解
- 进 [`../sft/02b_real_world_reference.md`](../sft/02b_real_world_reference.md) 继续 SFT 学习（路径 B 的当前位置）
- 看 [`../03_reward_design.md`](../03_reward_design.md) → reward 设计是 verl 编程的核心
- 看 [`../REFERENCES.md`](../REFERENCES.md) 的"框架选型景观"章节，确认我们的选型决策
