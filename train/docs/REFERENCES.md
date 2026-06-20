# References — Learning Material

Curated reading list. Star ratings reflect how essential each is for **this project**, not general importance.

## Tier 1 — read these first (★★★)

### Circuit-Think (the direct template)
- **Title**: Circuit-Think: A Multimodal Reasoning Framework for Automated Circuit-to-Netlist Translation with Trajectory-Guided Reinforcement Learning
- **Authors**: Jiang Y., Hu Y., Deng J., Qiu X., Cui Y., He X., Li R., Sun Q., Zhuo C.
- **Venue**: AAAI 2026 (Oral), DOI 10.1609/aaai.v40i7.37465
- **PDF**: `https://ojs.aaai.org/index.php/AAAI/article/download/37465/41427` (when OJS is up)
- **What to extract**: TGRL recipe (SFT → GRPO with multi-level reward), trajectory format, reflective learning mechanism, ablation methodology
- **Key tables**: Table 1 (TGRL vs SFT/DPO/PPO/GRPO), Table 5 (per-step ablation), Table 6 (component ablation)
- **Caveat**: their TGRL ≈ GRPO + reward design + reflective learning; the algorithmic novelty is small

### DeepSeek-R1 (the GRPO origin)
- **Title**: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
- **Authors**: Guo D., Yang D., Zhang H., et al.
- **arXiv**: 2501.12948
- **What to extract**: SFT cold-start → GRPO recipe; reward design for math/code; emergence of reasoning capability
- **Why critical**: this is THE blueprint for SFT-then-GRPO on code-like tasks

### DeepSeekMath (GRPO original)
- **Title**: DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
- **Authors**: Shao Z., et al.
- **arXiv**: 2402.03300
- **What to extract**: GRPO derivation, group-baseline advantage; comparison with PPO
- **Why critical**: the GRPO objective in mathematical detail

## Tier 2 — read when relevant (★★)

### HuggingFace TRL Documentation
- **URL**: https://huggingface.co/docs/trl
- **What to extract**: `SFTTrainer`, `GRPOTrainer` API; reward function signature
- **When**: before writing `train_sft/train.py` and `train_rl/train_grpo.py`

### Qwen2.5-Coder Technical Report
- **arXiv**: 2502.13923 (Qwen2.5-VL; Coder report similar)
- **What to extract**: chat template, tokenizer, special tokens, supported languages
- **When**: when configuring base model and special tokens

### AMSBench (related benchmark)
- **arXiv**: 2505.24138
- **What to extract**: AMS circuit evaluation methodology, cross-source generalization concerns
- **Why useful**: helps avoid the over-fitting pitfalls Circuit-Think glossed over

### LLM-for-EDA Survey
- **arXiv**: 2508.20030
- **What to extract**: landscape; what others have tried for hardware code generation
- **When**: when motivating the project; not needed for implementation

## Tier 3 — supporting / background (★)

### PPO original
- **Schulman et al., "Proximal Policy Optimization Algorithms"** (arXiv 1707.06347)
- For understanding the PPO objective that GRPO simplifies

### InstructGPT
- **Ouyang et al., 2022** (arXiv 2203.02155)
- The original SFT-then-RLHF recipe

### Process Reward Models
- **PRM-800K**: Lightman et al., "Let's Verify Step by Step" (arXiv 2305.20050)
- For background on step-wise reward; not strictly used here (we use task-grounded verifiers)

### Auto-SPICE (related task)
- **arXiv**: 2411 (Bhandari et al. 2024)
- LLM for SPICE netlist extraction from images — earlier work in Circuit-Think's space

### AMSnet (dataset reference)
- **Tao et al. 2024**
- Image-netlist dataset used by Circuit-Think

## Tier 4 — code references (∗)

### Repositories to study (paper-companion code)
- **deepseek-math**: official paper artifacts (GRPO reference implementation)
- **CircuitThink dataset**: https://github.com/7jiangyq/CircuitThink (only 100/3100 items released, but trajectory format is canonical reference)
- **RTL-Coder**: https://github.com/hkust-zhiyao/RTL-Coder — **the real-world SFT pipeline we study line-by-line**; see [`docs/sft/02b_real_world_reference.md`](./sft/02b_real_world_reference.md)

### Existing project artifacts (in this repo)
- `EVAS/README.md` — simulator interface
- `EVAS/CLAUDE.md` — agent contract for EVAS
- `veriloga-skills/` — domain skills, reference templates
- `behavioral-veriloga-eval/docs/VAEVAS_VALIDATION_PIPELINE.md` — validation pipeline (do NOT mine for training data without audit)

---

## 框架选型景观 (Framework Landscape)

**这部分是工具栈的决策记录**，不是阅读材料。决定我们在每个阶段实际用什么库。

> 🔍 **想深入了解某个工具？** 进 [`tools/`](./tools/) 子目录看深度学习笔记。当前已有 [`tools/vllm.md`](./tools/vllm.md) 和 [`tools/verl.md`](./tools/verl.md)。

### 路线图全景

```
       ┌─────────────────────────────────┐
       │ SFT 阶段                         │   学：RTL-Coder mle.py
       │   ├ trl.SFTTrainer (HF 官方) ⭐  │   生产候选：trl 或 LLaMA-Factory
       │   ├ LLaMA-Factory                │
       │   └ Axolotl                      │
       └─────────────────────────────────┘
                       ↓ SFT checkpoint
       ┌─────────────────────────────────┐
       │ GRPO / RL 阶段                   │   学：trl.GRPOTrainer
       │   ├ trl.GRPOTrainer              │   生产候选：verl (主流)
       │   ├ verl ⭐⭐ (字节)              │
       │   ├ slime (THUDM, agentic)       │
       │   └ OpenRLHF                     │
       └─────────────────────────────────┘
                       ↓ 需要快速 rollout 生成 N 个 completion
       ┌─────────────────────────────────┐
       │ 推理框架 (RL 训练时和 eval 都用)  │
       │   ├ vLLM ⭐⭐ (业界标准)          │
       │   └ SGLang (约束解码)            │
       └─────────────────────────────────┘
```

### 按用途分组的对比

#### SFT 训练框架

| 框架 | 出品方 | 出现时间 | 杀手特性 | 我们何时用 |
|---|---|---|---|---|
| `trl.SFTTrainer` | HuggingFace | 2023+ | 与 `transformers` 紧密集成，文档全 | **学习 + 小规模实验** |
| **LLaMA-Factory** | hiyouga（北航） | 2023+ | YAML 一站式，国内最易用 | 生产 SFT（候选） |
| Axolotl | Open Access AI Collective | 2023+ | 配置灵活，西方社区版的 LLaMA-Factory | 可选替代 |

#### RL 训练框架

| 框架 | 出品方 | 出现时间 | 杀手特性 | 我们何时用 |
|---|---|---|---|---|
| `trl.GRPOTrainer` | HuggingFace | 2025 | 标准 GRPO 实现，文档清晰 | **GRPO 概念学习 + 小规模实验** |
| **verl** | 字节跳动 | 2024-2025 | **rollout 用 vLLM**，训练-推理混部，可 scale 到几百卡 | **生产 GRPO（首选）** |
| slime | 智谱 / THUDM | 2025 | Agentic RL，把 tool-calling 当一等公民 | 多轮 agent RL（未来） |
| OpenRLHF | OpenLLMAI | 2024 | 生产级，多节点 | verl 的备选 |

#### 推理框架

| 框架 | 出品方 | 出现时间 | 杀手特性 | 我们何时用 |
|---|---|---|---|---|
| **vLLM** | UC Berkeley Sky Lab | 2023-2024 | PagedAttention + continuous batching，比 HF generate 快 10-24× | **必学，部署 / RL rollout / eval 都用** |
| SGLang | LMSYS（vLLM 团队 fork）| 2024 | constrained decoding，强制输出符合 JSON / 正则 / EBNF | 推理时强制结构化输出（可选） |

### 我们的选型决策（vaEvas SFT/GRPO 项目）

| 阶段 | 我们选什么 | 备选 | 决策理由 |
|---|---|---|---|
| Phase 2 SFT **学习** | RTL-Coder `mle.py` 逐行读 | - | 透明、无封装、能看到每一步 |
| Phase 2 SFT **生产** | `trl.SFTTrainer` 起步；卡了切 LLaMA-Factory | LLaMA-Factory 直接起 | trl 控制力强、Bug 时好调；LF 配置便利 |
| Phase 3 GRPO **学习** | `trl.GRPOTrainer` | - | 标准实现，与 SFT 一脉相承 |
| Phase 3 GRPO **生产** | **verl** | OpenRLHF | rollout 必须用 vLLM（不然慢 5-10×），verl 已集成 |
| Phase 4 **eval** | **vLLM** | HF `generate` | 10× 快，跑 50-100 条 eval 节省几小时 |
| 未来：EVAS 多轮闭环 agent | slime 或 verl agentic 模式 | - | 多轮 tool-calling 才需要 |

### 学习路径推荐

**现在（SFT 学习阶段）**：
1. 读 RTL-Coder upstream `train/mle.py`
2. 扫一眼 [LLaMA-Factory 中文文档](https://llamafactory.readthedocs.io/zh-cn/latest/) 的 SFT 例子（30 分钟感受 YAML 风格）
3. 知道 [`trl.SFTTrainer`](https://huggingface.co/docs/trl/sft_trainer) 在做什么

**GRPO 开始之前**：

4. **学 vLLM 基础**（必须）：知道 `LLM(model=...)` / `SamplingParams` / `llm.generate(prompts, sampling_params)` 三个 API。这是 RL rollout 的核心。
5. 看一遍 [verl 的 GRPO example](https://verl.org.cn/en/latest/index.html)，不一定要跑通，但要知道**它怎么把"训练 step + vLLM rollout"串起来**

**长期备用**：

6. **slime** 收藏；只在我们考虑"EVAS 作为工具被模型反复调用"的 agent 闭环时再深入
7. **SGLang** 收藏；如果发现 `R_format` reward 浪费太多 RL 训练步，考虑推理时用 SGLang 约束输出（训练时不用）

### 关键 URLs（书签存这里）

| 框架 | 官方文档 | GitHub |
|---|---|---|
| trl | https://huggingface.co/docs/trl | github.com/huggingface/trl |
| LLaMA-Factory | https://llamafactory.readthedocs.io/zh-cn/latest/ | github.com/hiyouga/LLaMA-Factory |
| Axolotl | https://axolotl.ai/ | github.com/axolotl-ai-cloud/axolotl |
| verl | https://verl.org.cn/en/latest/index.html | github.com/volcengine/verl |
| slime | https://thudm.github.io/slime/zh/index.html | github.com/THUDM/slime |
| OpenRLHF | https://openrlhf.readthedocs.io/ | github.com/OpenLLMAI/OpenRLHF |
| vLLM | https://docs.vllm.com.cn/en/latest/ | github.com/vllm-project/vllm |
| SGLang | https://docs.sglang.com.cn/ | github.com/sgl-project/sglang |

### 这些选型什么时候应该被推翻

- **trl → LLaMA-Factory**：如果我们在 Phase 2 写 `train_sft/train.py` 时发现要配置的样板代码超过 200 行，切 LLaMA-Factory（YAML 就能搞定）。
- **verl → OpenRLHF / trl**：如果 verl 文档不足以让我们独立跑通 vaEvas reward function 集成，回退 trl + 自己用 vLLM 做 rollout（更脆弱但可控）。
- **vLLM → HF generate**：如果某种 reward function 严重依赖 HF 输出格式或 hidden states，eval 时回退 HF。**RL rollout 绝不回退** —— 那是性能必需的。

任何上述推翻发生时，在 [`docs/adr/`](../../docs/adr/) 里写一个 ADR 记录。

---

## How to use this list

- **Before Phase 1**: read Tier 1 fully. Take notes. Decide your trajectory format.
- **During Phase 2**: read TRL docs, study `trl.SFTTrainer` source.
- **Before Phase 3**: re-read DeepSeekMath sections 4-5. Study `trl.GRPOTrainer`.
- **Before Phase 4**: read AMSBench section on cross-source eval.

## Add new entries here when

- A paper directly informs a design decision in `train/`.
- A library decision is made.
- A blog post / talk gives a clean explanation worth re-reading.

Do NOT add:
- Wide-ranging surveys you'll never re-open.
- Papers you read but didn't apply.
- "Inspiration" without traceable design impact.
