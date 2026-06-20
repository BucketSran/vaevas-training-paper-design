# 02 — GRPO Principles

What Group Relative Policy Optimization does, why it's the right RL flavor for this task, what to watch.

## The 90-second summary

GRPO (DeepSeek-R1, 2025) is **PPO without a value (critic) network**. For each prompt, you sample N completions, compute reward per completion, and use the **group's mean and std as an implicit baseline** for advantage estimation.

```
For each prompt x:
    Sample N completions {o_1, ..., o_N} from current policy
    Compute reward R_i for each completion
    Compute group-normalized advantage:
        Â_i = (R_i - mean({R})) / std({R})
    Apply PPO clipped objective:
        L_i = min(
            (π_θ(o_i|x) / π_old(o_i|x)) · Â_i,
            clip((π_θ / π_old), 1-ε, 1+ε) · Â_i
        )
    Total objective: (1/N) Σ L_i  - β · KL(π_θ || π_ref)
    Update θ via gradient ascent.
```

That's the whole thing. No critic, no value head, no GAE. Cheaper and stabler than PPO on language tasks.

## Why GRPO, not PPO or DPO

| Method | Why we don't use it here |
|---|---|
| **PPO** | Requires a value network → 2x model size in memory, harder to train, less stable on long sequences |
| **DPO** | Needs preference pairs `(chosen, rejected)` — we don't have humans labeling Verilog-A pairs at scale |
| **REINFORCE** | High variance without a baseline; group baseline solves this without a critic |
| **RLOO** | Sibling of GRPO, leaves-one-out baseline; very similar performance, GRPO has stronger library support in `trl` |

**Default for `train/`**: GRPO via `trl.GRPOTrainer`. Switch to RLOO if GRPO instability appears.

## Hyperparameters that actually matter

| Param | What it controls | Default | When to change |
|---|---|---|---|
| Group size N | Variance of advantage; quality of group baseline | 8 | larger N (16) if A100 memory allows; helps stabilize sparse-reward tasks |
| Steps | Total optimization steps | 100-200 | Circuit-Think used 200; tune up if reward still climbing |
| Learning rate | Standard | 1e-6 | very low because SFT init is already good |
| KL coefficient β | Pulls model toward reference (SFT) | 0.04 | raise if KL explodes; lower if model stops exploring |
| Clip ε | PPO clip range | 0.2 | rarely needs tuning |
| Reference model | Where the KL is anchored | Frozen SFT checkpoint | not the base; the SFT checkpoint |
| Temperature / top-p | Sampling for the N completions | T=0.7-1.0, top-p=0.95 | higher T = more exploration |

## The reward design constraint

GRPO's quality is **bounded by the reward function**. The algorithm itself does NOT solve reward design; it just exploits whatever reward you give it.

**This is the most important sentence in this doc**: 80% of the work in adapting Circuit-Think to vaBench is in `rewards/`, not in `train_rl/train_grpo.py`.

See `03_reward_design.md`.

## What "group-relative" buys you

The group baseline `mean({R_i})` cancels two annoying things:
1. **Per-prompt reward scale**: a hard prompt where everyone scores 0.2 vs an easy one where everyone scores 0.9 are both normalized to advantage ~0 if everyone scores the same. The model only learns from variation *within* a group.
2. **Reward function shifts**: if you tweak reward weights, the group baseline absorbs the offset; only the relative ordering matters.

The std normalization handles scale:
- For prompts where group rewards cluster tightly, advantages are sharp (small differences amplified).
- For prompts with huge reward variance, advantages are dampened (one big-reward sample doesn't dominate).

## Failure modes to watch

| Symptom | Cause | Fix |
|---|---|---|
| `R_total` flat from step 1 | Group reward variance = 0 (everyone fails or everyone passes) | Adjust prompt difficulty distribution; group size; SFT not strong enough |
| KL > 20 nat and rising | Model running away from SFT init | raise β, lower LR, or stop and revert |
| Output entropy collapsing | Mode collapse — model finds one "safe" completion | raise temperature, lower β so model can explore |
| Reward hacking | Model finds a trivial way to earn reward | inspect samples, fix reward function (NOT raise β as a workaround) |
| `R_compile` rising but `R_correct` flat | Model produces compileable garbage | check reward weights; metric reward may be under-incentivized |
| Catastrophic forgetting of unrelated abilities | β too low, model drifts | raise β |

## Anatomy of a healthy run

Per Circuit-Think (Figure 5):
- Format reward: 0.85 → 1.00 within ~40 steps (easy signal)
- Stepwise reasoning reward: 0.4 → 0.8 over 200 steps
- Netlist (answer) reward: 0.2 → 0.75 over 200 steps (slowest climber)
- Graph consistency: 0.4 → 0.85 over 200 steps

What we'd hope to see for vaBench:
- Format reward: same fast-climb to ~1.0
- Compile reward: 0.5 → 0.9 over 100 steps
- Simulate reward: 0.3 → 0.7 over 200 steps
- Metric reward: 0.1 → 0.5 over 200 steps (hardest)
- Trajectory reward: 0.4 → 0.8 over 200 steps

If any reward plateaus low after 100 steps, that reward is poorly designed OR the model lacks capability — investigate before scaling.

## Library choice

| Lib | Pros | Cons |
|---|---|---|
| **`trl.GRPOTrainer`** | HuggingFace standard, well-maintained, integrates with `accelerate` / `deepspeed` | API can shift between minor versions; pin tightly |
| **`verl`** | Newer, more optimized for RL workloads, better separation of policy/reward | Less battle-tested in community |
| **OpenRLHF** | Production-grade, multi-node | Heavier setup |

**Default**: `trl.GRPOTrainer` for Phase 3. Revisit if scale demands.

## Key references

See `REFERENCES.md` section "GRPO" for:
- DeepSeek-R1 paper (Guo et al. 2025)
- DeepSeekMath GRPO original paper (Shao et al. 2024)
- HuggingFace TRL docs
- Circuit-Think paper (Jiang et al. 2026) — concrete example
