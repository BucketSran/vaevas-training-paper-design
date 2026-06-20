# train_rl/

GRPO reinforcement learning on top of the SFT checkpoint, with multi-level rewards rooted in EVAS execution.

## Files (to be implemented in Phase 3)

| File | Role |
|---|---|
| `train_grpo.py` | GRPO training loop, wires reward modules from `../rewards/` |
| `config/grpo_default.yaml` | Hyperparameters: group size, clip ε, β (KL), LR, max steps |
| `launch.sh` | Wrapper for remote (2× A100) launch |

## Recipe (mirrors Circuit-Think TGRL — see `../docs/02_grpo_principles.md`)

| Item | Value (default) | Notes |
|---|---|---|
| Base | SFT checkpoint from `../train_sft/` | Required cold start |
| Algorithm | GRPO (Group Relative Policy Optimization) | DeepSeek-R1 style |
| Library | `trl.GRPOTrainer` or `verl` | Use existing infra |
| Group size N | 8 | Same as Circuit-Think |
| Steps | 100-200 | Circuit-Think used 200 |
| Batch size | 16 | Per-device or global, depends on impl |
| Learning rate | 1e-6 | Conservative, since RL pushes from SFT init |
| KL coefficient β | 0.04 (default) | Tune if KL explodes |
| Clip ε | 0.2 (PPO standard) | |
| Reference model | Frozen SFT checkpoint | For KL term |
| Precision | bf16 | |
| Logging | wandb + local logs/ | |

## Reward configuration

See `../rewards/README.md`. Default weights:
```yaml
reward_weights:
  format: 0.1
  compile: 0.2
  simulate: 0.2
  metric: 0.3
  trajectory: 0.2
```

Step gating: τ1 schedule from 0.5 → 0.8 over training, matching Circuit-Think.

Reflective learning: triggered when `R_total < 0.7`, λ_ref = 0.6.

## What to watch during training

| Signal | Healthy | Warning | Stop |
|---|---|---|---|
| `R_total` | trending up | flat for 30+ steps | dropping consistently |
| KL divergence | < 5 nat | 5-20 nat (tune β) | > 20 nat (divergence) |
| Output entropy | > 0.3 floor | < 0.3 (mode collapse) | ~0 (deterministic, broken) |
| Compile rate per checkpoint | rising | flat | dropping |

## Validation before scaling

1. **Tiny run**: 10 steps, 20 samples, group=4 — verify loop converges, rewards are computed correctly.
2. **Reward smoke test**: hand-craft a known-good and known-bad completion; verify reward values match expectations.
3. **Gradient sanity**: log `||grad||` per step; not NaN, not exploding.

## Ablations (required for Phase 3 KPI)

Each as a short (50 step) run:
- `--no-trajectory-reward`
- `--no-reflective`
- `--no-step-gating`
- `--no-r-metric`
- SFT-only baseline (no RL)

## Logging

Every run produces `../logs/grpo_<run-id>/`:
- `config.yaml`
- `rewards.csv` (per-step: R_format, R_compile, ..., R_total, KL, entropy)
- `samples/<step>/` — 8 completions every 10 steps for human review
- `checkpoint/` (gitignored)
- `eval_results.json`
