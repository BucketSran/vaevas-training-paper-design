# infra/

Remote server / cluster scripts. Two A100 GPUs configured for SFT and GRPO.

## Files (to be implemented in Phase 2)

| File | Role |
|---|---|
| `setup_remote.sh` | One-time: clone repo, install deps, pre-download base model |
| `sync.sh` | rsync code and data between local and remote |
| `launch_sft.sh` | Launch SFT on remote, attach via `tmux`/`screen` |
| `launch_grpo.sh` | Launch GRPO on remote |
| `monitor.sh` | tail loss/reward logs, GPU utilization |
| `pull_logs.sh` | Pull `logs/<run-id>/` back to local for review |
| `job_templates/` | Templates for SLURM / nohup / docker if needed |

## Topology assumption

- Remote: single node, 2× NVIDIA A100 (80GB)
- Local: M-series Mac for code + monitoring
- Data flow: synthesize locally OR on remote (Phase 1 decision); train on remote; pull logs back

## Environment

| Item | Pinned at |
|---|---|
| CUDA | 12.4 |
| PyTorch | 2.4+ (cu124) |
| transformers | 4.45+ |
| trl | 0.11+ (GRPO support) |
| accelerate / deepspeed | latest stable |
| flash-attn | 2.6+ |

Pinned versions go into `requirements.txt` once Phase 2 starts.

## Security / hygiene rules

- No long-lived SSH keys checked into the repo.
- Remote `.env` for HF tokens, wandb keys (not committed).
- All `rsync` operations use `--dry-run` first when going FROM remote TO local (to prevent accidental overwrites of local work).

## Cost / time estimates

| Task | Est. wall time | GPU-hours |
|---|---|---|
| Base model download | 30 min | 0 |
| Full SFT 2 epochs on 2500 samples | 4-8 hours | 8-16 |
| GRPO 200 steps | 8-12 hours | 16-24 |
| Full eval on 50-100 held-out | 20-40 min | <1 |

## Phase 0 placeholders

No scripts exist yet. This README defines the contract; implementation lands in Phase 2.
