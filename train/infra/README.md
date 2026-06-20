# infra/

Remote server / cluster handoff scripts. Current mode is GitHub-mediated:
the server pulls this branch, runs a small smoke job, then pushes small result
artifacts back to GitHub for local inspection.

## Current Files

| File | Role |
|---|---|
| `run_github_handoff_smoke.sh` | Server-side smoke: environment snapshot + Phase 1 manifest/admission/SFT/GRPO pack loop |
| `run_server_platform_probe.sh` | Server-side platform probe: CUDA/PyTorch/package/model-path readiness without training |
| `run_gpu_blocker_diagnosis.sh` | Server-side GPU blocker triage after `BLOCKED_GPU` without admin changes |
| `run_tiny_sft_smoke.sh` | Server-side two-step LoRA SFT smoke on toy admitted data; commits evidence only |
| `jobs/` | YAML job manifests that describe what the server should run |
| `results/` | Small result upload area; no checkpoints or raw training data |

## Later SSH/Phase 2 Files

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
- Current data flow: GitHub branch -> server smoke -> GitHub result commit.
- Later data flow: synthesize locally OR on remote; train on remote; pull logs back.

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

Full SFT/GRPO launch scripts remain deferred until Phase 2. The current scripts
only validate handoff, platform readiness, environment visibility, toy Phase 1
data plumbing, and tiny toy-data SFT training-loop health.
