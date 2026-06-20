# Server Platform Probe

Status: next GitHub-mediated remote task after the handoff smoke, 2026-06-20.

This probe checks whether the remote machine is ready for SFT/GRPO smoke runs.
It is deliberately non-destructive: no training, no checkpoint writes, no
package installs, and no driver changes.

## Why This Exists

The first server handoff smoke proved that GitHub exchange and the toy Phase 1
data pipeline work on the remote host. It also showed a likely GPU blocker:
`nvidia-smi` could not communicate with the NVIDIA driver.

Before any SFT or GRPO run, we need a precise platform status:

- is CUDA visible to PyTorch?
- can PyTorch allocate a CUDA tensor?
- are the SFT packages importable?
- are the GRPO rollout packages importable?
- are the historical 7B model paths still present?

## Command

On the remote server:

```bash
git fetch origin
git checkout training-paper-design-20260620
git pull --ff-only
bash train/infra/run_server_platform_probe.sh \
  --job train/infra/jobs/phase1_server_platform_probe.yaml
```

If the intended Python environment is known, pass it explicitly:

```bash
bash train/infra/run_server_platform_probe.sh \
  --job train/infra/jobs/phase1_server_platform_probe.yaml \
  --python /data/jinzhihong/envs/vaevas-rl/bin/python
```

## Result Status

| Status | Meaning | Next action |
| --- | --- | --- |
| `READY` | GPU, SFT stack, and GRPO stack are visible. | Prepare tiny SFT platform smoke. |
| `READY_SFT_ONLY` | GPU and SFT stack are visible, but GRPO packages are incomplete. | Run SFT smoke first; fix GRPO later. |
| `BLOCKED_GPU` | Python stack may exist, but CUDA/GPU is not usable by PyTorch. | Do not train; report GPU/driver state. |
| `BLOCKED_IMPORTS` | Required core Python packages are missing. | Do not train; report missing packages. |

Missing `flash_attn` is only a warning at this stage.

For `BLOCKED_GPU`, follow `SERVER_GPU_BLOCKER_TRIAGE.md` and ask remote Codex
to execute `train/infra/jobs/REMOTE_CODEX_TASK_GPU_BLOCKER.md`.

## Upload Result Back

After running the probe:

```bash
git status --short
git add train/infra/results/<run_id>
git commit -m "Upload server platform probe result <run_id>"
git push origin training-paper-design-20260620
```

Upload the result directory even if the status is blocked.

## Expected Result Files

```text
train/infra/results/<run_id>/
├── commands.log
├── environment.txt
├── gpu_probe.json
├── model_paths.json
├── python_packages.json
├── status.txt
└── summary.json
```

The local agent should read `summary.json` first. `commands.log` is only for
debugging if the script itself behaved unexpectedly.

## Forbidden Actions

Do not do any of the following during this probe:

- install or upgrade packages;
- change NVIDIA drivers;
- start SFT, GRPO, vLLM serving, or Ray clusters;
- write checkpoints or adapters;
- upload secrets, tokens, `.env` files, model weights, or full datasets.
