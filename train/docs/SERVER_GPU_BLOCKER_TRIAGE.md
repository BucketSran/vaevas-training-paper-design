# Server GPU Blocker Triage

Status: follow-up to the `BLOCKED_GPU` platform probe result, 2026-06-20.

The remote platform probe showed that the Python stack and model paths are
mostly ready, but CUDA is not usable:

- `nvidia-smi` cannot communicate with the NVIDIA driver;
- PyTorch is installed with CUDA 12.4 support;
- `torch.cuda.is_available()` is false;
- no SFT or GRPO training should run until this is fixed.

## Diagnosis Job

Run:

```bash
bash train/infra/run_gpu_blocker_diagnosis.sh \
  --job train/infra/jobs/phase1_gpu_blocker_diagnosis.yaml \
  --python /data/jinzhihong/envs/vaevas-rl/bin/python
```

For a remote Codex session, use:

```text
train/infra/jobs/REMOTE_CODEX_TASK_GPU_BLOCKER.md
```

## Interpreting Status

| Status | Meaning | Safe response |
| --- | --- | --- |
| `GPU_READY` | GPU is visible and PyTorch CUDA smoke passes. | Rerun platform probe; proceed only if it reports `READY` or `READY_SFT_ONLY`. |
| `BLOCKED_NO_NVIDIA_SMI` | GPU runtime is not available in this shell/image. | Use the correct GPU node/image/session. |
| `BLOCKED_NO_DEVICE_FILES` | `/dev/nvidia*` is absent. | Request GPU allocation or restart container/session with GPU passthrough. |
| `BLOCKED_NVIDIA_DRIVER` | `nvidia-smi` exists but cannot talk to driver. | If not a GPU session, switch sessions; otherwise ask platform admin. |
| `BLOCKED_TORCH_CUDA` | Driver is visible but PyTorch CUDA fails. | Check Python env and `CUDA_VISIBLE_DEVICES`; do not train. |
| `BLOCKED_UNKNOWN_GPU` | Signals conflict or are incomplete. | Upload evidence and escalate to platform admin. |

## Repair Boundary

Remote Codex may only perform safe session-level repair:

- move from a CPU/login node to a GPU node;
- enter the site's normal GPU allocation/session;
- restart a user-owned container with GPU passthrough if the site already
  provides that command;
- set `CUDA_VISIBLE_DEVICES` to the allocated GPU IDs if the allocation exists.

Remote Codex must not:

- install packages;
- run `sudo`;
- change NVIDIA drivers or kernel modules;
- restart host services;
- launch SFT/GRPO while status is blocked.
