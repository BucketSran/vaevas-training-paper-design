# jobs/

Server job manifests for GitHub-mediated handoff.

These YAML files are lightweight execution contracts. The current smoke script
does not need a full YAML parser; the manifest is mainly for humans and for
future automation to know what command was intended.

## Current Job

| File | Purpose |
| --- | --- |
| `phase1_github_handoff_smoke.yaml` | Validate server can run the local Phase 1 manifest/admission/SFT/GRPO pack loop and upload small results. |
| `phase1_server_platform_probe.yaml` | Validate CUDA/PyTorch/SFT/GRPO package readiness without training. |
| `REMOTE_CODEX_TASK_PLATFORM_PROBE.md` | Copy-paste instructions for a remote Codex session controlling the server. |

## Job Rules

- A job must have a stable `job_id`.
- A job must state whether it is allowed to use GPUs.
- A job must state whether it may write checkpoints.
- A job must list expected upload files.
- Default smoke jobs must not launch full SFT/GRPO training.
- Blocked platform probes must still upload their result directory for diagnosis.
