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
| `phase1_gpu_blocker_diagnosis.yaml` | Diagnose `BLOCKED_GPU` without package installs, driver changes, or training. |
| `phase1_tiny_sft_smoke.yaml` | Run two-step LoRA SFT smoke on toy admitted data and upload small evidence files. |
| `phase1_prompt_gate.yaml` | Validate pilot prompt rendering without LLM calls, training, EVAS, or Spectre. |
| `phase1_contract_synthesis_smoke.yaml` | Generate five draft contract YAMLs from clean-room prompts and upload a small review result directory. |
| `phase1_contract_batch_overnight.yaml` | Generate 25 draft contracts, review them, generate draft Verilog-A for accepted contracts, and pack draft unadmitted SFT/GRPO JSONL. |
| `phase1_evas_rust_toolchain_fix.yaml` | Resolve remote EVAS PR12 Rust backend blocker by installing/exposing user-local cargo/rustc and rerunning the pinned evas-rust smoke. |
| `phase1_remote_worker_queue.yaml` | Define the GitHub-mediated remote queue used to avoid repeated copy-paste handoffs. |
| `phase1_evas_failure_repair.yaml` | Repair the 10 EVAS Rust smoke failures from contract-batch-0025, rerun EVAS, and rebuild draft packs only if all candidates pass. |
| `REMOTE_CODEX_TASK_PLATFORM_PROBE.md` | Copy-paste instructions for a remote Codex session controlling the server. |
| `REMOTE_CODEX_TASK_GPU_BLOCKER.md` | Copy-paste instructions for remote Codex to diagnose GPU visibility and rerun platform probe only after safe session-level fixes. |
| `REMOTE_CODEX_TASK_TINY_SFT_SMOKE.md` | Copy-paste instructions for remote Codex to run tiny SFT smoke after platform `READY`. |
| `REMOTE_CODEX_TASK_PROMPT_GATE.md` | Copy-paste instructions for remote Codex to validate the prompt gate. |
| `REMOTE_CODEX_TASK_CONTRACT_SYNTHESIS_SMOKE.md` | Copy-paste instructions for remote Codex to run the one-shot contract synthesis smoke and return summary directly. |
| `REMOTE_CODEX_TASK_CONTRACT_BATCH_OVERNIGHT.md` | Copy-paste instructions for remote Codex to run the larger overnight contract + artifact + draft training-pack job. |
| `REMOTE_CODEX_TASK_EVAS_RUST_TOOLCHAIN_FIX.md` | Copy-paste instructions for remote Codex to fix the missing Rust toolchain blocker and rebuild pinned EVAS PR12. |

## Job Rules

- A job must have a stable `job_id`.
- A job must state whether it is allowed to use GPUs.
- A job must state whether it may write checkpoints.
- A job must list expected upload files.
- Default smoke jobs must not launch full SFT/GRPO training.
- Blocked platform probes must still upload their result directory for diagnosis.
- GPU blocker diagnosis must not run `sudo`, install packages, or change drivers.
- Tiny SFT smoke may write adapters/checkpoints only to an untracked work directory; never commit them.
- Prompt gate jobs must not call external LLM APIs or commit rendered prompt JSONL.
- Contract synthesis smoke may commit exactly five draft contract YAML files plus their generated contract index as review evidence; it must not commit Verilog-A artifacts or training packs.
- Contract batch overnight may commit draft Verilog-A artifacts and draft SFT/GRPO JSONL, but those outputs must remain explicitly unadmitted and must not include checkpoints, simulator dumps, EVAS/Spectre claims, or model-training results.
- EVAS Rust toolchain fix may install or expose user-local Rust only; it must not use `sudo`, run Spectre, run training, or write checkpoints.
- Queue worker jobs must finish with `train/infra/remote_worker/finish_job.sh`
  so `pending/`, `running/`, `done/`, and `failed/` stay consistent.
- EVAS failure repair jobs must preserve original batch result directories and
  write repaired draft artifacts under a new result directory.
