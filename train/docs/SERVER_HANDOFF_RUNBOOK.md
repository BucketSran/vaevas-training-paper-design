# GitHub-Based Server Handoff Runbook

Status: current operating procedure while direct SSH is unavailable, 2026-06-20.

This runbook defines how the remote SFT/GRPO server interacts with the design
repo through GitHub. It is intentionally conservative: first run small smoke
checks, upload small evidence files, and do not launch full training.

## Current Constraint

The local agent cannot reliably SSH into the server right now. The available
control path is:

```text
local repo
  -> push public GitHub branch
  -> user controls server by screen-sharing
  -> server pulls GitHub branch
  -> server runs smoke command
  -> server commits/pushes small result directory
  -> local agent reads result from GitHub/local sync
```

This is enough to revalidate the remote platform without giving the local agent
interactive shell access.

## Branch and Repo

Use the public design branch:

```bash
git clone https://github.com/BucketSran/vaevas-training-paper-design.git
cd vaevas-training-paper-design
git checkout training-paper-design-20260620
git pull --ff-only
```

If the repo already exists on the server:

```bash
cd <server-copy>/vaevas-training-paper-design
git fetch origin
git checkout training-paper-design-20260620
git pull --ff-only
```

## First Server Command

Run only the GitHub handoff smoke first:

```bash
bash train/infra/run_github_handoff_smoke.sh \
  --job train/infra/jobs/phase1_github_handoff_smoke.yaml
```

The script should:

1. record hostname, date, git commit, Python version, GPU visibility, and key
   package imports;
2. run the Phase 1 manifest validator;
3. build a protected-index smoke manifest;
4. run contamination positive and negative controls;
5. generate an admitted manifest from reports;
6. pack SFT JSONL from the generated admitted manifest;
7. pack GRPO prompt JSONL without answer leakage;
8. write a small result directory under `train/infra/results/<run_id>/`.

This does not train a model and does not write checkpoints.

## Upload Result Back to GitHub

After the smoke completes:

```bash
git status --short
git add train/infra/results/<run_id>
git commit -m "Upload server handoff smoke result <run_id>"
git push origin training-paper-design-20260620
```

Commit only the result directory produced by the smoke. Do not commit:

- model checkpoints;
- LoRA adapters;
- full generated training datasets;
- raw simulator dumps;
- secret tokens;
- `.env` files;
- wandb local directories.

## Result Files to Expect

The result directory should contain:

```text
train/infra/results/<run_id>/
├── commands.log
├── environment.txt
├── summary.json
├── status.txt
└── artifacts/
    ├── admitted_manifest.yaml
    ├── admitted_manifest_reject.yaml
    ├── contamination_clean.yaml
    ├── contamination_reject.yaml
    ├── grpo_prompt_manifest.yaml
    ├── protected_index.yaml
    └── sft_pack_manifest.yaml
```

The `summary.json` file is the main machine-readable handoff artifact.

## How to Interpret Results

| Signal | Meaning |
| --- | --- |
| `status.txt = PASS` | GitHub handoff and local Phase 1 data plumbing work on the server. |
| `pipeline.validate_fixture = true` | The server can parse and validate manifest fixtures. |
| `contamination.clean_count = 1` | The fixture positive path is accepted. |
| `contamination.reject_count = 1` | The exact-hash negative control is rejected. |
| `admission.admitted_count = 1` | Evidence reports can generate an admitted manifest. |
| `admission.reject_control_count = 1` | Dirty evidence does not enter admission. |
| `sft.train_lines = 1` | SFT packer can emit training JSONL from admitted data. |
| `grpo.prompt_lines = 1` | GRPO packer can emit prompt JSONL from admitted data. |
| `grpo.no_answer_leak = true` | GRPO prompt does not expose supervised answer fields. |

If this passes, the next server-side smoke can check historical SFT platform
paths and package versions. If it fails, upload the result directory anyway.

## When SSH Returns

When direct SSH works again, keep this protocol as the reproducible baseline:

- use GitHub result directories for small evidence;
- use SSH/rsync only for large logs or private artifacts;
- keep checkpoints out of GitHub;
- keep the same `run_id` naming convention across both modes.
