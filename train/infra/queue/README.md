# Remote Queue

GitHub-mediated remote work queue for the training-paper branch.

The goal is to stop copying long task prompts between local and remote Codex
windows. Local Codex commits Markdown jobs under `pending/`. Remote Codex runs
the worker prompt once, claims the next job, executes it, commits results, moves
the job to `done/` or `failed/`, and pushes.

## Directories

| Directory | Meaning |
| --- | --- |
| `pending/` | Jobs ready for the remote server. |
| `running/` | Jobs claimed by one remote worker. |
| `done/` | Jobs completed with a committed result. |
| `failed/` | Jobs that hit a hard blocker but still uploaded diagnostics. |
| `control/` | Human control flags such as pausing the training line. |

## Rules

- Queue jobs are execution contracts, not paper claims.
- Every job must state boundaries, expected result files, and final response fields.
- Remote workers must return a final result directly; no intermediate confirmation loops.
- Result directories must stay small: no checkpoints, simulator dumps, package caches, `.env`, or secrets.
- Draft training data stays draft until contamination, EVAS, Spectre-shadow, and admission gates pass.
- Active control flags override new queue creation and claiming.
