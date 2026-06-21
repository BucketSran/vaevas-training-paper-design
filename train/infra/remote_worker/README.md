# Remote Worker

This folder contains the one-time remote Codex workflow for GitHub-mediated
jobs.

## Human usage

After the remote machine has this branch checked out, tell remote Codex only:

```text
请读取并执行 train/infra/remote_worker/REMOTE_CODEX_WORKER_PROMPT.md。
不要向我进行中间确认；如果有 pending job，直接 claim、执行、提交、push，并返回最终结果。
```

If the remote checkout is stale, first run:

```bash
git fetch origin training-paper-design-20260620
git checkout training-paper-design-20260620
git pull --ff-only origin training-paper-design-20260620
```

## Worker commands

| Command | Role |
| --- | --- |
| `bash train/infra/remote_worker/poll_once.sh` | Pull latest branch, claim the oldest pending job, print the job body. |
| `bash train/infra/remote_worker/finish_job.sh --status done --job <running-job> --result <result-dir> --summary "<summary>"` | Move a claimed job to `done/`, add result evidence, commit, and push. |
| `bash train/infra/remote_worker/finish_job.sh --status failed --job <running-job> --result <result-dir> --summary "<blocker>"` | Move a claimed job to `failed/`, add diagnostics, commit, and push. |

The scripts do not run experiments by themselves. They provide reliable queue
bookkeeping so the remote Codex session can focus on the job content.

