# Remote Codex Task: Server Platform Probe

You are operating on the remote training server. Run only the non-destructive
platform probe and upload the small result directory back to GitHub.

## Hard Constraints

- Do not train a model.
- Do not install, upgrade, or remove packages.
- Do not change NVIDIA drivers or conda environments.
- Do not start Ray clusters, vLLM services, SFT jobs, or GRPO jobs.
- Do not commit checkpoints, LoRA adapters, model weights, `.env` files, tokens,
  full datasets, or simulator dumps.

## Commands

If the repo already exists:

```bash
cd /data/jinzhihong/vaevas-training-paper-design
git fetch origin
git checkout training-paper-design-20260620
git pull --ff-only
```

If the repo does not exist:

```bash
cd /data/jinzhihong
git clone https://github.com/BucketSran/vaevas-training-paper-design.git
cd vaevas-training-paper-design
git checkout training-paper-design-20260620
git pull --ff-only
```

Run the probe. Prefer the known environment from the previous smoke if it exists:

```bash
bash train/infra/run_server_platform_probe.sh \
  --job train/infra/jobs/phase1_server_platform_probe.yaml \
  --python /data/jinzhihong/envs/vaevas-rl/bin/python
```

If that Python path does not exist, use:

```bash
bash train/infra/run_server_platform_probe.sh \
  --job train/infra/jobs/phase1_server_platform_probe.yaml
```

The script prints a result directory like:

```text
result_dir=train/infra/results/server-platform-probe-YYYYMMDDTHHMMSSZ
summary=train/infra/results/server-platform-probe-YYYYMMDDTHHMMSSZ/summary.json
```

Upload exactly that result directory:

```bash
git status --short
git add train/infra/results/server-platform-probe-YYYYMMDDTHHMMSSZ
git commit -m "Upload server platform probe result server-platform-probe-YYYYMMDDTHHMMSSZ"
git push origin training-paper-design-20260620
```

Replace `server-platform-probe-YYYYMMDDTHHMMSSZ` with the actual run ID printed
by the script.

## Expected Outcomes

- `READY`: GPU and SFT/GRPO stack are usable.
- `READY_SFT_ONLY`: SFT can proceed; GRPO packages need later work.
- `BLOCKED_GPU`: do not train; upload result for local diagnosis.
- `BLOCKED_IMPORTS`: do not train; upload result for local diagnosis.

Even if the result is blocked, upload the result directory. The local agent will
read `summary.json`, `gpu_probe.json`, and `python_packages.json`.
