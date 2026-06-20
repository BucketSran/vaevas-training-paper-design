# Remote Codex Task: GPU Blocker Diagnosis

The previous platform probe returned `BLOCKED_GPU`. Your task is to diagnose the
GPU visibility problem, apply only safe non-admin fixes if obvious, and upload
the result back to GitHub.

## Hard Constraints

- Do not train a model.
- Do not install, upgrade, or remove packages.
- Do not run `sudo`.
- Do not change NVIDIA drivers, kernel modules, or system services.
- Do not start Ray clusters, vLLM services, SFT jobs, or GRPO jobs.
- Do not commit checkpoints, adapters, model weights, `.env` files, tokens, full
  datasets, or simulator dumps.

## What Is Known

The latest platform probe on `huaxiyun085` reported:

- Python environment: `/data/jinzhihong/envs/vaevas-rl/bin/python`
- training packages present: `torch`, `transformers`, `trl`, `datasets`,
  `accelerate`, `vllm`, `verl`, `ray`
- known 7B model paths exist
- blocker: `nvidia-smi` cannot communicate with the NVIDIA driver and
  `torch.cuda.is_available()` is false

This means the next step is GPU/session/driver diagnosis, not SFT/GRPO.

## Commands

Update the repo:

```bash
cd /data/jinzhihong/vaevas-training-paper-design
git fetch origin
git checkout training-paper-design-20260620
git pull --ff-only
```

Run the GPU blocker diagnosis:

```bash
bash train/infra/run_gpu_blocker_diagnosis.sh \
  --job train/infra/jobs/phase1_gpu_blocker_diagnosis.yaml \
  --python /data/jinzhihong/envs/vaevas-rl/bin/python
```

The script prints:

```text
result_dir=train/infra/results/gpu-blocker-diagnosis-YYYYMMDDTHHMMSSZ
summary=train/infra/results/gpu-blocker-diagnosis-YYYYMMDDTHHMMSSZ/summary.json
```

Upload exactly that result directory:

```bash
git status --short
git add train/infra/results/gpu-blocker-diagnosis-YYYYMMDDTHHMMSSZ
git commit -m "Upload GPU blocker diagnosis result gpu-blocker-diagnosis-YYYYMMDDTHHMMSSZ"
git push origin training-paper-design-20260620
```

Replace the run ID with the actual one printed by the script.

## Safe Fix Policy

You may fix only safe session-placement problems:

- If the current shell is on a login/CPU node, move to an ordinary GPU node or
  allocated GPU session using the site's normal command, then rerun the platform
  probe.
- If this is a container/session without GPU passthrough and the site provides a
  standard GPU-enabled launch command, restart into that GPU-enabled session,
  then rerun the platform probe.
- If `CUDA_VISIBLE_DEVICES` is empty or wrong but a GPU allocation exists, set it
  according to the allocation and rerun the platform probe.

Do not attempt host-level repair. If diagnosis indicates driver/module/service
failure, upload the result and ask the platform administrator.

## After Any Safe Fix

If you move to a GPU-capable session, rerun:

```bash
bash train/infra/run_server_platform_probe.sh \
  --job train/infra/jobs/phase1_server_platform_probe.yaml \
  --python /data/jinzhihong/envs/vaevas-rl/bin/python
```

Then upload the new `server-platform-probe-*` result directory as before.
