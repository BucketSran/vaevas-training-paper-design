# Remote Codex Task: Tiny SFT Smoke

The latest platform probe is `READY`. Run the tiny SFT smoke only. This verifies
the training loop; it is not a real experiment and not a paper metric.

## Hard Constraints

- Do not run full SFT.
- Do not run GRPO.
- Do not install, upgrade, or remove packages.
- Do not commit checkpoints, LoRA adapters, merged models, model weights,
  `.env` files, tokens, full datasets, or simulator dumps.
- Only commit the small result directory under `train/infra/results/<run_id>/`.

## Commands

Update the repo:

```bash
cd /data/jinzhihong/vaevas-training-paper-design
git fetch origin
git checkout training-paper-design-20260620
git pull --ff-only
```

Run the tiny SFT smoke on one visible GPU:

```bash
bash train/infra/run_tiny_sft_smoke.sh \
  --job train/infra/jobs/phase1_tiny_sft_smoke.yaml \
  --python /data/jinzhihong/envs/vaevas-rl/bin/python \
  --cuda-visible-devices 0
```

The script prints:

```text
result_dir=train/infra/results/tiny-sft-smoke-YYYYMMDDTHHMMSSZ
summary=train/infra/results/tiny-sft-smoke-YYYYMMDDTHHMMSSZ/summary.json
```

Upload exactly that result directory:

```bash
git status --short
git add train/infra/results/tiny-sft-smoke-YYYYMMDDTHHMMSSZ
git commit -m "Upload tiny SFT smoke result tiny-sft-smoke-YYYYMMDDTHHMMSSZ"
git push origin training-paper-design-20260620
```

Replace the run ID with the actual one printed by the script.

## Expected Result

`summary.json` should report:

- `status = PASS`
- `toy_data_only = true`
- `max_steps = 2`
- `adapter_dir_committed = false`
- `lora.reload_check_ok = true`

If it fails, still upload the result directory. The local agent will inspect
`summary.json`, `commands.log`, and `loss.csv` if present.
