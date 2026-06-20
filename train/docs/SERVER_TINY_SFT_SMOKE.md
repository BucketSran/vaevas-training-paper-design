# Server Tiny SFT Smoke

Status: next remote task after platform probe `READY`, 2026-06-21.

This smoke checks the smallest useful SFT training loop on the remote machine:

1. rebuild the toy admitted manifest from Phase 1 fixtures;
2. pack one admitted toy item into SFT JSONL;
3. duplicate it into a tiny train/eval set inside memory;
4. load `/data/jinzhihong/vaEVAS/models/base/Qwen2.5-Coder-7B-Instruct-vaevas`;
5. run a two-step LoRA update on one GPU;
6. save and reload the adapter in an untracked work directory;
7. commit only small evidence files back to GitHub.

It is not a paper result and must not be used as a model-quality claim.

## Command

```bash
bash train/infra/run_tiny_sft_smoke.sh \
  --job train/infra/jobs/phase1_tiny_sft_smoke.yaml \
  --python /data/jinzhihong/envs/vaevas-rl/bin/python \
  --cuda-visible-devices 0
```

For a remote Codex session, use:

```text
train/infra/jobs/REMOTE_CODEX_TASK_TINY_SFT_SMOKE.md
```

## Expected Committed Files

```text
train/infra/results/tiny-sft-smoke-<run_id>/
├── adapter_manifest.json
├── admitted_manifest.yaml
├── commands.log
├── environment.txt
├── loss.csv
├── sample_generation.txt
├── sft_pack_manifest.yaml
├── status.txt
└── summary.json
```

## Files That Must Not Be Committed

The script writes the adapter/checkpoint work area outside the repo by default:

```text
/tmp/vaevas-training-paper-design-tiny-sft/<run_id>/
```

Do not commit anything from that work directory. The committed
`adapter_manifest.json` records where the adapter was saved and confirms it was
not committed.

## Pass Criteria

| Signal | Required |
| --- | --- |
| `status.txt` | `PASS` |
| `summary.status` | `PASS` |
| `summary.toy_data_only` | `true` |
| `summary.max_steps` | `2` |
| `summary.lora.reload_check_ok` | `true` |
| `loss.csv` | exists and contains trainer logs |

If this passes, the next task is not GRPO yet. The next local step should be a
real clean-room pilot-batch generator/admission path or a slightly larger SFT
smoke using admitted clean-room data.
