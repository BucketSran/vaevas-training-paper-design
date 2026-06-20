# logs/

Per-run experiment logs. **Contents gitignored** except this README.

## Layout (per run)

```
logs/
├── README.md
├── .gitignore
├── sft_<run-id>/
│   ├── config.yaml         # snapshot of config used
│   ├── loss.csv            # per-step loss
│   ├── eval_results.json   # final eval metrics
│   ├── data_hash.txt       # SHA256 of train.jsonl
│   ├── model_commit.txt    # HF revision or local commit
│   ├── reward_commit.txt   # commit of rewards/ (for RL runs)
│   ├── stdout.log          # full stdout
│   └── stderr.log
└── grpo_<run-id>/
    ├── config.yaml
    ├── rewards.csv         # per-step: each R component, R_total, KL, entropy
    ├── samples/<step>/     # 8 completions sampled every N steps
    ├── eval_results.json
    └── ...
```

## Run ID convention

`<stage>_<YYYY-MM-DD>_<short-tag>`, e.g. `sft_2026-06-10_baseline`, `grpo_2026-06-15_full-reward`.

## What to log

| Type | Where |
|---|---|
| Per-step training loss / reward | `loss.csv` / `rewards.csv` (append-only) |
| Final eval metrics | `eval_results.json` (one-shot at end) |
| Reproducibility metadata | `config.yaml`, `data_hash.txt`, `model_commit.txt`, `reward_commit.txt` |
| Samples for human review | `samples/<step>/` (8 completions per checkpoint, every 20 steps) |
| Console output | `stdout.log`, `stderr.log` (captured by launch wrapper) |

## What NOT to log here

- Checkpoint weights → `../models/` (or remote).
- Raw training data → `../data/`.
- Aggregated reports → `../eval/reports/`.

## Retention policy

- Keep ALL logs for runs that produced reported numbers.
- Keep failed runs for at least 30 days (debugging value).
- Beyond 30 days, prune failed runs but keep their `config.yaml` + `eval_results.json` for the historical record.
