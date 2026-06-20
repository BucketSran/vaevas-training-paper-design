# train_sft/

Supervised fine-tuning of `Qwen2.5-Coder-7B` on EVAS-verified `<prompt, trajectory + answer>` pairs.

## Files (to be implemented in Phase 2)

| File | Role |
|---|---|
| `train.py` | Full-parameter SFT script using `transformers` + `trl.SFTTrainer` |
| `config/qwen25_coder_7b.yaml` | Hyperparameters: batch size, LR, epochs, seq length, deepspeed/accelerate config |
| `launch.sh` | Wrapper for remote (2× A100) launch |

## Recipe (mirrors Circuit-Think SFT stage)

| Item | Value (default, tune in Phase 2) |
|---|---|
| Base model | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| Strategy | Full-parameter SFT (not LoRA) |
| Data | `../data/sft/train.jsonl` (~1000-2500 items at start) |
| Epochs | 2 |
| Batch (per-device) | 1-2 (depends on seq length) |
| Gradient accumulation | 8-16 |
| Sequence length | 4096-8192 |
| Optimizer | AdamW |
| Learning rate | 1e-5 to 5e-5 (cosine schedule, warmup 50 steps) |
| Weight decay | 0.01 |
| Precision | bf16 |
| Distributed | DeepSpeed ZeRO-3 or `accelerate` FSDP |

## Why these choices

- **Full SFT, not LoRA**: Circuit-Think used full SFT (paper section "Implementation Details"). LoRA may underfit on small data; revisit if compute becomes tight.
- **2 epochs**: Circuit-Think used 2 epochs. More risks overfitting (paper Table 1: SFT loss starts climbing back).
- **bf16**: A100 supports it natively; halves memory vs fp32.
- **2× A100 (80GB)**: fits 7B full SFT with ZeRO-3.

## Validation before claiming done

1. **Smoke test**: 10-20 samples, batch=1, 1 epoch — verify the loop runs end-to-end.
2. **Loss sanity**: training loss should drop from ~2-3 to <1 within the first epoch. NaN or divergence = stop, debug.
3. **Inference works**: load checkpoint, generate on a held-out spec, verify output parseable.
4. **Floor check**: base model eval rate measured before SFT; SFT must beat it on compile rate by ≥15 pp.

## Logging

Every run produces `../logs/sft_<run-id>/`:
- `config.yaml` (snapshot)
- `loss.csv`
- `checkpoint/` (gitignored)
- `eval_results.json`
- `data_hash.txt` (hash of `train.jsonl`)
- `model_commit.txt` (HF revision or local commit)

## When to advance to GRPO

Per `../KPI.md` Phase 2 gate: held-out compile rate ≥ 60%. Do not start RL on a checkpoint that didn't pass this gate.
