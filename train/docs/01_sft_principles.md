# 01 — SFT Principles

What supervised fine-tuning does, why we need it before RL, what to learn.

## The 60-second summary

SFT = take a pre-trained LM, give it `(prompt, target)` pairs, minimize next-token cross-entropy loss. Same recipe as the original pretraining objective, but on **curated** data.

```
loss = - Σ_t log P(target_t | target_{<t}, prompt)
```

That's it. No reward, no preference, no exploration. The model learns to **imitate** the target distribution.

## Why SFT before GRPO

Three reasons, ordered by importance:

1. **RL on a cold base produces zero reward signal**.
   For Verilog-A generation, `Qwen2.5-Coder-7B` zero-shot will produce code that often won't compile. If 95% of completions have `R_compile = 0`, GRPO's advantage `A_i = (R_i - mean(R)) / std(R)` is undefined or noisy. SFT lifts the base into a regime where ~50%+ samples earn non-zero reward, making RL gradient meaningful.

2. **SFT teaches the output format**.
   We want completions structured as `<think> step1 step2 step3 </think> <answer> ...va code... </answer>`. The base model has never seen this format. Teaching it via reward shaping is hopeless. Via SFT it's two epochs.

3. **SFT injects domain priors that RL cannot create from data alone**.
   Naming conventions, module declarations, port directions, common idioms — all subtle structural facts. RL can sharpen these but cannot invent them from a few hundred prompts.

Conversely: **SFT cannot, on its own, optimize for "does this code actually work"**. The teacher signal in SFT is "produce these tokens," not "produce code that compiles." That's RL's job.

## Full-parameter vs LoRA

| | Full SFT | LoRA / QLoRA |
|---|---|---|
| Memory | ~14× model size (Adam state, grad, activations) | 1-2× model size |
| Quality on small data | Usually better | Can underfit if rank is too small |
| Used by Circuit-Think | ✅ Full SFT | ❌ |
| Default for `train/` | Full SFT (per Circuit-Think baseline) | Fallback if 2×A100 cannot fit |

**Default**: Full SFT with DeepSpeed ZeRO-3. Switch to LoRA only if 80GB×2 cannot hold Qwen2.5-Coder-7B with seq_len=4096 and batch=1. Empirically this fits; revisit only if it doesn't.

## Hyperparameters that actually matter

| Param | Why it matters | Default |
|---|---|---|
| Learning rate | Too high → forgets pretraining; too low → won't fit | 1e-5 to 5e-5 |
| Epochs | Too few → underfits; too many → overfits | 2 (Circuit-Think) |
| Sequence length | Determines max trajectory + answer length | 4096-8192 |
| Batch × accumulation | Effective batch size; affects LR scaling | effective 16-32 |
| Warmup | Stabilizes early steps | 50 steps |
| Weight decay | Slight regularization | 0.01 |
| Precision | bf16 standard on A100 | bf16 |

## Failure modes to watch

| Symptom | Cause | Fix |
|---|---|---|
| Loss NaN early | LR too high, bf16 instability | drop LR 10×, check data for malformed tokens |
| Loss flat | LR too low, frozen layers | check `requires_grad` everywhere, raise LR |
| Loss drops fast then plateaus high | Data too repetitive, model memorizes prompt format | check data diversity |
| Eval drops as loss drops | Overfitting on small data | reduce epochs, augment data |
| Output is gibberish at inference | Format token confusion | check special tokens are properly tokenized |
| Output ignores `<think>` tags | Tags not in tokenizer, treated as text | add as special tokens BEFORE SFT |

## What we expect to see (Circuit-Think Table 1, adapted)

Circuit-Think's SFT on 1000 image-netlist pairs:
- step 50: 37.13% accuracy
- step 100: 36.85%
- step 150: 36.01%
- step 200: 34.78% ← starts decreasing (overfitting onset)

**Takeaway**: pure SFT on a small dataset **plateaus quickly and starts to overfit**. We expect similar behavior on Verilog-A. The SFT checkpoint is a STARTING POINT for RL, not a final product.

## Key references

See `REFERENCES.md` section "SFT" for:
- "InstructGPT" (Ouyang et al. 2022) — original RLHF, includes SFT-first recipe
- DeepSeek-R1 (Guo et al. 2025) — modern SFT-then-RL paradigm
- HuggingFace `trl.SFTTrainer` docs
- Qwen2.5-Coder technical report (Bai et al. 2025)
