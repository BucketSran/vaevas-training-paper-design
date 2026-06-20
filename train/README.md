# train/

Experimental subproject: train a local Verilog-A reasoning model on the vaEVAS stack via **SFT + GRPO**, inspired by Circuit-Think (AAAI'26).

> **Status: PHASE 0 DESIGN.** No Phase 1 clean-room training data has been
> admitted. Historical SFT platform smoke evidence exists in
> `docs/sft/09_smoke_lessons.md`; it validates plumbing, not model quality or
> paper claims.

## Scope

This directory is a **research exploration**, **not** part of the vaEVAS paper mainline. Top-level `AGENTS.md` (line 24) explicitly lists local SFT/fine-tuning as a non-goal of the benchmark paper. This subproject reopens that scope as a **parallel experimental track** so vaBench results remain trustworthy.

The hard rule: **nothing inside `train/` may contaminate the vaBench benchmark evaluation**. See [`SCOPE_BOUNDARY.md`](./SCOPE_BOUNDARY.md) — this is enforced as a firewall, not a guideline.

## Entry Points

Read in this order:

1. [`BRIEF.md`](./BRIEF.md) — what we are building and why
2. [`SCOPE_BOUNDARY.md`](./SCOPE_BOUNDARY.md) — contamination firewall, read before touching data
3. [`KPI.md`](./KPI.md) — how we know each phase is done
4. [`ROADMAP.md`](./ROADMAP.md) — phased plan: framework → data → SFT → GRPO → eval
5. [`PLAN.md`](./PLAN.md) — concrete execution plan for the current phase
6. [`AGENTS.md`](./AGENTS.md) — operating contract for agents working inside `train/`
7. [`docs/`](./docs/) — design notes and SFT/GRPO learning material

## Quick Layout

```
train/
├── README.md, AGENTS.md, BRIEF.md, KPI.md, ROADMAP.md, PLAN.md, SCOPE_BOUNDARY.md
├── docs/         — design notes, SFT/GRPO principles, references, decisions
├── data/         — contracts → synthesized → manifests → admitted → sft/rl/eval (gitignored)
├── pipelines/    — data synthesis + EVAS verification + trajectory building
├── rewards/      — diagnostic reward implementations (format / contract / compile / sim / property / repair)
├── models/       — base model config + checkpoints (gitignored)
├── train_sft/    — SFT training scripts and configs
├── train_rl/     — GRPO training scripts and configs
├── eval/         — held-out evaluation harness (NOT vabench)
├── infra/        — remote server / cluster scripts (2× A100)
└── logs/         — experiment logs (gitignored)
```

## Reference Paper

Method baseline: Circuit-Think (Jiang et al., AAAI 2026, DOI 10.1609/aaai.v40i7.37465).

Key adaptations vs Circuit-Think:
- **Text-only** task (Verilog-A code, not circuit images) → base is `Qwen2.5-Coder-7B`, not VL.
- **Audited verifier**: EVAS is the fast deterministic training evaluator, while Spectre remains the reference oracle for shadow audits and reportable claims.
- **Multi-task**: benchmark-aligned `level`, `task_form`, and `category` slices, not a single task.
- **Training-paper method**: clean-room contract synthesis, diagnostic rewards, and EVAS-as-audited accelerator, not a new RL optimizer.
