# Roadmap

Five phases, each gated by `KPI.md`. Phases run sequentially; advance only when the previous gate passes.

```
Phase 0  ─────►  Phase 1  ─────►  Phase 2  ─────►  Phase 3  ─────►  Phase 4
Scaffold         Data Pipeline    SFT End-to-End   GRPO End-to-End  Full Eval Report
(now)            (1k-2k pilot,    (EVAS compile    (EVAS compile    (reproducible)
                 2k-3k scale)    ≥60%)            ≥80%)
```

## Phase 0 — Scaffold (CURRENT)

**Goal**: Have a complete framework for SFT/GRPO research, with scope rules locked in and learning materials in place.

**Tasks**:
1. Create directory tree under `train/`.
2. Write top-level docs (`README`, `AGENTS`, `BRIEF`, `KPI`, `ROADMAP`, `PLAN`, `SCOPE_BOUNDARY`).
3. Write subdir `README.md` placeholders explaining what goes where.
4. Write learning notes in `docs/` (SFT, GRPO, reward design, trajectory, data, eval, references).
5. Set up `.gitignore` for data/, models/, logs/.
6. Get user approval on the scaffold before touching data.

**Exit**: Phase 0 KPI gate satisfied.

## Phase 1 — Data Pipeline

**Goal**: Build a contamination-safe pipeline that produces contract-backed, EVAS-verified, Spectre-audited training data with step-by-step trajectory annotations.

**Tasks**:
1. **Source inventory** (3-5 days):
   - Build a safe seed catalog from clean-room contracts, public references, and explicitly audited internal examples.
   - Use `behavioral-veriloga-eval/tasks/` only for taxonomy/error-type inspiration unless an item receives explicit contamination clearance.
   - Record provenance and vaBench-overlap audit for every admitted seed.
   - Build a starting catalog of 150-250 safe seed contracts and artifacts for the pilot.
2. **Synthesis pipeline** (1 week):
   - Implement `pipelines/synthesize.py` — use a strong LLM to generate contract-conditioned artifact proposals.
   - Implement `pipelines/verify_evas.py` — run EVAS compile/elaboration + simulation diagnostics.
   - Implement Spectre shadow audit on the promotion slice.
   - Drop candidates that fail admission gates.
3. **Trajectory builder** (3-5 days):
   - Implement `pipelines/build_trajectory.py` — given a verified `<spec, va, tb>`, decompose into step-by-step CoT (see `docs/04_trajectory_format.md`).
4. **Contamination check** (1-2 days):
   - Implement `pipelines/check_contamination.py` from `docs/CONTAMINATION_CHECKER_SPEC.md`.
   - Build protected/candidate indexes, match tiers, split-leakage report, and admission decisions.
   - Run on every batch before promotion.
5. **Spectre shadow audit**:
   - Implement pilot manifests from `docs/SPECTRE_SHADOW_AUDIT_PROTOCOL.md`.
   - Run Spectre on required high-risk slices before paper-facing claims.
6. **Held-out construction**:
   - Set aside 100-300 clean-room eval items, disjoint at spec and `split_key` level.
   - Set aside 50-100 OOD eval items, primarily by held-out circuit category plus optional L2-hard holdout.

**Exit**: 1k-2k pilot admitted data points, audit clean, diversity report passed, eval/OOD splits created. Serious SFT/GRPO claims wait for 2k-3k admitted scale-up.

## Phase 2 — SFT End-to-End

**Goal**: A working SFT pipeline producing a non-degenerate checkpoint on 2× A100.

**Tasks**:
1. **Infra** (2-3 days):
   - `infra/setup_remote.sh` — clone, install, environment.
   - `infra/sync.sh` — rsync data and code to remote.
2. **Training script** (3-5 days):
   - `train_sft/train.py` — full-parameter SFT on `Qwen2.5-Coder-7B` using HuggingFace `transformers` + `trl.SFTTrainer` or `deepspeed`/`accelerate`.
   - `train_sft/config/qwen25_coder_7b.yaml` — batch, LR, epochs, sequence length.
3. **Validation** (1-2 days):
   - Run SFT on a tiny subset (10-20 samples) end-to-end first — verify the loop works before scaling.
   - Run full SFT (1 epoch).
   - Run inference on `train/data/eval/`.
   - Run base model on `train/data/eval/` as baseline floor.
4. **Stop checks**:
   - Loss curve sane.
   - Inference outputs in correct format.
   - EVAS compile/elaboration rate measurable.

**Exit**: Phase 2 KPI gate (EVAS compile/elaboration ≥ 60%).

## Phase 3 — GRPO End-to-End

**Goal**: GRPO training loop with diagnostic reward profiles, producing a model that improves over the SFT baseline.

**Tasks**:
1. **Reward modules** (1 week):
   - Implement reward functions from `docs/DIAGNOSTIC_REWARD_SPEC.md`.
   - Implement repair/feedback reward support where the active `task_form` requires it.
   - Write `rewards/tests/test_rewards.py` — unit tests for each.
2. **GRPO loop** (1 week):
   - `train_rl/train_grpo.py` — adapt HuggingFace `trl.GRPOTrainer` or `verl`.
   - Wire reward functions via `reward_funcs=[...]`.
   - Implement reward weighting and step gating.
3. **Tiny-scale validation**:
   - Run GRPO for 10 steps on 20 samples to verify the loop converges.
4. **Full GRPO run**:
   - 100-200 steps depending on convergence behavior.
   - Log reward curves, KL, entropy.
5. **Reward ablations** (mandatory):
   - w/o property reward
   - w/o L2 decomposition reward
   - w/o repair/feedback reward
   - compile-only / simulate-only reward baselines
   - Each ablation: short run (50 steps), capture final metrics.

**Exit**: Phase 3 KPI gate (EVAS compile/elaboration ≥ 80% on held-out, active reward profile verified).

## Phase 4 — Full Evaluation Report

**Goal**: A defensible report on the trained model's capability, with ablations and contamination audit.

**Tasks**:
1. **Held-out eval** (full):
   - Run `eval/run_eval.py` on all `train/data/eval/`.
   - Stratify by `level`, `task_form`, benchmark-aligned `category`, and difficulty.
   - Compute confidence intervals via bootstrap.
2. **Ablation table**:
   - Final SFT-only vs final TGRL-style.
   - Each reward component removed.
3. **Contamination re-audit**:
   - Re-run `check_contamination.py` on the final training set.
   - State the audit result in the report header.
4. **Optional vaBench probe**:
   - Only if the user explicitly requests, and only with separate logging.
   - Treat as a one-time experiment, not a recurring benchmark run.

**Exit**: Report in `eval/reports/<date>_final.md` passes audit.

## Out of scope (for now)

These are deferred until after Phase 4 succeeds:
- Multi-modal extension (image inputs).
- Larger base models (32B+).
- Distillation, quantization, deployment.
- Integrating the trained model back into `behavioral-veriloga-eval/` as a baseline (would require explicit scope reopening).
