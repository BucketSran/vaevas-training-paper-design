# KPI

Phase-gated acceptance criteria. Each phase has a single binary `gate_passed` check plus diagnostic metrics. Do not advance phases without passing the gate.

## Phase 0 — Framework Scaffold (current)

**Gate**: User can read `BRIEF.md`, `SCOPE_BOUNDARY.md`, and `ROADMAP.md` and approve the subproject before any data is collected.

**Diagnostics**:
- `train/` directory tree exists with all subdirs.
- All top-level docs (`README.md`, `BRIEF.md`, `KPI.md`, `ROADMAP.md`, `PLAN.md`, `SCOPE_BOUNDARY.md`, `AGENTS.md`) present and reviewed.
- All subdir `README.md` files present.
- `docs/01_sft_principles.md` ... `06_eval_protocol.md` + `REFERENCES.md` present.

**Exit signal**: User says "framework approved, proceed to Phase 1."

---

## Phase 1 — Data Pipeline + Contamination Firewall

**Gate**: At least **300 EVAS-verified data points** in `train/data/verified/` with provenance metadata, contract records, and zero vaBench-release overlap (`check_contamination.py` returns clean).

**Diagnostics**:
- `train/data/verified/` count ≥ 300.
- Each entry has a `.meta.yaml` with `vabench_audit.release_overlap == false`.
- Distribution across `level`, `task_form`, and benchmark-aligned `category` — log the counts; uneven is acceptable, but every admitted bucket must be intentional.
- `train/pipelines/check_contamination.py` runs in CI / pre-commit hook.
- `train/data/eval/` has at least 50 held-out items, disjoint at the spec level.

**Stop conditions**:
- Cannot reach 300 within available sources → escalate to user, do not relax the contamination rule to compensate.

---

## Phase 2 — SFT Pipeline End-to-End

**Gate**: One full SFT run produces a checkpoint that, on `train/data/eval/`, achieves **EVAS compile/elaboration ≥ 60%** on generated artifacts.

**Diagnostics**:
- Training loss curve in `logs/<run>/loss.csv` is monotonic-ish (no divergence).
- Checkpoint loadable in inference mode on the same hardware.
- `eval/run_eval.py` produces a structured JSON report.
- Base model (no SFT) EVAS compile/elaboration rate measured for the same eval set — used as the floor.
- Improvement over base ≥ +15 pp on EVAS compile/elaboration rate.

**Stop conditions**:
- SFT diverges (loss NaN, gradient explosion) → diagnose before retry.
- EVAS compile/elaboration rate < base after SFT → bug in data or training script; do NOT proceed.

---

## Phase 3 — GRPO Pipeline End-to-End

**Gate**: One full GRPO run on top of the SFT checkpoint, with the appropriate diagnostic reward profile active, produces a model whose `R_total` curve trends upward across training steps, and where final EVAS compile/elaboration rate ≥ **80%** on `train/data/eval/`.

**Diagnostics**:
- Reward curves logged per step: `R_format`, `R_contract`, `R_static`, `R_compile_diag`, `R_sim_health`, `R_property`, optional `R_l2_decomp`, optional `R_repair_delta`, optional `R_feedback_use`, `P_anti_hack`, `R_total`.
- KL divergence from reference model stays bounded (< 5 nat by default).
- No `R_total` collapse to a single mode (entropy of sampled outputs above floor).
- Simulation rate ≥ 50% on held-out.
- Functional correctness ≥ 30% on held-out.

**Stop conditions**:
- Reward hacking detected (e.g., model outputs trivially-passing dummy code) → revisit reward design.
- KL divergence explodes → reduce LR, increase β.
- EVAS compile/elaboration rate after GRPO < after SFT alone → roll back to SFT.

---

## Phase 4 — Full Evaluation Report

**Gate**: A reproducible report in `eval/reports/<date>_final.md` with:

- compile / simulate / correct rates on `train/data/eval/`, with confidence intervals.
- Breakdown by `level`, `task_form`, and benchmark-aligned `category`.
- Ablation: w/o trajectory reward, w/o reflective learning, w/o GRPO (SFT-only), base model zero-shot.
- Contamination audit re-run and certified.
- Optional: a separate gated experiment on a small vaBench slice with EXPLICIT prior contamination clearance.

**Diagnostics**:
- Report passes `train/eval/audit_report.py` (checks all required fields present).
- Numbers reproducible from `logs/<run>/` with `eval/run_eval.py --replay`.

**Stop conditions**:
- If any number in the report cannot be traced to a `logs/<run>/` artifact, it must be removed from the report.

---

## Cross-Cutting Quality Gates

These apply at every phase:

| Gate | Check |
|---|---|
| Contamination | `check_contamination.py` returns 0 hits on training set vs vaBench release. |
| Reproducibility | Every reported number maps to a `logs/<run-id>/` directory with seed, config, data hash. |
| Documentation | Any new abstraction has a docstring; any new file type has a schema in `docs/`. |
| `.gitignore` | Checkpoints, raw data, logs are gitignored; only configs and reports are tracked. |
