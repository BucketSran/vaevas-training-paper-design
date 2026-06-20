# eval/

Held-out evaluation harness for `train/` models. **Strictly disjoint** from `behavioral-veriloga-eval/benchmark-vabench-release-v1/`.

## What "held-out" means here

The eval set in `../data/eval/` is constructed under these rules:

1. **Spec-level disjoint from training**: not just prompt-disjoint. Two prompts that ask for the same circuit by different wording count as overlapping.
2. **Includes an OOD slice**: at least one circuit category or one task family is absent from training (see `../PLAN.md` for the chosen OOD axis).
3. **Provenance audited**: every eval item has `vabench_audit.release_overlap == false`.

## Files (to be implemented in Phase 2-4)

| File | Role |
|---|---|
| `run_eval.py` | Evaluate a model checkpoint on `../data/eval/`, report metrics |
| `audit_report.py` | Verify a report contains all required fields (per `../KPI.md` Phase 4 gate) |
| `reports/` | Final reports, dated, with provenance |

## Metrics

| Metric | Definition | Target (Phase 4) |
|---|---|---|
| `compile_rate` | % of generated `.va` files passing EVAS Rust parser/elaboration | ≥ 85% |
| `sim_rate` | % that EVAS simulates without error, conditional on compile | ≥ 70% |
| `correct_rate` | % whose simulation matches reference within tolerance, conditional on sim | ≥ 50% |
| `compile_rate_OOD` | Same as compile_rate, restricted to held-out slice | ≥ 60% |
| `sim_rate_OOD` | Same as sim_rate, restricted to held-out slice | ≥ 40% |

## Reporting

Reports in `reports/<date>_<run-id>.md` must include:

- Top-line table of metrics with bootstrap 95% CIs.
- Per-task-family breakdown.
- OOD slice breakdown.
- Per-circuit-category breakdown.
- Comparison: SFT-only vs final GRPO; per-reward ablations.
- Floor: base model (no SFT, no RL) zero-shot.
- Contamination audit re-run with date and hash.
- Pointer to `../logs/<run-id>/` for reproduction.

## How to run (Phase 2+, sketch)

```
python -m train.eval.run_eval \
  --checkpoint ../models/grpo-run-N \
  --eval-set ../data/eval \
  --out reports/2026-MM-DD_grpo-N.md
```

## What this is NOT

- **Not a vaBench runner.** Do not invoke this on `behavioral-veriloga-eval/benchmark-vabench-release-v1/` without an explicit, separately-gated experiment per `SCOPE_BOUNDARY.md`.
- **Not a continuous benchmark.** Run on milestones (end of phase), not on every commit.
