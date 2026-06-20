# 06 — Eval Protocol

How we measure model quality. **Disjoint from vaBench by construction.**

## Three eval layers

1. **Sanity eval (during SFT)** — small, fast; runs after every epoch
2. **Held-out eval (end of phase)** — `data/eval/` full sweep; produces reports
3. **OOD probe** — subset of held-out, models a circuit category or task form the training set didn't see

## What we measure

| Metric | Formula | Where it comes from |
|---|---|---|
| `compile_rate` | (# items where EVAS compile/elaboration succeeds) / total | `pipelines/verify_evas.py` re-used at eval time |
| `sim_rate` | (# items where EVAS sim succeeds) / total | same |
| `correct_rate` | (# items where R_metric ≥ threshold) / total | `rewards/metric_reward.py` |
| `trajectory_step1` ... | per-step trajectory reward, averaged | `rewards/trajectory_reward.py` |
| Conditional rates | sim_rate given compile, correct given sim | derived |

## Conditional reporting matters

Don't report just `correct_rate`. Report the **funnel**:

```
N = 50 held-out items
  ├─ compile:    42 / 50  (84%)
  │   ├─ sim:        35 / 42  (83% conditional)
  │   │   └─ correct:  22 / 35  (63% conditional)
  │   │       ⇒ end-to-end correct: 22 / 50 = 44%
```

This is much more informative than a single 44% number — it tells you whether the bottleneck is compile, sim, or correctness.

## Statistical reporting

For small N (50-100 items), point estimates are misleading. Report **95% bootstrap CIs**:

```
compile_rate = 84% [95% CI 72%, 92%]
```

Bootstrap: resample with replacement 10000 times, take 2.5% and 97.5% percentiles. `numpy` one-liner.

## Stratification

Always report stratified:
- By `task_form`
- By `level`
- By circuit category (comparator / dac / sar / dwa / etc)
- By difficulty if labeled

Stratification reveals where the model fails — important for next-iteration decisions.

## OOD probe

The OOD axis is chosen in Phase 1 (see `PLAN.md` decision #2). Options:
- **Task-form-held-out**: e.g., train without any `tb-generation`, test on `tb-generation`
- **Circuit-held-out**: e.g., train without any comparators, test on comparators
- **Difficulty-held-out**: e.g., train on easy/medium, test on hard

Default if undecided: **circuit-held-out** with 1-2 categories withheld. Why: most realistic generalization probe; family-held-out is too harsh, difficulty-held-out is too easy.

Report OOD metrics SEPARATELY from in-distribution metrics. Do not collapse them into one average.

## Comparison baselines

Every full eval report must include:

| Baseline | What it measures |
|---|---|
| `Qwen2.5-Coder-7B` zero-shot | Floor; no training |
| SFT-only checkpoint | What SFT alone achieves |
| Final SFT+GRPO checkpoint | Our system |
| GPT-4-class zero-shot (optional) | Closed-model floor |

The headline number is **(final SFT+GRPO) minus (SFT-only)** on `correct_rate`. That delta is the value-add of the RL stage.

## Ablations (required for Phase 4 report)

| Ablation | Removes | Expected impact |
|---|---|---|
| `--no-trajectory-reward` | R_trajectory + step gating | Should drop accuracy substantially (Circuit-Think saw 73→35) |
| `--no-reflective` | reflective learning mechanism | Smaller drop, expected 5-10 pp |
| `--no-r-metric` | R_metric (only compile/sim) | Should show model produces compileable garbage |
| `--no-step-gating` | gating, R_trajectory still computed | Smaller drop than no-trajectory |
| `--single-reward (compile-only)` | Just R_compile | Worst case; confirms multi-reward is needed |

## Sample inspection

For every eval run, save 10 random samples (correct), 10 random samples (incorrect), 10 random samples (didn't compile) to `eval/reports/<date>/samples/`. Human inspection is the best bug-finder.

## Reporting checklist

A Phase 4 report at `eval/reports/<date>_final.md` MUST contain:

- [ ] Headline metrics table with CIs
- [ ] Per-task-form stratification
- [ ] Per-circuit-category stratification
- [ ] OOD probe results (separately)
- [ ] Comparison baselines (zero-shot + SFT-only)
- [ ] Ablation table
- [ ] Contamination audit re-run with date
- [ ] Sample completions (10 + 10 + 10)
- [ ] Pointer to `../logs/<run-id>/`
- [ ] Reproduction command (`run_eval.py --replay ...`)

If any item is missing, the report is not done.

## What we do NOT do

- **Cherry-pick best samples for the headline number.**
- **Average over a single run.** Always report the bootstrap CI.
- **Report a single accuracy without the funnel.**
- **Compare against an undefined baseline.** Every comparison needs an explicit baseline source.
- **Run on vaBench tasks** unless explicitly cleared per `SCOPE_BOUNDARY.md`.
