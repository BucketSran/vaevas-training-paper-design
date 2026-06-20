# Brief

## What we are building

A local training subproject that takes `Qwen2.5-Coder-7B` and fine-tunes it on Verilog-A generation tasks via a **two-stage pipeline**: full-parameter SFT for cold start, followed by GRPO reinforcement learning with contract-driven rewards rooted in EVAS execution and audited against Spectre. The approach mirrors Circuit-Think (AAAI'26), adapted from circuit-image-to-netlist to vaBench-style behavioral Verilog-A tasks described by existing benchmark fields: `level`, `task_form`, and `category`.

## Why

1. **Test the Circuit-Think hypothesis on a structurally similar task.** vaBench has the same properties that made Circuit-Think work: a deterministic verifier (EVAS), a natural step-wise decomposition, and a strong base model. If the hypothesis holds, a 7B local model should out-perform much larger closed models on this narrow task.
2. **Get a local baseline stronger than prompt-only GPT-4o.** Useful as a future vaBench baseline and as a controllable research artifact.
3. **Build reusable training infrastructure** (data pipelines, reward functions, GRPO loop wired to EVAS) that the lab can apply to other domain-specific code tasks.

## Success Condition

The trained model satisfies **all** of the following on the held-out eval set defined in `train/data/eval/` (NOT vaBench):

1. **EVAS compile/elaboration rate** ≥ 85% on generated artifacts.
2. **EVAS simulation-health rate** ≥ 70% with required observables present and sane.
3. **Functional correctness** ≥ 50% by contract properties, with Spectre used as the audit oracle for reportable claims.
4. **Improvement over base model**: ≥ +25 percentage points absolute on at least one of compile / sim / correct, with the base being `Qwen2.5-Coder-7B` zero-shot.
5. **No contamination**: `train/pipelines/check_contamination.py` shows zero overlap between training data and vaBench release.

Acceptable failure mode: any of (1)-(3) below threshold but (4) and (5) hold — record as a partial success, document the gap, do not claim equivalence with stronger thresholds.

## Non-goals

- **Not a vaBench number generator.** Any vaBench evaluation done after training is a separate gated experiment, not the primary deliverable.
- **Not a production model.** No serving, no distillation, no quantization unless explicitly reopened.
- **Not a multi-modal model.** vaBench is text-only; image inputs are out of scope.
- **Not an algorithm-only paper.** The method claim is the verifier-grounded training system: clean-room contracts, LLM synthetic expansion, diagnostic rewards, and EVAS-as-audited accelerator. It is not a claim of a new RL algorithm.
- **Not a replacement for the vaBench paper line.** vaBench remains the primary research artifact; `train/` is a parallel exploration.

## Compatibility Constraints

- **vaEVAS mainline integrity**: `train/` MUST NOT modify anything under `behavioral-veriloga-eval/`, `EVAS/`, or `veriloga-skills/`. If `train/` needs an EVAS API change, file it as a separate vaEVAS-side issue first.
- **Contamination firewall**: see `SCOPE_BOUNDARY.md`. No exception.
- **Reproducibility**: every training run must produce a `logs/<run-id>/` directory with seed, config, dataset hash, base model commit, and reward implementation commit.

## Acceptance Metrics (what "done" looks like for each phase)

See `KPI.md`. Briefly:
- **Phase 0 (now)**: scaffold complete, scope docs read by the user, framework approved.
- **Phase 1**: 300+ EVAS-verified contract-backed data points available, contamination-audited.
- **Phase 2**: SFT runs end-to-end on the target GPU setup, produces a checkpoint whose generated artifacts pass EVAS compile/elaboration on ≥ 60% of held-out tasks.
- **Phase 3**: GRPO runs end-to-end with the diagnostic reward profile active; reward curves trend upward; final EVAS compile/elaboration ≥ 80%.
- **Phase 4**: held-out eval reports for compile / sim / correct with confidence intervals.

## Required Reusable Artifacts

By the end of this subproject:

1. `train/pipelines/synthesize.py` — generate candidate `<spec, va, tb>` triples from seed prompts.
2. `train/pipelines/verify_evas.py` — run candidates through EVAS, label pass/fail, capture diagnostics.
3. `train/pipelines/build_trajectory.py` — turn verified pairs into step-by-step CoT annotations.
4. `train/rewards/*.py` — reward modules importable from any future GRPO run.
5. `train/eval/run_eval.py` — held-out evaluation with provenance manifest.
6. `train/docs/01-06*.md` — SFT/GRPO/reward/trajectory/data/eval design notes that any future contributor can read in <1 hour.

## Residual Risks (as of scaffold time)

- **Data scarcity**: we may not be able to synthesize 1000+ high-quality EVAS-verified triples without significant effort. Mitigation: start with `EVAS/evas/examples/` (~30) + `veriloga-skills/` references, iterate.
- **Reward hacking**: GRPO will exploit reward weaknesses. Mitigation: keep compile/sim/property gates, anti-hack penalties, Spectre shadow audits, and reward ablations active.
- **Base model fit**: `Qwen2.5-Coder-7B` may already saturate on simple Verilog-A; the headroom for RL may be small on easy tasks. Mitigation: stratify eval by difficulty.
- **EVAS coverage gaps**: any task that EVAS cannot simulate cannot be used as RL training (no reward signal). Mitigation: enumerate the supported subset in `train/docs/05_data_pipeline.md`.
