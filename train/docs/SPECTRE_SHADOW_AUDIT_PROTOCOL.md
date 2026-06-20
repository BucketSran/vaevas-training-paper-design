# Spectre Shadow Audit Protocol

This document defines how the training track uses Spectre to audit EVAS-based
training rewards. It is a Phase 0 design artifact. Implementation belongs to
Phase 1+.

## Purpose

EVAS Rust is the fast training-time evaluator. Spectre is the reference oracle
for reportable correctness claims. The shadow audit protocol makes that split
explicit:

- EVAS may drive SFT filtering, GRPO reward, debugging, and speed measurements.
- Spectre must audit EVAS judgments before paper-facing correctness claims.
- EVAS PASS / Spectre FAIL is a zero-tolerance false-positive condition.
- Spectre PASS / EVAS FAIL is EVAS false-negative debt and is repaired in
  scheduled batches rather than ad hoc during every rollout.

This protocol is not optional for any experiment that claims EVAS-accelerated
training or model correctness.

## Scope

Audited units are not only final model completions. The audit covers:

- contract examples,
- generated DUT/testbench/checker artifacts,
- SFT training data admission batches,
- GRPO reward calibration samples,
- checkpoint completions during training,
- final held-out evaluation items,
- repair examples derived from failed attempts.

The protocol is independent of vaBench scoring. `train/` uses its own held-out
data and contamination firewall.

## Terminology

| Term | Meaning |
| --- | --- |
| EVAS PASS | EVAS compile/elaboration, simulation health, and required contract properties pass under the active reward profile. |
| EVAS FAIL | EVAS rejects the artifact or any required gate fails. |
| Spectre PASS | Spectre simulation plus the same observable property checks pass within declared tolerances. |
| Spectre FAIL | Spectre fails to run, produces invalid observables, or violates the same properties. |
| Shadow audit | Running Spectre on a selected EVAS-evaluated slice and comparing results. |
| Audit round | One scheduled batch of Spectre checks with fixed input manifest and output report. |
| False positive | EVAS PASS / Spectre FAIL. Dangerous because reward may train the model toward simulator-specific hacks. |
| False negative | Spectre PASS / EVAS FAIL. Costly because EVAS rejects valid examples and weakens training coverage. |

## Non-Negotiable Rules

1. **No paper-facing correctness claim from EVAS-only evidence.**
2. **No unresolved EVAS PASS / Spectre FAIL in the claimed slice.**
3. **Every audit item must trace to contract, artifact hashes, reward profile,
   EVAS version, Spectre version, checker version, and tolerance profile.**
4. **Spectre PASS / EVAS FAIL cases are batched into an EVAS false-negative
   backlog and repaired after fixed audit intervals.**
5. **Do not change checker thresholds to hide simulator disagreement.**
   Threshold calibration is allowed only after waveform semantics agree and the
   residual is numerical/windowing tolerance.

## Audit Layers

### Layer A: Contract Admission Audit

Run before a contract group becomes a source for generated training data.

Minimum slice:

- at least one item per new `level`,
- at least one item per new `task_form`,
- at least one item per new `category`,
- every new checker/property type,
- every new simulator feature class in `allowed_features`.

Admission requirement:

- no EVAS PASS / Spectre FAIL;
- documented Spectre PASS / EVAS FAIL cases either fixed or marked out of
  reward scope until fixed;
- property tolerances documented in the contract.

### Layer B: Pre-Training Parity Gate

Run after the first verified data batch and before GRPO scaling.

Purpose:

- calibrate reward profiles,
- verify that EVAS reward gates align with Spectre on the selected subset,
- catch checker/windowing mistakes before they become RL incentives.

Suggested pilot minimum:

- 30 total items, if available;
- every active reward profile represented;
- all L2 items included until at least 20 L2 items have passed audit;
- all new checker/property types included.

Scaling rule:

- For small batches (`N <= 100`), audit at least 30% and all high-risk items.
- For larger batches, audit a stratified slice with a floor of 50 items or 10%,
  whichever is larger, until the mismatch rate is stable.

### Layer C: During-Training Shadow Audit

Run on checkpoint completions while GRPO is active.

Audit rounds should include:

- the SFT checkpoint before GRPO,
- the first GRPO checkpoint after reward becomes nonzero,
- fixed interval checkpoints,
- any checkpoint where reward distribution shifts sharply,
- any checkpoint selected for a reported result.

Suggested pilot schedule:

| Stage | Audit trigger | Minimum slice |
| --- | --- | ---: |
| SFT baseline | before GRPO | 30 completions |
| GRPO warmup | first nonzero reward plateau | 30 completions |
| GRPO interval | every 25-50 optimizer steps in pilot | 30 completions |
| Reward anomaly | reward spike, entropy collapse, or KL spike | 50 completions |
| Candidate final | before final evaluation | all headline eval items or a declared audited subset |

Sampling mix per round:

| Bucket | Share | Why |
| --- | ---: | --- |
| High EVAS reward / EVAS PASS | 40% | Detect reward false positives. |
| Borderline property score | 20% | Calibrate tolerances and windows. |
| EVAS FAIL with localized diagnostics | 15% | Discover EVAS false negatives and repair opportunities. |
| L2 and new categories | 15% | Protect the hardest and newest slices. |
| Random stratified sample | 10% | Avoid only auditing known-risk examples. |

If a bucket is unavailable, redistribute its share to L2/new categories and
random stratified sampling.

### Layer D: Final Held-Out Spectre Evaluation

Paper-facing correctness must be Spectre-backed.

Preferred rule:

- run Spectre on every held-out item used in the headline table.

Fallback if Spectre cost is prohibitive:

- report EVAS metrics as development metrics only;
- separately report the Spectre-audited subset with its sampling method;
- do not call the EVAS-only number final correctness.

## Mismatch Matrix

| Case | Meaning | Required action |
| --- | --- | --- |
| EVAS PASS / Spectre PASS | Trusted under current tolerance. | Keep candidate and reward profile. |
| EVAS PASS / Spectre FAIL | False positive. | Quarantine affected contract, checker, category, reward profile, and checkpoint slice. Stop claims until repaired and re-audited. |
| Spectre PASS / EVAS FAIL | False negative. | Add to EVAS false-negative backlog. Repair after scheduled audit interval. Do not silently lower quality or broaden reward acceptance. |
| EVAS FAIL / Spectre FAIL | Invalid artifact or model failure. | Keep as failed candidate or repair data if diagnostics are useful. |

## False-Positive Triage

For every EVAS PASS / Spectre FAIL, record one primary cause:

| Cause | Examples | Owner |
| --- | --- | --- |
| EVAS semantic gap | scheduling, event crossing, source update, unsupported operator mismatch | EVAS compatibility backlog |
| Checker bug | property uses wrong observable, wrong window, wrong polarity | checker/property owner |
| Contract ambiguity | spec permits behavior that checker rejects | contract owner |
| Spectre harness issue | include path, discipline, simulator option, timeout | audit harness owner |
| Reward exploit | hardcoded constants, hidden checker bypass, artificial delays | reward owner |

Required remediation:

1. distill a minimal reproducer if EVAS semantics are suspected;
2. fix EVAS/checker/contract/reward at the general rule level;
3. rerun EVAS and Spectre on the quarantined slice;
4. only then unquarantine the affected slice.

Do not special-case task IDs, model names, category names, or checkpoint IDs in
EVAS core or reward code.

## False-Negative Backlog

Spectre PASS / EVAS FAIL is not an emergency stop unless it dominates a slice,
but it cannot be ignored.

Batching rule:

- collect false negatives for one audit interval;
- group by diagnostic class and simulator feature;
- repair the most general EVAS semantic class first;
- rerun the full backlog after repair;
- record residual cases as unsupported scope if they are intentionally out of
  EVAS coverage.

Training policy:

- do not admit false-negative examples as EVAS-verified training data until EVAS
  can score them correctly;
- they may be stored as `spectre_only_candidate` or EVAS regression fixtures;
- report the number of blocked but Spectre-valid cases as training coverage
  loss.

## Audit Manifest

Each audit round writes a manifest before execution and a report after
execution.

### Input manifest

```yaml
audit_id: spectre_shadow_2026mmdd_round001
created_at: 2026-MM-DDTHH:MM:SSZ
purpose: pre_training_parity | grpo_interval | final_eval | anomaly
selection_policy:
  sample_seed: 1234
  buckets:
    high_evas_reward: 12
    borderline_property: 6
    evas_fail_localized: 5
    l2_new_category: 5
    random_stratified: 2
versions:
  evas_commit: <git-sha>
  reward_commit: <git-sha>
  checker_version: <semver-or-sha>
  spectre_version: <version-string>
  cadence_host_profile: <profile-id>
items:
  - item_id: <completion-or-artifact-id>
    contract_id: <contract-id>
    artifact_hashes:
      prompt_sha256: <sha256>
      completion_sha256: <sha256>
      dut_sha256: <sha256-or-null>
      tb_sha256: <sha256-or-null>
      checker_sha256: <sha256-or-null>
    classification:
      level: L1
      task_form: dut
      category: Comparator and Decision Circuits
    reward_profile: L1_dut
    evas_result_ref: <path>
    expected_properties: [prop_delay, hysteresis_width]
```

### Output report

```yaml
audit_id: spectre_shadow_2026mmdd_round001
status: pass | fail | quarantined | infra_failed
summary:
  total: 30
  evas_pass_spectre_pass: 24
  evas_pass_spectre_fail: 0
  spectre_pass_evas_fail: 3
  both_fail: 3
  infra_fail: 0
timing:
  evas_wall_time_s_sum: 12.4
  spectre_wall_time_s_sum: 845.0
  same_slice_speedup_sum: 68.1
mismatches:
  - item_id: <id>
    case: spectre_pass_evas_fail
    primary_cause: evas_false_negative_pending
    action: backlog
    owner: evas_compat
    due_audit_round: spectre_shadow_2026mmdd_round002
claim_gate:
  correctness_claim_allowed: true
  speed_claim_allowed: true
  notes: []
```

## Metrics

Report per audit round and cumulatively:

- `false_positive_count`: EVAS PASS / Spectre FAIL.
- `false_negative_count`: Spectre PASS / EVAS FAIL.
- `pass_agreement_rate`: both pass divided by total audited items.
- `decision_agreement_rate`: same binary decision divided by total audited items.
- `spectre_infra_fail_rate`: Spectre infrastructure failures divided by total.
- `same_slice_speedup_sum`: sum Spectre wall time divided by sum EVAS wall time.
- `median_per_item_speedup`, plus p25/p75.
- mismatch counts by `level`, `task_form`, `category`, checker type, and reward profile.

Do not hide infra failures in mismatch counts. Report them separately.

## Claim Gates

| Claim | Required audit evidence |
| --- | --- |
| EVAS reward is a trusted proxy on a slice | Latest audit round has zero EVAS PASS / Spectre FAIL for that slice. |
| EVAS accelerates training | Same-slice EVAS and Spectre wall times, same artifacts, same properties, zero unresolved false positives. |
| Final model correctness | Spectre-backed held-out evaluation or explicitly labeled Spectre-audited subset. |
| Diagnostic reward improves behavior | Reward ablation results plus shadow audit showing the improvement is not EVAS-specific hacking. |
| L2 improvement | L2-specific Spectre audit coverage and no unresolved L2 false positives. |

## Stop Conditions

Pause scaling or strong claims when any condition occurs:

- any EVAS PASS / Spectre FAIL in a claimed slice;
- false positives repeat in the same checker/property type;
- a reward profile has high EVAS reward but Spectre failure concentration;
- Spectre infra failures exceed 10% in an audit round;
- audit manifest cannot reproduce EVAS or Spectre inputs;
- artifact hashes or contract IDs are missing.

Stop conditions block claims, not necessarily all local debugging. Debug runs may
continue if clearly labeled as untrusted.

## Integration With Training

SFT:

- use Spectre-audited data for high-value seeds and all L2 seeds;
- use EVAS-only verified data only as provisional data until audit coverage is
  sufficient;
- preserve audit refs in packed SFT metadata.

GRPO:

- EVAS computes reward online;
- Spectre audits checkpoint completions offline;
- reward profiles are frozen within each audit interval;
- if false positives appear, quarantine the profile/slice and rerun ablations
  after repair.

Repair:

- Spectre PASS / EVAS FAIL examples become EVAS compatibility tests;
- EVAS FAIL / Spectre FAIL examples can become repair prompts if the failure is
  model-caused and diagnostics are useful;
- EVAS PASS / Spectre FAIL examples are not training positives.

## Implementation Roadmap

Phase 1 should add scripts only after this protocol is accepted:

1. `pipelines/sample_spectre_shadow.py`: build input manifests.
2. `pipelines/run_spectre_shadow.py`: execute Spectre on manifest items.
3. `pipelines/compare_evas_spectre.py`: normalize results and emit mismatch report.
4. `pipelines/update_evas_false_negative_backlog.py`: batch false negatives.
5. `eval/audit_report.py`: enforce claim gates in final reports.

No script should infer missing provenance. Missing provenance is a hard failure.

## Paper Reporting Template

For every reported training run:

```text
EVAS was used for online reward computation. Spectre shadow audit was run on
<N> selected completions across <levels/forms/categories>. The audited slice had
<FP> EVAS PASS / Spectre FAIL false positives and <FN> Spectre PASS / EVAS FAIL
false negatives. Headline correctness is reported on <all held-out items | the
declared Spectre-audited subset>. EVAS/Spectre same-slice timing showed <speed>
speedup under the recorded host and Cadence configuration.
```

If `<FP> > 0`, do not write a headline correctness or speed claim for that
slice.

## Open Decisions Before Phase 1

- Exact pilot audit budget in Spectre wall-clock hours.
- Which host profile is the stable Spectre audit environment.
- Whether final held-out Spectre evaluation is exhaustive or an explicitly
  stratified audited subset.
- Minimum L2 audit coverage before L2 GRPO scaling.
