# Phase 1 Parameter Decisions

Status: accepted baseline, 2026-06-20.

This document fixes the Phase 1-v0.1 parameters for the clean-room data factory.
It replaces the earlier ambiguous "300 examples" target with a pilot target and
a scale-up target.

## Core Interpretation

Phase 1 data is **not** vaBench benchmark data. The existing vaBench release is
reserved for benchmark evaluation and separately gated frozen-model probes.

Phase 1 builds a new clean-room dataset:

- contract-first,
- contamination-audited,
- EVAS-verified,
- Spectre-shadow-audited,
- split into train/validation/internal held-out eval.

## Accepted Targets

| Quantity | Phase 1 pilot | Phase 1 scale-up | Why |
| --- | ---: | ---: | --- |
| Seed contracts | 150-250 | 250-500 | Avoid deriving thousands of samples from a tiny template pool. |
| Generated candidates | 3k-6k | 8k-15k | Candidate generation is cheap relative to verification and filtering. |
| EVAS-passing candidates | 1.5k-3k | 4k-8k | Provides enough material for filtering, diversity, and repair data. |
| Admitted clean-room data | 1k-2k | 2k-3k minimum | Sufficient for SFT pilot without leaning on vaBench. |
| SFT train split | 800-1600 | 1600-2400 | Main imitation-learning set. |
| Validation split | 100-200 | 200-300 | Used for training diagnostics and early sanity checks, not final claims. |
| Clean-room eval split | 100-300 | 300-500 | Internal held-out set for training-paper evaluation. |
| OOD eval subset | 50-100 | 100-200 | Tests generalization beyond in-distribution templates. |

The pilot is the first successful end-to-end factory run. The scale-up target is
the minimum data volume for serious SFT and later GRPO experiments.

## Distribution Targets

Initial admitted-data mix:

| Axis | Target |
| --- | --- |
| `level` | L0 5-10%, L1 60-70%, L2 20-30% |
| `task_form` | `dut` 35-45%, `tb` 15-25%, `bugfix` 15-25%, `e2e` 10-20%, `conformance` 3-8% |
| `category` | No single category above 25% in the admitted pilot. |
| `split_key` | No split key shared across train/validation/eval. |
| checker/property type | Every promoted checker type must have Spectre audit coverage. |

Rationale:

- L1 remains the core training surface.
- L2 must be large enough for cold-start SFT before L2-heavy GRPO.
- L0 is useful for simulator-scope and conformance behavior but should not
  dominate training.
- End-to-end tasks are important, but too many early L2/e2e examples make reward
  sparse before the model has learned stable format and interfaces.

## OOD Definition

OOD means **out-of-distribution relative to the newly constructed clean-room
training set**, not relative to vaBench.

Accepted Phase 1 OOD axes:

1. **Circuit-category held-out**: reserve one or more benchmark-aligned
   categories from SFT/GRPO training and use them only in clean-room eval.
2. **L2-hard held-out**: reserve harder composed flows for evaluation after the
   model sees easier L2 or L1 components.
3. **Parameter-range held-out**: train on one parameter region, evaluate on a
   non-overlapping but physically meaningful region.

Default Phase 1 choice:

- primary OOD: circuit-category held-out;
- secondary OOD: L2-hard held-out if enough L2 contracts exist.

Do not use full `task_form` held-out as the first OOD setting. If the model never
sees a task format, failure may reflect missing format learning rather than
circuit-function generalization.

## Spectre Audit Budget

Spectre audit is part of data admission, not an afterthought.

Accepted pilot audit policy:

| Slice | Audit policy |
| --- | --- |
| Pre-training admitted pilot | Audit `max(100, 20%)` of admitted items. |
| L2 admitted items | Audit at least 50%; audit all L2 items until 50 audited L2 passes exist. |
| New checker/property type | Audit all first 10 admitted examples. |
| New category | Audit at least 10 examples or all examples if fewer than 10. |
| High EVAS reward / borderline tolerance | Oversample in each audit round. |
| Final clean-room eval headline | Prefer all Spectre; otherwise report a declared Spectre-audited subset. |

Red lines:

- EVAS PASS / Spectre FAIL tolerance is 0 in any claimed slice.
- Spectre PASS / EVAS FAIL goes to the EVAS false-negative backlog and is fixed
  after scheduled audit intervals.
- EVAS-only results are development metrics unless the declared audit gate
  licenses the slice.

## Contamination Threshold Policy

Initial policy is conservative:

| Signal | Action |
| --- | --- |
| Exact release prompt/code/checker overlap | Reject. |
| Release task ID or gold path in lineage | Reject. |
| High prompt/spec similarity | Needs review. |
| Identifier-blind code similarity | Needs review or reject, depending on strength. |
| Same `split_key` across train/eval | Block. |
| Missing provenance | Quarantine. |
| Parser/index failure | Quarantine. |

The first implementation should prefer false blocks over false clean admissions.
Data volume can be recovered through more synthetic generation; contaminated
training cannot be recovered without retraining.

## Split Policy

For the pilot admitted set:

| Split | Target |
| --- | ---: |
| SFT train | 80% |
| Validation | 10% |
| Clean-room eval | 10% |

For scale-up, preserve at least:

- 300 clean-room eval items,
- 100 OOD eval items,
- no `split_key` overlap across splits,
- no near-duplicate prompt/spec/code across train and eval.

## Admission Gates

An item is admitted only if:

1. provenance is complete;
2. contract schema validates;
3. EVAS compile/elaboration passes;
4. EVAS simulation health passes;
5. required contract properties pass under EVAS;
6. contamination checker returns clean;
7. split-leakage check passes;
8. Spectre audit exists if required by this policy;
9. diversity filter does not reject the item as template duplication.

Failed candidates may be retained only as diagnostics, repair negatives, or EVAS
false-negative backlog entries with explicit state labels.

## Implementation Consequences

Phase 1 scripts should be implemented in this order:

1. manifest schemas for protected index, candidate index, contamination report,
   and Spectre shadow report;
2. `check_contamination.py`;
3. safe seed catalog builder;
4. contract-conditioned LLM synthesis;
5. EVAS verification;
6. Spectre shadow sampling and report comparison;
7. diversity filtering;
8. packers for SFT/RL/eval splits.

Do not start bulk generation before the protected index and contamination report
format exist.

## Decisions Still Deferred

- Exact held-out categories.
- Exact numeric similarity thresholds after the first protected-index dry run.
- Which Spectre host profile is the official audit environment.
- Whether final headline eval is all-Spectre or a declared Spectre-audited
  subset due to wall-clock cost.
