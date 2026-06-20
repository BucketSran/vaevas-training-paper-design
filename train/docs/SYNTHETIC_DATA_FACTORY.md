# Contract-First Synthetic Data Factory

Status: design draft. This document defines the data factory protocol; it does
not claim that any data has been generated or admitted.

This factory is the clean-room data construction route for the training paper.
It turns audited safe seeds into task contracts, generated artifacts, verifier
evidence, SFT examples, GRPO prompts, and repair examples without relying on
historical experiment outputs.

## Purpose

The factory exists to generate enough diverse, verified Verilog-A training data
for SFT + GRPO while avoiding three failure modes:

1. **Contamination**: synthetic data must not paraphrase or leak vaBench release
   tasks.
2. **Self-consistent hallucination**: an LLM must not be allowed to generate a
   spec, DUT, testbench, and checker that agree with each other but test the
   wrong behavior.
3. **Template collapse**: thousands of examples that differ only by names or
   constants should not be counted as broad training data.

## Core Principle

Synthetic generation starts from a contract, not from a free-form prompt:

```text
safe seed
  -> task contract
  -> independent artifact proposals
  -> EVAS verification
  -> Spectre shadow audit
  -> diversity/contamination gates
  -> admitted SFT / GRPO / repair data
```

The LLM is a proposer. The contract, independent checks, EVAS, Spectre audit,
and diversity filters decide whether an example is admitted.

## Inputs

### Safe seed catalog

Allowed seed tiers:

| Tier | Source | Use |
| --- | --- | --- |
| A | EVAS examples, `veriloga-skills`, manual specs, public source material with license/provenance | Primary seed source. |
| B | LLM-generated contracts derived from Tier A abstractions | Main expansion source after checks. |
| C | Controlled mutations of clean verified contracts | Repair data and robustness source. |
| D | Historical outputs or benchmark-adjacent artifacts | Excluded by default; may inform taxonomy only unless re-audited item by item. |

Every seed catalog entry should contain:

```yaml
seed_id: seed_manual_cmp_hysteresis_0001
source_tier: A_safe_seed
source_kind: manual
source_ref: null
license: internal_clean_room
allowed_uses: [contract_proposal, artifact_proposal]
forbidden_uses: [copy_release_prompt, copy_release_gold]
notes: "Manual clean-room behavioral intent, not copied from vaBench release."
```

### Contract schema

The canonical schema is `train/data/contracts/schema.yaml`.

Contracts must use:

- `category`: vaBench circuit-role taxonomy,
- `level`: `L0`, `L1`, or `L2`,
- `task_form`: `dut`, `tb`, `bugfix`, `e2e`, or `conformance`.

Do not use `family` to mean circuit category.

## Factory Stages

### Stage 0: Batch plan

Before generating examples, define a batch plan:

```yaml
batch_id: synth-batch-0001
target_counts:
  L0_conformance: 20
  L1_dut: 120
  L1_tb: 40
  L1_bugfix: 80
  L2_e2e: 30
categories:
  Comparator and Decision Circuits: 80
  Sampling and Analog Memory: 60
  Data Converter Models: 60
  Measurement Instrumentation Flows: 40
ood_holdout:
  split_key_prefixes: [data_converter/sar_adc_mini_loop]
admission_policy:
  require_evas: true
  require_spectre_shadow_for:
    - high_reward_samples
    - L2_e2e
    - new_property_type
```

The batch plan prevents opportunistic generation from drifting toward easy
templates.

### Stage 1: Contract proposal

Input:

- safe seed catalog entry,
- target `category`,
- target `level`,
- target `task_form`,
- allowed EVAS/Spectre scope,
- diversity constraints.

Output:

- draft contract YAML matching `schema.yaml`.

Prompt requirements:

- Ask for observable behavior, not implementation.
- Require `properties` before artifacts.
- Require `split_key`.
- Require `forbidden_constructs`.
- Prohibit copying or paraphrasing known vaBench release wording.
- Prohibit hidden checker logic.

Reject contract if:

- it lacks measurable properties,
- it depends on current-domain/KCL/KVL/device-level behavior,
- it collapses to an L0 primitive while labeled L1/L2,
- it duplicates an existing `split_key`,
- it overlaps release prompt/gold fingerprints,
- it cannot be audited by EVAS/Spectre within current scope.

### Stage 2: Contract review

Review can be automated first, then manually sampled.

Automated checks:

- YAML parses.
- Required fields exist.
- `category`, `level`, and `task_form` are valid.
- L2 has decomposed subclaims and a system-level metric.
- `bugfix` has `fault_model`.
- `tb` names a reference DUT interface in `generation_plan`.
- `split_key` is unique or intentionally grouped.
- forbidden constructs are present and relevant.

Manual/sample checks:

- Is the behavior a real analog/mixed-signal behavioral modeling task?
- Are properties sufficient to catch common wrong implementations?
- Is the contract too close to vaBench release wording or gold behavior?
- Is the contract useful for SFT, GRPO, repair, or audit?

Admission states:

```text
draft_contract -> ready_for_generation -> rejected
```

Only `ready_for_generation` contracts may proceed to artifact proposal.

### Stage 3: Multi-view artifact proposal

Generate artifacts from the contract using separate views. Do not let one model
call generate everything in one self-consistent bundle unless the batch is only
for scratch diagnostics.

Recommended views:

| View | Produces | Should not see |
| --- | --- | --- |
| DUT proposer | DUT `.va` or modules | Checker implementation details. |
| Testbench proposer | `.scs` or equivalent EVAS/Spectre harness | Gold DUT internals beyond public interface. |
| Checker/property proposer | Property extraction/checker code or checker config | DUT implementation details. |
| Fault proposer | Broken artifact mutations for repair | Fixed patch. |
| Critic/reviewer | Rejection reasons and missing properties | None, but cannot admit data alone. |

For high-value L2 data, use at least two independent proposals:

- one for the system/DUT,
- one for checker/testbench,
- then a reviewer that only compares them against the contract.

### Stage 4: Artifact assembly

Package candidate artifacts into a staging record:

```yaml
candidate_id: cand_cg_l1_hysteresis_comparator_dut_0001_a03
contract_id: cg_l1_hysteresis_comparator_dut_0001
artifact_hashes:
  dut_va: sha256:...
  tb_scs: sha256:...
  checker_yaml: sha256:...
proposal:
  dut_model: null
  tb_model: null
  checker_model: null
  prompt_hashes: []
status: generated
```

Keep failed candidates. They are useful for reward calibration and repair data,
but they are not admitted as SFT gold.

### Stage 5: Static and contract checks

Run cheap checks before EVAS:

- artifact set matches `task_form`,
- module/port/parameter names match contract,
- forbidden constructs are absent,
- no benchmark task IDs or checker-note strings,
- no hardcoded expected outputs tied to a fixed sweep,
- source size and module count are within limits,
- checker uses public observables only.

Outcomes:

| Outcome | Action |
| --- | --- |
| Pass | Proceed to EVAS. |
| Soft fail | Keep as negative/reward calibration candidate. |
| Hard exploit | Reject and add anti-hack pattern. |

### Stage 6: EVAS verification

EVAS is the fast verifier used for scale.

Required outputs:

- compile/elaboration status,
- simulation status,
- required observables present,
- trace health summary,
- property scores,
- runtime and timeout status,
- EVAS commit/profile,
- checker/property version.

Candidate states:

```text
generated -> static_checked -> evas_verified -> rejected
```

EVAS failure does not automatically discard the candidate. It may become:

- repair prompt,
- reward calibration negative,
- EVAS false-negative candidate if Spectre later passes.

### Stage 7: Spectre shadow audit

Detailed protocol: `SPECTRE_SHADOW_AUDIT_PROTOCOL.md`.

Spectre is not required for every candidate during pilot generation, but it is
required for claims and for high-risk slices.

Always prioritize Spectre audit for:

- high EVAS-reward candidates,
- L2 candidates,
- new property/checker types,
- new categories,
- candidates near tolerance boundaries,
- repair examples with high `R_repair_delta`,
- any category or `split_key` slice with recent EVAS/Spectre mismatch.

Mismatch handling:

| Case | Action |
| --- | --- |
| EVAS PASS / Spectre PASS | Candidate may become admitted evidence. |
| EVAS PASS / Spectre FAIL | Quarantine the affected category, `split_key`, contract, checker, and artifact slice; repair EVAS/checker before claims. |
| Spectre PASS / EVAS FAIL | Add to EVAS false-negative backlog; do not admit as EVAS-verified training data until policy says how to use it. |
| Both fail | Keep only as failed candidate or repair input if useful. |

### Stage 8: Contamination check

Detailed specification: `CONTAMINATION_CHECKER_SPEC.md`.

Run contamination checks before any candidate enters SFT/GRPO/eval.

Minimum checks:

- exact prompt/code overlap,
- normalized n-gram overlap,
- module/port signature overlap,
- AST/fingerprint similarity where available,
- `split_key` and seed lineage review.

Reject if:

- source resembles vaBench release prompt or gold code,
- only superficial paraphrasing separates it from release material,
- provenance is missing,
- seed lineage touches forbidden release assets.

### Stage 9: Diversity filtering

Do not admit examples only because they pass verification. Enforce diversity.

Track at least:

- category counts,
- level counts,
- task_form counts,
- base_function counts,
- split_key counts,
- property type counts,
- port signature clusters,
- AST/template fingerprints,
- parameter-space coverage,
- error class counts for repair data.

Batch admission should include a diversity report:

```yaml
diversity_report:
  examples_total: 1200
  unique_split_keys: 140
  max_examples_per_split_key: 5
  category_entropy: 2.1
  task_form_counts: {dut: 120, tb: 40, bugfix: 80, e2e: 30, conformance: 30}
  property_type_counts: {threshold_crossing: 44, event_sequence: 31}
  ast_duplicate_rate: 0.04
```

The exact metrics can change, but the report must exist.

### Stage 10: Packing

Admitted candidates produce different outputs depending on use:

| Output | Source | Contents |
| --- | --- | --- |
| SFT direct generation | verified clean candidates | prompt/contract summary + gold artifact completion. |
| SFT feedback repair | mutation/on-policy repair candidates | broken artifact + feedback -> fixed artifact. |
| GRPO prompt pool | contracts and harness metadata | prompt/contract without target completion. |
| Eval set | held-out split contracts/candidates | never used for synthesis prompt iteration or reward tuning. |

Packing must preserve:

- contract ID,
- candidate ID,
- split key,
- source tier,
- artifact hashes,
- verifier evidence refs,
- contamination check refs,
- reward/checker version.

## Prompt Families

Prompt templates should be tracked and hashed. The exact wording can evolve, but
these roles should remain separate.

### Contract proposer prompt

Purpose: propose a measurable contract from a safe seed.

Must include:

- target category/level/task_form,
- EVAS scope constraints,
- required property types,
- forbidden constructs,
- split and provenance requirements.

Must not include:

- vaBench release prompts,
- gold code from release tasks,
- checker internals from held-out eval.

### DUT proposer prompt

Purpose: propose implementation artifacts from a contract.

Must include:

- public contract fields,
- required ports/parameters,
- forbidden constructs,
- output artifact format.

Must not include:

- checker implementation,
- hidden expected values,
- eval split identities.

### Testbench/checker proposer prompt

Purpose: propose independent verification artifacts.

Must include:

- public interface,
- stimulus_space,
- observables,
- properties,
- tolerances.

Must not include:

- DUT implementation details beyond public behavior,
- hidden shortcuts tied to one generated DUT.

### Fault proposer prompt

Purpose: create repair examples.

Allowed fault classes:

- syntax/local declaration errors,
- port/interface mismatch,
- wrong threshold sign,
- wrong event direction,
- missing reset/hold behavior,
- wrong delay/aperture timing,
- unsupported construct insertion,
- wrong parameter unit/scale,
- incomplete L2 connection,
- incorrect state/sequence update.

Faults must be realistic and tied to `fault_model`.

## Admission Matrix

| Candidate state | May enter SFT gold | May enter GRPO prompts | May enter repair data | May enter eval |
| --- | --- | --- | --- | --- |
| Draft contract | No | No | No | No |
| Ready contract | No | Yes, only for rollout pilots | No | No |
| Generated, unverified | No | No | Possible negative only | No |
| EVAS verified | Possible | Yes | Yes | No unless split/audit complete |
| Spectre shadow verified | Yes | Yes | Yes | Possible |
| Contamination unclear | No | No | No | No |
| Historical, not re-audited | No | No | No | No |

## Clean-Room Rules

- Do not copy from `behavioral-veriloga-eval/benchmark-vabench-release-v1/`.
- Do not use release prompts as seeds.
- Do not paraphrase release prompts.
- Do not reuse release gold code as templates.
- Do not use old experiment outputs as training data unless re-audited item by
  item.
- Do not tune generation prompts on final held-out eval failures.

## Batch Acceptance Checklist

A synthetic batch is accepted only if:

- batch plan exists,
- all contracts parse against the schema,
- contract review completed,
- every admitted example has provenance,
- contamination check passed,
- EVAS evidence exists for admitted training examples,
- Spectre shadow audit exists for required slices,
- diversity report exists,
- split manifest exists,
- no EVAS PASS / Spectre FAIL false positives remain unresolved,
- generated examples can be reproduced from prompt hashes and seed refs.

## Implementation Order

Do not start with a bulk generator. Start with checks and manifests.

1. `check_contracts.py`: parse schema/examples and enforce cross-field rules.
2. `build_seed_catalog.py`: collect safe Tier A seeds and provenance.
3. `propose_contracts.py`: generate draft contracts from seed catalog.
4. `review_contracts.py`: static contract review and duplicate/scope checks.
5. `propose_artifacts.py`: generate DUT/TB/checker/fault candidates.
6. `check_static_artifacts.py`: contract alignment and anti-hack screening.
7. `verify_evas.py`: EVAS compile/sim/property verification.
8. `check_contamination.py`: release overlap gate.
9. `sample_spectre_shadow.py`: audit sampling manifest.
10. `pack_sft.py` / `pack_grpo.py` / `pack_repair.py`: final packing.

This ordering prevents the project from accumulating unverifiable synthetic
data before gates exist.

## Pilot Batch

First pilot should be deliberately small:

| Slice | Count |
| --- | ---: |
| L0 conformance | 5 |
| L1 dut | 10 |
| L1 tb | 5 |
| L1 bugfix | 10 |
| L2 e2e | 5 |

Pilot success criteria:

- all contracts parse,
- at least 60% candidates pass static checks,
- at least 30% candidates reach EVAS verification,
- no contamination hits,
- at least one Spectre shadow audit run succeeds,
- failed candidates produce useful error taxonomy labels.

If pilot fails, fix schema/prompts/checkers before scaling.

## Relationship to Other Specs

- Contract schema: `../data/contracts/schema.yaml`
- Reward spec: `DIAGNOSTIC_REWARD_SPEC.md`
- Experiment design: `TRAINING_PAPER_EXPERIMENT_DESIGN.md`
- Scope firewall: `../SCOPE_BOUNDARY.md`
