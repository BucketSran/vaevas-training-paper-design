# Phase 1 Manifest Schemas

Status: design draft, 2026-06-20.

This document defines the Phase 1 manifest interfaces for the clean-room data
factory. It is intentionally upstream of SFT and GRPO training code.

The goal is to prevent each downstream script from inventing its own view of
data provenance, verification state, contamination status, split ownership, and
reward metadata.

## Relationship to Existing SFT/GRPO Platform Work

Existing notes show that the SFT platform has already been smoke-tested:

- `docs/sft/09_smoke_lessons.md` records an end-to-end LLaMA-Factory/Qwen2.5
  smoke run through environment checks, tokenizer/trajectory tokens, mini SFT,
  merge, and chat.
- `train_sft/README.md` defines the intended Phase 2 full SFT recipe on
  `Qwen/Qwen2.5-Coder-7B-Instruct`.

GRPO is less complete:

- `docs/grpo/00_environment_prep.md` and `docs/tools/verl.md` define the
  intended `verl` + `vLLM` platform and rollout configuration.
- The EVAS-backed reward service and `verl` integration have not yet been
  validated end-to-end.

The connection is:

```text
Phase 1 manifests
  -> admitted clean-room items
  -> SFT pack manifest + train/val jsonl
  -> SFT checkpoint
  -> GRPO prompt manifest + reward runtime metadata
  -> GRPO rollout/reward training
```

The platform work should consume these manifests. It should not decide which
data are clean, admitted, Spectre-audited, or split-safe.

## Global Conventions

All manifests must follow these conventions unless a later schema explicitly
overrides them.

| Field | Rule |
| --- | --- |
| `schema_version` | Semantic version string, starting at `phase1.v0.1`. |
| IDs | Stable, lowercase, snake-case or slug IDs. No timestamp-only IDs. |
| paths | Relative to `train/` unless explicitly marked as external. |
| hashes | `sha256:<hex>` for immutable file contents. |
| timestamps | ISO-8601 with timezone. |
| producer | Script/module name plus git commit if available. |
| source refs | Must distinguish clean-room seeds from protected benchmark assets. |
| mutable state | State transitions create a new manifest or report; do not silently edit prior evidence. |

A manifest item is not admissible if any required hash, source reference,
contract ID, split key, or gate result is missing.

## State Model

Phase 1 data moves through explicit states:

```text
seeded
  -> contract_ready
  -> artifact_generated
  -> static_checked
  -> evas_verified
  -> contamination_clean
  -> spectre_shadow_checked_if_required
  -> diversity_accepted
  -> split_assigned
  -> admitted
  -> packed_for_sft_or_grpo
```

Failed candidates should be retained with explicit state labels only when they
serve one of these purposes:

- repair data,
- reward calibration negative,
- EVAS false-negative backlog,
- prompt or checker debugging.

They must not enter SFT gold, GRPO prompt pools, or eval sets unless the relevant
admission gates pass.

## Seed Catalog

File:

```text
data/seeds/seed_catalog.<catalog_id>.yaml
```

Purpose: record clean-room seed candidates before any LLM synthesis. A seed
catalog is not training data and does not admit examples by itself.

Required top-level fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | Manifest version. |
| `catalog_id` | Stable seed catalog ID. |
| `purpose` | Why this catalog exists. |
| `policy_version` | Clean-room seed policy version. |
| `producer` | Script/agent and commit if available. |
| `seed_catalog_hash` | Hash of the normalized catalog payload. |
| `items` | Seed entries. |

Required seed fields:

| Field | Meaning |
| --- | --- |
| `seed_id` | Stable seed ID. |
| `source_tier` | Clean-room source tier, such as `A_manual_clean_room`. |
| `source_kind` | Manual, EVAS example, public source, or reviewed skill source. |
| `source_ref` | Source path or citation. Must not point at protected vaBench release assets. |
| `license` | Usage/license statement. |
| `allowed_uses` | Explicit allowed downstream planning uses. |
| `forbidden_uses` | Explicit forbidden uses, including benchmark evaluation and release copying. |
| `intended_levels` | Target L0/L1/L2 coverage. |
| `intended_task_forms` | Target `dut`, `tb`, `bugfix`, `e2e`, or `conformance` coverage. |
| `category` | Benchmark-aligned circuit-role category. |
| `status` | `seed_candidate`, `accepted_seed`, or `rejected`. |
| `contamination_review` | Review state and protected-source flags. |

Rules:

- `seed_candidate` entries require review before generation.
- `accepted_seed` still does not bypass candidate-level contamination checks.
- `rejected` seeds must not be referenced by a batch plan.

## Pilot Batch Plan

File:

```text
data/manifests/pilot/batch_plan.<batch_id>.yaml
```

Purpose: lock target distribution, seed coverage, split policy, generation
budget, and verification queue before a first pilot generation run.

Required fields extend the base batch plan with:

| Field | Meaning |
| --- | --- |
| `seed_catalog_ref` | Seed catalog path relative to `train/`. |
| `seed_catalog_hash` | Hash of the referenced catalog. |
| `planned_seed_ids` | Seed IDs intended for this pilot. |
| `split_policy` | Train/validation/OOD split constraints. |
| `generation_budget` | Candidate-per-contract and total candidate limits. |
| `verification_queue` | Static, EVAS, Spectre-shadow queue requirements. |
| `pilot_plan_hash` | Hash of the normalized pilot plan payload. |

The current pilot gate is implemented by
`train/pipelines/validate_pilot_plan.py`.

## Prompt Gate Records

File:

```text
<temp>/pilot_generation_prompts.jsonl
```

Purpose: render prompt-template records from the seed catalog and pilot batch
plan before any LLM generation. These records are scratch synthesis requests,
not model outputs and not training data.

Current renderer:

```text
train/pipelines/render_pilot_prompts.py
```

Current prompt kinds:

| Kind | Purpose |
| --- | --- |
| `contract_proposal` | Ask an LLM to draft a new contract YAML from a clean-room seed. |
| `contract_review` | Ask a reviewer to apply `CONTRACT_REVIEW_CHECKLIST.md`. |
| `artifact_proposal` | Ask an LLM to generate the artifact requested by an already reviewed contract. |

Rules:

- The renderer must not call external LLM APIs.
- Rendered records must not include target completions or hidden checker secrets.
- Rendered records should go to a temp directory unless intentionally promoted
  as a tiny fixture.

## Synthesis Run Plan

File:

```text
data/manifests/synthesis/synthesis_run.<run_id>.yaml
```

Purpose: define a bounded synthesis job before any model creates draft
contracts or artifacts.

Current smoke and overnight runs:

```text
data/manifests/synthesis/synthesis_run.contract-smoke-0001.yaml
data/manifests/synthesis/synthesis_run.contract-batch-0025.yaml
```

Required policy blocks:

| Block | Purpose |
| --- | --- |
| `prompt_selection` | Which rendered prompt kinds can be used and how many requests are allowed. |
| `model_policy` | Whether external APIs are allowed and how many contracts may be created. |
| `output_policy` | What the job may write and what it must not write. |
| `review_policy` | Manual/mechanical review requirements before downstream stages. |
| `validation_policy` | Required validators before outputs can be committed. |
| `remote_handoff` | One-shot return and commit behavior for server-mediated work. |

The smoke permits only `contract_proposal` prompts and forbids Verilog-A
artifacts, SFT/GRPO packs, simulator outputs, and checkpoints. The overnight
batch permits draft Verilog-A artifact generation and draft unadmitted SFT/GRPO
JSONL packing, but still forbids checkpoints, simulator dumps, and admitted
training-data claims.

## Generated Contract Index

File:

```text
train/infra/results/<run_id>/generated_contract_index.yaml
```

Purpose: index draft contract YAMLs produced by a synthesis smoke run. This is
review evidence, not admitted training data.

Required fields:

| Field | Meaning |
| --- | --- |
| `run_id` | Synthesis run ID. |
| `synthesis_run_hash` | Hash of the run plan. |
| `source_refs` | Run plan, request JSONL, and contracts directory. |
| `summary` | Counts by decision, level, task form, and category. |
| `items` | Contract refs, hashes, prompt refs, seed refs, and mechanical review output. |
| `generated_contract_index_hash` | Stable hash of the index payload. |

Current validators:

```text
train/pipelines/write_generated_contract_index.py
train/pipelines/validate_generated_contracts.py
```

## Contract Review Manifest

File:

```text
train/infra/results/<run_id>/review_manifest.yaml
```

Purpose: record review decisions after mechanical contract validation. Accepted
contracts may proceed to artifact generation; other contracts remain in
revision, quarantine, or reject states.

Current validators:

```text
train/pipelines/write_contract_review_manifest.py
train/pipelines/validate_contract_review_manifest.py
```

## Artifact Candidate Index

File:

```text
train/infra/results/<run_id>/artifact_candidate_index.yaml
```

Purpose: hash-track generated Verilog-A artifacts as draft candidate items. This
index enables draft pack generation, but it is not an admitted manifest and does
not imply EVAS/Spectre success.

Current validators:

```text
train/pipelines/write_artifact_candidate_index.py
train/pipelines/validate_artifact_candidate_index.py
```

## Draft Training Pack Manifest

File:

```text
train/infra/results/<run_id>/draft_training_pack_manifest.yaml
```

Purpose: record draft SFT/GRPO JSONL generated from artifact candidates for
inspection and later pipeline testing. It must carry
`status: draft_unadmitted_training_pack` and cannot be used for paper claims
until contamination, EVAS, Spectre-shadow policy, and admitted-manifest gates
pass.

## Protected Index

File:

```text
data/manifests/protected_index.yaml
```

Purpose: fingerprint protected benchmark-facing assets so Phase 1 generation can
reject release overlap and split leakage.

Required top-level fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | Manifest version. |
| `protected_root_ref` | Path or external reference for protected assets. |
| `benchmark_release_id` | Frozen release identifier if available. |
| `source_manifest_refs` | Release manifests used to enumerate protected files. |
| `indexed_at` | Timestamp. |
| `producer` | Index builder and commit. |
| `protected_index_hash` | Hash of the normalized index payload. |
| `items` | Protected asset entries. |

Required item fields:

| Field | Meaning |
| --- | --- |
| `protected_id` | Stable protected item ID. |
| `source_kind` | `release_prompt`, `gold_va`, `testbench`, `checker`, `metadata`, or `report`. |
| `source_path` | Protected file path or manifest path. |
| `task_id` | Benchmark task ID if applicable. |
| `counted_in_score` | Whether the protected task is benchmark-scored. |
| `asset_role` | Role inside the protected release. |
| `raw_sha256` | Raw file hash. |
| `normalized_sha256` | Hash after whitespace/comment normalization. |
| `ngram_minhash` | Optional locality-sensitive prompt/code fingerprint. |
| `module_signature_hash` | Optional identifier-normalized interface fingerprint. |
| `property_signature_hash` | Optional checker/property fingerprint. |

Rules:

- The protected index is read-only input for Phase 1.
- A protected asset hash must never appear as a generated candidate hash.
- Phase 1 scripts must fail closed if this index is absent or stale.

## Candidate Index

File:

```text
data/manifests/candidate_index.<batch_id>.yaml
```

Purpose: record every generated candidate before admission. This includes failed
or rejected candidates.

Required top-level fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | Manifest version. |
| `batch_id` | Generation batch ID. |
| `batch_plan_ref` | Batch plan used to set distribution targets. |
| `producer` | Candidate assembler and commit. |
| `candidate_index_hash` | Hash of normalized index payload. |
| `items` | Candidate entries. |

Required item fields:

| Field | Meaning |
| --- | --- |
| `candidate_id` | Stable candidate ID. |
| `contract_id` | Contract ID. |
| `contract_ref` | Contract YAML path. |
| `contract_sha256` | Contract hash. |
| `state` | Current candidate state. |
| `level` | `L0`, `L1`, or `L2`. |
| `task_form` | `dut`, `tb`, `bugfix`, `e2e`, or `conformance`. |
| `category` | Benchmark-aligned circuit-role category. |
| `split_key` | Leakage-prevention key. |
| `seed_refs` | Clean-room seed references and source tiers. |
| `artifact_refs` | DUT/TB/checker/repair artifact paths. |
| `artifact_hashes` | Hash for each artifact. |
| `allowed_features` | EVAS/Spectre feature subset used by the contract. |
| `forbidden_constructs` | Contract-level forbidden constructs. |
| `reward_profile` | Reward profile name from `DIAGNOSTIC_REWARD_SPEC.md`. |
| `intended_uses` | Booleans for SFT, GRPO, repair, eval, diagnostics. |
| `lineage` | Prompt/model/proposer metadata and prompt hashes. |
| `evidence_refs` | Static, EVAS, Spectre, contamination, diversity reports. |

Rules:

- `artifact_refs` may be absent for ready contracts, but must exist after
  `artifact_generated`.
- `intended_uses.sft_gold` requires an artifact that can serve as a gold
  completion.
- `intended_uses.grpo_prompt` must not expose target completion content.
- `level`, `task_form`, `category`, and `split_key` must match the contract.

## Static Check Report

File:

```text
data/manifests/static_check_report.<batch_id>.yaml
```

Purpose: capture cheap deterministic checks before EVAS.

Required fields:

| Field | Meaning |
| --- | --- |
| `candidate_index_hash` | Candidate index under test. |
| `summary` | Counts by pass/soft_fail/hard_fail/quarantine. |
| `items` | Per-candidate results. |

Per-candidate fields:

| Field | Meaning |
| --- | --- |
| `candidate_id` | Candidate ID. |
| `decision` | `pass`, `soft_fail`, `hard_fail`, or `quarantine`. |
| `checks` | Named check results. |
| `messages` | Human-readable diagnostic messages. |

Minimum checks:

- contract schema validates,
- artifact set matches `task_form`,
- module/port/parameter names match the contract,
- forbidden constructs are absent,
- no protected IDs, release paths, or checker notes appear,
- outputs are observable through public interfaces,
- artifact size and module count are within limits.

## EVAS Verification Report

File:

```text
data/manifests/evas_verification_report.<batch_id>.yaml
```

Purpose: record EVAS Rust compile/elaboration, simulation, property, and runtime
evidence.

Required top-level fields:

| Field | Meaning |
| --- | --- |
| `candidate_index_hash` | Candidate index under test. |
| `evas_commit` | EVAS Rust commit or build ID. |
| `evas_profile` | Feature/profile configuration. |
| `checker_version` | Checker/property evaluator version. |
| `summary` | Compile, simulate, property, timeout, and crash counts. |
| `items` | Per-candidate evidence. |

Per-candidate fields:

| Field | Meaning |
| --- | --- |
| `candidate_id` | Candidate ID. |
| `compile_status` | EVAS parser/elaboration status. |
| `simulate_status` | EVAS simulation status. |
| `property_status` | Contract property status. |
| `trace_health` | Required observables and waveform sanity. |
| `reward_components` | Diagnostic reward components, if computed. |
| `runtime_s` | EVAS wall-clock runtime. |
| `log_ref` | Log path. |
| `artifact_hashes` | Repeated hashes for evidence binding. |

Rules:

- Phase 1 compile status means EVAS Rust parser/elaboration status, not
  OpenVAF compilation.
- EVAS is the fast reference used at scale, but Spectre remains the external
  audit oracle for claimed slices.
- EVAS PASS / Spectre FAIL is not allowed in a claimed slice.

## Contamination Report

File:

```text
data/manifests/contamination_report.<batch_id>.yaml
```

Purpose: prove that candidate data do not overlap protected benchmark assets or
leak across training/eval splits.

Required top-level fields:

| Field | Meaning |
| --- | --- |
| `protected_index_hash` | Protected index used for matching. |
| `candidate_index_hash` | Candidate index under test. |
| `policy_version` | Threshold and decision policy version. |
| `summary` | Counts by clean/needs_review/quarantined/rejected. |
| `split_report` | Split-key and near-duplicate leakage checks. |
| `items` | Per-candidate decisions. |

Per-candidate fields:

| Field | Meaning |
| --- | --- |
| `candidate_id` | Candidate ID. |
| `decision` | `clean`, `needs_review`, `quarantined`, or `rejected`. |
| `strongest_match_tier` | Exact, high, medium, weak, or none. |
| `matched_protected_ids` | Protected matches if any. |
| `signals` | Hash, n-gram, signature, lineage, or split-key signals. |
| `reviewer` | Manual reviewer ID if applicable. |
| `notes` | Review notes. |

Rules:

- Missing provenance is quarantine.
- Exact protected overlap is rejection.
- High similarity is manual review or rejection, never automatic clean.
- Split leakage blocks admission even if release contamination is clean.

## Spectre Shadow Manifests

Two files are used:

```text
data/manifests/spectre_shadow_manifest.<audit_id>.yaml
data/manifests/spectre_shadow_report.<audit_id>.yaml
```

The manifest defines what will be audited. The report records results.

### Audit Manifest Fields

| Field | Meaning |
| --- | --- |
| `audit_id` | Stable audit ID. |
| `purpose` | Pilot admission, L2 audit, new checker audit, claim gate, etc. |
| `selection_policy` | Sampling rule and oversampling rationale. |
| `candidate_index_hash` | Candidate source. |
| `evas_report_hash` | EVAS report used for preselection. |
| `spectre_profile` | Host, version, model/library path, and runner profile. |
| `items` | Candidate IDs and artifact hashes selected for audit. |

### Audit Report Fields

| Field | Meaning |
| --- | --- |
| `audit_id` | Audit ID. |
| `spectre_profile` | Actual Spectre environment used. |
| `summary` | Pass/pass, pass/fail, fail/pass, fail/fail, timeout counts. |
| `claim_gate` | Whether the audited slice can support a claim. |
| `items` | Per-candidate EVAS/Spectre comparison. |
| `mismatches` | Detailed mismatch list and triage state. |

Per-item comparison fields:

| Field | Meaning |
| --- | --- |
| `candidate_id` | Candidate ID. |
| `evas_status` | Bound EVAS result. |
| `spectre_status` | Spectre result. |
| `property_delta` | Numeric/property deltas. |
| `evas_wall_time_s` | EVAS timing if available. |
| `spectre_wall_time_s` | Spectre timing. |
| `decision` | `agree_pass`, `agree_fail`, `evas_false_positive`, `evas_false_negative`, or `inconclusive`. |
| `triage_ref` | Mismatch issue or backlog entry. |

Rules:

- `evas_false_positive` means EVAS PASS / Spectre FAIL and blocks claims.
- `evas_false_negative` means Spectre PASS / EVAS FAIL and enters scheduled EVAS
  repair debt.
- Spectre PASS / EVAS FAIL candidates are not admitted as EVAS-verified training
  data until the policy explicitly allows that state.

## Diversity Report

File:

```text
data/manifests/diversity_report.<batch_id>.yaml
```

Purpose: prevent template collapse and overfitting.

Required fields:

| Field | Meaning |
| --- | --- |
| `candidate_index_hash` | Candidate source. |
| `admission_pool_hash` | Pool after verification/contamination filters. |
| `distribution` | Counts by level, task_form, category, property type, and split key. |
| `duplicate_clusters` | Prompt/spec/code/template clusters. |
| `decisions` | Per-candidate accept/reject decisions. |

Minimum distribution constraints for the accepted Phase 1 baseline are defined in
`PHASE1_PARAMETER_DECISIONS.md`.

## Split Manifest

File:

```text
data/manifests/split_manifest.<batch_id>.yaml
```

Purpose: define train/validation/internal eval/OOD ownership.

Required fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | Manifest version. |
| `batch_id` | Batch ID. |
| `policy_version` | Split policy version. |
| `source_pool_hash` | Hash of candidate/admission pool. |
| `splits` | Item IDs per split. |
| `ood_axes` | Held-out category, L2-hard, or parameter-range axes. |
| `leakage_checks` | Split-key and near-duplicate evidence. |

Rules:

- No `split_key` may appear in more than one of train, validation, eval, or OOD.
- OOD means out-of-distribution relative to the clean-room training set, not
  relative to vaBench.
- Eval and OOD items must not be used for prompt iteration or reward tuning.

## Admitted Manifest

File:

```text
data/manifests/admitted_manifest.<batch_id>.yaml
```

Purpose: list the only items that downstream packers may consume.

Required top-level fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | Manifest version. |
| `batch_id` | Batch ID. |
| `source_hashes` | Candidate, EVAS, contamination, Spectre, diversity, and split report hashes. |
| `admission_policy` | Policy version and gate settings. |
| `summary` | Counts by split, level, task_form, category, and reward profile. |
| `items` | Admitted item entries. |

Required item fields:

| Field | Meaning |
| --- | --- |
| `item_id` | Stable admitted item ID. |
| `candidate_id` | Candidate ID. |
| `contract_id` | Contract ID. |
| `split` | `sft_train`, `validation`, `clean_room_eval`, `ood_eval`, or `grpo_train`. |
| `split_key` | Leakage-prevention key. |
| `level` | `L0`, `L1`, or `L2`. |
| `task_form` | Task form. |
| `category` | Circuit-role category. |
| `artifact_hashes` | Bound hashes. |
| `evidence_refs` | Static, EVAS, contamination, Spectre, diversity reports. |
| `reward_profile` | Reward profile. |
| `use_flags` | SFT, GRPO, repair, eval eligibility. |
| `pack_targets` | Downstream packers allowed to consume the item. |

Admission gates:

1. provenance complete;
2. contract schema valid;
3. static checks pass;
4. EVAS compile/elaboration passes;
5. EVAS simulation health passes;
6. required properties pass under EVAS;
7. contamination decision is clean;
8. split leakage check passes;
9. Spectre shadow audit exists if required;
10. diversity filter accepts the item.

## SFT Pack Manifest

File:

```text
data/manifests/sft_pack_manifest.<run_id>.yaml
```

Purpose: bind SFT jsonl files to admitted items and tokenizer/model assumptions.

Required fields:

| Field | Meaning |
| --- | --- |
| `admitted_manifest_hash` | Source admitted manifest. |
| `output_files` | `data/sft/train.jsonl`, `data/sft/val.jsonl`, etc. |
| `output_hashes` | Hashes of packed jsonl files. |
| `format` | `llamafactory_alpaca`, `chat_messages`, or another explicit format. |
| `base_model_ref` | Base model path or HF revision. |
| `tokenizer_ref` | Tokenizer path or revision. |
| `special_tokens` | Required trajectory/control tokens. |
| `dataset_info_ref` | LLaMA-Factory dataset info file if used. |
| `item_ids` | Admitted items included in each split. |

Rules:

- Every packed item ID must exist in the admitted manifest.
- SFT output may include gold completions only for items with
  `use_flags.sft_gold = true`.
- The packer must validate jsonl keys against the target training framework.
- Historical smoke data may validate engineering plumbing, but not Phase 1
  training claims.

## GRPO Prompt Manifest

File:

```text
data/manifests/grpo_prompt_manifest.<run_id>.yaml
```

Purpose: bind GRPO prompt files to reward runtime metadata without leaking gold
answers.

Required fields:

| Field | Meaning |
| --- | --- |
| `admitted_manifest_hash` | Source admitted manifest. |
| `output_files` | `data/rl/prompts.jsonl`, parquet, or framework-specific prompt files. |
| `output_hashes` | Hashes of packed prompt files. |
| `format` | `trl_grpo`, `verl_parquet`, or another explicit format. |
| `base_checkpoint_ref` | SFT checkpoint used as GRPO initialization. |
| `reward_runtime_ref` | Reward configuration and implementation commit. |
| `reward_profiles` | Reward profiles used by packed prompts. |
| `group_size_hint` | Intended rollout group size, e.g. 8. |
| `item_ids` | Admitted prompt items. |

Rules:

- GRPO prompts must not include target completions.
- Reward runtime metadata must include contract, checker, EVAS profile, timeout,
  and anti-hack policy references.
- GRPO train prompts must not overlap SFT train or clean-room eval split keys
  unless the experiment explicitly declares an on-policy continuation setup.

## Required Gate Order

Phase 1 implementations should enforce this order:

1. build protected index;
2. build candidate index;
3. run static checks;
4. run EVAS verification;
5. run contamination report;
6. sample and run Spectre shadow audit where required;
7. run diversity filtering;
8. assign splits;
9. write admitted manifest;
10. pack SFT and GRPO data.

Do not run `pack_sft.py` or `pack_grpo.py` directly on raw generated candidates.

## Acceptance Tests for Manifest Writers

The first implementation should include tests for these invariants:

- all YAML files parse;
- every manifest hash is reproducible from normalized content;
- every admitted item has complete provenance;
- every admitted item has static, EVAS, contamination, and split evidence;
- every required Spectre audit reference exists;
- no EVAS PASS / Spectre FAIL item appears in an admitted claimed slice;
- no protected hash appears in candidate artifacts;
- no train/eval split-key collision exists;
- all packed SFT/GRPO item IDs are subsets of the admitted manifest;
- GRPO prompt files contain no gold completion fields.

## Implementation Order

Implement the manifest layer before bulk generation:

1. schema definitions and loaders for protected/candidate/admitted manifests;
2. protected index writer;
3. candidate index writer;
4. static check report writer;
5. EVAS verification report adapter;
6. contamination report writer;
7. Spectre shadow manifest/report writer;
8. diversity and split manifest writers;
9. SFT and GRPO pack manifests.

Bulk LLM synthesis should wait until these writers exist and fail closed.

## Open Decisions

- Exact YAML schema enforcement mechanism: Pydantic models, JSON Schema, or both.
- Exact similarity thresholds after the first protected-index dry run.
- Exact paths for per-candidate raw artifacts and logs.
- Exact `verl` prompt file format once GRPO smoke is implemented.
- Whether Spectre audit reports also become speed-measurement artifacts for the
  training paper, or remain correctness-only until the main vaEVAS speed gate is
  complete.
