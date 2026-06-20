# Contamination Checker Specification

This document defines the expected behavior of the future
`train/pipelines/check_contamination.py` gate. It is a Phase 0 design artifact,
not an implementation.

## Purpose

The training paper is only credible if training, validation, and held-out data
are clean-room with respect to vaBench release tasks. The contamination checker
must prevent:

- direct reuse of release prompts, gold Verilog-A, testbenches, or checkers;
- paraphrases that preserve the same task identity;
- code/template copying with superficial renaming;
- leakage between train/validation/eval splits inside `train/`;
- LLM synthetic data that traces back to forbidden release assets.

The checker is an admission gate. It does not make a dirty item clean; it only
classifies evidence for accept, review, quarantine, or reject.

## Protected Material

The protected index is built read-only from vaBench paper-facing assets.

Forbidden by default:

- `behavioral-veriloga-eval/benchmark-vabench-release-v1/**`;
- any row counted by the current score denominator manifest;
- prompts, metadata, gold code, testbench assets, checkers, and checker notes
  associated with release/scored tasks;
- known paraphrases or derived versions of release prompts;
- generated outputs that were produced using release gold assets as examples.

Historical or adjacent assets under `behavioral-veriloga-eval/` are not
automatically safe. They require explicit item-level audit and provenance.

## Candidate Material

The candidate index covers every artifact that may influence training:

- contract text and metadata,
- public prompt/spec text,
- generated DUT code,
- generated testbench code,
- property/checker configuration,
- repair prompt and broken artifact,
- SFT completion trajectory,
- RL prompt,
- eval prompt,
- seed prompt used for LLM synthesis,
- provenance metadata and source references.

Do not check only the final `.va` file. Contamination often enters through spec
text, checker expectations, or prompt lineage.

## Outputs

The checker produces:

1. a protected index;
2. a candidate index;
3. a match report;
4. an admission decision per candidate;
5. a split-leakage report for `train/data/sft`, `train/data/rl`, and
   `train/data/eval`;
6. a machine-readable summary used by CI/pre-commit and batch promotion.

Every admitted item must carry:

```yaml
vabench_audit:
  checked: true
  release_overlap: false
  checker_version: <git-sha-or-semver>
  protected_index_hash: <sha256>
  candidate_index_hash: <sha256>
  report_ref: <path>
  reviewed_by: null
  reviewed_at: null
```

If human review is needed, `release_overlap` remains unknown until the review is
resolved. Unknown is not clean.

## Decision States

| State | Meaning | Can enter training? |
| --- | --- | --- |
| `clean` | No blocking match, provenance complete, split-leakage clear. | Yes |
| `needs_review` | Similarity is high enough that a human must inspect. | No |
| `quarantined` | Evidence incomplete, source unclear, or audit infra failed. | No |
| `rejected` | Direct or derived release overlap detected. | No |

The checker must fail closed. Missing source, missing hash, missing protected
index, or parser failure yields `quarantined` unless manually reviewed.

## Normalization

All text/code views should be normalized before matching.

### Text normalization

- Unicode normalize to NFC.
- Lowercase English text.
- Strip Markdown formatting that does not affect meaning.
- Replace numeric literals with `<NUM>` for a separate normalized view.
- Replace identifiers with stable placeholders for a separate identifier-blind
  view.
- Remove boilerplate license headers and standard simulator preambles when they
  would otherwise dominate similarity.
- Keep a raw view for exact hashing.

### Verilog-A/code normalization

- Remove comments for semantic matching, while preserving raw hashes.
- Normalize whitespace.
- Normalize module/port declarations into sorted signatures.
- Replace local identifiers with placeholders in an identifier-blind view.
- Preserve numeric constants, parameter defaults, event expressions, analog
  contributions, and observable names in semantic views.
- Extract a lightweight feature vector:
  - module count,
  - module names,
  - port names/directions/disciplines,
  - parameter names/defaults,
  - analog operators,
  - event controls,
  - contribution targets,
  - observable/check names.

The first implementation may use regex/lightweight parsing. A future parser can
replace it, but the output fields should remain stable.

## Fingerprints

Each protected and candidate artifact should have multiple fingerprints:

| Fingerprint | Applies to | Use |
| --- | --- | --- |
| `raw_sha256` | all files/text blobs | Exact reuse detection. |
| `normalized_sha256` | text/code | Formatting-insensitive exact detection. |
| `ngram_minhash` | prompts/specs/trajectories | Paraphrase and near-duplicate detection. |
| `identifier_blind_hash` | code/specs | Renaming-insensitive detection. |
| `module_signature_hash` | Verilog-A | Interface reuse detection. |
| `property_signature_hash` | checker/property configs | Hidden checker leakage detection. |
| `split_key` | candidate metadata | Train/eval cluster leakage prevention. |
| `lineage_hash` | LLM synthesis seed chain | Derived-data tracking. |

## Match Tiers

### Tier 0: Exact block

Reject immediately when any of these match protected material:

- raw file hash,
- normalized text/code hash,
- known release task ID,
- known release module name plus matching port signature,
- protected gold code embedded as a substring.

### Tier 1: Strong similarity

Reject or require review when:

- prompt/spec n-gram similarity is very high;
- identifier-blind code hash matches;
- module signature plus behavior/operator fingerprint matches;
- checker/property signature matches;
- same spec structure with only parameter names/numeric values changed.

Default policy: `needs_review` unless there is an obvious exact release-derived
path, in which case `rejected`.

### Tier 2: Structural overlap

Require review when:

- the same circuit role, port signature, property set, and stimulus pattern
  appear together;
- generated text shares a rare phrase or checker assertion with a protected
  task;
- a candidate belongs to a protected `split_key` cluster;
- code similarity is low but the testbench/checker is near-identical.

### Tier 3: Benign domain overlap

Allow if provenance is clean and only generic domain vocabulary overlaps:

- common module names such as comparator, ADC, DAC, sampler;
- standard Verilog-A boilerplate;
- common analog terms such as threshold, gain, hysteresis, delay;
- textbook-level equations without release-specific structure.

The report should distinguish benign overlap from release-derived overlap.

## Threshold Policy

Initial thresholds should be conservative and audited after pilot batches.

| Signal | Default action |
| --- | --- |
| exact raw/normalized hash match | `rejected` |
| release task ID or file path in metadata | `rejected` |
| protected gold substring above a small code-block threshold | `rejected` |
| prompt/spec n-gram similarity above high threshold | `needs_review` |
| identifier-blind code similarity above high threshold | `needs_review` |
| module signature + property signature match | `needs_review` |
| only generic vocabulary overlap | `clean` if provenance is complete |

Do not tune thresholds to maximize admitted data count. Tune them to minimize
false clean decisions.

## Lineage Rules

Every LLM-synthesized candidate must record the full seed chain:

```yaml
lineage:
  source_kind: llm_synth
  seed_contract_ids: [contract_l1_comparator_001]
  seed_artifact_hashes: [sha256:...]
  generator_model: <model-name>
  generator_prompt_hash: <sha256>
  generation_run_id: <run-id>
  parent_candidate_ids: []
```

Reject if any parent is rejected or protected. Quarantine if any parent is
missing.

Paraphrasing a protected prompt is still protected. Changing identifiers,
parameter values, or prose order does not break lineage.

## Split-Leakage Check

The checker also enforces disjoint splits inside `train/`.

Forbidden:

- same `split_key` in train and eval;
- same contract ID in train and eval unless the eval item is explicitly a
  parameter-hidden robustness variant and the paper labels it as such;
- near-duplicate prompt/spec/code across SFT, RL, and eval;
- repair examples where the broken artifact or reference answer appears in the
  held-out eval split.

Required report fields:

```yaml
split_report:
  train_count: 0
  rl_count: 0
  eval_count: 0
  split_key_collisions: []
  near_duplicate_pairs: []
  blocked_eval_items: []
```

## Human Review Protocol

Human review is allowed for `needs_review`, not for exact protected matches.

Reviewer checklist:

1. identify protected candidate pair;
2. compare spec, interface, behavior, stimulus, checker, and code structure;
3. decide whether overlap is generic domain knowledge or release-derived;
4. record rationale in the report;
5. if accepted, set `reviewed_by`, `reviewed_at`, and `review_note`.

Review cannot override missing provenance. Missing provenance remains
`quarantined`.

## Exit Codes

The future script should use deterministic exit codes:

| Code | Meaning |
| ---: | --- |
| 0 | All candidates clean. |
| 1 | Rejected contamination found. |
| 2 | Review-required matches found. |
| 3 | Quarantine condition: missing provenance, missing protected index, parser failure, or infra error. |
| 4 | Split leakage found. |

CI/pre-commit should treat nonzero as blocking.

## Report Schema

```yaml
report_id: contamination_2026mmdd_batch001
created_at: 2026-MM-DDTHH:MM:SSZ
checker_version: <git-sha>
protected_index:
  root_ref: behavioral-veriloga-eval/benchmark-vabench-release-v1
  index_hash: <sha256>
  item_count: 0
candidate_index:
  root_ref: train/data/verified
  index_hash: <sha256>
  item_count: 0
summary:
  clean: 0
  needs_review: 0
  quarantined: 0
  rejected: 0
  split_leakage: 0
matches:
  - candidate_id: <id>
    protected_id: <id>
    state: needs_review
    tier: 1
    signals:
      prompt_ngram_similarity: 0.86
      module_signature_match: true
      property_signature_match: false
    action: block_until_review
    review: null
split_report:
  split_key_collisions: []
  near_duplicate_pairs: []
admission_manifest_ref: <path-or-null>
```

## Integration Points

### Data factory

Run after EVAS verification and before data promotion. Failed/generated
candidates may remain in diagnostics directories, but they cannot enter SFT,
RL, or eval splits.

### SFT packing

`pack_sft.py` must refuse any item without:

- `vabench_audit.checked == true`,
- `vabench_audit.release_overlap == false`,
- clean split-leakage status.

### GRPO prompts

`pack_rl.py` must apply the same gate. RL prompts are training data even if they
do not include answers.

### Evaluation

`train/data/eval/` must pass both release-overlap and train/eval split-leakage
checks. Evaluation data cannot be used as SFT or RL data.

## What Not To Do

- Do not ask an LLM to decide whether data is contaminated.
- Do not use release prompts as few-shot examples for synthetic generation.
- Do not copy release checker text into training feedback.
- Do not mark paraphrases as clean merely because exact hashes differ.
- Do not lower thresholds after a batch fails unless a documented false-positive
  analysis justifies it.
- Do not publish vaBench performance from `train/` without a separate frozen
  model and contamination clearance gate.

## Implementation Roadmap

Phase 1 implementation should proceed in this order:

1. protected index builder;
2. candidate index builder;
3. exact and normalized hash matching;
4. prompt/spec n-gram similarity;
5. Verilog-A signature and identifier-blind fingerprints;
6. checker/property signature matching;
7. split-leakage check;
8. YAML/JSON report writer;
9. CI/pre-commit integration;
10. reviewer workflow for `needs_review`.

The first version should prefer conservative blocking over high recall of clean
data. Data volume is less important than paper-valid provenance.

## Acceptance Test Ideas

The implementation should include fixture cases:

- exact copied prompt -> `rejected`;
- prompt paraphrase with same rare structure -> `needs_review`;
- copied gold code with renamed ports -> `rejected` or `needs_review`;
- generic comparator prompt from clean contract -> `clean`;
- missing provenance -> `quarantined`;
- train/eval same `split_key` -> exit code 4;
- generated candidate whose parent was rejected -> `rejected`;
- protected checker phrase copied into training feedback -> `needs_review` or
  `rejected`.

## Open Decisions Before Phase 1

- Exact protected-index source list for the current vaBench release.
- Initial numeric thresholds for n-gram and code similarity.
- Whether human review reports live under `train/data/audits/` or `train/logs/`.
- Whether protected-index hashes should be pinned per paper submission.
