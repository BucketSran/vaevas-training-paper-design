# Contract Review Checklist

Status: Phase 1 gate, 2026-06-21.

This checklist is the manual review surface between clean-room seeds and any
LLM-generated contracts or artifacts. Passing this checklist does not admit
training data; it only allows the item to move to candidate generation and
mechanical verification.

## Review Inputs

For each contract or generated proposal, reviewers must see:

- seed ID and source tier;
- source reference and license/provenance note;
- target `category`, `level`, and `task_form`;
- complete contract YAML;
- generator prompt hash, if LLM-generated;
- contamination status and release-overlap status;
- intended use flags: SFT, GRPO, repair, eval, diagnostics.

## Immediate Rejects

Reject the item if any condition is true:

- source or wording is copied/paraphrased from protected benchmark release assets;
- source provenance is missing or ambiguous;
- contract lacks observable properties or only describes implementation style;
- `split_key` is missing, too broad, or inconsistent with related variants;
- L2 contract lacks decomposed subclaims and a system-level metric;
- bugfix contract lacks `fault_model` and `repair_delta`;
- TB contract hides checker logic or hardcodes expected answers;
- any construct violates the Phase 1 voltage-domain EVAS/Spectre scope;
- any EVAS PASS / Spectre FAIL evidence is present;
- GRPO-visible prompt includes target completion, answer, gold code, or checker-only secrets.

## Required Checks

| Gate | Question | Pass Condition |
| --- | --- | --- |
| Provenance | Is the seed source allowed and reviewable? | Tier A/B/C only, explicit source ref, no protected release source. |
| Taxonomy | Are category/level/task_form canonical? | Values match `train/data/contracts/schema.yaml`. |
| Observability | Can properties be checked from public observables? | Every property names observables, expected behavior, and tolerance. |
| EVAS Scope | Is the task inside voltage-domain behavioral Verilog-A? | Forbidden constructs include unsupported features; no current-domain dependency. |
| Diversity | Does the contract add a distinct split key or variation axis? | Not only variable/module renaming. |
| Split Safety | Can related variants be held together? | `split_key` groups topology/template variants. |
| SFT Safety | Is the completion source independent and verified? | No SFT gold until EVAS and contamination gates pass. |
| GRPO Safety | Can the prompt be shown without answers? | Model-visible fields contain no completion, answer, or hidden checker logic. |
| Spectre Audit | Is the shadow policy clear? | L2, first property type, and high-reward samples enter shadow queue. |

## Level-Specific Review

### L0 Conformance

- Purpose is simulator semantic coverage, not headline model capability.
- The property must isolate one semantic class, such as event direction or
  scheduling behavior.
- Admission should keep L0 separate from L1/L2 circuit-function scoring.

### L1 Component or Flow

- The task must describe a reusable circuit function, measurement flow, or
  repair objective that a designer would recognize.
- Properties should include at least one functional behavior and one structural
  or safety constraint.
- TB tasks must test public interface behavior without inspecting internals.

### L2 Composed Flow

- The contract must combine multiple L1-style behaviors into a mini-system.
- It must include decomposed subclaims and a final system-level metric.
- It must receive higher Spectre-shadow priority before any paper-facing claim.

## Decision Labels

Use these labels in review notes:

- `accept_for_generation`: safe to generate candidates, still not admitted data.
- `needs_revision`: reviewer found fixable contract/provenance issues.
- `quarantine`: source or similarity risk blocks use until audit.
- `reject`: unsafe, out of scope, duplicate, or not meaningful for training.

## Reviewer Output

Reviewer notes should record:

```yaml
review_decision: accept_for_generation | needs_revision | quarantine | reject
reviewer: <name-or-agent>
reviewed_at: <ISO-8601>
blocking_findings: []
required_edits: []
allowed_next_stage: contract_generation | artifact_generation | none
```

No item may enter `train/data/sft/`, `train/data/rl/`, or `train/data/eval/`
from this checklist alone.

