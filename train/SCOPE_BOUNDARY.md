# Contamination Firewall — read before touching data

**This is a hard rule, not a guideline.** Violating it makes every result inside `train/` paper-invalid and forces a full retrain.

## The Problem

`train/` is downstream of `behavioral-veriloga-eval/` (vaBench). If training data leaks from the scored vaBench tasks, the trained model "remembers" the test set and any reported accuracy is meaningless. This is the same trap Circuit-Think partially fell into and what AMSBench was designed to expose.

## Hard Rules

### ❌ FORBIDDEN as training data

| Source | Why forbidden |
|---|---|
| `behavioral-veriloga-eval/benchmark-vabench-release-v1/**` | This is the paper-facing scored benchmark. Direct contamination. |
| Any task whose ID appears in `score_denominator_manifest.json` `counted_in_score` | This is the live test set. |
| Any `prompt.md`, `meta.json`, `gold/`, `checks.yaml` whose task ID is in the release | Same as above, finer-grained. |
| LLM-generated paraphrases of release prompts | Surface-level rewording does not de-contaminate. |
| Anything that reuses release `gold/` Verilog-A code as a starting template | Code-level leakage. |

### ⚠️ REQUIRES AUDIT before use

| Source | Audit step |
|---|---|
| `behavioral-veriloga-eval/tasks/` (historical) | For every task pulled, verify the task ID is NOT in `score_denominator_manifest.json::counted_in_score`. Log the audit. |
| `behavioral-veriloga-eval/_archives/` | Same as above. |
| Any future inventory under `behavioral-veriloga-eval/` | Same as above. |

### ✅ ALLOWED without audit

| Source | Notes |
|---|---|
| `EVAS/evas/examples/` | Bundled simulator examples, public, not in vaBench. |
| `veriloga-skills/` reference templates | Skill docs and worked examples. |
| Public Verilog-A code (textbooks, open IP, papers) | Cite source. |
| LLM-synthesized novel specs **+ EVAS-verified** | Must not paraphrase any release task. Document the seed prompt and synthesis pipeline. |

## Audit Procedure

Every data file landed in `train/data/verified/` MUST carry provenance metadata:

```yaml
# train/data/verified/<id>.meta.yaml
provenance:
  source_kind: evas_example | veriloga_skill | textbook | llm_synth | manual
  source_ref: <path or URL>
  vabench_audit:
    checked: true
    release_overlap: false
    overlap_check_script: train/pipelines/check_contamination.py
    overlap_check_run: 2026-MM-DD
created_by: <agent or user>
created_at: <ISO date>
```

A data file without `vabench_audit.checked == true` and `release_overlap == false` MUST NOT enter `train/data/sft/` or `train/data/rl/`.

## Held-out Evaluation Rule

`train/eval/` evaluates on **its own held-out set** (see `train/data/eval/`). This held-out set:

- MUST be disjoint from the training set at the **specification level**, not just the prompt level.
- SHOULD include a topology held-out slice — at least one circuit family the model has not seen during training (e.g., train without comparators, then test on comparators).
- MUST NOT be used to make claims about vaBench performance unless an explicit, separately gated experiment is run AFTER training is frozen.

## Reporting Rule

When reporting any number out of `train/`:

- State the training data source and audit status.
- State the evaluation set explicitly (`train/data/eval/` ≠ vaBench).
- If any vaBench number is reported, declare the contamination audit result.

## Enforcement

Agents reading this document MUST refuse to:
- Copy files from `behavioral-veriloga-eval/benchmark-vabench-release-v1/` into `train/data/`.
- Run training on any file lacking provenance metadata.
- Report vaBench accuracy without an explicit contamination clearance step.

If unsure whether a candidate data source is safe, **stop and ask the user** before proceeding.
