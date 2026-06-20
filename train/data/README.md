# data/

All training, validation, and held-out evaluation data lives here. **Contents are gitignored** — only structure and metadata schemas are tracked.

> **Hard rule**: nothing lands in `sft/`, `rl/`, or `eval/` without provenance metadata showing `vabench_audit.release_overlap == false`. See `../SCOPE_BOUNDARY.md` and `../docs/CONTAMINATION_CHECKER_SPEC.md`.

## Flow

```text
contracts/      →    synthesized/     →    manifests/      →    admitted/        →    sft/  rl/  eval/
(task contracts      (LLM-generated       (EVAS/Spectre,       (clean-room         (final, packed
 + schemas)           candidates,          contamination,       admitted item       for SFT,
                      including failures)  split evidence)      references)         GRPO, eval)
```

## Subdirectories

| Dir | Holds | When written |
|---|---|---|
| `seeds/` | Clean-room seed catalogs with source tier, allowed uses, forbidden uses, and review status. These are planning inputs, not training examples. | Phase 1 seed gate |
| `contracts/` | Contract schema and example task contracts for clean-room synthetic data generation. These are metadata/design artifacts, not training samples. | Phase 0/1 design gate |
| `raw/` | Source materials before any transformation: EVAS example copies, skill templates, textbook excerpts. One subdir per source. | Phase 1 task 1 (source inventory) |
| `synthesized/` | LLM-generated `<spec, va, tb>` candidates, not yet verified. Includes failed candidates for diagnostic logging. | Phase 1 task 2 (synthesis) |
| `manifests/` | Protected/candidate/EVAS/contamination/Spectre/diversity/split/admitted/pack manifests. See `../docs/PHASE1_MANIFEST_SCHEMAS.md`. | Phase 1 gates |
| `verified/` | Legacy/staging name for EVAS-verified candidates. New scripts should prefer manifest-bound admitted items over free-form verified dirs. | Phase 1 task 2/3 |
| `sft/` | Final SFT-ready data: `train.jsonl`, `val.jsonl`, plus SFT pack manifest. Each line is one prompt-completion pair with trajectory. | Phase 1 task 5 (packing) |
| `rl/` | GRPO prompt files plus reward runtime metadata. Training prompts must not include gold completions. | Phase 1 task 5 (packing) |
| `eval/` | Held-out evaluation set. See `eval/README.md` for split rules. | Phase 1 task 5 (held-out) |

## Contract schema

The current contract-first schema lives at `contracts/schema.yaml`.
Representative draft contracts live under `contracts/examples/`.

Contracts reuse vaBench terminology:

- `category`: circuit-role taxonomy, such as `Comparator and Decision Circuits`.
- `level`: `L0`, `L1`, or `L2` functional granularity.
- `task_form`: `dut`, `tb`, `bugfix`, `e2e`, or `conformance`.

Do not use `family` for circuit category. Existing vaBench metadata often uses
`family` for task forms such as `spec-to-va`, `tb-generation`, `end-to-end`, and
`bugfix`.

## Legacy verified-data metadata sketch

The current source of truth is `../docs/PHASE1_MANIFEST_SCHEMAS.md`. The older
`.meta.yaml` sketch below is retained only to show the original intent of
per-item provenance; new Phase 1 implementations should emit manifests instead
of relying on this free-form metadata.

Each legacy entry in `verified/` carried a `.meta.yaml`:

```yaml
id: <stable-id>
provenance:
  source_kind: evas_example | veriloga_skill | textbook | llm_synth | manual
  source_ref: <path or URL>
  vabench_audit:
    checked: true
    release_overlap: false
    overlap_check_script: ../pipelines/check_contamination.py
    overlap_check_run: 2026-MM-DD
    report_ref: <path-to-contamination-report>
artifacts:
  spec: <path-to-spec.md>
  gold_va: <path-to-gold.va>
  gold_tb: <path-to-tb.scs>
  evas_run_log: <path-to-evas-log>
  evas_tran_csv: <path-to-tran-csv>
verifier_evidence:
  evas: <path-or-id>
  spectre_shadow: <path-or-id-or-null>
classification:
  level: L0 | L1 | L2
  task_form: dut | tb | bugfix | e2e | conformance
  category: <benchmark-aligned circuit category>
  difficulty: simple | medium | hard
trajectory:
  step1_ports: <yaml-or-json>
  step2_behavior: <yaml-or-json>
  step3_testbench: <yaml-or-json>
created_by: <agent or user>
created_at: <ISO date>
```

## What NOT to put here

- Anything from `behavioral-veriloga-eval/benchmark-vabench-release-v1/` (see `SCOPE_BOUNDARY.md`).
- Model checkpoints (those go in `../models/`).
- Training logs (those go in `../logs/`).
- Result tables (those go in `../eval/reports/`).
