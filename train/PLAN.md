# Current-Phase Plan

> Updated at the start of each phase. Phase 1 active.

## Phase 1 — Manifest Bootstrap

### Status
In progress. Phase 0 design docs are complete enough to proceed. Phase 1
parameter choices, manifest interfaces, minimal toy fixtures, local closed-loop
pipeline tools, remote platform memory, and the server-free local execution plan
are now materialized under `docs/`, `data/manifests/examples/`, and `pipelines/`.

No clean-room training data has been admitted yet. Current work is limited to
schema discipline, toy manifest fixtures, local validators, and learning
documentation that explains how contracts become SFT/GRPO training examples.

### Concrete tasks

| # | Task | Output | Validation |
|---|---|---|---|
| 1 | Freeze Phase 1 parameters | `docs/PHASE1_PARAMETER_DECISIONS.md` | Data/OOD/audit targets accepted by user |
| 2 | Define manifest interfaces | `docs/PHASE1_MANIFEST_SCHEMAS.md`, `data/manifests/README.md` | YAML examples parse; fields cover provenance, verification, admission, SFT, and GRPO |
| 3 | Add minimal manifest fixtures | `data/manifests/examples/*.yaml` | Candidate → report → admitted → SFT/GRPO refs are internally consistent |
| 4 | Add fixture validator | `pipelines/validate_manifest_fixtures.py` | `python3 -m train.pipelines.validate_manifest_fixtures` passes |
| 5 | Add learning guide | `docs/MANIFEST_FIXTURE_GUIDE.md` | Explains contract, manifest, admitted item, SFT pack, and GRPO prompt with a toy example |
| 6 | Draft reusable schema modules | `pipelines/manifest_schemas.py` | Fixture validator and packers import shared Pydantic models |
| 7 | Draft local packers | `pipelines/pack_sft.py`, `pipelines/pack_grpo.py` | Emits JSONL from admitted manifests; GRPO prompt leak check passes |
| 8 | Draft contamination gate | `pipelines/build_protected_index.py`, `pipelines/check_contamination.py` | Clean fixture passes; exact-hash negative control rejects |

### Decisions already accepted

1. **Clean-room rebuild** — historical experiment outputs and benchmark-adjacent artifacts are excluded by default and may only inform taxonomy/error types unless re-audited item by item.
2. **OOD held-out strategy** — primary circuit-category held-out; secondary L2-hard held-out if enough L2 contracts exist.
3. **EVAS/Spectre role split** — EVAS Rust is the Phase 1 fast compile/sim filter; Spectre remains the shadow oracle and final reportable simulator.
4. **EVAS false-negative handling** — `Spectre PASS / EVAS FAIL` cases can be batched for later EVAS repair after enough evidence accumulates.
5. **Spectre audit budget** — pilot policy: audit `max(100, 20%)`, L2 at least 50%, and all first 10 examples for new checker/property types.
6. **Contamination policy** — exact release overlap rejects; high similarity needs review; split leakage blocks.
7. **Remote infra memory** — 7B SFT smoke/setup evidence exists, but live server status is not required for Phase 1 local data-manifest work.

### Open decisions before bulk generation

1. **Synthesis LLM choice** — Claude / GPT / both, plus cost and review budget.
2. **First batch size** — keep small enough for manual audit but large enough to test L0/L1/L2/task-form diversity.
3. **Spectre shadow schedule** — choose exact pilot cadence once real candidates exist.
4. **First real synthesis protocol** — decide how many clean-room candidates to generate before EVAS verification and manual review.

### Non-tasks

- Do not generate bulk training data before validator and contamination gates exist.
- Do not run SFT/GRPO before admitted SFT/GRPO packs exist and are hash-tracked.
- Do not touch `behavioral-veriloga-eval/`, `EVAS/`, or `veriloga-skills/` from this subproject.
- Do not report vaBench numbers from `train/` without a separate contamination clearance gate.
- Do not require live server access for Phase 1 local manifest work.

### Exit checklist for Phase 1

- [ ] Candidate, verification, contamination, admission, split, SFT, and GRPO manifests validate locally.
- [ ] At least one real clean-room pilot batch is generated and admitted under hash-tracked manifests.
- [ ] SFT JSONL and GRPO prompt packs are generated from admitted manifests only.
- [ ] Contamination report is present and blocks exact release overlap.
- [ ] Spectre shadow-audit queue is materialized for the accepted pilot budget.
- [ ] No generated training data, simulator dumps, checkpoints, or raw logs are committed by accident.
