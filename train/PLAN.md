# Current-Phase Plan

> Updated at the start of each phase. Phase 0 active.

## Phase 0 — Scaffold

### Status
In progress. Directory tree created. Top-level docs written. Training-paper
experiment design, contract-first data schema, diagnostic reward spec, synthetic
factory protocol, Spectre audit protocol, and contamination-checker spec are now
drafted. Phase 1 parameter baseline is accepted in
`docs/PHASE1_PARAMETER_DECISIONS.md`. Phase 1 manifest interfaces are drafted in
`docs/PHASE1_MANIFEST_SCHEMAS.md`. Remote SFT platform memory and the
server-free local Phase 1 bridge plan are captured in
`docs/REMOTE_PLATFORM_STATUS.md` and `docs/PHASE1_LOCAL_EXECUTION_PLAN.md`.

### Concrete tasks (this session)

| # | Task | Output | Validation |
|---|---|---|---|
| 1 | Create directory tree | `tree train/` shows all subdirs | `ls train/` |
| 2 | Top-level docs | `README`, `AGENTS`, `BRIEF`, `KPI`, `ROADMAP`, `PLAN`, `SCOPE_BOUNDARY` | All files exist, user reads them |
| 3 | Subdir READMEs | One `README.md` per subdir | `find train/ -name README.md \| wc -l` |
| 4 | Learning notes | `docs/00-06*.md` + `REFERENCES.md` | Each note >100 lines, references cited |
| 5 | gitignore | `train/.gitignore`, `data/.gitignore`, `models/.gitignore`, `logs/.gitignore` | Confirms checkpoints / raw data excluded |
| 6 | User review | User reads `BRIEF.md` + `SCOPE_BOUNDARY.md` and approves | Explicit "Phase 0 approved, proceed to Phase 1" |
| 7 | Training-paper design | `docs/TRAINING_PAPER_EXPERIMENT_DESIGN.md`, `data/contracts/schema.yaml`, example contracts, `docs/DIAGNOSTIC_REWARD_SPEC.md`, `docs/SYNTHETIC_DATA_FACTORY.md`, `docs/SPECTRE_SHADOW_AUDIT_PROTOCOL.md`, `docs/CONTAMINATION_CHECKER_SPEC.md`, `docs/PHASE1_PARAMETER_DECISIONS.md`, `docs/PHASE1_MANIFEST_SCHEMAS.md`, `docs/REMOTE_PLATFORM_STATUS.md`, `docs/PHASE1_LOCAL_EXECUTION_PLAN.md` | YAML parses; L0/L1/L2 and `dut/tb/bugfix/e2e` covered; reward spec has profiles and calibration gates; factory protocol has admission gates; audit/checker specs define manifests and stop conditions; Phase 1 data/OOD/audit targets accepted; manifest interfaces bind evidence, admission, SFT packing, and GRPO prompts; remote platform memory is separated from current data claims |

### Decisions to surface to the user before Phase 1

1. **Data source clarification** — current decision: rebuild a clean-room training set. Historical experiment outputs and benchmark-adjacent artifacts are excluded by default and may only inform taxonomy/error types unless re-audited item by item.
2. **OOD held-out strategy** — accepted baseline: primary circuit-category held-out; secondary L2-hard held-out if enough L2 contracts exist.
3. **Remote infra access** — remote SFT smoke evidence exists, but live server status must be revalidated before Phase 2 infra scripts or GRPO package changes are made.
4. **Synthesis LLM choice** — Claude / GPT-4 / both for data synthesis? Cost budget?
5. **Contract schema approval** — review `data/contracts/schema.yaml` and the example contracts before writing synthesis or verifier pipelines.
6. **Synthetic factory and manifest approval** — review `docs/SYNTHETIC_DATA_FACTORY.md` and `docs/PHASE1_MANIFEST_SCHEMAS.md` before running bulk LLM generation.
7. **Spectre audit budget** — accepted pilot policy: audit `max(100, 20%)`, L2 at least 50%, and all first 10 examples for new checker/property types.
8. **Contamination threshold policy** — accepted conservative baseline: exact release overlap rejects; high similarity needs review; split leakage blocks.

### Non-tasks (deliberately deferred to later phases)

- Writing actual `synthesize.py` / `verify_evas.py` / training scripts. The scaffold defines where they live; implementation is Phase 1+.
- Standing up the remote server. The scaffold defines the `infra/` layout; setup is Phase 2.
- Picking exact hyperparameters. Configs are template-only at Phase 0; tuned at Phase 2/3.

### Exit checklist for Phase 0

- [ ] User has read `BRIEF.md`.
- [ ] User has read `SCOPE_BOUNDARY.md` and agreed to the firewall.
- [ ] User has reviewed `ROADMAP.md` and approved the phased plan.
- [ ] All scaffold files exist, no Python skeletons (those come in Phase 1+).
- [ ] User explicitly says "proceed to Phase 1".

## When Phase 0 exits

Update this file with the Phase 1 plan, mirroring the same structure (status / tasks / decisions / non-tasks / exit checklist).
