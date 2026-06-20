# docs/

Design notes, learning materials, decision logs. **The first thing a new contributor reads** after the top-level scope docs.

## Index

| File | Purpose |
|---|---|
| `TRAINING_PAPER_EXPERIMENT_DESIGN.md` | Current design for the separate SFT/GRPO training paper: clean-room data, contract-first synthesis, diagnostic rewards, EVAS/Spectre audit, and evaluation gates. |
| `00_overview.md` | One-page summary of how SFT, GRPO, rewards, data, and eval fit together. |
| `01_sft_principles.md` | What SFT is, why we use it, how it differs from RL, key hyperparameters, failure modes. |
| `02_grpo_principles.md` | GRPO algorithm in detail, advantage normalization, KL term, why it works on this task. |
| `03_reward_design.md` | Historical simple reward sketch kept for context; superseded by `DIAGNOSTIC_REWARD_SPEC.md`. |
| `DIAGNOSTIC_REWARD_SPEC.md` | Current concrete GRPO reward design: component formulas, profiles, L2 decomposition, repair reward, anti-hack policy, and calibration gates. |
| `SPECTRE_SHADOW_AUDIT_PROTOCOL.md` | Current Spectre audit protocol: sampling schedule, mismatch triage, claim gates, manifests, and speed/correctness reporting boundaries. |
| `CONTAMINATION_CHECKER_SPEC.md` | Current contamination-checker specification: protected/candidate indexes, match tiers, split leakage, report schema, and Phase 1 implementation plan. |
| `PHASE1_PARAMETER_DECISIONS.md` | Accepted Phase 1-v0.1 data scale, OOD split, Spectre audit budget, contamination policy, and admission gates. |
| `PHASE1_MANIFEST_SCHEMAS.md` | Phase 1 manifest interfaces linking clean-room generation, EVAS/Spectre evidence, contamination gates, splits, SFT packing, and GRPO prompts. |
| `MANIFEST_FIXTURE_GUIDE.md` | Learning note explaining manifest fixtures, validator checks, and how manifests become SFT/GRPO training files. |
| `PHASE1_CLOSED_LOOP_PIPELINES.md` | Implementation map connecting protected index, contamination, SFT packing, and GRPO prompt packing. |
| `SERVER_HANDOFF_RUNBOOK.md` | GitHub-based server handoff procedure for screen-share-controlled remote execution and result upload. |
| `REMOTE_PLATFORM_STATUS.md` | Local memory snapshot of the remote 7B/LLaMA-Factory smoke-tested platform and what still needs live revalidation. |
| `PHASE1_LOCAL_EXECUTION_PLAN.md` | Server-free Phase 1 bridge plan: manifest fixtures, protected index dry-run, seed catalog, batch plan, and packer contracts. |
| `04_trajectory_format.md` | The step-by-step CoT structure for vaBench tasks (analog of Circuit-Think's port→device→connection). |
| `05_data_pipeline.md` | How data flows from raw sources → synthesized → EVAS-verified → SFT/RL ready. |
| `SYNTHETIC_DATA_FACTORY.md` | Current contract-first LLM data factory protocol: batch planning, multi-view artifact generation, verification gates, diversity filtering, and admission rules. |
| `06_eval_protocol.md` | Held-out construction, contamination check, metrics, statistical reporting. |
| `REFERENCES.md` | Papers, tutorials, codebases, blog posts to read. |
| `../../docs/adr/` | ADR-style records of major design decisions. One file per decision. |
| `references/` | Saved PDFs / scratch notes from referenced material. |
| `tools/` | Deep-dive notes on specific tools (vLLM, verl, etc). Replaceable as tools evolve. |
| `sft/` | SFT 深度学习笔记系列（foundations / data / training loop / 等）。|

## When to add a document

- A new training-stage concept enters the design → add to `docs/`.
- A reward function or trajectory rule changes meaningfully → log in `../../docs/adr/`.
- A paper directly informs the design → add a one-paragraph summary in `REFERENCES.md`.

## When NOT to add a document

- Anything that's already in the code as a clear docstring.
- Conversation transcripts (those belong in `logs/`).
- Phase-specific status (that belongs in `PLAN.md`).
