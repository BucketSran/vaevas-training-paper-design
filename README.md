# vaEVAS Training Paper Design Snapshot

This repository is a clean design snapshot for the separate SFT + GRPO training-paper track.

It intentionally does **not** publish vaBench release data, generated candidates, simulator outputs, model checkpoints, logs, or historical experiment sweeps.

## Scope

- `CONTEXT.md` records the current project-level positioning.
- `docs/adr/` records accepted design decisions.
- `train/` contains the training-track scope docs, KPI gates, contract schema, reward design, synthetic-data protocol, and evaluation protocol.

## Current Position

- vaBench benchmark paper and this training paper are separate tracks.
- Spectre remains the reference oracle for reportable correctness claims.
- EVAS Rust is used as a fast training-time evaluator only under Spectre shadow audit.
- Training data is rebuilt clean-room through contract-first synthesis.
- SFT provides the cold start; GRPO optimizes diagnostic reward profiles.

## Publication Rules

- Do not add raw `train/data/` contents except schema and contract examples.
- Do not add `train/logs/`, model checkpoints, or generated simulator artifacts.
- Do not use benchmark release prompts or gold code as training data without explicit contamination clearance.
- Keep this branch as a design/paper-planning artifact until Phase 1 implementation begins.
