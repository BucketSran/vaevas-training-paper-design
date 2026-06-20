# `train/` Agent Contract

Operating rules for any agent (Claude or otherwise) working inside `train/`. Read top-level `vaEvas/AGENTS.md` first.

## Scope of this subproject

`train/` is a **research exploration** of SFT + GRPO for Verilog-A generation. It is NOT part of the vaEVAS paper mainline. See `BRIEF.md` for full scope.

## Hard rules

1. **Contamination firewall** (`SCOPE_BOUNDARY.md`): no exception. If data provenance is unclear, stop and ask the user.
2. **Do not modify** anything under `behavioral-veriloga-eval/`, `EVAS/`, or `veriloga-skills/` from within `train/`. File a separate vaEVAS-side issue if a change is needed.
3. **Do not report vaBench numbers** out of `train/` without an explicit, separately-gated contamination clearance step.
4. **Every reported metric must trace** to a `logs/<run-id>/` directory with seed, config, data hash, and reward implementation commit.
5. **No half-finished implementations.** Phase 0 has docs only. Phase 1+ implementations land as complete, tested units, not stubs.

## Phase awareness

Read `PLAN.md` first. It defines what is in scope this session. Tasks outside the current phase are deferred — surface them, do not silently expand scope.

## Default behaviors

- **Validate before claiming success.** SFT or RL "ran" only if a loss curve / reward curve exists and the resulting model loads. Pre-commit smoke tests required.
- **Log experiments under `logs/`.** Even failed runs. Failed runs have learning value.
- **Prefer the smallest validation.** Run training on 10-20 samples first to verify the loop before scaling.
- **Use existing libraries.** HuggingFace `transformers` + `trl` for SFT/GRPO; `accelerate` or `deepspeed` for distributed; `wandb` for logging. Do not roll custom infra unless an existing tool truly does not work.

## What to escalate to the user

- Any expansion beyond the current phase scope.
- Any data source whose vaBench overlap is ambiguous.
- Any reward-design change that affects scoring semantics.
- Any infra requirement that changes the compute footprint (GPUs, storage, time).

## Output standard

End every meaningful change with:
1. **What changed** — files and rationale.
2. **What was verified** — concrete command output, not narrative.
3. **KPI status** — phase gate progress.
4. **Residual risk** — what could still break.
5. **Next templated artifact** — what reusable thing should be lifted out.

## Cross-references

- vaEVAS top-level: `../AGENTS.md`, `../README.md`, `../EVAS_COMPATIBILITY_POLICY.md`
- vaBench source-of-truth: `../behavioral-veriloga-eval/docs/VAEVAS_MAINLINE_PLAN.md`
- This subproject: `BRIEF.md`, `KPI.md`, `ROADMAP.md`, `PLAN.md`, `SCOPE_BOUNDARY.md`
