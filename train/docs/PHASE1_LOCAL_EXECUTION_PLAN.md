# Phase 1 Local Execution Plan

Status: design-to-implementation bridge, 2026-06-20.

This plan lists what can be advanced locally while server access is unavailable.
It does not start bulk data generation and does not require remote GPUs.

## Current Position

Completed design inputs:

- experiment design: `TRAINING_PAPER_EXPERIMENT_DESIGN.md`;
- contract schema and examples: `../data/contracts/schema.yaml`;
- reward design: `DIAGNOSTIC_REWARD_SPEC.md`;
- synthetic data factory: `SYNTHETIC_DATA_FACTORY.md`;
- Spectre shadow audit protocol: `SPECTRE_SHADOW_AUDIT_PROTOCOL.md`;
- contamination checker spec: `CONTAMINATION_CHECKER_SPEC.md`;
- Phase 1 parameters: `PHASE1_PARAMETER_DECISIONS.md`;
- manifest interfaces: `PHASE1_MANIFEST_SCHEMAS.md`;
- remote platform memory: `REMOTE_PLATFORM_STATUS.md`.

The next work should convert the design into small, testable local units before
any large LLM generation or remote training.

## Workstreams

| Order | Workstream | Output | Server needed | Stop condition |
| ---: | --- | --- | --- | --- |
| 1 | Manifest schema hardening | Pydantic/JSON schema choice and example fixtures | No | Example manifests validate locally. |
| 2 | Protected index dry-run | Protected index for vaBench release assets | No | Hashes and normalized signatures are reproducible. |
| 3 | Seed catalog | Clean-room seed catalog with provenance tiers | No | Every seed has allowed/forbidden uses. |
| 4 | Pilot batch plan | `synth-batch-0001` counts, categories, OOD choice | No | Distribution matches Phase 1 targets. |
| 5 | Contract review checklist | Local reviewer checklist for L0/L1/L2 contracts | No | Contracts can be accepted/rejected consistently. |
| 6 | Prompt templates | Contract/DUT/TB/checker/fault proposer templates | No | Templates expose no protected benchmark content. |
| 7 | EVAS verification adapter | Interface spec or local smoke if EVAS is available | Maybe local only | Report matches manifest schema. |
| 8 | Spectre audit handoff | Audit manifest generation without running Spectre | No | Audit sample can be selected deterministically. |
| 9 | SFT/GRPO packer contract | Pack schemas for LLaMA-Factory and `verl`/`trl` | No | Packed examples are subsets of admitted manifest. |
| 10 | Remote revalidation | Confirm 7B/LF/vLLM/verl/EVAS once network returns | Yes | Remote status is refreshed without package changes. |

## Immediate Local Sequence

### Step 1: Example manifests

Create minimal valid examples for:

- `protected_index.yaml`;
- `candidate_index.yaml`;
- `evas_verification_report.yaml`;
- `contamination_report.yaml`;
- `spectre_shadow_manifest.yaml`;
- `admitted_manifest.yaml`;
- `sft_pack_manifest.yaml`;
- `grpo_prompt_manifest.yaml`.

These examples should use fake toy IDs and hashes. They should not import or
copy benchmark release assets.

### Step 2: Schema validator shape

Choose one enforcement mechanism:

| Option | Pros | Cons | Recommendation |
| --- | --- | --- | --- |
| Pydantic only | Strong Python typing; easy for scripts. | Less language-neutral. | Best first implementation. |
| JSON Schema only | Tool-neutral; easy CI validation. | More verbose cross-field logic. | Add after Pydantic stabilizes. |
| Pydantic + generated JSON Schema | Good implementation and external contract. | Slightly more maintenance. | Target after first local pass. |

Default: implement Pydantic models first, then export JSON Schema once fields
settle.

### Step 3: Protected index dry-run

Build the protected index from the existing local release package without using
its contents as training seeds.

Allowed outputs:

- file paths;
- raw hashes;
- normalized hashes;
- module/interface signatures;
- prompt/checker fingerprints.

Forbidden outputs:

- generated training prompts derived from release prompts;
- reusable code templates copied from release gold;
- repair examples derived from release candidates.

### Step 4: Clean-room seed catalog

Define the first seed catalog with tiers:

- manual clean-room specifications;
- EVAS examples only after provenance review;
- Verilog-A skill examples only when not derived from vaBench release;
- public source material only with license/provenance;
- historical outputs excluded unless re-audited item by item.

The seed catalog should be smaller than the generated candidate pool. Its job is
to provide diverse intents, not direct training examples.

Status: first pilot seed catalog prepared at
`../data/seeds/seed_catalog.phase1-pilot-0001.yaml`. It contains review-pending
seed candidates only and does not admit training data.

### Step 5: Pilot batch plan

Draft `synth-batch-0001` around the accepted pilot targets:

- 150-250 seed contracts eventually;
- 3k-6k generated candidates eventually;
- 1k-2k admitted clean-room items eventually;
- first local smoke much smaller, around 20-50 contracts and no bulk generation.

The first batch should intentionally cover all interfaces:

- at least one L0 conformance contract;
- at least one L1 DUT contract;
- at least one L1 TB contract;
- at least one L1 bugfix contract;
- at least one L2 E2E contract.

Status: first pilot batch plan prepared at
`../data/manifests/pilot/batch_plan.synth-batch-pilot-0001.yaml` and validated
by `../pipelines/validate_pilot_plan.py`.

### Step 6: Contract review checklist and prompt templates

Before calling any LLM, prepare:

- a manual review checklist for generated contracts;
- contract proposal prompt template;
- contract review prompt template;
- artifact proposal prompt template;
- a renderer that writes scratch prompt records outside the repository by
  default.

Status: prompt gate prepared at `CONTRACT_REVIEW_CHECKLIST.md`,
`../data/prompts/templates/`, and `../pipelines/render_pilot_prompts.py`. It
renders 15 scratch prompt records for the five pilot seed candidates.

### Step 7: Contract synthesis smoke package

Before asking the remote server to create draft contracts, prepare:

- synthesis run plan with hash-stable policy fields;
- request packer that filters only `contract_proposal` prompts;
- generated contract index writer;
- generated contract validator;
- one-shot remote Codex task that returns final summary directly.

Status: prepared as
`../data/manifests/synthesis/synthesis_run.contract-smoke-0001.yaml`,
`../pipelines/prepare_contract_synthesis_requests.py`,
`../pipelines/write_generated_contract_index.py`,
`../pipelines/validate_generated_contracts.py`, and
`../infra/jobs/REMOTE_CODEX_TASK_CONTRACT_SYNTHESIS_SMOKE.md`.

## Decisions Not Reopened

Do not reopen these unless new evidence breaks them:

- training data is clean-room, not vaBench release data;
- OOD means clean-room train distribution shift, not vaBench distribution shift;
- EVAS Rust compile/elaboration is the Phase 1 compile signal;
- Spectre remains the external audit oracle;
- GRPO prompts do not contain gold completions;
- remote SFT platform smoke evidence is useful but not a Phase 1 data claim.

## Implementation Gate

Before writing bulk synthesis scripts, require:

1. example manifests exist and validate;
2. protected index writer exists and passes a local dry-run;
3. contamination report writer exists and fails closed;
4. one tiny admitted-manifest fixture can be packed into SFT and GRPO formats;
5. the remote platform status has been rechecked or explicitly marked as stale.

## Recommended Next Artifact

The next concrete artifact should be:

```text
train/data/manifests/examples/
```

with minimal example YAML files for the manifests above.

This is the smallest useful bridge from design to implementation because it
forces field-level consistency without requiring LLM generation, EVAS, Spectre,
or server access.

Status: prepared as tracked YAML fixtures in
`../data/manifests/examples/README.md`. The next implementation step is a
fixture validator or Pydantic model layer that checks these files mechanically.

Validator status: implemented as
`../pipelines/validate_manifest_fixtures.py`. Run it with:

```bash
python3 -m train.pipelines.validate_manifest_fixtures
```

GitHub handoff status: prepared as `SERVER_HANDOFF_RUNBOOK.md` plus
`../infra/run_github_handoff_smoke.sh`. This lets the screen-share-controlled
server pull the branch, run a toy Phase 1 smoke, and push small result artifacts
back to GitHub without direct SSH.

Remote platform probe status: prepared as `SERVER_PLATFORM_PROBE.md`,
`../infra/run_server_platform_probe.sh`, and
`../infra/jobs/REMOTE_CODEX_TASK_PLATFORM_PROBE.md`. This is the next
non-destructive server task after handoff; it checks CUDA/PyTorch/SFT/GRPO
readiness and uploads only small diagnostic JSON/text files.
