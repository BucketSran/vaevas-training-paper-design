# manifests/

Phase 1 manifest examples, pilot plans, and future generated manifest outputs.

The source-of-truth schema description is
`../../docs/PHASE1_MANIFEST_SCHEMAS.md`.

Tracked contents in this directory are schema examples only. Real generated
manifests from bulk synthesis or verifier runs should be reviewed before being
committed.

## Current Examples

`examples/` contains a single toy candidate flowing through the minimum Phase 1
evidence chain:

```text
protected_index
  + candidate_index
  + static_check_report
  + evas_verification_report
  + contamination_report
  + spectre_shadow_report
  + diversity_report
  + split_manifest
  -> admitted_manifest
  -> sft_pack_manifest / grpo_prompt_manifest
```

These files are not training data. Report-level hashes remain fixture values for
cross-reference testing, while the toy candidate's contract and artifact hashes
now bind to real fixture files so local packers can fail closed on hash mismatch.

The current toy candidate also includes tiny fixture-only artifacts under
`examples/artifacts/` so local packers can verify file hashes and emit temporary
SFT/GRPO JSONL under `/private/tmp` during smoke tests. These artifacts are not
paper training data.

Validate the current examples with:

```bash
python3 -m train.pipelines.validate_manifest_fixtures
```

The learning guide is `../../docs/MANIFEST_FIXTURE_GUIDE.md`.

## Current Pilot Plan

`pilot/` contains the first clean-room planning batch:

```text
data/seeds/seed_catalog.phase1-pilot-0001.yaml
  -> data/manifests/pilot/batch_plan.synth-batch-pilot-0001.yaml
  -> future contract/candidate generation
```

Validate it with:

```bash
python3 -m train.pipelines.validate_pilot_plan
```

The pilot plan is not admitted data; it only fixes target counts, seed coverage,
OOD holdout policy, and verification gates before generation.

Prompt rendering for this pilot is handled by:

```bash
python3 -m train.pipelines.render_pilot_prompts --out-dir /tmp/vaevas_phase1_prompt_gate
```

Rendered prompt JSONL is scratch synthesis input and should not be committed
unless a later task explicitly promotes it as a small fixture.

## Current Synthesis Run Plan

`synthesis/` contains the first one-shot contract synthesis smoke plan and a
larger overnight draft-data plan:

```text
data/manifests/synthesis/synthesis_run.contract-smoke-0001.yaml
  -> /tmp/vaevas_contract_synthesis_requests/contract_synthesis_requests.jsonl
  -> train/infra/results/contract-synthesis-smoke-<timestamp>/

data/manifests/synthesis/synthesis_run.contract-batch-0025.yaml
  -> /tmp/vaevas_contract-batch-0025-<timestamp>_requests/contract_synthesis_requests.jsonl
  -> train/infra/results/contract-batch-0025-<timestamp>/
  -> contracts + review_manifest + artifacts + artifact_candidate_index
  -> draft_training/draft_sft + draft_training/draft_grpo
```

Validate and prepare requests with:

```bash
python3 -m train.pipelines.validate_synthesis_run
python3 -m train.pipelines.prepare_contract_synthesis_requests --out-dir /tmp/vaevas_contract_synthesis_requests
```

The smoke run may produce five draft contract YAML files for review evidence,
but it must not generate Verilog-A artifacts, admitted data, SFT JSONL, or GRPO
JSONL.

The overnight batch may generate draft Verilog-A artifacts and draft SFT/GRPO
JSONL for inspection. These outputs are not admitted data; later contamination,
EVAS, Spectre-shadow policy, and admission gates are still required.
