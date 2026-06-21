# Remote Codex Task: Contract + Verilog-A + Draft Training Overnight

Run one complete Phase 1 overnight data-preparation job and return the final
results directly in one response.

## Objective

Create a useful overnight draft data pool:

1. Generate exactly 25 clean-room draft contract YAML files.
2. Mechanically validate and structurally review the contracts.
3. Promote usable contracts to artifact generation.
4. Generate Verilog-A draft artifacts for accepted contracts.
5. Build a hash-tracked artifact candidate index.
6. Pack draft unadmitted SFT JSONL and GRPO prompt JSONL.
7. Commit only the small result directory and return all summaries.

This is allowed to generate Verilog-A and draft training JSONL. It must not run
SFT/GRPO model training, write checkpoints, run EVAS/Spectre, or claim admitted
paper-quality training data.

## Non-Negotiable Boundaries

- Do not ask for confirmation between steps; run the full job and return the
  final result directly.
- Do not call external LLM APIs from scripts. The remote Codex session itself
  may author the YAML/code artifacts.
- Do not write model checkpoints.
- Do not run SFT or GRPO training.
- Do not run EVAS or Spectre in this task unless a local no-install command is
  already obvious; this task is primarily data preparation.
- Do not modify files outside `train/`.
- Do not commit simulator dumps, raw logs beyond `commands.log`, package caches,
  `.env`, secrets, or checkpoints.
- Label outputs as draft/unadmitted unless later EVAS, contamination, Spectre
  shadow, and admission gates pass.

## Setup

Work on the latest `training-paper-design-20260620` branch.

```bash
set -euo pipefail

git fetch origin training-paper-design-20260620
git checkout training-paper-design-20260620
git pull --ff-only origin training-paper-design-20260620

RUN_ID=contract-batch-0025-$(date -u +%Y%m%dT%H%M%SZ)
REQUEST_DIR=/tmp/vaevas_${RUN_ID}_requests
RESULT_DIR=train/infra/results/${RUN_ID}
RUN_PLAN=train/data/manifests/synthesis/synthesis_run.contract-batch-0025.yaml
export RUN_ID REQUEST_DIR RESULT_DIR RUN_PLAN

mkdir -p "${REQUEST_DIR}" "${RESULT_DIR}/contracts" "${RESULT_DIR}/artifacts"
: > "${RESULT_DIR}/commands.log"
```

Append important commands and their stdout to `${RESULT_DIR}/commands.log`.

## Step 1: Validate Run Plan and Prepare 25 Contract Requests

```bash
python3 -m train.pipelines.validate_synthesis_run \
  --run-plan "${RUN_PLAN}" | tee -a "${RESULT_DIR}/commands.log"

python3 -m train.pipelines.prepare_contract_synthesis_requests \
  --run-plan "${RUN_PLAN}" \
  --out-dir "${REQUEST_DIR}" | tee -a "${RESULT_DIR}/commands.log"

cat "${REQUEST_DIR}/contract_synthesis_request_summary.json" | tee -a "${RESULT_DIR}/commands.log"
cat "${REQUEST_DIR}/contract_synthesis_requests.jsonl"
```

## Step 2: Generate 25 Draft Contract YAML Files

For each JSONL request, write exactly one draft contract YAML under:

```text
${RESULT_DIR}/contracts/
```

Contract requirements:

- Must match `train/data/contracts/schema.yaml`.
- Must pass `train.pipelines.contract_review_gate`.
- Must use the request `category`, `level`, and `task_form`.
- Must create a distinct contract for every `variant_id`; do not only rename
  the same task.
- Must use a new `id` and a new `split_key` reflecting the variant focus.
- Must not reuse vaBench release IDs or protected benchmark material.
- Must set `provenance.source_tier: B_llm_synthetic`.
- Must set `provenance.source_kind: llm_synthetic`.
- Must set `provenance.seed_refs` to the request `seed_id`.
- Must set `provenance.generator.prompt_id` to the request `request_id`.
- Must set `provenance.generator.prompt_hash` to the request `prompt_sha256`.
- Must set EVAS/Spectre fields to `not_run`, `pending`, or equivalent
  non-executed states allowed by the schema.
- Must set `admission.status: draft_contract` or `ready_for_generation`.
- Must include blockers or notes stating that contamination, EVAS, Spectre, and
  admission gates remain required before final training claims.

Good source references:

- `train/data/contracts/schema.yaml`
- prior small result contracts under
  `train/infra/results/contract-synthesis-smoke-*/contracts/`
- prompt requirements in `${REQUEST_DIR}/contract_synthesis_requests.jsonl`

## Step 3: Index and Validate Contracts

Repair simple YAML/schema/mechanical issues and rerun up to three times.

```bash
python3 -m train.pipelines.write_generated_contract_index \
  --contracts-dir "${RESULT_DIR}/contracts" \
  --request-jsonl "${REQUEST_DIR}/contract_synthesis_requests.jsonl" \
  --synthesis-run "${RUN_PLAN}" \
  --out "${RESULT_DIR}/generated_contract_index.yaml" \
  --model-ref remote_codex_default \
  --expect-count 25 | tee -a "${RESULT_DIR}/commands.log"

python3 -m train.pipelines.validate_generated_contracts \
  --index "${RESULT_DIR}/generated_contract_index.yaml" \
  --synthesis-run "${RUN_PLAN}" \
  --expect-count 25 | tee -a "${RESULT_DIR}/commands.log"
```

## Step 4: Structured Review and Promotion

Create the conservative review manifest:

```bash
python3 -m train.pipelines.write_contract_review_manifest \
  --index "${RESULT_DIR}/generated_contract_index.yaml" \
  --out "${RESULT_DIR}/review_manifest.yaml" \
  --report-out "${RESULT_DIR}/review_report.md" | tee -a "${RESULT_DIR}/commands.log"
```

Then perform a real review pass:

- Promote usable contracts to `accept_for_generation`.
- Target at least 10 accepted contracts if quality permits.
- Keep plausible but underspecified contracts as `needs_revision`.
- Use `quarantine` for provenance/contamination ambiguity.
- Use `reject` for incoherent, non-analog, schema-invalid, or non-Verilog-A
  suitable tasks.
- For every accepted item:
  - `blocking_findings: []`
  - `required_edits: []`
  - `allowed_next_stage: artifact_generation`
  - rationale explaining why draft artifact generation is worth trying.
- Update `review_report.md` to match final decisions.

Validate the edited review manifest:

```bash
python3 -m train.pipelines.validate_contract_review_manifest \
  --review-manifest "${RESULT_DIR}/review_manifest.yaml" \
  --generated-contract-index "${RESULT_DIR}/generated_contract_index.yaml" \
  --expect-count 25 | tee -a "${RESULT_DIR}/commands.log"
```

## Step 5: Prepare Artifact Requests

```bash
python3 -m train.pipelines.prepare_artifact_synthesis_requests \
  --generated-contract-index "${RESULT_DIR}/generated_contract_index.yaml" \
  --review-manifest "${RESULT_DIR}/review_manifest.yaml" \
  --out-dir "${RESULT_DIR}/artifact_requests" | tee -a "${RESULT_DIR}/commands.log"

cat "${RESULT_DIR}/artifact_requests/artifact_synthesis_request_summary.json" | tee -a "${RESULT_DIR}/commands.log"
cat "${RESULT_DIR}/artifact_requests/artifact_synthesis_requests.jsonl"
```

## Step 6: Generate Verilog-A Draft Artifacts

For every artifact request, create:

```text
${RESULT_DIR}/artifacts/<generated_contract_id>/solution.va
```

Optional helpful files:

```text
${RESULT_DIR}/artifacts/<generated_contract_id>/tb.scs
${RESULT_DIR}/artifacts/<generated_contract_id>/checker.yaml
${RESULT_DIR}/artifacts/<generated_contract_id>/buggy.va
${RESULT_DIR}/artifacts/<generated_contract_id>/fixed.va
${RESULT_DIR}/artifacts/<generated_contract_id>/notes.md
```

Artifact requirements:

- `solution.va` is required for each accepted contract and is the supervised
  draft target used by `pack_draft_training`.
- Use clean-room behavioral Verilog-A only.
- Keep to voltage-domain/event-driven subset.
- Avoid forbidden constructs listed in the contract.
- Avoid task-ID special casing, benchmark IDs, release paths, hidden checker
  logic, or hardcoded one-trace answers.
- Include `module ... endmodule`.
- Prefer simple, readable, EVAS-friendly constructs over complex analog
  operators.
- For `bugfix` contracts, include `buggy.va` and `fixed.va` if possible, but
  `solution.va` must contain the fixed target.
- For `tb` contracts, `solution.va` may be a Verilog-A stimulus/measurement
  helper and `tb.scs` may provide the surrounding Spectre-style harness.

Do not run model training after generating artifacts.

## Step 7: Build Artifact Candidate Index

```bash
python3 -m train.pipelines.write_artifact_candidate_index \
  --generated-contract-index "${RESULT_DIR}/generated_contract_index.yaml" \
  --review-manifest "${RESULT_DIR}/review_manifest.yaml" \
  --artifact-root "${RESULT_DIR}/artifacts" \
  --out "${RESULT_DIR}/artifact_candidate_index.yaml" \
  --expect-min-candidates 5 | tee -a "${RESULT_DIR}/commands.log"

python3 -m train.pipelines.validate_artifact_candidate_index \
  --candidate-index "${RESULT_DIR}/artifact_candidate_index.yaml" \
  --expect-min-candidates 5 | tee -a "${RESULT_DIR}/commands.log"
```

If this fails because an accepted contract lacks `solution.va`, either add the
missing artifact or demote that contract in `review_manifest.yaml`, then rerun
review validation and candidate indexing.

## Step 8: Pack Draft SFT/GRPO Training JSONL

This creates draft training files for inspection. They are not admitted training
data and must not be used for final claims before later gates.

```bash
python3 -m train.pipelines.pack_draft_training \
  --candidate-index "${RESULT_DIR}/artifact_candidate_index.yaml" \
  --out-dir "${RESULT_DIR}/draft_training" \
  --manifest-out "${RESULT_DIR}/draft_training_pack_manifest.yaml" \
  --run-id "${RUN_ID}" \
  --validation-every 5 | tee -a "${RESULT_DIR}/commands.log"
```

Expected files:

```text
${RESULT_DIR}/draft_training/draft_sft/train.jsonl
${RESULT_DIR}/draft_training/draft_sft/val.jsonl
${RESULT_DIR}/draft_training/draft_grpo/prompts.jsonl
${RESULT_DIR}/draft_training_pack_manifest.yaml
```

## Step 9: Status and Summary Files

Write:

```bash
printf "PASS\n" > "${RESULT_DIR}/status.txt"

python3 - <<'PY'
import json
import os
from pathlib import Path
import yaml

result_dir = Path(os.environ["RESULT_DIR"])
index = yaml.safe_load((result_dir / "generated_contract_index.yaml").read_text())
review = yaml.safe_load((result_dir / "review_manifest.yaml").read_text())
artifact_summary = json.loads(
    (result_dir / "artifact_requests" / "artifact_synthesis_request_summary.json").read_text()
)
candidate_index = yaml.safe_load((result_dir / "artifact_candidate_index.yaml").read_text())
draft_pack = yaml.safe_load((result_dir / "draft_training_pack_manifest.yaml").read_text())

summary = {
    "run_id": result_dir.name,
    "status": "PASS",
    "contract_count": index["summary"]["contract_count"],
    "review_decisions": review["summary"]["by_decision"],
    "accepted_for_generation": review["summary"]["accepted_for_generation"],
    "needs_revision": review["summary"]["needs_revision"],
    "artifact_request_count": artifact_summary["request_count"],
    "artifact_candidate_count": len(candidate_index["items"]),
    "draft_sft_train_count": draft_pack["counts"]["sft_train"],
    "draft_sft_val_count": draft_pack["counts"]["sft_val"],
    "draft_grpo_prompt_count": draft_pack["counts"]["grpo_prompts"],
    "generated_contract_index_hash": index["generated_contract_index_hash"],
    "contract_review_manifest_hash": review["contract_review_manifest_hash"],
    "artifact_request_jsonl_hash": artifact_summary["request_jsonl_hash"],
    "artifact_candidate_index_hash": candidate_index["candidate_index_hash"],
    "draft_training_pack_hash": draft_pack["draft_training_pack_hash"],
    "llm_api_called": False,
    "verilog_a_generated": True,
    "draft_training_data_generated": True,
    "admitted_training_data_generated": False,
    "sft_run": False,
    "grpo_run": False,
    "evas_run": False,
    "spectre_run": False,
    "checkpoint_written": False,
}
(result_dir / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
PY
```

## Step 10: Commit Results

Commit only the result directory. Use `git add -f` for `commands.log` and JSONL
files if needed because logs or generated data may be ignored.

```bash
git status -sb
git add "${RESULT_DIR}/status.txt" \
  "${RESULT_DIR}/summary.json" \
  "${RESULT_DIR}/generated_contract_index.yaml" \
  "${RESULT_DIR}/review_manifest.yaml" \
  "${RESULT_DIR}/review_report.md" \
  "${RESULT_DIR}/artifact_candidate_index.yaml" \
  "${RESULT_DIR}/draft_training_pack_manifest.yaml" \
  "${RESULT_DIR}/contracts" \
  "${RESULT_DIR}/artifact_requests" \
  "${RESULT_DIR}/artifacts" \
  "${RESULT_DIR}/draft_training"
git add -f "${RESULT_DIR}/commands.log" "${RESULT_DIR}/draft_training"

git commit -m "Upload overnight draft artifact training result ${RUN_ID}" \
  -m "Constraint: GitHub-mediated remote handoff with draft Verilog-A artifacts and draft unadmitted SFT/GRPO JSONL only." \
  -m "Confidence: medium" \
  -m "Scope-risk: moderate" \
  -m "Directive: Do not treat draft training packs as admitted data until contamination, EVAS, Spectre-shadow policy, and admitted manifests pass." \
  -m "Tested: validate_synthesis_run; prepare_contract_synthesis_requests; write_generated_contract_index; validate_generated_contracts; write_contract_review_manifest; validate_contract_review_manifest; prepare_artifact_synthesis_requests; write_artifact_candidate_index; validate_artifact_candidate_index; pack_draft_training." \
  -m "Not-tested: EVAS, Spectre, SFT training, GRPO training, and model loading intentionally not run."
git push origin training-paper-design-20260620
```

## Final Response Contract

Return one final response with exactly these fields:

```text
RUN_ID=<run id>
COMMIT=<commit hash>
VALIDATION_OUTPUT=<stdout from validation/preparation/packing commands>
SUMMARY_JSON=<full summary.json>
REVIEW_SUMMARY=<decision counts and notable blockers>
ARTIFACT_SUMMARY=<artifact candidate count plus generated file types>
DRAFT_TRAINING_SUMMARY=<full draft_training_pack_manifest counts and hashes>
NEXT_RISKS=<contamination/EVAS/Spectre/admission gaps>
```

If the job fails, still commit the small result directory if it contains useful
diagnostics and return `status=FAIL` with the failing command. Do not commit
checkpoints, simulator dumps, package caches, `.env`, or secret files under any
circumstance.
