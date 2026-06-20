# Remote Codex Task: Contract Synthesis Smoke

Run one complete Phase 1 contract synthesis smoke and return the final results
in one response. Do not stop for intermediate confirmation unless a command
would be destructive, credential-gated, or changes the compute footprint.

## Purpose

Create exactly five draft contract YAML files from the clean-room pilot
`contract_proposal` requests, validate them mechanically, commit only the small
result directory, and return the final summary.

This is not SFT, not GRPO, not EVAS, not Spectre, and not admitted training data.

## Hard Constraints

- Do not call external LLM APIs.
- Do not generate Verilog-A artifacts.
- Do not run EVAS or Spectre.
- Do not run SFT or GRPO.
- Do not write checkpoints, model weights, simulator outputs, `.env` files, or
  secrets.
- Do not commit `/tmp` request files.
- Commit only `train/infra/results/<run_id>/`.

## Commands and Work

Update the repo:

```bash
cd /data/jinzhihong/vaevas-training-paper-design
git fetch origin
git checkout training-paper-design-20260620
git pull --ff-only
```

Prepare request records:

```bash
RUN_ID=contract-synthesis-smoke-$(date -u +%Y%m%dT%H%M%SZ)
REQUEST_DIR=/tmp/vaevas_${RUN_ID}_requests
RESULT_DIR=train/infra/results/${RUN_ID}

python3 -m train.pipelines.validate_synthesis_run
python3 -m train.pipelines.prepare_contract_synthesis_requests --out-dir "${REQUEST_DIR}"
mkdir -p "${RESULT_DIR}/contracts"
```

Now read:

```bash
cat "${REQUEST_DIR}/contract_synthesis_requests.jsonl"
```

For each of the five JSONL records, write exactly one draft contract YAML under:

```text
${RESULT_DIR}/contracts/
```

Contract requirements:

- YAML only; no Verilog-A code.
- Must match `train/data/contracts/schema.yaml`.
- Must be based on the request's `seed_id`, `category`, `level`, and
  `task_form`.
- Use new contract IDs and split keys; do not reuse vaBench release IDs.
- Set `provenance.source_tier: B_llm_synthetic`.
- Set `provenance.source_kind: llm_synthetic`.
- Put the request `seed_id` in `provenance.seed_refs`.
- Set EVAS status to `not_run` and Spectre shadow status to `pending` or
  `not_run`; do not claim simulator success.
- Set `admission.status: draft_contract`.
- Include blockers for missing manual review, contamination check, and EVAS
  evidence.

Then validate and index:

```bash
python3 -m train.pipelines.write_generated_contract_index \
  --contracts-dir "${RESULT_DIR}/contracts" \
  --request-jsonl "${REQUEST_DIR}/contract_synthesis_requests.jsonl" \
  --out "${RESULT_DIR}/generated_contract_index.yaml" \
  --model-ref remote_codex_default \
  --expect-count 5

python3 -m train.pipelines.validate_generated_contracts \
  --index "${RESULT_DIR}/generated_contract_index.yaml" \
  --expect-count 5

python3 -m train.pipelines.validate_manifest_fixtures
```

Write result files:

```bash
printf "PASS\n" > "${RESULT_DIR}/status.txt"
cat > "${RESULT_DIR}/commands.log" <<EOF
Remote contract synthesis smoke.
RUN_ID=${RUN_ID}
REQUEST_DIR=${REQUEST_DIR}
RESULT_DIR=${RESULT_DIR}
Commands:
- python3 -m train.pipelines.validate_synthesis_run
- python3 -m train.pipelines.prepare_contract_synthesis_requests --out-dir ${REQUEST_DIR}
- wrote five draft contract YAML files under ${RESULT_DIR}/contracts
- python3 -m train.pipelines.write_generated_contract_index --contracts-dir ${RESULT_DIR}/contracts --request-jsonl ${REQUEST_DIR}/contract_synthesis_requests.jsonl --out ${RESULT_DIR}/generated_contract_index.yaml --model-ref remote_codex_default --expect-count 5
- python3 -m train.pipelines.validate_generated_contracts --index ${RESULT_DIR}/generated_contract_index.yaml --expect-count 5
- python3 -m train.pipelines.validate_manifest_fixtures
EOF

python3 - <<PY
import json
from pathlib import Path
import yaml
result_dir = Path("${RESULT_DIR}")
index = yaml.safe_load((result_dir / "generated_contract_index.yaml").read_text())
summary = {
    "status": "PASS",
    "run_id": "${RUN_ID}",
    "contract_count": index["summary"]["contract_count"],
    "review_decisions": index["summary"]["review_decisions"],
    "levels": index["summary"]["levels"],
    "task_forms": index["summary"]["task_forms"],
    "categories": index["summary"]["categories"],
    "synthesis_run_hash": index["synthesis_run_hash"],
    "training_data_admitted": False,
    "verilog_a_generated": False,
    "evas_run": False,
    "spectre_run": False,
    "sft_run": False,
    "grpo_run": False,
}
(result_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\\n")
PY
```

Commit and push only the result directory:

```bash
git status --short
git add "${RESULT_DIR}/status.txt" "${RESULT_DIR}/summary.json" \
  "${RESULT_DIR}/generated_contract_index.yaml" "${RESULT_DIR}/contracts"
git add -f "${RESULT_DIR}/commands.log"
git commit -m "Upload contract synthesis smoke result ${RUN_ID}"
git push origin training-paper-design-20260620
```

## Final Response Required

Return all of this in one response:

```text
RUN_ID=<run id>
COMMIT=<commit hash>
VALIDATION_OUTPUT=<stdout from validate_synthesis_run / prepare requests / write index / validate generated contracts / validate fixtures>
SUMMARY_JSON=<contents of train/infra/results/<run_id>/summary.json>
GENERATED_CONTRACT_INDEX_HEAD=<first 80 lines of generated_contract_index.yaml>
```

If the job fails, still return the full traceback and do not commit partial
outputs unless `generated_contract_index.yaml` was successfully written.
