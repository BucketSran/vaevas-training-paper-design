# Queue Job: Repair 10 EVAS Failures from Contract Batch 0025

## Job Metadata

- job_id: phase1_evas_failure_repair_contract_batch_0025
- priority: P0
- created_at_utc: 2026-06-21T00:00:00Z
- expected_result_dir: `train/infra/results/evas-repair-contract-batch-0025-<YYYYMMDDTHHMMSSZ>`
- finish_with: `train/infra/remote_worker/finish_job.sh`

## Objective

Repair the 10 draft Verilog-A artifacts that failed the first EVAS Rust bulk
smoke, then rerun EVAS Rust smoke on all 25 candidates.

If and only if all 25 candidates pass EVAS Rust smoke, rebuild the draft
unadmitted SFT/GRPO packs from the repaired candidate index. These packs remain
draft and not admitted.

## Inputs

- Source draft batch:
  `train/infra/results/contract-batch-0025-20260621T053934Z/`
- Source candidate index:
  `train/infra/results/contract-batch-0025-20260621T053934Z/artifact_candidate_index.yaml`
- Prior EVAS report:
  `train/infra/results/evas-bulk-smoke-contract-batch-0025-20260621T100418Z/evas_bulk_smoke_report.yaml`
- EVAS profile:
  `train/infra/results/evas-profile-pr12-e1e73c0-rustfix-20260621T065830Z/evas_profile.yaml`
- EVAS checkout:
  `/data/jinzhihong/vaevas-evaluator/EVAS`
- Python:
  `/data/jinzhihong/envs/vaevas-rl/bin/python`
- EVAS commit:
  `e1e73c056ceb91dbda47c5029d8b37f57d9c0fa7`

## Boundaries

- Do not run Spectre.
- Do not run SFT or GRPO model training.
- Do not write or commit checkpoints.
- Do not modify files in the original source batch result directory.
- Do not mutate the original `artifact_candidate_index.yaml`.
- Do not commit raw EVAS output directories, CSV waveforms, plots, package
  caches, `.env`, secrets, or model weights.
- Do not mark any data as admitted.
- Do not call external LLM APIs from scripts.
- This is a syntax/subset compatibility repair job. Preserve contract intent;
  do not replace tasks with unrelated easier tasks.

## Known Failures to Repair

### Failure mode A: `transition()` contribution inside conditional/event/loop/case

Five L0 conformance candidates failed because `transition()` is contributed
inside an `if` branch:

- `cand_cg_batch_l0_cross_event_latch_boundary_tolerance_0001`
- `cand_cg_batch_l0_cross_event_latch_observable_clarity_0001`
- `cand_cg_batch_l0_cross_event_latch_ood_split_naming_0001`
- `cand_cg_batch_l0_cross_event_latch_parameter_diversity_0001`
- `cand_cg_batch_l0_cross_event_latch_waveform_coverage_0001`

Repair rule:

- Keep the event latch behavior.
- Keep `@(cross(...))` only for state update.
- Move the contribution out of the conditional branch.
- Use one unconditional contribution in the analog block, e.g. compute or inline
  the target voltage and use:

  ```verilog
  V(vflag) <+ transition(latched ? vhigh : vlow, tdetect, tdetect);
  ```

- Do not put `V(...) <+ transition(...)` inside `if`, event, loop, or `case`.

### Failure mode B: unsupported `$rtoi()`

Five L2 SAR ADC loop candidates failed because EVAS Rust rejected `$rtoi()`:

- `cand_cg_batch_l2_sar_adc_loop_boundary_tolerance_0001`
- `cand_cg_batch_l2_sar_adc_loop_observable_clarity_0001`
- `cand_cg_batch_l2_sar_adc_loop_ood_split_naming_0001`
- `cand_cg_batch_l2_sar_adc_loop_parameter_diversity_0001`
- `cand_cg_batch_l2_sar_adc_loop_waveform_coverage_0001`

Repair rule:

- Preserve the 4-bit SAR/quantization intent.
- Remove `$rtoi()`.
- Prefer an explicit threshold ladder over unsupported conversion functions,
  for example:

  ```verilog
  if (sample_value <= 0.0) code_value = 0;
  else if (sample_value >= vref) code_value = 15;
  else if (sample_value >= (14.5/15.0)*vref) code_value = 15;
  else if (sample_value >= (13.5/15.0)*vref) code_value = 14;
  ...
  else if (sample_value >= (0.5/15.0)*vref) code_value = 1;
  else code_value = 0;
  ```

- Avoid `$rtoi`, `$floor`, `$ceil`, `min`, and `max` unless EVAS smoke proves
  they are accepted.
- Keep output contributions unconditional in the analog block.

## Required Steps

### 1. Setup

```bash
set -euo pipefail

git fetch origin training-paper-design-20260620
git checkout training-paper-design-20260620
git pull --ff-only origin training-paper-design-20260620

RUN_ID=evas-repair-contract-batch-0025-$(date -u +%Y%m%dT%H%M%SZ)
RESULT_DIR=train/infra/results/${RUN_ID}
SOURCE_BATCH=train/infra/results/contract-batch-0025-20260621T053934Z
SOURCE_INDEX=${SOURCE_BATCH}/artifact_candidate_index.yaml
PRIOR_REPORT=train/infra/results/evas-bulk-smoke-contract-batch-0025-20260621T100418Z/evas_bulk_smoke_report.yaml
PROFILE=train/infra/results/evas-profile-pr12-e1e73c0-rustfix-20260621T065830Z/evas_profile.yaml
EVAS_DIR=/data/jinzhihong/vaevas-evaluator/EVAS
PY=/data/jinzhihong/envs/vaevas-rl/bin/python
EVAS_COMMIT=e1e73c056ceb91dbda47c5029d8b37f57d9c0fa7

mkdir -p "${RESULT_DIR}/artifacts" "${RESULT_DIR}/harnesses" "${RESULT_DIR}/logs"
: > "${RESULT_DIR}/commands.log"
```

Append important commands and compact stdout/stderr summaries to
`${RESULT_DIR}/commands.log`.

### 2. Verify EVAS Profile

```bash
cd "${EVAS_DIR}"
source "$HOME/.cargo/env" 2>/dev/null || true
test "$(git rev-parse HEAD)" = "${EVAS_COMMIT}"
test -f evas/rust_core/target/release/libevas_rust_core.so
sha256sum evas/rust_core/target/release/libevas_rust_core.so
"${PY}" -m evas list
"${PY}" -m evas simulate --help
```

If the commit or shared library is wrong, stop with a `failed` queue finish and
write a blocked report. Do not switch EVAS commits.

### 3. Materialize a Repaired Artifact Root

Create a new artifact root under `${RESULT_DIR}/artifacts`.

- For the 15 prior `evas_pass` items, copy the source artifact directory from
  `${SOURCE_BATCH}/artifacts/<contract_id>/`.
- For the 10 prior `evas_fail` items, copy the source artifact directory first,
  then edit only `solution.va` using the repair rules above.
- Add or update `notes.md` for repaired items with:
  - prior failure diagnostic
  - repair class
  - source artifact path
  - statement that this is still draft/unadmitted

Do not edit `${SOURCE_BATCH}` in place.

### 4. Rebuild Candidate Index

Use the existing index writer so hashes point at the repaired artifact root:

```bash
"${PY}" -m train.pipelines.write_artifact_candidate_index \
  --generated-contract-index "${SOURCE_BATCH}/generated_contract_index.yaml" \
  --review-manifest "${SOURCE_BATCH}/review_manifest.yaml" \
  --artifact-root "${RESULT_DIR}/artifacts" \
  --out "${RESULT_DIR}/artifact_candidate_index.yaml" \
  --expect-min-candidates 25 | tee -a "${RESULT_DIR}/commands.log"

"${PY}" -m train.pipelines.validate_artifact_candidate_index \
  --candidate-index "${RESULT_DIR}/artifact_candidate_index.yaml" \
  --expect-min-candidates 25 | tee -a "${RESULT_DIR}/commands.log"
```

### 5. Rerun EVAS Rust Smoke on All 25

Generate minimal harnesses under `${RESULT_DIR}/harnesses/<candidate_id>/` and
compact logs under `${RESULT_DIR}/logs/<candidate_id>.log`, following the same
approach as the previous bulk smoke.

Run each candidate with:

```bash
"${PY}" -m evas simulate "<harness tb.scs>" --engine evas-rust -o "<tmp output dir>"
```

Write `${RESULT_DIR}/evas_bulk_smoke_report.yaml` with the same schema as the
prior report and include all 25 items. Keep output directories under `/tmp` and
do not commit them.

### 6. Pack Draft Training Only if EVAS All-Pass

If `evas_pass == 25` and `evas_fail == 0`, run:

```bash
"${PY}" -m train.pipelines.pack_draft_training \
  --candidate-index "${RESULT_DIR}/artifact_candidate_index.yaml" \
  --out-dir "${RESULT_DIR}/draft_training" \
  --manifest-out "${RESULT_DIR}/draft_training_pack_manifest.yaml" \
  --run-id "${RUN_ID}" \
  --validation-every 5 | tee -a "${RESULT_DIR}/commands.log"
```

If any EVAS failures remain, do not pack training data. Instead, write
`draft_training_pack_manifest.yaml` as absent and set
`draft_training_data_generated: false` in `summary.json`.

### 7. Write Required Reports

Write `${RESULT_DIR}/repair_manifest.yaml`:

```yaml
schema_version: phase1.evas_repair.v0.1
run_id: <RUN_ID>
source_batch: <SOURCE_BATCH>
prior_evas_report: <PRIOR_REPORT>
repair_counts:
  copied_without_change: 15
  repaired_transition_inside_conditional_or_event: 5
  repaired_unsupported_rtoi: 5
items:
  - candidate_id: <id>
    contract_id: <id>
    action: copied_without_change | repaired
    repair_class: none | transition_inside_conditional_or_event | unsupported_rtoi
    source_solution_ref: <path>
    repaired_solution_ref: <path>
    prior_diagnostic: <string or null>
```

Write `${RESULT_DIR}/summary.json`:

```json
{
  "status": "PASS",
  "candidate_count": 25,
  "repair_count": 10,
  "evas_pass": 25,
  "evas_fail": 0,
  "draft_training_data_generated": true,
  "admitted_training_data_generated": false,
  "spectre_run": false,
  "sft_run": false,
  "grpo_run": false,
  "checkpoints_written": false
}
```

Set `status` to `PARTIAL` if any EVAS failures remain. Set `status` to
`BLOCKED` only if the job cannot run the repair/evaluation loop at all.

Write `${RESULT_DIR}/status.txt` with the same status.

### 8. Finish Queue Job

For `PASS` or `PARTIAL`, finish as done:

```bash
bash train/infra/remote_worker/finish_job.sh \
  --status done \
  --job "<JOB_FILE_FROM_POLL_ONCE>" \
  --result "${RESULT_DIR}" \
  --summary "EVAS failure repair completed for contract-batch-0025; see summary.json"
```

For hard blockers, finish as failed:

```bash
bash train/infra/remote_worker/finish_job.sh \
  --status failed \
  --job "<JOB_FILE_FROM_POLL_ONCE>" \
  --result "${RESULT_DIR}" \
  --summary "EVAS failure repair blocked; see summary.json and commands.log"
```

## Final Response Contract

Return one final response with:

```text
JOB_FILE=<running queue job path>
FINISH_STATUS=<done|failed>
COMMIT=<finish commit hash>
RESULT_DIR=<result dir>
SUMMARY_JSON=<full summary.json>
REPAIR_COUNTS=<copied/repaired_transition/repaired_rtoi>
EVAS_COUNTS=<evas_pass/evas_fail/not_runnable_no_harness/blocked>
DRAFT_TRAINING=<generated true|false, counts and hashes if generated>
VALIDATION=<candidate-index validation, EVAS profile verification, EVAS smoke summary>
NEXT_RISKS=<Spectre shadow audit still local; repaired data still unadmitted>
```

