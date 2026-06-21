# Queue Job: EVAS Rust Bulk Smoke for 25 Draft Artifacts

## Job Metadata

- job_id: phase1_evas_bulk_smoke_contract_batch_0025
- priority: P0
- created_at_utc: 2026-06-21T00:00:00Z
- expected_result_dir: `train/infra/results/evas-bulk-smoke-contract-batch-0025-<YYYYMMDDTHHMMSSZ>`
- finish_with: `train/infra/remote_worker/finish_job.sh`

## Objective

Run a bounded EVAS Rust smoke-validation pass for the 25 draft Verilog-A
artifacts produced in:

```text
train/infra/results/contract-batch-0025-20260621T053934Z/
```

This job should determine which draft artifacts are mechanically runnable or
blocked under the pinned remote EVAS Rust profile. It is not a final functional
correctness gate and must not claim Spectre parity or admitted training data.

## Inputs

- Candidate index:
  `train/infra/results/contract-batch-0025-20260621T053934Z/artifact_candidate_index.yaml`
- Draft artifacts:
  `train/infra/results/contract-batch-0025-20260621T053934Z/artifacts/*/solution.va`
- EVAS profile:
  `train/infra/results/evas-profile-pr12-e1e73c0-rustfix-20260621T065830Z/evas_profile.yaml`
- EVAS checkout:
  `/data/jinzhihong/vaevas-evaluator/EVAS`
- Python:
  `/data/jinzhihong/envs/vaevas-rl/bin/python`
- EVAS commit:
  `e1e73c056ceb91dbda47c5029d8b37f57d9c0fa7`
- EVAS engine:
  `evas-rust`

## Boundaries

- Do not run Spectre.
- Do not run SFT or GRPO training.
- Do not write or commit checkpoints.
- Do not modify the draft candidate artifacts.
- Do not commit raw simulator output directories or waveforms.
- Do not mark any item admitted for training.
- Do not call external LLM APIs from scripts.
- If an artifact lacks enough harness information, mark it
  `not_runnable_no_harness`; do not block the entire job.
- If the EVAS CLI cannot run arbitrary local harnesses, mark the job
  `BLOCKED_EVAS_CLI_ARBITRARY_HARNESS` with CLI help and stop.

## Required Steps

### 1. Setup

```bash
set -euo pipefail

git fetch origin training-paper-design-20260620
git checkout training-paper-design-20260620
git pull --ff-only origin training-paper-design-20260620

RUN_ID=evas-bulk-smoke-contract-batch-0025-$(date -u +%Y%m%dT%H%M%SZ)
RESULT_DIR=train/infra/results/${RUN_ID}
CANDIDATE_INDEX=train/infra/results/contract-batch-0025-20260621T053934Z/artifact_candidate_index.yaml
PROFILE=train/infra/results/evas-profile-pr12-e1e73c0-rustfix-20260621T065830Z/evas_profile.yaml
EVAS_DIR=/data/jinzhihong/vaevas-evaluator/EVAS
PY=/data/jinzhihong/envs/vaevas-rl/bin/python
EVAS_COMMIT=e1e73c056ceb91dbda47c5029d8b37f57d9c0fa7

mkdir -p "${RESULT_DIR}/harnesses" "${RESULT_DIR}/logs"
: > "${RESULT_DIR}/commands.log"
```

Append commands and short stdout/stderr summaries to
`${RESULT_DIR}/commands.log`.

### 2. Verify EVAS Profile

Verify and record:

```bash
cd "${EVAS_DIR}"
source "$HOME/.cargo/env" 2>/dev/null || true
test "$("${PY}" -m evas list >/tmp/evas-list.txt 2>&1; echo $?)" = "0"
test "$(git rev-parse HEAD)" = "${EVAS_COMMIT}"
test -f evas/rust_core/target/release/libevas_rust_core.so
sha256sum evas/rust_core/target/release/libevas_rust_core.so
cat /tmp/evas-list.txt
"${PY}" -m evas --help || true
"${PY}" -m evas run --help || true
```

If the commit or shared library is missing, rebuild only the pinned EVAS Rust
backend. Do not upgrade EVAS to a floating branch.

### 3. Create Minimal Harnesses

For each item in `${CANDIDATE_INDEX}`:

1. Read `artifact_refs.solution_va`.
2. Extract the Verilog-A module name, port list, and simple parameters.
3. Create a minimal smoke testbench under:

   ```text
   ${RESULT_DIR}/harnesses/<candidate_id>/tb.scs
   ```

4. Copy the draft `solution.va` into the same harness directory.
5. Drive all input-like ports with conservative voltage sources and terminate
   output-like ports with high impedance or voltage probes.
6. Use a short transient run, for example `tran tran stop=100n step=100p`, unless
   the contract clearly needs a shorter or longer safe window.

This harness is allowed to be shallow. Its purpose is parser/netlist/scheduler
smoke evidence, not functional scoring.

If a safe harness cannot be inferred, record the item as
`not_runnable_no_harness` and continue.

### 4. Run EVAS Rust Smoke

For every generated harness, run EVAS Rust using the supported CLI form on this
checkout. First discover whether direct `.scs` invocation is supported from
`"${PY}" -m evas --help` and `run --help`.

For each candidate, save a small text log:

```text
${RESULT_DIR}/logs/<candidate_id>.log
```

Keep only compact logs and the harness files. Do not commit EVAS `output/`
directories, CSV waveforms, or plots.

### 5. Write Reports

Write `${RESULT_DIR}/evas_bulk_smoke_report.yaml` with:

```yaml
schema_version: phase1.evas_bulk_smoke.v0.1
run_id: <RUN_ID>
status: PASS | FAIL | BLOCKED_EVAS_CLI_ARBITRARY_HARNESS
candidate_index: <path>
candidate_index_hash: <hash from candidate index>
evaluator_profile:
  profile_ref: <PROFILE>
  evaluator_profile_id: evas-pr12-e1e73c0
  git_commit: e1e73c056ceb91dbda47c5029d8b37f57d9c0fa7
  shared_library_sha256: <sha256>
boundaries:
  spectre_run: false
  sft_run: false
  grpo_run: false
  checkpoints_written: false
  admitted_training_data_generated: false
summary:
  total_candidates: 25
  harness_generated: <int>
  evas_pass: <int>
  evas_fail: <int>
  not_runnable_no_harness: <int>
  blocked: <int>
items:
  - candidate_id: <id>
    contract_id: <id>
    level: <L0|L1|L2>
    task_form: <form>
    artifact_ref: <solution.va path>
    harness_ref: <harness path or null>
    log_ref: <log path or null>
    status: evas_pass | evas_fail | not_runnable_no_harness | blocked
    command: <exact command or null>
    return_code: <int or null>
    diagnostic: <short diagnostic>
```

Write `${RESULT_DIR}/summary.json` with the same counts, hashes, EVAS commit,
and booleans:

```json
{
  "status": "PASS",
  "candidate_count": 25,
  "harness_generated": 0,
  "evas_pass": 0,
  "evas_fail": 0,
  "not_runnable_no_harness": 0,
  "blocked": 0,
  "spectre_run": false,
  "sft_run": false,
  "grpo_run": false,
  "checkpoints_written": false,
  "admitted_training_data_generated": false
}
```

Also write `${RESULT_DIR}/status.txt` as `PASS`, `FAIL`, or the blocked status.

### 6. Commit and Finish Queue Job

Use `finish_job.sh`; do not make a separate ad hoc queue-state commit.

For pass or partial pass:

```bash
printf "PASS\n" > "${RESULT_DIR}/status.txt"
bash train/infra/remote_worker/finish_job.sh \
  --status done \
  --job "<JOB_FILE_FROM_POLL_ONCE>" \
  --result "${RESULT_DIR}" \
  --summary "EVAS bulk smoke completed for contract-batch-0025; see summary.json"
```

For hard blocker:

```bash
printf "BLOCKED_EVAS_CLI_ARBITRARY_HARNESS\n" > "${RESULT_DIR}/status.txt"
bash train/infra/remote_worker/finish_job.sh \
  --status failed \
  --job "<JOB_FILE_FROM_POLL_ONCE>" \
  --result "${RESULT_DIR}" \
  --summary "Blocked before candidate smoke; EVAS CLI arbitrary harness support unclear"
```

## Final Response Contract

Return one final response with:

```text
JOB_FILE=<running queue job path>
FINISH_STATUS=<done|failed>
COMMIT=<finish commit hash>
RESULT_DIR=<result dir>
SUMMARY_JSON=<full summary.json>
COUNTS=<harness_generated/evas_pass/evas_fail/not_runnable_no_harness/blocked>
PROFILE=<EVAS commit and shared library hash>
VALIDATION=<EVAS profile verification and smoke command summary>
NEXT_RISKS=<Spectre shadow audit still local; this is not admission>
```


## Worker Claim

- claimed_at_utc: 2026-06-21T10:02:49Z
- worker_host: huaxiyun085
- worker_user: jinzhihong
- branch: training-paper-design-20260620
- repo_commit_at_claim: abcd32844eb85ba1021c7ff4ea8daaf98bd4715b

## Worker Completion

- finished_at_utc: 2026-06-21T12:07:44Z
- finish_status: done
- worker_host: huaxiyun085
- repo_commit_before_finish: 584899c51026630d9222d1fbee0c393d00da0a07
- summary: EVAS bulk smoke completed for contract-batch-0025; see summary.json
- result_paths:
  - train/infra/results/evas-bulk-smoke-contract-batch-0025-20260621T100418Z
