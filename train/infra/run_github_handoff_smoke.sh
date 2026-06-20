#!/usr/bin/env bash
set -euo pipefail

JOB_PATH="train/infra/jobs/phase1_github_handoff_smoke.yaml"
RUN_ID="server-handoff-$(date -u +%Y%m%dT%H%M%SZ)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job)
      JOB_PATH="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

RESULT_DIR="train/infra/results/$RUN_ID"
ARTIFACT_DIR="$RESULT_DIR/artifacts"
SFT_DIR="$ARTIFACT_DIR/sft"
GRPO_DIR="$ARTIFACT_DIR/grpo"
COMMAND_LOG="$RESULT_DIR/commands.log"
ENV_LOG="$RESULT_DIR/environment.txt"
SUMMARY_JSON="$RESULT_DIR/summary.json"
STATUS_FILE="$RESULT_DIR/status.txt"

mkdir -p "$SFT_DIR" "$GRPO_DIR"
: > "$COMMAND_LOG"

log_cmd() {
  echo "\$ $*" | tee -a "$COMMAND_LOG"
  "$@" 2>&1 | tee -a "$COMMAND_LOG"
}

write_environment() {
  {
    echo "run_id=$RUN_ID"
    echo "job_path=$JOB_PATH"
    echo "repo_root=$REPO_ROOT"
    echo "utc_time=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "hostname=$(hostname 2>/dev/null || true)"
    echo "git_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || true)"
    echo "python=$($PYTHON_BIN --version 2>&1)"
    echo
    echo "[gpu]"
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv || true
    else
      echo "nvidia-smi not found"
    fi
    echo
    echo "[python-imports]"
    "$PYTHON_BIN" - <<'PY' || true
packages = ["yaml", "pydantic", "torch", "transformers", "trl", "datasets", "accelerate", "vllm", "verl", "ray", "flash_attn"]
for name in packages:
    try:
        module = __import__(name)
        print(f"{name}=OK version={getattr(module, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"{name}=MISSING {type(exc).__name__}: {str(exc)[:160]}")
PY
  } > "$ENV_LOG"
}

write_environment

log_cmd "$PYTHON_BIN" -m train.pipelines.validate_manifest_fixtures

log_cmd "$PYTHON_BIN" -m train.pipelines.build_protected_index \
  --root train/data/contracts/examples \
  --out "$ARTIFACT_DIR/protected_index.yaml" \
  --benchmark-release-id clean-room-contract-smoke \
  --source-manifest-ref data/contracts/schema.yaml

log_cmd "$PYTHON_BIN" -m train.pipelines.check_contamination \
  --protected-index train/data/manifests/examples/protected_index.yaml \
  --candidate-index train/data/manifests/examples/candidate_index.synth-batch-toy-0001.yaml \
  --out "$ARTIFACT_DIR/contamination_clean.yaml"

log_cmd "$PYTHON_BIN" -m train.pipelines.check_contamination \
  --protected-index "$ARTIFACT_DIR/protected_index.yaml" \
  --candidate-index train/data/manifests/examples/candidate_index.synth-batch-toy-0001.yaml \
  --out "$ARTIFACT_DIR/contamination_reject.yaml"

log_cmd "$PYTHON_BIN" -m train.pipelines.write_admitted_manifest \
  --candidate-index train/data/manifests/examples/candidate_index.synth-batch-toy-0001.yaml \
  --static-check-report train/data/manifests/examples/static_check_report.synth-batch-toy-0001.yaml \
  --evas-report train/data/manifests/examples/evas_verification_report.synth-batch-toy-0001.yaml \
  --contamination-report train/data/manifests/examples/contamination_report.synth-batch-toy-0001.yaml \
  --diversity-report train/data/manifests/examples/diversity_report.synth-batch-toy-0001.yaml \
  --split-manifest train/data/manifests/examples/split_manifest.synth-batch-toy-0001.yaml \
  --spectre-report train/data/manifests/examples/spectre_shadow_report.audit-toy-0001.yaml \
  --require-spectre-shadow \
  --out "$ARTIFACT_DIR/admitted_manifest.yaml"

log_cmd "$PYTHON_BIN" -m train.pipelines.write_admitted_manifest \
  --candidate-index train/data/manifests/examples/candidate_index.synth-batch-toy-0001.yaml \
  --static-check-report train/data/manifests/examples/static_check_report.synth-batch-toy-0001.yaml \
  --evas-report train/data/manifests/examples/evas_verification_report.synth-batch-toy-0001.yaml \
  --contamination-report "$ARTIFACT_DIR/contamination_reject.yaml" \
  --diversity-report train/data/manifests/examples/diversity_report.synth-batch-toy-0001.yaml \
  --split-manifest train/data/manifests/examples/split_manifest.synth-batch-toy-0001.yaml \
  --spectre-report train/data/manifests/examples/spectre_shadow_report.audit-toy-0001.yaml \
  --require-spectre-shadow \
  --out "$ARTIFACT_DIR/admitted_manifest_reject.yaml"

log_cmd "$PYTHON_BIN" -m train.pipelines.pack_sft \
  --admitted-manifest "$ARTIFACT_DIR/admitted_manifest.yaml" \
  --candidate-index train/data/manifests/examples/candidate_index.synth-batch-toy-0001.yaml \
  --out-dir "$SFT_DIR" \
  --manifest-out "$ARTIFACT_DIR/sft_pack_manifest.yaml" \
  --run-id "$RUN_ID-sft"

log_cmd "$PYTHON_BIN" -m train.pipelines.pack_grpo \
  --admitted-manifest "$ARTIFACT_DIR/admitted_manifest.yaml" \
  --candidate-index train/data/manifests/examples/candidate_index.synth-batch-toy-0001.yaml \
  --out-dir "$GRPO_DIR" \
  --manifest-out "$ARTIFACT_DIR/grpo_prompt_manifest.yaml" \
  --run-id "$RUN_ID-grpo"

"$PYTHON_BIN" - "$RESULT_DIR" "$RUN_ID" "$JOB_PATH" <<'PY'
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

result_dir = Path(sys.argv[1])
run_id = sys.argv[2]
job_path = sys.argv[3]
artifact_dir = result_dir / "artifacts"

clean = yaml.safe_load((artifact_dir / "contamination_clean.yaml").read_text())
reject = yaml.safe_load((artifact_dir / "contamination_reject.yaml").read_text())
admitted = yaml.safe_load((artifact_dir / "admitted_manifest.yaml").read_text())
admission_reject = yaml.safe_load((artifact_dir / "admitted_manifest_reject.yaml").read_text())
sft_manifest = yaml.safe_load((artifact_dir / "sft_pack_manifest.yaml").read_text())
grpo_manifest = yaml.safe_load((artifact_dir / "grpo_prompt_manifest.yaml").read_text())

forbidden = {"output", "completion", "gold_completion", "answer"}
prompt_path = artifact_dir / "grpo" / "prompts.jsonl"
prompt_lines = 0
no_answer_leak = True
for line in prompt_path.read_text().splitlines():
    prompt_lines += 1
    record = json.loads(line)
    if forbidden & set(record):
        no_answer_leak = False

sft_train_path = artifact_dir / "sft" / "train.jsonl"
sft_train_lines = sum(1 for _ in sft_train_path.open())

def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


summary = {
    "schema_version": "phase1.handoff_result.v0.1",
    "run_id": run_id,
    "job_path": job_path,
    "status": "PASS",
    "git": {
        "branch": git_value("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": git_value("rev-parse", "HEAD"),
    },
    "pipeline": {
        "validate_fixture": True,
        "protected_index_hash": yaml.safe_load((artifact_dir / "protected_index.yaml").read_text())["protected_index_hash"],
    },
    "contamination": {
        "clean_count": clean["summary"]["clean"],
        "reject_count": reject["summary"]["rejected"],
    },
    "admission": {
        "admitted_count": admitted["summary"]["total"],
        "rejected_count": admitted["summary"]["rejected"],
        "reject_control_count": admission_reject["summary"]["rejected"],
        "admitted_manifest_hash": admitted["admitted_manifest_hash"],
    },
    "sft": {
        "train_lines": sft_train_lines,
        "train_hash": sft_manifest["output_hashes"]["train"],
        "val_hash": sft_manifest["output_hashes"]["val"],
    },
    "grpo": {
        "prompt_lines": prompt_lines,
        "prompt_hash": grpo_manifest["output_hashes"]["prompts"],
        "no_answer_leak": no_answer_leak,
    },
}

assert summary["contamination"]["clean_count"] == 1, summary
assert summary["contamination"]["reject_count"] == 1, summary
assert summary["admission"]["admitted_count"] == 1, summary
assert summary["admission"]["reject_control_count"] == 1, summary
assert summary["sft"]["train_lines"] == 1, summary
assert summary["grpo"]["prompt_lines"] == 1, summary
assert summary["grpo"]["no_answer_leak"] is True, summary

(result_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
(result_dir / "status.txt").write_text("PASS\n")
print("github_handoff_smoke_ok=1")
print(f"result_dir={result_dir}")
PY

echo "PASS" > "$STATUS_FILE"
echo "result_dir=$RESULT_DIR"
echo "summary=$SUMMARY_JSON"
