#!/usr/bin/env bash
set -euo pipefail

JOB_PATH="train/infra/jobs/phase1_tiny_sft_smoke.yaml"
RUN_ID="tiny-sft-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="/data/jinzhihong/vaEVAS/models/base/Qwen2.5-Coder-7B-Instruct-vaevas"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0}"
MAX_STEPS="2"
MAX_LENGTH="1024"
TRAIN_RECORDS="4"
EVAL_RECORDS="1"
DRY_RUN="0"
WORK_ROOT="${TMPDIR:-/tmp}/vaevas-training-paper-design-tiny-sft"

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
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --cuda-visible-devices)
      CUDA_VISIBLE_DEVICES_VALUE="$2"
      shift 2
      ;;
    --max-steps)
      MAX_STEPS="$2"
      shift 2
      ;;
    --max-length)
      MAX_LENGTH="$2"
      shift 2
      ;;
    --train-records)
      TRAIN_RECORDS="$2"
      shift 2
      ;;
    --eval-records)
      EVAL_RECORDS="$2"
      shift 2
      ;;
    --work-root)
      WORK_ROOT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="1"
      shift
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
WORK_DIR="$WORK_ROOT/$RUN_ID"
ARTIFACT_DIR="$WORK_DIR/artifacts"
SFT_DIR="$ARTIFACT_DIR/sft"
COMMAND_LOG="$RESULT_DIR/commands.log"
ENV_LOG="$RESULT_DIR/environment.txt"

mkdir -p "$SFT_DIR" "$RESULT_DIR"
: > "$COMMAND_LOG"

log_cmd() {
  echo "\$ $*" | tee -a "$COMMAND_LOG"
  "$@" 2>&1 | tee -a "$COMMAND_LOG"
}

{
  echo "run_id=$RUN_ID"
  echo "job_path=$JOB_PATH"
  echo "repo_root=$REPO_ROOT"
  echo "work_dir=$WORK_DIR"
  echo "python_bin=$PYTHON_BIN"
  echo "model_path=$MODEL_PATH"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES_VALUE"
  echo "max_steps=$MAX_STEPS"
  echo "dry_run=$DRY_RUN"
  echo "utc_time=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  echo "git_commit=$(git rev-parse HEAD 2>/dev/null || true)"
  echo "python=$($PYTHON_BIN --version 2>&1)"
} > "$ENV_LOG"

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

log_cmd "$PYTHON_BIN" -m train.pipelines.pack_sft \
  --admitted-manifest "$ARTIFACT_DIR/admitted_manifest.yaml" \
  --candidate-index train/data/manifests/examples/candidate_index.synth-batch-toy-0001.yaml \
  --out-dir "$SFT_DIR" \
  --manifest-out "$ARTIFACT_DIR/sft_pack_manifest.yaml" \
  --run-id "$RUN_ID-sft-pack" \
  --base-model-ref "$MODEL_PATH" \
  --tokenizer-ref "$MODEL_PATH"

TRAIN_ARGS=(
  -m train.train_sft.tiny_sft_smoke
  --sft-train-jsonl "$SFT_DIR/train.jsonl"
  --result-dir "$RESULT_DIR"
  --work-dir "$WORK_DIR"
  --run-id "$RUN_ID"
  --model-path "$MODEL_PATH"
  --max-steps "$MAX_STEPS"
  --max-length "$MAX_LENGTH"
  --train-records "$TRAIN_RECORDS"
  --eval-records "$EVAL_RECORDS"
)

if [[ "$DRY_RUN" == "1" ]]; then
  TRAIN_ARGS+=(--dry-run)
fi

echo "export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_VALUE" | tee -a "$COMMAND_LOG"
export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE"
log_cmd "$PYTHON_BIN" "${TRAIN_ARGS[@]}"

cp "$ARTIFACT_DIR/sft_pack_manifest.yaml" "$RESULT_DIR/sft_pack_manifest.yaml"
cp "$ARTIFACT_DIR/admitted_manifest.yaml" "$RESULT_DIR/admitted_manifest.yaml"

echo "result_dir=$RESULT_DIR"
echo "summary=$RESULT_DIR/summary.json"
