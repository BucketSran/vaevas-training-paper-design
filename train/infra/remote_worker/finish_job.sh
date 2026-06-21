#!/usr/bin/env bash
set -euo pipefail

BRANCH="${BRANCH:-training-paper-design-20260620}"
REMOTE="${REMOTE:-origin}"
QUEUE_ROOT="${QUEUE_ROOT:-train/infra/queue}"
STATUS=""
JOB_FILE=""
SUMMARY=""
PUSH="1"
RESULT_PATHS=()

usage() {
  cat <<'EOF'
Usage: finish_job.sh --status done|failed --job train/infra/queue/running/<job>.md [--result PATH ...] [--summary TEXT] [--no-push]

Move a claimed queue job to done/ or failed/, append completion metadata, add
small result evidence, commit, and push.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --status)
      STATUS="$2"
      shift 2
      ;;
    --job)
      JOB_FILE="$2"
      shift 2
      ;;
    --result)
      RESULT_PATHS+=("$2")
      shift 2
      ;;
    --summary)
      SUMMARY="$2"
      shift 2
      ;;
    --branch)
      BRANCH="$2"
      shift 2
      ;;
    --remote)
      REMOTE="$2"
      shift 2
      ;;
    --no-push)
      PUSH="0"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$STATUS" != "done" && "$STATUS" != "failed" ]]; then
  echo "--status must be done or failed" >&2
  usage >&2
  exit 2
fi

if [[ -z "$JOB_FILE" || ! -f "$JOB_FILE" ]]; then
  echo "--job must point to an existing running queue job" >&2
  usage >&2
  exit 2
fi

case "$JOB_FILE" in
  "$QUEUE_ROOT"/running/*.md) ;;
  *)
    echo "--job must be under $QUEUE_ROOT/running/" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p "$QUEUE_ROOT/done" "$QUEUE_ROOT/failed"

DEST="$QUEUE_ROOT/$STATUS/$(basename "$JOB_FILE")"
FINISHED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
HEAD_COMMIT="$(git rev-parse HEAD)"
WORKER_HOST="$(hostname 2>/dev/null || echo unknown-host)"

TMP_FILE="$(mktemp)"
cat "$JOB_FILE" > "$TMP_FILE"
{
  echo
  echo "## Worker Completion"
  echo
  echo "- finished_at_utc: $FINISHED_AT"
  echo "- finish_status: $STATUS"
  echo "- worker_host: $WORKER_HOST"
  echo "- repo_commit_before_finish: $HEAD_COMMIT"
  if [[ -n "$SUMMARY" ]]; then
    echo "- summary: $SUMMARY"
  fi
  if [[ "${#RESULT_PATHS[@]}" -gt 0 ]]; then
    echo "- result_paths:"
    for RESULT_PATH in "${RESULT_PATHS[@]}"; do
      echo "  - $RESULT_PATH"
    done
  fi
} >> "$TMP_FILE"

git mv "$JOB_FILE" "$DEST"
cat "$TMP_FILE" > "$DEST"
rm -f "$TMP_FILE"

git add "$QUEUE_ROOT/running" "$QUEUE_ROOT/$STATUS" "$DEST"

for RESULT_PATH in "${RESULT_PATHS[@]}"; do
  case "$RESULT_PATH" in
    train/infra/results/*|train/logs/*)
      if [[ -e "$RESULT_PATH" ]]; then
        git add -f "$RESULT_PATH"
      else
        echo "result path does not exist: $RESULT_PATH" >&2
        exit 2
      fi
      ;;
    *)
      echo "refusing to add result outside train/infra/results or train/logs: $RESULT_PATH" >&2
      exit 2
      ;;
  esac
done

JOB_ID="$(basename "$JOB_FILE" .md)"

git commit -m "Finish remote queue job $JOB_ID as $STATUS" \
  -m "Constraint: GitHub-mediated remote worker queue; only queue state and small declared evidence are committed." \
  -m "Confidence: medium" \
  -m "Scope-risk: narrow" \
  -m "Directive: Treat failed queue jobs as diagnostic evidence, not as missing work; create a new pending job for retries." \
  -m "Tested: finish_job.sh moved $JOB_ID to $STATUS and added declared result paths." \
  -m "Not-tested: Job-specific scientific claims require the result summary and validation logs."

if [[ "$PUSH" == "1" ]]; then
  git push "$REMOTE" "$BRANCH"
fi

echo "FINISHED=1"
echo "STATUS=$STATUS"
echo "JOB_FILE=$DEST"
echo "COMMIT=$(git rev-parse HEAD)"
if [[ "${#RESULT_PATHS[@]}" -gt 0 ]]; then
  printf 'RESULT_PATHS='
  printf '%s ' "${RESULT_PATHS[@]}"
  printf '\n'
fi

