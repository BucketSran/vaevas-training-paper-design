#!/usr/bin/env bash
set -euo pipefail

BRANCH="${BRANCH:-training-paper-design-20260620}"
REMOTE="${REMOTE:-origin}"
QUEUE_ROOT="${QUEUE_ROOT:-train/infra/queue}"
PUSH="1"

usage() {
  cat <<'EOF'
Usage: poll_once.sh [--branch BRANCH] [--remote REMOTE] [--no-push]

Pull latest branch, claim the oldest pending queue job, commit the claim, push,
and print the running job path plus job body.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

git fetch "$REMOTE" "$BRANCH"
git checkout "$BRANCH"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "BLOCKED_DIRTY_WORKTREE=1"
  git status -sb
  exit 3
fi

git pull --ff-only "$REMOTE" "$BRANCH"

mkdir -p "$QUEUE_ROOT/pending" "$QUEUE_ROOT/running" "$QUEUE_ROOT/done" "$QUEUE_ROOT/failed"

mapfile -t PENDING_JOBS < <(find "$QUEUE_ROOT/pending" -maxdepth 1 -type f -name '*.md' | sort)

if [[ "${#PENDING_JOBS[@]}" -eq 0 ]]; then
  echo "NO_PENDING=1"
  echo "BRANCH=$BRANCH"
  echo "QUEUE_ROOT=$QUEUE_ROOT"
  exit 0
fi

PENDING_JOB="${PENDING_JOBS[0]}"
JOB_BASENAME="$(basename "$PENDING_JOB")"
RUNNING_JOB="$QUEUE_ROOT/running/$JOB_BASENAME"
CLAIMED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
WORKER_HOST="$(hostname 2>/dev/null || echo unknown-host)"
WORKER_USER="${USER:-unknown-user}"
HEAD_COMMIT="$(git rev-parse HEAD)"

TMP_FILE="$(mktemp)"
cat "$PENDING_JOB" > "$TMP_FILE"
{
  echo
  echo "## Worker Claim"
  echo
  echo "- claimed_at_utc: $CLAIMED_AT"
  echo "- worker_host: $WORKER_HOST"
  echo "- worker_user: $WORKER_USER"
  echo "- branch: $BRANCH"
  echo "- repo_commit_at_claim: $HEAD_COMMIT"
} >> "$TMP_FILE"

git mv "$PENDING_JOB" "$RUNNING_JOB"
cat "$TMP_FILE" > "$RUNNING_JOB"
rm -f "$TMP_FILE"

JOB_ID="${JOB_BASENAME%.md}"

git add "$QUEUE_ROOT/pending" "$QUEUE_ROOT/running"
git commit -m "Claim remote queue job $JOB_ID" \
  -m "Constraint: GitHub-mediated remote queue claim only; no experiment execution is performed by poll_once.sh." \
  -m "Confidence: high" \
  -m "Scope-risk: narrow" \
  -m "Directive: Finish this claimed job with train/infra/remote_worker/finish_job.sh so queue state cannot remain ambiguous." \
  -m "Tested: poll_once.sh reached git claim bookkeeping for $JOB_ID." \
  -m "Not-tested: Job-specific validation is deferred to the claimed queue job."

if [[ "$PUSH" == "1" ]]; then
  git push "$REMOTE" "$BRANCH"
fi

echo "CLAIMED=1"
echo "JOB_FILE=$RUNNING_JOB"
echo "JOB_ID=$JOB_ID"
echo "BRANCH=$BRANCH"
echo
echo "----- BEGIN JOB BODY -----"
cat "$RUNNING_JOB"
echo "----- END JOB BODY -----"

