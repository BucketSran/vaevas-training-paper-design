# Remote Codex Task: EVAS Rust Toolchain Fix

Fix the blocker from:

```text
train/infra/results/evas-profile-pr12-e1e73c0-20260621T064622Z/
```

The prior run synchronized the correct EVAS commit and installed the Python
package into `/data/jinzhihong/envs/vaevas-rl/bin/python`, but `cargo` and
`rustc` were not available, so the Rust backend shared library was not built.

## Objective

Install or expose a user-local Rust toolchain, rebuild EVAS PR12 at the exact
commit, rerun the `evas-rust` smoke, and upload a new profile result directory.

## Fixed EVAS Version

- Repo: `https://github.com/BucketSran/EVAS.git`
- Branch: `codex/stochastic-semantics-and-cross-law`
- Commit: `e1e73c056ceb91dbda47c5029d8b37f57d9c0fa7`
- Upstream PR: `https://github.com/Arcadia-1/EVAS/pull/12`
- Python env: `/data/jinzhihong/envs/vaevas-rl/bin/python`

Do not use a floating branch for the final profile. Always verify:

```bash
test "$(git rev-parse HEAD)" = "e1e73c056ceb91dbda47c5029d8b37f57d9c0fa7"
```

## Boundaries

- Do not use `sudo`.
- Do not run SFT or GRPO training.
- Do not run Spectre.
- Do not write checkpoints.
- Do not modify files outside the EVAS checkout and the result directory.
- If Rust cannot be installed or exposed without admin privileges, commit a
  `BLOCKED_RUST_TOOLCHAIN` report with the attempted commands and stop.

## Setup

```bash
set -euo pipefail

PY=/data/jinzhihong/envs/vaevas-rl/bin/python
EVAS_COMMIT=e1e73c056ceb91dbda47c5029d8b37f57d9c0fa7
WORKDIR=/data/jinzhihong/vaevas-evaluator
EVAS_DIR="${WORKDIR}/EVAS"
RUN_ID=evas-profile-pr12-e1e73c0-rustfix-$(date -u +%Y%m%dT%H%M%SZ)
RESULT_DIR=train/infra/results/${RUN_ID}

mkdir -p "${WORKDIR}" "${RESULT_DIR}"
: > "${RESULT_DIR}/commands.log"
```

## Step 1: Sync EVAS to Exact Commit

```bash
cd "${WORKDIR}"
if [ ! -d "${EVAS_DIR}/.git" ]; then
  git clone https://github.com/BucketSran/EVAS.git "${EVAS_DIR}"
fi

cd "${EVAS_DIR}"
git fetch origin codex/stochastic-semantics-and-cross-law
git checkout --detach "${EVAS_COMMIT}"
test "$(git rev-parse HEAD)" = "${EVAS_COMMIT}"
```

## Step 2: Find or Install Rust Without sudo

First try existing locations:

```bash
set +e
command -v cargo
command -v rustc
find "$HOME" /data/jinzhihong -maxdepth 4 -type f \( -name cargo -o -name rustc \) 2>/dev/null | head -50
set -e
```

If a valid `cargo` is found, prepend its parent directories to `PATH`.

If no Rust toolchain exists and outbound network is available, install Rust in
user space:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup-init.sh
sh /tmp/rustup-init.sh -y --profile minimal --default-toolchain stable
source "$HOME/.cargo/env"
```

If `curl` or the network fails, try the package manager for the existing user
environment only if it is already installed:

```bash
set +e
command -v micromamba
command -v mamba
command -v conda
set -e
```

Use one of these only if available, targeting the existing env:

```bash
micromamba install -y -n vaevas-rl -c conda-forge rust
# or
mamba install -y -n vaevas-rl -c conda-forge rust
# or
conda install -y -n vaevas-rl -c conda-forge rust
```

After any path, verify:

```bash
cargo --version
rustc --version
```

## Step 3: Reinstall Python Package and Build Rust Backend

```bash
cd "${EVAS_DIR}"
"${PY}" -m pip install -e ".[dev]"
cargo build --manifest-path evas/rust_core/Cargo.toml --release
test -f evas/rust_core/target/release/libevas_rust_core.so
```

If the shared library extension is different on this host, record the actual
filename under `evas/rust_core/target/release/`.

## Step 4: Smoke Tests

```bash
"${PY}" -m evas list
"${PY}" -m evas run clk_div --engine evas-rust
```

Optional, only if the first smoke passes:

```bash
"${PY}" -m pytest tests -q -m rust_backend
```

## Step 5: Write Result Files

Write `evas_profile.yaml` and `evas_build_report.json` under `${RESULT_DIR}`.
The profile must include:

```yaml
evaluator_profile_id: evas-pr12-e1e73c0
evaluator_kind: evas_rust
repo: https://github.com/BucketSran/EVAS.git
upstream_pr: https://github.com/Arcadia-1/EVAS/pull/12
branch: codex/stochastic-semantics-and-cross-law
git_commit: e1e73c056ceb91dbda47c5029d8b37f57d9c0fa7
rust_build_command: cargo build --manifest-path evas/rust_core/Cargo.toml --release
engine_selector: evas-rust
status: built_smoke_tested
```

If blocked, set:

```yaml
status: blocked_missing_rust_toolchain
```

and include the exact failed install/build commands.

## Step 6: Commit and Return

Commit only the result directory:

```bash
git status -sb
git add "${RESULT_DIR}/evas_profile.yaml" "${RESULT_DIR}/evas_build_report.json"
git add -f "${RESULT_DIR}/commands.log"
git commit -m "Upload EVAS Rust toolchain fix result ${RUN_ID}" \
  -m "Constraint: User-local Rust toolchain only; no sudo, no Spectre, no training, no checkpoints." \
  -m "Confidence: medium" \
  -m "Scope-risk: narrow" \
  -m "Directive: Pin EVAS validation to commit e1e73c056ceb91dbda47c5029d8b37f57d9c0fa7 until a new evaluator profile is explicitly created." \
  -m "Tested: cargo/rustc probe; EVAS editable install; rust backend build; evas list; evas run clk_div --engine evas-rust when available." \
  -m "Not-tested: Spectre, SFT, GRPO, and model checkpoints intentionally not run."
git push origin training-paper-design-20260620
```

Final response must include:

```text
RUN_ID=<run id>
COMMIT=<commit hash or none if blocked before commit>
STATUS=<built_smoke_tested or blocked_missing_rust_toolchain>
CARGO_VERSION=<cargo --version or unavailable>
RUSTC_VERSION=<rustc --version or unavailable>
BUILD_OUTPUT=<summary>
SMOKE_OUTPUT=<summary>
NEXT_BLOCKER=<none or exact blocker>
```
