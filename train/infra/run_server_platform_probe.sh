#!/usr/bin/env bash
set -euo pipefail

JOB_PATH="train/infra/jobs/phase1_server_platform_probe.yaml"
RUN_ID="server-platform-probe-$(date -u +%Y%m%dT%H%M%SZ)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATHS=()

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
      MODEL_PATHS+=("$2")
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ${#MODEL_PATHS[@]} -eq 0 ]]; then
  MODEL_PATHS+=("/data/jinzhihong/vaEVAS/models/base/Qwen2.5-Coder-7B-Instruct-vaevas")
  MODEL_PATHS+=("/data/jinzhihong/vaEVAS/models/base/Qwen2.5-Coder-7B-Instruct")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

RESULT_DIR="train/infra/results/$RUN_ID"
mkdir -p "$RESULT_DIR"

COMMAND_LOG="$RESULT_DIR/commands.log"
: > "$COMMAND_LOG"

{
  echo "run_id=$RUN_ID"
  echo "job_path=$JOB_PATH"
  echo "python_bin=$PYTHON_BIN"
  echo "model_paths=${MODEL_PATHS[*]}"
  echo "repo_root=$REPO_ROOT"
} | tee -a "$COMMAND_LOG" >/dev/null

echo "\$ $PYTHON_BIN - <platform_probe.py>" | tee -a "$COMMAND_LOG"

"$PYTHON_BIN" - "$RESULT_DIR" "$RUN_ID" "$JOB_PATH" "${MODEL_PATHS[@]}" <<'PY' 2>&1 | tee -a "$COMMAND_LOG"
from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


result_dir = Path(sys.argv[1])
run_id = sys.argv[2]
job_path = sys.argv[3]
model_paths = [Path(value) for value in sys.argv[4:]]


def run_command(args: list[str], timeout_s: int = 20) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            args,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
        )
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except FileNotFoundError as exc:
        return {"ok": False, "returncode": None, "stdout": "", "stderr": str(exc)}
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "returncode": None,
            "stdout": (exc.stdout or "").strip() if isinstance(exc.stdout, str) else "",
            "stderr": f"timeout after {timeout_s}s",
        }


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def package_status(name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(name)
    status: dict[str, Any] = {"available": spec is not None, "version": None, "error": None}
    if spec is None:
        return status
    try:
        status["version"] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        try:
            module = __import__(name)
            status["version"] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            status["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    return status


def disk_status(path: Path) -> dict[str, Any]:
    try:
        usage = shutil.disk_usage(path)
        return {
            "exists": path.exists(),
            "total_gb": round(usage.total / 1024**3, 2),
            "used_gb": round(usage.used / 1024**3, 2),
            "free_gb": round(usage.free / 1024**3, 2),
        }
    except Exception as exc:
        return {"exists": path.exists(), "error": f"{type(exc).__name__}: {str(exc)[:160]}"}


def collect_torch() -> dict[str, Any]:
    status: dict[str, Any] = {
        "available": False,
        "version": None,
        "cuda_build": None,
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
        "bf16_supported": False,
        "cuda_tensor_smoke": False,
        "error": None,
    }
    if importlib.util.find_spec("torch") is None:
        status["error"] = "torch import spec not found"
        return status
    try:
        import torch

        status["available"] = True
        status["version"] = torch.__version__
        status["cuda_build"] = torch.version.cuda
        status["cuda_available"] = bool(torch.cuda.is_available())
        status["device_count"] = int(torch.cuda.device_count()) if status["cuda_available"] else 0
        status["bf16_supported"] = bool(torch.cuda.is_bf16_supported()) if status["cuda_available"] else False
        devices = []
        for idx in range(status["device_count"]):
            props = torch.cuda.get_device_properties(idx)
            devices.append(
                {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "total_memory_gb": round(props.total_memory / 1024**3, 2),
                    "major": props.major,
                    "minor": props.minor,
                }
            )
        status["devices"] = devices
        if status["cuda_available"]:
            tensor = torch.ones((1,), device="cuda")
            status["cuda_tensor_smoke"] = bool(float((tensor + 1).item()) == 2.0)
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    return status


package_names = [
    "yaml",
    "pydantic",
    "torch",
    "transformers",
    "trl",
    "datasets",
    "accelerate",
    "vllm",
    "verl",
    "ray",
    "flash_attn",
    "deepspeed",
    "peft",
]
packages = {name: package_status(name) for name in package_names}
torch_status = collect_torch()

nvidia_smi = run_command(
    [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version",
        "--format=csv,noheader",
    ],
    timeout_s=20,
)
nvcc = run_command(["nvcc", "--version"], timeout_s=10)

model_checks = []
for path in model_paths:
    model_checks.append(
        {
            "path": str(path),
            "exists": path.exists(),
            "is_dir": path.is_dir(),
            "config_json": (path / "config.json").exists(),
            "tokenizer_config_json": (path / "tokenizer_config.json").exists(),
        }
    )

core_imports = ["yaml", "pydantic", "torch", "transformers", "trl", "datasets", "accelerate"]
grpo_imports = ["vllm", "verl", "ray"]
missing_core = [name for name in core_imports if not packages[name]["available"]]
missing_grpo = [name for name in grpo_imports if not packages[name]["available"]]
gpu_ready = bool(torch_status["cuda_available"] and torch_status["device_count"] >= 1 and torch_status["cuda_tensor_smoke"])
sft_stack_ready = not missing_core and gpu_ready
grpo_stack_ready = sft_stack_ready and not missing_grpo
known_model_found = any(item["exists"] and item["is_dir"] for item in model_checks)

blocking_reasons: list[str] = []
warnings: list[str] = []
if missing_core:
    blocking_reasons.append("missing_core_imports:" + ",".join(missing_core))
if not gpu_ready:
    blocking_reasons.append("gpu_not_ready")
if missing_grpo:
    warnings.append("missing_grpo_imports:" + ",".join(missing_grpo))
if not known_model_found:
    warnings.append("known_model_path_not_found")
if not packages["flash_attn"]["available"]:
    warnings.append("flash_attn_missing_optional")

if missing_core:
    status = "BLOCKED_IMPORTS"
elif not gpu_ready:
    status = "BLOCKED_GPU"
elif missing_grpo:
    status = "READY_SFT_ONLY"
else:
    status = "READY"

selected_env = {
    key: os.environ.get(key)
    for key in [
        "CUDA_VISIBLE_DEVICES",
        "CONDA_DEFAULT_ENV",
        "VIRTUAL_ENV",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "HF_HUB_CACHE",
        "TORCH_HOME",
        "WANDB_MODE",
        "WANDB_DISABLED",
    ]
    if os.environ.get(key) is not None
}

summary = {
    "schema_version": "phase1.platform_probe_result.v0.1",
    "run_id": run_id,
    "job_path": job_path,
    "status": status,
    "git": {
        "branch": git_value("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": git_value("rev-parse", "HEAD"),
    },
    "host": {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
    },
    "readiness": {
        "gpu_ready": gpu_ready,
        "sft_stack_ready": sft_stack_ready,
        "grpo_stack_ready": grpo_stack_ready,
        "known_model_found": known_model_found,
        "blocking_reasons": blocking_reasons,
        "warnings": warnings,
    },
    "torch": torch_status,
    "packages": packages,
    "commands": {
        "nvidia_smi": nvidia_smi,
        "nvcc": nvcc,
    },
    "model_paths": model_checks,
    "disk": {
        "repo_root": disk_status(Path.cwd()),
        "data_jinzhihong": disk_status(Path("/data/jinzhihong")),
    },
    "env": selected_env,
}

(result_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
(result_dir / "status.txt").write_text(status + "\n")
(result_dir / "environment.txt").write_text(
    "\n".join(
        [
            f"run_id={run_id}",
            f"job_path={job_path}",
            f"status={status}",
            f"hostname={summary['host']['hostname']}",
            f"python_executable={summary['host']['python_executable']}",
            f"python_version={summary['host']['python_version']}",
            f"git_branch={summary['git']['branch']}",
            f"git_commit={summary['git']['commit']}",
            f"gpu_ready={gpu_ready}",
            f"sft_stack_ready={sft_stack_ready}",
            f"grpo_stack_ready={grpo_stack_ready}",
            f"known_model_found={known_model_found}",
            f"blocking_reasons={','.join(blocking_reasons)}",
            f"warnings={','.join(warnings)}",
        ]
    )
    + "\n"
)
(result_dir / "python_packages.json").write_text(json.dumps(packages, indent=2, sort_keys=True) + "\n")
(result_dir / "gpu_probe.json").write_text(
    json.dumps({"torch": torch_status, "nvidia_smi": nvidia_smi, "nvcc": nvcc}, indent=2, sort_keys=True) + "\n"
)
(result_dir / "model_paths.json").write_text(json.dumps(model_checks, indent=2, sort_keys=True) + "\n")

print(f"server_platform_probe_status={status}")
print(f"result_dir={result_dir}")
print(f"gpu_ready={gpu_ready}")
print(f"sft_stack_ready={sft_stack_ready}")
print(f"grpo_stack_ready={grpo_stack_ready}")
if blocking_reasons:
    print("blocking_reasons=" + ",".join(blocking_reasons))
if warnings:
    print("warnings=" + ",".join(warnings))
PY

echo "result_dir=$RESULT_DIR"
echo "summary=$RESULT_DIR/summary.json"
