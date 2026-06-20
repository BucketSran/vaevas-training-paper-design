#!/usr/bin/env bash
set -euo pipefail

JOB_PATH="train/infra/jobs/phase1_gpu_blocker_diagnosis.yaml"
RUN_ID="gpu-blocker-diagnosis-$(date -u +%Y%m%dT%H%M%SZ)"
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
mkdir -p "$RESULT_DIR"

COMMAND_LOG="$RESULT_DIR/commands.log"
: > "$COMMAND_LOG"

{
  echo "run_id=$RUN_ID"
  echo "job_path=$JOB_PATH"
  echo "python_bin=$PYTHON_BIN"
  echo "repo_root=$REPO_ROOT"
} | tee -a "$COMMAND_LOG" >/dev/null

echo "\$ $PYTHON_BIN - <gpu_blocker_diagnosis.py>" | tee -a "$COMMAND_LOG"

"$PYTHON_BIN" - "$RESULT_DIR" "$RUN_ID" "$JOB_PATH" <<'PY' 2>&1 | tee -a "$COMMAND_LOG"
from __future__ import annotations

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


def run_shell(command: str, timeout_s: int = 20) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            shell=True,
            executable="/bin/bash",
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
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return {
            "ok": False,
            "returncode": None,
            "stdout": stdout.strip(),
            "stderr": (stderr.strip() + f"\ntimeout after {timeout_s}s").strip(),
        }


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def collect_torch() -> dict[str, Any]:
    status: dict[str, Any] = {
        "available": False,
        "version": None,
        "cuda_build": None,
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
        "cuda_tensor_smoke": False,
        "bf16_supported": False,
        "error": None,
    }
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
        status["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
    return status


def command_text(command_result: dict[str, Any]) -> str:
    return "\n".join([command_result.get("stdout") or "", command_result.get("stderr") or ""]).strip()


commands = {
    "hostname": run_shell("hostname"),
    "uname": run_shell("uname -a"),
    "id": run_shell("id"),
    "groups": run_shell("groups"),
    "which_nvidia_smi": run_shell("command -v nvidia-smi"),
    "nvidia_smi": run_shell("nvidia-smi", timeout_s=30),
    "nvidia_smi_L": run_shell("nvidia-smi -L", timeout_s=30),
    "nvidia_smi_query": run_shell(
        "nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version --format=csv,noheader",
        timeout_s=30,
    ),
    "dev_nvidia": run_shell("ls -l /dev/nvidia* 2>/dev/null || true"),
    "proc_driver_nvidia": run_shell("cat /proc/driver/nvidia/version 2>/dev/null || true"),
    "lsmod_nvidia": run_shell("lsmod 2>/dev/null | grep -E '^(nvidia|nouveau)' || true"),
    "lspci_nvidia": run_shell("lspci 2>/dev/null | grep -i nvidia || true"),
    "systemctl_nvidia_persistenced": run_shell("systemctl is-active nvidia-persistenced 2>/dev/null || true"),
    "which_nvcc": run_shell("command -v nvcc"),
    "nvcc_version": run_shell("nvcc --version 2>/dev/null || true"),
    "slurm_env": run_shell("env | grep -E '^(SLURM|CUDA|NVIDIA|CONDA|VIRTUAL_ENV|LD_LIBRARY_PATH)=' | sort || true"),
}
torch_status = collect_torch()

selected_env = {
    key: os.environ.get(key)
    for key in [
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "CONDA_DEFAULT_ENV",
        "VIRTUAL_ENV",
        "LD_LIBRARY_PATH",
        "SLURM_JOB_ID",
        "SLURM_JOB_GPUS",
        "SLURM_STEP_GPUS",
        "SLURM_GPUS",
        "SLURM_JOB_NODELIST",
        "PBS_JOBID",
    ]
    if os.environ.get(key) is not None
}

dev_nvidia_stdout = commands["dev_nvidia"]["stdout"]
has_nvidia_device_files = "/dev/nvidia" in dev_nvidia_stdout
nvidia_smi_text = command_text(commands["nvidia_smi"])
nvidia_smi_ok = bool(commands["nvidia_smi"]["ok"])
torch_gpu_ok = bool(torch_status["cuda_available"] and torch_status["cuda_tensor_smoke"])

safe_next_actions: list[str] = []
likely_cause = "unknown"

if nvidia_smi_ok and torch_gpu_ok:
    status = "GPU_READY"
    likely_cause = "no_gpu_blocker_detected"
    safe_next_actions.append("rerun train/infra/run_server_platform_probe.sh to confirm READY status")
elif not commands["which_nvidia_smi"]["ok"]:
    status = "BLOCKED_NO_NVIDIA_SMI"
    likely_cause = "nvidia_smi_not_on_path_or_gpu_runtime_absent"
    safe_next_actions.append("run from the intended GPU image/environment or ask admin for the correct GPU node")
    safe_next_actions.append("do not install drivers from this repository task")
elif "couldn't communicate with the NVIDIA driver" in nvidia_smi_text:
    status = "BLOCKED_NVIDIA_DRIVER"
    likely_cause = "host_driver_unavailable_or_container_not_attached_to_driver"
    safe_next_actions.append("if this is a scheduler cluster, move to an allocated GPU node/session")
    safe_next_actions.append("if this is a container, restart it with GPU passthrough such as --gpus all or the cluster equivalent")
    safe_next_actions.append("if /dev/nvidia* exists but nvidia-smi still fails, ask platform admin to repair/restart the NVIDIA driver")
elif not has_nvidia_device_files:
    status = "BLOCKED_NO_DEVICE_FILES"
    likely_cause = "current_session_has_no_gpu_device_mount_or_no_gpu_allocation"
    safe_next_actions.append("switch to a GPU node or request an interactive GPU allocation")
    safe_next_actions.append("if using a container, ensure /dev/nvidia* devices are mounted into the container")
elif nvidia_smi_ok and not torch_gpu_ok:
    status = "BLOCKED_TORCH_CUDA"
    likely_cause = "driver_visible_but_python_torch_cuda_unusable"
    safe_next_actions.append("verify CUDA_VISIBLE_DEVICES and the Python environment")
    safe_next_actions.append("rerun with the intended Python path using --python /data/jinzhihong/envs/vaevas-rl/bin/python")
else:
    status = "BLOCKED_UNKNOWN_GPU"
    likely_cause = "gpu_failure_requires_admin_or_scheduler_context"
    safe_next_actions.append("inspect gpu_blocker_commands.json and ask platform admin if driver/session state is unclear")

summary = {
    "schema_version": "phase1.gpu_blocker_diagnosis_result.v0.1",
    "run_id": run_id,
    "job_path": job_path,
    "status": status,
    "likely_cause": likely_cause,
    "safe_next_actions": safe_next_actions,
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
    "env": selected_env,
    "signals": {
        "nvidia_smi_ok": nvidia_smi_ok,
        "has_nvidia_device_files": has_nvidia_device_files,
        "torch_gpu_ok": torch_gpu_ok,
        "torch_cuda_available": torch_status["cuda_available"],
        "torch_device_count": torch_status["device_count"],
    },
    "torch": torch_status,
}

diagnosis_text = [
    f"status={status}",
    f"likely_cause={likely_cause}",
    "safe_next_actions:",
]
diagnosis_text.extend(f"- {item}" for item in safe_next_actions)

(result_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
(result_dir / "gpu_blocker_commands.json").write_text(json.dumps(commands, indent=2, sort_keys=True) + "\n")
(result_dir / "environment.txt").write_text(
    "\n".join(
        [
            f"run_id={run_id}",
            f"job_path={job_path}",
            f"status={status}",
            f"likely_cause={likely_cause}",
            f"hostname={summary['host']['hostname']}",
            f"python_executable={summary['host']['python_executable']}",
            f"python_version={summary['host']['python_version']}",
            f"git_branch={summary['git']['branch']}",
            f"git_commit={summary['git']['commit']}",
            f"nvidia_smi_ok={nvidia_smi_ok}",
            f"has_nvidia_device_files={has_nvidia_device_files}",
            f"torch_gpu_ok={torch_gpu_ok}",
            f"torch_cuda_available={torch_status['cuda_available']}",
            f"torch_device_count={torch_status['device_count']}",
        ]
    )
    + "\n"
)
(result_dir / "diagnosis.txt").write_text("\n".join(diagnosis_text) + "\n")
(result_dir / "status.txt").write_text(status + "\n")

print(f"gpu_blocker_diagnosis_status={status}")
print(f"likely_cause={likely_cause}")
print(f"result_dir={result_dir}")
for item in safe_next_actions:
    print(f"safe_next_action={item}")
PY

echo "result_dir=$RESULT_DIR"
echo "summary=$RESULT_DIR/summary.json"
