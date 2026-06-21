"""
Shared helpers for Phase 1 local pipeline scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from .manifest_schemas import read_yaml_mapping, resolve_train_ref, sha256_file
except ImportError:  # pragma: no cover - direct script execution fallback
    from manifest_schemas import read_yaml_mapping, resolve_train_ref, sha256_file

SYSTEM_PROMPT = (
    "You are an expert Verilog-A engineer. Reason about ports, behavior, "
    "constraints, and verifier feedback before emitting final Verilog-A."
)


def load_contract(contract_ref: str, train_root: Path) -> tuple[Path, dict[str, Any]]:
    path = resolve_train_ref(contract_ref, train_root)
    if not path.exists():
        raise FileNotFoundError(f"contract_ref does not exist: {contract_ref}")
    return path, read_yaml_mapping(path)


def verify_file_hash(path: Path, expected_hash: str, label: str) -> str:
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError(f"{label} hash mismatch: expected {expected_hash}, got {actual_hash}")
    return actual_hash


def format_ports(contract: dict[str, Any]) -> str:
    ports = contract.get("ports", [])
    rendered: list[str] = []
    for port in ports:
        rendered.append(
            f"{port.get('name')}({port.get('direction')}, "
            f"{port.get('discipline')}, role={port.get('role')})"
        )
    return ", ".join(rendered)


def format_parameters(contract: dict[str, Any]) -> str:
    parameters = contract.get("parameters", [])
    rendered: list[str] = []
    for param in parameters:
        rendered.append(
            f"{param.get('name')}={param.get('default')} {param.get('unit')} "
            f"range={param.get('range')}"
        )
    return "; ".join(rendered)


def format_properties(contract: dict[str, Any]) -> str:
    properties = contract.get("properties", [])
    rendered: list[str] = []
    for prop in properties:
        rendered.append(f"{prop.get('id')}[{prop.get('type')}]: {prop.get('description')}")
    return "\n".join(f"- {line}" for line in rendered)


def public_contract_summary(contract: dict[str, Any]) -> str:
    lines = [
        f"Contract ID: {contract.get('id')}",
        f"Category: {contract.get('category')}",
        f"Level/task_form: {contract.get('level')} / {contract.get('task_form')}",
        f"Base function: {contract.get('base_function')}",
        f"Intent: {contract.get('intent')}",
        f"Ports: {format_ports(contract)}",
        f"Parameters: {format_parameters(contract)}",
        "Properties:",
        format_properties(contract),
        "Forbidden constructs:",
        ", ".join(contract.get("forbidden_constructs", [])),
    ]
    return "\n".join(line for line in lines if line)


def instruction_from_contract(contract: dict[str, Any]) -> str:
    return (
        f"Generate a Verilog-A {contract.get('task_form')} artifact for "
        f"{contract.get('base_function')} under the given behavioral contract."
    )


def grpo_prompt_from_contract(contract: dict[str, Any]) -> str:
    return f"{instruction_from_contract(contract)}\n\n{public_contract_summary(contract)}"


def sft_output_from_artifact(contract: dict[str, Any], artifact_text: str) -> str:
    reasoning = (
        "<think>\n"
        f"<port>{format_ports(contract)}</port>\n"
        f"<behavior>{contract.get('intent')}</behavior>\n"
        "</think>"
    )
    answer = f"<answer>\n{artifact_text.rstrip()}\n</answer>"
    return f"{reasoning}\n{answer}"


def sft_output_from_dut(contract: dict[str, Any], dut_text: str) -> str:
    return sft_output_from_artifact(contract, dut_text)
