"""
Mechanical review checks for Phase 1 generated contract YAML files.

The checks here are intentionally conservative. Passing them means a contract is
well-formed enough for human review or downstream verifier planning; it does not
admit SFT/GRPO data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from .manifest_schemas import read_yaml_mapping
except ImportError:  # pragma: no cover - direct script execution fallback
    from manifest_schemas import read_yaml_mapping


FORBIDDEN_TEXT_FRAGMENTS = (
    "benchmark-vabench-release-v1",
    "score_denominator_manifest",
    "gold_completion",
    "target_completion",
    "reference_answer",
)
ALLOWED_LEVELS = {"L0", "L1", "L2"}
ALLOWED_TASK_FORMS = {"dut", "tb", "bugfix", "e2e", "conformance"}
ALLOWED_DOMAINS = {"voltage"}
ALLOWED_ADMISSION_STATUSES = {"draft_contract", "ready_for_generation"}
REQUIRED_PROPERTY_FIELDS = {"id", "type", "description", "observables", "expected", "tolerance_ref"}
REQUIRED_PORT_FIELDS = {"name", "direction", "discipline", "role"}
REQUIRED_PARAMETER_FIELDS = {"name", "unit", "default", "range"}


def _append_missing(mapping: dict[str, Any], required_fields: list[str], findings: list[str], prefix: str) -> None:
    missing = [field for field in required_fields if field not in mapping]
    if missing:
        findings.append(f"{prefix} missing fields: {missing}")


def _seed_ids_from_contract(contract: dict[str, Any]) -> list[str]:
    provenance = contract.get("provenance", {})
    raw_refs = provenance.get("seed_refs", [])
    seed_ids: list[str] = []
    if isinstance(raw_refs, list):
        for raw_ref in raw_refs:
            if isinstance(raw_ref, str) and raw_ref.startswith("seed_"):
                seed_ids.append(raw_ref)
    return seed_ids


def load_contract_schema(train_root: Path) -> dict[str, Any]:
    return read_yaml_mapping(train_root / "data/contracts/schema.yaml")


def validate_contract_payload(
    contract: dict[str, Any],
    schema: dict[str, Any],
    *,
    allowed_seed_ids: set[str] | None = None,
) -> dict[str, Any]:
    blocking_findings: list[str] = []
    warnings: list[str] = []
    required_fields = list(schema.get("required_fields", []))
    _append_missing(contract, required_fields, blocking_findings, "contract")

    contract_text = str(contract).lower()
    leaked_fragments = [fragment for fragment in FORBIDDEN_TEXT_FRAGMENTS if fragment in contract_text]
    if leaked_fragments:
        blocking_findings.append(f"contract contains forbidden fragments: {leaked_fragments}")

    schema_version = str(contract.get("schema_version"))
    if schema_version != str(schema.get("schema_version")):
        blocking_findings.append(f"schema_version must be {schema.get('schema_version')!r}, got {schema_version!r}")

    canonical_categories = set(schema.get("canonical_axes", {}).get("category", {}).get("canonical_values", []))
    category = contract.get("category")
    if category not in canonical_categories:
        blocking_findings.append(f"category is not canonical: {category!r}")

    level = contract.get("level")
    task_form = contract.get("task_form")
    domain = contract.get("domain")
    if level not in ALLOWED_LEVELS:
        blocking_findings.append(f"level is invalid: {level!r}")
    if task_form not in ALLOWED_TASK_FORMS:
        blocking_findings.append(f"task_form is invalid: {task_form!r}")
    if domain not in ALLOWED_DOMAINS:
        blocking_findings.append(f"domain is outside Phase 1 scope: {domain!r}")

    ports = contract.get("ports", [])
    if not isinstance(ports, list) or not ports:
        blocking_findings.append("ports must be a non-empty list")
    else:
        for index, port in enumerate(ports):
            if not isinstance(port, dict):
                blocking_findings.append(f"ports[{index}] must be a mapping")
                continue
            _append_missing(port, sorted(REQUIRED_PORT_FIELDS), blocking_findings, f"ports[{index}]")
            if port.get("discipline") != "electrical":
                blocking_findings.append(f"ports[{index}].discipline must be electrical")

    parameters = contract.get("parameters", [])
    if not isinstance(parameters, list):
        blocking_findings.append("parameters must be a list")
    else:
        for index, parameter in enumerate(parameters):
            if not isinstance(parameter, dict):
                blocking_findings.append(f"parameters[{index}] must be a mapping")
                continue
            _append_missing(parameter, sorted(REQUIRED_PARAMETER_FIELDS), warnings, f"parameters[{index}]")

    stimulus_space = contract.get("stimulus_space", {})
    if not isinstance(stimulus_space, dict):
        blocking_findings.append("stimulus_space must be a mapping")
    else:
        if not stimulus_space.get("waveforms"):
            blocking_findings.append("stimulus_space.waveforms must be non-empty")
        if "sweeps" not in stimulus_space:
            blocking_findings.append("stimulus_space.sweeps is required")

    observables = contract.get("observables", [])
    if not isinstance(observables, list) or not observables:
        blocking_findings.append("observables must be a non-empty list")

    properties = contract.get("properties", [])
    if not isinstance(properties, list) or not properties:
        blocking_findings.append("properties must be a non-empty list")
    else:
        property_types = set()
        for index, property_payload in enumerate(properties):
            if not isinstance(property_payload, dict):
                blocking_findings.append(f"properties[{index}] must be a mapping")
                continue
            _append_missing(property_payload, sorted(REQUIRED_PROPERTY_FIELDS), blocking_findings, f"properties[{index}]")
            property_types.add(str(property_payload.get("type")))
            prop_observables = property_payload.get("observables", [])
            if not isinstance(prop_observables, list) or not prop_observables:
                blocking_findings.append(f"properties[{index}].observables must be non-empty")

        if task_form == "bugfix" and "repair_delta" not in property_types:
            blocking_findings.append("bugfix contracts require a repair_delta property")
        if level == "L2" and len(properties) < 3:
            blocking_findings.append("L2 contracts require multiple decomposed/system properties")

    forbidden_constructs = contract.get("forbidden_constructs", [])
    if not isinstance(forbidden_constructs, list) or not forbidden_constructs:
        blocking_findings.append("forbidden_constructs must be a non-empty list")
    if "task_id_special_case" not in forbidden_constructs:
        warnings.append("forbidden_constructs should include task_id_special_case")

    split_key = contract.get("split_key")
    if not isinstance(split_key, str) or "/" not in split_key:
        blocking_findings.append("split_key must be a structured string with / separators")

    generation_plan = contract.get("generation_plan", {})
    if task_form == "tb" and not generation_plan.get("reference_dut_interface"):
        blocking_findings.append("tb contracts require generation_plan.reference_dut_interface")
    if level == "L2":
        decomposed = generation_plan.get("decomposed_reward_subclaims", [])
        if not isinstance(decomposed, list) or len(decomposed) < 2:
            blocking_findings.append("L2 contracts require generation_plan.decomposed_reward_subclaims")
    if task_form == "bugfix" and not contract.get("fault_model"):
        blocking_findings.append("bugfix contracts require fault_model")

    provenance = contract.get("provenance", {})
    if not isinstance(provenance, dict):
        blocking_findings.append("provenance must be a mapping")
    else:
        source_tier = provenance.get("source_tier")
        source_kind = provenance.get("source_kind")
        if source_tier not in {"B_llm_synthetic", "C_controlled_mutation"}:
            blocking_findings.append(f"generated contracts must use B/C provenance, got {source_tier!r}")
        if source_kind not in {"llm_synthetic", "controlled_mutation"}:
            blocking_findings.append(f"generated source_kind is invalid: {source_kind!r}")
        seed_ids = _seed_ids_from_contract(contract)
        if not seed_ids:
            blocking_findings.append("provenance.seed_refs must include at least one seed_* ID")
        if allowed_seed_ids is not None:
            unknown_seed_ids = sorted(set(seed_ids) - allowed_seed_ids)
            if unknown_seed_ids:
                blocking_findings.append(f"provenance.seed_refs contains non-planned seeds: {unknown_seed_ids}")
        contamination = provenance.get("contamination", {})
        if not isinstance(contamination, dict):
            blocking_findings.append("provenance.contamination must be a mapping")
        elif contamination.get("release_overlap") is True:
            blocking_findings.append("contract claims release_overlap=true")

    verifier = contract.get("verifier_evidence", {})
    if not isinstance(verifier, dict):
        blocking_findings.append("verifier_evidence must be a mapping")
    else:
        evas_status = verifier.get("evas", {}).get("status") if isinstance(verifier.get("evas"), dict) else None
        spectre_status = verifier.get("spectre_shadow", {}).get("status") if isinstance(verifier.get("spectre_shadow"), dict) else None
        if evas_status not in {"not_run", "pending"}:
            blocking_findings.append(f"new draft contract cannot claim EVAS status {evas_status!r}")
        if spectre_status not in {"not_run", "pending", "not_required"}:
            blocking_findings.append(f"new draft contract cannot claim Spectre status {spectre_status!r}")

    admission = contract.get("admission", {})
    if not isinstance(admission, dict):
        blocking_findings.append("admission must be a mapping")
    else:
        if admission.get("status") not in ALLOWED_ADMISSION_STATUSES:
            blocking_findings.append(f"generated contract admission status must be draft-like, got {admission.get('status')!r}")
        blockers = admission.get("blockers", [])
        if not isinstance(blockers, list) or not blockers:
            blocking_findings.append("draft generated contracts must carry admission.blockers")

    seed_ids = _seed_ids_from_contract(contract)
    decision = "needs_manual_review" if not blocking_findings else "reject"
    return {
        "decision": decision,
        "blocking_findings": blocking_findings,
        "warnings": warnings,
        "seed_ids": seed_ids,
        "checks": {
            "blocking_count": len(blocking_findings),
            "warning_count": len(warnings),
            "manual_review_required": True,
            "not_admitted_training_data": True,
        },
    }

