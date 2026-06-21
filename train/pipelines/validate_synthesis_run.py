"""
Validate a Phase 1 synthesis run plan before any LLM generation.

This validator checks metadata and references only. It does not call an LLM and
does not generate contracts, Verilog-A artifacts, EVAS reports, Spectre reports,
or training data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from .manifest_schemas import (
        SCHEMA_VERSION,
        PilotBatchPlan,
        SeedCatalog,
        SynthesisRunPlan,
        canonical_payload_hash,
        read_yaml_mapping,
        require_sha256,
        resolve_train_ref,
        write_yaml_mapping,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from manifest_schemas import (
        SCHEMA_VERSION,
        PilotBatchPlan,
        SeedCatalog,
        SynthesisRunPlan,
        canonical_payload_hash,
        read_yaml_mapping,
        require_sha256,
        resolve_train_ref,
        write_yaml_mapping,
    )


DEFAULT_TRAIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_PLAN = DEFAULT_TRAIN_ROOT / "data/manifests/synthesis/synthesis_run.contract-smoke-0001.yaml"
ALLOWED_PROMPT_KINDS = {"contract_proposal", "contract_review", "artifact_proposal"}
ALWAYS_FORBIDDEN_OUTPUT_KEYWORDS = {"model checkpoints", "simulator dumps", "secrets or .env files"}
TRAINING_PACK_FORBIDDEN_OUTPUT_KEYWORDS = {"SFT JSONL", "GRPO JSONL"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def rewrite_hash(path: Path, payload: dict[str, Any], hash_field: str) -> dict[str, Any]:
    payload = dict(payload)
    payload[hash_field] = canonical_payload_hash(payload, hash_field)
    write_yaml_mapping(path, payload)
    return read_yaml_mapping(path)


def validate_hash(payload: dict[str, Any], hash_field: str) -> None:
    value = payload.get(hash_field)
    require(isinstance(value, str), f"{hash_field} must be present")
    require_sha256(hash_field, value)
    expected = canonical_payload_hash(payload, hash_field)
    require(value == expected, f"{hash_field} mismatch: expected {expected}, got {value}")


def load_run_plan(path: Path, rewrite_hashes: bool = False) -> SynthesisRunPlan:
    payload = read_yaml_mapping(path)
    if rewrite_hashes:
        payload = rewrite_hash(path, payload, "synthesis_run_hash")
    validate_hash(payload, "synthesis_run_hash")
    plan = SynthesisRunPlan.model_validate(payload)
    require(plan.schema_version == SCHEMA_VERSION, f"unexpected schema_version={plan.schema_version!r}")
    return plan


def validate_run_plan(plan: SynthesisRunPlan, train_root: Path) -> None:
    seed_catalog_path = resolve_train_ref(plan.seed_catalog_ref, train_root)
    pilot_plan_path = resolve_train_ref(plan.pilot_plan_ref, train_root)
    template_dir = resolve_train_ref(plan.prompt_template_dir, train_root)

    require(seed_catalog_path.exists(), f"seed_catalog_ref does not exist: {plan.seed_catalog_ref}")
    require(pilot_plan_path.exists(), f"pilot_plan_ref does not exist: {plan.pilot_plan_ref}")
    require(template_dir.exists() and template_dir.is_dir(), f"prompt_template_dir does not exist: {plan.prompt_template_dir}")

    seed_payload = read_yaml_mapping(seed_catalog_path)
    pilot_payload = read_yaml_mapping(pilot_plan_path)
    validate_hash(seed_payload, "seed_catalog_hash")
    validate_hash(pilot_payload, "pilot_plan_hash")

    seed_catalog = SeedCatalog.model_validate(seed_payload)
    pilot_plan = PilotBatchPlan.model_validate(pilot_payload)
    require(plan.seed_catalog_hash == seed_catalog.seed_catalog_hash, "seed_catalog_hash mismatch")
    require(plan.pilot_plan_hash == pilot_plan.pilot_plan_hash, "pilot_plan_hash mismatch")
    require(pilot_plan.seed_catalog_hash == seed_catalog.seed_catalog_hash, "pilot plan does not reference seed catalog")

    include_kinds = set(plan.prompt_selection.get("include_prompt_kinds", []))
    exclude_kinds = set(plan.prompt_selection.get("exclude_prompt_kinds", []))
    require(include_kinds, "prompt_selection.include_prompt_kinds must be non-empty")
    require(include_kinds <= ALLOWED_PROMPT_KINDS, f"unknown included prompt kinds: {sorted(include_kinds - ALLOWED_PROMPT_KINDS)}")
    require(exclude_kinds <= ALLOWED_PROMPT_KINDS, f"unknown excluded prompt kinds: {sorted(exclude_kinds - ALLOWED_PROMPT_KINDS)}")
    require(not include_kinds & exclude_kinds, "prompt_selection cannot include and exclude the same kind")
    require(
        include_kinds == {"contract_proposal"},
        "contract synthesis smoke must include only contract_proposal prompts",
    )

    max_requests = int(plan.prompt_selection.get("max_requests", 0))
    variants_per_seed = int(plan.prompt_selection.get("variants_per_seed", 1))
    require(max_requests > 0, "prompt_selection.max_requests must be positive")
    require(variants_per_seed > 0, "prompt_selection.variants_per_seed must be positive when present")
    if plan.prompt_selection.get("require_one_request_per_planned_seed"):
        require(max_requests == len(pilot_plan.planned_seed_ids), "max_requests must match planned_seed_ids count")
    else:
        require(
            max_requests == len(pilot_plan.planned_seed_ids) * variants_per_seed,
            "max_requests must match planned_seed_ids * variants_per_seed",
        )

    max_contracts = int(plan.model_policy.get("max_contracts", 0))
    require(max_contracts == max_requests, "model_policy.max_contracts must match prompt max_requests")
    external_api_allowed = bool(plan.model_policy.get("external_api_allowed", True))
    if not external_api_allowed:
        require(
            plan.validation_policy.get("require_no_llm_api_when_external_api_allowed_false") is True,
            "validation_policy must enforce no external LLM API when external_api_allowed is false",
        )

    require(plan.output_policy.get("commit_result_dir") is True, "smoke result dir should be committed for handoff")
    forbidden_outputs = set(plan.output_policy.get("forbidden_outputs", []))
    required_forbidden = set(ALWAYS_FORBIDDEN_OUTPUT_KEYWORDS)
    if not plan.output_policy.get("allow_draft_training_packs", False):
        required_forbidden |= TRAINING_PACK_FORBIDDEN_OUTPUT_KEYWORDS
    missing_forbidden = required_forbidden - forbidden_outputs
    require(not missing_forbidden, f"output_policy.forbidden_outputs missing {sorted(missing_forbidden)}")
    require(
        plan.review_policy.get("manual_review_required_before_artifact_generation") is True,
        "manual review must be required before artifact generation",
    )
    require(plan.remote_handoff.get("one_shot_required") is True, "remote handoff must be one-shot")
    require(plan.remote_handoff.get("return_summary_in_final_response") is True, "remote must return summary directly")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-plan", type=Path, default=DEFAULT_RUN_PLAN)
    parser.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--rewrite-hashes", action="store_true")
    args = parser.parse_args()

    plan = load_run_plan(args.run_plan, rewrite_hashes=args.rewrite_hashes)
    validate_run_plan(plan, args.train_root)
    print(f"PASS validate_synthesis_run run_id={plan.run_id} hash={plan.synthesis_run_hash}")


if __name__ == "__main__":
    main()
