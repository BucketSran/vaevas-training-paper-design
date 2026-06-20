"""
Validate the Phase 1 clean-room seed catalog and pilot batch plan.

This script checks planning metadata only. It does not generate contracts,
candidate artifacts, EVAS reports, Spectre reports, or training data.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

try:
    from .manifest_schemas import (
        SCHEMA_VERSION,
        PilotBatchPlan,
        SeedCatalog,
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
        canonical_payload_hash,
        read_yaml_mapping,
        require_sha256,
        resolve_train_ref,
        write_yaml_mapping,
    )


DEFAULT_TRAIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED_CATALOG = DEFAULT_TRAIN_ROOT / "data/seeds/seed_catalog.phase1-pilot-0001.yaml"
DEFAULT_PILOT_PLAN = DEFAULT_TRAIN_ROOT / "data/manifests/pilot/batch_plan.synth-batch-pilot-0001.yaml"

FORBIDDEN_SOURCE_REF_FRAGMENTS = (
    "behavioral-veriloga-eval",
    "benchmark-vabench-release-v1",
    "score_denominator_manifest",
    "/tasks/",
    "vabench",
)
REQUIRED_FORBIDDEN_USES = {"benchmark_eval", "copy_release_prompt", "copy_release_gold"}
REQUIRED_LEVELS = {"L0", "L1", "L2"}
REQUIRED_TASK_FORMS = {"conformance", "dut", "tb", "bugfix", "e2e"}


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


def load_catalog(path: Path, rewrite_hashes: bool) -> tuple[SeedCatalog, dict[str, Any]]:
    payload = read_yaml_mapping(path)
    if rewrite_hashes:
        payload = rewrite_hash(path, payload, "seed_catalog_hash")
    validate_hash(payload, "seed_catalog_hash")
    catalog = SeedCatalog.model_validate(payload)
    require(catalog.schema_version == SCHEMA_VERSION, f"unexpected seed catalog schema_version={catalog.schema_version!r}")
    return catalog, payload


def load_pilot_plan(
    path: Path,
    seed_catalog_hash: str,
    rewrite_hashes: bool,
) -> tuple[PilotBatchPlan, dict[str, Any]]:
    payload = read_yaml_mapping(path)
    if rewrite_hashes:
        payload = dict(payload)
        payload["seed_catalog_hash"] = seed_catalog_hash
        payload["pilot_plan_hash"] = canonical_payload_hash(payload, "pilot_plan_hash")
        write_yaml_mapping(path, payload)
        payload = read_yaml_mapping(path)
    validate_hash(payload, "pilot_plan_hash")
    plan = PilotBatchPlan.model_validate(payload)
    require(plan.schema_version == SCHEMA_VERSION, f"unexpected pilot plan schema_version={plan.schema_version!r}")
    require(plan.seed_catalog_hash == seed_catalog_hash, "pilot plan seed_catalog_hash does not match seed catalog")
    return plan, payload


def validate_seed_catalog(catalog: SeedCatalog, train_root: Path) -> None:
    seed_ids = [item.seed_id for item in catalog.items]
    require(len(seed_ids) == len(set(seed_ids)), "seed_id values must be unique")
    require(catalog.items, "seed catalog must contain at least one item")

    for item in catalog.items:
        require(item.status != "rejected", f"rejected seed is still present: {item.seed_id}")
        require(item.allowed_uses, f"{item.seed_id} allowed_uses must be non-empty")
        require(item.forbidden_uses, f"{item.seed_id} forbidden_uses must be non-empty")
        missing_forbidden = REQUIRED_FORBIDDEN_USES - set(item.forbidden_uses)
        require(not missing_forbidden, f"{item.seed_id} forbidden_uses missing {sorted(missing_forbidden)}")
        require(item.source_tier.startswith("A_"), f"{item.seed_id} must use a reviewed Tier A-style source tier")
        require(item.source_ref, f"{item.seed_id} source_ref must be explicit")

        lowered_ref = item.source_ref.lower()
        blocked = [fragment for fragment in FORBIDDEN_SOURCE_REF_FRAGMENTS if fragment in lowered_ref]
        require(not blocked, f"{item.seed_id} source_ref contains forbidden fragments: {blocked}")

        source_path = resolve_train_ref(item.source_ref, train_root)
        require(source_path.exists(), f"{item.seed_id} source_ref does not exist: {item.source_ref}")
        require(source_path.is_file(), f"{item.seed_id} source_ref must point to a file: {item.source_ref}")

        review = item.contamination_review
        require(review.get("protected_release_source") is False, f"{item.seed_id} must not use protected release source")
        if item.status == "seed_candidate":
            require(
                review.get("review_required_before_generation") is True,
                f"{item.seed_id} seed_candidate must require review before generation",
            )


def validate_pilot_plan(plan: PilotBatchPlan, catalog: SeedCatalog, train_root: Path) -> None:
    catalog_path = resolve_train_ref(plan.seed_catalog_ref, train_root)
    require(catalog_path.exists(), f"seed_catalog_ref does not exist: {plan.seed_catalog_ref}")
    require(plan.planned_seed_ids, "planned_seed_ids must be non-empty")
    require(len(plan.planned_seed_ids) == len(set(plan.planned_seed_ids)), "planned_seed_ids must be unique")

    catalog_by_id = {item.seed_id: item for item in catalog.items}
    missing = sorted(set(plan.planned_seed_ids) - set(catalog_by_id))
    require(not missing, f"planned_seed_ids missing from seed catalog: {missing}")

    covered_levels: set[str] = set()
    covered_task_forms: set[str] = set()
    covered_categories: set[str] = set()
    for seed_id in plan.planned_seed_ids:
        item = catalog_by_id[seed_id]
        require(item.status != "rejected", f"pilot plan references rejected seed: {seed_id}")
        covered_levels.update(item.intended_levels)
        covered_task_forms.update(item.intended_task_forms)
        covered_categories.add(item.category)

    require(REQUIRED_LEVELS <= covered_levels, f"pilot planned seeds must cover levels {sorted(REQUIRED_LEVELS)}")
    require(
        REQUIRED_TASK_FORMS <= covered_task_forms,
        f"pilot planned seeds must cover task forms {sorted(REQUIRED_TASK_FORMS)}",
    )

    seed_contracts = plan.target_counts.get("seed_contracts", 0)
    generated_candidates = plan.target_counts.get("generated_candidates", 0)
    admitted_min = plan.target_counts.get("admitted_min", 0)
    spectre_min = plan.target_counts.get("spectre_shadow_queue_min", 0)
    require(seed_contracts > 0, "target_counts.seed_contracts must be positive")
    require(generated_candidates > 0, "target_counts.generated_candidates must be positive")
    require(admitted_min > 0, "target_counts.admitted_min must be positive")
    require(sum(plan.categories.values()) == seed_contracts, "category counts must sum to target_counts.seed_contracts")

    min_candidates = int(plan.generation_budget.get("min_candidates_per_contract", 0))
    max_total = int(plan.generation_budget.get("max_total_candidates", 0))
    require(min_candidates >= 1, "generation_budget.min_candidates_per_contract must be positive")
    require(max_total == generated_candidates, "generation_budget.max_total_candidates must match target generated_candidates")
    require(generated_candidates >= seed_contracts * min_candidates, "generated_candidates is too small for min_candidates_per_contract")

    required_shadow = max(1, math.ceil(admitted_min * 0.2))
    require(spectre_min >= required_shadow, "spectre shadow queue must cover at least 20% of admitted_min")
    require(
        float(plan.verification_queue.get("spectre_shadow_min_l2_fraction", 0.0)) >= 0.5,
        "verification_queue.spectre_shadow_min_l2_fraction must be at least 0.5",
    )

    admission = plan.admission_policy
    require(admission.get("require_contract_review") is True, "admission_policy.require_contract_review must be true")
    require(admission.get("require_contamination_clean") is True, "admission_policy.require_contamination_clean must be true")
    require(admission.get("require_evas") is True, "admission_policy.require_evas must be true")
    require(admission.get("reject_evas_pass_spectre_fail") is True, "EVAS PASS / Spectre FAIL must be rejected")
    require(admission.get("grpo_prompt_leakage_block") is True, "GRPO prompt leakage block must be enabled")
    require("L2_e2e" in set(admission.get("require_spectre_shadow_for", [])), "L2_e2e must be shadow-audited")

    heldout_categories = set(plan.ood_holdout.get("heldout_categories", []))
    require(heldout_categories <= set(plan.categories), "heldout_categories must be a subset of planned categories")
    require(heldout_categories & covered_categories, "heldout_categories must be represented by planned seeds")

    require(plan.split_policy.get("forbid_spec_overlap_between_train_and_eval") is True, "split leakage block must be enabled")
    require(plan.split_policy.get("holdout_before_sft_packing") is True, "holdout must happen before SFT packing")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-catalog", type=Path, default=DEFAULT_SEED_CATALOG)
    parser.add_argument("--pilot-plan", type=Path, default=DEFAULT_PILOT_PLAN)
    parser.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--rewrite-hashes", action="store_true", help="rewrite stable hash fields before validation")
    args = parser.parse_args()

    catalog, _catalog_payload = load_catalog(args.seed_catalog, args.rewrite_hashes)
    validate_seed_catalog(catalog, args.train_root)

    plan, _plan_payload = load_pilot_plan(args.pilot_plan, catalog.seed_catalog_hash, args.rewrite_hashes)
    validate_pilot_plan(plan, catalog, args.train_root)

    print(
        "PASS validate_pilot_plan "
        f"catalog={catalog.catalog_id} seeds={len(catalog.items)} "
        f"batch={plan.batch_id} target_candidates={plan.target_counts['generated_candidates']}"
    )


if __name__ == "__main__":
    main()
