"""
Validate a generated contract index and its referenced draft contracts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .contract_review_gate import load_contract_schema, validate_contract_payload
    from .manifest_schemas import (
        GeneratedContractIndex,
        canonical_payload_hash,
        read_yaml_mapping,
        resolve_train_ref,
        sha256_file,
    )
    from .validate_synthesis_run import DEFAULT_RUN_PLAN, DEFAULT_TRAIN_ROOT, load_run_plan, validate_run_plan
except ImportError:  # pragma: no cover - direct script execution fallback
    from contract_review_gate import load_contract_schema, validate_contract_payload
    from manifest_schemas import (
        GeneratedContractIndex,
        canonical_payload_hash,
        read_yaml_mapping,
        resolve_train_ref,
        sha256_file,
    )
    from validate_synthesis_run import DEFAULT_RUN_PLAN, DEFAULT_TRAIN_ROOT, load_run_plan, validate_run_plan


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--synthesis-run", type=Path, default=DEFAULT_RUN_PLAN)
    parser.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--expect-count", type=int, default=None)
    args = parser.parse_args()

    run_plan = load_run_plan(args.synthesis_run)
    validate_run_plan(run_plan, args.train_root)
    payload = read_yaml_mapping(args.index)
    expected_hash = canonical_payload_hash(payload, "generated_contract_index_hash")
    require(payload.get("generated_contract_index_hash") == expected_hash, "generated_contract_index_hash mismatch")
    index = GeneratedContractIndex.model_validate(payload)
    require(index.synthesis_run_hash == run_plan.synthesis_run_hash, "synthesis_run_hash mismatch")
    if args.expect_count is not None:
        require(len(index.items) == args.expect_count, f"expected {args.expect_count} contracts, got {len(index.items)}")

    schema = load_contract_schema(args.train_root)
    pilot_payload = read_yaml_mapping(args.train_root / run_plan.pilot_plan_ref)
    allowed_seed_ids = set(pilot_payload["planned_seed_ids"])
    seen_contract_ids: set[str] = set()
    for item in index.items:
        require(item.generated_contract_id not in seen_contract_ids, f"duplicate generated_contract_id: {item.generated_contract_id}")
        seen_contract_ids.add(item.generated_contract_id)
        contract_path = resolve_train_ref(item.contract_ref, args.train_root)
        require(contract_path.exists(), f"contract_ref does not exist: {item.contract_ref}")
        require(sha256_file(contract_path) == item.contract_sha256, f"contract hash mismatch: {item.contract_ref}")
        contract = read_yaml_mapping(contract_path)
        review = validate_contract_payload(contract, schema, allowed_seed_ids=allowed_seed_ids)
        require(review["decision"] != "reject", f"{item.generated_contract_id} failed review: {review['blocking_findings']}")
        require(item.mechanical_review["decision"] == review["decision"], f"mechanical_review decision drift: {item.generated_contract_id}")
        require(item.intended_next_stage == "manual_contract_review", "generated contracts must route to manual review")

    require(index.summary.get("training_data_admitted") is False, "generated index must not admit training data")
    require(index.summary.get("verilog_a_generated") is False, "generated index must not claim Verilog-A artifacts")
    print(f"PASS validate_generated_contracts index={args.index} contracts={len(index.items)}")


if __name__ == "__main__":
    main()
