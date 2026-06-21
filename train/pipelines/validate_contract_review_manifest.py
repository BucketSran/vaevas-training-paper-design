"""
Validate a structured contract review manifest.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

try:
    from .manifest_schemas import (
        ContractReviewManifest,
        GeneratedContractIndex,
        canonical_payload_hash,
        read_yaml_mapping,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from manifest_schemas import (
        ContractReviewManifest,
        GeneratedContractIndex,
        canonical_payload_hash,
        read_yaml_mapping,
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--generated-contract-index", type=Path, required=True)
    parser.add_argument("--expect-count", type=int, default=None)
    args = parser.parse_args()

    review_payload = read_yaml_mapping(args.review_manifest)
    expected_review_hash = canonical_payload_hash(review_payload, "contract_review_manifest_hash")
    require(
        review_payload.get("contract_review_manifest_hash") == expected_review_hash,
        "contract_review_manifest_hash mismatch",
    )
    review_manifest = ContractReviewManifest.model_validate(review_payload)

    index_payload = read_yaml_mapping(args.generated_contract_index)
    expected_index_hash = canonical_payload_hash(index_payload, "generated_contract_index_hash")
    require(index_payload.get("generated_contract_index_hash") == expected_index_hash, "generated_contract_index_hash mismatch")
    index = GeneratedContractIndex.model_validate(index_payload)
    require(review_manifest.generated_contract_index_hash == index.generated_contract_index_hash, "generated_contract_index_hash ref mismatch")

    index_ids = {item.generated_contract_id for item in index.items}
    review_ids = {item.generated_contract_id for item in review_manifest.items}
    require(review_ids == index_ids, f"review IDs mismatch: missing={sorted(index_ids - review_ids)} extra={sorted(review_ids - index_ids)}")
    if args.expect_count is not None:
        require(len(review_manifest.items) == args.expect_count, f"expected {args.expect_count} review items")

    by_decision = Counter(item.decision for item in review_manifest.items)
    require(dict(sorted(by_decision.items())) == review_manifest.summary.get("by_decision"), "summary.by_decision mismatch")
    for item in review_manifest.items:
        if item.decision == "accept_for_generation":
            require(item.allowed_next_stage == "artifact_generation", f"{item.generated_contract_id} accepted item has wrong next stage")
            require(not item.blocking_findings, f"{item.generated_contract_id} accepted item still has blocking findings")
        if item.decision in {"reject", "quarantine"}:
            require(item.allowed_next_stage == "none", f"{item.generated_contract_id} rejected/quarantined item has next stage")

    print(
        "PASS validate_contract_review_manifest "
        f"items={len(review_manifest.items)} "
        f"accepted={by_decision.get('accept_for_generation', 0)} "
        f"needs_revision={by_decision.get('needs_revision', 0)}"
    )


if __name__ == "__main__":
    main()
