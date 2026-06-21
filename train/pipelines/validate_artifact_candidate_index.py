"""
Validate an artifact candidate index and referenced artifact hashes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .manifest_schemas import CandidateIndex, canonical_payload_hash, read_yaml_mapping, resolve_train_ref, sha256_file
except ImportError:  # pragma: no cover - direct script execution fallback
    from manifest_schemas import CandidateIndex, canonical_payload_hash, read_yaml_mapping, resolve_train_ref, sha256_file


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-index", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, default=Path("train"))
    parser.add_argument("--expect-min-candidates", type=int, default=1)
    args = parser.parse_args()

    payload = read_yaml_mapping(args.candidate_index)
    expected_hash = canonical_payload_hash(payload, "candidate_index_hash")
    require(payload.get("candidate_index_hash") == expected_hash, "candidate_index_hash mismatch")
    index = CandidateIndex.model_validate(payload)
    require(len(index.items) >= args.expect_min_candidates, f"expected at least {args.expect_min_candidates} candidates")

    for item in index.items:
        for key, ref in item.artifact_refs.items():
            path = resolve_train_ref(ref, args.train_root)
            require(path.exists(), f"{item.candidate_id}.{key} missing artifact: {ref}")
            actual_hash = sha256_file(path)
            require(
                item.artifact_hashes.get(key) == actual_hash,
                f"{item.candidate_id}.{key} hash mismatch: expected {item.artifact_hashes.get(key)}, got {actual_hash}",
            )
        main_key = item.lineage.get("sft_gold_artifact_key")
        require(main_key in item.artifact_refs, f"{item.candidate_id} missing lineage.sft_gold_artifact_key artifact")
        require(item.lineage.get("admission_status") == "not_admitted", f"{item.candidate_id} must remain not_admitted")

    print(
        "PASS validate_artifact_candidate_index "
        f"candidates={len(index.items)} candidate_index_hash={index.candidate_index_hash}"
    )


if __name__ == "__main__":
    main()
