"""
Check candidate-index entries against a protected-index manifest.

The first implementation is intentionally conservative and deterministic. It
handles exact hash overlap, protected lineage/path references, missing
provenance, and split-key collisions. Higher-similarity MinHash or parser-based
matching can be added behind the same report schema later.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

try:
    from .manifest_schemas import (
        EMPTY_SHA256,
        SCHEMA_VERSION,
        CandidateIndex,
        ContaminationReport,
        Producer,
        ProtectedIndex,
        canonical_payload_hash,
        is_sha256,
        load_manifest,
        write_yaml_mapping,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from manifest_schemas import (
        EMPTY_SHA256,
        SCHEMA_VERSION,
        CandidateIndex,
        ContaminationReport,
        Producer,
        ProtectedIndex,
        canonical_payload_hash,
        is_sha256,
        load_manifest,
        write_yaml_mapping,
    )


def flatten_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from flatten_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from flatten_strings(item)


def protected_hash_map(protected_index: ProtectedIndex, attr: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for item in protected_index.items:
        value = getattr(item, attr)
        if value:
            out.setdefault(value, []).append(item.protected_id)
    return out


def decide_candidate(item: Any, protected_index: ProtectedIndex) -> dict[str, Any]:
    raw_hashes = protected_hash_map(protected_index, "raw_sha256")
    normalized_hashes = protected_hash_map(protected_index, "normalized_sha256")
    protected_paths = {protected.source_path for protected in protected_index.items}
    protected_task_ids = {protected.task_id for protected in protected_index.items if protected.task_id}

    candidate_hashes = set(item.artifact_hashes.values())
    candidate_hashes.add(item.contract_sha256)
    raw_matches = sorted({pid for h in candidate_hashes for pid in raw_hashes.get(h, [])})
    normalized_matches = sorted({pid for h in candidate_hashes for pid in normalized_hashes.get(h, [])})

    lineage_strings = set(flatten_strings(item.lineage))
    lineage_strings |= {seed.source_ref for seed in item.seed_refs}
    forbidden_lineage = any("behavioral-veriloga-eval" in value for value in lineage_strings)
    forbidden_lineage |= any(path and path in value for path in protected_paths for value in lineage_strings)

    missing_provenance = not item.seed_refs or not is_sha256(item.contract_sha256)
    split_key_collision = item.split_key in protected_task_ids
    matched_ids = sorted(set(raw_matches) | set(normalized_matches))

    if matched_ids:
        decision = "rejected"
        strongest_match_tier = "exact_hash"
        notes = ["Exact protected raw or normalized hash matched."]
    elif forbidden_lineage:
        decision = "rejected"
        strongest_match_tier = "forbidden_lineage"
        notes = ["Lineage or seed reference points at protected benchmark material."]
    elif missing_provenance:
        decision = "quarantined"
        strongest_match_tier = "missing_provenance"
        notes = ["Candidate provenance or hash metadata is incomplete."]
    elif split_key_collision:
        decision = "needs_review"
        strongest_match_tier = "split_key_collision"
        notes = ["Candidate split_key collides with a protected task ID."]
    else:
        decision = "clean"
        strongest_match_tier = "none"
        notes = ["No exact protected overlap or forbidden lineage detected."]

    return {
        "candidate_id": item.candidate_id,
        "decision": decision,
        "strongest_match_tier": strongest_match_tier,
        "matched_protected_ids": matched_ids,
        "signals": {
            "raw_hash_match": bool(raw_matches),
            "normalized_hash_match": bool(normalized_matches),
            "ngram_similarity_max": 0.0,
            "module_signature_match": False,
            "property_signature_match": False,
            "forbidden_lineage": forbidden_lineage,
            "split_key_collision": split_key_collision,
            "missing_provenance": missing_provenance,
        },
        "reviewer": None,
        "notes": notes,
    }


def build_contamination_report(protected_index: ProtectedIndex, candidate_index: CandidateIndex) -> dict[str, Any]:
    items = [decide_candidate(item, protected_index) for item in candidate_index.items]
    summary = {
        "total": len(items),
        "clean": sum(item["decision"] == "clean" for item in items),
        "needs_review": sum(item["decision"] == "needs_review" for item in items),
        "quarantined": sum(item["decision"] == "quarantined" for item in items),
        "rejected": sum(item["decision"] == "rejected" for item in items),
    }
    split_key_collisions = [item["candidate_id"] for item in items if item["signals"]["split_key_collision"]]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protected_index_hash": protected_index.protected_index_hash,
        "candidate_index_hash": candidate_index.candidate_index_hash,
        "policy_version": "contamination-policy-phase1-local-v0",
        "producer": Producer(name="train.pipelines.check_contamination", version="phase1-local-v0").model_dump(),
        "summary": summary,
        "split_report": {
            "split_key_collisions": len(split_key_collisions),
            "collision_candidate_ids": split_key_collisions,
            "near_duplicate_train_eval_pairs": 0,
            "blocked_split_keys": split_key_collisions,
        },
        "items": items,
        "contamination_report_hash": EMPTY_SHA256,
    }
    payload["contamination_report_hash"] = canonical_payload_hash(payload, "contamination_report_hash")
    ContaminationReport.model_validate(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Check candidate-index manifest against protected-index manifest.")
    parser.add_argument("--protected-index", type=Path, required=True)
    parser.add_argument("--candidate-index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    protected_index = load_manifest(args.protected_index, ProtectedIndex)
    candidate_index = load_manifest(args.candidate_index, CandidateIndex)
    payload = build_contamination_report(protected_index, candidate_index)
    write_yaml_mapping(args.out, payload)
    print("contamination_check_ok=1")
    print(f"out={args.out}")
    print(f"total={payload['summary']['total']}")
    print(f"clean={payload['summary']['clean']}")
    print(f"needs_review={payload['summary']['needs_review']}")
    print(f"quarantined={payload['summary']['quarantined']}")
    print(f"rejected={payload['summary']['rejected']}")


if __name__ == "__main__":
    main()
