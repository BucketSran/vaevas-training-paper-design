"""
Write a structured review manifest and Markdown report for generated contracts.

The default output is conservative: contracts that pass mechanical checks are
marked `needs_revision` until a human/agent reviewer explicitly promotes them to
`accept_for_generation`. This script does not generate artifacts or training
data.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .manifest_schemas import (
        SCHEMA_VERSION,
        ContractReviewManifest,
        GeneratedContractIndex,
        Producer,
        canonical_payload_hash,
        read_yaml_mapping,
        write_yaml_mapping,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from manifest_schemas import (
        SCHEMA_VERSION,
        ContractReviewManifest,
        GeneratedContractIndex,
        Producer,
        canonical_payload_hash,
        read_yaml_mapping,
        write_yaml_mapping,
    )


def decision_for_item(item: Any, default_pass_decision: str) -> str:
    review = item.mechanical_review
    blocking_findings = review.get("blocking_findings", [])
    if blocking_findings:
        return "reject"
    return default_pass_decision


def allowed_next_stage(decision: str) -> str:
    if decision == "accept_for_generation":
        return "artifact_generation"
    if decision == "needs_revision":
        return "contract_revision"
    return "none"


def required_edits_for_item(item: Any, decision: str) -> list[str]:
    if decision == "accept_for_generation":
        return []
    if decision == "needs_revision":
        return [
            "Manual reviewer must confirm observable properties are meaningful and non-duplicative.",
            "Manual reviewer must confirm provenance and split_key are clean-room safe.",
            "Manual reviewer must confirm this contract should enter artifact generation.",
        ]
    return list(item.mechanical_review.get("blocking_findings", []))


def build_manifest(index_payload: dict[str, Any], reviewer: str, default_pass_decision: str) -> dict[str, Any]:
    index = GeneratedContractIndex.model_validate(index_payload)
    items: list[dict[str, Any]] = []
    for item in index.items:
        decision = decision_for_item(item, default_pass_decision)
        items.append(
            {
                "generated_contract_id": item.generated_contract_id,
                "contract_ref": item.contract_ref,
                "decision": decision,
                "reviewer": reviewer,
                "blocking_findings": list(item.mechanical_review.get("blocking_findings", [])),
                "required_edits": required_edits_for_item(item, decision),
                "allowed_next_stage": allowed_next_stage(decision),
                "rationale": (
                    "Mechanical checks passed; manual review still required."
                    if decision == "needs_revision"
                    else "Mechanical checks passed and reviewer approved artifact generation."
                    if decision == "accept_for_generation"
                    else "Mechanical checks found blockers."
                ),
            }
        )

    decision_counts = Counter(item["decision"] for item in items)
    next_stage_counts = Counter(item["allowed_next_stage"] for item in items)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": index.run_id,
        "generated_contract_index_hash": index.generated_contract_index_hash,
        "producer": Producer(name="train.pipelines.write_contract_review_manifest", version="phase1.v0.1").model_dump(),
        "review_policy": {
            "reviewer": reviewer,
            "default_pass_decision": default_pass_decision,
            "manual_review_required_before_artifact_generation": True,
            "not_admitted_training_data": True,
        },
        "summary": {
            "total": len(items),
            "by_decision": dict(sorted(decision_counts.items())),
            "by_allowed_next_stage": dict(sorted(next_stage_counts.items())),
            "accepted_for_generation": decision_counts.get("accept_for_generation", 0),
            "needs_revision": decision_counts.get("needs_revision", 0),
            "quarantined": decision_counts.get("quarantine", 0),
            "rejected": decision_counts.get("reject", 0),
        },
        "items": items,
        "fixture_notes": [
            "Contract review manifest is a review gate only.",
            "No contract is admitted as SFT/GRPO/eval data by this manifest.",
        ],
        "contract_review_manifest_hash": "",
    }
    payload["contract_review_manifest_hash"] = canonical_payload_hash(payload, "contract_review_manifest_hash")
    ContractReviewManifest.model_validate(payload)
    return payload


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Contract Review Report",
        "",
        f"Run ID: `{manifest['run_id']}`",
        f"Review manifest hash: `{manifest['contract_review_manifest_hash']}`",
        "",
        "## Summary",
        "",
    ]
    for key, value in manifest["summary"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Items", ""])
    for item in manifest["items"]:
        lines.extend(
            [
                f"### `{item['generated_contract_id']}`",
                "",
                f"- decision: `{item['decision']}`",
                f"- allowed_next_stage: `{item['allowed_next_stage']}`",
                f"- contract_ref: `{item['contract_ref']}`",
                f"- rationale: {item['rationale']}",
                f"- blocking_findings: {item['blocking_findings']}",
                f"- required_edits: {item['required_edits']}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--reviewer", default="mechanical_contract_review")
    parser.add_argument(
        "--default-pass-decision",
        choices=["accept_for_generation", "needs_revision"],
        default="needs_revision",
    )
    args = parser.parse_args()

    index_payload = read_yaml_mapping(args.index)
    manifest = build_manifest(index_payload, args.reviewer, args.default_pass_decision)
    write_yaml_mapping(args.out, manifest)
    write_report(args.report_out, manifest)
    print(
        "PASS write_contract_review_manifest "
        f"items={manifest['summary']['total']} "
        f"accepted={manifest['summary']['accepted_for_generation']} "
        f"needs_revision={manifest['summary']['needs_revision']}"
    )


if __name__ == "__main__":
    main()
