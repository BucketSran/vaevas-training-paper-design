"""
Write an admitted manifest from candidate evidence reports.

This is the admission gate between verification/contamination evidence and
SFT/GRPO packers. Packers should consume this output, not raw candidates.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .manifest_schemas import (
        EMPTY_SHA256,
        SCHEMA_VERSION,
        AdmittedManifest,
        CandidateIndex,
        ContaminationReport,
        DiversityReport,
        EvasVerificationReport,
        Producer,
        SpectreShadowReport,
        SplitManifest,
        StaticCheckReport,
        canonical_payload_hash,
        load_manifest,
        write_yaml_mapping,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from manifest_schemas import (
        EMPTY_SHA256,
        SCHEMA_VERSION,
        AdmittedManifest,
        CandidateIndex,
        ContaminationReport,
        DiversityReport,
        EvasVerificationReport,
        Producer,
        SpectreShadowReport,
        SplitManifest,
        StaticCheckReport,
        canonical_payload_hash,
        load_manifest,
        write_yaml_mapping,
    )


def admitted_id_for(candidate_id: str) -> str:
    if candidate_id.startswith("cand_"):
        return f"adm_{candidate_id[len('cand_'):]}"
    return f"adm_{candidate_id}"


def split_targets_for(item_id: str, split_manifest: SplitManifest) -> tuple[str, list[str]]:
    targets = [split for split, ids in split_manifest.splits.items() if item_id in ids]
    if not targets:
        return "unassigned", []
    primary = "sft_train" if "sft_train" in targets else targets[0]
    return primary, targets


def gate_candidate(
    candidate: Any,
    static_by_candidate: dict[str, Any],
    evas_by_candidate: dict[str, Any],
    contamination_by_candidate: dict[str, Any],
    diversity_by_candidate: dict[str, Any],
    spectre_by_candidate: dict[str, Any],
    require_spectre_shadow: bool,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    static = static_by_candidate.get(candidate.candidate_id)
    evas = evas_by_candidate.get(candidate.candidate_id)
    contamination = contamination_by_candidate.get(candidate.candidate_id)
    diversity = diversity_by_candidate.get(candidate.candidate_id)
    spectre = spectre_by_candidate.get(candidate.candidate_id)

    if static is None or static.decision != "pass":
        reasons.append("static_not_pass")
    if evas is None:
        reasons.append("evas_missing")
    else:
        if evas.compile_status.get("status") != "pass":
            reasons.append("evas_compile_not_pass")
        if evas.simulate_status.get("status") != "pass":
            reasons.append("evas_simulate_not_pass")
        if evas.property_status.get("status") != "pass":
            reasons.append("evas_property_not_pass")
    if contamination is None or contamination.decision != "clean":
        reasons.append("contamination_not_clean")
    if diversity is None or diversity.get("decision") != "accept":
        reasons.append("diversity_not_accept")
    if require_spectre_shadow and spectre is None:
        reasons.append("spectre_shadow_missing")
    if spectre is not None and spectre.get("decision") == "evas_false_positive":
        reasons.append("evas_pass_spectre_fail")
    return not reasons, reasons


def build_admitted_manifest(
    candidate_index: CandidateIndex,
    static_report: StaticCheckReport,
    evas_report: EvasVerificationReport,
    contamination_report: ContaminationReport,
    diversity_report: DiversityReport,
    split_manifest: SplitManifest,
    spectre_report: SpectreShadowReport | None,
    require_spectre_shadow: bool,
    policy_version: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    static_by_candidate = {item.candidate_id: item for item in static_report.items}
    evas_by_candidate = {item.candidate_id: item for item in evas_report.items}
    contamination_by_candidate = {item.candidate_id: item for item in contamination_report.items}
    diversity_by_candidate = {str(item["candidate_id"]): item for item in diversity_report.decisions}
    spectre_by_candidate = {
        str(item["candidate_id"]): item
        for item in (spectre_report.items if spectre_report is not None else [])
    }

    admitted_items: list[dict[str, Any]] = []
    rejected_items: list[dict[str, Any]] = []
    for candidate in candidate_index.items:
        item_id = admitted_id_for(candidate.candidate_id)
        split, pack_targets = split_targets_for(item_id, split_manifest)
        ok, reasons = gate_candidate(
            candidate,
            static_by_candidate,
            evas_by_candidate,
            contamination_by_candidate,
            diversity_by_candidate,
            spectre_by_candidate,
            require_spectre_shadow,
        )
        if split == "unassigned":
            ok = False
            reasons.append("split_unassigned")
        if not ok:
            rejected_items.append({"candidate_id": candidate.candidate_id, "reasons": reasons})
            continue

        admitted_items.append(
            {
                "item_id": item_id,
                "candidate_id": candidate.candidate_id,
                "contract_id": candidate.contract_id,
                "split": split,
                "split_key": candidate.split_key,
                "level": candidate.level,
                "task_form": candidate.task_form,
                "category": candidate.category,
                "artifact_hashes": candidate.artifact_hashes,
                "evidence_refs": candidate.evidence_refs,
                "reward_profile": candidate.reward_profile,
                "use_flags": {
                    "sft_gold": bool(candidate.intended_uses.get("sft_gold") and "sft_train" in pack_targets),
                    "grpo_prompt": bool(candidate.intended_uses.get("grpo_prompt") and "grpo_train" in pack_targets),
                    "repair_data": bool(candidate.intended_uses.get("repair_data")),
                    "eval": bool(candidate.intended_uses.get("eval")),
                    "diagnostics": bool(candidate.intended_uses.get("diagnostics")),
                },
                "pack_targets": pack_targets,
            }
        )

    split_counts = Counter(item["split"] for item in admitted_items)
    level_counts = Counter(item["level"] for item in admitted_items)
    task_form_counts = Counter(item["task_form"] for item in admitted_items)
    category_counts = Counter(item["category"] for item in admitted_items)
    reward_counts = Counter(item["reward_profile"] for item in admitted_items)

    source_hashes = {
        "candidate_index": candidate_index.candidate_index_hash,
        "static_check_report": static_report.static_check_report_hash,
        "evas_verification_report": evas_report.evas_report_hash,
        "contamination_report": contamination_report.contamination_report_hash,
        "diversity_report": diversity_report.diversity_report_hash,
        "split_manifest": split_manifest.split_manifest_hash,
    }
    if spectre_report is not None:
        source_hashes["spectre_shadow_report"] = spectre_report.spectre_shadow_report_hash

    payload = {
        "schema_version": SCHEMA_VERSION,
        "batch_id": candidate_index.batch_id,
        "source_hashes": source_hashes,
        "producer": Producer(name="train.pipelines.write_admitted_manifest", version="phase1-local-v0").model_dump(),
        "admission_policy": {
            "policy_version": policy_version,
            "require_static_pass": True,
            "require_evas_compile_pass": True,
            "require_evas_sim_pass": True,
            "require_evas_property_pass": True,
            "require_contamination_clean": True,
            "require_split_leakage_clear": True,
            "require_spectre_shadow_if_selected": require_spectre_shadow,
            "require_diversity_accept": True,
        },
        "summary": {
            "total": len(admitted_items),
            "rejected": len(rejected_items),
            "by_split": dict(sorted(split_counts.items())),
            "by_level": dict(sorted(level_counts.items())),
            "by_task_form": dict(sorted(task_form_counts.items())),
            "by_category": dict(sorted(category_counts.items())),
            "by_reward_profile": dict(sorted(reward_counts.items())),
        },
        "items": admitted_items,
        "admitted_manifest_hash": EMPTY_SHA256,
    }
    payload["admitted_manifest_hash"] = canonical_payload_hash(payload, "admitted_manifest_hash")
    AdmittedManifest.model_validate(payload)
    return payload, rejected_items


def main() -> None:
    parser = argparse.ArgumentParser(description="Write admitted manifest from candidate evidence reports.")
    parser.add_argument("--candidate-index", type=Path, required=True)
    parser.add_argument("--static-check-report", type=Path, required=True)
    parser.add_argument("--evas-report", type=Path, required=True)
    parser.add_argument("--contamination-report", type=Path, required=True)
    parser.add_argument("--diversity-report", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--spectre-report", type=Path)
    parser.add_argument("--require-spectre-shadow", action="store_true")
    parser.add_argument("--policy-version", default="admission-policy-phase1-local-v0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    candidate_index = load_manifest(args.candidate_index, CandidateIndex)
    static_report = load_manifest(args.static_check_report, StaticCheckReport)
    evas_report = load_manifest(args.evas_report, EvasVerificationReport)
    contamination_report = load_manifest(args.contamination_report, ContaminationReport)
    diversity_report = load_manifest(args.diversity_report, DiversityReport)
    split_manifest = load_manifest(args.split_manifest, SplitManifest)
    spectre_report = load_manifest(args.spectre_report, SpectreShadowReport) if args.spectre_report else None

    payload, rejected_items = build_admitted_manifest(
        candidate_index=candidate_index,
        static_report=static_report,
        evas_report=evas_report,
        contamination_report=contamination_report,
        diversity_report=diversity_report,
        split_manifest=split_manifest,
        spectre_report=spectre_report,
        require_spectre_shadow=args.require_spectre_shadow,
        policy_version=args.policy_version,
    )
    write_yaml_mapping(args.out, payload)
    print("admitted_manifest_ok=1")
    print(f"out={args.out}")
    print(f"admitted_count={len(payload['items'])}")
    print(f"rejected_count={len(rejected_items)}")
    print(f"admitted_manifest_hash={payload['admitted_manifest_hash']}")


if __name__ == "__main__":
    main()
