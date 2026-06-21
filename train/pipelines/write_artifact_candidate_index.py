"""
Write a candidate index from generated contracts plus generated artifacts.

This script is the bridge from reviewed contract YAMLs to draft training-pack
inputs. It records artifact hashes and provenance, but it does not run EVAS,
Spectre, contamination checks, admission, SFT, or GRPO training.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .manifest_schemas import (
        SCHEMA_VERSION,
        CandidateIndex,
        ContractReviewManifest,
        GeneratedContractIndex,
        Producer,
        SeedCatalog,
        canonical_payload_hash,
        load_manifest,
        read_yaml_mapping,
        sha256_file,
        train_relative_ref,
        write_yaml_mapping,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from manifest_schemas import (
        SCHEMA_VERSION,
        CandidateIndex,
        ContractReviewManifest,
        GeneratedContractIndex,
        Producer,
        SeedCatalog,
        canonical_payload_hash,
        load_manifest,
        read_yaml_mapping,
        sha256_file,
        train_relative_ref,
        write_yaml_mapping,
    )


DEFAULT_TRAIN_ROOT = Path("train")
DEFAULT_SEED_CATALOG = DEFAULT_TRAIN_ROOT / "data/seeds/seed_catalog.phase1-pilot-0001.yaml"
FORBIDDEN_ARTIFACT_FRAGMENTS = (
    "benchmark-vabench-release-v1",
    "score_denominator_manifest",
    "gold_completion",
    "reference_answer",
    "task_id_special_case",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_seed_refs(seed_catalog_path: Path) -> dict[str, dict[str, Any]]:
    catalog = load_manifest(seed_catalog_path, SeedCatalog)
    return {item.seed_id: item.model_dump() for item in catalog.items}


def artifact_files_for(candidate_dir: Path) -> dict[str, Path]:
    artifact_refs: dict[str, Path] = {}
    known = {
        "solution.va": "solution_va",
        "dut.va": "dut_va",
        "tb.scs": "tb_scs",
        "checker.yaml": "checker_yaml",
        "buggy.va": "buggy_va",
        "fixed.va": "fixed_va",
        "system.va": "system_va",
        "notes.md": "notes_md",
    }
    for filename, key in known.items():
        path = candidate_dir / filename
        if path.exists():
            artifact_refs[key] = path
    return artifact_refs


def check_main_artifact(path: Path) -> list[str]:
    findings: list[str] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    lowered = text.lower()
    if "module" not in lowered or "endmodule" not in lowered:
        findings.append("main Verilog-A artifact must contain module/endmodule")
    forbidden = [fragment for fragment in FORBIDDEN_ARTIFACT_FRAGMENTS if fragment.lower() in lowered]
    if forbidden:
        findings.append(f"main artifact contains forbidden fragments: {forbidden}")
    if len(text.strip()) < 80:
        findings.append("main artifact is too short to be useful training output")
    return findings


def build_candidate_items(
    index: GeneratedContractIndex,
    review: ContractReviewManifest,
    artifact_root: Path,
    train_root: Path,
    seed_refs_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    review_by_id = {item.generated_contract_id: item for item in review.items}
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for item in index.items:
        review_item = review_by_id[item.generated_contract_id]
        if review_item.decision != "accept_for_generation":
            rejected.append({"generated_contract_id": item.generated_contract_id, "reason": review_item.decision})
            continue

        candidate_dir = artifact_root / item.generated_contract_id
        artifact_paths = artifact_files_for(candidate_dir)
        main_artifact = artifact_paths.get("solution_va") or artifact_paths.get("dut_va") or artifact_paths.get("system_va")
        if main_artifact is None:
            rejected.append({"generated_contract_id": item.generated_contract_id, "reason": "missing main Verilog-A artifact"})
            continue

        findings = check_main_artifact(main_artifact)
        if findings:
            rejected.append({"generated_contract_id": item.generated_contract_id, "reason": "; ".join(findings)})
            continue

        contract_path = train_root / item.contract_ref
        contract = read_yaml_mapping(contract_path)
        seed_ids = item.seed_ids
        seed_refs = []
        for seed_id in seed_ids:
            seed_ref = seed_refs_by_id.get(seed_id)
            require(seed_ref is not None, f"seed_id not found in seed catalog: {seed_id}")
            seed_refs.append(
                {
                    key: seed_ref[key]
                    for key in [
                        "seed_id",
                        "source_tier",
                        "source_kind",
                        "source_ref",
                        "license",
                        "allowed_uses",
                        "forbidden_uses",
                    ]
                }
            )

        artifact_refs = {
            key: train_relative_ref(path, train_root)
            for key, path in sorted(artifact_paths.items())
        }
        artifact_hashes = {
            key: sha256_file(path)
            for key, path in sorted(artifact_paths.items())
        }
        main_key = next(key for key, path in artifact_paths.items() if path == main_artifact)
        candidate_id = f"cand_{item.generated_contract_id}"
        result_ref = train_relative_ref(artifact_root.parent, train_root)
        evidence_refs = {
            "generated_contract_index": f"{result_ref}/generated_contract_index.yaml",
            "contract_review_manifest": f"{result_ref}/review_manifest.yaml",
            "artifact_root": train_relative_ref(artifact_root, train_root),
        }
        candidates.append(
            {
                "candidate_id": candidate_id,
                "contract_id": item.generated_contract_id,
                "contract_ref": item.contract_ref,
                "contract_sha256": item.contract_sha256,
                "state": "artifact_generated_unverified",
                "level": item.level,
                "task_form": item.task_form,
                "category": item.category,
                "split_key": item.split_key,
                "seed_refs": seed_refs,
                "artifact_refs": artifact_refs,
                "artifact_hashes": artifact_hashes,
                "allowed_features": {
                    "domain": contract.get("domain", "voltage"),
                    "event_driven": True,
                    "unsupported_constructs_disallowed": True,
                },
                "forbidden_constructs": list(contract.get("forbidden_constructs", [])),
                "reward_profile": f"{item.level}_{item.task_form}",
                "intended_uses": {
                    "sft_gold": True,
                    "grpo_prompt": True,
                    "repair_data": item.task_form == "bugfix",
                    "eval": False,
                    "diagnostics": True,
                },
                "lineage": {
                    "source_generated_contract_index_hash": index.generated_contract_index_hash,
                    "source_contract_review_manifest_hash": review.contract_review_manifest_hash,
                    "artifact_generator": "remote_codex_default",
                    "sft_gold_artifact_key": main_key,
                    "admission_status": "not_admitted",
                },
                "evidence_refs": evidence_refs,
            }
        )
    return candidates, rejected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-contract-index", type=Path, required=True)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-catalog", type=Path, default=DEFAULT_SEED_CATALOG)
    parser.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--expect-min-candidates", type=int, default=1)
    args = parser.parse_args()

    index = load_manifest(args.generated_contract_index, GeneratedContractIndex)
    review = load_manifest(args.review_manifest, ContractReviewManifest)
    require(review.generated_contract_index_hash == index.generated_contract_index_hash, "review/index hash mismatch")
    seed_refs_by_id = load_seed_refs(args.seed_catalog)
    candidates, rejected = build_candidate_items(index, review, args.artifact_root, args.train_root, seed_refs_by_id)
    require(
        len(candidates) >= args.expect_min_candidates,
        f"expected at least {args.expect_min_candidates} candidates, got {len(candidates)}; rejected={rejected}",
    )

    level_counts = Counter(item["level"] for item in candidates)
    task_form_counts = Counter(item["task_form"] for item in candidates)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "batch_id": args.batch_id or f"artifact-{index.run_id}",
        "batch_plan_ref": "data/manifests/synthesis/synthesis_run.contract-batch-0025.yaml",
        "producer": Producer(name="train.pipelines.write_artifact_candidate_index", version="phase1.v0.1").model_dump(),
        "candidate_index_hash": "",
        "items": candidates,
        "fixture_notes": [
            "Artifact candidates are generated and hash-tracked but not admitted training data.",
            "EVAS, Spectre, contamination, and admission gates remain required for paper claims.",
            f"Rejected or skipped generated contracts: {rejected}",
            f"level_counts={dict(sorted(level_counts.items()))}; task_form_counts={dict(sorted(task_form_counts.items()))}",
        ],
    }
    payload["candidate_index_hash"] = canonical_payload_hash(payload, "candidate_index_hash")
    CandidateIndex.model_validate(payload)
    write_yaml_mapping(args.out, payload)
    print(
        "PASS write_artifact_candidate_index "
        f"candidates={len(candidates)} rejected={len(rejected)} "
        f"candidate_index_hash={payload['candidate_index_hash']}"
    )


if __name__ == "__main__":
    main()
