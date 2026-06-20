"""
Pack admitted Phase 1 items into GRPO prompt JSONL.

The generated model-visible records intentionally omit gold outputs and answer
fields. Reward/runtime metadata is kept in the manifest.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from .manifest_schemas import (
        FORBIDDEN_GRPO_COMPLETION_KEYS,
        SCHEMA_VERSION,
        AdmittedManifest,
        CandidateIndex,
        GrpoPromptManifest,
        Producer,
        load_manifest,
        train_relative_ref,
        write_jsonl,
        write_yaml_mapping,
    )
    from .pipeline_common import grpo_prompt_from_contract, load_contract, verify_file_hash
except ImportError:  # pragma: no cover - direct script execution fallback
    from manifest_schemas import (
        FORBIDDEN_GRPO_COMPLETION_KEYS,
        SCHEMA_VERSION,
        AdmittedManifest,
        CandidateIndex,
        GrpoPromptManifest,
        Producer,
        load_manifest,
        train_relative_ref,
        write_jsonl,
        write_yaml_mapping,
    )
    from pipeline_common import grpo_prompt_from_contract, load_contract, verify_file_hash


def build_grpo_records(
    admitted_manifest: AdmittedManifest,
    candidate_index: CandidateIndex,
    train_root: Path,
    reward_runtime_ref: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    candidates = {item.candidate_id: item for item in candidate_index.items}
    records: list[dict[str, Any]] = []
    item_ids: list[str] = []
    reward_profiles: set[str] = set()

    for admitted_item in admitted_manifest.items:
        if not admitted_item.use_flags.get("grpo_prompt", False):
            continue
        if "grpo_train" not in admitted_item.pack_targets:
            continue

        candidate = candidates.get(admitted_item.candidate_id)
        if candidate is None:
            raise ValueError(f"admitted item references unknown candidate_id={admitted_item.candidate_id}")

        contract_path, contract = load_contract(candidate.contract_ref, train_root)
        verify_file_hash(contract_path, candidate.contract_sha256, f"{candidate.candidate_id}.contract")
        record = {
            "item_id": admitted_item.item_id,
            "prompt": grpo_prompt_from_contract(contract),
            "contract_ref": candidate.contract_ref,
            "reward_profile": admitted_item.reward_profile,
            "reward_runtime_ref": reward_runtime_ref,
        }
        leaked = FORBIDDEN_GRPO_COMPLETION_KEYS & set(record)
        if leaked:
            raise ValueError(f"GRPO record leaks forbidden completion keys: {sorted(leaked)}")
        records.append(record)
        item_ids.append(admitted_item.item_id)
        reward_profiles.add(admitted_item.reward_profile)

    return records, item_ids, sorted(reward_profiles)


def pack_grpo(
    admitted_manifest_path: Path,
    candidate_index_path: Path,
    out_dir: Path,
    manifest_out: Path,
    run_id: str,
    base_checkpoint_ref: str,
    train_root: Path,
    evas_profile: str,
    timeout_s: int,
    group_size_hint: int,
) -> dict[str, Any]:
    admitted_manifest = load_manifest(admitted_manifest_path, AdmittedManifest)
    candidate_index = load_manifest(candidate_index_path, CandidateIndex)
    reward_runtime_ref = {
        "config_ref": "train/docs/DIAGNOSTIC_REWARD_SPEC.md",
        "implementation_commit": None,
        "evas_profile": evas_profile,
        "timeout_s": timeout_s,
        "anti_hack_policy": "diagnostic-reward-spec-v0",
    }
    records, item_ids, reward_profiles = build_grpo_records(
        admitted_manifest,
        candidate_index,
        train_root,
        reward_runtime_ref,
    )

    prompt_path = out_dir / "prompts.jsonl"
    prompt_hash = write_jsonl(prompt_path, records)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "admitted_manifest_hash": admitted_manifest.admitted_manifest_hash,
        "producer": Producer(name="train.pipelines.pack_grpo", version="phase1-local-v0").model_dump(),
        "output_files": {"prompts": train_relative_ref(prompt_path, train_root)},
        "output_hashes": {"prompts": prompt_hash},
        "format": "trl_grpo",
        "base_checkpoint_ref": base_checkpoint_ref,
        "reward_runtime_ref": reward_runtime_ref,
        "reward_profiles": reward_profiles,
        "group_size_hint": group_size_hint,
        "item_ids": {"prompts": item_ids},
        "prompt_format_contract": {
            "required_jsonl_keys": [
                "item_id",
                "prompt",
                "contract_ref",
                "reward_profile",
                "reward_runtime_ref",
            ],
            "forbidden_jsonl_keys": sorted(FORBIDDEN_GRPO_COMPLETION_KEYS),
            "model_visible_fields": ["public_prompt", "public_contract_summary"],
            "model_hidden_fields": ["artifact_hashes", "contamination_report", "spectre_shadow_report"],
        },
    }
    GrpoPromptManifest.model_validate(payload)
    write_yaml_mapping(manifest_out, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack admitted Phase 1 items into GRPO prompt JSONL.")
    parser.add_argument("--admitted-manifest", type=Path, required=True)
    parser.add_argument("--candidate-index", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--run-id", default="phase1-local-grpo")
    parser.add_argument("--base-checkpoint-ref", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--train-root", type=Path, default=Path("train"))
    parser.add_argument("--evas-profile", default="balanced")
    parser.add_argument("--timeout-s", type=int, default=10)
    parser.add_argument("--group-size-hint", type=int, default=8)
    args = parser.parse_args()

    payload = pack_grpo(
        admitted_manifest_path=args.admitted_manifest,
        candidate_index_path=args.candidate_index,
        out_dir=args.out_dir,
        manifest_out=args.manifest_out,
        run_id=args.run_id,
        base_checkpoint_ref=args.base_checkpoint_ref,
        train_root=args.train_root,
        evas_profile=args.evas_profile,
        timeout_s=args.timeout_s,
        group_size_hint=args.group_size_hint,
    )
    print("grpo_pack_ok=1")
    print(f"prompt_count={len(payload['item_ids']['prompts'])}")
    print(f"prompt_hash={payload['output_hashes']['prompts']}")


if __name__ == "__main__":
    main()
