"""
Pack draft SFT/GRPO JSONL directly from generated artifact candidates.

This is an overnight data-preparation tool, not the final admission path. The
output is labeled draft/unadmitted and must not be reported as clean training
data until contamination, EVAS, Spectre-shadow policy, and admitted manifests
are available.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from .manifest_schemas import (
        FORBIDDEN_GRPO_COMPLETION_KEYS,
        SCHEMA_VERSION,
        CandidateIndex,
        Producer,
        canonical_payload_hash,
        load_manifest,
        sha256_file,
        train_relative_ref,
        resolve_train_ref,
        write_jsonl,
        write_yaml_mapping,
    )
    from .pipeline_common import (
        SYSTEM_PROMPT,
        grpo_prompt_from_contract,
        instruction_from_contract,
        load_contract,
        public_contract_summary,
        sft_output_from_artifact,
        verify_file_hash,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from manifest_schemas import (
        FORBIDDEN_GRPO_COMPLETION_KEYS,
        SCHEMA_VERSION,
        CandidateIndex,
        Producer,
        canonical_payload_hash,
        load_manifest,
        sha256_file,
        train_relative_ref,
        resolve_train_ref,
        write_jsonl,
        write_yaml_mapping,
    )
    from pipeline_common import (
        SYSTEM_PROMPT,
        grpo_prompt_from_contract,
        instruction_from_contract,
        load_contract,
        public_contract_summary,
        sft_output_from_artifact,
        verify_file_hash,
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def build_records(
    candidate_index: CandidateIndex,
    train_root: Path,
    validation_every: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
    sft_train: list[dict[str, Any]] = []
    sft_val: list[dict[str, Any]] = []
    grpo_prompts: list[dict[str, Any]] = []
    item_ids = {"sft_train": [], "sft_val": [], "grpo_prompts": []}

    for offset, candidate in enumerate(candidate_index.items, start=1):
        contract_path, contract = load_contract(candidate.contract_ref, train_root)
        verify_file_hash(contract_path, candidate.contract_sha256, f"{candidate.candidate_id}.contract")
        main_key = candidate.lineage.get("sft_gold_artifact_key")
        require(isinstance(main_key, str) and main_key in candidate.artifact_refs, f"{candidate.candidate_id} missing main artifact")
        artifact_ref = candidate.artifact_refs[main_key]
        artifact_path = resolve_train_ref(artifact_ref, train_root)
        verify_file_hash(artifact_path, candidate.artifact_hashes[main_key], f"{candidate.candidate_id}.{main_key}")
        artifact_text = artifact_path.read_text(encoding="utf-8")

        if candidate.intended_uses.get("sft_gold", False):
            sft_record = {
                "instruction": instruction_from_contract(contract),
                "input": public_contract_summary(contract),
                "output": sft_output_from_artifact(contract, artifact_text),
                "system": SYSTEM_PROMPT,
                "metadata": {
                    "candidate_id": candidate.candidate_id,
                    "contract_id": candidate.contract_id,
                    "draft_unadmitted": True,
                    "gold_artifact_key": main_key,
                },
            }
            if validation_every > 0 and offset % validation_every == 0:
                sft_val.append(sft_record)
                item_ids["sft_val"].append(candidate.candidate_id)
            else:
                sft_train.append(sft_record)
                item_ids["sft_train"].append(candidate.candidate_id)

        if candidate.intended_uses.get("grpo_prompt", False):
            grpo_record = {
                "item_id": candidate.candidate_id,
                "prompt": grpo_prompt_from_contract(contract),
                "contract_ref": candidate.contract_ref,
                "reward_profile": candidate.reward_profile,
                "reward_runtime_ref": {
                    "config_ref": "train/docs/DIAGNOSTIC_REWARD_SPEC.md",
                    "implementation_commit": None,
                    "evas_profile": "pending",
                    "timeout_s": 10,
                    "draft_unadmitted": True,
                },
            }
            leaked = FORBIDDEN_GRPO_COMPLETION_KEYS & set(grpo_record)
            require(not leaked, f"GRPO record leaks forbidden completion keys: {sorted(leaked)}")
            grpo_prompts.append(grpo_record)
            item_ids["grpo_prompts"].append(candidate.candidate_id)

    return sft_train, sft_val, grpo_prompts, item_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-index", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--train-root", type=Path, default=Path("train"))
    parser.add_argument("--validation-every", type=int, default=5)
    parser.add_argument("--base-model-ref", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    args = parser.parse_args()

    candidate_index = load_manifest(args.candidate_index, CandidateIndex)
    sft_train, sft_val, grpo_prompts, item_ids = build_records(candidate_index, args.train_root, args.validation_every)
    require(sft_train or sft_val or grpo_prompts, "no draft records generated")

    sft_dir = args.out_dir / "draft_sft"
    grpo_dir = args.out_dir / "draft_grpo"
    train_path = sft_dir / "train.jsonl"
    val_path = sft_dir / "val.jsonl"
    prompts_path = grpo_dir / "prompts.jsonl"
    train_hash = write_jsonl(train_path, sft_train)
    val_hash = write_jsonl(val_path, sft_val)
    prompts_hash = write_jsonl(prompts_path, grpo_prompts)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "status": "draft_unadmitted_training_pack",
        "candidate_index_hash": candidate_index.candidate_index_hash,
        "producer": Producer(name="train.pipelines.pack_draft_training", version="phase1.v0.1").model_dump(),
        "output_files": {
            "sft_train": train_relative_ref(train_path, args.train_root),
            "sft_val": train_relative_ref(val_path, args.train_root),
            "grpo_prompts": train_relative_ref(prompts_path, args.train_root),
        },
        "output_hashes": {
            "sft_train": train_hash,
            "sft_val": val_hash,
            "grpo_prompts": prompts_hash,
        },
        "format": {
            "sft": "llamafactory_alpaca_draft",
            "grpo": "trl_grpo_prompt_draft",
        },
        "base_model_ref": args.base_model_ref,
        "item_ids": item_ids,
        "counts": {
            "sft_train": len(sft_train),
            "sft_val": len(sft_val),
            "grpo_prompts": len(grpo_prompts),
        },
        "admission_boundary": {
            "admitted_training_data": False,
            "requires_contamination_gate": True,
            "requires_evas_gate": True,
            "requires_spectre_shadow_policy": True,
            "requires_admitted_manifest_before_paper_claim": True,
        },
        "draft_training_pack_hash": "",
    }
    payload["draft_training_pack_hash"] = canonical_payload_hash(payload, "draft_training_pack_hash")
    write_yaml_mapping(args.manifest_out, payload)
    print(
        "PASS pack_draft_training "
        f"sft_train={len(sft_train)} sft_val={len(sft_val)} grpo_prompts={len(grpo_prompts)} "
        f"draft_training_pack_hash={payload['draft_training_pack_hash']}"
    )
    print(f"sft_train_hash={sha256_file(train_path)}")
    print(f"sft_val_hash={sha256_file(val_path)}")
    print(f"grpo_prompt_hash={sha256_file(prompts_path)}")


if __name__ == "__main__":
    main()
