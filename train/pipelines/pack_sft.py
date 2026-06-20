"""
Pack admitted Phase 1 items into SFT JSONL.

The packer consumes only admitted manifests plus candidate metadata. It fails
closed if a referenced gold artifact is missing or its hash does not match.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from .manifest_schemas import (
        SCHEMA_VERSION,
        AdmittedManifest,
        CandidateIndex,
        Producer,
        SftPackManifest,
        load_manifest,
        resolve_train_ref,
        sha256_file,
        train_relative_ref,
        write_jsonl,
        write_yaml_mapping,
    )
    from .pipeline_common import (
        SYSTEM_PROMPT,
        instruction_from_contract,
        load_contract,
        public_contract_summary,
        sft_output_from_dut,
        verify_file_hash,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from manifest_schemas import (
        SCHEMA_VERSION,
        AdmittedManifest,
        CandidateIndex,
        Producer,
        SftPackManifest,
        load_manifest,
        resolve_train_ref,
        sha256_file,
        train_relative_ref,
        write_jsonl,
        write_yaml_mapping,
    )
    from pipeline_common import (
        SYSTEM_PROMPT,
        instruction_from_contract,
        load_contract,
        public_contract_summary,
        sft_output_from_dut,
        verify_file_hash,
    )


def build_sft_records(
    admitted_manifest: AdmittedManifest,
    candidate_index: CandidateIndex,
    train_root: Path,
    gold_artifact_key: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
    candidates = {item.candidate_id: item for item in candidate_index.items}
    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []
    item_ids = {"train": [], "val": []}

    for admitted_item in admitted_manifest.items:
        if not admitted_item.use_flags.get("sft_gold", False):
            continue
        if "sft_train" not in admitted_item.pack_targets and admitted_item.split != "validation":
            continue

        candidate = candidates.get(admitted_item.candidate_id)
        if candidate is None:
            raise ValueError(f"admitted item references unknown candidate_id={admitted_item.candidate_id}")

        contract_path, contract = load_contract(candidate.contract_ref, train_root)
        verify_file_hash(contract_path, candidate.contract_sha256, f"{candidate.candidate_id}.contract")

        artifact_ref = candidate.artifact_refs.get(gold_artifact_key)
        expected_hash = candidate.artifact_hashes.get(gold_artifact_key)
        if not artifact_ref or not expected_hash:
            raise ValueError(f"{candidate.candidate_id} missing {gold_artifact_key} artifact ref/hash")
        artifact_path = resolve_train_ref(artifact_ref, train_root)
        if not artifact_path.exists():
            raise FileNotFoundError(f"gold artifact does not exist: {artifact_ref}")
        verify_file_hash(artifact_path, expected_hash, f"{candidate.candidate_id}.{gold_artifact_key}")

        dut_text = artifact_path.read_text(encoding="utf-8")
        record = {
            "instruction": instruction_from_contract(contract),
            "input": public_contract_summary(contract),
            "output": sft_output_from_dut(contract, dut_text),
            "system": SYSTEM_PROMPT,
        }
        if admitted_item.split == "validation":
            val_records.append(record)
            item_ids["val"].append(admitted_item.item_id)
        else:
            train_records.append(record)
            item_ids["train"].append(admitted_item.item_id)

    return train_records, val_records, item_ids


def pack_sft(
    admitted_manifest_path: Path,
    candidate_index_path: Path,
    out_dir: Path,
    manifest_out: Path,
    run_id: str,
    base_model_ref: str,
    tokenizer_ref: str,
    train_root: Path,
    gold_artifact_key: str,
) -> dict[str, Any]:
    admitted_manifest = load_manifest(admitted_manifest_path, AdmittedManifest)
    candidate_index = load_manifest(candidate_index_path, CandidateIndex)
    train_records, val_records, item_ids = build_sft_records(
        admitted_manifest,
        candidate_index,
        train_root,
        gold_artifact_key,
    )

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    train_hash = write_jsonl(train_path, train_records)
    val_hash = write_jsonl(val_path, val_records)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "admitted_manifest_hash": admitted_manifest.admitted_manifest_hash,
        "producer": Producer(name="train.pipelines.pack_sft", version="phase1-local-v0").model_dump(),
        "output_files": {
            "train": train_relative_ref(train_path, train_root),
            "val": train_relative_ref(val_path, train_root),
        },
        "output_hashes": {"train": train_hash, "val": val_hash},
        "format": "llamafactory_alpaca",
        "base_model_ref": base_model_ref,
        "tokenizer_ref": tokenizer_ref,
        "special_tokens": ["<think>", "</think>", "<answer>", "</answer>"],
        "dataset_info_ref": "data/llamafactory/dataset_info.json",
        "item_ids": item_ids,
        "pack_format_contract": {
            "required_jsonl_keys": ["instruction", "input", "output", "system"],
            "model_visible_fields": ["public_prompt", "public_contract_summary", "gold_completion"],
            "model_hidden_fields": ["artifact_hashes", "contamination_report", "spectre_shadow_report"],
        },
    }
    SftPackManifest.model_validate(payload)
    write_yaml_mapping(manifest_out, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack admitted Phase 1 items into SFT JSONL.")
    parser.add_argument("--admitted-manifest", type=Path, required=True)
    parser.add_argument("--candidate-index", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--run-id", default="phase1-local-sft")
    parser.add_argument("--base-model-ref", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--tokenizer-ref", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--train-root", type=Path, default=Path("train"))
    parser.add_argument("--gold-artifact-key", default="dut_va")
    args = parser.parse_args()

    payload = pack_sft(
        admitted_manifest_path=args.admitted_manifest,
        candidate_index_path=args.candidate_index,
        out_dir=args.out_dir,
        manifest_out=args.manifest_out,
        run_id=args.run_id,
        base_model_ref=args.base_model_ref,
        tokenizer_ref=args.tokenizer_ref,
        train_root=args.train_root,
        gold_artifact_key=args.gold_artifact_key,
    )
    print("sft_pack_ok=1")
    print(f"train_count={len(payload['item_ids']['train'])}")
    print(f"val_count={len(payload['item_ids']['val'])}")
    print(f"train_hash={payload['output_hashes']['train']}")
    print(f"val_hash={payload['output_hashes']['val']}")


if __name__ == "__main__":
    main()
