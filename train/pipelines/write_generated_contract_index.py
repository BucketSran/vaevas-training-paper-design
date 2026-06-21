"""
Validate generated draft contracts and write a generated contract index.

This script receives small draft contract YAML files from a synthesis smoke run.
It does not admit data, generate Verilog-A, run EVAS/Spectre, or pack SFT/GRPO.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .contract_review_gate import load_contract_schema, validate_contract_payload
    from .manifest_schemas import (
        SCHEMA_VERSION,
        GeneratedContractIndex,
        canonical_payload_hash,
        read_yaml_mapping,
        sha256_file,
        train_relative_ref,
        write_yaml_mapping,
    )
    from .validate_synthesis_run import DEFAULT_RUN_PLAN, DEFAULT_TRAIN_ROOT, load_run_plan, validate_run_plan
except ImportError:  # pragma: no cover - direct script execution fallback
    from contract_review_gate import load_contract_schema, validate_contract_payload
    from manifest_schemas import (
        SCHEMA_VERSION,
        GeneratedContractIndex,
        canonical_payload_hash,
        read_yaml_mapping,
        sha256_file,
        train_relative_ref,
        write_yaml_mapping,
    )
    from validate_synthesis_run import DEFAULT_RUN_PLAN, DEFAULT_TRAIN_ROOT, load_run_plan, validate_run_plan


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(record)
    return records


def build_prompt_map(request_jsonl: Path | None) -> dict[str, dict[str, Any]]:
    if request_jsonl is None:
        return {}
    records = read_jsonl(request_jsonl)
    prompt_map: dict[str, dict[str, Any]] = {}
    for record in records:
        prompt_map[str(record["prompt_id"])] = record
        prompt_map[str(record.get("request_id", record["prompt_id"]))] = record
    return prompt_map


def collect_contract_paths(contracts_dir: Path) -> list[Path]:
    require(contracts_dir.exists() and contracts_dir.is_dir(), f"contracts-dir does not exist: {contracts_dir}")
    paths = sorted(contracts_dir.glob("*.yaml"))
    require(paths, f"no generated contract YAML files found in {contracts_dir}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthesis-run", type=Path, default=DEFAULT_RUN_PLAN)
    parser.add_argument("--contracts-dir", type=Path, required=True)
    parser.add_argument("--request-jsonl", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--model-ref", default="remote_codex_default")
    parser.add_argument("--generation-kind", default="contract_proposal")
    parser.add_argument("--expect-count", type=int, default=None)
    args = parser.parse_args()

    run_plan = load_run_plan(args.synthesis_run)
    validate_run_plan(run_plan, args.train_root)
    prompt_map = build_prompt_map(args.request_jsonl)
    schema = load_contract_schema(args.train_root)
    contract_paths = collect_contract_paths(args.contracts_dir)

    require(len(contract_paths) <= int(run_plan.model_policy["max_contracts"]), "too many generated contracts for run plan")
    if args.expect_count is not None:
        require(len(contract_paths) == args.expect_count, f"expected {args.expect_count} contracts, got {len(contract_paths)}")
    allowed_seed_ids = set(read_yaml_mapping(args.train_root / run_plan.pilot_plan_ref)["planned_seed_ids"])

    items: list[dict[str, Any]] = []
    review_counter: Counter[str] = Counter()
    level_counter: Counter[str] = Counter()
    task_form_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()

    for contract_path in contract_paths:
        contract = read_yaml_mapping(contract_path)
        review = validate_contract_payload(contract, schema, allowed_seed_ids=allowed_seed_ids)
        require(review["decision"] != "reject", f"{contract_path} failed mechanical review: {review['blocking_findings']}")

        seed_ids = list(review["seed_ids"])
        require(seed_ids, f"{contract_path} has no seed IDs")
        generator = contract.get("provenance", {}).get("generator", {})
        source_prompt_id = str(generator.get("prompt_id") or f"{seed_ids[0]}.contract_proposal")
        prompt_record = prompt_map.get(source_prompt_id, {})
        prompt_sha256 = prompt_record.get("prompt_sha256") or generator.get("prompt_hash")

        item = {
            "generated_contract_id": str(contract["id"]),
            "contract_ref": train_relative_ref(contract_path, args.train_root),
            "contract_sha256": sha256_file(contract_path),
            "source_prompt_id": source_prompt_id,
            "prompt_sha256": prompt_sha256,
            "seed_ids": seed_ids,
            "category": str(contract["category"]),
            "level": str(contract["level"]),
            "task_form": str(contract["task_form"]),
            "split_key": str(contract["split_key"]),
            "model_ref": args.model_ref,
            "generation_kind": args.generation_kind,
            "mechanical_review": review,
            "intended_next_stage": "manual_contract_review",
            "notes": [
                "Generated draft contract only.",
                "Not admitted for SFT/GRPO/eval.",
            ],
        }
        items.append(item)
        review_counter[item["mechanical_review"]["decision"]] += 1
        level_counter[item["level"]] += 1
        task_form_counter[item["task_form"]] += 1
        category_counter[item["category"]] += 1

    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_plan.run_id,
        "synthesis_run_hash": run_plan.synthesis_run_hash,
        "producer": {
            "name": "write_generated_contract_index",
            "version": "phase1.v0.1",
            "git_commit": None,
        },
        "source_refs": {
            "synthesis_run": train_relative_ref(args.synthesis_run, args.train_root),
            "request_jsonl": str(args.request_jsonl) if args.request_jsonl else "",
            "contracts_dir": str(args.contracts_dir),
        },
        "summary": {
            "contract_count": len(items),
            "review_decisions": dict(sorted(review_counter.items())),
            "levels": dict(sorted(level_counter.items())),
            "task_forms": dict(sorted(task_form_counter.items())),
            "categories": dict(sorted(category_counter.items())),
            "training_data_admitted": False,
            "verilog_a_generated": False,
            "evas_run": False,
            "spectre_run": False,
        },
        "items": items,
        "fixture_notes": [
            "Generated contract index records draft contracts only.",
            "Manual review and contamination checks are still required before artifact generation.",
        ],
        "generated_contract_index_hash": "",
    }
    payload["generated_contract_index_hash"] = canonical_payload_hash(payload, "generated_contract_index_hash")
    GeneratedContractIndex.model_validate(payload)
    write_yaml_mapping(args.out, payload)
    print(f"PASS write_generated_contract_index contracts={len(items)} out={args.out}")


if __name__ == "__main__":
    main()
