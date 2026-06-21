"""
Prepare artifact-synthesis request JSONL from accepted contract reviews.

This script consumes a generated contract index plus a contract review manifest.
It emits prompt records for contracts marked `accept_for_generation`. It does
not generate Verilog-A artifacts, run EVAS/Spectre, or create training data.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

try:
    from .manifest_schemas import (
        ContractReviewManifest,
        GeneratedContractIndex,
        canonical_payload_hash,
        read_yaml_mapping,
        resolve_train_ref,
        sha256_text,
        write_jsonl,
    )
    from .render_pilot_prompts import DEFAULT_TEMPLATE_DIR, safe_inline, validate_prompt_text
except ImportError:  # pragma: no cover - direct script execution fallback
    from manifest_schemas import (
        ContractReviewManifest,
        GeneratedContractIndex,
        canonical_payload_hash,
        read_yaml_mapping,
        resolve_train_ref,
        sha256_text,
        write_jsonl,
    )
    from render_pilot_prompts import DEFAULT_TEMPLATE_DIR, safe_inline, validate_prompt_text


DEFAULT_TRAIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = Path(tempfile.gettempdir()) / "vaevas_artifact_synthesis_requests"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_review(review_manifest_path: Path, index_path: Path) -> tuple[ContractReviewManifest, GeneratedContractIndex]:
    review_payload = read_yaml_mapping(review_manifest_path)
    review_hash = canonical_payload_hash(review_payload, "contract_review_manifest_hash")
    require(review_payload.get("contract_review_manifest_hash") == review_hash, "contract_review_manifest_hash mismatch")
    review = ContractReviewManifest.model_validate(review_payload)

    index_payload = read_yaml_mapping(index_path)
    index_hash = canonical_payload_hash(index_payload, "generated_contract_index_hash")
    require(index_payload.get("generated_contract_index_hash") == index_hash, "generated_contract_index_hash mismatch")
    index = GeneratedContractIndex.model_validate(index_payload)
    require(review.generated_contract_index_hash == index.generated_contract_index_hash, "review/index hash mismatch")
    return review, index


def render_request(
    template: str,
    item: Any,
    contract: dict[str, Any],
    contract_yaml: str,
    generated_contract_index_hash: str,
) -> dict[str, Any]:
    seed_id = item.seed_ids[0] if item.seed_ids else "unknown_seed"
    prompt = template.format(
        seed_id=seed_id,
        contract_id=contract.get("id"),
        category=contract.get("category"),
        level=contract.get("level"),
        task_form=contract.get("task_form"),
        base_function=contract.get("base_function"),
        forbidden_constructs=safe_inline(contract.get("forbidden_constructs", [])),
        contract_yaml=contract_yaml.rstrip(),
    )
    validate_prompt_text("artifact_proposal", prompt)
    return {
        "request_id": f"{item.generated_contract_id}.artifact_proposal",
        "prompt_kind": "artifact_proposal",
        "generated_contract_id": item.generated_contract_id,
        "contract_ref": item.contract_ref,
        "contract_sha256": item.contract_sha256,
        "seed_ids": item.seed_ids,
        "category": item.category,
        "level": item.level,
        "task_form": item.task_form,
        "model_visible_prompt": prompt,
        "prompt_sha256": sha256_text(prompt),
        "hidden_metadata": {
            "generated_contract_index_hash": generated_contract_index_hash,
            "not_training_data": True,
            "expected_output": "one draft artifact matching the reviewed contract task_form",
        },
    }


def write_summary(out_dir: Path, records: list[dict[str, Any]], request_hash: str, review: ContractReviewManifest, index: GeneratedContractIndex) -> Path:
    summary = {
        "status": "PASS",
        "run_id": index.run_id,
        "accepted_contract_count": len(records),
        "request_count": len(records),
        "request_jsonl_hash": request_hash,
        "contract_review_manifest_hash": review.contract_review_manifest_hash,
        "generated_contract_index_hash": index.generated_contract_index_hash,
        "artifact_generated": False,
        "training_data_generated": False,
        "evas_run": False,
        "spectre_run": False,
    }
    path = out_dir / "artifact_synthesis_request_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-contract-index", type=Path, required=True)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--template-dir", type=Path, default=DEFAULT_TEMPLATE_DIR)
    parser.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    review, index = load_review(args.review_manifest, args.generated_contract_index)
    review_by_id = {item.generated_contract_id: item for item in review.items}
    template = (args.template_dir / "artifact_proposal.md").read_text(encoding="utf-8")
    records: list[dict[str, Any]] = []
    for item in index.items:
        review_item = review_by_id[item.generated_contract_id]
        if review_item.decision != "accept_for_generation":
            continue
        contract_path = resolve_train_ref(item.contract_ref, args.train_root)
        contract_yaml = contract_path.read_text(encoding="utf-8")
        contract = read_yaml_mapping(contract_path)
        records.append(render_request(template, item, contract, contract_yaml, index.generated_contract_index_hash))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    request_path = args.out_dir / "artifact_synthesis_requests.jsonl"
    request_hash = write_jsonl(request_path, records)
    summary_path = write_summary(args.out_dir, records, request_hash, review, index)
    print(
        "PASS prepare_artifact_synthesis_requests "
        f"request_count={len(records)} request_jsonl={request_path} summary={summary_path}"
    )


if __name__ == "__main__":
    main()
