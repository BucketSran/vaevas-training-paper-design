"""
Render and validate Phase 1 pilot prompt-template records.

This script does not call an LLM and does not create training data. It renders
generation/review request records from the clean-room seed catalog, pilot plan,
and prompt templates so the next synthesis step has a reproducible prompt gate.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path
from string import Formatter
from typing import Any

try:
    from .manifest_schemas import (
        PilotBatchPlan,
        SeedCatalog,
        canonical_payload_hash,
        read_yaml_mapping,
        resolve_train_ref,
        sha256_text,
        write_jsonl,
    )
    from .pipeline_common import public_contract_summary
except ImportError:  # pragma: no cover - direct script execution fallback
    from manifest_schemas import (
        PilotBatchPlan,
        SeedCatalog,
        canonical_payload_hash,
        read_yaml_mapping,
        resolve_train_ref,
        sha256_text,
        write_jsonl,
    )
    from pipeline_common import public_contract_summary


DEFAULT_TRAIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED_CATALOG = DEFAULT_TRAIN_ROOT / "data/seeds/seed_catalog.phase1-pilot-0001.yaml"
DEFAULT_PILOT_PLAN = DEFAULT_TRAIN_ROOT / "data/manifests/pilot/batch_plan.synth-batch-pilot-0001.yaml"
DEFAULT_TEMPLATE_DIR = DEFAULT_TRAIN_ROOT / "data/prompts/templates"
DEFAULT_OUT_DIR = Path(tempfile.gettempdir()) / "vaevas_phase1_prompt_gate"

TEMPLATE_FILES = {
    "contract_proposal": "contract_proposal.md",
    "contract_review": "contract_review.md",
    "artifact_proposal": "artifact_proposal.md",
}
REQUIRED_TEMPLATE_FIELDS = {
    "contract_proposal": {
        "seed_id",
        "source_tier",
        "source_kind",
        "source_ref",
        "category",
        "intended_levels",
        "intended_task_forms",
        "allowed_uses",
        "forbidden_uses",
        "contamination_review",
        "source_contract_summary",
        "schema_required_fields",
    },
    "contract_review": {
        "seed_id",
        "source_tier",
        "source_ref",
        "allowed_uses",
        "forbidden_uses",
        "contract_yaml",
    },
    "artifact_proposal": {
        "seed_id",
        "contract_id",
        "category",
        "level",
        "task_form",
        "base_function",
        "forbidden_constructs",
        "contract_yaml",
    },
}
FORBIDDEN_PROMPT_FRAGMENTS = (
    "benchmark-vabench-release-v1",
    "score_denominator_manifest",
    "gold_completion",
    "target_completion",
    "reference_answer",
)
REQUIRED_PROMPT_KINDS = {"contract_proposal", "contract_review", "artifact_proposal"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_yaml_text(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    payload = read_yaml_mapping(path)
    return payload, text


def load_inputs(seed_catalog_path: Path, pilot_plan_path: Path) -> tuple[SeedCatalog, PilotBatchPlan]:
    catalog = SeedCatalog.model_validate(read_yaml_mapping(seed_catalog_path))
    plan = PilotBatchPlan.model_validate(read_yaml_mapping(pilot_plan_path))
    expected_catalog_hash = canonical_payload_hash(read_yaml_mapping(seed_catalog_path), "seed_catalog_hash")
    require(catalog.seed_catalog_hash == expected_catalog_hash, "seed catalog hash mismatch")
    require(plan.seed_catalog_hash == catalog.seed_catalog_hash, "pilot plan seed_catalog_hash mismatch")
    expected_plan_hash = canonical_payload_hash(read_yaml_mapping(pilot_plan_path), "pilot_plan_hash")
    require(plan.pilot_plan_hash == expected_plan_hash, "pilot plan hash mismatch")
    return catalog, plan


def template_fields(template: str) -> set[str]:
    fields: set[str] = set()
    for _literal_text, field_name, _format_spec, _conversion in Formatter().parse(template):
        if field_name:
            fields.add(field_name)
    return fields


def load_templates(template_dir: Path) -> dict[str, str]:
    templates: dict[str, str] = {}
    for kind, filename in TEMPLATE_FILES.items():
        path = template_dir / filename
        require(path.exists(), f"missing prompt template: {path}")
        text = path.read_text(encoding="utf-8")
        fields = template_fields(text)
        missing = REQUIRED_TEMPLATE_FIELDS[kind] - fields
        require(not missing, f"{filename} missing required placeholders: {sorted(missing)}")
        templates[kind] = text
    return templates


def safe_inline(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def validate_prompt_text(prompt_kind: str, text: str) -> None:
    lowered = text.lower()
    blocked = [fragment for fragment in FORBIDDEN_PROMPT_FRAGMENTS if fragment in lowered]
    require(not blocked, f"{prompt_kind} prompt contains forbidden fragments: {blocked}")
    require("gold_completion" not in lowered, f"{prompt_kind} prompt leaks gold_completion")
    require("<answer>" not in lowered, f"{prompt_kind} prompt contains answer tag")


def schema_required_fields(train_root: Path) -> str:
    schema = read_yaml_mapping(train_root / "data/contracts/schema.yaml")
    fields = schema.get("required_fields")
    require(isinstance(fields, list), "contract schema required_fields must be a list")
    return "\n".join(f"- {field}" for field in fields)


def record_id(seed_id: str, prompt_kind: str) -> str:
    return f"{seed_id}.{prompt_kind}"


def render_records(
    catalog: SeedCatalog,
    plan: PilotBatchPlan,
    templates: dict[str, str],
    train_root: Path,
) -> list[dict[str, Any]]:
    catalog_by_id = {item.seed_id: item for item in catalog.items}
    required_fields_text = schema_required_fields(train_root)
    records: list[dict[str, Any]] = []

    for seed_id in plan.planned_seed_ids:
        item = catalog_by_id[seed_id]
        contract_path = resolve_train_ref(item.source_ref, train_root)
        contract, contract_yaml = load_yaml_text(contract_path)
        summary = public_contract_summary(contract)
        common = {
            "seed_id": item.seed_id,
            "source_tier": item.source_tier,
            "source_kind": item.source_kind,
            "source_ref": item.source_ref,
            "category": item.category,
            "intended_levels": safe_inline(item.intended_levels),
            "intended_task_forms": safe_inline(item.intended_task_forms),
            "allowed_uses": safe_inline(item.allowed_uses),
            "forbidden_uses": safe_inline(item.forbidden_uses),
            "contamination_review": safe_inline(item.contamination_review),
            "source_contract_summary": summary,
            "schema_required_fields": required_fields_text,
            "contract_yaml": contract_yaml.rstrip(),
            "contract_id": str(contract.get("id")),
            "level": str(contract.get("level")),
            "task_form": str(contract.get("task_form")),
            "base_function": str(contract.get("base_function")),
            "forbidden_constructs": safe_inline(contract.get("forbidden_constructs", [])),
        }

        for prompt_kind in ("contract_proposal", "contract_review", "artifact_proposal"):
            text = templates[prompt_kind].format(**common)
            validate_prompt_text(prompt_kind, text)
            records.append(
                {
                    "prompt_id": record_id(seed_id, prompt_kind),
                    "prompt_kind": prompt_kind,
                    "seed_id": seed_id,
                    "source_ref": item.source_ref,
                    "contract_id": str(contract.get("id")),
                    "category": str(contract.get("category")),
                    "level": str(contract.get("level")),
                    "task_form": str(contract.get("task_form")),
                    "requires_manual_review": item.status == "seed_candidate",
                    "model_visible_prompt": text,
                    "prompt_sha256": sha256_text(text),
                    "hidden_metadata": {
                        "seed_catalog_hash": catalog.seed_catalog_hash,
                        "pilot_plan_hash": plan.pilot_plan_hash,
                        "seed_status": item.status,
                        "contract_ref": item.source_ref,
                    },
                }
            )
    return records


def validate_records(records: list[dict[str, Any]], plan: PilotBatchPlan) -> None:
    expected_count = len(plan.planned_seed_ids) * len(REQUIRED_PROMPT_KINDS)
    require(len(records) == expected_count, f"expected {expected_count} prompt records, got {len(records)}")
    ids = [str(record["prompt_id"]) for record in records]
    require(len(ids) == len(set(ids)), "prompt_id values must be unique")
    kinds = {str(record["prompt_kind"]) for record in records}
    require(kinds == REQUIRED_PROMPT_KINDS, f"unexpected prompt kinds: {sorted(kinds)}")
    for record in records:
        prompt = str(record["model_visible_prompt"])
        validate_prompt_text(str(record["prompt_kind"]), prompt)
        require(str(record["prompt_sha256"]) == sha256_text(prompt), f"prompt hash mismatch: {record['prompt_id']}")
        hidden = record.get("hidden_metadata")
        require(isinstance(hidden, dict), f"hidden_metadata missing for {record['prompt_id']}")
        require("model_visible_prompt" not in hidden, f"hidden_metadata nests prompt text for {record['prompt_id']}")


def write_summary(out_dir: Path, records: list[dict[str, Any]], prompt_hash: str, catalog: SeedCatalog, plan: PilotBatchPlan) -> Path:
    counter = Counter(str(record["prompt_kind"]) for record in records)
    summary = {
        "status": "PASS",
        "prompt_count": len(records),
        "prompt_kinds": dict(sorted(counter.items())),
        "prompt_jsonl_hash": prompt_hash,
        "seed_catalog_hash": catalog.seed_catalog_hash,
        "pilot_plan_hash": plan.pilot_plan_hash,
        "batch_id": plan.batch_id,
        "catalog_id": catalog.catalog_id,
        "llm_api_called": False,
        "training_data_generated": False,
    }
    path = out_dir / "prompt_plan_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-catalog", type=Path, default=DEFAULT_SEED_CATALOG)
    parser.add_argument("--pilot-plan", type=Path, default=DEFAULT_PILOT_PLAN)
    parser.add_argument("--template-dir", type=Path, default=DEFAULT_TEMPLATE_DIR)
    parser.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    catalog, plan = load_inputs(args.seed_catalog, args.pilot_plan)
    templates = load_templates(args.template_dir)
    records = render_records(catalog, plan, templates, args.train_root)
    validate_records(records, plan)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = args.out_dir / "pilot_generation_prompts.jsonl"
    prompt_hash = write_jsonl(prompt_path, records)
    summary_path = write_summary(args.out_dir, records, prompt_hash, catalog, plan)

    print(
        "PASS render_pilot_prompts "
        f"prompt_count={len(records)} "
        f"prompt_jsonl={prompt_path} "
        f"summary={summary_path}"
    )


if __name__ == "__main__":
    main()
