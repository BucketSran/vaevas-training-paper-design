"""
Prepare contract-synthesis request JSONL from the Phase 1 synthesis run plan.

This script renders pilot prompts, filters the contract-proposal prompts selected
by the run plan, and writes scratch request records. It does not call an LLM and
does not generate contracts or training data.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

try:
    from .manifest_schemas import write_jsonl
    from .render_pilot_prompts import load_inputs, load_templates, render_records
    from .validate_synthesis_run import DEFAULT_RUN_PLAN, DEFAULT_TRAIN_ROOT, load_run_plan, validate_run_plan
except ImportError:  # pragma: no cover - direct script execution fallback
    from manifest_schemas import write_jsonl
    from render_pilot_prompts import load_inputs, load_templates, render_records
    from validate_synthesis_run import DEFAULT_RUN_PLAN, DEFAULT_TRAIN_ROOT, load_run_plan, validate_run_plan


DEFAULT_OUT_DIR = Path(tempfile.gettempdir()) / "vaevas_contract_synthesis_requests"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_summary(out_dir: Path, requests: list[dict], request_hash: str, run_id: str, run_hash: str) -> Path:
    summary = {
        "status": "PASS",
        "run_id": run_id,
        "synthesis_run_hash": run_hash,
        "request_count": len(requests),
        "prompt_kinds": sorted({request["prompt_kind"] for request in requests}),
        "request_jsonl_hash": request_hash,
        "llm_api_called": False,
        "contracts_generated": False,
        "training_data_generated": False,
    }
    path = out_dir / "contract_synthesis_request_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-plan", type=Path, default=DEFAULT_RUN_PLAN)
    parser.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    run_plan = load_run_plan(args.run_plan)
    validate_run_plan(run_plan, args.train_root)

    seed_catalog_path = args.train_root / run_plan.seed_catalog_ref
    pilot_plan_path = args.train_root / run_plan.pilot_plan_ref
    template_dir = args.train_root / run_plan.prompt_template_dir
    catalog, pilot_plan = load_inputs(seed_catalog_path, pilot_plan_path)
    templates = load_templates(template_dir)
    prompt_records = render_records(catalog, pilot_plan, templates, args.train_root)

    selected_kinds = set(run_plan.prompt_selection["include_prompt_kinds"])
    requests: list[dict] = []
    for record in prompt_records:
        if record["prompt_kind"] not in selected_kinds:
            continue
        requests.append(
            {
                "request_id": record["prompt_id"],
                "prompt_id": record["prompt_id"],
                "prompt_kind": record["prompt_kind"],
                "seed_id": record["seed_id"],
                "contract_id_hint": record["contract_id"],
                "category": record["category"],
                "level": record["level"],
                "task_form": record["task_form"],
                "model_visible_prompt": record["model_visible_prompt"],
                "prompt_sha256": record["prompt_sha256"],
                "hidden_metadata": {
                    **record["hidden_metadata"],
                    "synthesis_run_hash": run_plan.synthesis_run_hash,
                    "expected_output": "one draft contract YAML matching train/data/contracts/schema.yaml",
                    "not_training_data": True,
                },
            }
        )

    max_requests = int(run_plan.prompt_selection["max_requests"])
    require(len(requests) == max_requests, f"expected {max_requests} requests, got {len(requests)}")
    require({request["prompt_kind"] for request in requests} == {"contract_proposal"}, "only contract_proposal requests are allowed")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    request_path = args.out_dir / "contract_synthesis_requests.jsonl"
    request_hash = write_jsonl(request_path, requests)
    summary_path = write_summary(args.out_dir, requests, request_hash, run_plan.run_id, run_plan.synthesis_run_hash)
    print(
        "PASS prepare_contract_synthesis_requests "
        f"request_count={len(requests)} request_jsonl={request_path} summary={summary_path}"
    )


if __name__ == "__main__":
    main()

