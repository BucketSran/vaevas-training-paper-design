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
    from .manifest_schemas import sha256_text, write_jsonl
    from .render_pilot_prompts import load_inputs, load_templates, render_records
    from .validate_synthesis_run import DEFAULT_RUN_PLAN, DEFAULT_TRAIN_ROOT, load_run_plan, validate_run_plan
except ImportError:  # pragma: no cover - direct script execution fallback
    from manifest_schemas import sha256_text, write_jsonl
    from render_pilot_prompts import load_inputs, load_templates, render_records
    from validate_synthesis_run import DEFAULT_RUN_PLAN, DEFAULT_TRAIN_ROOT, load_run_plan, validate_run_plan


DEFAULT_OUT_DIR = Path(tempfile.gettempdir()) / "vaevas_contract_synthesis_requests"
DEFAULT_VARIATION_FOCI = [
    "boundary conditions and tolerance stress",
    "alternative stimulus waveform coverage",
    "parameter sweep diversity",
    "checker-observable clarity",
    "OOD-style naming and split-key diversity",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_summary(out_dir: Path, requests: list[dict], request_hash: str, run_id: str, run_hash: str) -> Path:
    summary = {
        "status": "PASS",
        "run_id": run_id,
        "synthesis_run_hash": run_hash,
        "request_count": len(requests),
        "seed_count": len({request["seed_id"] for request in requests}),
        "variant_count": len({request["variant_id"] for request in requests}),
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
    variants_per_seed = int(run_plan.prompt_selection.get("variants_per_seed", 1))
    variation_foci = list(run_plan.prompt_selection.get("variation_foci", DEFAULT_VARIATION_FOCI))
    if not variation_foci:
        variation_foci = DEFAULT_VARIATION_FOCI
    requests: list[dict] = []
    for record in prompt_records:
        if record["prompt_kind"] not in selected_kinds:
            continue
        for variant_index in range(1, variants_per_seed + 1):
            variant_id = f"v{variant_index:02d}"
            variation_focus = variation_foci[(variant_index - 1) % len(variation_foci)]
            variant_prompt = (
                f"{record['model_visible_prompt'].rstrip()}\n\n"
                "## Variant Requirements\n\n"
                f"- variant_id: `{variant_id}`\n"
                f"- variation_focus: {variation_focus}\n"
                "- Create a distinct contract, not only a rename of the seed contract.\n"
                "- Use a new `id` and a new `split_key` that includes the variant focus.\n"
                "- Preserve the requested category, level, task_form, and clean-room provenance.\n"
            )
            requests.append(
                {
                    "request_id": f"{record['prompt_id']}.{variant_id}",
                    "prompt_id": record["prompt_id"],
                    "prompt_kind": record["prompt_kind"],
                    "seed_id": record["seed_id"],
                    "variant_id": variant_id,
                    "variation_focus": variation_focus,
                    "contract_id_hint": record["contract_id"],
                    "category": record["category"],
                    "level": record["level"],
                    "task_form": record["task_form"],
                    "model_visible_prompt": variant_prompt,
                    "prompt_sha256": sha256_text(variant_prompt),
                    "hidden_metadata": {
                        **record["hidden_metadata"],
                        "synthesis_run_hash": run_plan.synthesis_run_hash,
                        "expected_output": "one draft contract YAML matching train/data/contracts/schema.yaml",
                        "not_training_data": True,
                        "variant_id": variant_id,
                        "variation_focus": variation_focus,
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
