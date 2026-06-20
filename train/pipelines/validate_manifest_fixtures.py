"""
Validate Phase 1 manifest fixtures.

This script is intentionally local-only: it validates the toy YAML fixtures under
train/data/manifests/examples/ without requiring EVAS, Spectre, remote GPUs, or
generated artifacts.

Usage:
    python3 -m train.pipelines.validate_manifest_fixtures
    python3 train/pipelines/validate_manifest_fixtures.py --examples-dir train/data/manifests/examples
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .manifest_schemas import validate_fixture_dir
except ImportError:  # pragma: no cover - direct script execution fallback
    from manifest_schemas import validate_fixture_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Phase 1 manifest fixture YAML files.")
    parser.add_argument(
        "--examples-dir",
        type=Path,
        default=Path("train/data/manifests/examples"),
        help="Directory containing the toy manifest fixtures.",
    )
    args = parser.parse_args()

    bundle = validate_fixture_dir(args.examples_dir)
    candidate_ids = sorted(item.candidate_id for item in bundle.candidate_index.items)
    admitted_item_ids = sorted(item.item_id for item in bundle.admitted_manifest.items)

    print(f"manifest_fixture_validation_ok=1")
    print(f"examples_dir={args.examples_dir}")
    print(f"candidate_count={len(candidate_ids)}")
    print(f"admitted_count={len(admitted_item_ids)}")
    print(f"candidate_ids={','.join(candidate_ids)}")
    print(f"admitted_item_ids={','.join(admitted_item_ids)}")


if __name__ == "__main__":
    main()
