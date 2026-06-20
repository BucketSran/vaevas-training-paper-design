"""
Build a Phase 1 protected-index manifest from a read-only asset tree.

This is a local metadata tool. It fingerprints files and writes a manifest; it
does not modify protected assets.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from .manifest_schemas import (
        EMPTY_SHA256,
        SCHEMA_VERSION,
        Producer,
        ProtectedIndex,
        canonical_payload_hash,
        normalized_sha256_file,
        read_yaml_mapping,
        sha256_file,
        sha256_text,
        slugify,
        train_relative_ref,
        write_yaml_mapping,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from manifest_schemas import (
        EMPTY_SHA256,
        SCHEMA_VERSION,
        Producer,
        ProtectedIndex,
        canonical_payload_hash,
        normalized_sha256_file,
        read_yaml_mapping,
        sha256_file,
        sha256_text,
        slugify,
        train_relative_ref,
        write_yaml_mapping,
    )

TEXT_SUFFIXES = {".md", ".txt", ".yaml", ".yml", ".json", ".va", ".vams", ".scs", ".py"}


def guess_source_kind(path: Path) -> tuple[str, str]:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if suffix in {".va", ".vams"}:
        return "gold_va", "veriloga"
    if suffix == ".scs":
        return "testbench", "spectre_testbench"
    if "checker" in name:
        return "checker", "checker"
    if suffix in {".md", ".txt"}:
        return "release_prompt", "prompt"
    if suffix in {".yaml", ".yml", ".json"}:
        return "metadata", "metadata"
    return "report", "file"


def extract_task_id(path: Path) -> str | None:
    if path.suffix.lower() not in {".yaml", ".yml"}:
        return None
    try:
        data = read_yaml_mapping(path)
    except Exception:
        return None
    task_id = data.get("id") or data.get("task_id")
    if isinstance(task_id, str):
        return task_id
    return None


def extract_module_signature_hash(path: Path) -> str | None:
    if path.suffix.lower() not in {".va", ".vams"}:
        return None
    text = path.read_bytes().decode("utf-8", errors="replace")
    signatures: list[str] = []
    for match in re.finditer(r"\bmodule\s+([A-Za-z_][A-Za-z0-9_$]*)\s*\((.*?)\)\s*;", text, re.DOTALL):
        module_name = match.group(1)
        ports = [port.strip() for port in match.group(2).replace("\n", " ").split(",") if port.strip()]
        signatures.append(f"{module_name}({','.join(sorted(ports))})")
    if not signatures:
        return None
    return sha256_text("\n".join(sorted(signatures)))


def extract_property_signature_hash(path: Path) -> str | None:
    if path.suffix.lower() not in {".yaml", ".yml"}:
        return None
    try:
        data = read_yaml_mapping(path)
    except Exception:
        return None
    properties = data.get("properties")
    if not isinstance(properties, list):
        return None
    signatures: list[str] = []
    for prop in properties:
        if isinstance(prop, dict):
            signatures.append(f"{prop.get('id')}:{prop.get('type')}:{prop.get('tolerance_ref')}")
    if not signatures:
        return None
    return sha256_text("\n".join(sorted(signatures)))


def iter_asset_files(root: Path) -> list[Path]:
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES and ".git" not in path.parts
    ]
    return sorted(files)


def build_protected_index_payload(
    root: Path,
    benchmark_release_id: str,
    source_manifest_refs: list[str],
    counted_in_score_default: bool,
    train_root: Path,
) -> dict[str, Any]:
    files = iter_asset_files(root)
    if not files:
        raise ValueError(f"no indexable text assets found under {root}")

    items: list[dict[str, Any]] = []
    for path in files:
        relative_id = slugify(str(path.relative_to(root)))
        source_kind, asset_role = guess_source_kind(path)
        items.append(
            {
                "protected_id": f"prot_{relative_id}",
                "source_kind": source_kind,
                "source_path": train_relative_ref(path, train_root),
                "task_id": extract_task_id(path),
                "counted_in_score": counted_in_score_default,
                "asset_role": asset_role,
                "raw_sha256": sha256_file(path),
                "normalized_sha256": normalized_sha256_file(path),
                "ngram_minhash": None,
                "module_signature_hash": extract_module_signature_hash(path),
                "property_signature_hash": extract_property_signature_hash(path),
            }
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "protected_root_ref": train_relative_ref(root, train_root),
        "benchmark_release_id": benchmark_release_id,
        "source_manifest_refs": source_manifest_refs,
        "indexed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "producer": Producer(name="train.pipelines.build_protected_index", version="phase1-local-v0").model_dump(),
        "protected_index_hash": EMPTY_SHA256,
        "items": items,
    }
    hash_payload = dict(payload)
    hash_payload.pop("indexed_at", None)
    payload["protected_index_hash"] = canonical_payload_hash(hash_payload, "protected_index_hash")
    ProtectedIndex.model_validate(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a protected-index manifest from a read-only asset tree.")
    parser.add_argument("--root", type=Path, required=True, help="Read-only asset tree to fingerprint.")
    parser.add_argument("--out", type=Path, required=True, help="Output protected_index YAML path.")
    parser.add_argument("--benchmark-release-id", required=True, help="Release or source ID for the protected tree.")
    parser.add_argument("--source-manifest-ref", action="append", default=[], help="Source manifest ref; repeatable.")
    parser.add_argument("--counted-in-score-default", action="store_true", help="Mark indexed items as counted in score.")
    parser.add_argument("--train-root", type=Path, default=Path("train"), help="Path root for train-relative refs.")
    args = parser.parse_args()

    payload = build_protected_index_payload(
        root=args.root,
        benchmark_release_id=args.benchmark_release_id,
        source_manifest_refs=args.source_manifest_ref,
        counted_in_score_default=args.counted_in_score_default,
        train_root=args.train_root,
    )
    write_yaml_mapping(args.out, payload)
    print("protected_index_ok=1")
    print(f"out={args.out}")
    print(f"item_count={len(payload['items'])}")
    print(f"protected_index_hash={payload['protected_index_hash']}")


if __name__ == "__main__":
    main()
