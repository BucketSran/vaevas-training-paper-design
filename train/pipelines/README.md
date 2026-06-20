# pipelines/

Data preparation and verification pipelines. Each script is **one self-contained tool**, runnable from the CLI with `python -m train.pipelines.<name>`.

Phase 1 implementations must emit and consume the manifests defined in
`../docs/PHASE1_MANIFEST_SCHEMAS.md`. Training packers must read admitted
manifests, not raw generated candidates.

## Modules

| File | Role | Inputs | Outputs |
|---|---|---|---|
| `synthesize.py` | Future LLM-based generation of contract-conditioned candidates from safe seeds | seed catalog + contract manifest | `data/synthesized/<batch>/` + candidate index |
| `verify_evas.py` | Future EVAS Rust compile/elaboration + simulation for each candidate | candidate index | EVAS verification report |
| `build_trajectory.py` | Future trajectory builder for verified `<spec, va, tb>` items | `data/verified/<batch>/` | adds `trajectory.json` per entry |
| `build_protected_index.py` | Build protected material fingerprints from a read-only asset tree | protected asset root | protected index |
| `check_contamination.py` | Compare candidate index against protected index using hash, normalized, signature, and split-leakage checks | protected index + candidate index | contamination report |
| `write_admitted_manifest.py` | Apply admission gates across static, EVAS, contamination, Spectre, diversity, and split reports | evidence reports | admitted manifest |
| `pack_sft.py` | Concatenate admitted SFT-eligible items into `train.jsonl`/`val.jsonl` in SFT format | admitted manifest + candidate index | SFT JSONL + SFT pack manifest |
| `pack_grpo.py` | Extract GRPO prompts and reward runtime metadata without gold completions | admitted manifest + candidate index | GRPO prompt JSONL + GRPO prompt manifest |
| `validate_manifest_fixtures.py` | Validate toy Phase 1 manifest fixtures and cross-manifest references | `data/manifests/examples/` | exit 0 if schemas and references are coherent |
| `validate_pilot_plan.py` | Validate the clean-room seed catalog and pilot batch plan before generation | seed catalog + pilot batch plan | exit 0 if hashes, provenance, coverage, and gates are coherent |
| `manifest_schemas.py` | Shared Pydantic models and hash/IO helpers | YAML/jsonl payloads | validated manifest objects |
| `pipeline_common.py` | Shared contract-summary and packer helpers | contract YAML + artifacts | model-visible prompt/record text |

## Design rules

1. **Idempotency**: re-running with the same input must not duplicate outputs. Use stable IDs.
2. **Provenance**: every output batch carries a manifest or `meta.yaml` with source + script version + hash evidence.
3. **Streaming**: large datasets must stream, not load in memory. Use `.jsonl`, not `.json`.
4. **Verifier honesty**: `verify_evas.py` MUST report EVAS errors verbatim, never re-interpret. EVAS is the fast reference for Phase 1 scale.
5. **Spectre boundary**: Spectre audit reports decide claim eligibility for audited slices; no EVAS PASS / Spectre FAIL item may enter a claimed slice.
6. **Contamination check** is a pre-commit hook for any branch that adds admitted or packed data.

## CLI contract

```
python -m train.pipelines.build_protected_index --root <protected-root> --out <protected-index.yaml> --benchmark-release-id <id>
python -m train.pipelines.synthesize  --seeds <path> --contracts <path> --n 100 --model <llm>
python -m train.pipelines.verify_evas --candidate-index <path>
python -m train.pipelines.build_trajectory --batch <id>
python -m train.pipelines.check_contamination --protected-index <path> --candidate-index <path> --out <report.yaml>
python -m train.pipelines.write_admitted_manifest --candidate-index <path> --static-check-report <path> --evas-report <path> --contamination-report <path> --diversity-report <path> --split-manifest <path> --out <admitted.yaml>
python -m train.pipelines.pack_sft --admitted-manifest <path> --candidate-index <path> --out-dir <dir> --manifest-out <manifest.yaml>
python -m train.pipelines.pack_grpo --admitted-manifest <path> --candidate-index <path> --out-dir <dir> --manifest-out <manifest.yaml>
python -m train.pipelines.validate_manifest_fixtures
python -m train.pipelines.validate_pilot_plan
```

Toy closed-loop smoke commands are documented in
`../docs/PHASE1_CLOSED_LOOP_PIPELINES.md`.

## Dependencies (Phase 1)

- EVAS Rust evaluator (already in vaEVAS workspace)
- `anthropic` / `openai` SDKs for synthesis LLMs
- `pyyaml`, `pydantic` for metadata schemas
