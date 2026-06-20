# pipelines/

Data preparation and verification pipelines. Each script is **one self-contained tool**, runnable from the CLI with `python -m train.pipelines.<name>`.

Phase 1 implementations must emit and consume the manifests defined in
`../docs/PHASE1_MANIFEST_SCHEMAS.md`. Training packers must read admitted
manifests, not raw generated candidates.

## Modules (to be implemented in Phase 1)

| File | Role | Inputs | Outputs |
|---|---|---|---|
| `synthesize.py` | LLM-based generation of contract-conditioned candidates from safe seeds | seed catalog + contract manifest | `data/synthesized/<batch>/` + candidate index |
| `verify_evas.py` | EVAS Rust compile/elaboration + simulation for each candidate; label pass/fail; capture logs | candidate index | EVAS verification report |
| `build_trajectory.py` | Given verified `<spec, va, tb>`, decompose into step-by-step CoT trajectory | `data/verified/<batch>/` | adds `trajectory.json` per entry |
| `check_contamination.py` | Compare candidate index against protected index using hash, normalized, signature, and split-leakage checks | protected index + candidate index | contamination report |
| `pack_sft.py` | Concatenate admitted SFT-eligible items into `train.jsonl`/`val.jsonl` in SFT format | admitted manifest | `data/sft/{train,val}.jsonl` + SFT pack manifest |
| `pack_grpo.py` | Extract GRPO prompts and reward runtime metadata without gold completions | admitted manifest | `data/rl/prompts.*` + GRPO prompt manifest |
| `validate_manifest_fixtures.py` | Validate toy Phase 1 manifest fixtures and cross-manifest references | `data/manifests/examples/` | exit 0 if schemas and references are coherent |

## Design rules

1. **Idempotency**: re-running with the same input must not duplicate outputs. Use stable IDs.
2. **Provenance**: every output file carries a `meta.yaml` with source + script version + timestamp.
3. **Streaming**: large datasets must stream, not load in memory. Use `.jsonl`, not `.json`.
4. **Verifier honesty**: `verify_evas.py` MUST report EVAS errors verbatim, never re-interpret. EVAS is the fast reference for Phase 1 scale.
5. **Spectre boundary**: Spectre audit reports decide claim eligibility for audited slices; no EVAS PASS / Spectre FAIL item may enter a claimed slice.
6. **Contamination check** is a pre-commit hook for any branch that adds admitted or packed data.

## CLI contract (proposed)

```
python -m train.pipelines.build_protected_index --release-dir ../behavioral-veriloga-eval/benchmark-vabench-release-v1
python -m train.pipelines.synthesize  --seeds <path> --contracts <path> --n 100 --model <llm>
python -m train.pipelines.verify_evas --candidate-index <path>
python -m train.pipelines.build_trajectory --batch <id>
python -m train.pipelines.check_contamination --protected-index <path> --candidate-index <path>
python -m train.pipelines.pack_sft   --admitted-manifest <path> --out data/sft
python -m train.pipelines.pack_grpo  --admitted-manifest <path> --out data/rl
python -m train.pipelines.validate_manifest_fixtures
```

## Dependencies (Phase 1)

- EVAS Rust evaluator (already in vaEVAS workspace)
- `anthropic` / `openai` SDKs for synthesis LLMs
- `pyyaml`, `pydantic` for metadata schemas
