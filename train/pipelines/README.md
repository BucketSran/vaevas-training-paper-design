# pipelines/

Data preparation and verification pipelines. Each script is **one self-contained tool**, runnable from the CLI with `python -m train.pipelines.<name>`.

## Modules (to be implemented in Phase 1)

| File | Role | Inputs | Outputs |
|---|---|---|---|
| `synthesize.py` | LLM-based generation of `<spec, va, tb>` candidates from seed prompts | seed catalog | `data/synthesized/<batch>/` |
| `verify_evas.py` | Compile + simulate each candidate via EVAS; label pass/fail; capture logs | `data/synthesized/<batch>/` | `data/verified/<batch>/` + meta yaml |
| `build_trajectory.py` | Given verified `<spec, va, tb>`, decompose into step-by-step CoT trajectory | `data/verified/<batch>/` | adds `trajectory.json` per entry |
| `check_contamination.py` | Hash-match against vaBench release prompts and gold code | training dir + vaBench release dir | exit 0 if clean, 1 + diff if overlap |
| `pack_sft.py` | Concatenate verified items into `train.jsonl`/`val.jsonl` in SFT format | `data/verified/` | `data/sft/{train,val}.jsonl` |
| `pack_rl.py` | Extract RL prompts and reference answers for reflective learning | `data/verified/` | `data/rl/{prompts,ref_answers}.jsonl` |

## Design rules

1. **Idempotency**: re-running with the same input must not duplicate outputs. Use stable IDs.
2. **Provenance**: every output file carries a `meta.yaml` with source + script version + timestamp.
3. **Streaming**: large datasets must stream, not load in memory. Use `.jsonl`, not `.json`.
4. **Verifier honesty**: `verify_evas.py` MUST report EVAS errors verbatim, never re-interpret. The verifier is ground truth.
5. **Contamination check** is a pre-commit hook for any branch that adds to `data/verified/`.

## CLI contract (proposed)

```
# Phase 1 sketch — to be implemented
python -m train.pipelines.synthesize  --seeds <path>  --n 100  --model claude-opus
python -m train.pipelines.verify_evas --batch <id>
python -m train.pipelines.build_trajectory --batch <id>
python -m train.pipelines.check_contamination --train-dir data/verified --release-dir ../behavioral-veriloga-eval/benchmark-vabench-release-v1
python -m train.pipelines.pack_sft   --in data/verified --out data/sft   --val-frac 0.1
python -m train.pipelines.pack_rl    --in data/verified --out data/rl
```

## Dependencies (Phase 1)

- `evas-sim` (already in vaEVAS workspace)
- `openvaf` binary (already used by EVAS)
- `anthropic` / `openai` SDKs for synthesis LLMs
- `pyyaml`, `pydantic` for metadata schemas
