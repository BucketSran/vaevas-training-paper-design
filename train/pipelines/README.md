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
| `render_pilot_prompts.py` | Render contract proposal/review/artifact prompt records from the pilot plan without calling an LLM | seed catalog + pilot plan + templates | scratch JSONL + summary under temp output dir |
| `validate_synthesis_run.py` | Validate a synthesis run plan before any LLM generation | synthesis run YAML | exit 0 if hashes, refs, prompt selection, and handoff policy are coherent |
| `prepare_contract_synthesis_requests.py` | Write contract-proposal request JSONL for a synthesis run without calling an LLM | synthesis run YAML | scratch request JSONL + summary |
| `contract_review_gate.py` | Shared mechanical contract review checks | contract YAML + schema | review decision, blockers, warnings |
| `write_generated_contract_index.py` | Validate generated draft contracts and write an index | contracts dir + request JSONL + synthesis run | generated contract index YAML |
| `validate_generated_contracts.py` | Revalidate a generated contract index and referenced contracts | generated contract index | exit 0 if contract refs, hashes, and mechanical review are coherent |
| `write_contract_review_manifest.py` | Write a structured review manifest for generated contracts | generated contract index | review manifest + Markdown report |
| `validate_contract_review_manifest.py` | Validate review decisions and next-stage permissions | review manifest + generated contract index | exit 0 if decisions are internally coherent |
| `prepare_artifact_synthesis_requests.py` | Write artifact-generation request JSONL for accepted contracts | generated contract index + review manifest | scratch/request-dir JSONL + summary |
| `write_artifact_candidate_index.py` | Hash-track generated Verilog-A artifacts as candidate items | generated contract index + review manifest + artifacts | artifact candidate index YAML |
| `validate_artifact_candidate_index.py` | Revalidate artifact candidate refs and hashes | artifact candidate index | exit 0 if artifact refs and hashes are coherent |
| `pack_draft_training.py` | Pack draft unadmitted SFT/GRPO JSONL from artifact candidates | artifact candidate index | draft SFT JSONL + draft GRPO prompt JSONL + manifest |
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
python -m train.pipelines.render_pilot_prompts --out-dir /tmp/vaevas_phase1_prompt_gate
python -m train.pipelines.validate_synthesis_run
python -m train.pipelines.prepare_contract_synthesis_requests --out-dir /tmp/vaevas_contract_synthesis_requests
python -m train.pipelines.write_generated_contract_index --contracts-dir <contracts-dir> --request-jsonl <request-jsonl> --out <generated_contract_index.yaml> --expect-count 5
python -m train.pipelines.validate_generated_contracts --index <generated_contract_index.yaml> --synthesis-run <synthesis_run.yaml> --expect-count 5
python -m train.pipelines.write_contract_review_manifest --index <generated_contract_index.yaml> --out <review_manifest.yaml> --report-out <review_report.md>
python -m train.pipelines.validate_contract_review_manifest --review-manifest <review_manifest.yaml> --generated-contract-index <generated_contract_index.yaml>
python -m train.pipelines.prepare_artifact_synthesis_requests --generated-contract-index <generated_contract_index.yaml> --review-manifest <review_manifest.yaml> --out-dir <artifact-request-dir>
python -m train.pipelines.write_artifact_candidate_index --generated-contract-index <generated_contract_index.yaml> --review-manifest <review_manifest.yaml> --artifact-root <artifacts-dir> --out <artifact_candidate_index.yaml>
python -m train.pipelines.validate_artifact_candidate_index --candidate-index <artifact_candidate_index.yaml>
python -m train.pipelines.pack_draft_training --candidate-index <artifact_candidate_index.yaml> --out-dir <draft-training-dir> --manifest-out <draft_training_pack_manifest.yaml> --run-id <run-id>
```

Toy closed-loop smoke commands are documented in
`../docs/PHASE1_CLOSED_LOOP_PIPELINES.md`.

## Dependencies (Phase 1)

- EVAS Rust evaluator (already in vaEVAS workspace)
- `anthropic` / `openai` SDKs for synthesis LLMs
- `pyyaml`, `pydantic` for metadata schemas
