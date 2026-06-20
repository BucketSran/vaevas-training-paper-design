# Phase 1 Closed-Loop Pipeline Map

Status: local implementation guide, 2026-06-20.

This document connects the current Phase 1 scripts into one minimal data loop.
The loop is still toy-scale, but it is now executable end to end without server
access, EVAS, Spectre, or model training.

## Why This Step Matters

The training paper needs more than generated examples. It needs a defensible
data-control chain:

```text
protected assets
  -> protected index
  -> candidate index
  -> static / EVAS / contamination / Spectre shadow / diversity / split reports
  -> admitted manifest
  -> SFT JSONL
  -> GRPO prompt JSONL
```

The current work turns that chain from a document into local code. The scripts
do not yet prove simulator correctness; they prove that downstream SFT/GRPO
packing can only consume manifest-admitted, hash-bound items.

## Implemented Tools

| Step | Tool | Input | Output | Role |
| --- | --- | --- | --- | --- |
| 1 | `manifest_schemas.py` | YAML payloads | Pydantic objects | Shared type contract across scripts. |
| 2 | `build_protected_index.py` | Read-only asset tree | `protected_index.yaml` | Fingerprints protected material that candidates must not overlap. |
| 3 | `check_contamination.py` | protected index + candidate index | `contamination_report.yaml` | Rejects exact protected overlap and quarantines unsafe provenance. |
| 4 | `write_admitted_manifest.py` | candidate index + evidence reports | `admitted_manifest.yaml` | Applies gates and creates the only packer-consumable item list. |
| 5 | `pack_sft.py` | admitted manifest + candidate index | `train.jsonl`, `val.jsonl`, SFT pack manifest | Creates supervised examples with gold output. |
| 6 | `pack_grpo.py` | admitted manifest + candidate index | `prompts.jsonl`, GRPO prompt manifest | Creates reward-training prompts without gold output. |
| 7 | `validate_manifest_fixtures.py` | toy manifest directory | pass/fail summary | Regression check for the fixture chain. |

## Concrete Toy Flow

The fixture candidate is:

```text
cand_toy_l1_hysteresis_comparator_dut_0001
```

It now has real fixture artifact files:

```text
data/manifests/examples/artifacts/cand_toy_l1_hysteresis_comparator_dut_0001/dut.va
data/manifests/examples/artifacts/cand_toy_l1_hysteresis_comparator_dut_0001/tb.scs
data/manifests/examples/artifacts/cand_toy_l1_hysteresis_comparator_dut_0001/checker.yaml
```

The candidate index binds these files by SHA-256. `pack_sft.py` refuses to pack
the item if the DUT file is missing or if its hash no longer matches the
manifest.

## Positive Path

The normal toy path uses the fixture protected index:

```bash
python3 -m train.pipelines.check_contamination \
  --protected-index train/data/manifests/examples/protected_index.yaml \
  --candidate-index train/data/manifests/examples/candidate_index.synth-batch-toy-0001.yaml \
  --out /private/tmp/vaevas_phase1_closed_loop/contamination_report.yaml
```

Expected shape:

```text
contamination_check_ok=1
total=1
clean=1
rejected=0
```

Then SFT packing:

```bash
python3 -m train.pipelines.write_admitted_manifest \
  --candidate-index train/data/manifests/examples/candidate_index.synth-batch-toy-0001.yaml \
  --static-check-report train/data/manifests/examples/static_check_report.synth-batch-toy-0001.yaml \
  --evas-report train/data/manifests/examples/evas_verification_report.synth-batch-toy-0001.yaml \
  --contamination-report train/data/manifests/examples/contamination_report.synth-batch-toy-0001.yaml \
  --diversity-report train/data/manifests/examples/diversity_report.synth-batch-toy-0001.yaml \
  --split-manifest train/data/manifests/examples/split_manifest.synth-batch-toy-0001.yaml \
  --spectre-report train/data/manifests/examples/spectre_shadow_report.audit-toy-0001.yaml \
  --require-spectre-shadow \
  --out /private/tmp/vaevas_phase1_closed_loop/admitted_manifest.yaml
```

Expected shape:

```text
admitted_manifest_ok=1
admitted_count=1
rejected_count=0
```

```bash
python3 -m train.pipelines.pack_sft \
  --admitted-manifest /private/tmp/vaevas_phase1_closed_loop/admitted_manifest.yaml \
  --candidate-index train/data/manifests/examples/candidate_index.synth-batch-toy-0001.yaml \
  --out-dir /private/tmp/vaevas_phase1_closed_loop/sft \
  --manifest-out /private/tmp/vaevas_phase1_closed_loop/sft_pack_manifest.yaml \
  --run-id toy-sft-local
```

The SFT record contains:

```json
{
  "instruction": "...",
  "input": "contract summary ...",
  "output": "<think>...</think><answer>Verilog-A DUT...</answer>",
  "system": "..."
}
```

This is supervised learning: the model sees the target answer.

Then GRPO packing:

```bash
python3 -m train.pipelines.pack_grpo \
  --admitted-manifest /private/tmp/vaevas_phase1_closed_loop/admitted_manifest.yaml \
  --candidate-index train/data/manifests/examples/candidate_index.synth-batch-toy-0001.yaml \
  --out-dir /private/tmp/vaevas_phase1_closed_loop/grpo \
  --manifest-out /private/tmp/vaevas_phase1_closed_loop/grpo_prompt_manifest.yaml \
  --run-id toy-grpo-local
```

The GRPO record contains:

```json
{
  "item_id": "...",
  "prompt": "contract summary without answer ...",
  "contract_ref": "...",
  "reward_profile": "L1_dut",
  "reward_runtime_ref": {...}
}
```

It must not contain `output`, `completion`, `gold_completion`, or `answer`.
GRPO learns by sampling completions and receiving reward feedback.

## Negative Path

A useful contamination checker must also reject something. If we build a
protected index from the same clean-room contract directory and compare the toy
candidate against it, the candidate is rejected because its `contract_sha256`
matches protected material exactly:

```bash
python3 -m train.pipelines.build_protected_index \
  --root train/data/contracts/examples \
  --out /private/tmp/vaevas_phase1_closed_loop/protected_index.yaml \
  --benchmark-release-id clean-room-contract-smoke \
  --source-manifest-ref data/contracts/schema.yaml

python3 -m train.pipelines.check_contamination \
  --protected-index /private/tmp/vaevas_phase1_closed_loop/protected_index.yaml \
  --candidate-index train/data/manifests/examples/candidate_index.synth-batch-toy-0001.yaml \
  --out /private/tmp/vaevas_phase1_closed_loop/contamination_reject_report.yaml
```

Expected shape:

```text
clean=0
rejected=1
strongest_match_tier=exact_hash
```

This validates the fail-closed direction of the gate.

## What This Still Does Not Do

The current closed loop does not:

- run EVAS;
- run Spectre;
- generate new LLM candidates;
- compute MinHash/embedding similarity;
- run SFT or GRPO training;
- produce paper metrics.

Those are later phases. This step only proves that manifest-controlled data can
flow into SFT/GRPO pack formats without bypassing admission metadata.

## Next Step

The next substantive implementation should be the first real clean-room batch
generator:

```text
seed catalog + contract templates
  -> candidate_index.<batch>.yaml
  -> EVAS verifier
  -> contamination checker
  -> admitted_manifest.<batch>.yaml
```

Bulk LLM synthesis should wait until protected-index construction and
contamination checking are run on the actual protected vaBench release assets.
