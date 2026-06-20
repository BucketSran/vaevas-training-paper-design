# Minimal Manifest Fixtures

These YAML files are minimal Phase 1 manifest fixtures. They are deliberately
small and use toy IDs, fake hashes, and placeholder artifact paths.

They are useful for:

- validating manifest loaders;
- documenting required fields with concrete examples;
- testing that SFT/GRPO packers consume `admitted_manifest` instead of raw
  generated candidates;
- keeping provenance, contamination, split, and reward metadata separate from
  model-visible training text.

They are not useful for:

- training a model;
- reporting benchmark metrics;
- proving EVAS/Spectre correctness;
- proving remote platform health.

## Files

| File | Role |
| --- | --- |
| `batch_plan.synth-batch-toy-0001.yaml` | Tiny batch-plan reference used by the candidate index. |
| `protected_index.yaml` | One protected toy asset fingerprint. |
| `candidate_index.synth-batch-toy-0001.yaml` | One generated candidate tied to a clean-room contract. |
| `static_check_report.synth-batch-toy-0001.yaml` | Static checks for the candidate. |
| `evas_verification_report.synth-batch-toy-0001.yaml` | EVAS Rust compile/sim/property evidence. |
| `contamination_report.synth-batch-toy-0001.yaml` | Protected-overlap and split-leakage decision. |
| `spectre_shadow_manifest.audit-toy-0001.yaml` | Spectre audit selection. |
| `spectre_shadow_report.audit-toy-0001.yaml` | Toy EVAS/Spectre agreement result. |
| `diversity_report.synth-batch-toy-0001.yaml` | Template-collapse decision. |
| `split_manifest.synth-batch-toy-0001.yaml` | Split ownership for admitted item IDs. |
| `admitted_manifest.synth-batch-toy-0001.yaml` | The only fixture downstream packers should consume. |
| `sft_pack_manifest.toy-run-0001.yaml` | SFT JSONL packing metadata. |
| `grpo_prompt_manifest.toy-run-0001.yaml` | GRPO prompt packing metadata without gold completions. |

## Fixture IDs

The single candidate is:

```text
candidate_id: cand_toy_l1_hysteresis_comparator_dut_0001
contract_id: cg_l1_hysteresis_comparator_dut_0001
item_id: adm_toy_l1_hysteresis_comparator_dut_0001
split_key: comparator/hysteresis_toy_fixture
```

The fixture reuses a tracked clean-room contract example path but does not copy
or derive from protected vaBench release assets.
