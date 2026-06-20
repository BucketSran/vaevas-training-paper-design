# contracts/

Contract schemas and example task contracts for the training-paper data factory.

These files are **not training data**. They define the machine-readable task
contract format that future synthesis, verification, SFT packing, GRPO reward,
and audit pipelines should consume.

## Files

| File | Purpose |
|---|---|
| `schema.yaml` | Human-readable YAML schema for clean-room synthetic task contracts. |
| `examples/*.yaml` | Representative draft contracts covering L0/L1/L2 and direct/repair/e2e task forms. |

## Naming policy

Reuse vaBench language:

- `category` follows vaBench circuit-role taxonomy.
- `level` is `L0`, `L1`, or `L2`.
- `task_form` is `dut`, `tb`, `bugfix`, `e2e`, or `conformance`.

Do not use `family` for circuit category. In existing vaBench metadata, `family`
often means task form such as `spec-to-va`, `tb-generation`, `end-to-end`, or
`bugfix`.

## Admission rule

An example contract can guide data generation only after it has explicit
provenance, split information, contamination status, and verifier-evidence
placeholders. It can enter SFT/GRPO data only after the future data pipeline
fills those fields with concrete audit results.
