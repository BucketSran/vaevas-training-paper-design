# Clean-Room Contract Proposal

You are proposing a new clean-room Verilog-A task contract. Produce only YAML
matching `train/data/contracts/schema.yaml`. Do not emit Verilog-A code.

## Safety Rules

- Do not copy or paraphrase protected benchmark release prompts, code, checks, or metadata.
- Do not use hidden checker logic or target answers.
- Keep the task inside voltage-domain behavioral Verilog-A.
- Define observable behavior before implementation details.
- Use `category`, `level`, and `task_form`; do not invent a parallel taxonomy.
- Use a new `split_key` that groups related variants together.

## Seed Metadata

- seed_id: `{seed_id}`
- source_tier: `{source_tier}`
- source_kind: `{source_kind}`
- source_ref: `{source_ref}`
- category: `{category}`
- intended_levels: `{intended_levels}`
- intended_task_forms: `{intended_task_forms}`
- allowed_uses: `{allowed_uses}`
- forbidden_uses: `{forbidden_uses}`
- contamination_review: `{contamination_review}`

## Source Contract Summary

```text
{source_contract_summary}
```

## Required Contract Fields

```text
{schema_required_fields}
```

## Output Requirements

- Return one draft contract.
- Use `schema_version: "0.1"`.
- Set `provenance.source_tier` to `B_llm_synthetic`.
- Set `provenance.source_kind` to `llm_synthetic`.
- Include the seed ID in `provenance.seed_refs`.
- Set verifier evidence to `not_run` or `pending`; do not claim EVAS/Spectre success.
- Set admission status to `draft_contract`.

