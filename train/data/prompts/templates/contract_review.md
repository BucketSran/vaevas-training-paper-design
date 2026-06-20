# Contract Review Request

Review the draft contract against the Phase 1 checklist. Do not generate code.

## Review Checklist Source

Use `train/docs/CONTRACT_REVIEW_CHECKLIST.md` as the policy source.

## Seed Metadata

- seed_id: `{seed_id}`
- source_tier: `{source_tier}`
- source_ref: `{source_ref}`
- allowed_uses: `{allowed_uses}`
- forbidden_uses: `{forbidden_uses}`

## Draft Contract

```yaml
{contract_yaml}
```

## Required Output

Return YAML only:

```yaml
review_decision: accept_for_generation | needs_revision | quarantine | reject
reviewer: llm_contract_reviewer
blocking_findings: []
required_edits: []
allowed_next_stage: contract_generation | artifact_generation | none
```

