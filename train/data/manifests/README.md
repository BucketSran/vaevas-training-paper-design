# manifests/

Phase 1 manifest examples and future generated manifest outputs.

The source-of-truth schema description is
`../../docs/PHASE1_MANIFEST_SCHEMAS.md`.

Tracked contents in this directory are schema examples only. Real generated
manifests from bulk synthesis or verifier runs should be reviewed before being
committed.

## Current Examples

`examples/` contains a single toy candidate flowing through the minimum Phase 1
evidence chain:

```text
protected_index
  + candidate_index
  + static_check_report
  + evas_verification_report
  + contamination_report
  + spectre_shadow_report
  + diversity_report
  + split_manifest
  -> admitted_manifest
  -> sft_pack_manifest / grpo_prompt_manifest
```

These files are not training data. They use fake hashes and placeholder artifact
paths to test field names, references, and state transitions.
