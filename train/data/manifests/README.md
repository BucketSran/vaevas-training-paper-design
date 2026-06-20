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

These files are not training data. Report-level hashes remain fixture values for
cross-reference testing, while the toy candidate's contract and artifact hashes
now bind to real fixture files so local packers can fail closed on hash mismatch.

The current toy candidate also includes tiny fixture-only artifacts under
`examples/artifacts/` so local packers can verify file hashes and emit temporary
SFT/GRPO JSONL under `/private/tmp` during smoke tests. These artifacts are not
paper training data.

Validate the current examples with:

```bash
python3 -m train.pipelines.validate_manifest_fixtures
```

The learning guide is `../../docs/MANIFEST_FIXTURE_GUIDE.md`.
