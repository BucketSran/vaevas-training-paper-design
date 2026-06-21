# Control: Pause Training Line

## Status

- control_id: pause_training_line_20260621
- status: active
- requested_by: user
- requested_at_local: 2026-06-21

## Directive

Pause the SFT/GRPO training-paper execution line until the user explicitly
reopens it.

Do not create, claim, or execute new jobs whose primary purpose is:

- SFT training
- GRPO training
- model checkpoint generation
- admitted training-data construction
- draft training-pack expansion beyond already-materialized evidence
- large-scale synthetic data generation for training

## Current Running Exception

At the time this pause was recorded, the remote worker had already claimed:

```text
train/infra/queue/running/20260621-repair-evas-failures-10.md
```

That job is not model training; it is EVAS smoke repair for draft artifacts. If
the remote worker is still running and can safely stop, stop it and finish the
queue job as `failed` or `PARTIAL` with diagnostics. If it has already completed
or cannot be safely interrupted, keep its result only as draft diagnostic
evidence. Do not use it to launch SFT/GRPO or make training claims.

## Resume Condition

Resume this line only after the user explicitly asks to return to SFT/GRPO
training work.

