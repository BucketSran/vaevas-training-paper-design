# Pilot Batch Manifests

This directory stores non-bulk Phase 1 pilot planning manifests.

Pilot batch plans define target distributions, seed references, split policy,
and verification queues before any LLM generation starts. They are intended to
make generation reproducible and auditable, not to admit data by themselves.

Run:

```bash
python3 -m train.pipelines.validate_pilot_plan
```

before using a pilot plan for generation.

