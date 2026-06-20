# Clean-Room Seed Catalogs

This directory stores seed catalogs for Phase 1 synthetic-data planning.

Seed catalogs are not training data. They describe audited or review-pending
intent sources that may later be expanded into task contracts and generated
candidate artifacts.

Rules:

- Do not copy assets from `behavioral-veriloga-eval/benchmark-vabench-release-v1/`.
- Do not promote a seed into SFT/GRPO data without contamination review.
- Keep `allowed_uses` and `forbidden_uses` explicit for every seed.
- Treat `seed_candidate` entries as planning inputs only until review is done.

