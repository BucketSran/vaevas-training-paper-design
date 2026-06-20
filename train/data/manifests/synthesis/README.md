# Synthesis Run Manifests

This directory stores Phase 1 synthesis run plans.

Run plans are not generated data. They define which prompt records may be sent
to an LLM or remote Codex task, where outputs should be written, and which
validators must pass before any generated contract can move to artifact
generation.

Rules:

- Keep generated contracts out of `train/data/sft/`, `train/data/rl/`, and
  `train/data/eval/`.
- Use `train/infra/results/<run_id>/` for small smoke outputs that need GitHub
  handoff.
- Keep bulk generated candidates in ignored scratch directories until admitted.
- Do not treat a generated contract index as admitted SFT/GRPO data.

