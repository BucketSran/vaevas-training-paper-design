# rewards/

Reward function modules used by GRPO. Each reward is a **pure function** of `(prompt, completion, contract, runtime)` returning a float in `[0, 1]`, plus diagnostic metadata.

> **Design philosophy**: rewards are the soul of GRPO. Spend more time here than on the GRPO loop itself. The current detailed design is `../docs/DIAGNOSTIC_REWARD_SPEC.md`; `../docs/03_reward_design.md` is the older simple sketch.

## Reward components

| File | Reward | Range | Source of signal |
|---|---|---|---|
| `contract_reward.py` | `R_format`, `R_contract` | 0-1 | Output parses and aligns to the task contract. |
| `static_semantics.py` | `R_static` | 0-1 | Forbidden constructs and EVAS/Spectre scope checks. |
| `compile_diag_reward.py` | `R_compile_diag` | 0-1 | EVAS Rust parser/elaboration diagnostics. |
| `sim_health_reward.py` | `R_sim_health` | 0-1 | EVAS simulation termination, observables, finite traces, coverage. |
| `property_reward.py` | `R_property` | 0-1 | Contract property scorers. |
| `l2_decomp_reward.py` | `R_l2_decomp` | 0-1 | Interface, subblock, timing, and system metric subclaims for L2. |
| `repair_reward.py` | `R_repair_delta`, `R_feedback_use` | 0-1 | Improvement from broken artifact and correct use of verifier feedback. |
| `anti_hack.py` | `P_anti_hack` | 0-0.25 or hard ban | Reward-hacking penalties and zero-score bans. |

## Aggregation

See `../docs/DIAGNOSTIC_REWARD_SPEC.md` for profiles by `level` and
`task_form`. The scalar reward is a clipped weighted sum of diagnostic
components minus anti-hack penalties.

## Testing

`tests/test_rewards.py` (to be added in Phase 3):
- Unit tests with known-good and known-bad completions.
- Property tests: rewards in `[0, 1]`, gating works correctly.
- Integration test: full reward computation on a sample completion.

## Design rules

1. **No model calls in reward functions.** Rewards must be deterministic, not LLM-judged.
2. **Use EVAS as the training-time verifier**, not heuristics. If EVAS can't decide, the reward is undefined (not 0.5), and Spectre shadow audit decides whether EVAS is a trusted proxy on the claimed slice.
3. **Reward shaping is opinionated.** Document any non-obvious choice with a comment in the reward file AND a paragraph in `docs/03_reward_design.md`.
4. **Avoid reward hacking**: anything that lets the model trivially earn reward without solving the task → fix it before scaling training.
