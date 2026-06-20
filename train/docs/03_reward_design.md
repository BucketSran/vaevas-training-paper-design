# 03 — Reward Design

The most important file in this directory.

> Status note: this file is the original simple reward sketch. The current
> concrete GRPO reward design is `DIAGNOSTIC_REWARD_SPEC.md`, which replaces the
> old OpenVAF-centric compile reward with contract-driven EVAS diagnostics and
> Spectre shadow audit.

## Principle

> "RL is reward design with a fancy gradient." — paraphrased from every RL practitioner.

We get one chance to define what "good" means. GRPO will then ruthlessly optimize for whatever we wrote down. Every degree of freedom in the reward function is a degree of freedom the model will exploit.

## The five rewards

| Reward | What it captures | Source | Range |
|---|---|---|---|
| `R_format` | Output has correct XML-style structure | string parsing | 0/1 |
| `R_compile` | Generated `.va` compiles via OpenVAF | OpenVAF exit code | 0/1 |
| `R_simulate` | EVAS runs the compiled model end-to-end | EVAS exit code | 0/1 |
| `R_metric` | Simulation output matches reference within tolerance | numerical comparison | 0-1 continuous |
| `R_trajectory` | Step-wise correctness across the reasoning trajectory | step-by-step verification | 0-1 continuous |

Plus a meta-mechanism:
- **Reflective learning**: when `R_total < τ_2`, inject the reference answer as a hint, let the model re-attempt, and reward only the improvement.

## Aggregation

```
R_total = α·R_format + β·R_compile + γ·R_simulate + δ·R_metric + ε·R_trajectory

α = 0.1   (format — hygiene only)
β = 0.2   (compile — gating)
γ = 0.2   (simulate — gating)
δ = 0.3   (metric — the real target)
ε = 0.2   (trajectory — the differentiator)
```

**Defaults to be ablated.** See `02_grpo_principles.md` for ablation list.

## Gating logic

If a reward cannot be computed, the dependent ones MUST be zero, not skipped:

```
if R_compile == 0:
    R_simulate = 0
    R_metric = 0
if R_simulate == 0:
    R_metric = 0
```

Reason: GRPO needs a defined reward to compute advantage. NaN or "skipped" rewards corrupt the group statistics.

## Each reward in detail

### R_format

Output must contain (in order):
- `<think>...</think>` block
- inside `<think>`: `<port>...</port>`, `<behavior>...</behavior>`, `<testbench>...</testbench>` (per `04_trajectory_format.md`)
- `<answer>...</answer>` block containing valid Verilog-A code

```
R_format = 1 if all tags present in correct order, else 0
```

**Why a hard 0/1, not a soft score**: format compliance is a precondition. Partial credit makes the model converge to "75% formatted" rather than "fully formatted." Same as Circuit-Think.

### R_compile

```
R_compile = 1 if OpenVAF exits 0 on the generated .va, else 0
```

**Pitfalls**:
- Treat warnings as pass (we care about compile, not lint).
- Time out compile at 30s — anything longer is a probable hang.
- Capture stderr for diagnostics, but don't penalize for stderr content alone.

### R_simulate

```
R_simulate = 1 if EVAS produces non-empty tran.csv with no error, else 0
```

**Pitfalls**:
- EVAS simulation can succeed but produce garbage (constant zero, NaN). Treat as `R_simulate = 1, R_metric = 0`, not `R_simulate = 0`. Reason: simulator ran, the metric reward is the right place to penalize.
- Time out simulation at 60s. Generally EVAS is fast; a slow run usually means an infinite loop in the model code.

### R_metric

The hardest reward. The model's output must produce simulation results that **match the reference behavior**.

Options (in order of preference):
1. **Direct trace comparison**: if the gold has a reference `tran.csv`, compare key signals via MSE / dynamic time warping. Threshold-based score.
2. **Functional features**: derived features like "comparator output goes high after t=X" — compare boolean events.
3. **Strobed values**: if the testbench has `$strobe` style checks, compare strobe outputs string-wise.

Score:
```
R_metric = max(0, 1 - error / tolerance)
```
where `error` is task-specific.

**Pitfalls**:
- A metric reward that's too tight makes EVERY sample score 0 → no signal.
- A metric reward that's too loose makes EVERY sample score 1 → no signal.
- Calibrate by running the reward function on `gold + gold` (should be 1.0) and `gold + random noisy variant` (should be < 0.5).

### R_trajectory

The Circuit-Think trick. The trajectory is decomposed into K steps; each step is scored independently; failure of step t zeroes out steps > t.

```
R_trajectory = Σ_t w_t · R_step(t),   with R_step(t) = 0 if R_step(t-1) < τ_1(t)

τ_1 schedule: 0.5 → 0.8 over training steps (curriculum)
```

For vaBench `spec-to-va`:
- Step 1: ports identified correctly (names, directions, types) — semantic match
- Step 2: behavioral structure declared (analog block, branches, parameters) — structural match
- Step 3: behavior equations match reference — symbolic / numerical match

For vaBench `tb-generation`:
- Step 1: DUT interface understood
- Step 2: stimulus designed
- Step 3: checks/assertions added

For `end-to-end` and `bugfix`, see `04_trajectory_format.md`.

**Why this matters**: without trajectory reward, the model gets credit only for the final answer. The trajectory reward forces "show your work correctly." Circuit-Think Table 6: removing step-by-step drops accuracy 73 → 35.

### Reflective learning

When `R_total < τ_2` (default 0.7), inject the reference answer as a hint and let the model re-attempt:

```
R_hat = R_total + λ_ref · max(0, R_hint - R_total) / (1 - R_total + ε)
λ_ref = 0.6, ε = 1e-6, τ_2 = 0.7
```

Only positive improvement is rewarded. If the model can't improve even with the hint, no extra reward (and small implicit penalty via the std normalization).

**Why**: when SFT init is weak, RL gets stuck. The hint provides a temporary scaffold without giving away the answer. Same idea as DAgger / curriculum learning.

## Things that look like rewards but are NOT

| Tempting | Why we don't |
|---|---|
| "Use a stronger LLM to judge correctness" | Non-deterministic, gameable, expensive |
| "Reward shorter answers" | Encourages dropping necessary content |
| "Reward longer trajectories" | Encourages verbose nonsense |
| "Penalize specific anti-patterns" | Easy to overfit to a blacklist; better to fix root cause in training data |

## Red flags during training

| Symptom | Likely cause |
|---|---|
| Model converges to one repeated completion | Reward has a trivial pass; check format / compile rewards for shortcut |
| Compile rate climbs but simulate doesn't | `R_compile = 1` rewards compileable garbage; check whether your compile check is too lenient |
| All samples score ~0.5 | Reward saturation; not enough variance for GRPO to learn |
| Trajectory reward saturates fast (≥0.9) | Step verification too lax |

## Implementation contract

Every reward function in `rewards/*.py` must:
1. Be a pure function of `(prompt, completion, ground_truth)` returning `(score: float, diagnostic: dict)`.
2. Be deterministic.
3. Have a unit test in `rewards/tests/test_<name>.py` with at least one positive and one negative case.
4. Document its tolerance / threshold parameters in the docstring AND in this file.
5. Handle malformed input gracefully — return `(0.0, {"error": "..."})`, never raise.

## When to update this file

- New reward component → add a section here, update aggregation formula.
- Weight change → log the rationale here AND in `decisions/` ADR.
- Threshold change (τ_1, τ_2, λ_ref) → log here, with calibration evidence.
