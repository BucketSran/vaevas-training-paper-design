# Diagnostic GRPO Reward Spec

Status: design draft. No reward implementation or training result is claimed
here.

This document defines the concrete GRPO reward design for the training paper.
It supersedes the older five-component sketch in `03_reward_design.md`.

## Design Goal

The reward must train executable behavioral Verilog-A generation, not merely
format compliance or compile success. It should provide enough dense signal for
GRPO while preserving Spectre-grounded credibility.

The reward is:

- **contract-driven**: every score is derived from `train/data/contracts/schema.yaml`;
- **diagnostic**: failures are classified and partially scored when useful;
- **verifier-grounded**: EVAS Rust provides training-time signals;
- **Spectre-audited**: Spectre checks reward false positives through shadow audit;
- **anti-hacking**: shortcuts that satisfy weak checkers without solving the
  contract are penalized or hard-rejected.

## Inputs and Outputs

### Reward function signature

Future implementation should expose one top-level deterministic function:

```python
def compute_reward(
    prompt: str,
    completion: str,
    contract: dict,
    runtime: dict,
) -> tuple[float, dict]:
    ...
```

Return:

```python
(
  R_total: float,          # clipped to [0, 1]
  diagnostic: dict,        # component scores, gates, error taxonomy, artifact refs
)
```

`runtime` should include verifier paths, timeout settings, EVAS profile, current
training stage, and optional cached verifier outputs. Reward functions must not
call LLMs.

### Required diagnostic payload

Every completion should produce:

```yaml
score:
  total: 0.0
  components:
    R_format: 0.0
    R_contract: 0.0
    R_static: 0.0
    R_compile_diag: 0.0
    R_sim_health: 0.0
    R_property: 0.0
    R_l2_decomp: 0.0
    R_repair_delta: 0.0
    R_feedback_use: 0.0
    P_anti_hack: 0.0
gates:
  hard_ban: false
  structure_ok: false
  compile_ok: false
  simulate_ok: false
  property_scored: false
error_taxonomy:
  primary: null
  secondary: []
artifacts:
  parsed_answer_ref: null
  staged_run_ref: null
  evas_result_ref: null
  spectre_shadow_ref: null
```

This diagnostic object is part of the paper evidence. It makes reward curves
explainable instead of a single opaque scalar.

## Scalar Aggregation

### Gates

Define binary gates:

```text
G_structure = 1 if output can be parsed into the required artifact set else 0
G_compile   = 1 if EVAS compile/elaboration passes else 0
G_sim       = 1 if EVAS simulation completes with required outputs else 0
G_hard_ban  = 1 if a hard exploit or unsafe artifact is detected else 0
```

Hard rule:

```text
if G_hard_ban == 1:
    R_total = 0
```

Soft rule:

```text
R_sim_health = 0 if G_compile == 0
R_property   = 0 if G_compile == 0 or G_sim == 0
R_l2_decomp  = 0 if G_compile == 0 or G_sim == 0
```

`R_compile_diag` can still be nonzero when compile fails, because parser and
elaboration diagnostics provide useful near-miss signal.

### Formula

For a selected reward profile `W`:

```text
R_raw =
    W_format       * R_format
  + W_contract     * R_contract
  + W_static       * R_static
  + W_compile_diag * R_compile_diag
  + W_sim_health   * R_sim_health
  + W_property     * R_property
  + W_l2_decomp    * R_l2_decomp
  + W_repair_delta * R_repair_delta
  + W_feedback_use * R_feedback_use
  - P_anti_hack

R_total = clip(R_raw, 0, 1)
```

Weights are chosen by `level`, `task_form`, and curriculum stage. The component
weights should sum to 1 before subtracting `P_anti_hack`.

## Default Reward Profiles

These are initial defaults for pilot experiments. They must be ablated before
paper claims.

| Profile | `R_format` | `R_contract` | `R_static` | `R_compile_diag` | `R_sim_health` | `R_property` | `R_l2_decomp` | `R_repair_delta` | `R_feedback_use` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `L0_conformance` | 0.05 | 0.10 | 0.10 | 0.15 | 0.20 | 0.40 | 0.00 | 0.00 | 0.00 |
| `L1_dut` | 0.05 | 0.10 | 0.05 | 0.15 | 0.15 | 0.50 | 0.00 | 0.00 | 0.00 |
| `L1_tb` | 0.05 | 0.15 | 0.10 | 0.10 | 0.15 | 0.45 | 0.00 | 0.00 | 0.00 |
| `L1_bugfix` | 0.05 | 0.05 | 0.05 | 0.10 | 0.10 | 0.25 | 0.00 | 0.35 | 0.05 |
| `L2_e2e` | 0.05 | 0.10 | 0.05 | 0.10 | 0.10 | 0.25 | 0.35 | 0.00 | 0.00 |
| `L2_bugfix` | 0.05 | 0.05 | 0.05 | 0.10 | 0.10 | 0.15 | 0.20 | 0.25 | 0.05 |

Interpretation:

- L1 direct generation puts most mass on functional properties.
- L2 shifts mass from single final metric into decomposed system subclaims.
- Bugfix profiles reward actual improvement over the broken artifact.
- Format stays low; it is a hygiene signal, not a research contribution.

## Component Definitions

### `R_format`

Measures whether the completion can be parsed into the expected top-level
sections and artifacts.

Suggested scoring:

| Check | Weight |
| --- | ---: |
| Required answer/code block present | 0.35 |
| Required reasoning/diagnosis block present when requested | 0.20 |
| All required artifact filenames or module markers present | 0.25 |
| No extra unsupported artifact channels | 0.20 |

If the output cannot be parsed at all, `R_format = 0` and `G_structure = 0`.

### `R_contract`

Measures alignment with the contract before running the simulator.

Suggested scoring:

| Check | Weight |
| --- | ---: |
| Required modules/artifacts present for `task_form` | 0.25 |
| Ports match contract names, directions, and disciplines | 0.30 |
| Parameters match names, units, defaults, and declared ranges | 0.20 |
| Observable signals are exposed or saved as required | 0.15 |
| `category`, `level`, and `task_form` constraints respected | 0.10 |

For `tb`, the testbench must instantiate the reference DUT interface. For
`bugfix`, the patch must target the broken artifact rather than replacing the
task with unrelated code.

### `R_static`

Measures static legality and EVAS/Spectre scope compatibility.

Suggested scoring:

| Check | Weight |
| --- | ---: |
| No forbidden constructs from the contract | 0.35 |
| No current-domain/KCL/KVL/device-level behavior in voltage-domain tasks | 0.20 |
| No unsupported analog operators unless contract explicitly allows them | 0.20 |
| No task ID, checker note, or benchmark-row special casing | 0.15 |
| Source size and module count within contract bounds | 0.10 |

If a hard exploit is detected, set `G_hard_ban = 1` even if some static checks
would otherwise earn partial credit.

### `R_compile_diag`

Measures EVAS Rust frontend/compile/elaboration health. This replaces the old
OpenVAF-only compile reward.

If compile/elaboration passes:

```text
R_compile_diag = 1
G_compile = 1
```

If it fails, use diagnostics:

| Diagnostic class | Partial score |
| --- | ---: |
| Answer extracted but parser fails early | 0.10 |
| Verilog-A syntax mostly valid but localized syntax error remains | 0.25 |
| Modules parse but declarations/parameters are invalid | 0.40 |
| Modules elaborate but port/interface mismatch remains | 0.55 |
| Only unsupported construct or simulator-scope issue remains | 0.65 |
| Compile times out or crashes infrastructure | 0.00 |

The exact class must be logged under `error_taxonomy.primary`.

### `R_sim_health`

Measures whether executable output produces usable simulation traces, not
whether it is functionally correct.

Suggested scoring:

| Check | Weight |
| --- | ---: |
| Simulation terminates within timeout | 0.25 |
| Required observables exist in output traces/logs | 0.20 |
| Values are finite: no NaN/Inf/exploding rails | 0.20 |
| Signals are non-degenerate when contract expects dynamics | 0.20 |
| Event/sample coverage reaches required windows | 0.15 |

Constant garbage can receive `R_sim_health > 0` if the simulator ran, but it
should score near zero on `R_property`.

### `R_property`

Scores each contract property independently and averages by property weight.

For contract properties `p_i`:

```text
R_property = Σ_i property_weight_i * score_property(p_i)
```

If no explicit weights exist, use equal weights, except for L2 contracts where
the final system metric should not exceed 40% of total property weight.

#### Property scorers

| Property type | Score |
| --- | --- |
| `voltage_bounds` | Fraction of checked samples within tolerance, with excess scaled by `voltage_abs_v`. |
| `threshold_crossing` | Average of correct crossing count, direction, and event-time error score. |
| `event_count` | `max(0, 1 - abs(observed - expected) / max(1, expected))`, with exact mode when requested. |
| `event_sequence` | Normalized sequence similarity; exact score if contract sets `event_sequence_exact`. |
| `timing_delay` | `max(0, 1 - abs(observed_delay - expected_delay) / timing_tol)`. |
| `settling_time` | Score combines final value error and whether settling occurs before required time. |
| `gain_estimate` | `max(0, 1 - relative_error / relative_tolerance)`. |
| `code_mapping` | `max(0, 1 - abs(code_error_lsb) / allowed_lsb_error)`. |
| `state_sequence` | Normalized edit-distance similarity over declared states. |
| `monotonicity` | Fraction of ordered comparisons satisfying monotonic relation. |
| `repair_delta` | Use `R_repair_delta`; do not double-count inside `R_property` unless explicitly configured. |

All property scorers must return diagnostics: observed value, expected value,
tolerance, and failure reason.

### `R_l2_decomp`

Only used for L2 mini-systems. It exists because single end-to-end metrics are
too sparse for GRPO.

Default decomposition:

```text
R_l2_decomp =
    0.20 * R_interface
  + 0.25 * R_subblock
  + 0.20 * R_timing_sequence
  + 0.35 * R_system_metric
```

Definitions:

| Subscore | Meaning |
| --- | --- |
| `R_interface` | Modules connect through declared ports, rails, clocks, and parameters. |
| `R_subblock` | Key internal component behaviors satisfy local properties where observable. |
| `R_timing_sequence` | Events, state transitions, and handshakes occur in legal order. |
| `R_system_metric` | Final system outcome, e.g. SAR code, PLL lock, calibration residual. |

For L2 GRPO, log each subscore separately. If `R_system_metric` is zero but
interface and sequence are nonzero, the rollout is still useful for learning.

### `R_repair_delta`

Only used for `bugfix` or feedback-repair prompts.

Let `S_before` be the verifier score of the broken artifact and `S_after` be
the verifier score of the patched artifact under the same contract:

```text
R_repair_delta = max(0, S_after - S_before) / (1 - S_before + 1e-6)
```

`S_before` and `S_after` should use the executable subset of the reward:

```text
S_exec = 0.20 * R_compile_diag + 0.20 * R_sim_health + 0.60 * R_property
```

For L2 repair, replace `R_property` with:

```text
0.45 * R_property + 0.55 * R_l2_decomp
```

This rewards improvement, not merely producing code that looks clean.

### `R_feedback_use`

Low-weight signal for whether the model used verifier feedback correctly.

Suggested scoring:

| Check | Weight |
| --- | ---: |
| Identifies the primary error class from verifier feedback | 0.35 |
| Patch location matches the failing artifact region | 0.25 |
| Explanation is consistent with the applied patch | 0.15 |
| Feedback use is validated by positive `R_repair_delta` | 0.25 |

Hard guard:

```text
if R_repair_delta == 0:
    R_feedback_use = min(R_feedback_use, 0.25)
```

This prevents fluent but ineffective debugging narratives from being rewarded.

### `P_anti_hack`

Penalty for shortcuts. Suggested cap:

```text
0 <= P_anti_hack <= 0.25
```

Soft penalties:

| Pattern | Penalty |
| --- | ---: |
| Excessive code length beyond contract budget | 0.03-0.08 |
| Dead modules or unused outputs | 0.03-0.08 |
| Artificial delays unrelated to contract timing | 0.05-0.12 |
| Fragile dependence on one fixed stimulus value | 0.05-0.15 |
| Repeated boilerplate that does not affect behavior | 0.03-0.10 |

Hard bans set `G_hard_ban = 1`:

- hardcoding task IDs, checker note strings, or expected answer constants;
- modifying reference testbench/checker when task form does not allow it;
- file-output tricks that bypass observable behavior;
- simulator hang or runaway resource use;
- using forbidden constructs to escape the declared EVAS/Spectre scope;
- special-casing benchmark/release identifiers.

## Curriculum Schedule

Weights should change slowly over training, not jump between unrelated reward
functions.

### Stage A: Reward smoke and early GRPO

Goal: avoid all-zero rewards.

- Use mostly L1 prompts and a small L2 slice.
- Keep compile/sim/diagnostic weights relatively high.
- Require group reward standard deviation to be nonzero before scaling.

Suggested adjustment:

```text
R_property_weight_multiplier = 0.80
R_compile_diag_weight_multiplier = 1.15
R_sim_health_weight_multiplier = 1.10
```

Renormalize weights after applying multipliers.

### Stage B: Main L1 GRPO

Goal: optimize functional correctness.

- Use default `L1_dut`, `L1_tb`, and `L1_bugfix` profiles.
- Track property subscore curves separately by property type.

### Stage C: Mixed L1/L2 GRPO

Goal: make L2 reward variance usable.

- Increase L2 prompt fraction only after L1 compile/sim health is stable.
- Use `L2_e2e` with decomposed subclaims.
- Do not rely on final system metric alone.

### Stage D: L2-focused GRPO

Goal: improve mini-system correctness.

- Raise L2 prompt fraction.
- Keep L1 prompts as anchors to prevent regression.
- Increase `R_l2_decomp` and final system metric within the `L2_e2e` profile only
  after shadow audit stays clean.

## Spectre Shadow Audit

Spectre is not part of every GRPO step. It is an audit channel.

### During training

Sample from each checkpoint:

- high EVAS reward completions,
- low EVAS reward completions,
- L2 completions,
- repair completions with large `R_repair_delta`,
- completions from categories with recent EVAS/Spectre drift.

### Audit outcomes

| Outcome | Reward implication |
| --- | --- |
| EVAS PASS / Spectre PASS | Keep reward configuration. |
| EVAS PASS / Spectre FAIL | Zero-tolerance false positive; quarantine affected reward/checker/contract family. |
| Spectre PASS / EVAS FAIL | Add to EVAS false-negative backlog; do not silently lower candidate quality. |
| EVAS FAIL / Spectre FAIL | Model or contract failure; use diagnostics for repair data. |

Do not retroactively report EVAS-only reward as correctness if shadow audit is
not available for the claimed slice.

## Calibration Protocol

Before any GRPO run, calibrate rewards on a small fixed panel:

1. **Gold artifacts**: should score `R_total >= 0.95`.
2. **Near-miss mutations**: should land in `0.35-0.80`, depending on severity.
3. **Severe invalid mutations**: should score `< 0.30`.
4. **Known hard hacks**: should score `0` through hard ban.
5. **Random/base outputs**: should not all be exactly zero after SFT cold start.

GRPO readiness criteria:

```text
For group size N=8:
- >= 60% of prompts have nonzero reward variance after SFT cold start.
- Median group reward std >= 0.05 on the pilot prompt pool.
- No component saturates above 0.95 for >80% of samples except R_format.
- Gold examples score high and controlled bad examples score low.
```

If these fail, fix the reward or curriculum before scaling training.

## Logging Requirements

Every GRPO run must log:

- total reward and all component rewards;
- gates and hard-ban rate;
- error taxonomy histogram;
- property-type score histogram;
- L1 vs L2 reward curves;
- direct-generation vs repair reward curves;
- EVAS runtime and timeout rate;
- Spectre shadow audit sample IDs and mismatch table;
- reward config hash and contract schema version.

Component logs are mandatory for paper ablations.

## Required Ablations

| Ablation | Expected evidence |
| --- | --- |
| Binary pass/fail reward | Shows diagnostic reward improves reward variance and final correctness. |
| No `R_l2_decomp` | Tests whether L2 decomposed subclaims matter. |
| No `R_repair_delta` | Tests whether feedback repair training matters. |
| No `P_anti_hack` | Tests reward-hacking exposure. |
| L2 GRPO without L2 SFT cold start | Tests the L2 cold-start decision. |
| EVAS-only without Spectre shadow audit | Not a trusted training result; used only to quantify audit necessity. |

## Implementation Modules

Future code should replace the old simple module list with:

| Module | Responsibility |
| --- | --- |
| `rewards/extract_artifacts.py` | Parse completion into required artifacts. |
| `rewards/contract_reward.py` | `R_format`, `R_contract`, schema alignment. |
| `rewards/static_semantics.py` | `R_static`, hard-ban checks. |
| `rewards/evas_runner.py` | EVAS compile/sim execution with timeout and caching. |
| `rewards/compile_diag_reward.py` | `R_compile_diag` and compile taxonomy. |
| `rewards/sim_health_reward.py` | `R_sim_health`. |
| `rewards/property_reward.py` | Per-property scorers from contract schema. |
| `rewards/l2_decomp_reward.py` | L2 interface/subblock/timing/system scoring. |
| `rewards/repair_reward.py` | `R_repair_delta` and `R_feedback_use`. |
| `rewards/anti_hack.py` | Soft penalties and hard bans. |
| `rewards/aggregate.py` | Weight profiles, gates, clipping, diagnostics. |

No implementation should start until this spec and the contract schema are
reviewed together.
