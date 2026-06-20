# Training Paper Experiment Design

Status: design draft. No training result is claimed here.

This document consolidates the current plan for a separate training paper on
SFT + GRPO for executable behavioral Verilog-A generation. It is intentionally
separate from the vaBench benchmark paper: the benchmark paper establishes the
benchmark/evaluator artifact, while this paper studies verifier-grounded model
training using clean-room data.

## Positioning

### Working thesis

Verifier-grounded training can improve local Verilog-A generation when the
training signal is executable, diagnostic, and Spectre-audited:

> SFT gives a local code model executable behavioral-modeling structure; GRPO
> uses decomposed verifier rewards to improve functional and system-level
> correctness, especially for L2 mini-systems.

### Main method components

1. **Clean-room training set**: rebuild training data with provenance,
   contamination checks, and verifier evidence instead of reusing old sweeps.
2. **Contract-first synthetic data factory**: generate data from explicit
   behavioral contracts, not free-form LLM self-consistency.
3. **Diagnostic verifier reward**: score structure, diagnostics, simulation
   health, functional properties, repair progress, and anti-hacking guards.
4. **EVAS-accelerated, Spectre-grounded training**: EVAS Rust drives fast reward
   feedback only under continuous Spectre shadow audit.
5. **Direct generation plus feedback repair**: train both first-attempt code
   generation and use of verifier feedback for repair.

### Non-goals

- Do not use vaBench release rows as training data.
- Do not report vaBench accuracy from the training loop.
- Do not treat EVAS as a replacement for Spectre.
- Do not claim a new RL algorithm; the contribution is the verifier-grounded
  training system and evidence.
- Do not import historical experiment outputs into training without item-level
  re-audit.

## Terminology

Reuse the vaBench taxonomy instead of inventing a parallel one.

| Term | Meaning in this design |
| --- | --- |
| `category` | vaBench circuit-role taxonomy, e.g. `Comparator and Decision Circuits`, `Data Converter Models`, `Sampling and Analog Memory`. |
| `level` | Functional granularity: `L0` primitive, `L1` component, `L2` mini-system. |
| `task_form` | vaBench-style task form: `dut`, `tb`, `bugfix`, or `e2e`. |
| `base_function` | The specific reusable behavior, e.g. `hysteresis_comparator` or `track_and_hold`. |
| `split_key` | A leakage-control key used to keep related templates, circuits, and variants in one split. |

Avoid using `family` to mean circuit category. In existing vaBench metadata,
`family` often means task family such as `spec-to-va`; this design uses
`task_form` for that axis.

## Research Questions

### RQ1: Does SFT create an executable Verilog-A cold start?

Compare base model vs SFT on clean held-out direct-generation tasks. The
expected improvement is strongest on formatting, interface correctness,
compile/simulation health, and L1 component behavior.

### RQ2: Does GRPO improve beyond SFT when rewards are diagnostic?

Compare SFT vs SFT+GRPO using the same held-out splits. The expected gain should
appear in functional correctness and all-pass rate, not merely compile rate.

### RQ3: Does decomposed GRPO matter more for L2 mini-systems?

Report L1 and L2 separately. The target hypothesis is that SFT is sufficient to
teach many L1 patterns, while verifier-decomposed GRPO is needed for L2
composition, timing, and end-to-end metrics.

### RQ4: Can EVAS accelerate training without losing Spectre credibility?

Use EVAS for training-time rewards, but quantify agreement through Spectre
shadow audit and final Spectre evaluation. EVAS speed is a method claim only
when same-slice EVAS/Spectre timing and parity evidence exist.

### RQ5: Does contract-first synthetic expansion improve robustness?

Compare contract-first synthetic data against smaller clean seed-only data and,
if feasible, against free-form synthetic generation. The expected benefit is
better diversity, less overfitting, and stronger OOD performance.

## Clean-Room Data Policy

### Source tiers

| Tier | Source | Default use |
| --- | --- | --- |
| A | EVAS examples, `veriloga-skills` knowledge, manual clean specs, public sources with provenance | Safe seeds after provenance capture. |
| B | LLM synthetic expansion from safe contracts | Main scale-up path after verifier filtering. |
| C | Controlled mutation from clean gold contracts | Main repair-data and diversity path. |
| D | Historical runs, benchmark-adjacent artifacts, old sweeps | Excluded by default; taxonomy inspiration only unless item-level re-audited. |

### Dataset partitions

Keep three datasets separate:

1. **SFT dataset**: supervised examples for direct generation and repair.
2. **GRPO prompt pool**: prompts/contracts without target completions; rollouts
   are scored by the verifier reward.
3. **Held-out evaluation set**: never used for reward tuning, early stopping, or
   prompt iteration.

### Scale targets

| Stage | Target scale | Purpose |
| --- | ---: | --- |
| Pilot | 300-500 verified examples | Validate data factory, reward service, and split logic. |
| Paper minimum | 2k-3k verified examples plus 300-500 held-out tasks | Run defensible SFT/GRPO comparisons. |
| Strong target | 8k-15k verified examples | Reduce template memorization and support category/OOD analysis. |

These are not claim numbers. They are planning targets; final claims require
actual dataset manifests and run logs.

### Split rules

- Split by `split_key`, not by individual prompt string.
- Keep variants of the same contract/template in the same split.
- Reserve at least one category or base-function cluster as OOD evaluation.
- Keep final held-out prompts invisible to synthesis prompts and repair prompts.
- Record dataset hash, contract hash, generator prompt hash, verifier commit, and
  reward commit in every training log.

## Contract-First Synthetic Data Factory

### Why contract-first

If the same LLM freely generates the spec, DUT, testbench, and checker, it can
produce self-consistent but wrong data. EVAS may pass such data if the checker is
weak or checks the wrong property. The contract-first design makes the behavior
to be checked explicit before generating artifacts.

### Minimal contract schema

The schema is not a new benchmark taxonomy. It is the smallest machine-readable
contract needed to generate, verify, split, and audit training examples.

```yaml
id: cg_l1_hysteresis_comparator_0001
category: Comparator and Decision Circuits
level: L1
task_form: dut
base_function: hysteresis_comparator
domain: voltage
difficulty: medium

intent: >
  Differential input with hysteresis drives a rail-like output and avoids
  chatter near the threshold.

ports:
  - {name: inp, direction: input, discipline: electrical, role: positive_input}
  - {name: inn, direction: input, discipline: electrical, role: negative_input}
  - {name: out, direction: output, discipline: electrical, role: decision_output}

parameters:
  - {name: vhi, unit: V, default: 1.2, range: [0.8, 1.8]}
  - {name: vlo, unit: V, default: 0.0, range: [0.0, 0.1]}
  - {name: vhyst, unit: V, default: 0.02, range: [0.005, 0.08]}

stimulus_space:
  waveforms: [slow_ramp, noisy_ramp, stepped_differential]
  sweeps:
    vhyst: [0.01, 0.02, 0.05]
    slope_v_per_s: [1e6, 5e6, 1e7]

observables:
  - out
  - inp_minus_inn

properties:
  - id: rail_bounds
    type: voltage_bounds
    signal: out
    expected: {low: 0.0, high: 1.2, tolerance_v: 0.05}
  - id: rising_threshold
    type: threshold_crossing
    condition: inp_minus_inn crosses +vhyst/2
  - id: falling_threshold
    type: threshold_crossing
    condition: inp_minus_inn crosses -vhyst/2
  - id: no_chatter
    type: event_count_window
    max_extra_edges: 0

forbidden_constructs:
  - current_contribution
  - idt
  - ddt
  - laplace
  - noise_operators
  - hardcoded_testbench_time

tolerances:
  voltage_abs_v: 0.02
  timing_abs_s: 1e-10

split_key: comparator/hysteresis/template-v1
provenance:
  source_kind: clean_room_synthetic
  seed_refs: []
  generator_prompt_hash: null
  contamination_checked: false
verifier_evidence:
  evas: null
  spectre_shadow: null
```

### Factory pipeline

1. **Seed abstraction**: extract safe behavioral intents from Tier A sources.
2. **Contract proposal**: LLM proposes contracts under vaBench category/level
   rules.
3. **Contract review**: reject contracts that are outside EVAS scope, duplicate
   release rows, or lack measurable properties.
4. **Multi-view artifact proposal**: generate DUT, testbench, checker, variants,
   and faults from the contract. Prefer separate prompts or models for DUT and
   checker to reduce self-consistent hallucination.
5. **Verifier filtering**: run EVAS first for scale; run Spectre shadow audit on
   sampled and high-risk examples.
6. **Diversity filtering**: enforce category, level, task_form, template, AST,
   and parameter-space diversity.
7. **Packaging**: emit SFT examples, GRPO prompts, repair examples, metadata, and
   manifests.

### Data admitted to training

An example can enter SFT or GRPO only if it has:

- contract metadata,
- provenance metadata,
- contamination check result,
- EVAS verifier evidence,
- split assignment,
- artifact hashes,
- checker/property version,
- Spectre shadow evidence when required by the audit schedule.

## L0/L1/L2 Training Use

`level` is a curriculum and evaluation axis, not merely a difficulty label.

| Level | Training role | Risk if overused |
| --- | --- | --- |
| L0 | Teach primitive Verilog-A/EVAS semantics and support parity calibration. | Model becomes a primitive/syntax model rather than a circuit model. |
| L1 | Main executable component modeling surface. | Can overfit to single-block templates. |
| L2 | Mini-system composition, interface alignment, and delayed system metrics. | Reward too sparse if used before cold start. |

### SFT curriculum

Initial target mix:

- L0: 5-10%
- L1: 65-75%
- L2: 15-25%

L2 appears in SFT to provide a cold start for module composition and output
structure. It is not expected to solve L2 correctness by itself.

### GRPO curriculum

1. **Reward smoke**: tiny prompt set, verify non-NaN rewards and nonzero group
   variance.
2. **L1-heavy GRPO**: optimize executable component behavior with dense
   diagnostic rewards.
3. **Mixed L1/L2 GRPO**: introduce L2 once compile/sim health is stable.
4. **L2-focused GRPO**: increase L2 weight and rely on decomposed system reward.

Run an ablation comparing `SFT without L2 -> L2 GRPO` against `SFT with small
L2 -> L2 GRPO` to test whether L2 cold start matters.

## Training Modes

### Direct generation

Input:

```text
contract/spec + required artifacts
```

Output:

```text
Verilog-A DUT, testbench, or mini-system artifacts depending on task_form
```

Measures first-attempt generation quality.

### Feedback repair

Input:

```text
contract/spec + broken artifact + verifier feedback + failing properties
```

Output:

```text
patched artifact
```

Measures whether the model can use compile, simulation, checker, and audit
feedback to repair code.

### Repair data sources

Use clean sources first:

1. **Mutation-induced repair**: controlled faults injected into verified clean
   artifacts.
2. **On-policy repair**: failures produced by current base/SFT/GRPO checkpoints
   on clean prompts.
3. **Historical repair**: excluded by default; use only after item-level audit.

Each repair sample should record before/after verifier scores and the error
taxonomy.

## Diagnostic Verifier Reward

The reward should be a vector with named components before aggregation. This is
necessary for debugging, ablations, and paper evidence.

### Reward components

| Component | Signal | Purpose |
| --- | --- | --- |
| `R_contract_structure` | Required artifacts, module names, ports, disciplines, parameters, output tags. | Avoid wasting RL on malformed outputs. |
| `R_static_semantics` | Forbidden constructs, unsupported operators, obvious EVAS/Spectre scope violations. | Prevent invalid shortcuts and unsupported Verilog-A. |
| `R_compile_diag` | Parser/elaboration/interface diagnostics, not just pass/fail. | Give partial signal for near-miss code. |
| `R_sim_health` | Simulation terminates, signals exist, no NaN, no constant garbage, event counts plausible. | Distinguish runnable behavior from broken dynamics. |
| `R_property` | Per-property functional assertions from the contract. | Main correctness target. |
| `R_l2_decomposition` | Interface connection, subblock behavior, timing sequence, and system metric. | Make L2 reward dense enough for GRPO. |
| `R_repair_delta` | Improvement from broken artifact to patched artifact. | Train use of verifier feedback. |
| `R_anti_hack` | Penalty for hardcoding, excessive length, dead code, artificial delays, checker bypass. | Reduce reward exploitation. |

### Gating policy

Use gated continuous rewards, not pure binary rewards:

- If structure is invalid, executable rewards are zero.
- If compile/elaboration fails, simulation and property rewards are zero, but
  compile diagnostics still provide partial signal.
- If simulation fails, property rewards are zero, but simulation diagnostics
  still provide partial signal.
- If simulation succeeds, score every contract property independently.
- For L2, score subclaims even when the final system metric fails.

### Error taxonomy

Every failed completion should be classified when possible:

- format/structure,
- Verilog-A syntax,
- elaboration/interface,
- unsupported construct,
- source/testbench issue,
- analog event scheduling,
- waveform/source semantics,
- signal health,
- functional property failure,
- checker/windowing issue,
- infrastructure timeout or bridge failure.

This taxonomy feeds repair data, reward debugging, and EVAS false-negative
backlog triage.

## EVAS/Spectre Audit Plan

Detailed protocol: `SPECTRE_SHADOW_AUDIT_PROTOCOL.md`.

### Roles

- Spectre is the reference oracle for paper-facing correctness claims.
- EVAS Rust is the fast training-time verifier and reward engine.
- EVAS-derived reward is credible only under parity gates and shadow audit.

### Three audit layers

1. **Pre-training parity gate**: run EVAS/Spectre agreement on representative
   contracts by category, level, and task_form before using EVAS rewards at
   scale.
2. **During-training Spectre shadow audit**: sample completions from checkpoints
   and run Spectre, prioritizing high EVAS-reward samples and category/L2
   coverage.
3. **Final held-out Spectre evaluation**: evaluate the final model with Spectre
   for headline correctness.

### Mismatch policy

| Case | Interpretation | Action |
| --- | --- | --- |
| EVAS PASS / Spectre FAIL | Reward false positive; dangerous. | Zero tolerance. Pause strong claims and repair EVAS/checker before continuing. |
| Spectre PASS / EVAS FAIL | EVAS false negative; reduces training coverage. | Batch into EVAS false-negative backlog and repair after fixed audit intervals. |
| Both fail | Model failure or invalid task. | Use diagnostics for repair data and reward analysis. |
| Both pass | Candidate trusted under declared tolerance. | Eligible for training/eval evidence. |

### Speed claim gate

Do not claim EVAS training acceleration unless the same slice records:

- EVAS wall time,
- Spectre wall time,
- row/task count,
- parity mismatch count,
- machine and bridge/Cadence configuration,
- timing aggregation such as sum speedup and median per-row speedup.

## Evaluation Protocol

### Main result tables

Report at least:

1. Base model zero-shot.
2. SFT.
3. SFT + GRPO.
4. Optional SFT + repair data.
5. Optional SFT + GRPO + repair curriculum.

Metrics:

- compile/elaboration success,
- simulation success,
- functional property score,
- all-pass rate,
- repair success rate,
- repair delta,
- L1 and L2 separately,
- category-stratified scores,
- OOD split scores,
- EVAS/Spectre audit mismatch rates.

Final paper-facing correctness should be Spectre-backed. EVAS-only metrics are
development and speed evidence unless the audit gate explicitly licenses them.

### Required ablations

| Ablation | Question answered |
| --- | --- |
| SFT-only vs SFT+GRPO | Does GRPO add value beyond imitation? |
| No L2 in SFT before L2 GRPO | Does L2 cold start matter? |
| Binary pass/fail reward vs diagnostic reward | Does reward decomposition matter? |
| No feedback-repair data | Does explicit repair training matter? |
| No contract-first filtering | Does contract-first synthesis reduce bad data or overfit? |
| No Spectre shadow audit | How much trust depends on EVAS auditing? |
| Smaller clean seed-only data | Does synthetic expansion improve scale/generalization? |

### Overfitting checks

- train/held-out split by `split_key`,
- template and AST fingerprint overlap,
- prompt n-gram overlap,
- category/base-function OOD slices,
- hidden parameter sweeps,
- checker hidden cases,
- repeated evaluation after changing stimulus parameters.

## Claim Gates

No claim is allowed unless the corresponding evidence exists:

| Claim | Required evidence |
| --- | --- |
| SFT works | Loss curve, checkpoint loads, held-out generation, verifier metrics. |
| GRPO works | Reward curves, KL/entropy logs, SFT comparison, held-out improvement. |
| Diagnostic reward matters | Reward ablations and component logs. |
| L2 GRPO matters | L1/L2 split results and L2-specific ablation. |
| EVAS accelerates training | Same-slice EVAS/Spectre timing and parity evidence. |
| Training data is clean | Provenance manifest, contamination audit, split manifest. |
| vaBench performance | Frozen-model, separately gated vaBench probe with contamination clearance. |

## Residual Risks

| Risk | Mitigation |
| --- | --- |
| Self-consistent synthetic data is wrong | Contract-first schema, multi-view generation, property-based checks, Spectre audit. |
| Synthetic data collapses into templates | Diversity filters, split keys, OOD slices, AST/template fingerprints. |
| EVAS proxy bias | Continuous Spectre shadow audit and final Spectre evaluation. |
| GRPO reward hacking | Anti-hack reward, sample inspection, hidden sweeps, reward ablations. |
| L2 reward remains sparse | L2 SFT cold start and decomposed L2 rewards. |
| Repair data is artificial | Mix mutation-induced and on-policy failures; keep historical data excluded by default. |
| Data scale creates false confidence | Report diversity and split metrics, not only example count. |

## Immediate Next Artifacts

1. `train/pipelines/check_contamination.py`: implement
   `CONTAMINATION_CHECKER_SPEC.md` as the first Phase 1 data-admission gate.
2. `train/pipelines/sample_spectre_shadow.py`: implement
   `SPECTRE_SHADOW_AUDIT_PROTOCOL.md` after the first verified candidate batch.
3. Phase 1 pilot manifests: protected index, candidate index, audit manifest,
   contamination report, and Spectre shadow report.

## Local Decision Records

This design is grounded in the current root-level domain notes and ADRs:

- `../../CONTEXT.md`
- `../../docs/adr/0001-use-evas-as-audited-training-accelerator.md`
- `../../docs/adr/0002-train-direct-generation-and-feedback-repair.md`
- `../../docs/adr/0003-build-clean-room-training-data.md`
- `../../docs/adr/0004-use-llm-synthetic-expansion-as-a-method-component.md`
- `../../docs/adr/0005-use-l2-sft-cold-start-before-l2-grpo.md`
