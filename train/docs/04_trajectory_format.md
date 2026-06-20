# 04 — Trajectory Format

How we structure the model's step-by-step reasoning so that each step is **verifiable**, **trainable**, and **rewardable**.

> Status note: this is the original trajectory sketch. Keep the tagged-output
> idea, but use the current contract schema and reward profiles in
> `data/contracts/schema.yaml` and `DIAGNOSTIC_REWARD_SPEC.md` for new work.

## Why structured trajectories matter

Free-form CoT ("just think step by step") makes RL hard:
- Steps aren't separable → step-wise reward impossible
- Step content is unconstrained → no verifier can score it
- Model can dump nonsense reasoning that still leads to a correct answer → reward hacking

Structured trajectory:
- Each step is a tagged section with a defined format
- Each step has a verifier (string match, numerical check, EVAS run)
- Failure at step t can be detected → step gating works

This is Circuit-Think's key non-obvious move. We adopt the same shape.

## Top-level template

```
<think>
  <port>
    ...port identification...
  </port>
  <behavior>
    ...behavioral structure...
  </behavior>
  <testbench>
    ...testbench design (only for tb-gen / end-to-end)...
  </testbench>
</think>

<answer>
  ...Verilog-A code (and/or testbench)...
</answer>
```

## Per task form

### `spec-to-va`

Goal: text spec → Verilog-A module.

```
<think>
  <port>
    inputs: vin (electrical), clk (logic, 1-bit)
    outputs: vout (electrical)
    parameters: gain=1.0, vth=0.5
  </port>
  <behavior>
    analog block. Branches:
      - vout = gain * vin when clk high
      - vout = 0 when clk low
    Time-domain: event-driven on clk edges; @cross or @above on vin? Not needed for level-sensitive
  </behavior>
</think>

<answer>
`include "disciplines.vams"
module sample_hold (clk, vin, vout);
  input clk, vin;
  output vout;
  electrical vin, vout;
  electrical clk;

  parameter real gain = 1.0;
  ...
endmodule
</answer>
```

Step verifiers:
- `<port>`: parse, compare port list and types against `meta.json` ports — string set match
- `<behavior>`: looser semantic match against reference behavior tags
- `<answer>`: EVAS compile/elaboration + EVAS simulate + contract property checks

### `tb-generation`

Goal: spec + DUT (given) → testbench.

```
<think>
  <port>
    DUT ports: vin (input, electrical), vout (output, electrical)
    Required stimulus: vin sweeps 0→1V over 1us
    Required checks: vout = vin / 2 at t=0.5us
  </port>
  <behavior>
    Stimulus sources needed: V0 (vin), with PWL or vsource
    Strobe / measurement points: at t=0.5us, capture vout, compare against 0.5
  </behavior>
  <testbench>
    .scs structure:
      simulator lang=spectre
      ahdl_include "dut.va"
      X1 dut vin vout
      V0 vin 0 ...
      simulator lang=spice
      tran tran stop=1u
  </testbench>
</think>

<answer>
simulator lang=spectre
...
</answer>
```

Step verifiers:
- `<port>`: DUT interface understood (string compare to meta)
- `<behavior>`: stimulus and check plan structured (presence check)
- `<testbench>`: outline matches the actual answer (cross-verification)
- `<answer>`: EVAS runs the testbench against gold DUT; checks pass (R_simulate, R_metric)

### `end-to-end`

Goal: spec → both VA + testbench. Trajectory is the **union** of `spec-to-va` and `tb-generation`, in that order:
```
<think>
  <port>...</port>
  <behavior>...</behavior>
  <testbench>...</testbench>
</think>
<answer>
  <va>...</va>
  <tb>...</tb>
</answer>
```

Step gating respects the dependency: bad `<port>` → no credit for downstream.

### `bugfix`

Goal: broken VA → fixed VA. Trajectory is diagnostic, not generative:

```
<think>
  <symptom>
    EVAS log shows: "convergence error at t=2.3e-7"
  </symptom>
  <root_cause>
    discontinuous current at branch X due to missing @cross around state transition
  </root_cause>
  <fix>
    add @cross(vin - vth, +1) around the transition logic
  </fix>
</think>

<answer>
  ...fixed VA code...
</answer>
```

Step verifiers:
- `<symptom>`: match against the actual error in the given broken code (presence in EVAS log)
- `<root_cause>`: semantic match against ground-truth diagnosis (LLM-free for now: keyword match against `meta.json` `fix_category`)
- `<fix>`: presence of the fix pattern in the answer (regex / AST match)
- `<answer>`: compiles + simulates correctly (R_compile, R_simulate, R_metric)

## Building trajectories from gold

`pipelines/build_trajectory.py` (Phase 1) takes a verified `<spec, gold_va, gold_tb>` triple and produces the trajectory:

- For `<port>`: parse module declaration, extract port list and types.
- For `<behavior>`: extract analog block structure (assignments, contributions, @-events).
- For `<testbench>`: parse `.scs`, extract sources and measurement points.

For `bugfix`, requires extra metadata: the broken version + the diff describing the fix.

## Why tags?

- **Verifiable**: regex / parser can extract each step.
- **Separable**: rewards compute per-step.
- **Resilient**: if the model omits a step, format reward zero-marks it; loss is well-defined.
- **Same shape as Circuit-Think**: maintains the analogy.

## Pitfalls

1. **Too rigid → model can't express edge cases.** Allow free text within tags; only the tag structure is required.
2. **Tag overload → input/output context length explodes.** Keep tag content concise; use bullet lines, not prose.
3. **Mismatch between trajectory format and `<answer>`.** Build_trajectory must guarantee the trajectory IS consistent with the answer; otherwise the model learns to ignore trajectory.
4. **Tokenizer interaction.** Add tags as special tokens BEFORE training so they aren't split.

## Schema (to be formalized in Phase 1)

A trajectory entry in `data/verified/<id>/trajectory.json`:

```json
{
  "id": "evas_example_001",
  "level": "L1",
  "task_form": "dut",
  "category": "Comparator and Decision Circuits",
  "spec": "...",
  "trajectory": {
    "port": {...},
    "behavior": {...}
  },
  "answer": {
    "va": "..."
  },
  "step_verifiers": {
    "port": {"type": "set_match", "ground_truth": ["vin", "vout", "clk"]},
    "behavior": {"type": "regex_presence", "patterns": ["analog", "@cross"]}
  }
}
```
