# 00 — Overview

How SFT, GRPO, rewards, data, and evaluation fit together.

## Picture

```
                ┌──────────────────────────────────────┐
                │   Contract-reviewed sources          │
                │   - clean-room contracts             │
                │   - audited public references        │
                │   - audited internal examples        │
                │   - LLM-synthesized variants         │
                └────────────────┬─────────────────────┘
                                 │  pipelines/synthesize.py
                                 ▼
                ┌──────────────────────────────────────┐
                │   data/synthesized/                  │
                │   candidate <spec, va, tb>           │
                └────────────────┬─────────────────────┘
                                 │  pipelines/verify_evas.py
                                 │  (drop anything that fails admission gates)
                                 ▼
                ┌──────────────────────────────────────┐
                │   data/verified/                     │
                │   (with provenance, EVAS diagnostics,│
                │    Spectre shadow-audit status)      │
                └────────────────┬─────────────────────┘
                                 │  pipelines/build_trajectory.py
                                 │  pipelines/check_contamination.py
                                 ▼
                ┌──────────────────────────────────────┐
                │   data/sft/    data/rl/   data/eval/ │
                └──────────┬───────────────────────────┘
                           │
        ┌──────────────────┼────────────────────────┐
        │                  │                        │
        ▼                  ▼                        ▼
  ┌──────────┐      ┌─────────────┐         ┌─────────────┐
  │  Phase 2 │      │   Phase 3   │         │   Phase 4   │
  │  SFT     │ ──▶  │   GRPO      │ ──▶     │   Eval      │
  │  loop    │      │   loop      │         │   harness   │
  └────┬─────┘      └──────┬──────┘         └─────────────┘
       │                   │  diagnostic reward profiles
       ▼                   │  + repair/feedback rewards
  base + SFT ckpt          ▼
                     SFT + GRPO ckpt
```

## Why each piece exists

| Piece | Role | Justified by |
|---|---|---|
| Contracts → synthesized | Expand clean, typed task definitions into diverse candidates | Phase 1 KPI (need ≥300 verified) |
| Verifier (EVAS) | Filter broken candidates, label diagnostics, provide fast reward signal | Without it, reward signal is unverifiable |
| Spectre shadow audit | Check that reportable EVAS judgments remain aligned with the reference simulator | Prevents training on EVAS false positives |
| Trajectory builder | Decompose artifacts into verifiable tagged steps | Required for structured reward and SFT cold start |
| SFT | Cold-start: teach the model output format + basic capability | Without SFT, RL reward is sparse → won't converge |
| GRPO | Push the model past SFT plateau, optimize for verifier-grounded rewards | Circuit-Think: SFT alone gets 35%, +GRPO with right reward gets 73% |
| Repair/feedback learning | Train the model to use verifier diagnostics and improve broken artifacts | Makes GRPO feedback useful beyond binary pass/fail |
| Held-out eval | Honest measure of generalization, OOD probe | Contamination firewall has no value without it |

## The Circuit-Think → vaBench mapping

| Circuit-Think | vaBench equivalent |
|---|---|
| Input: circuit image | Input: text spec |
| Output: SPICE netlist | Output: Verilog-A module (+ optionally testbench) |
| Base: Qwen2.5-VL-7B | Base: Qwen2.5-Coder-7B |
| Verifier: SPICE syntax + graph similarity | Verifier: EVAS compile/sim/property diagnostics + Spectre shadow audit |
| Trajectory: port → device → connection | Trajectory: contract → interface → behavior → testbench/checker |
| Reward components: format/answer/logic/think | Reward components: format/contract/static/compile/sim/property/L2/repair |
| Dataset: 3,100 image-netlist pairs | Dataset: contract-first clean-room set; target 300+ verified |
| Compute: paper-reported multi-GPU RL run | Compute: target GPU setup and step budget recorded per run |

## Where to read next

1. `01_sft_principles.md` — what SFT actually does
2. `02_grpo_principles.md` — what GRPO actually does
3. `DIAGNOSTIC_REWARD_SPEC.md` — the current reward design
4. `04_trajectory_format.md` — how we structure the step-by-step CoT
5. `SYNTHETIC_DATA_FACTORY.md` — contract-first data generation protocol
6. `06_eval_protocol.md` — how we judge success
7. `REFERENCES.md` — papers, codebases, tutorials
