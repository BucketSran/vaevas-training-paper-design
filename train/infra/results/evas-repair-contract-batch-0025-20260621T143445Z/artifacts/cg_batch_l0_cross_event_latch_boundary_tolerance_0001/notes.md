# Draft artifact for cg_batch_l0_cross_event_latch_boundary_tolerance_0001

Generated for overnight data-preparation only.
Not EVAS verified, not Spectre verified, and not admitted training data.

## EVAS Rust subset repair

- prior_failure_diagnostic: ERROR: Failed to compile Verilog-A file solution.va: Spectre-incompatible Verilog-A: transition() contribution is inside a conditional/event/loop/case statement
- repair_class: transition_inside_conditional_or_event
- source_artifact_path: train/infra/results/contract-batch-0025-20260621T053934Z/artifacts/cg_batch_l0_cross_event_latch_boundary_tolerance_0001/solution.va
- repaired_solution_ref: train/infra/results/evas-repair-contract-batch-0025-20260621T143445Z/artifacts/cg_batch_l0_cross_event_latch_boundary_tolerance_0001/solution.va
- admission_status: draft_unadmitted
- note: This repair only targets EVAS Rust parser/subset smoke compatibility. It is not Spectre-validated and not admitted training data.
