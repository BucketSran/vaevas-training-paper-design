# Clean-Room Artifact Proposal

Generate the requested Verilog-A-facing artifact from the reviewed contract.
Emit only the artifact requested by `task_form`; do not emit extra explanation.

## Safety Rules

- Do not copy protected benchmark release assets.
- Do not include target answers, hidden checker logic, or checker-only secrets.
- Do not special-case task IDs, module names, or benchmark IDs.
- Stay inside the allowed EVAS/Spectre voltage-domain subset.
- Prefer clear behavioral Verilog-A over clever simulator-specific hacks.
- If the contract is inconsistent or outside scope, emit a short refusal comment
  explaining the blocker instead of fabricating an answer.

## Requested Artifact

- seed_id: `{seed_id}`
- contract_id: `{contract_id}`
- category: `{category}`
- level: `{level}`
- task_form: `{task_form}`
- base_function: `{base_function}`
- forbidden_constructs: `{forbidden_constructs}`

## Public Contract

```yaml
{contract_yaml}
```

## Output Contract

- For `dut`, emit one Verilog-A module implementing the public interface.
- For `tb`, emit a transient testbench or testbench-facing artifact matching the reference interface.
- For `bugfix`, emit a patched artifact only; do not include the broken artifact unless explicitly requested.
- For `e2e`, emit the composed top-level behavior and any helper modules needed by the public interface.
- For `conformance`, emit the minimal artifact needed to exercise the semantic case.

