# Remote Codex Task: Prompt Gate Validation

Run the prompt-gate validation only. This verifies that the remote checkout can
read the clean-room seed catalog, pilot batch plan, and prompt templates. It
does not call an LLM, generate training data, run EVAS, run Spectre, run SFT, or
run GRPO.

## Hard Constraints

- Do not run full SFT.
- Do not run GRPO.
- Do not call external LLM APIs.
- Do not generate or commit training data.
- Do not install, upgrade, or remove packages.
- Do not commit checkpoints, model weights, `.env` files, tokens, simulator
  outputs, or rendered prompt JSONL unless explicitly requested later.

## Commands

Update the repo:

```bash
cd /data/jinzhihong/vaevas-training-paper-design
git fetch origin
git checkout training-paper-design-20260620
git pull --ff-only
```

Run local validators:

```bash
python3 -m train.pipelines.validate_pilot_plan
python3 -m train.pipelines.render_pilot_prompts \
  --out-dir /tmp/vaevas_phase1_prompt_gate_remote
python3 -m train.pipelines.validate_manifest_fixtures
```

Report back:

```bash
cat /tmp/vaevas_phase1_prompt_gate_remote/prompt_plan_summary.json
```

## Expected Result

The commands should report:

- `PASS validate_pilot_plan`
- `PASS render_pilot_prompts`
- `prompt_count = 15`
- prompt kinds include `contract_proposal`, `contract_review`, and `artifact_proposal`

If a command fails, paste the full traceback and the content of
`/tmp/vaevas_phase1_prompt_gate_remote/` if it exists. Do not commit failure
artifacts unless asked.

