# Prompt Templates

This directory stores prompt templates for Phase 1 clean-room synthesis.

The templates are used to render generation requests from seed catalogs and
pilot batch plans. They are not training data, model outputs, or admitted
examples.

Run:

```bash
python3 -m train.pipelines.render_pilot_prompts
```

The default output goes to the system temp directory under
`vaevas_phase1_prompt_gate/` so rendered prompt records are not committed
accidentally.

Rules:

- Templates must not include protected benchmark release content.
- GRPO-visible prompt fields must not contain target completions or gold code.
- Contract proposal prompts produce draft YAML only; artifact proposal prompts
  produce candidate artifacts only after review permits generation.
