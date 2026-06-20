# models/

Base model configuration and checkpoint pointers. **Checkpoints are gitignored** — they live on remote storage.

## Base model

| Item | Value |
|---|---|
| Name | `Qwen2.5-Coder-7B-Instruct` |
| HuggingFace | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| Why | Strong code base, supports Verilog-A-adjacent syntax, fits 2× A100 with full-parameter SFT and GRPO |
| Alternatives (deferred) | `Qwen2.5-Coder-32B`, `DeepSeek-Coder-V2-Lite` |

## Files

| File | Role |
|---|---|
| `base_config.yaml` | Base model identity, tokenizer settings, HF revision pin |
| `.gitignore` | Excludes all `*.bin`, `*.safetensors`, `checkpoint*/`, etc. |

## Checkpoint layout (under remote storage, NOT under this dir)

```
remote:/path/to/models/
├── base/Qwen2.5-Coder-7B-Instruct/    (downloaded once)
├── sft-run-001/
├── sft-run-002/
├── grpo-run-001/
└── grpo-run-002/
```

Local `models/` keeps only:
- Config files (`base_config.yaml`, run-specific YAML).
- Pointer files: `<run-id>.checkpoint_ref` containing the remote path.

## Why no local checkpoints

A 7B bf16 checkpoint is ~14GB. Two-three iterations and the local repo bloats. Checkpoint storage is the remote's job; this directory tracks identity and config only.

## Restore-from-remote (sketch, Phase 2+)

```
# Pull a specific checkpoint locally only when needed for inference debugging
rsync -avz user@remote:/path/to/models/grpo-run-001/ /tmp/ckpt/
```

## Phase 0 placeholders

No `base_config.yaml` exists yet — created in Phase 2 when SFT scripts land.
