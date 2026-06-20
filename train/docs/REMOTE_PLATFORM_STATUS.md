# Remote Platform Status

Status: local memory snapshot, not live-verified in this session.

This document records the existing remote SFT/GRPO platform state so the
training-paper plan does not accidentally treat already completed setup work as
missing. It should be updated once SSH access to the server is available again.

## Known Completed Work

Source of evidence: `docs/sft/09_smoke_lessons.md`.

The SFT platform has already passed an engineering smoke test on the remote
server:

| Layer | Evidence |
| --- | --- |
| Hardware/runtime | 8 × A100, CUDA, bf16 matrix check passed. |
| Base model | `Qwen2.5-Coder-7B-Instruct` loaded, chat template worked, 50-token generation worked. |
| Tokenizer adaptation | 16 trajectory tokens registered and initialized. |
| Mini SFT | LLaMA-Factory SFT ran 12 steps in 17.8 s; eval loss moved from 1.81 to 1.55. |
| Merge/inference | LoRA merged into a 15 GB model; generated output contained `module`/`endmodule`. |

The historical smoke proves:

- remote Python/CUDA/model/LLaMA-Factory plumbing worked at least once;
- the 7B model was downloaded and loadable;
- the `-vaevas` tokenizer-adapted base model existed;
- the training/merge/chat loop was operational.

It does not prove:

- Phase 1 clean-room data exists;
- a production SFT run has been completed;
- GRPO with EVAS reward has been run;
- the remote environment is still unchanged today.

## Known Remote Artifact Map

Historical paths from the smoke report:

```text
/data/jinzhihong/vaEVAS/
├── models/base/Qwen2.5-Coder-7B-Instruct/
├── models/base/Qwen2.5-Coder-7B-Instruct-vaevas/
├── models/qwen2.5-coder-7b-vaevas-smoke/
├── outputs/qwen2.5-coder-7b/lora/smoke/
├── data/llamafactory/dataset_info.json
├── data/llamafactory/vaevas_sft.jsonl
├── configs/llamafactory/vaevas_lora_sft.yaml
├── configs/llamafactory/vaevas_lora_sft_smoke.yaml
├── configs/llamafactory/vaevas_merge_lora.yaml
├── configs/llamafactory/vaevas_merge_lora_smoke.yaml
├── configs/llamafactory/vaevas_inference_lora.yaml
├── configs/llamafactory/vaevas_inference_lora_smoke.yaml
└── pipelines/preprocess_tokenizer.py
```

Operational rule:

- Future SFT and GRPO should prefer
  `models/base/Qwen2.5-Coder-7B-Instruct-vaevas/` as the base model because it
  contains the trajectory tokens.
- Smoke outputs validate plumbing only. They are not clean-room Phase 1 data and
  should not be used as paper-facing training evidence.

## GRPO Platform Status

The GRPO side is partially specified but not proven end-to-end in the tracked
notes.

Known from `docs/grpo/00_environment_prep.md`:

| Component | Status in notes |
| --- | --- |
| LLaMA-Factory conda env | Installed and usable as base environment. |
| `torch`, `transformers`, `trl`, `accelerate`, `datasets`, `modelscope` | Recorded as installed in the LF env. |
| `vLLM` | Planned/required for rollout; current install status needs live check. |
| `verl` | Planned/required for production GRPO; current install status needs live check. |
| EVAS reward service | Not implemented or smoke-tested end-to-end. |
| Spectre on remote | Needs live check before any audit claim. |

User memory indicates that the relevant platform may already have been downloaded
or staged on the server. Until SSH is available, treat this as a likely
engineering asset, not current verification evidence.

## Live Check When Server Access Returns

Run only non-destructive checks first:

```bash
ssh <remote> 'hostname; date; nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv'
ssh <remote> 'ls -ld /data/jinzhihong/vaEVAS/models/base/Qwen2.5-Coder-7B-Instruct*'
ssh <remote> 'source /data/home/maxiaokang/miniconda3/etc/profile.d/conda.sh && conda activate /data/jinzhihong/envs/llamafactory && python - <<PY
import torch, transformers, trl
print(torch.__version__)
print(transformers.__version__)
print(trl.__version__)
print(torch.cuda.is_available())
PY'
ssh <remote> 'source /data/home/maxiaokang/miniconda3/etc/profile.d/conda.sh && conda activate /data/jinzhihong/envs/llamafactory && python - <<PY
for name in ["vllm", "verl", "ray", "flash_attn"]:
    try:
        mod = __import__(name)
        print(name, "OK", getattr(mod, "__version__", "unknown"))
    except Exception as exc:
        print(name, "MISSING", type(exc).__name__, str(exc)[:120])
PY'
```

Record the result here before changing remote packages.

## Impact on Current Local Work

No server network is required for the current Phase 1 local work:

- finalize manifest and admission contracts;
- define clean-room seed and batch-plan formats;
- define packer interfaces for existing LLaMA-Factory paths;
- keep GRPO prompt manifests compatible with `verl`/`vLLM`, but do not require
  remote GRPO smoke until data and reward runtime exist.

Do not spend time rebuilding the remote SFT platform unless live checks show it
is missing or corrupted.
