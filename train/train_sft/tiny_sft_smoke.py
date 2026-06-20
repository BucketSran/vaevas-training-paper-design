"""
Tiny SFT smoke for the remote training platform.

This is not a production trainer. It verifies that the current server can load
the tokenizer/model, run a minimal LoRA supervised update on admitted SFT JSONL,
save a tiny adapter in an untracked work directory, reload that adapter, and
emit small evidence files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any


REQUIRED_KEYS = {"instruction", "input", "output", "system"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        missing = REQUIRED_KEYS - set(record)
        if missing:
            raise ValueError(f"{path}:{line_no} missing keys {sorted(missing)}")
        records.append(record)
    if not records:
        raise ValueError(f"empty SFT JSONL: {path}")
    return records


def expand_records(records: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    expanded: list[dict[str, Any]] = []
    index = 0
    while len(expanded) < count:
        expanded.append(records[index % len(records)])
        index += 1
    return expanded


def fallback_prompt(record: dict[str, Any], include_answer: bool) -> str:
    user = f"{record['instruction'].strip()}\n\n{record['input'].strip()}".strip()
    if include_answer:
        return f"{record['system'].strip()}\n\nUser:\n{user}\n\nAssistant:\n{record['output'].strip()}"
    return f"{record['system'].strip()}\n\nUser:\n{user}\n\nAssistant:\n"


def format_text(tokenizer: Any, record: dict[str, Any], include_answer: bool) -> str:
    user = f"{record['instruction'].strip()}\n\n{record['input'].strip()}".strip()
    messages = [
        {"role": "system", "content": record["system"].strip()},
        {"role": "user", "content": user},
    ]
    if include_answer:
        messages.append({"role": "assistant", "content": record["output"].strip()})
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=not include_answer,
        )
    except Exception:
        return fallback_prompt(record, include_answer)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_loss_csv(path: Path, log_history: list[dict[str, Any]]) -> None:
    fieldnames = ["step", "epoch", "loss", "eval_loss", "learning_rate", "grad_norm"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in log_history:
            row = {name: item.get(name, "") for name in fieldnames}
            writer.writerow(row)


def first_float(log_history: list[dict[str, Any]], key: str) -> float | None:
    for item in log_history:
        value = item.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def last_float(log_history: list[dict[str, Any]], key: str) -> float | None:
    for item in reversed(log_history):
        value = item.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def run_dry(args: argparse.Namespace) -> dict[str, Any]:
    records = read_jsonl(args.sft_train_jsonl)
    train_records = expand_records(records, args.train_records)
    eval_records = expand_records(records, args.eval_records)
    summary = {
        "schema_version": "phase1.tiny_sft_smoke_result.v0.1",
        "run_id": args.run_id,
        "status": "DRY_RUN",
        "dry_run": True,
        "input": {
            "sft_train_jsonl": str(args.sft_train_jsonl),
            "source_records": len(records),
            "train_records": len(train_records),
            "eval_records": len(eval_records),
        },
        "checks": {
            "required_keys": sorted(REQUIRED_KEYS),
            "first_record_has_required_keys": REQUIRED_KEYS <= set(records[0]),
        },
        "notes": [
            "No model was loaded.",
            "No optimizer step was run.",
            "No checkpoint or adapter was written.",
        ],
    }
    write_json(args.result_dir / "summary.json", summary)
    (args.result_dir / "status.txt").write_text("DRY_RUN\n", encoding="utf-8")
    return summary


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    import gc

    import torch
    from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false; run platform probe before SFT smoke")

    records = read_jsonl(args.sft_train_jsonl)
    train_records = expand_records(records, args.train_records)
    eval_records = expand_records(records, args.eval_records)
    if not eval_records:
        eval_records = train_records[:1]

    args.work_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.work_dir / "checkpoints"
    adapter_dir = args.work_dir / "adapter"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_texts = [format_text(tokenizer, record, include_answer=True) for record in train_records]
    eval_texts = [format_text(tokenizer, record, include_answer=True) for record in eval_records]
    generation_prompt = format_text(tokenizer, eval_records[0], include_answer=False)

    class TextDataset(Dataset[Any]):
        def __init__(self, texts: list[str]) -> None:
            self.encoded = [
                tokenizer(
                    text,
                    truncation=True,
                    max_length=args.max_length,
                    add_special_tokens=False,
                )
                for text in texts
            ]

        def __len__(self) -> int:
            return len(self.encoded)

        def __getitem__(self, index: int) -> dict[str, list[int]]:
            return self.encoded[index]

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total_params = sum(param.numel() for param in model.parameters())

    training_kwargs: dict[str, Any] = {
        "output_dir": str(checkpoint_dir),
        "overwrite_output_dir": True,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "logging_steps": 1,
        "eval_steps": 1,
        "save_strategy": "no",
        "bf16": True,
        "report_to": [],
        "disable_tqdm": True,
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
    }
    import inspect

    if "eval_strategy" in inspect.signature(TrainingArguments).parameters:
        training_kwargs["eval_strategy"] = "steps"
    else:
        training_kwargs["evaluation_strategy"] = "steps"

    train_args = TrainingArguments(**training_kwargs)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=TextDataset(train_texts),
        eval_dataset=TextDataset(eval_texts),
        data_collator=data_collator,
    )

    started_at = time.time()
    train_output = trainer.train()
    wall_time_s = time.time() - started_at
    log_history = trainer.state.log_history

    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    adapter_config = PeftConfig.from_pretrained(adapter_dir)

    model.eval()
    inputs = tokenizer(generation_prompt, return_tensors="pt", truncation=True, max_length=args.max_length).to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_text = tokenizer.decode(generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False)

    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    reloaded_model = PeftModel.from_pretrained(base_model, adapter_dir)
    reload_check_ok = adapter_config.base_model_name_or_path is not None and reloaded_model is not None
    del reloaded_model
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    loss_csv = args.result_dir / "loss.csv"
    write_loss_csv(loss_csv, log_history)
    (args.result_dir / "sample_generation.txt").write_text(generated_text + "\n", encoding="utf-8")

    final_loss = last_float(log_history, "loss")
    final_eval_loss = last_float(log_history, "eval_loss")
    status = "PASS" if reload_check_ok and final_loss is not None else "FAIL"

    summary = {
        "schema_version": "phase1.tiny_sft_smoke_result.v0.1",
        "run_id": args.run_id,
        "status": status,
        "dry_run": False,
        "model_path": args.model_path,
        "sft_train_jsonl": str(args.sft_train_jsonl),
        "work_dir": str(args.work_dir),
        "adapter_dir": str(adapter_dir),
        "adapter_dir_committed": False,
        "paper_metric": False,
        "toy_data_only": True,
        "train_records": len(train_records),
        "eval_records": len(eval_records),
        "max_steps": args.max_steps,
        "max_length": args.max_length,
        "wall_time_s": round(wall_time_s, 3),
        "train_result": {
            "global_step": getattr(train_output, "global_step", None),
            "training_loss": getattr(train_output, "training_loss", None),
        },
        "loss": {
            "first_train_loss": first_float(log_history, "loss"),
            "final_train_loss": final_loss,
            "first_eval_loss": first_float(log_history, "eval_loss"),
            "final_eval_loss": final_eval_loss,
            "loss_csv": "loss.csv",
        },
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "trainable_params": trainable_params,
            "total_params": total_params,
            "trainable_fraction": trainable_params / total_params if total_params else None,
            "reload_check_ok": reload_check_ok,
        },
        "generation": {
            "sample_generation_ref": "sample_generation.txt",
            "max_new_tokens": args.max_new_tokens,
        },
        "artifacts": {
            "loss_csv": "loss.csv",
            "sample_generation": "sample_generation.txt",
            "adapter_manifest": "adapter_manifest.json",
        },
    }
    write_json(args.result_dir / "summary.json", summary)
    write_json(
        args.result_dir / "adapter_manifest.json",
        {
            "adapter_dir": str(adapter_dir),
            "adapter_dir_committed": False,
            "contains_model_weights": True,
            "commit_policy": "do not commit adapter/checkpoint/model files to GitHub",
            "reload_check_ok": reload_check_ok,
        },
    )
    (args.result_dir / "status.txt").write_text(status + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny remote SFT smoke.")
    parser.add_argument("--sft-train-jsonl", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-path", default="/data/jinzhihong/vaEVAS/models/base/Qwen2.5-Coder-7B-Instruct-vaevas")
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--train-records", type=int, default=4)
    parser.add_argument("--eval-records", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    try:
        if args.dry_run:
            summary = run_dry(args)
        else:
            summary = run_train(args)
        print(f"tiny_sft_smoke_status={summary['status']}")
        print(f"result_dir={args.result_dir}")
    except Exception as exc:
        failure = {
            "schema_version": "phase1.tiny_sft_smoke_result.v0.1",
            "run_id": args.run_id,
            "status": "FAIL",
            "dry_run": args.dry_run,
            "error": f"{type(exc).__name__}: {str(exc)}",
            "traceback": traceback.format_exc(),
        }
        write_json(args.result_dir / "summary.json", failure)
        (args.result_dir / "status.txt").write_text("FAIL\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
