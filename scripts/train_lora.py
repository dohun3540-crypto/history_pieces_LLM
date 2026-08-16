"""Permission-gated LoRA entry point for a separate CUDA training server."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        value = value.strip('"')
        if value.lower() in {"true", "false"}:
            values[key] = value.lower() == "true"
        elif value.replace(".", "", 1).isdigit():
            values[key] = float(value) if "." in value else int(value)
        else:
            values[key] = value
    return values


def validate_sft_file(path: Path) -> int:
    if not path.is_file():
        raise RuntimeError(f"SFT file not found: {path}")
    count = 0
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("training_eligible") is not True:
            raise RuntimeError(f"{path}:{number} is not training eligible")
        evidence = record.get("source_evidence")
        if not evidence or any(item.get("allowed_for_training") is not True for item in evidence):
            raise RuntimeError(f"{path}:{number} has unapproved source evidence")
        messages = record.get("messages")
        if not isinstance(messages, list) or [item.get("role") for item in messages][-1:] != ["assistant"]:
            raise RuntimeError(f"{path}:{number} has invalid chat messages")
        count += 1
    if not count:
        raise RuntimeError(f"SFT file is empty: {path}")
    return count


def preflight(config: dict[str, Any]) -> dict[str, Any]:
    model = Path(str(config["model_name_or_path"]))
    tokenizer = Path(str(config["tokenizer_name_or_path"]))
    output = Path(str(config["output_dir"]))
    if not model.is_dir() or not any(model.glob("*.safetensors")):
        raise RuntimeError("local model checkpoint is unavailable")
    if not tokenizer.is_dir() or not (tokenizer / "tokenizer_config.json").is_file():
        raise RuntimeError("local tokenizer is unavailable")
    if model.resolve() == output.resolve() or output.resolve() in model.resolve().parents:
        raise RuntimeError("output_dir must not overwrite the base checkpoint")
    missing = [name for name in ("torch", "transformers", "peft", "trl", "datasets") if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError("missing training packages: " + ", ".join(missing))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch and a training GPU are required")
    train_count = validate_sft_file(Path(str(config["train_file"])))
    validation_count = validate_sft_file(Path(str(config["validation_file"])))
    return {
        "model": str(model), "tokenizer": str(tokenizer),
        "train_samples": train_count, "validation_samples": validation_count,
        "gpu": torch.cuda.get_device_name(0), "output_dir": str(output),
    }


def train(config: dict[str, Any]) -> None:
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(
        config["tokenizer_name_or_path"], local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"], local_files_only=True, device_map="auto",
        torch_dtype="auto",
    )
    dataset = load_dataset("json", data_files={
        "train": config["train_file"], "validation": config["validation_file"],
    })
    args = TrainingArguments(
        output_dir=config["output_dir"], seed=int(config["seed"]),
        num_train_epochs=float(config["num_train_epochs"]),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(config["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        learning_rate=float(config["learning_rate"]), bf16=bool(config["bf16"]),
        gradient_checkpointing=bool(config["gradient_checkpointing"]),
        save_strategy=str(config["save_strategy"]),
        eval_strategy=str(config["eval_strategy"]),
        load_best_model_at_end=bool(config["load_best_model_at_end"]),
        metric_for_best_model=str(config["metric_for_best_model"]),
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, args=args,
        train_dataset=dataset["train"], eval_dataset=dataset["validation"],
        max_seq_length=int(config["max_seq_length"]),
        peft_config=LoraConfig(
            r=int(config["lora_r"]), lora_alpha=int(config["lora_alpha"]),
            lora_dropout=float(config["lora_dropout"]), task_type="CAUSAL_LM",
        ),
        formatting_func=lambda row: tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False
        ),
    )
    trainer.train()
    trainer.evaluate()
    trainer.save_model(config["output_dir"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    try:
        report = preflight(config)
    except RuntimeError as error:
        print(json.dumps({"status": "blocked", "reason": str(error)}, indent=2))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.run:
        train(config)
    else:
        print("preflight passed; training was not started (add --run to start)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
