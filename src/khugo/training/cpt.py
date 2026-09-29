"""Reproducible QLoRA continued-pretraining path for Khugo-Base."""

from __future__ import annotations

import inspect
import json
from dataclasses import MISSING, asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from ..data.registry import load_source_registry
from ..data.schema import TrainingDocument
from ..data.validation import assess_training_eligibility
from ..evaluation.artifacts import git_revision, runtime_environment, write_json
from ..reproducibility import seed_everything
from .config import RunConfig, load_mapping


@dataclass(frozen=True, slots=True)
class CPTConfig(RunConfig):
    """All parameters required for a small, auditable QLoRA CPT experiment."""

    validation_data: str
    model_revision: str
    tokenizer_revision: str
    fp16: bool = True
    allow_remote_model: bool = False
    use_qlora: bool = True
    load_in_4bit: bool = True
    quantization_type: str = "nf4"
    gradient_checkpointing: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    )
    logging_steps: int = 10
    save_steps: int = 100
    resume_from_checkpoint: str | None = None

    def __post_init__(self) -> None:
        RunConfig.__post_init__(self)
        required = (self.validation_data, self.model_revision, self.tokenizer_revision)
        if not all(value.strip() for value in required):
            raise ValueError("validation_data, model_revision, and tokenizer_revision are required")
        if self.bf16:
            raise ValueError("the GTX 1660 SUPER CPT profile requires bf16: false")
        if not self.fp16:
            raise ValueError("the GTX 1660 SUPER CPT profile requires fp16: true")
        if not self.use_qlora or not self.load_in_4bit or self.quantization_type != "nf4":
            raise ValueError("the initial CPT profile requires 4-bit NF4 QLoRA")
        if self.lora_r <= 0 or self.lora_alpha <= 0 or not 0 <= self.lora_dropout < 1:
            raise ValueError("invalid LoRA configuration")
        if not self.lora_target_modules or self.logging_steps <= 0 or self.save_steps <= 0:
            raise ValueError("LoRA targets, logging_steps, and save_steps must be non-empty/positive")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CPTConfig":
        fields = cls.__dataclass_fields__
        unexpected = set(value) - set(fields)
        if unexpected:
            raise ValueError(f"unexpected CPT configuration keys: {sorted(unexpected)}")
        missing = [name for name, field in fields.items() if field.default is MISSING and name not in value]
        if missing:
            raise ValueError(f"missing CPT configuration keys: {sorted(missing)}")
        prepared = dict(value)
        target_modules = prepared.get("lora_target_modules")
        if isinstance(target_modules, list) and all(isinstance(item, str) for item in target_modules):
            prepared["lora_target_modules"] = tuple(target_modules)
        return cls(**prepared)

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


def load_cpt_config(path: Path) -> CPTConfig:
    """Load and validate a local CPT configuration."""
    return CPTConfig.from_mapping(load_mapping(path))


def run_cpt(*, config: CPTConfig, registry_path: Path, repository_root: Path) -> Path:
    """Run QLoRA CPT over explicitly supplied, locally prepared JSONL splits."""
    _validate_training_data(config, registry_path)
    output_dir = Path(config.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not config.resume_from_checkpoint:
        raise FileExistsError("output directory is non-empty; set resume_from_checkpoint explicitly")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
            DataCollatorForLanguageModeling, Trainer, TrainingArguments,
        )
    except ImportError as error:
        raise RuntimeError("install Khugo training dependencies before running CPT") from error
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA CPT requires a CUDA GPU; no CUDA device is available")

    local_files_only = not config.allow_remote_model
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path, revision=config.tokenizer_revision,
        local_files_only=local_files_only, trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type=config.quantization_type,
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path, revision=config.model_revision,
        local_files_only=local_files_only, trust_remote_code=False,
        quantization_config=quantization, device_map="auto",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=config.gradient_checkpointing
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=config.lora_r, lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout, target_modules=list(config.lora_target_modules), bias="none",
        ),
    )
    raw_datasets = load_dataset(
        "json", data_files={"train": config.train_data, "validation": config.validation_data}
    )

    def tokenize(batch: Mapping[str, list[str]]) -> Mapping[str, list[list[int]]]:
        return tokenizer(batch["text"], truncation=True, max_length=config.max_sequence_length)

    tokenized = raw_datasets.map(
        tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
    )
    tokens_per_epoch = sum(len(item) for item in tokenized["train"]["input_ids"])
    write_json(
        output_dir / "dataset_statistics.json",
        {
            "train_documents": len(raw_datasets["train"]),
            "validation_documents": len(raw_datasets["validation"]),
            "train_tokens_after_truncation": tokens_per_epoch,
            "max_sequence_length": config.max_sequence_length,
        },
    )
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": _training_arguments(TrainingArguments, config, output_dir),
        "train_dataset": tokenized["train"],
        "eval_dataset": tokenized["validation"],
        "data_collator": DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    }
    trainer_kwargs["processing_class" if "processing_class" in inspect.signature(Trainer).parameters else "tokenizer"] = tokenizer
    trainer = Trainer(
        **trainer_kwargs,
    )
    train_result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    evaluation = trainer.evaluate()
    trainer.save_model(str(output_dir / "adapter"))
    tokenizer.save_pretrained(output_dir / "adapter")
    trainer.save_state()
    observed_tokens = getattr(trainer.state, "num_input_tokens_seen", 0) or (
        tokens_per_epoch * config.num_train_epochs
    )
    write_json(
        output_dir / "training_metrics.json",
        {"train": train_result.metrics, "validation": evaluation, "tokens_processed": observed_tokens},
    )
    write_json(
        output_dir / "training_manifest.json",
        {
            "config": config.to_mapping(), "git_revision": git_revision(repository_root),
            "environment": runtime_environment(),
            "checkpoint_information": sorted(
                item.name for item in output_dir.iterdir() if item.name.startswith("checkpoint-")
            ),
            "trainable_parameters": _trainable_parameter_count(model),
            "artifacts": ["dataset_statistics.json", "training_metrics.json", "trainer_state.json", "adapter/"],
        },
    )
    return output_dir


def _validate_training_data(config: CPTConfig, registry_path: Path) -> None:
    registry = load_source_registry(registry_path)
    for dataset_path in (Path(config.train_data), Path(config.validation_data)):
        if not dataset_path.is_file():
            raise FileNotFoundError(f"prepared dataset not found: {dataset_path}")
        with dataset_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    document = TrainingDocument.from_mapping(json.loads(line))
                    decision = assess_training_eligibility(document, registry)
                    if not decision.accepted:
                        raise ValueError(
                            f"ineligible training document at {dataset_path}:{line_number}: {decision.reason}"
                        )


def _training_arguments(training_arguments_class: Any, config: CPTConfig, output_dir: Path) -> Any:
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir), "run_name": config.run_name,
        "num_train_epochs": config.num_train_epochs,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate, "fp16": config.fp16, "bf16": config.bf16,
        "gradient_checkpointing": config.gradient_checkpointing,
        "logging_steps": config.logging_steps, "save_steps": config.save_steps,
        "save_total_limit": 2, "report_to": "none", "seed": config.seed,
        "data_seed": config.seed, "optim": "paged_adamw_8bit", "save_strategy": "steps",
    }
    names = inspect.signature(training_arguments_class).parameters
    kwargs["eval_strategy" if "eval_strategy" in names else "evaluation_strategy"] = "epoch"
    if "include_num_input_tokens_seen" in names:
        kwargs["include_num_input_tokens_seen"] = True
    return training_arguments_class(**kwargs)


def _trainable_parameter_count(model: Any) -> dict[str, int]:
    return {
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "total": sum(parameter.numel() for parameter in model.parameters()),
    }
