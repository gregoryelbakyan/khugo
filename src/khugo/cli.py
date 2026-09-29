"""Command-line entry points for the small initial Khugo interfaces."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Callable, Sequence

from .data.deduplicate import deduplicate_documents
from .data.filter import FilterConfig, assess_document
from .data.normalize import normalize_document
from .data.schema import TrainingDocument
from .data.statistics import compute_statistics
from .data.registry import load_source_registry
from .data.split import train_validation_split
from .data.validation import assess_training_eligibility
from .evaluation.translation import bleu_score
from .evaluation.comparison import compare_experiments
from .evaluation.runner import BaselineConfig, run_baseline
from .logging import configure_logging
from .reproducibility import seed_everything
from .training.cpt import load_cpt_config, run_cpt
from .training.sft import load_sft_config

LOGGER = logging.getLogger(__name__)


def prepare_data_main(arguments: Sequence[str] | None = None) -> int:
    """Normalize, filter, and deduplicate an explicitly supplied local JSONL corpus."""
    parser = argparse.ArgumentParser(description="Prepare a local provenance-preserving JSONL corpus.")
    parser.add_argument("--input", type=Path, required=True, help="Local source JSONL path.")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSONL path.")
    parser.add_argument(
        "--validation-output", type=Path, required=True, help="Destination validation JSONL path."
    )
    parser.add_argument(
        "--rejected-output", type=Path, required=True, help="Destination audit JSONL path."
    )
    parser.add_argument(
        "--statistics-output", type=Path, required=True, help="Destination statistics JSON path."
    )
    parser.add_argument("--registry", type=Path, default=Path("data_registry/sources.yaml"))
    parser.add_argument("--min-characters", type=int, default=20)
    parser.add_argument("--dedup-threshold", type=float, default=0.90)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(arguments)
    configure_logging()

    documents = _read_documents(args.input)
    registry = load_source_registry(args.registry)
    normalized = [normalize_document(document) for document in documents]
    filter_config = FilterConfig(min_characters=args.min_characters)
    accepted: list[TrainingDocument] = []
    rejected: list[dict[str, object]] = []
    for document in normalized:
        quality = assess_document(document, filter_config)
        eligibility = assess_training_eligibility(document, registry)
        decision = quality if not quality.accepted else eligibility
        if decision.accepted:
            accepted.append(document)
        else:
            rejected.append({"document": document.to_mapping(), "reason": decision.reason})
    deduplicated = deduplicate_documents(accepted, threshold=args.dedup_threshold)
    train, validation = train_validation_split(
        deduplicated, validation_ratio=args.validation_ratio, seed=args.seed
    )
    _write_documents(args.output, train)
    _write_documents(args.validation_output, validation)
    _write_records(args.rejected_output, rejected)
    statistics = {
        "input": compute_statistics(documents).to_mapping(),
        "accepted_before_deduplication": compute_statistics(accepted).to_mapping(),
        "train": compute_statistics(train).to_mapping(),
        "validation": compute_statistics(validation).to_mapping(),
        "rejected_documents": len(rejected),
        "seed": args.seed,
        "validation_ratio": args.validation_ratio,
    }
    args.statistics_output.parent.mkdir(parents=True, exist_ok=True)
    args.statistics_output.write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    LOGGER.info(
        "data preparation completed",
        extra={
            "event": "data_prepared",
            "details": {
                "input_documents": len(documents),
                "accepted_documents": len(accepted),
                "train_documents": len(train),
                "validation_documents": len(validation),
                "rejected_documents": len(rejected),
                "statistics": statistics,
            },
        },
    )
    return 0


def train_cpt_main(arguments: Sequence[str] | None = None) -> int:
    """Run the explicitly configured local-data QLoRA CPT experiment."""
    parser = argparse.ArgumentParser(description="Run a reproducible local-data Khugo CPT experiment.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=Path("data_registry/sources.yaml"))
    args = parser.parse_args(arguments)
    configure_logging()
    config = load_cpt_config(args.config)
    output = run_cpt(config=config, registry_path=args.registry, repository_root=Path.cwd())
    LOGGER.info("CPT completed", extra={"event": "cpt_completed", "details": {"output": output}})
    return 0


def train_sft_main(arguments: Sequence[str] | None = None) -> int:
    """Validate and record an SFT configuration; a full trainer is intentionally deferred."""
    return _validate_training_config(arguments, load_sft_config, "sft_config_validated")


def evaluate_main(arguments: Sequence[str] | None = None) -> int:
    """Compute BLEU from local JSONL predictions and references."""
    parser = argparse.ArgumentParser(description="Evaluate local translation outputs.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    args = parser.parse_args(arguments)
    configure_logging()
    predictions = _read_text_lines(args.predictions, "prediction")
    references = _read_text_lines(args.references, "reference")
    score = bleu_score(predictions, [references])
    LOGGER.info("evaluation completed", extra={"event": "bleu_computed", "details": {"bleu": score}})
    return 0


def baseline_main(arguments: Sequence[str] | None = None) -> int:
    """Run the explicitly requested model baseline and save reproducible artifacts."""
    parser = argparse.ArgumentParser(description="Run a local JSONL benchmark against a language model.")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("experiments/baselines"))
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--adapter", type=Path, help="Local PEFT adapter directory to evaluate.")
    parser.add_argument("--load-in-4bit", action="store_true", help="Use NF4 loading on a CUDA GPU.")
    parser.add_argument(
        "--allow-remote-model",
        action="store_true",
        help="Permit an explicit Hugging Face model download for this run.",
    )
    args = parser.parse_args(arguments)
    configure_logging()
    result = run_baseline(
        benchmark_path=args.benchmark,
        output_root=args.output_root,
        config=BaselineConfig(
            model_name_or_path=args.model,
            model_revision=args.revision,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            allow_remote_model=args.allow_remote_model,
            adapter_path=str(args.adapter) if args.adapter else None,
            load_in_4bit=args.load_in_4bit,
        ),
        repository_root=Path.cwd(),
    )
    LOGGER.info("baseline completed", extra={"event": "baseline_completed", "details": {"output": result}})
    return 0


def compare_main(arguments: Sequence[str] | None = None) -> int:
    """Compare two completed baseline/CPT evaluation artifact directories."""
    parser = argparse.ArgumentParser(description="Compare matching Khugo experiment evaluations.")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--khugo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(arguments)
    configure_logging()
    report = compare_experiments(base_dir=args.base, khugo_dir=args.khugo, output_path=args.output)
    LOGGER.info("comparison completed", extra={"event": "comparison_completed", "details": report})
    return 0


def _validate_training_config(
    arguments: Sequence[str] | None, loader: Callable[[Path], object], event: str
) -> int:
    parser = argparse.ArgumentParser(description="Validate a reproducible local training configuration.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(arguments)
    configure_logging()
    config = loader(args.config)
    seed_everything(config.seed)
    LOGGER.info("training scaffold ready", extra={"event": event, "details": config.to_mapping()})
    return 0


def _read_documents(path: Path) -> list[TrainingDocument]:
    if not path.is_file():
        raise FileNotFoundError(f"input corpus not found: {path}")
    documents: list[TrainingDocument] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    documents.append(TrainingDocument.from_mapping(json.loads(line)))
                except (TypeError, ValueError, json.JSONDecodeError) as error:
                    raise ValueError(f"invalid document at {path}:{number}: {error}") from error
    return documents


def _write_documents(path: Path, documents: Sequence[TrainingDocument]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for document in documents:
            handle.write(json.dumps(document.to_mapping(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _write_records(path: Path, records: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _read_text_lines(path: Path, field: str) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"evaluation file not found: {path}")
    values: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)[field]
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid {field} at {path}:{number}") from error
            if not isinstance(value, str):
                raise ValueError(f"{field} at {path}:{number} must be a string")
            values.append(value)
    return values
