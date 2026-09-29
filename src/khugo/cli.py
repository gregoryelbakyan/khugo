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
from .evaluation.translation import bleu_score
from .logging import configure_logging
from .reproducibility import seed_everything
from .training.cpt import load_cpt_config
from .training.sft import load_sft_config

LOGGER = logging.getLogger(__name__)


def prepare_data_main(arguments: Sequence[str] | None = None) -> int:
    """Normalize, filter, and deduplicate an explicitly supplied local JSONL corpus."""
    parser = argparse.ArgumentParser(description="Prepare a local provenance-preserving JSONL corpus.")
    parser.add_argument("--input", type=Path, required=True, help="Local source JSONL path.")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSONL path.")
    parser.add_argument("--min-characters", type=int, default=20)
    parser.add_argument("--dedup-threshold", type=float, default=0.90)
    args = parser.parse_args(arguments)
    configure_logging()

    documents = _read_documents(args.input)
    normalized = [normalize_document(document) for document in documents]
    filter_config = FilterConfig(min_characters=args.min_characters)
    accepted = [document for document in normalized if assess_document(document, filter_config).accepted]
    deduplicated = deduplicate_documents(accepted, threshold=args.dedup_threshold)
    _write_documents(args.output, deduplicated)
    LOGGER.info(
        "data preparation completed",
        extra={
            "event": "data_prepared",
            "details": {
                "input_documents": len(documents),
                "accepted_documents": len(accepted),
                "output_documents": len(deduplicated),
                "statistics": compute_statistics(deduplicated).to_mapping(),
            },
        },
    )
    return 0


def train_cpt_main(arguments: Sequence[str] | None = None) -> int:
    """Validate and record a CPT configuration; a full trainer is intentionally deferred."""
    return _validate_training_config(arguments, load_cpt_config, "cpt_config_validated")


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
