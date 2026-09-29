"""Shared strict configuration loading for training commands."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Settings required to make a training experiment traceable and repeatable."""

    run_name: str
    model_name_or_path: str
    train_data: str
    output_dir: str
    seed: int
    max_sequence_length: int
    learning_rate: float
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    bf16: bool

    def __post_init__(self) -> None:
        required_names = (self.run_name, self.model_name_or_path, self.train_data, self.output_dir)
        if not all(value.strip() for value in required_names):
            raise ValueError("run_name, model_name_or_path, train_data, and output_dir are required")
        if self.seed < 0 or self.max_sequence_length <= 0:
            raise ValueError("seed must be non-negative and max_sequence_length must be positive")
        if self.learning_rate <= 0 or self.num_train_epochs <= 0:
            raise ValueError("learning_rate and num_train_epochs must be positive")
        if self.per_device_train_batch_size <= 0 or self.gradient_accumulation_steps <= 0:
            raise ValueError("batch size and gradient accumulation steps must be positive")

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


def load_mapping(path: Path) -> Mapping[str, Any]:
    """Load a non-empty YAML mapping from an explicit, local configuration file."""
    if not path.is_file():
        raise FileNotFoundError(f"configuration file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, Mapping):
        raise ValueError("configuration must be a YAML mapping")
    return loaded
