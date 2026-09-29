"""Supervised fine-tuning configuration, without a trainer implementation yet."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import RunConfig, load_mapping


@dataclass(frozen=True, slots=True)
class SFTConfig(RunConfig):
    """Reproducible parameters for a future supervised fine-tuning run."""

    assistant_only_loss: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SFTConfig":
        fields = set(cls.__dataclass_fields__)
        unexpected = set(value) - fields
        if unexpected:
            raise ValueError(f"unexpected SFT configuration keys: {sorted(unexpected)}")
        missing = fields - set(value)
        if missing:
            raise ValueError(f"missing SFT configuration keys: {sorted(missing)}")
        return cls(**{name: value[name] for name in fields})

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


def load_sft_config(path: Path) -> SFTConfig:
    """Load and validate a local SFT configuration."""
    return SFTConfig.from_mapping(load_mapping(path))
