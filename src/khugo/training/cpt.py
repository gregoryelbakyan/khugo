"""Continued-pretraining configuration, without a trainer implementation yet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import RunConfig, load_mapping


@dataclass(frozen=True, slots=True)
class CPTConfig(RunConfig):
    """Reproducible parameters for a continued-pretraining run."""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CPTConfig":
        return cls(**_shared_fields(value))


def load_cpt_config(path: Path) -> CPTConfig:
    """Load and validate a local CPT configuration."""
    return CPTConfig.from_mapping(load_mapping(path))


def _shared_fields(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = set(RunConfig.__dataclass_fields__)
    unexpected = set(value) - fields
    if unexpected:
        raise ValueError(f"unexpected CPT configuration keys: {sorted(unexpected)}")
    missing = fields - set(value)
    if missing:
        raise ValueError(f"missing CPT configuration keys: {sorted(missing)}")
    return {name: value[name] for name in fields}
