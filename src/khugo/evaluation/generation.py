"""Serializable generation records for product checks and research review."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    """One generated response with the prompt and inference settings that produced it."""

    prompt_id: str
    prompt: str
    completion: str
    model_id: str
    generation_config: Mapping[str, Any]

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)
