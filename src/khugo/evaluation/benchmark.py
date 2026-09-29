"""Versioned, local JSONL benchmark schema for Khugo model evaluations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

BENCHMARK_CATEGORIES = {
    "completion",
    "qa",
    "translation_en_hyw",
    "translation_hyw_en",
    "dialect_diagnostic",
    "perplexity",
}


@dataclass(frozen=True, slots=True)
class BenchmarkExample:
    """One immutable benchmark prompt with optional references and diagnostic metadata."""

    id: str
    category: str
    prompt: str
    input_language: str
    target_language: str | None = None
    references: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.prompt.strip():
            raise ValueError("benchmark id and prompt must not be empty")
        if self.category not in BENCHMARK_CATEGORIES:
            raise ValueError(f"unsupported benchmark category: {self.category}")
        if self.category.startswith("translation_") and not self.references:
            raise ValueError("translation examples require at least one reference")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BenchmarkExample":
        references = value.get("references", [])
        metadata = value.get("metadata", {})
        if not isinstance(references, list) or not all(isinstance(item, str) for item in references):
            raise ValueError("benchmark references must be a list of strings")
        if not isinstance(metadata, Mapping):
            raise ValueError("benchmark metadata must be an object")
        return cls(
            id=str(value.get("id", "")).strip(),
            category=str(value.get("category", "")).strip(),
            prompt=str(value.get("prompt", "")),
            input_language=str(value.get("input_language", "")).strip(),
            target_language=_optional_string(value.get("target_language")),
            references=tuple(references),
            metadata=dict(metadata),
        )

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


def load_benchmark(path: Path) -> list[BenchmarkExample]:
    """Read a non-empty local benchmark file and reject duplicate example IDs."""
    if not path.is_file():
        raise FileNotFoundError(f"benchmark file not found: {path}")
    examples: list[BenchmarkExample] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    raise ValueError("benchmark line must be an object")
                examples.append(BenchmarkExample.from_mapping(payload))
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"invalid benchmark example at {path}:{line_number}: {error}") from error
    if not examples:
        raise ValueError("benchmark must contain at least one example")
    if len({example.id for example in examples}) != len(examples):
        raise ValueError("benchmark contains duplicate example IDs")
    return examples


def benchmark_sha256(path: Path) -> str:
    """Return the content fingerprint recorded in every experiment manifest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None
