"""Version-controlled source registry and training-eligibility rules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True, slots=True)
class SourceRecord:
    """Legal and provenance metadata for one corpus source."""

    source_id: str
    source_name: str
    url: str
    author: str | None
    license: str
    copyright_status: str
    retrieved_at: str | None
    training_allowed: bool
    redistribution_allowed: bool
    notes: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SourceRecord":
        required = ("source_id", "source_name", "url", "license", "copyright_status")
        missing = [key for key in required if not str(value.get(key, "")).strip()]
        if missing:
            raise ValueError(f"source record missing required fields: {missing}")
        for key in ("training_allowed", "redistribution_allowed"):
            if not isinstance(value.get(key), bool):
                raise ValueError(f"source record field '{key}' must be a boolean")
        return cls(
            source_id=str(value["source_id"]).strip(),
            source_name=str(value["source_name"]).strip(),
            url=str(value["url"]).strip(),
            author=_optional_string(value.get("author")),
            license=str(value["license"]).strip(),
            copyright_status=str(value["copyright_status"]).strip(),
            retrieved_at=_optional_string(value.get("retrieved_at")),
            training_allowed=value["training_allowed"],
            redistribution_allowed=value["redistribution_allowed"],
            notes=_optional_string(value.get("notes")),
        )


def load_source_registry(path: Path) -> dict[str, SourceRecord]:
    """Load a local registry and reject duplicate source identifiers."""
    if not path.is_file():
        raise FileNotFoundError(f"source registry not found: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("sources"), list):
        raise ValueError("source registry must contain a 'sources' list")
    records = [SourceRecord.from_mapping(item) for item in payload["sources"] if isinstance(item, Mapping)]
    if len(records) != len(payload["sources"]):
        raise ValueError("every source registry entry must be an object")
    result = {record.source_id: record for record in records}
    if len(result) != len(records):
        raise ValueError("source registry contains duplicate source_id values")
    return result


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
