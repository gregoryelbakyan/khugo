"""JSONL-compatible document and source-provenance schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class SourceMetadata:
    """Provenance required to assess a document's training eligibility."""

    name: str
    license: str
    url: str | None = None
    retrieved_at: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SourceMetadata":
        name = str(value.get("name", "")).strip()
        license_name = str(value.get("license", "")).strip()
        if not name or not license_name:
            raise ValueError("source metadata requires non-empty 'name' and 'license'")
        return cls(
            name=name,
            license=license_name,
            url=_optional_string(value.get("url")),
            retrieved_at=_optional_string(value.get("retrieved_at")),
        )

    def to_mapping(self) -> dict[str, str]:
        result = {"name": self.name, "license": self.license}
        if self.url:
            result["url"] = self.url
        if self.retrieved_at:
            result["retrieved_at"] = self.retrieved_at
        return result


@dataclass(frozen=True, slots=True)
class TrainingDocument:
    """A training document whose one-or-more sources travel with its text."""

    id: str
    text: str
    sources: tuple[SourceMetadata, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("document id must not be empty")
        if not self.sources:
            raise ValueError("every training document requires source metadata")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TrainingDocument":
        document_id = str(value.get("id", "")).strip()
        text = value.get("text")
        if not isinstance(text, str):
            raise ValueError("document text must be a string")
        raw_sources = value.get("sources")
        if raw_sources is None:
            raw_source = value.get("source")
            raw_sources = [raw_source] if raw_source is not None else []
        if not isinstance(raw_sources, list):
            raise ValueError("document sources must be a list or a source object")
        sources = tuple(SourceMetadata.from_mapping(item) for item in raw_sources if isinstance(item, Mapping))
        if len(sources) != len(raw_sources):
            raise ValueError("every source must be an object")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("document metadata must be an object")
        return cls(id=document_id, text=text, sources=sources, metadata=dict(metadata))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "sources": [source.to_mapping() for source in self.sources],
            "metadata": dict(self.metadata),
        }


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
