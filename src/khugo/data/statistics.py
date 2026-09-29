"""Corpus-level statistics for data review and experiment reporting."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable

from .schema import TrainingDocument


@dataclass(frozen=True, slots=True)
class CorpusStatistics:
    document_count: int
    character_count: int
    mean_characters: float
    source_count: int
    documents_by_license: dict[str, int]

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


def compute_statistics(documents: Iterable[TrainingDocument]) -> CorpusStatistics:
    """Compute aggregate information without dropping document provenance."""
    materialized = list(documents)
    character_count = sum(len(document.text) for document in materialized)
    licenses = Counter(
        source.license for document in materialized for source in document.sources
    )
    source_names = {source.name for document in materialized for source in document.sources}
    return CorpusStatistics(
        document_count=len(materialized),
        character_count=character_count,
        mean_characters=character_count / len(materialized) if materialized else 0.0,
        source_count=len(source_names),
        documents_by_license=dict(sorted(licenses.items())),
    )
