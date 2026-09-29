"""Data schemas and transformations for provenance-preserving corpora."""

from .schema import SourceMetadata, TrainingDocument
from .registry import SourceRecord, load_source_registry

__all__ = ["SourceMetadata", "SourceRecord", "TrainingDocument", "load_source_registry"]
