"""Auditable, intentionally simple corpus quality filters."""

from __future__ import annotations

from dataclasses import dataclass

from .schema import TrainingDocument


@dataclass(frozen=True, slots=True)
class FilterConfig:
    min_characters: int = 20
    max_characters: int = 100_000
    min_alphabetic_ratio: float = 0.20

    def __post_init__(self) -> None:
        if self.min_characters < 0 or self.max_characters < self.min_characters:
            raise ValueError("invalid character limits")
        if not 0.0 <= self.min_alphabetic_ratio <= 1.0:
            raise ValueError("min_alphabetic_ratio must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class FilterDecision:
    accepted: bool
    reason: str | None = None


def assess_document(document: TrainingDocument, config: FilterConfig = FilterConfig()) -> FilterDecision:
    """Return an explicit keep/drop decision without silently changing text."""
    length = len(document.text)
    if length < config.min_characters:
        return FilterDecision(False, "too_short")
    if length > config.max_characters:
        return FilterDecision(False, "too_long")
    non_space = [character for character in document.text if not character.isspace()]
    alpha_ratio = sum(character.isalpha() for character in non_space) / max(len(non_space), 1)
    if alpha_ratio < config.min_alphabetic_ratio:
        return FilterDecision(False, "low_alphabetic_ratio")
    return FilterDecision(True)


def filter_documents(documents: list[TrainingDocument], config: FilterConfig = FilterConfig()) -> list[TrainingDocument]:
    """Keep documents passing ``assess_document`` in their original order."""
    return [document for document in documents if assess_document(document, config).accepted]
