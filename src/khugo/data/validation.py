"""Explicit language, script, and registry validation for corpus documents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .registry import SourceRecord
from .schema import TrainingDocument


@dataclass(frozen=True, slots=True)
class EligibilityDecision:
    accepted: bool
    reason: str | None = None


def armenian_script_ratio(text: str) -> float:
    """Return the fraction of alphabetic characters in the Armenian Unicode blocks."""
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return 0.0
    armenian = sum("\u0531" <= character <= "\u058f" for character in letters)
    return armenian / len(letters)


def assess_training_eligibility(
    document: TrainingDocument,
    registry: Mapping[str, SourceRecord],
    *,
    min_armenian_ratio: float = 0.50,
) -> EligibilityDecision:
    """Accept only registered, training-permitted Western Armenian documents."""
    if document.language != "hyw":
        return EligibilityDecision(False, "language_not_hyw")
    if armenian_script_ratio(document.text) < min_armenian_ratio:
        return EligibilityDecision(False, "insufficient_armenian_script")
    for source in document.sources:
        if not source.source_id:
            return EligibilityDecision(False, "missing_source_id")
        record = registry.get(source.source_id)
        if record is None:
            return EligibilityDecision(False, "unregistered_source")
        if not record.training_allowed:
            return EligibilityDecision(False, "training_not_allowed")
        if source.license != record.license:
            return EligibilityDecision(False, "license_mismatch")
    return EligibilityDecision(True)
