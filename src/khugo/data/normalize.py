"""Conservative Unicode and whitespace normalization for corpus text."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import replace

from .schema import TrainingDocument

_INLINE_WHITESPACE = re.compile(r"[^\S\r\n]+")
_EXCESS_NEWLINES = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Apply NFC and stable whitespace cleanup without altering Armenian orthography."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    normalized = _INLINE_WHITESPACE.sub(" ", normalized)
    normalized = _EXCESS_NEWLINES.sub("\n\n", normalized)
    return normalized.strip()


def normalize_document(document: TrainingDocument) -> TrainingDocument:
    """Return a provenance-preserving copy with normalized text."""
    return replace(document, text=normalize_text(document.text))
