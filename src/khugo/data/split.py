"""Stable train/validation splitting without random state hidden in process memory."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from .schema import TrainingDocument


def train_validation_split(
    documents: Sequence[TrainingDocument], *, validation_ratio: float, seed: int
) -> tuple[list[TrainingDocument], list[TrainingDocument]]:
    """Split by a SHA-256 score derived from the document ID and explicit seed."""
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    threshold = int(validation_ratio * (2**64))
    train: list[TrainingDocument] = []
    validation: list[TrainingDocument] = []
    for document in documents:
        digest = hashlib.sha256(f"{seed}:{document.id}".encode("utf-8")).digest()
        score = int.from_bytes(digest[:8], "big")
        (validation if score < threshold else train).append(document)
    return train, validation
