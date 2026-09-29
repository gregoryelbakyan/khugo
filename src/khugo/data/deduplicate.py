"""Deterministic exact and near-duplicate detection with provenance retention."""

from __future__ import annotations

from dataclasses import replace

from datasketch import MinHash, MinHashLSH

from .schema import SourceMetadata, TrainingDocument


def deduplicate_documents(
    documents: list[TrainingDocument], *, threshold: float = 0.90, num_perm: int = 128
) -> list[TrainingDocument]:
    """Merge exact or token-near duplicates while retaining all source metadata.

    Processing order determines the canonical document, making the outcome reproducible for a
    fixed input order and parameters. ``datasketch`` is seeded explicitly for stable MinHash
    signatures across runs.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    if num_perm <= 0:
        raise ValueError("num_perm must be positive")

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    retained: list[TrainingDocument] = []
    exact_index: dict[str, int] = {}
    lsh_index: dict[str, int] = {}

    for document in documents:
        fingerprint = document.text.casefold().strip()
        candidate_index = exact_index.get(fingerprint)
        signature = _minhash(fingerprint, num_perm)
        if candidate_index is None:
            candidates = [lsh_index[key] for key in lsh.query(signature)]
            candidate_index = min(candidates) if candidates else None

        if candidate_index is None:
            key = f"document-{len(retained)}"
            lsh.insert(key, signature)
            lsh_index[key] = len(retained)
            exact_index[fingerprint] = len(retained)
            retained.append(document)
            continue

        canonical = retained[candidate_index]
        merged_sources = _merge_sources(canonical.sources, document.sources)
        duplicate_ids = list(canonical.metadata.get("deduplicated_ids", []))
        duplicate_ids.append(document.id)
        metadata = {**canonical.metadata, "deduplicated_ids": duplicate_ids}
        retained[candidate_index] = replace(canonical, sources=merged_sources, metadata=metadata)

    return retained


def _minhash(text: str, num_perm: int) -> MinHash:
    signature = MinHash(num_perm=num_perm, seed=42)
    for token in sorted(set(text.split())):
        signature.update(token.encode("utf-8"))
    return signature


def _merge_sources(
    existing: tuple[SourceMetadata, ...], incoming: tuple[SourceMetadata, ...]
) -> tuple[SourceMetadata, ...]:
    return tuple(dict.fromkeys((*existing, *incoming)))
