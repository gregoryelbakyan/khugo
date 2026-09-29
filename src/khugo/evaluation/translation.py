"""Translation evaluation built on sacreBLEU's reproducible metric implementation."""

from __future__ import annotations

from collections.abc import Sequence


def bleu_score(hypotheses: Sequence[str], references: Sequence[Sequence[str]]) -> float:
    """Return corpus BLEU for hypotheses and one-or-more aligned reference streams."""
    if not hypotheses:
        raise ValueError("hypotheses must not be empty")
    if not references or any(len(reference) != len(hypotheses) for reference in references):
        raise ValueError("every reference stream must align with hypotheses")
    from sacrebleu import corpus_bleu

    return float(corpus_bleu(list(hypotheses), [list(reference) for reference in references]).score)


def chrf_score(hypotheses: Sequence[str], references: Sequence[Sequence[str]]) -> float:
    """Return corpus chrF, Khugo's primary translation metric."""
    if not hypotheses:
        raise ValueError("hypotheses must not be empty")
    if not references or any(len(reference) != len(hypotheses) for reference in references):
        raise ValueError("every reference stream must align with hypotheses")
    from sacrebleu import corpus_chrf

    return float(corpus_chrf(list(hypotheses), [list(reference) for reference in references]).score)
