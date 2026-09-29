"""A transparent exploratory lexical signal, not a dialect-classification claim."""

from __future__ import annotations

import re
from collections.abc import Collection

_TOKEN = re.compile(r"[^\W\d_]+", flags=re.UNICODE)


def western_armenian_lexicon_ratio(text: str, western_lexicon: Collection[str]) -> float:
    """Return the share of alphabetic tokens found in a supplied Western-Armenian lexicon.

    This is intentionally a research probe rather than a language-quality or identity metric.
    Benchmark owners must document their lexicon, sampling, and limitations.
    """
    tokens = _TOKEN.findall(text.casefold())
    if not tokens:
        return 0.0
    normalized_lexicon = {term.casefold() for term in western_lexicon}
    return sum(token in normalized_lexicon for token in tokens) / len(tokens)
