"""Framework-neutral perplexity calculation helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence


def perplexity_from_token_nlls(token_negative_log_likelihoods: Sequence[float]) -> float:
    """Calculate perplexity from per-token negative log-likelihood values."""
    if not token_negative_log_likelihoods:
        raise ValueError("at least one token loss is required")
    if any(not math.isfinite(loss) for loss in token_negative_log_likelihoods):
        raise ValueError("token losses must be finite")
    return math.exp(sum(token_negative_log_likelihoods) / len(token_negative_log_likelihoods))
