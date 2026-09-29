"""Research evaluation interfaces that live alongside model/product code."""

from .perplexity import perplexity_from_token_nlls
from .translation import bleu_score, chrf_score
from .westernness import western_armenian_lexicon_ratio

__all__ = ["bleu_score", "chrf_score", "perplexity_from_token_nlls", "western_armenian_lexicon_ratio"]
