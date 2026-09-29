"""Deterministic-seed utilities for repeatable research runs."""

from __future__ import annotations

import os
import random


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed available RNGs; deterministic Torch kernels may trade performance for repeatability."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
