"""Opt-in boilerplate removal hooks; source text is never changed implicitly."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

from .schema import TrainingDocument

BoilerplateHook = Callable[[str], str]


def apply_boilerplate_hooks(
    document: TrainingDocument, hooks: Sequence[BoilerplateHook]
) -> TrainingDocument:
    """Apply explicitly supplied hooks and record that text transformation occurred."""
    text = document.text
    for hook in hooks:
        text = hook(text)
    if text == document.text:
        return document
    metadata = {**document.metadata, "boilerplate_hooks_applied": len(hooks)}
    return replace(document, text=text, metadata=metadata)
