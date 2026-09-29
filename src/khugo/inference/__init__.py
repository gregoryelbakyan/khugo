"""Stable product-facing inference request interfaces."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """Validated, backend-independent generation input for future serving code."""

    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.7

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt must not be empty")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
