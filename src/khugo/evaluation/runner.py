"""Explicit model-loading baseline runner; importing this module never downloads a model."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .artifacts import ExperimentManifest, git_revision, runtime_environment, utc_timestamp, write_json, write_jsonl
from .benchmark import BenchmarkExample, benchmark_sha256, load_benchmark
from .translation import bleu_score, chrf_score
from .westernness import western_armenian_lexicon_ratio
from ..reproducibility import seed_everything


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    model_name_or_path: str = "Qwen/Qwen3-1.7B-Base"
    model_revision: str = "main"
    seed: int = 42
    max_new_tokens: int = 256
    temperature: float = 0.0
    allow_remote_model: bool = False
    adapter_path: str | None = None
    load_in_4bit: bool = False

    def __post_init__(self) -> None:
        if not self.model_name_or_path.strip() or not self.model_revision.strip():
            raise ValueError("model name/path and revision must not be empty")
        if self.seed < 0 or self.max_new_tokens <= 0 or self.temperature < 0:
            raise ValueError("invalid generation configuration")

    def generation_parameters(self) -> dict[str, Any]:
        return {
            "do_sample": self.temperature > 0,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "load_in_4bit": self.load_in_4bit,
        }


def run_baseline(
    *, benchmark_path: Path, output_root: Path, config: BaselineConfig, repository_root: Path
) -> Path:
    """Generate a reproducible baseline result from an explicit local/remote model request."""
    examples = load_benchmark(benchmark_path)
    benchmark_hash = benchmark_sha256(benchmark_path)
    experiment_id = _experiment_id(benchmark_hash, config)
    output_dir = output_root / experiment_id
    if output_dir.exists():
        raise FileExistsError(f"baseline output already exists: {output_dir}")

    seed_everything(config.seed)
    tokenizer, model = _load_model_and_tokenizer(config)
    output_dir.mkdir(parents=True)
    generations = [_generate(example, tokenizer, model, config) for example in examples]
    metrics = _compute_metrics(examples, generations)
    metrics.update(_compute_perplexity(examples, tokenizer, model))
    manifest = ExperimentManifest(
        experiment_id=experiment_id,
        created_at=utc_timestamp(),
        benchmark_path=str(benchmark_path),
        benchmark_sha256=benchmark_hash,
        model_name_or_path=config.model_name_or_path,
        model_revision=config.model_revision,
        adapter_path=config.adapter_path,
        seed=config.seed,
        generation_parameters=config.generation_parameters(),
        git_revision=git_revision(repository_root),
        environment=runtime_environment(),
    )
    write_json(output_dir / "manifest.json", manifest.to_mapping())
    write_jsonl(output_dir / "generations.jsonl", generations)
    write_json(output_dir / "metrics.json", metrics)
    return output_dir


def _load_model_and_tokenizer(config: BaselineConfig) -> tuple[Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        raise RuntimeError("install Khugo's model dependencies before running a baseline") from error
    local_files_only = not config.allow_remote_model
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        revision=config.model_revision,
        local_files_only=local_files_only,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, Any] = {
        "revision": config.model_revision,
        "local_files_only": local_files_only,
        "trust_remote_code": False,
        "torch_dtype": "auto",
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    if config.load_in_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit model loading requires a CUDA GPU")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        **model_kwargs,
    )
    model.eval()
    if config.adapter_path:
        try:
            from peft import PeftModel
        except ImportError as error:
            raise RuntimeError("install peft to evaluate a Khugo adapter") from error
        adapter = Path(config.adapter_path)
        if not adapter.is_dir():
            raise FileNotFoundError(f"adapter directory not found: {adapter}")
        model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
    return tokenizer, model


def _generate(example: BenchmarkExample, tokenizer: Any, model: Any, config: BaselineConfig) -> dict[str, Any]:
    import torch

    encoded = tokenizer(example.prompt, return_tensors="pt")
    device = next(model.parameters()).device
    encoded = {name: value.to(device) for name, value in encoded.items()}
    with torch.inference_mode():
        generated = model.generate(**encoded, **config.generation_parameters())
    completion_tokens = generated[0, encoded["input_ids"].shape[-1] :]
    completion = tokenizer.decode(completion_tokens, skip_special_tokens=True)
    return {
        "id": example.id,
        "category": example.category,
        "prompt": example.prompt,
        "completion": completion,
        "input_language": example.input_language,
        "target_language": example.target_language,
        "references": list(example.references),
        "metadata": dict(example.metadata),
    }


def _compute_metrics(
    examples: list[BenchmarkExample], generations: list[Mapping[str, Any]]
) -> dict[str, Any]:
    by_id = {example.id: example for example in examples}
    translated: dict[str, list[str]] = {"translation_en_hyw": [], "translation_hyw_en": []}
    references: dict[str, list[list[str]]] = {"translation_en_hyw": [], "translation_hyw_en": []}
    diagnostics: list[dict[str, float]] = []
    for row in generations:
        example = by_id[row["id"]]
        if example.category in translated:
            translated[example.category].append(str(row["completion"]))
            references[example.category].append(list(example.references))
        if example.category == "dialect_diagnostic":
            western = example.metadata.get("western_markers", [])
            eastern = example.metadata.get("eastern_markers", [])
            if isinstance(western, list) and isinstance(eastern, list):
                diagnostics.append(
                    {
                        "western_marker_ratio": western_armenian_lexicon_ratio(str(row["completion"]), western),
                        "eastern_marker_ratio": western_armenian_lexicon_ratio(str(row["completion"]), eastern),
                    }
                )
    translation_metrics: dict[str, dict[str, float]] = {}
    for category, hypotheses in translated.items():
        if not hypotheses:
            continue
        max_references = max(len(items) for items in references[category])
        aligned_references = [
            [items[index] if index < len(items) else items[0] for items in references[category]]
            for index in range(max_references)
        ]
        translation_metrics[category] = {
            "chrf": chrf_score(hypotheses, aligned_references),
            "bleu": bleu_score(hypotheses, aligned_references),
        }
    return {
        "translation": translation_metrics,
        "dialect_diagnostic": {
            "examples": len(diagnostics),
            "mean_western_marker_ratio": _mean([item["western_marker_ratio"] for item in diagnostics]),
            "mean_eastern_marker_ratio": _mean([item["eastern_marker_ratio"] for item in diagnostics]),
            "disclaimer": "Exploratory lexical signal only; not a dialect-classification result.",
        },
        "manual_review_categories": ["completion", "qa"],
    }


def _compute_perplexity(examples: list[BenchmarkExample], tokenizer: Any, model: Any) -> dict[str, Any]:
    texts = [example.prompt for example in examples if example.category == "perplexity"]
    if not texts:
        return {}
    import math
    import torch

    losses: list[float] = []
    device = next(model.parameters()).device
    for text in texts:
        encoded = tokenizer(text, return_tensors="pt", truncation=True)
        encoded = {name: value.to(device) for name, value in encoded.items()}
        if encoded["input_ids"].shape[-1] < 2:
            continue
        with torch.inference_mode():
            result = model(**encoded, labels=encoded["input_ids"])
        losses.append(float(result.loss))
    if not losses:
        return {"perplexity": None}
    return {"perplexity": math.exp(sum(losses) / len(losses))}


def _experiment_id(benchmark_hash: str, config: BaselineConfig) -> str:
    payload = f"{benchmark_hash}:{config.model_name_or_path}:{config.model_revision}:{config.adapter_path}:{config.load_in_4bit}:{config.seed}:{config.max_new_tokens}:{config.temperature}"
    return f"baseline-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]}"


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
