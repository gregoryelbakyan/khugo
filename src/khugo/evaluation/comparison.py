"""Compare two completed evaluations without manufacturing missing measurements."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .artifacts import write_json


def compare_experiments(*, base_dir: Path, khugo_dir: Path, output_path: Path) -> dict[str, Any]:
    """Create a metric delta report only for matching benchmark revisions."""
    base_manifest = _read_json(base_dir / "manifest.json")
    khugo_manifest = _read_json(khugo_dir / "manifest.json")
    if base_manifest["benchmark_sha256"] != khugo_manifest["benchmark_sha256"]:
        raise ValueError("experiments must use the same benchmark revision")
    base_metrics = _read_json(base_dir / "metrics.json")
    khugo_metrics = _read_json(khugo_dir / "metrics.json")
    report = {
        "base_experiment_id": base_manifest["experiment_id"],
        "khugo_experiment_id": khugo_manifest["experiment_id"],
        "benchmark_sha256": base_manifest["benchmark_sha256"],
        "translation_delta": _translation_deltas(base_metrics, khugo_metrics),
        "perplexity": _perplexity_comparison(base_metrics, khugo_metrics),
        "notes": "Completion and QA require manual inspection of linked raw generations.",
    }
    write_json(output_path, report)
    return report


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required experiment artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"artifact must contain a JSON object: {path}")
    return payload


def _translation_deltas(base: Mapping[str, Any], khugo: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    base_translation = base.get("translation", {})
    khugo_translation = khugo.get("translation", {})
    if not isinstance(base_translation, Mapping) or not isinstance(khugo_translation, Mapping):
        return result
    for category in sorted(set(base_translation) & set(khugo_translation)):
        left, right = base_translation[category], khugo_translation[category]
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            continue
        deltas = {
            metric: float(right[metric]) - float(left[metric])
            for metric in ("chrf", "bleu")
            if metric in left and metric in right
        }
        if deltas:
            result[str(category)] = deltas
    return result


def _perplexity_comparison(base: Mapping[str, Any], khugo: Mapping[str, Any]) -> dict[str, float] | None:
    base_value, khugo_value = base.get("perplexity"), khugo.get("perplexity")
    if not isinstance(base_value, (float, int)) or not isinstance(khugo_value, (float, int)):
        return None
    return {
        "base": float(base_value),
        "khugo": float(khugo_value),
        "delta": float(khugo_value) - float(base_value),
    }
