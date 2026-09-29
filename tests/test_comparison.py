import json
from pathlib import Path

import pytest

from khugo.evaluation.comparison import compare_experiments


def _experiment(path: Path, experiment_id: str, benchmark_hash: str, chrf: float) -> None:
    path.mkdir()
    (path / "manifest.json").write_text(
        json.dumps({"experiment_id": experiment_id, "benchmark_sha256": benchmark_hash}), encoding="utf-8"
    )
    (path / "metrics.json").write_text(
        json.dumps({"translation": {"translation_en_hyw": {"chrf": chrf, "bleu": chrf / 2}}}),
        encoding="utf-8",
    )


def test_comparison_calculates_only_matching_benchmark_deltas(tmp_path: Path) -> None:
    base, khugo = tmp_path / "base", tmp_path / "khugo"
    _experiment(base, "base", "same", 20.0)
    _experiment(khugo, "khugo", "same", 24.0)

    report = compare_experiments(base_dir=base, khugo_dir=khugo, output_path=tmp_path / "report.json")

    assert report["translation_delta"]["translation_en_hyw"]["chrf"] == 4.0


def test_comparison_rejects_mismatched_benchmarks(tmp_path: Path) -> None:
    base, khugo = tmp_path / "base", tmp_path / "khugo"
    _experiment(base, "base", "one", 20.0)
    _experiment(khugo, "khugo", "two", 24.0)
    with pytest.raises(ValueError, match="same benchmark"):
        compare_experiments(base_dir=base, khugo_dir=khugo, output_path=tmp_path / "report.json")
