from pathlib import Path

import pytest

from khugo.evaluation.benchmark import BenchmarkExample, benchmark_sha256, load_benchmark
from khugo.evaluation.runner import BaselineConfig


def test_benchmark_requires_translation_references() -> None:
    with pytest.raises(ValueError, match="require at least one reference"):
        BenchmarkExample(
            id="translation", category="translation_en_hyw", prompt="Translate.", input_language="en"
        )


def test_benchmark_hash_is_content_stable(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.jsonl"
    path.write_text(
        '{"id":"qa","category":"qa","prompt":"Question","input_language":"hyw"}\n',
        encoding="utf-8",
    )
    assert len(load_benchmark(path)) == 1
    assert benchmark_sha256(path) == benchmark_sha256(path)


def test_baseline_config_does_not_allow_remote_loading_by_default() -> None:
    assert not BaselineConfig().allow_remote_model
