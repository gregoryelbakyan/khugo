"""Reproducible experiment manifests and JSON artifact writers."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class ExperimentManifest:
    """Metadata needed to reproduce or audit one model-evaluation run."""

    experiment_id: str
    created_at: str
    benchmark_path: str
    benchmark_sha256: str
    model_name_or_path: str
    model_revision: str
    adapter_path: str | None
    seed: int
    generation_parameters: Mapping[str, Any]
    git_revision: str | None
    environment: Mapping[str, str]

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def git_revision(repository_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={repository_root}", "rev-parse", "HEAD"],
            cwd=repository_root,
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def runtime_environment() -> dict[str, str]:
    """Collect installed package and runtime versions without triggering model downloads."""
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("torch", "transformers", "peft", "bitsandbytes", "datasets", "accelerate"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
