# Khugo

Khugo is an open-source language-model project and ecosystem for Western Armenian. This repository keeps product/model code, reproducible training configuration, and research evaluation together.

## Status

This initial scaffold provides tested local-data processing, configuration validation, and small evaluation primitives. It does **not** download datasets or implement a full training loop yet.

## Setup

Khugo requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run pytest
```

## Layout

- `src/khugo/data/`: normalization, filtering, deduplication, and corpus statistics.
- `src/khugo/training/`: reproducible CPT and SFT configuration interfaces.
- `src/khugo/evaluation/`: perplexity, translation, generation, and Western-Armenian research metrics.
- `src/khugo/inference/`: product-facing inference request interfaces.
- `configs/`: version-controlled experiment configurations.
- `benchmarks/` and `experiments/`: research assets and ignored local outputs.
- `data/`: ignored local data storage; never commit corpora here.

## Data contract and provenance

Preparation accepts local JSONL only. Every record needs `id`, `text`, and a `source` object with `name` and `license`; `url` and `retrieved_at` are encouraged. Deduplication retains all merged-source metadata in the output `sources` array.

```json
{"id":"example-001","text":"Բարեւ աշխարհ","source":{"name":"Example corpus","license":"CC BY 4.0","url":"https://example.org"}}
```

```bash
uv run khugo-prepare-data --input local/raw.jsonl --output local/clean.jsonl
```

No command downloads a corpus automatically. Verify each source's licence and intended use before training.

## Training configurations

`configs/cpt.yaml` and `configs/sft.yaml` pin model ID, local data path, seed, precision, sequence length, and output directory. The current commands validate and log them only:

```bash
uv run khugo-train-cpt --config configs/cpt.yaml
uv run khugo-train-sft --config configs/sft.yaml
```

## Development

```bash
uv run ruff check .
uv run pytest
```

Raw datasets, checkpoints, adapters, and model weights are ignored by Git. Commit code, configuration, aggregate benchmarks, and public-safe documentation only. Choose and add a repository licence before the first public release.
