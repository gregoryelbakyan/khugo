# Khugo
Khugo is an open-source language-model project and ecosystem for Western Armenian. This repository keeps product/model code, reproducible training configuration, and research evaluation together.

## Mission

Khugo exists to support the preservation and living use of Western Armenian. It aims to give Western Armenian speakers, diaspora communities, and people reconnecting with their family heritage useful language technology: a model that understands and generates Western Armenian without needlessly shifting into Eastern Armenian.

The project is named in honour of the Khugoyan family. Khugo is an open, evidence-driven effort: language quality claims must come from documented evaluations and review by speakers or linguists, not from marketing language.

## Long-term direction

Khugo's first public model line is planned as **Khugo-Base** followed by **Khugo-Instruct**. The immediate milestone is a clean, reproducible Western-Armenian continued-pretraining experiment. Over time, the project aims to grow into a small open-source team, collaborate with language experts and researchers, and publish research with reproducible evidence.

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

Preparation accepts local JSONL only. Every record needs `id`, `text`, `language` (`hyw` or `hye`), and provenance. Sources are registered in `data_registry/sources.yaml`; only registered `hyw` documents whose sources explicitly allow training enter the CPT split. Deduplication retains all merged-source metadata in the output `sources` array.

```json
{"id":"example-001","text":"Բարեւ աշխարհ","language":"hyw","source":{"source_id":"example-corpus","name":"Example corpus","license":"CC BY 4.0","url":"https://example.org"}}
```

```bash
uv run khugo-prepare-data --input local/raw.jsonl --output local/train.jsonl --validation-output local/validation.jsonl --rejected-output local/rejected.jsonl --statistics-output local/statistics.json
```

No command downloads a corpus automatically. Verify each source's licence and intended use before training.

Add verified source records to `data_registry/sources.yaml` before preparation. Keep `training_allowed: false` for sources whose licence or intended use is unclear; the pipeline preserves them in its audit output but excludes them from CPT.

## Training configurations

`configs/cpt.yaml` and `configs/sft.yaml` pin model ID, local data path, seed, precision, sequence length, and output directory. The current commands validate and log them only:

```bash
uv run khugo-train-cpt --config configs/cpt.yaml
uv run khugo-train-sft --config configs/sft.yaml
```

The initial CPT profile targets the repository's GTX 1660 SUPER: 4-bit NF4 QLoRA, fp16, context length 2048, batch size 1, gradient accumulation 16, and one epoch. It trains only local, already prepared `hyw` JSONL data. Set `allow_remote_model: true` only when you intentionally want the configured base model fetched from Hugging Face.

## Reproducible baseline

The provisional benchmark fixture is versioned at `benchmarks/fixtures/baseline-v0.jsonl`; it is intentionally small and requires linguistic review before any public quality claim. Run the original model only after explicitly allowing a model download:

```bash
uv run khugo-baseline --benchmark benchmarks/fixtures/baseline-v0.jsonl --allow-remote-model
```

Each run writes an immutable directory containing `manifest.json`, `generations.jsonl`, and `metrics.json`. To evaluate a CPT adapter on the 6 GB GPU, pass its local adapter directory using `--adapter --load-in-4bit`; no model weights are uploaded by these commands.

## Development

```bash
uv run ruff check .
uv run pytest
```

Raw datasets, checkpoints, adapters, and model weights are ignored by Git. Commit code, configuration, aggregate benchmarks, and public-safe documentation only. Khugo source code is licensed under the [MIT License](LICENSE).
