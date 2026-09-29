# Khugo research brief

## Why Khugo

Western Armenian is a heritage language for communities around the world. Khugo is an open-source effort to support its preservation and active use through language technology. It is named in honour of the Khugoyan family.

The project is intended for Western Armenian speakers and for people reconnecting with the language of their families. Its first responsibility is not to make broad performance claims, but to create a reproducible and legally careful path toward a useful public model.

## Initial research question

Can parameter-efficient continued pretraining on clean, provenance-preserving Western Armenian (`hyw`) text improve a base language model's Western Armenian capability while avoiding unnecessary interference from Eastern Armenian (`hye`)?

## Method

1. Evaluate the original `Qwen/Qwen3-1.7B-Base` on a versioned Western Armenian benchmark.
2. Prepare local, licence-reviewed `hyw` data while preserving source provenance and excluding uncertain sources from training.
3. Run a reproducible 4-bit NF4 QLoRA continued-pretraining experiment.
4. Evaluate the adapted model on exactly the same benchmark and compare metrics, raw generations, and `hyw`/`hye` diagnostics.

The initial benchmark is deliberately small and provisional. It requires review by Western Armenian speakers or linguists before supporting any external quality claim.

## Evidence and safeguards

- Every training document retains source and licence metadata.
- Unknown or disallowed sources cannot enter CPT by default.
- Experiment artifacts record the model revision, seed, generation parameters, environment, data statistics, logs, and checkpoints.
- Raw generations accompany automatic metrics for manual inspection.
- No datasets are downloaded automatically and no model is published automatically.

## Collaboration goal

Khugo is being developed as a long-term open-source project led by its founder, with the aim of building a small contributor team and collaborating with academic language-technology researchers. A future paper should be led by evidence from this repository and credit all contributors accurately.
