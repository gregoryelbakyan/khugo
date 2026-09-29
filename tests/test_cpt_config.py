import pytest

from khugo.training.cpt import CPTConfig


def _config() -> dict[str, object]:
    return {
        "run_name": "test", "model_name_or_path": "local/model", "model_revision": "abc",
        "tokenizer_revision": "abc", "train_data": "train.jsonl", "validation_data": "validation.jsonl",
        "output_dir": "outputs/test", "seed": 42, "max_sequence_length": 2048,
        "learning_rate": 2e-5, "num_train_epochs": 1, "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16, "bf16": False,
    }


def test_initial_cpt_config_defaults_to_fp16_nf4_qlora() -> None:
    config = CPTConfig.from_mapping(_config())
    assert config.fp16
    assert config.use_qlora
    assert config.quantization_type == "nf4"


def test_initial_cpt_config_rejects_bf16() -> None:
    config = _config()
    config["bf16"] = True
    with pytest.raises(ValueError, match="bf16"):
        CPTConfig.from_mapping(config)
