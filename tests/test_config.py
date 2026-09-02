import pytest

from trainer_zero import TrainConfig


def test_unknown_config_values_are_preserved():
    config = TrainConfig.from_mapping({"output_dir": "x", "custom_value": 3})
    assert config.output_dir == "x"
    assert config.extra["custom_value"] == 3


def test_accumulation_must_be_positive():
    with pytest.raises(ValueError):
        TrainConfig(gradient_accumulation_steps=0)
