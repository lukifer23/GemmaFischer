import pytest
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.config_validation import (
    ConfigValidationError,
    validate_lora_config,
)


def test_validate_lora_config_valid():
    cfg = {
        "r": 8,
        "lora_alpha": 16,
        "target_modules": ["q_proj", "k_proj"],
        "dropout": 0.1,
    }
    result = validate_lora_config(cfg)
    assert result["r"] == 8
    assert result["lora_alpha"] == 16
    assert result["lora_dropout"] == 0.1


@pytest.mark.parametrize(
    "cfg",
    [
        {"r": -1, "lora_alpha": 16, "target_modules": ["q_proj"]},
        {"r": 8, "lora_alpha": 0, "target_modules": ["q_proj"]},
        {"r": 8, "lora_alpha": 16, "target_modules": [], "lora_dropout": 0.1},
        {"r": 8, "lora_alpha": 16, "target_modules": ["q_proj"], "lora_dropout": 1.5},
    ],
)
def test_validate_lora_config_invalid(cfg):
    with pytest.raises(ConfigValidationError):
        validate_lora_config(cfg)
