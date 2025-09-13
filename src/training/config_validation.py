from typing import Any, Dict, List

from pydantic import BaseModel, Field, ValidationError, ConfigDict, field_validator, model_validator


class ConfigValidationError(Exception):
    """Raised when a LoRA configuration is invalid."""


class LoraConfigModel(BaseModel):
    r: int
    lora_alpha: int
    target_modules: List[str]
    lora_dropout: float = Field(0.0, ge=0.0, le=1.0)

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    def handle_aliases(cls, values: Any) -> Any:
        if isinstance(values, dict):
            if "dropout" in values and "lora_dropout" not in values:
                values["lora_dropout"] = values.pop("dropout")
        return values

    @field_validator("r", "lora_alpha")
    def check_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be positive")
        return v

    @field_validator("target_modules")
    def check_modules(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("target_modules must not be empty")
        return v


def validate_lora_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a LoRA configuration dictionary.

    Args:
        config: The dictionary loaded from YAML/JSON.

    Returns:
        The validated configuration dictionary with normalized fields.

    Raises:
        ConfigValidationError: If validation fails.
    """
    try:
        model = LoraConfigModel(**config)
    except ValidationError as e:
        raise ConfigValidationError(str(e)) from e
    return model.model_dump()
