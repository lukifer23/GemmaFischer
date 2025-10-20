#!/usr/bin/env python3
"""Simplified configuration management for ChessGemma using Pydantic settings.

Replaces the overly complex configuration system with a clean, validated,
and maintainable approach using Pydantic BaseSettings.
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import logging

from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """Model configuration settings."""
    pretrained_model_path: str = Field(default="google/gemma-3-270m", description="HuggingFace model ID or local path")
    local_model_path: Optional[str] = Field(default=None, description="Local model directory path")
    model_id_env: str = Field(default="CHESSGEMMA_MODEL_ID", description="Environment variable for model ID")
    model_path_env: str = Field(default="CHESSGEMMA_MODEL_PATH", description="Environment variable for model path")

    @validator('local_model_path')
    def resolve_model_path(cls, v, values):
        """Resolve model path relative to project root."""
        if v:
            project_root = Path(__file__).resolve().parents[2]
            return str(project_root / v)
        return v


class ChessEngineConfig(BaseModel):
    """Chess engine configuration."""
    primary: str = Field(default="lc0", description="Primary chess engine")
    lc0: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "engine_path": "/opt/homebrew/bin/lc0",
        "weights_file": "models/lc0_weights/network.pb.gz",
        "backend": "metal",
        "threads": 4,
        "nn_cache_size": 262144,
        "time_limit": 1.5,
        "search_paths": ["/opt/homebrew/bin/lc0", "/usr/local/bin/lc0", "/usr/bin/lc0", "lc0"]
    })
    fallback: Dict[str, Any] = Field(default_factory=lambda: {
        "engine_path": "/opt/homebrew/bin/stockfish",
        "threads": 4,
        "hash": 256,
        "skill_level": 20,
        "show_wdl": True,
        "depth": 16,
        "time_limit": 0.8,
        "search_paths": ["/opt/homebrew/bin/stockfish", "/usr/local/bin/stockfish", "/usr/bin/stockfish", "/usr/games/stockfish", "stockfish"]
    })

    @validator('lc0')
    def resolve_lc0_paths(cls, v):
        """Resolve LC0 paths relative to project root."""
        if "weights_file" in v and v["weights_file"]:
            project_root = Path(__file__).resolve().parents[2]
            v["weights_file"] = str(project_root / v["weights_file"])
        return v


class TrainingConfig(BaseModel):
    """Training configuration settings."""
    output_dir: str = Field(default="checkpoints/lora_default")
    per_device_train_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    max_steps: int = Field(default=1200, ge=1)
    learning_rate: float = Field(default=1.6e-4, gt=0)
    num_train_epochs: int = Field(default=1, ge=1)
    warmup_steps: int = Field(default=120, ge=0)
    weight_decay: float = Field(default=0.01, ge=0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    logging_steps: int = Field(default=25, ge=1)
    save_steps: int = Field(default=150, ge=1)
    eval_strategy: str = Field(default="steps")
    eval_steps: int = Field(default=150, ge=1)
    save_strategy: str = Field(default="steps")
    load_best_model_at_end: bool = Field(default=True)
    metric_for_best_model: str = Field(default="eval_loss")
    greater_is_better: bool = Field(default=False)
    lr_scheduler_type: str = Field(default="cosine")
    fp16: bool = Field(default=False)
    bf16: bool = Field(default=False)
    remove_unused_columns: bool = Field(default=False)
    dataloader_pin_memory: bool = Field(default=False)
    dataloader_num_workers: int = Field(default=0, ge=0)

    @validator('output_dir')
    def resolve_output_dir(cls, v):
        """Resolve output directory relative to project root."""
        if not Path(v).is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            return str(project_root / v)
        return v


class LoRAConfig(BaseModel):
    """LoRA configuration settings."""
    r: int = Field(default=32, ge=1)
    lora_alpha: int = Field(default=64, ge=1)
    target_modules: List[str] = Field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"])
    dropout: float = Field(default=0.05, ge=0, le=1)
    bias: str = Field(default="none")
    task_type: str = Field(default="CAUSAL_LM")


class InferenceConfig(BaseModel):
    """Inference configuration settings."""
    max_new_tokens: int = Field(default=200, ge=1)
    temperature: float = Field(default=0.7, gt=0)
    top_p: float = Field(default=0.9, gt=0, le=1)
    do_sample: bool = Field(default=True)
    repetition_penalty: float = Field(default=1.1, ge=1)
    use_cache: bool = Field(default=True)
    pad_token_id: Optional[int] = Field(default=None)
    eos_token_id: Optional[int] = Field(default=None)

    # Expert-specific overrides
    engine_temperature: float = Field(default=0.0, ge=0)
    engine_top_p: float = Field(default=1.0, gt=0, le=1)
    engine_do_sample: bool = Field(default=False)
    engine_max_new_tokens: int = Field(default=8, ge=1)

    tutor_temperature: float = Field(default=0.7, gt=0)
    tutor_top_p: float = Field(default=0.9, gt=0, le=1)
    tutor_do_sample: bool = Field(default=True)
    tutor_max_new_tokens: int = Field(default=150, ge=1)

    director_temperature: float = Field(default=0.6, gt=0)
    director_top_p: float = Field(default=0.9, gt=0, le=1)
    director_do_sample: bool = Field(default=True)
    director_max_new_tokens: int = Field(default=200, ge=1)

    # Chess-aware decoding settings
    chess_aware_decoding: bool = Field(default=True)
    tutor_chess_aware_decoding: bool = Field(default=True)


class CacheConfig(BaseModel):
    """Cache configuration settings."""
    max_cache_size: int = Field(default=1024, ge=1)
    engine_cache_max: int = Field(default=2048, ge=1)
    cache_ttl_seconds: int = Field(default=3600, ge=1)
    enable_response_cache: bool = Field(default=True)
    enable_engine_cache: bool = Field(default=True)
    enable_kv_cache: bool = Field(default=True)


class SystemConfig(BaseModel):
    """System configuration settings."""
    debug_mode: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    device: str = Field(default="auto")
    seed: int = Field(default=42, ge=0)
    max_workers: int = Field(default=4, ge=1)
    timeout_minutes: int = Field(default=300, ge=1)
    memory_limit_gb: float = Field(default=16.0, gt=0)


class MoEConfig(BaseModel):
    """Mixture of Experts configuration."""
    enabled: bool = Field(default=True)
    router_checkpoint_path: Optional[str] = Field(default=None)
    router_learning_rate: float = Field(default=1.0e-3, gt=0)
    router_hidden_dim: int = Field(default=128, ge=1)
    router_training_steps: int = Field(default=1000, ge=1)
    ensemble_weight_threshold: float = Field(default=0.3, ge=0, le=1)
    adaptive_routing_enabled: bool = Field(default=True)
    performance_tracking_enabled: bool = Field(default=True)
    retraining_threshold: float = Field(default=0.1, ge=0, le=1)


class PerformanceConfig(BaseModel):
    """Performance configuration settings."""
    enable_performance_monitoring: bool = Field(default=True)
    metrics_collection_interval: int = Field(default=30, ge=1)
    slow_query_threshold: float = Field(default=1.5, gt=0)
    memory_monitoring_enabled: bool = Field(default=True)
    cache_performance_tracking: bool = Field(default=True)
    model_loading_optimization: bool = Field(default=True)
    batch_processing_enabled: bool = Field(default=True)
    parallel_inference_enabled: bool = Field(default=True)
    tokenizer_batch_size: int = Field(default=8, ge=1)
    inference_queue_size: int = Field(default=150, ge=1)


class ChessGemmaConfig(BaseSettings):
    """Unified configuration for ChessGemma with environment variable support."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    chess_engine: ChessEngineConfig = Field(default_factory=ChessEngineConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    moe: MoEConfig = Field(default_factory=MoEConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)

    # Expert-specific configurations
    experts: Dict[str, TrainingConfig] = Field(default_factory=lambda: {
        "uci": TrainingConfig(
            output_dir="checkpoints/lora_uci",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=12,
            max_steps=1600,
            learning_rate=2.0e-4,
            warmup_steps=160,
            logging_steps=20,
            save_steps=200,
            eval_steps=200
        ),
        "tutor": TrainingConfig(
            output_dir="checkpoints/lora_tutor",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            max_steps=1800,
            learning_rate=1.6e-4,
            warmup_steps=180,
            logging_steps=40,
            save_steps=200,
            eval_steps=200
        ),
        "director": TrainingConfig(
            output_dir="checkpoints/lora_director",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            max_steps=1600,
            learning_rate=1.6e-4,
            warmup_steps=160,
            logging_steps=40,
            save_steps=200,
            eval_steps=200
        )
    })

    class Config:
        env_prefix = "CHESSGEMMA_"
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Resolve model path from environment
        if self.model.model_id_env in os.environ:
            self.model.pretrained_model_path = os.environ[self.model.model_id_env]

        if self.model.model_path_env in os.environ and os.environ[self.model.model_path_env]:
            self.model.local_model_path = os.environ[self.model.model_path_env]

    @classmethod
    def from_yaml(cls, config_path: Optional[str] = None) -> "ChessGemmaConfig":
        """Load configuration from YAML file."""
        if config_path is None:
            # Try to find default config file
            possible_paths = [
                Path(__file__).parent / "default.yaml",
                Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
            ]

            for path in possible_paths:
                if path.exists():
                    config_path = str(path)
                    break

        if config_path and Path(config_path).exists():
            import yaml
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)

            # Filter out None values and merge with defaults
            filtered_data = {k: v for k, v in config_data.items() if v is not None}
            return cls(**filtered_data)

        return cls()

    def to_yaml(self, config_path: str):
        """Save configuration to YAML file."""
        import yaml

        # Convert to dict, filtering out None values
        config_dict = self.dict(exclude_none=True)

        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2, sort_keys=False)

        logger.info(f"Configuration saved to {config_path}")


# Global configuration instance
_config_instance: Optional[ChessGemmaConfig] = None


def get_config(config_path: Optional[str] = None) -> ChessGemmaConfig:
    """Get the global configuration instance."""
    global _config_instance

    if _config_instance is None:
        _config_instance = ChessGemmaConfig.from_yaml(config_path)

    return _config_instance


def reload_config(config_path: Optional[str] = None) -> ChessGemmaConfig:
    """Reload configuration from file."""
    global _config_instance
    _config_instance = ChessGemmaConfig.from_yaml(config_path)
    return _config_instance


def save_config(config: ChessGemmaConfig, config_path: str):
    """Save configuration to file."""
    config.to_yaml(config_path)
