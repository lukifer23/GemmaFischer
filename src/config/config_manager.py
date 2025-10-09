#!/usr/bin/env python3
"""
Unified Configuration Manager for ChessGemma

Consolidates all configuration files into a single, validated system with
environment-specific overrides and comprehensive documentation.
"""

from __future__ import annotations

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(project_root))

# Import common utilities
from ..utils.common import get_logger, get_environment_config

# Get utility functions
logger = get_logger(__name__)


@dataclass
class ModelConfig:
    """Model configuration settings."""
    pretrained_model_path: str = "google/gemma-3-270m"
    local_model_path: Optional[str] = None
    model_id_env: str = "CHESSGEMMA_MODEL_ID"
    model_path_env: str = "CHESSGEMMA_MODEL_PATH"


@dataclass
class TrainingConfig:
    """Training configuration settings."""
    output_dir: str = "checkpoints/lora_default"
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_steps: int = 1000
    learning_rate: float = 2e-4
    num_train_epochs: int = 1
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    logging_steps: int = 50
    save_steps: int = 200
    eval_strategy: str = "steps"
    eval_steps: int = 200
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    lr_scheduler_type: str = "cosine"
    fp16: bool = False
    bf16: bool = False
    remove_unused_columns: bool = False
    dataloader_pin_memory: bool = False
    dataloader_num_workers: int = 0


@dataclass
class LoRAConfig:
    """LoRA configuration settings."""
    r: int = 16
    lora_alpha: int = 32
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"])
    dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class ExpertConfig:
    """Expert-specific configuration settings."""
    uci: TrainingConfig = field(default_factory=TrainingConfig)
    tutor: TrainingConfig = field(default_factory=TrainingConfig)
    director: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        # Set expert-specific defaults
        self.uci.output_dir = "checkpoints/lora_uci"
        self.uci.max_steps = 2000
        self.uci.learning_rate = 2e-4
        self.uci.logging_steps = 25
        self.uci.save_steps = 100

        self.tutor.output_dir = "checkpoints/lora_tutor"
        self.tutor.max_steps = 1000
        self.tutor.learning_rate = 1.5e-4

        self.director.output_dir = "checkpoints/lora_director"
        self.director.max_steps = 1000
        self.director.learning_rate = 1.5e-4


@dataclass
class CurriculumConfig:
    """Curriculum training configuration."""
    phases: list = field(default_factory=list)

    def add_phase(self, steps: int, datasets: list):
        """Add a curriculum phase."""
        self.phases.append({
            'steps': steps,
            'datasets': datasets
        })


@dataclass
class InferenceConfig:
    """Inference configuration settings."""
    max_new_tokens: int = 200
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.1
    use_cache: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None

    # Expert-specific overrides
    engine_temperature: float = 0.0
    engine_top_p: float = 1.0
    engine_do_sample: bool = False
    engine_max_new_tokens: int = 8

    tutor_temperature: float = 0.7
    tutor_top_p: float = 0.9
    tutor_do_sample: bool = True
    tutor_max_new_tokens: int = 150

    director_temperature: float = 0.6
    director_top_p: float = 0.9
    director_do_sample: bool = True
    director_max_new_tokens: int = 200


@dataclass
class CacheConfig:
    """Caching configuration settings."""
    max_cache_size: int = 512
    engine_cache_max: int = 1024
    cache_ttl_seconds: int = 3600  # 1 hour
    enable_response_cache: bool = True
    enable_engine_cache: bool = True
    enable_kv_cache: bool = True


@dataclass
class SystemConfig:
    """System-wide configuration settings."""
    debug_mode: bool = False
    log_level: str = "INFO"
    device: str = "auto"  # auto, mps, cpu
    seed: int = 42
    max_workers: int = 4
    timeout_minutes: int = 300  # 5 hours
    memory_limit_gb: float = 16.0


@dataclass
class ChessGemmaConfig:
    """Unified configuration for ChessGemma."""

    # Core components
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    experts: ExpertConfig = field(default_factory=ExpertConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    # Dataset configuration
    datasets: list = field(default_factory=list)

    # Runtime overrides
    _overrides: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self):
        """Apply environment variable overrides after initialization."""
        self._apply_environment_overrides()

    def _apply_environment_overrides(self):
        """Apply environment variable overrides."""
        # Use common utility for environment configuration
        env_config = get_environment_config()

        # Apply model overrides
        if 'model_id' in env_config:
            self.model.pretrained_model_path = env_config['model_id']
        if 'model_path' in env_config:
            self.model.local_model_path = env_config['model_path']

        # Apply system overrides
        if 'debug_mode' in env_config:
            self.system.debug_mode = env_config['debug_mode']

        if 'timeout_minutes' in env_config:
            self.system.timeout_minutes = env_config['timeout_minutes']

        # Apply cache overrides
        if 'cache_size' in env_config:
            self.cache.max_cache_size = env_config['cache_size']

    def get_training_config(self, expert: str = "default") -> Dict[str, Any]:
        """Get training configuration for a specific expert."""
        if expert == "default":
            base_config = self.training
        elif expert == "uci":
            base_config = self.experts.uci
        elif expert == "tutor":
            base_config = self.experts.tutor
        elif expert == "director":
            base_config = self.experts.director
        else:
            base_config = self.training

        # Convert to dict and apply any runtime overrides
        config_dict = self._dataclass_to_dict(base_config)
        config_dict.update(self._overrides.get('training', {}))

        return config_dict

    def get_lora_config(self) -> Dict[str, Any]:
        """Get LoRA configuration."""
        config_dict = self._dataclass_to_dict(self.lora)
        config_dict.update(self._overrides.get('lora', {}))
        return config_dict

    def get_inference_config(self, expert: str = "tutor") -> Dict[str, Any]:
        """Get inference configuration for a specific expert."""
        base_config = self.inference

        # Apply expert-specific overrides
        if expert == "uci":
            config_dict = self._dataclass_to_dict(base_config)
            config_dict.update({
                'temperature': self.inference.engine_temperature,
                'top_p': self.inference.engine_top_p,
                'do_sample': self.inference.engine_do_sample,
                'max_new_tokens': self.inference.engine_max_new_tokens,
            })
        elif expert == "tutor":
            config_dict = self._dataclass_to_dict(base_config)
            config_dict.update({
                'temperature': self.inference.tutor_temperature,
                'top_p': self.inference.tutor_top_p,
                'do_sample': self.inference.tutor_do_sample,
                'max_new_tokens': self.inference.tutor_max_new_tokens,
            })
        elif expert == "director":
            config_dict = self._dataclass_to_dict(base_config)
            config_dict.update({
                'temperature': self.inference.director_temperature,
                'top_p': self.inference.director_top_p,
                'do_sample': self.inference.director_do_sample,
                'max_new_tokens': self.inference.director_max_new_tokens,
            })
        else:
            config_dict = self._dataclass_to_dict(base_config)

        config_dict.update(self._overrides.get('inference', {}))
        return config_dict

    def _dataclass_to_dict(self, obj) -> Dict[str, Any]:
        """Convert a dataclass instance to a dictionary."""
        if hasattr(obj, '__dataclass_fields__'):
            result = {}
            for field_name, field_def in obj.__dataclass_fields__.items():
                value = getattr(obj, field_name)
                if hasattr(value, '__dataclass_fields__'):
                    # Nested dataclass
                    result[field_name] = self._dataclass_to_dict(value)
                else:
                    result[field_name] = value
            return result
        return obj

    def override(self, section: str, key: str, value: Any):
        """Override a configuration value at runtime."""
        if section not in self._overrides:
            self._overrides[section] = {}
        self._overrides[section][key] = value

    def save_to_file(self, filepath: Union[str, Path]):
        """Save configuration to a YAML file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Create a copy without runtime overrides
        config_copy = ChessGemmaConfig()
        config_copy.model = self.model
        config_copy.training = self.training
        config_copy.lora = self.lora
        config_copy.experts = self.experts
        config_copy.curriculum = self.curriculum
        config_copy.inference = self.inference
        config_copy.cache = self.cache
        config_copy.system = self.system
        config_copy.datasets = self.datasets

        config_dict = self._dataclass_to_dict(config_copy)

        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Configuration saved to {filepath}")

    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> 'ChessGemmaConfig':
        """Load configuration from a YAML file."""
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Remove runtime overrides if present
        config_dict.pop('_overrides', None)

        # Create a new instance and populate it manually to avoid issues
        config = cls()

        # Set model config
        if 'model' in config_dict:
            model_data = config_dict['model']
            config.model.pretrained_model_path = model_data.get('pretrained_model_path', config.model.pretrained_model_path)
            config.model.local_model_path = model_data.get('local_model_path', config.model.local_model_path)
            config.model.model_id_env = model_data.get('model_id_env', config.model.model_id_env)
            config.model.model_path_env = model_data.get('model_path_env', config.model.model_path_env)

        # Set training config
        if 'training' in config_dict:
            training_data = config_dict['training']
            for field_name in config.training.__dataclass_fields__:
                if field_name in training_data:
                    setattr(config.training, field_name, training_data[field_name])

        # Set lora config
        if 'lora' in config_dict:
            lora_data = config_dict['lora']
            for field_name in config.lora.__dataclass_fields__:
                if field_name in lora_data:
                    setattr(config.lora, field_name, lora_data[field_name])

        # Set experts config
        if 'experts' in config_dict:
            experts_data = config_dict['experts']
            for expert_name in ['uci', 'tutor', 'director']:
                if expert_name in experts_data:
                    expert_data = experts_data[expert_name]
                    expert_config = getattr(config.experts, expert_name)
                    for field_name in expert_config.__dataclass_fields__:
                        if field_name in expert_data:
                            setattr(expert_config, field_name, expert_data[field_name])

        # Set inference config
        if 'inference' in config_dict:
            inference_data = config_dict['inference']
            for field_name in config.inference.__dataclass_fields__:
                if field_name in inference_data:
                    setattr(config.inference, field_name, inference_data[field_name])

        # Set cache config
        if 'cache' in config_dict:
            cache_data = config_dict['cache']
            for field_name in config.cache.__dataclass_fields__:
                if field_name in cache_data:
                    setattr(config.cache, field_name, cache_data[field_name])

        # Set system config
        if 'system' in config_dict:
            system_data = config_dict['system']
            for field_name in config.system.__dataclass_fields__:
                if field_name in system_data:
                    setattr(config.system, field_name, system_data[field_name])

        # Set datasets
        if 'datasets' in config_dict:
            config.datasets = config_dict['datasets']

        # Set curriculum
        if 'curriculum' in config_dict:
            curriculum_data = config_dict['curriculum']
            if 'phases' in curriculum_data:
                config.curriculum.phases = curriculum_data['phases']

        return config

    @classmethod
    def _dict_to_dataclass(cls, data: Dict[str, Any], dataclass_type):
        """Convert a dictionary to a dataclass instance recursively."""
        if not hasattr(dataclass_type, '__dataclass_fields__'):
            return data

        kwargs = {}
        for field_name, field_def in dataclass_type.__dataclass_fields__.items():
            field_type = field_def.type
            if field_name in data:
                value = data[field_name]
                # Handle nested dataclasses
                if (hasattr(field_type, '__dataclass_fields__') and
                    isinstance(value, dict)):
                    # Nested dataclass
                    kwargs[field_name] = cls._dict_to_dataclass(value, field_type)
                else:
                    kwargs[field_name] = value

        return dataclass_type(**kwargs)

    def validate(self) -> list:
        """Validate configuration and return list of validation errors."""
        errors = []

        # Validate model configuration
        if not self.model.pretrained_model_path:
            errors.append("Model pretrained_model_path is required")

        # Validate training configuration
        if self.training.per_device_train_batch_size <= 0:
            errors.append("Training per_device_train_batch_size must be positive")

        if self.training.max_steps <= 0 and self.training.num_train_epochs <= 0:
            errors.append("Either max_steps or num_train_epochs must be positive")

        if self.training.learning_rate <= 0:
            errors.append("Training learning_rate must be positive")

        # Validate LoRA configuration
        if self.lora.r <= 0:
            errors.append("LoRA r must be positive")

        if self.lora.lora_alpha <= 0:
            errors.append("LoRA lora_alpha must be positive")

        if not self.lora.target_modules:
            errors.append("LoRA target_modules cannot be empty")

        # Validate cache configuration
        if self.cache.max_cache_size <= 0:
            errors.append("Cache max_cache_size must be positive")

        return errors


class ConfigManager:
    """Manager for loading and managing ChessGemma configurations."""

    def __init__(self):
        self.config_dir = project_root / "configs"
        self.config_dir.mkdir(exist_ok=True)

        # Default configuration
        self._default_config = ChessGemmaConfig()

        # Loaded configurations cache
        self._config_cache: Dict[str, ChessGemmaConfig] = {}

    def get_config(self, name: str = "default") -> ChessGemmaConfig:
        """Get a configuration by name."""
        if name in self._config_cache:
            return self._config_cache[name]

        # Try to load from file
        config_file = self.config_dir / f"{name}.yaml"
        if config_file.exists():
            try:
                config = ChessGemmaConfig.load_from_file(config_file)
                self._config_cache[name] = config
                return config
            except Exception as e:
                logger.warning(f"Failed to load config {name} from {config_file}: {e}")

        # Return default config
        return self._default_config

    def save_config(self, config: ChessGemmaConfig, name: str = "default"):
        """Save a configuration to file."""
        config_file = self.config_dir / f"{name}.yaml"
        config.save_to_file(config_file)
        self._config_cache[name] = config

    def list_configs(self) -> list:
        """List available configuration files."""
        return [f.stem for f in self.config_dir.glob("*.yaml")]

    def create_expert_config(self, expert: str, base_config: str = "default") -> ChessGemmaConfig:
        """Create an expert-specific configuration."""
        base = self.get_config(base_config)

        # Apply expert-specific modifications
        if expert == "uci":
            config = ChessGemmaConfig()
            config.model = base.model
            config.lora = base.lora
            config.training = base.experts.uci
            config.inference = base.inference
            config.cache = base.cache
            config.system = base.system
            config.datasets = base.datasets

        elif expert == "tutor":
            config = ChessGemmaConfig()
            config.model = base.model
            config.lora = base.lora
            config.training = base.experts.tutor
            config.inference = base.inference
            config.cache = base.cache
            config.system = base.system
            config.datasets = base.datasets

        elif expert == "director":
            config = ChessGemmaConfig()
            config.model = base.model
            config.lora = base.lora
            config.training = base.experts.director
            config.inference = base.inference
            config.cache = base.cache
            config.system = base.system
            config.datasets = base.datasets

        else:
            config = base

        return config


# Global configuration manager instance
_config_manager = ConfigManager()


def get_config(name: str = "default") -> ChessGemmaConfig:
    """Get a configuration instance."""
    return _config_manager.get_config(name)


def create_default_config() -> ChessGemmaConfig:
    """Create and return the default configuration."""
    return ChessGemmaConfig()


def load_config_from_file(filepath: Union[str, Path]) -> ChessGemmaConfig:
    """Load configuration from a specific file."""
    return ChessGemmaConfig.load_from_file(filepath)


def save_config_to_file(config: ChessGemmaConfig, filepath: Union[str, Path], name: str = "default"):
    """Save configuration to a specific file."""
    _config_manager.save_config(config, name)
    config.save_to_file(filepath)
