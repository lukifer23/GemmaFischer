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
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Union, Callable, List
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
class MoEConfig:
    """Mixture of Experts configuration settings."""
    enabled: bool = True
    router_checkpoint_path: Optional[str] = None
    router_learning_rate: float = 1e-3
    router_hidden_dim: int = 128
    router_training_steps: int = 1000
    ensemble_weight_threshold: float = 0.3
    adaptive_routing_enabled: bool = True
    performance_tracking_enabled: bool = True
    retraining_threshold: float = 0.1  # Retrain if accuracy drops by 10%
    context_aware_routing: bool = True
    feedback_learning_rate: float = 0.01

    # Expert performance thresholds
    min_expert_confidence: float = 0.6
    max_expert_response_time: float = 2.0  # seconds
    ensemble_size: int = 2  # Number of experts to combine
    retrain_action: str = "monitor"
    retrain_focus_categories: List[str] = field(default_factory=list)


@dataclass
class PerformanceConfig:
    """Performance monitoring and optimization settings."""
    enable_performance_monitoring: bool = True
    metrics_collection_interval: int = 60  # seconds
    slow_query_threshold: float = 1.0  # seconds
    memory_monitoring_enabled: bool = True
    cache_performance_tracking: bool = True
    model_loading_optimization: bool = True
    batch_processing_enabled: bool = True
    parallel_inference_enabled: bool = True
    tokenizer_batch_size: int = 8
    inference_queue_size: int = 100


@dataclass
class LC0EngineSettings:
    """Configuration for LC0 engine."""
    enabled: bool = True
    engine_path: str = "lc0"
    weights_file: str = "models/lc0_weights/default.pb.gz"
    backend: str = "metal"
    threads: int = 2
    nn_cache_size: int = 200000
    search_paths: List[str] = field(default_factory=lambda: [
        "/opt/homebrew/bin/lc0",
        "/usr/local/bin/lc0",
        "/usr/bin/lc0",
        "lc0",
    ])
    debug: bool = False
    time_limit: float = 2.0
    depth: Optional[int] = None


@dataclass
class FallbackEngineSettings:
    """Configuration for fallback Stockfish engine."""
    enabled: bool = True
    engine_path: str = "/opt/homebrew/bin/stockfish"
    threads: int = 2
    hash: int = 128
    skill_level: int = 20
    show_wdl: bool = True
    search_paths: List[str] = field(default_factory=lambda: [
        "/opt/homebrew/bin/stockfish",
        "/usr/local/bin/stockfish",
        "/usr/bin/stockfish",
        "stockfish",
    ])
    depth: int = 18
    time_limit: float = 0.5


@dataclass
class ChessEngineConfig:
    """Configuration for chess engine orchestration."""
    primary: str = "lc0"
    lc0: LC0EngineSettings = field(default_factory=LC0EngineSettings)
    fallback: FallbackEngineSettings = field(default_factory=FallbackEngineSettings)


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
    chess_engine: ChessEngineConfig = field(default_factory=ChessEngineConfig)

    # Advanced features
    moe: MoEConfig = field(default_factory=MoEConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

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

        # Apply engine overrides
        if 'engine_primary' in env_config:
            self.chess_engine.primary = env_config['engine_primary']

        if 'lc0_path' in env_config:
            self.chess_engine.lc0.engine_path = env_config['lc0_path']

        if 'lc0_weights' in env_config:
            self.chess_engine.lc0.weights_file = env_config['lc0_weights']

        if 'lc0_backend' in env_config:
            self.chess_engine.lc0.backend = env_config['lc0_backend']

        if 'lc0_threads' in env_config:
            try:
                self.chess_engine.lc0.threads = int(env_config['lc0_threads'])
            except ValueError:
                pass

        # Optional LC0 time limit override
        if 'lc0_time_limit' in env_config:
            try:
                self.chess_engine.lc0.time_limit = float(env_config['lc0_time_limit'])
            except ValueError:
                pass

        if 'fallback_engine_path' in env_config:
            self.chess_engine.fallback.engine_path = env_config['fallback_engine_path']

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
        config_copy.chess_engine = self.chess_engine
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

        # Set chess engine config
        if 'chess_engine' in config_dict:
            engine_data = config_dict['chess_engine']
            if 'primary' in engine_data:
                config.chess_engine.primary = engine_data['primary']

            if 'lc0' in engine_data:
                lc0_data = engine_data['lc0']
                for field_name in config.chess_engine.lc0.__dataclass_fields__:
                    if field_name in lc0_data:
                        setattr(config.chess_engine.lc0, field_name, lc0_data[field_name])

            if 'fallback' in engine_data:
                fallback_data = engine_data['fallback']
                for field_name in config.chess_engine.fallback.__dataclass_fields__:
                    if field_name in fallback_data:
                        setattr(config.chess_engine.fallback, field_name, fallback_data[field_name])

        # Set datasets
        if 'datasets' in config_dict:
            config.datasets = config_dict['datasets']

        # Set curriculum
        if 'curriculum' in config_dict:
            curriculum_data = config_dict['curriculum']
            if 'phases' in curriculum_data:
                config.curriculum.phases = curriculum_data['phases']

        # Set MoE config
        if 'moe' in config_dict:
            moe_data = config_dict['moe']
            for field_name in config.moe.__dataclass_fields__:
                if field_name in moe_data:
                    setattr(config.moe, field_name, moe_data[field_name])

        # Set performance config
        if 'performance' in config_dict:
            perf_data = config_dict['performance']
            for field_name in config.performance.__dataclass_fields__:
                if field_name in perf_data:
                    setattr(config.performance, field_name, perf_data[field_name])

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
    """Manager for loading and managing ChessGemma configurations with hot-reloading support."""

    def __init__(self):
        self.config_dir = project_root / "configs"
        self.config_dir.mkdir(exist_ok=True)

        # Default configuration
        self._default_config = ChessGemmaConfig()

        # Loaded configurations cache
        self._config_cache: Dict[str, ChessGemmaConfig] = {}

        # Hot-reloading support
        self._file_watchers: Dict[str, Dict[str, Any]] = {}
        self._reload_callbacks: Dict[str, list] = {}
        self._watcher_thread: Optional[threading.Thread] = None
        self._watcher_running = False
        self._reload_interval = 5.0  # Check for changes every 5 seconds

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

    def enable_hot_reload(self, config_name: str = "default", callback: Optional[Callable[[ChessGemmaConfig], None]] = None):
        """Enable hot-reloading for a specific configuration file.

        Args:
            config_name: Name of the configuration to watch
            callback: Function to call when configuration is reloaded
        """
        config_file = self.config_dir / f"{config_name}.yaml"
        if not config_file.exists():
            logger.warning(f"Cannot enable hot reload for {config_name}: config file does not exist")
            return

        # Store file modification time
        self._file_watchers[config_name] = {
            'path': config_file,
            'last_modified': config_file.stat().st_mtime,
            'enabled': True
        }

        if callback:
            if config_name not in self._reload_callbacks:
                self._reload_callbacks[config_name] = []
            self._reload_callbacks[config_name].append(callback)

        # Start watcher thread if not already running
        if not self._watcher_running:
            self._start_watcher_thread()

        logger.info(f"Hot-reloading enabled for configuration: {config_name}")

    def disable_hot_reload(self, config_name: str = "default"):
        """Disable hot-reloading for a specific configuration."""
        if config_name in self._file_watchers:
            self._file_watchers[config_name]['enabled'] = False
            logger.info(f"Hot-reloading disabled for configuration: {config_name}")

    def _start_watcher_thread(self):
        """Start the file watcher thread."""
        self._watcher_running = True
        self._watcher_thread = threading.Thread(target=self._watch_files, daemon=True)
        self._watcher_thread.start()
        logger.debug("Configuration hot-reload watcher thread started")

    def _watch_files(self):
        """Watch configuration files for changes and reload when modified."""
        while self._watcher_running:
            try:
                for config_name, watcher in list(self._file_watchers.items()):
                    if not watcher['enabled']:
                        continue

                    config_file = watcher['path']
                    if not config_file.exists():
                        continue

                    current_mtime = config_file.stat().st_mtime
                    if current_mtime > watcher['last_modified']:
                        logger.info(f"Configuration file {config_name} changed, reloading...")

                        # Reload configuration
                        try:
                            new_config = ChessGemmaConfig.load_from_file(config_file)
                            self._config_cache[config_name] = new_config
                            watcher['last_modified'] = current_mtime

                            # Call reload callbacks
                            if config_name in self._reload_callbacks:
                                for callback in self._reload_callbacks[config_name]:
                                    try:
                                        callback(new_config)
                                    except Exception as e:
                                        logger.error(f"Error in reload callback for {config_name}: {e}")

                            logger.info(f"Configuration {config_name} reloaded successfully")

                        except Exception as e:
                            logger.error(f"Failed to reload configuration {config_name}: {e}")

                time.sleep(self._reload_interval)

            except Exception as e:
                logger.error(f"Error in configuration watcher: {e}")
                time.sleep(self._reload_interval)

    def get_hot_reload_status(self) -> Dict[str, Dict[str, Any]]:
        """Get the status of hot-reloading for all configurations."""
        status = {}
        for config_name, watcher in self._file_watchers.items():
            status[config_name] = {
                'enabled': watcher['enabled'],
                'file_path': str(watcher['path']),
                'last_modified': watcher['last_modified'],
                'has_callbacks': config_name in self._reload_callbacks and len(self._reload_callbacks[config_name]) > 0
            }
        return status

    def shutdown(self):
        """Shutdown the configuration manager and stop hot-reloading."""
        self._watcher_running = False
        if self._watcher_thread and self._watcher_thread.is_alive():
            self._watcher_thread.join(timeout=1.0)
        logger.debug("Configuration manager shutdown complete")


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


def enable_hot_reload(config_name: str = "default", callback: Optional[Callable[[ChessGemmaConfig], None]] = None):
    """Enable hot-reloading for a configuration file."""
    _config_manager.enable_hot_reload(config_name, callback)


def disable_hot_reload(config_name: str = "default"):
    """Disable hot-reloading for a configuration file."""
    _config_manager.disable_hot_reload(config_name)


def get_hot_reload_status() -> Dict[str, Dict[str, Any]]:
    """Get the status of hot-reloading for all configurations."""
    return _config_manager.get_hot_reload_status()


def shutdown_config_manager():
    """Shutdown the configuration manager."""
    _config_manager.shutdown()
