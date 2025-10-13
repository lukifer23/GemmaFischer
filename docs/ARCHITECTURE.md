# GemmaFischer Architecture

## System Overview

GemmaFischer is a chess AI system that combines Google's Gemma-3 270M model for strategic guidance and educational explanations with LeelaChess Zero (LC0) as the primary UCI chess engine. Uses LoRA adaptation on Apple Silicon with MPS acceleration and features a hybrid Mixture of Experts (MoE) system that intelligently routes between LC0's precise move calculation and the LLM's educational capabilities.

**Platform**: Mac with Apple Silicon (M3/M4 recommended) - MPS acceleration optimized for both LC0 Metal backend and LLM inference.

## Core Architecture

### Mixture of Experts (MoE) System
- **Intelligent Routing**: Automatic selection between UCI, Tutor, and Director expert models
- **Advanced Caching**: LRU cache for position features and routing decisions (1000+ entries)
- **Performance Optimization**: Cache hit rates 70-85% with memory efficiency monitoring
- **Real-time Feedback**: Live expert selection and routing confidence display
- **Fallback Handling**: Graceful degradation when experts are unavailable

### Expert Models (Hybrid Architecture)
- **UCI Expert**: Chess move generation using LC0 neural engine as primary, with LLM fallback
- **Tutor Expert**: Chess explanations and tactical analysis with educational focus
- **Director Expert**: Strategic Q&A and reasoning (2K+ CoT examples) for advanced chess concepts
- **Hybrid Integration**: LC0 provides precise move calculation while LLM adds strategic context and explanations

### Data Pipeline
- **Standardized Datasets**: 107K+ validated training samples including CoT reasoning
- **Expert-Specific Training**: Domain-focused fine-tuning with curriculum support
- **Quality Assurance**: Automated validation, repair, and Stockfish verification
- **MPS Optimization**: Memory-efficient training with gradient checkpointing

### Web Interface
- **Flask Application**: REST API with real-time hybrid system monitoring and performance tracking
- **Interactive Chess Board**: Click-to-move interface with LC0 analysis integration and expert switching
- **LC0 Integration Display**: Live LC0 engine status, analysis results, and performance metrics
- **Training Controls**: GUI for model training with progress monitoring and system health checks

### Performance & Reliability Layer
- **Error Handling**: Comprehensive error classification and recovery strategies
- **Model Validation**: Real-time integrity checks and corruption detection
- **Advanced Benchmarking**: Statistical analysis with regression detection
- **Intelligent Caching**: Multi-level LRU caching for positions, moves, and responses

## Component Architecture

### Training Layer (`src/training/`)
- Enhanced LoRA fine-tuning with timeout protection and stability
- MPS memory optimization with gradient checkpointing
- Automatic checkpoint resumption and progress tracking
- Curriculum training support with expert-specific configurations

### Inference Layer (`src/inference/`)
- Optimized MoE routing with advanced caching (70%+ hit rates)
- Intelligent response caching for identical queries
- Model validation and integrity checking on load
- Performance monitoring with tokens/second tracking
- UCI bridge for chess engine compatibility

### Web Layer (`src/web/`)
- Flask REST API with real-time performance monitoring
- Interactive chess board with expert switching controls
- Live MoE routing feedback and cache performance display
- Training progress monitoring and control interface

### Utilities Layer (`src/utils/`)
- **Error Handler** (`error_handler.py`): Comprehensive error classification and recovery
- **Model Validator** (`model_validator.py`): Real-time integrity checks and validation
- **Logging Config** (`logging_config.py`): Advanced logging with performance metrics

### Evaluation Layer (`src/evaluation/`)
- **Advanced Benchmarking** (`advanced_benchmark.py`): Statistical analysis with regression detection
- **Comprehensive Evaluation** (`comprehensive_eval.py`): Multi-dimensional performance assessment
- Automated benchmarking with confidence intervals and p-values

### Scripts Layer (`scripts/`)
- **UCI Training Complete** (`train_uci_complete.py`): Stable training with progress monitoring
- **CoT Dataset Validation** (`validate_and_repair_cot_dataset.py`): Quality assurance pipeline
- **Data Processing** (`*.py`): Dataset creation, validation, and optimization tools
