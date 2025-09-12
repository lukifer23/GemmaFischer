# ChessGemma Architecture

## System Overview

ChessGemma is a chess AI system that fine-tunes Google's Gemma-3 270M model using LoRA adaptation. It features Mixture of Experts (MoE) routing for intelligent expert selection and operates on Apple Silicon with MPS acceleration.

**Platform**: Mac with Apple Silicon (M3/M4 recommended) - MPS acceleration optimized.

## Core Architecture

### Mixture of Experts (MoE) System
- **Intelligent Routing**: Automatic selection between UCI, Tutor, and Director expert models
- **Query Analysis**: Position complexity and query type classification
- **Real-time Feedback**: Live expert selection and routing confidence display
- **Fallback Handling**: Graceful degradation when experts are unavailable

### Expert Models
- **UCI Expert**: Chess move generation in UCI format
- **Tutor Expert**: Chess explanations and tactical analysis
- **Director Expert**: Strategic Q&A and reasoning
- **LoRA Adapters**: Parameter-efficient fine-tuning for each domain

### Data Pipeline
- **Standardized Datasets**: 105K+ validated training samples
- **Expert-Specific Training**: Domain-focused fine-tuning
- **Quality Assurance**: Automated validation and Stockfish verification
- **MPS Optimization**: Memory-efficient training on Apple Silicon

### Web Interface
- **Flask Application**: REST API with real-time routing
- **Interactive Chess Board**: Click-to-move interface
- **Expert Status Display**: Live MoE routing feedback
- **Training Controls**: GUI for model training and evaluation

## Component Architecture

### Training Layer (`src/training/`)
- LoRA fine-tuning with Unsloth optimization
- Expert-specific configuration management
- MPS memory optimization and batch sizing
- Checkpoint management and progress tracking

### Inference Layer (`src/inference/`)
- MoE routing and expert selection
- Model loading and adapter switching
- UCI bridge for chess engine compatibility
- Response generation and validation

### Web Layer (`src/web/`)
- Flask REST API endpoints
- Real-time expert routing display
- Interactive chess analysis interface
- Training progress monitoring
