# ChessGemma Training Guide

## Overview

This guide covers the training process for ChessGemma's Mixture of Experts (MoE) system. The training uses LoRA fine-tuning with MPS optimization on Apple Silicon hardware.

**Platform**: Mac with Apple Silicon (M3/M4 recommended) - MPS acceleration optimized.
**Current Status**: 105K+ standardized training samples across three expert domains.

## Prerequisites

### System Requirements
- **Hardware**: Mac with Apple Silicon chip (M3/M4 recommended)
- **RAM**: 16GB+ (recommended for training)
- **Software**: Python 3.10+, PyTorch with MPS support
- **Storage**: 10GB+ free space for models and checkpoints
- **macOS**: 12.0+ (for MPS support)

### Installation
```bash
# Setup environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset Preparation

### Available Datasets
ChessGemma uses standardized training data across three expert domains:

```bash
# Current dataset sizes (validated and processed)
data/standardized/standardized_uci_expert.jsonl:     50,000 samples (UCI move generation)
data/standardized/standardized_tutor_expert.jsonl:   50,000 samples (Chess explanations)
data/standardized/standardized_director_expert.jsonl: 5,133 samples (Q&A reasoning)
```

### Data Quality
- **Validation**: 100% move legality verification with Stockfish
- **Format**: Standardized JSONL with consistent schema
- **Quality**: All samples validated for correctness
- **Coverage**: Comprehensive chess positions and scenarios

## Training Commands

### UCI Expert Training
Train the UCI expert for chess move generation:

```bash
# Recommended: Train for 1600 steps
python -m src.training.train_lora_poc --expert uci --config auto --max_steps_override 1600 --disable_eval
```

### Individual Expert Training

```bash
# UCI Expert (chess moves)
python -m src.training.train_lora_poc --expert uci --config auto --max_steps_override 1000 --disable_eval

# Tutor Expert (explanations)
python -m src.training.train_lora_poc --expert tutor --config auto --max_steps_override 1000 --disable_eval

# Director Expert (Q&A)
python -m src.training.train_lora_poc --expert director --config auto --max_steps_override 1000 --disable_eval
```

## Performance

### Training Performance (M3 Pro)
- **Memory Usage**: ~4-6GB peak
- **Training Speed**: ~2-3 steps/second
- **MPS Acceleration**: Native Apple Silicon optimization
- **Checkpoint Management**: Automatic saving every 100 steps
