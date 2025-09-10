# GemmaFischer Training Guide

## Overview

This guide covers the complete training process for GemmaFischer, from data preparation to model deployment. The system uses LoRA (Low-Rank Adaptation) fine-tuning with Unsloth optimization for efficient training on M3 Pro Macs with MPS acceleration, with support for dual-mode operation (engine and tutor modes) and chain-of-thought reasoning.

**Platform**: Mac-only (M3 Pro) with MPS acceleration - no CUDA/CPU fallbacks.

## Prerequisites

### System Requirements
- **Hardware**: Mac with M3 Pro chip (required for optimal performance)
- **RAM**: 16GB+ (recommended for training)
- **Software**: Python 3.10+, PyTorch with MPS support
- **Storage**: 10GB+ free space for models and checkpoints
- **macOS**: 12.0+ (required for MPS support)

### Dependencies
```bash
pip install -r requirements.txt
```

### MPS Optimization (M3 Pro)
```python
# Verify MPS availability
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

# Set device to MPS
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

## Training Pipeline

### 1. Data Preparation

#### ChessInstruct Dataset
The primary training data comes from the Thytu/ChessInstruct dataset:

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("Thytu/ChessInstruct", split="train")
print(f"Dataset size: {len(dataset)}")
```

#### Data Format Conversion
Convert raw dataset to chat format for training:

```python
def convert_to_chat_format(example):
    return {
        "conversations": [
            {"role": "system", "content": example["task"]},
            {"role": "user", "content": str(example["input"])},
            {"role": "assistant", "content": example["expected_output"]}
        ]
    }

dataset = dataset.map(convert_to_chat_format)
```

#### Custom Dataset Creation
Create focused chess Q&A datasets:

```bash
# Generate custom dataset
python create_finetune_dataset.py

# Create expanded dataset
python scripts/generate_expanded_dataset.py
```

### 2. Configuration Setup

#### Basic Configuration (`configs/lora.yaml`)
```yaml
# LoRA Configuration
r: 16
lora_alpha: 16
lora_dropout: 0.0
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
bias: none
use_gradient_checkpointing: unsloth
use_rslora: false
random_state: 3407
```

#### Full Training Configuration (`configs/lora_full.yaml`)
```yaml
model:
  pretrained_model_path: "models/unsloth-gemma-3-270m-it/..."
training:
  output_dir: "checkpoints/lora_full"
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 4
  num_train_epochs: 1
  max_steps: 2000
  learning_rate: 2e-4
  fp16: false
  logging_steps: 50
  save_steps: 200
lora:
  r: 16
  lora_alpha: 32
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
  dropout: 0.05
dataset:
  path: "data/datasets/chess_finetune_full.jsonl"

#### Optional: Dataset Mixer
```yaml
datasets:
  - path: "data/finetune/chess_finetune_refined.jsonl"
    weight: 0.3
  - path: "data/datasets/lichess_puzzles_1000_2000.jsonl"
    weight: 0.7
```

#### Optional: Curriculum Phases
```yaml
curriculum:
  - steps: 100
    datasets:
      - path: "data/finetune/chess_finetune_refined.jsonl"
        weight: 1.0
  - steps: 200
    datasets:
      - path: "data/finetune/chess_finetune_refined.jsonl"
        weight: 0.3
      - path: "data/datasets/lichess_puzzles_1000_2000.jsonl"
        weight: 0.7
```
```

### 3. Training Execution

#### Smoke Test (Recommended First)
```bash
# Quick 10-step test
python src/training/train.py --do_train --max_steps 10
```

#### Mixer & Curriculum via POC Trainer
```bash
python src/training/train_lora_poc.py --config src/training/configs/lora_finetune.yaml
```

#### Full Training
```bash
# Full training run
python src/training/train.py --config src/training/configs/lora_full.yaml
```

#### Resume Training
```bash
# Resume from checkpoint
python src/training/train.py --resume_from_checkpoint checkpoints/lora_full/checkpoint-1000
```

### 4. Training Monitoring

#### Real-time Monitoring
```bash
# Monitor training logs
tail -f checkpoints/lora_full/train_log.jsonl

# Check GPU memory usage
nvidia-smi  # or Activity Monitor on macOS
```

#### Key Metrics to Watch
- **Training Loss**: Should decrease steadily (target: <1.5)
- **Learning Rate**: Follows scheduler schedule
- **Memory Usage**: Should stay within available RAM
- **Steps per Second**: Indicates training speed

#### Expected Training Progress
```
Step 0:   Loss = 3.2,  LR = 2e-4
Step 50:  Loss = 2.8,  LR = 1.8e-4
Step 100: Loss = 2.4,  LR = 1.6e-4
Step 200: Loss = 2.0,  LR = 1.4e-4
Step 500: Loss = 1.6,  LR = 1.0e-4
Step 1000: Loss = 1.2, LR = 0.6e-4
```

## Advanced Training Strategies

### Dual-Mode Training

GemmaFischer supports two distinct training modes for different use cases:

#### Engine Mode Training
- **Purpose**: Fast, minimal move generation for UCI compatibility
- **Data Format**: Position → Move pairs
- **Output**: Single UCI move (e.g., "e2e4")
- **Use Case**: Chess software integration, rapid play

#### Tutor Mode Training
- **Purpose**: Educational move generation with explanations
- **Data Format**: Position → Analysis → Move
- **Output**: Step-by-step reasoning + UCI move
- **Use Case**: Learning, analysis, teaching

### Chain-of-Thought Integration

#### Structured Reasoning
```python
def create_cot_training_data(position, move, analysis):
    """Create chain-of-thought training examples"""
    return {
        "conversations": [
            {
                "role": "user", 
                "content": f"Position: {position.fen()}\nMode: Tutor\nAnalyze this position step by step."
            },
            {
                "role": "assistant",
                "content": f"""Let me analyze this position step by step:

1. **Material Count**: {analysis['material']}
2. **King Safety**: {analysis['king_safety']}
3. **Key Threats**: {analysis['threats']}
4. **Best Move**: {move.uci()}

Reasoning: {analysis['reasoning']}"""
            }
        ]
    }
```

#### CoT Templates
- **Tactical Analysis**: Identify threats, calculate variations, find best move
- **Positional Analysis**: Evaluate structure, plan development, assess imbalances
- **Endgame Analysis**: Count material, identify key squares, calculate technique

### Multi-Task Learning

#### Task Mixing Strategy
```python
def create_multi_task_dataset():
    """Create dataset with multiple chess tasks"""
    tasks = {
        'move_prediction': 0.3,      # Pure move generation
        'tactical_explanation': 0.25, # Puzzle solving with explanation
        'positional_analysis': 0.2,   # Strategic evaluation
        'opening_theory': 0.15,       # Opening identification and plans
        'endgame_technique': 0.1      # Endgame knowledge
    }
    return tasks
```

#### Curriculum Learning
1. **Phase 1**: Basic rules and legal moves
2. **Phase 2**: Simple tactics and patterns
3. **Phase 3**: Positional concepts and strategy
4. **Phase 4**: Complex combinations and endgames
5. **Phase 5**: Style conditioning and advanced analysis

### Training Data Preparation

```python
def prepare_dual_mode_data(chess_data):
    """Prepare data for dual-mode training"""
    engine_data = []
    tutor_data = []
    
    for game in chess_data:
        for position, move, analysis in game:
            # Engine mode: simple position → move
            engine_data.append({
                "conversations": [
                    {"role": "user", "content": f"Position: {position.fen()}\nMode: Engine\nGenerate best move."},
                    {"role": "assistant", "content": move.uci()}
                ]
            })
            
            # Tutor mode: position → analysis → move
            tutor_data.append({
                "conversations": [
                    {"role": "user", "content": f"Position: {position.fen()}\nMode: Tutor\nAnalyze and explain."},
                    {"role": "assistant", "content": f"{analysis}\n\nBest move: {move.uci()}"}
                ]
            })
    
    return engine_data, tutor_data
```

### Style Conditioning

Train the model to emulate different playing styles:

```python
def add_style_conditioning(data, style):
    """Add style conditioning to training data"""
    for item in data:
        for conv in item["conversations"]:
            if conv["role"] == "user":
                conv["content"] = f"Style: {style}\n{conv['content']}"
    return data

# Example styles
styles = ["fischer", "aggressive", "defensive", "balanced"]
for style in styles:
    style_data = add_style_conditioning(engine_data, style)
    # Train with style-conditioned data
```

### Chain-of-Thought Integration

Integrate step-by-step reasoning into the training:

```python
def create_cot_prompt(position, style="balanced"):
    """Create chain-of-thought prompt"""
    return f"""Position: {position.fen()}
Style: {style}
Mode: Tutor

Analyze this position step by step:
1. Evaluate the current position
2. Identify key threats and opportunities
3. Consider candidate moves
4. Choose the best move with reasoning

Respond with the best move in UCI format at the end."""
```

## Advanced Training Options

### 1. Hyperparameter Tuning

#### Learning Rate Schedules
```yaml
training:
  lr_scheduler_type: "cosine"  # or "linear", "constant"
  warmup_steps: 100
  learning_rate: 2e-4
```

#### Batch Size Optimization
```yaml
training:
  per_device_train_batch_size: 4    # Reduce if OOM
  gradient_accumulation_steps: 8     # Increase to maintain effective batch size
  # Effective batch size = per_device_train_batch_size * gradient_accumulation_steps
```

#### LoRA Configuration
```yaml
lora:
  r: 32                    # Higher rank = more parameters
  lora_alpha: 64          # Scaling factor
  dropout: 0.05           # Regularization
  target_modules:         # Which layers to adapt
    - "q_proj"
    - "v_proj"
    - "k_proj"
    - "o_proj"
```

### 2. Memory Optimization

#### Gradient Checkpointing
```python
# Enable in training config
use_gradient_checkpointing: "unsloth"
```

#### Mixed Precision (if supported)
```yaml
training:
  fp16: true    # Use with caution on MPS
  bf16: false   # Better for Apple Silicon
```

#### Model Quantization
```python
# 4-bit quantization (experimental)
load_in_4bit: true
```

### 3. Dataset Strategies

#### Curriculum Learning
```python
# Start with simple examples, gradually increase complexity
def create_curriculum_dataset():
    simple_examples = load_simple_chess_qa()
    complex_examples = load_complex_chess_qa()
    
    # Mix datasets with increasing complexity
    curriculum = simple_examples + complex_examples[:len(simple_examples)//2]
    return curriculum
```

#### Data Augmentation
```python
def augment_chess_data(example):
    # Add variations to chess positions
    # Rotate board, add context, etc.
    return augmented_example
```

## Troubleshooting

### Common Issues

#### Out of Memory (OOM)
**Symptoms**: Training crashes with memory error
**Solutions**:
1. Reduce batch size: `per_device_train_batch_size: 2`
2. Increase gradient accumulation: `gradient_accumulation_steps: 16`
3. Enable gradient checkpointing: `use_gradient_checkpointing: "unsloth"`
4. Use CPU fallback: Set `PYTORCH_ENABLE_MPS_FALLBACK=1`

#### Slow Training
**Symptoms**: Very low steps per second
**Solutions**:
1. Check MPS availability: `torch.backends.mps.is_available()`
2. Reduce sequence length: `max_seq_length: 512`
3. Use smaller model or quantized version
4. Check system resources (CPU, memory)

#### Loss Not Decreasing
**Symptoms**: Loss plateaus or increases
**Solutions**:
1. Check learning rate: Try `1e-5` to `5e-4`
2. Verify data quality: Check dataset format
3. Increase LoRA rank: `r: 32` or `r: 64`
4. Add warmup: `warmup_steps: 100`

#### Checkpoint Issues
**Symptoms**: Cannot resume training or corrupted checkpoints
**Solutions**:
1. Check checkpoint integrity: `scripts/adapter_integrity_test.py`
2. Clear corrupted checkpoints: Remove bad checkpoint directories
3. Start fresh: Delete all checkpoints and restart

### Debug Mode

#### Enable Debug Logging
```bash
export PYTHONPATH=$PWD
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
```

#### Model Inspection
```python
# Check model parameters
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

# Check LoRA adapters
for name, module in model.named_modules():
    if hasattr(module, 'lora_A'):
        print(f"LoRA module: {name}")
```

## Evaluation and Validation

### 1. Training Validation
```bash
# Run evaluation during training
python src/evaluation/chess_evaluation.py --checkpoint checkpoints/lora_full/checkpoint-1000
```

### 2. Model Comparison
```bash
# Compare before/after training
python compare_chess_qa.py
```

### 3. Chess-Specific Metrics
```python
# Evaluate chess understanding
from src.evaluation.chess_evaluation import ChessEvaluator

evaluator = ChessEvaluator(model_path, adapter_path)
results = evaluator.evaluate_test_set(test_questions)
print(f"Chess relevance: {results['average_chess_relevance']:.2%}")
```

## Deployment Preparation

### 1. Model Export
```python
# Save final model
model.save_pretrained("final_chess_model")
tokenizer.save_pretrained("final_chess_model")
```

### 2. Adapter Merging
```python
# Merge LoRA adapter into base model
merged_model = model.merge_and_unload()
merged_model.save_pretrained("merged_chess_model")
```

### 3. Model Optimization
```python
# Quantize for deployment
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
```

## Best Practices

### 1. Training Strategy
- Start with smoke tests (10-50 steps)
- Use validation sets for early stopping
- Save checkpoints frequently
- Monitor loss curves and adjust learning rate

### 2. Data Quality
- Validate chess moves with Stockfish
- Ensure diverse question types
- Balance difficulty levels
- Clean and preprocess data thoroughly

### 3. Resource Management
- Monitor memory usage continuously
- Use appropriate batch sizes
- Enable gradient checkpointing for large models
- Plan for checkpoint storage

### 4. Experiment Tracking
- Log all hyperparameters
- Save training configurations
- Track performance metrics
- Document successful configurations

## Next Steps

After successful training:

1. **Evaluate Performance**: Run comprehensive evaluation
2. **Deploy Model**: Set up inference pipeline
3. **Monitor Usage**: Track model performance in production
4. **Iterate**: Use feedback to improve training data and process

For more detailed information, see the [API Reference](API_REFERENCE.md) and [Architecture](ARCHITECTURE.md) documentation.
