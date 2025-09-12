# ChessGemma: Chess LLM Engine + Tutor

A chess AI system that fine-tunes Google's Gemma-3 270M model to function as both a chess engine (UCI-compatible) and chess tutor using LoRA adaptation on Apple Silicon with MPS acceleration. The system features Mixture of Experts (MoE) routing for intelligent expert selection based on query analysis.

## Key Features

### Core Capabilities
- **Mixture of Experts (MoE)**: Intelligent routing between UCI, Tutor, and Director expert models
- **MPS-Optimized Training**: LoRA fine-tuning optimized for Apple Silicon MPS acceleration
- **UCI Compatibility**: Full UCI protocol support for chess software integration
- **Multi-Mode Operation**: Engine (UCI moves), Tutor (explanations), and Director (Q&A) modes
- **Data Standardization**: Automated dataset validation and quality assurance pipeline
- **Web Interface**: Real-time MoE routing display and expert switching controls

### Current Status
- **Training Data**: 105K+ standardized samples across three expert domains
- **Model Checkpoints**: Multiple specialized LoRA adapters with robust checkpoint management
- **Data Quality**: 100% valid samples in expert datasets with comprehensive validation
- **MoE Routing**: Intelligent expert selection based on position complexity and query type
- **Web Interface**: Enhanced interface at http://localhost:5000 with MoE controls
- **Training Speed**: Optimized ~2-3 steps/second on M3 Pro with MPS memory optimization

### Current Challenges
- **UCI Expert Training**: Current checkpoint (1000 steps) generates suboptimal responses; needs additional training
- **CoT Dataset**: Missing comprehensive chain-of-thought reasoning dataset (5K+ examples exist but are invalid)
- **Expert Switching**: MoE routing working but could benefit from more specialized training per expert

### Next Steps
- **Complete UCI Training**: Resume training to 1600+ steps for better move generation
- **Fix CoT Dataset**: Investigate and repair invalid reasoning examples
- **Enhanced Evaluation**: Implement comprehensive model testing and comparison
- **Production Polish**: Optimize inference speed and memory usage


## Quick Start

### Training Commands

#### UCI Expert Training (Current Focus)
Train the UCI expert for improved chess move generation:

```bash
# Train UCI expert for 1600 steps (recommended for better performance)
cd /Users/admin/Downloads/VSCode/ChessGemma && python -m src.training.train_lora_poc --expert uci --config auto --max_steps_override 1600 --disable_eval
```

#### Individual Expert Training
Train specific experts individually:

```bash
# UCI Expert (chess move generation)
python -m src.training.train_lora_poc --expert uci --config auto --max_steps_override 1000 --disable_eval

# Tutor Expert (chess explanations)
python -m src.training.train_lora_poc --expert tutor --config auto --max_steps_override 1000 --disable_eval

# Director Expert (Q&A reasoning)
python -m src.training.train_lora_poc --expert director --config auto --max_steps_override 1000 --disable_eval
```

#### Web Interface
Launch the web interface for testing and evaluation:

```bash
# Start web interface
python -m src.web.run_web_app
# Visit: http://localhost:5000
```

### Prerequisites

- **Mac with Apple Silicon chip** (M3/M4 recommended for MPS performance)
- Python 3.10+
- 16GB+ RAM (recommended for training)
- macOS 12.0+ (for MPS support)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd ChessGemma

# Setup virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

1. **Start web interface:**
```bash
python -m src.web.run_web_app
# Visit: http://localhost:5000
```

2. **Run inference programmatically:**
```python
from src.inference.inference import get_inference_instance

# Load model and get inference
inference = get_inference_instance()
result = inference.generate_response("What is the best move for white?")
print(result['response'])
```

3. **Run training:**
```bash
# UCI Expert training (recommended next step)
python -m src.training.train_lora_poc --expert uci --config auto --max_steps_override 1600 --disable_eval
```

## Project Structure

```
ChessGemma/
├── src/
│   ├── training/       # LoRA fine-tuning scripts
│   ├── inference/      # Model inference and MoE routing
│   ├── web/           # Flask web interface
│   └── evaluation/    # Testing and benchmarking
├── data/
│   ├── standardized/  # 105K+ validated training samples
│   └── validation/    # Quality assessment reports
├── checkpoints/       # LoRA adapter checkpoints
└── docs/             # Documentation
```

## Architecture Overview

- **Mixture of Experts (MoE)**: Intelligent routing between UCI, Tutor, and Director experts
- **MPS Optimization**: Native Apple Silicon performance with memory-efficient training
- **LoRA Fine-tuning**: Parameter-efficient adaptation of Gemma-3 270M model
- **UCI Bridge**: Full chess engine protocol compatibility
- **Web Interface**: Real-time expert routing and interactive chess analysis

## License

This project is licensed under the MIT License.
