# ChessGemma: Fine-tuning Gemma-3 for Chess Q&A

A chess AI system that fine-tunes Google's Gemma-3 270M model for chess-related question answering using LoRA (Low-Rank Adaptation) on Apple Silicon with MPS acceleration.

## Project Overview

ChessGemma combines several components:
- **Fine-tuned Language Model**: Gemma-3 270M with LoRA adapters for chess knowledge
- **Chess Engine Integration**: Stockfish integration for move validation and analysis
- **Web Interface**: Flask-based web application for interaction
- **Evaluation Framework**: Chess-specific metrics and testing tools

## Key Features

- **Efficient Training**: LoRA fine-tuning with Unsloth optimization
- **Apple Silicon Support**: MPS acceleration for M-series Macs
- **Chess Engine Integration**: Stockfish for move validation and position analysis
- **Web Interface**: Interactive chess Q&A interface
- **Evaluation Tools**: Chess-specific metrics and testing framework
- **Multiple Training Configurations**: Flexible training setups for different use cases

## Quick Start

### Prerequisites

- Python 3.10+
- Apple Silicon Mac (M1/M2/M3) or compatible system
- 8GB+ RAM (16GB recommended)
- Stockfish chess engine (optional, for advanced features)

### Installation

1. **Clone and setup environment:**
```bash
git clone https://github.com/lukifer23/GemmaFischer.git
cd GemmaFischer
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install Stockfish (optional):**
```bash
# macOS with Homebrew
brew install stockfish

# Or download from https://stockfishchess.org/download/
```

### Basic Usage

1. **Run inference with pre-trained model:**
```bash
python src/inference/inference.py --download
```

2. **Start web interface:**
```bash
python src/web/app.py
# Visit http://localhost:5000
```

3. **Run training (smoke test):**
```bash
python src/training/train.py --do_train --max_steps 10
```

## Project Structure

```
GemmaFischer/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── docs/                     # Documentation
│   ├── ARCHITECTURE.md      # System architecture
│   ├── TRAINING_GUIDE.md    # Training instructions
│   ├── API_REFERENCE.md     # API documentation
│   └── PROJECT_PLAN.md      # Future development roadmap
├── src/                      # Source code
│   ├── training/            # Training scripts
│   │   ├── train.py         # Main training script
│   │   ├── train_lora_poc.py # LoRA proof-of-concept
│   │   └── configs/         # Training configurations
│   ├── inference/           # Inference and chess engine
│   │   ├── inference.py     # Model inference
│   │   └── chess_engine.py  # Stockfish integration
│   ├── evaluation/          # Evaluation and testing
│   │   └── chess_evaluation.py
│   └── web/                 # Web interface
│       ├── app.py           # Flask application
│       └── templates/       # HTML templates
├── data/                     # Data management
│   ├── raw/                 # Raw datasets
│   ├── processed/           # Processed data
│   └── datasets/            # Training datasets
├── models/                   # Model storage
├── checkpoints/             # Training checkpoints
├── tests/                   # Unit tests
└── scripts/                 # Utility scripts
```

## Architecture

### Core Components

1. **Training Pipeline** (`src/training/`)
   - LoRA fine-tuning with Unsloth
   - Multiple configuration options
   - Checkpoint management
   - Resume functionality

2. **Inference Engine** (`src/inference/`)
   - Model loading and caching
   - Chess-specific prompt formatting
   - Response generation and validation

3. **Chess Engine Integration** (`src/inference/chess_engine.py`)
   - Stockfish integration
   - Move validation
   - Position analysis
   - Tactical pattern recognition

4. **Web Interface** (`src/web/`)
   - Flask-based web application
   - Interactive Q&A interface
   - Chess board visualization
   - Model status monitoring

5. **Evaluation Framework** (`src/evaluation/`)
   - Chess-specific metrics
   - Model performance testing
   - Quality assessment tools

## Training Guide

### Configuration Files

Training configurations are located in `src/training/configs/`:

- `lora.yaml` - Basic LoRA configuration
- `lora_full.yaml` - Full training setup
- `lora_expanded.yaml` - Enhanced training with expanded dataset

### Training Process

1. **Prepare dataset:**
```bash
python data/prepare_dataset.py --output_dir data/processed
```

2. **Run training:**
```bash
python src/training/train.py --config src/training/configs/lora_full.yaml
```

3. **Monitor progress:**
```bash
# Training logs are saved to checkpoints/
tail -f checkpoints/lora_full/train_log.jsonl
```

### Training Parameters

Key parameters for fine-tuning:

- **Model**: `unsloth/gemma-3-270m-it` (270M parameters)
- **LoRA Rank**: 16-32 (adjustable)
- **Learning Rate**: 1e-4 to 2e-4
- **Batch Size**: 4-8 (depending on available memory)
- **Max Steps**: 1000-2000 for full training

## Configuration

### Model Configuration

The system supports multiple model configurations:

```yaml
model:
  pretrained_model_path: "models/unsloth-gemma-3-270m-it/..."
  max_seq_length: 2048
  dtype: "float16"

lora:
  r: 32
  lora_alpha: 64
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
  dropout: 0.05
```

### Training Configuration

```yaml
training:
  output_dir: "checkpoints/lora_full"
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 8
  learning_rate: 1e-4
  max_steps: 2000
  logging_steps: 50
  save_steps: 200
```

## Testing

### Run Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_inference.py
```

### Evaluation

```bash
# Run chess evaluation
python src/evaluation/chess_evaluation.py

# Compare model performance
python compare_chess_qa.py
```

## Performance

### Training Performance

- **Memory Usage**: ~4-6GB VRAM (with MPS)
- **Training Speed**: ~2-3 steps/second on M3 MacBook Pro
- **Convergence**: Loss typically drops from 3.0 to 1.2 in 100 steps

### Model Performance

- **Chess Relevance**: ~85% for chess-specific questions
- **Move Accuracy**: ~70% for basic chess moves
- **Response Quality**: Improved over base model

## Troubleshooting

### Common Issues

1. **MPS not available:**
   ```bash
   # Check MPS availability
   python -c "import torch; print(torch.backends.mps.is_available())"
   ```

2. **Out of memory:**
   - Reduce batch size in config
   - Enable gradient checkpointing
   - Use CPU fallback

3. **Stockfish not found:**
   ```bash
   # Install Stockfish
   brew install stockfish
   # Or set custom path in chess_engine.py
   ```

### Debug Mode

Enable debug logging:

```bash
export PYTHONPATH=$PWD
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run code formatting
black src/ tests/
flake8 src/ tests/
```

## Roadmap

### Completed
- [x] Basic LoRA fine-tuning pipeline
- [x] Chess engine integration
- [x] Web interface
- [x] Multiple training configurations
- [x] Evaluation framework

### In Progress
- [ ] Enhanced dataset generation
- [ ] Improved evaluation metrics
- [ ] Better error handling

### Planned
- [ ] Multi-model support
- [ ] Advanced chess analysis
- [ ] Mobile app interface
- [ ] Cloud deployment options

### Future Development
For detailed information about the evolution to GemmaFischer (chess engine + tutor), see [PROJECT_PLAN.md](docs/PROJECT_PLAN.md).

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Google**: For the Gemma-3 model
- **Unsloth**: For the efficient fine-tuning framework
- **Stockfish**: For the chess engine
- **Hugging Face**: For the transformers library
- **Thytu**: For the ChessInstruct dataset

## Support

- **Issues**: Report bugs and feature requests on GitHub
- **Discussions**: Join community discussions
- **Documentation**: Check the `docs/` folder for detailed guides

---

**ChessGemma** - A chess AI system for question answering and analysis.