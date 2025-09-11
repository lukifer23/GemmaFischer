# GemmaFischer: Chess LLM Engine + Tutor

A comprehensive chess AI system that fine-tunes Google's Gemma-3 270M model to function as both a chess engine (UCI-compatible) and a chess tutor/analyst using LoRA (Low-Rank Adaptation) on Apple Silicon with MPS acceleration. GemmaFischer represents a novel intersection of game AI and natural language processing, creating a compact yet powerful chess assistant that runs natively on Mac hardware.

## Project Overview

GemmaFischer combines several advanced components to create a comprehensive chess learning and analysis platform:

- **Fine-tuned Language Model**: Gemma-3 270M with LoRA adapters for chess knowledge
- **UCI Bridge Layer**: Full UCI protocol compatibility for chess software integration
- **Dual-Mode Operation**: Engine mode (fast moves) and Tutor mode (explanations)
- **Chain-of-Thought Reasoning**: Step-by-step analysis and explanation
- **Chess Engine Integration**: Stockfish integration for move validation and analysis
- **Web Interface**: Flask-based web application for interaction
- **Evaluation Framework**: Comprehensive chess-specific metrics and testing tools
- **Embedding-Based Retrieval**: Similar position lookup and context enhancement
- **Vision Module**: Board image to FEN conversion (planned)
- **Style Conditioning**: Historical player style emulation (planned)

## Key Features

### Core Capabilities
- **MPS-Optimized Training**: LoRA fine-tuning with Unsloth optimization for M3 Pro
- **Mac-Only Design**: Optimized exclusively for Apple Silicon with MPS acceleration
- **UCI Compatibility**: Full UCI protocol support for chess software integration
- **Dual-Mode Operation**: Engine mode (fast moves) and Tutor mode (explanations)
- **Chain-of-Thought Reasoning**: Step-by-step analysis and explanation
- **Style Conditioning**: Historical player style emulation (Fischer, aggressive, etc.)

### Advanced Features
- **Embedding-Based Retrieval**: Similar position lookup using vector search
- **Vision Module**: Board image to FEN conversion for real-world chess positions
- **Enhanced Analysis**: Blunder identification, tactical motif recognition, opening theory
- **Multi-Modal Support**: Text, image, and position-based interactions
- **Comprehensive Evaluation**: Automated benchmarking and performance metrics

### Technical Features
- **Chess Engine Integration**: Stockfish for move validation and position analysis
- **Web Interface**: Interactive chess Q&A interface with board visualization
- **Evaluation Tools**: Chess-specific metrics and testing framework
- **Modular Architecture**: Extensible design for easy feature additions

## Current Status

### Completed Milestones
- **Basic LoRA Pipeline**: Gemma-3 270M model fine-tuning with Unsloth optimization
- **Chess Engine Integration**: Stockfish UCI engine integration for move validation
- **Web Interface**: Flask-based web application with chess board visualization
- **UCI Bridge**: Full UCI protocol compatibility for chess software integration
- **Dual-Mode Operation**: Engine mode (fast moves) and Tutor mode (explanations)
- **Evaluation Framework**: Chess-specific metrics and testing tools
- **MPS Optimization**: Native Apple Silicon acceleration with Metal Performance Shaders

### Current Performance
- **Training Loss**: Improved from ~3.0 to ~1.2 in early epochs
- **Chess Relevance**: ~85% for chess-specific questions
- **Move Accuracy**: ~70% for basic chess moves
- **Training Speed**: ~2-3 steps/second on M3 Pro
- **Memory Usage**: ~4-6GB (MPS-optimized)

### Known Issues & Limitations
- **Dataset Quality**: Initial training data included overly long sequences with generic answers
- **Output Accuracy**: Model sometimes suggests illegal moves or incorrect explanations
- **Training Stability**: Issues with resume functionality and progress tracking
- **Evaluation Metrics**: Current evaluation is rudimentary and needs enhancement

### Next Priority Tasks
1. **Dataset Overhaul**: Curate high-quality, focused chess Q&A data
2. **Enhanced Evaluation**: Implement comprehensive benchmarking suite
3. **Embedding System**: Add similar position retrieval capabilities
4. **Vision Module**: Board image to FEN conversion
5. **Style Conditioning**: Historical player style emulation

## Quick Start

### Prerequisites

- **Mac with M3 Pro chip** (required for optimal MPS performance)
- Python 3.10+
- 16GB+ RAM (recommended for training)
- Stockfish chess engine (optional, for advanced features)
- macOS 12.0+ (for MPS support)

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

1. **Run inference (programmatic):**
```bash
python -c "from src.inference.inference import load_model,run_inference;print('load:',load_model());print(run_inference('Explain castling in chess')['response'])"
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

4. **Stockfish match evaluator (optional):**
```bash
python src/evaluation/stockfish_match_eval.py --file data/datasets/lichess_puzzles_1000_2000.jsonl --limit 50 --depth 8 --out stockfish_match_smoke.json
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

### Training Performance (M3 Pro)

- **Memory Usage**: ~4-6GB (MPS-optimized)
- **Training Speed**: ~2-3 steps/second on M3 Pro
- **Convergence**: Loss typically drops from 3.0 to 1.2 in 100 steps
- **MPS Acceleration**: All operations use Metal Performance Shaders
- **No CUDA/CPU Fallbacks**: Optimized exclusively for Apple Silicon

### Model Performance

- **Chess Relevance**: ~85% for chess-specific questions
- **Move Accuracy**: ~70% for basic chess moves
- **Response Quality**: Improved over base model

## Troubleshooting

### Common Issues (M3 Pro)

1. **MPS not available:**
   ```bash
   # Check MPS availability (should be True on M3 Pro)
   python -c "import torch; print(torch.backends.mps.is_available())"
   ```

2. **Out of memory:**
   - Reduce batch size in config
   - Enable gradient checkpointing
   - Reduce LoRA rank if needed

3. **Stockfish not found:**
   ```bash
   # Install Stockfish
   brew install stockfish
   # Or set custom path in chess_engine.py
   ```

4. **MPS performance issues:**
   - Ensure you're using M3 Pro (not M1/M2)
   - Check macOS version (12.0+ required)
   - Verify PyTorch MPS installation

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

## Comprehensive Roadmap

### Phase 1: Foundation (Completed)
- [x] Basic LoRA fine-tuning pipeline
- [x] Chess engine integration with Stockfish
- [x] Web interface with board visualization
- [x] UCI bridge for chess software compatibility
- [x] Dual-mode operation (engine/tutor)
- [x] Basic evaluation framework
- [x] MPS optimization for Apple Silicon

### Phase 2: Data & Quality (In Progress)
- [ ] **Dataset Overhaul**: Curate high-quality chess Q&A data
  - [ ] ChessInstruct v1.5 refinement
  - [ ] Lichess puzzle dataset integration
  - [ ] Annotated game commentary collection
  - [ ] Opening theory database integration
- [ ] **Enhanced Evaluation**: Comprehensive benchmarking suite
  - [ ] Move legality and syntax validation
  - [ ] Tactical puzzle success rate testing
  - [ ] Positional question answering accuracy
  - [ ] Stockfish match percentage analysis
- [ ] **Training Improvements**: Advanced fine-tuning strategies
  - [ ] Chain-of-thought reasoning integration
  - [ ] Multi-task learning optimization
  - [ ] Style conditioning implementation

### Phase 3: Advanced Features (Planned)
- [ ] **Embedding System**: Similar position retrieval
  - [ ] Vector database for chess positions
  - [ ] FAISS-based similarity search
  - [ ] Context enhancement for responses
- [ ] **Vision Module**: Board image processing
  - [ ] Chess piece detection and recognition
  - [ ] Board corner detection and correction
  - [ ] FEN generation from images
- [ ] **Enhanced Analysis**: Advanced chess capabilities
  - [ ] Blunder identification and explanation
  - [ ] Tactical motif recognition
  - [ ] Opening theory and naming
  - [ ] Endgame tablebase integration

### Phase 4: Polish & Deployment (Future)
- [ ] **Multi-Model Support**: Different model sizes and variants
- [ ] **Mobile Integration**: Core ML deployment for iOS
- [ ] **Performance Optimization**: Quantization and speed improvements
- [ ] **User Experience**: Advanced UI/UX features
- [ ] **Research Integration**: Academic collaboration and publication

### Research Inspirations
This project draws inspiration from:
- **ChessGPT (2023)**: Bridging policy learning and language modeling
- **Concept-Guided Chess Commentary (2024)**: Expert engine + LLM integration
- **Maia Chess**: Human-like playing styles and mistake understanding
- **Toolformer/ReAct**: LLM tool integration paradigms

For detailed implementation plans, see [PROJECT_PLAN.md](docs/PROJECT_PLAN.md).

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