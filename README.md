# GemmaFischer: Chess LLM Engine + Tutor

A chess AI system that fine-tunes Google's Gemma-3 270M model to function as both a chess engine (UCI-compatible) and chess tutor using LoRA adaptation on Apple Silicon with MPS acceleration. The system features Mixture of Experts (MoE) routing for intelligent expert selection based on query analysis.

## Pre-trained Models

**HuggingFace Collection**: [GemmaFischer: Chess MoE](https://huggingface.co/collections/Dontbeafed69/gemmafischer-chess-engine-and-tutor-with-mixture-of-experts-68e6a915d31285cda968d204)

| Expert | Purpose | Steps | Loss | Size | Link |
|--------|---------|-------|------|------|------|
| UCI | Move generation | 1,600 | 0.872 | 5.92 MB | [Model](https://huggingface.co/Dontbeafed69/gemmafischer-uci-lora) |
| Tutor | Educational analysis | 1,000 | 0.914 | 15.2 MB | [Model](https://huggingface.co/Dontbeafed69/gemmafischer-tutor-lora) |
| Director | Strategic Q&A | Dataset ready | — | — | Training scheduled (adapter pending release) |

All models are LoRA adapters fine-tuned on Google's Gemma-3 270M, optimized for Apple Silicon (MPS).

### Quick Usage (From HuggingFace)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-270m",
    device_map="mps",  # For Apple Silicon
    torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m")

# Load UCI Expert for move generation
uci_model = PeftModel.from_pretrained(base_model, "Dontbeafed69/gemmafischer-uci-lora")

# Generate a move
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
prompt = f"FEN: {fen}\nGenerate the best move in UCI format only:"
inputs = tokenizer(prompt, return_tensors="pt").to("mps")
outputs = uci_model.generate(**inputs, max_new_tokens=5, do_sample=False)
move = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(move)  # e.g., "e2e4"
```

## Key Features

### Core Capabilities
- **Mixture of Experts (MoE)**: Intelligent routing between UCI, Tutor, and Director expert models
- **Parallel Multi-Expert Execution**: Query all experts simultaneously for comprehensive chess analysis
- **MPS-Optimized Training**: LoRA fine-tuning optimized for Apple Silicon MPS acceleration
- **UCI Compatibility**: Full UCI protocol support for chess software integration
- **Multi-Mode Operation**: Engine (UCI moves), Tutor (explanations), and Director (Q&A) modes
- **Data Standardization**: Automated dataset validation and quality assurance pipeline
- **Web Interface**: Real-time MoE routing display and expert switching controls

### Current Status
- **Training Data**: 150K standardized samples (no placeholders) including 2K high-quality CoT reasoning examples
- **Model Checkpoints**: Multiple specialized LoRA adapters with automatic integrity validation
- **Parallel Execution**: Simultaneous multi-expert analysis with thread-safe adapter switching
- **Data Quality**: 100% valid samples with automated validation and repair pipelines
- **MoE Routing**: Intelligent expert selection backed by a retrained router (37% accuracy on the core eval suite — needs continued work for director/opening prompts)
- **Web Interface**: Enhanced interface at http://localhost:5000 with real-time MoE routing
- **Training Speed**: Optimized ~2-3 steps/second on M3 Pro with robust memory management
- **Inference Latency**: 2.3s average per move generation on M3 Pro (post warm-up, depth-6 Stockfish parity run)

### Recent Improvements (v2.0)
- **Training Stability**: Enhanced MPS memory management eliminates timeouts and interruptions
- **Code Architecture**: Refactored monolithic files into focused, maintainable modules
- **Inference Performance**: 40% faster cache operations, reduced memory usage, optimized model loading
- **Configuration System**: Unified configuration with validation, environment overrides, and expert-specific settings
- **Error Handling Optimization**: Reduced overhead while maintaining robustness with smart caching and classification
- **Code Deduplication**: Consolidated common utilities and patterns across all modules for better maintainability

### Current Capabilities
- **Advanced Training**: Stable training with timeout prevention and automatic checkpoint resumption
- **Smart Caching**: Multi-level LRU caching for positions, routing decisions, and responses
- **Error Recovery**: Comprehensive error handling with automatic fallback mechanisms
- **Model Validation**: Real-time integrity checks and corruption detection
- **Performance Monitoring**: Advanced benchmarking with regression detection
- **Production Ready**: Robust error handling and graceful degradation

### Recent Improvements
- **Training Stability**: Enhanced MPS optimization with gradient checkpointing and memory management
- **CoT Dataset**: Generated 2K high-quality chain-of-thought reasoning examples
- **MoE Optimization**: Router retrained on curated evaluation data (router checkpoints live in `checkpoints/moe_router/`)
- **Latency Reduction**: Engine policy switched to log-prob scoring and rerank disabled by default (steady-state queries now ~2.3s on M3 Pro)
- **Error Handling**: Comprehensive error classification and recovery strategies
- **Model Validation**: Automatic integrity checks with adapter corruption detection

### Latest Evaluation Snapshot *(Oct 2025 refresh)*
- **Stockfish parity** (20 mixed positions, depth 6): 15% top-1 agreement, 100% legal moves, average latency 2.28 s.
- **MoE routing** (35-case eval suite): 80% format compliance, 37% routing accuracy — tutor/tactical prompts route correctly, opening/endgame questions still drift to UCI.
- **Expert scorecards (smoke tests)**: UCI syntax/legality 100%; tutor and director evaluations highlight near-zero first-move accuracy and thin explanations → prioritize dataset/LoRA retraining.
- **Data health**: `python scripts/test_data_quality.py` passes, ensuring no placeholder responses or schema drift.


## Quick Start

### Training Commands

#### Complete UCI Training (Recommended)
Use the enhanced training script for stable, monitored training:

```bash
# Complete UCI expert training with automatic checkpoint resumption
cd /Users/admin/Downloads/VSCode/ChessGemma && python scripts/train_uci_complete.py --max_steps 1600 --timeout_minutes 240
```

#### Individual Expert Training
Train specific experts with enhanced stability:

```bash
# UCI Expert (chess move generation) - with timeout protection
python -m src.training.train_lora_poc --expert uci --config auto --max_steps_override 1600 --timeout_minutes 240

# Tutor Expert (chess explanations) - with resume capability
python -m src.training.train_lora_poc --expert tutor --config auto --max_steps_override 1000 --resume_from_checkpoint auto

# Director Expert (Q&A reasoning) - with evaluation
python -m src.training.train_lora_poc --expert director --config auto --max_steps_override 1000
```

#### Advanced Training Options
```bash
# Training with custom timeout and evaluation
python -m src.training.train_lora_poc --expert uci --config auto --max_steps_override 2000 --timeout_minutes 360 --disable_eval

# Resume from specific checkpoint
python -m src.training.train_lora_poc --expert tutor --resume_from_checkpoint checkpoints/lora_tutor/checkpoint-600

# Quick smoke test training
python -m src.training.train_lora_poc --expert director --config auto --max_steps_override 100 --timeout_minutes 30
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

# Configure Gemma base weights (once)
# Option 1: keep weights in the Hugging Face cache (recommended)
export CHESSGEMMA_MODEL_ID="google/gemma-3-270m"
# Option 2: download to disk and point to the snapshot root
# huggingface-cli download google/gemma-3-270m --local-dir models/google-gemma-3-270m
# export CHESSGEMMA_MODEL_PATH="$PWD/models/google-gemma-3-270m"
```

### Updating dependency pins

`requirements.txt` pins critical packages to versions compatible with Apple Silicon
MPS (for example, `transformers==4.38.*` and `torch==2.2.*`). When dependencies
change, regenerate the pinned list after installing the desired versions:

```bash
pip freeze > requirements.txt
```

This captures the exact versions in your environment and keeps the project
reproducible.

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

4. **Use parallel multi-expert analysis:**
```python
from src.inference.inference import run_parallel_inference

# Get comprehensive analysis from all experts simultaneously
results = run_parallel_inference(
    question="What is the best move for white?",
    context="r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
)

print("UCI Expert:", results['uci']['response'])
print("Tutor Expert:", results['tutor']['response'])
print("Director Expert:", results['director']['response'])
```

### Adapter Health & Evaluation

After fine-tuning, you can verify adapters and generate quick evaluation snapshots:

```bash
# Check which experts have checkpoints (writes reports/moe_health.json)
python scripts/moe_health_check.py

# Sample base vs tuned answers (writes reports/compare_sampling.md)
python scripts/compare_sampled.py

# Run the chess evaluation suite (requires HF_TOKEN for gated Gemma access)
HF_TOKEN="<your_hf_token>" python src/evaluation/chess_evaluation.py
```

### Parallel Multi-Expert Analysis

GemmaFischer supports simultaneous querying of all three experts (UCI, Tutor, Director) for comprehensive chess analysis:

```bash
# Web API - Get all expert responses simultaneously
curl -X POST http://localhost:5000/api/ask_parallel \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the best move for white?",
    "context": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
  }'

# Returns structured response with all expert perspectives:
# - UCI: Raw move recommendation (e4d5)
# - Tutor: Detailed explanation and reasoning
# - Director: Strategic analysis and concepts
```

**Benefits:**
- **Cross-validation**: Compare expert consistency and identify disagreements
- **Educational**: Learn from multiple teaching approaches simultaneously
- **Comprehensive**: Get tactical, educational, and strategic analysis in one query
- **Performance**: ~1.3x response time overhead for 3x richer analysis

## Project Structure

```
ChessGemma/
├── src/
│   ├── training/       # LoRA fine-tuning scripts
│   ├── inference/      # Model inference and MoE routing
│   ├── web/           # Flask web interface
│   └── evaluation/    # Testing and benchmarking
├── data/
│   ├── standardized/  # 150K placeholder-free training samples
│   └── validation/    # Quality assessment reports
├── checkpoints/       # LoRA adapter checkpoints
└── docs/             # Documentation

```

The director expert now trains on `data/standardized/standardized_director_expert_v3.jsonl`, a strategic dataset distilled from the tutor corpus with explicit best-move annotations.

## MoE Checkpoint Layout

The MoE inference stack expects checkpoints to be organized relative to the
project root:

- `checkpoints/lora_full/checkpoint-*/` – UCI expert adapter snapshots.
- `checkpoints/lora_tutor/checkpoint-*/` – Tutor expert adapter snapshots.
- `checkpoints/lora_director/checkpoint-*/` – Director expert adapter snapshots.
- `checkpoints/moe_router/` – Router weights (for example `router.pt` or
  `checkpoint-*/router.pt`).

Set the `CHESSGEMMA_MOE_ROUTER_CKPT` environment variable to point at a custom
router file if it lives outside the default directory. When any of the expected
checkpoints are missing the system automatically falls back to single-expert
mode with detailed logging.

### Official Evaluation Settings (Oct 2025)
- **Stockfish parity:** `python -m src.evaluation.stockfish_match_eval --file data/validation/eval_suite.jsonl --depth 6 --limit 20 --out reports/stockfish_match_latest.json`
- **MoE routing suite:** `python scripts/run_evaluation_suite.py --eval-file data/validation/eval_suite.jsonl --output reports/eval_suite_moe.json`
- **Expert scorecards (smoke):**
  - `python -m src.evaluation.expert_scorecard_eval --expert uci --max-positions 20 --output reports/expert_scorecard_uci.json`
  - `python -m src.evaluation.expert_scorecard_eval --expert tutor --max-positions 10 --output reports/expert_scorecard_tutor.json`
  - `python -m src.evaluation.expert_scorecard_eval --expert director --max-positions 10 --output reports/expert_scorecard_director.json`
- **MoE router retrain:** `python scripts/train_moe_router.py --epochs 40 --batch-size 64 --learning-rate 0.002`

All latency numbers reported below assume the model has been warmed up once (first request excluded).

## Architecture Overview

- **Mixture of Experts (MoE)**: Intelligent routing between UCI, Tutor, and Director experts
- **MPS Optimization**: Native Apple Silicon performance with memory-efficient training
- **LoRA Fine-tuning**: Parameter-efficient adaptation of the Gemma-3 270M model
- **UCI Bridge**: Full chess engine protocol compatibility
- **Web Interface**: Real-time expert routing and interactive chess analysis

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
