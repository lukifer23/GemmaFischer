# GemmaFischer: Chess LLM Engine + Tutor

A chess AI system that combines Google's Gemma-3 270M model for strategic guidance and educational explanations with LeelaChess Zero (LC0) as the primary UCI chess engine. Uses LoRA adaptation on Apple Silicon with MPS acceleration and features a hybrid Mixture of Experts (MoE) system that intelligently routes between LC0's precise move calculation and the LLM's educational capabilities.

> **Note:** Runtime environment variables still use the historical `CHESSGEMMA_*` prefix for compatibility. The rest of the project has been renamed to **GemmaFischer**.

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
- **Hybrid LLM/LC0 Architecture**: LC0 provides precise UCI move generation while LLM handles strategic guidance and educational explanations
- **Intelligent MoE Routing**: Automatic selection between LC0 (for moves) and LLM experts (for analysis and education)
- **MPS-Optimized Performance**: LoRA fine-tuning and inference optimized for Apple Silicon with Metal acceleration
- **UCI Compatibility**: Full UCI protocol support with LC0 as the primary chess engine
- **Multi-Expert Operation**: UCI (LC0 moves), Tutor (educational analysis), Director (strategic Q&A) modes
- **Real-time Analysis**: LC0 neural engine analysis combined with LLM explanations
- **Interactive Web Interface**: Real-time hybrid analysis display with move visualization and expert feedback
- **Educational Focus**: LLM provides strategic context and explanations for LC0's precise move recommendations

### Current Status
- **Hybrid Architecture**: LC0 neural engine provides primary UCI move generation with LLM strategic guidance and educational explanations
- **Training Data**: 150K standardized samples optimized for LLM educational capabilities and strategic reasoning
- **Model Checkpoints**: Specialized LoRA adapters for Tutor (explanations) and Director (strategic analysis) modes
- **LC0 Integration**: Metal-accelerated LC0 neural engine with optimized configuration for M3 Pro performance
- **Data Quality**: `python scripts/validate_and_repair_datasets.py --generate --repair` consolidates dataset generation and validation; `python scripts/test_data_quality.py` provides additional assertions.
- **Intelligent Routing**: MoE system intelligently routes UCI moves to LC0 and educational queries to LLM experts
- **Web Interface**: Enhanced interface at http://localhost:5000 with real-time LC0 analysis and LLM explanations
- **Performance**: Optimized for M3 Pro with LC0 Metal backend and efficient LLM inference
- **Response Quality**: LC0 provides precise moves while LLM adds strategic context and educational value

### Recent Improvements (v2.1 - Hybrid Architecture)
- **LC0 Integration**: LC0 neural engine now serves as primary UCI engine with Metal backend optimization for M3 Pro
- **Hybrid System Architecture**: Redesigned MoE system to leverage LC0 for precise moves and LLM for strategic guidance
- **Performance Optimization**: Enhanced caching and memory management for LC0 + LLM hybrid processing
- **Configuration Updates**: Optimized settings for LC0 Metal backend and hybrid inference patterns
- **UCI Bridge Enhancement**: Updated UCI protocol handler to prioritize LC0 over LLM for move generation
- **Expert Manager Updates**: Modified expert system to use hybrid engine for UCI queries when available
- **Web Interface Integration**: Enhanced UI to display LC0 analysis alongside LLM explanations

### Current Capabilities
- **Hybrid UCI Engine**: LC0 neural engine provides precise move generation with LLM strategic explanations
- **Advanced Training**: Stable LoRA training optimized for educational and strategic reasoning tasks
- **Intelligent Caching**: Multi-level LRU caching for positions, hybrid responses, and expert routing decisions
- **Educational Integration**: LLM provides strategic context and explanations for LC0's precise move recommendations
- **Error Recovery**: Comprehensive error handling with LC0 → LLM → Stockfish fallback mechanisms
- **Model Validation**: Real-time integrity checks for both LLM adapters and LC0 engine health
- **Performance Monitoring**: Advanced benchmarking with hybrid system regression detection
- **Production Ready**: Robust error handling and graceful degradation across the hybrid architecture

### Recent Improvements
- **Training Stability**: Enhanced MPS optimization with gradient checkpointing and memory management
- **CoT Dataset**: Generated 2K high-quality chain-of-thought reasoning examples
- **MoE Optimization**: Router retrained on curated evaluation data (router checkpoints live in `checkpoints/moe_router/`)
- **Latency Reduction**: Engine policy switched to log-prob scoring and rerank disabled by default (steady-state queries now ~2.3s on M3 Pro)
- **Error Handling**: Comprehensive error classification and recovery strategies
- **Model Validation**: Automatic integrity checks with adapter corruption detection

### Latest Evaluation Snapshot *(Oct 2025 refresh - Hybrid Architecture)*
- **LC0 Performance** (20 mixed positions, depth 8): 50%+ Stockfish agreement, 100% legal moves, average latency 1.8s with Metal acceleration.
- **Hybrid System**: LC0 provides primary move generation while LLM adds strategic context and educational explanations.
- **MoE routing** (35-case eval suite): UCI moves routed to LC0, educational queries to LLM experts with improved confidence scoring.
- **Expert Integration**: UCI expert now uses LC0 hybrid engine, Tutor/Director experts focus on educational and strategic analysis.
- **System Health**: LC0 Metal backend + LLM inference optimized for M3 Pro with comprehensive fallback mechanisms.
- **Data Quality**: `python scripts/test_data_quality.py` passes, ensuring high-quality educational and strategic training data.


## Quick Start

### Training Commands

#### Complete UCI Training (Recommended)
Use the enhanced training script for stable, monitored training:

```bash
# Complete UCI expert training with automatic checkpoint resumption
cd /Users/admin/Downloads/VSCode/GemmaFischer && python scripts/train_uci_complete.py --max_steps 1600 --timeout_minutes 240
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
# Start web interface with hybrid LC0 pool
./run_hybrid_webapp.sh
# Visit: http://localhost:5000

# Or launch manually
python -m src.web.run_web_app

# Disable the LC0 pool if you want a fresh engine instance per session
GEMMAFISCHER_DISABLE_LC0_POOL=1 ./run_hybrid_webapp.sh
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
cd GemmaFischer

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
./run_hybrid_webapp.sh
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

### Environment Variables

- `CHESSGEMMA_MODEL_ID` / `CHESSGEMMA_MODEL_PATH`: point to the Gemma-3 270M base weights (HF hub ID or local snapshot).
- `CHESSGEMMA_MOE_ROUTER_CKPT`: override the default MoE router checkpoint location.
- `CHESSGEMMA_LC0_USE_POOL`: set to `0` to disable the shared LC0 engine pool (the launcher sets this automatically when `GEMMAFISCHER_DISABLE_LC0_POOL=1`).
- `CHESSGEMMA_DEBUG`: enable verbose logging when set to `1`, `true`, etc.
- `GEMMAFISCHER_DISABLE_LC0_POOL`: convenience flag for `run_hybrid_webapp.sh`; when set to `1` the script exports `CHESSGEMMA_LC0_USE_POOL=0` before starting the server.

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
GemmaFischer/
├── src/
│   ├── training/       # LoRA fine-tuning scripts
│   ├── inference/      # Model inference and MoE routing
│   ├── web/           # Flask web interface
│   └── evaluation/    # Testing and benchmarking
├── data/
│   ├── standardized/  # 150K placeholder-free training samples
│   └── validation/    # Quality assessment reports
├── checkpoints/       # LoRA adapter checkpoints
├── run_hybrid_webapp.sh  # Hybrid launcher (LC0 + LLM web UI)
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

- **Hybrid LLM/LC0 System**: LC0 neural engine for precise UCI move generation, Gemma-3 LLM for strategic guidance and educational explanations
- **Intelligent MoE Routing**: Automatic selection between LC0 (moves) and LLM experts (analysis/education) based on query type
- **MPS Optimization**: Native Apple Silicon performance with Metal-accelerated LC0 and efficient LLM inference
- **LoRA Fine-tuning**: Parameter-efficient adaptation of Gemma-3 270M for educational and strategic reasoning
- **UCI Bridge**: Full chess engine protocol compatibility with LC0 as primary engine
- **Interactive Web Interface**: Real-time LC0 analysis display with LLM explanations and educational feedback

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
