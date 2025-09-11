# ChessGemma API Reference

## Overview

This document provides comprehensive API reference for ChessGemma, a multi-expert chess AI system with web interface, training controls, and evaluation capabilities. The system supports UCI-compatible engine play, educational analysis, and real-time Q&A through specialized expert models.

**Current Status**: Full web interface at http://localhost:5001 with REST API, multi-expert training support, and comprehensive evaluation tools.

## Training API

### Expert Training System (`src/training/train_lora_poc.py`)

Main expert-aware training system supporting UCI, Tutor, and Director modes.

#### Command Line Interface

```bash
python src/training/train_lora_poc.py --expert [EXPERT] [OPTIONS]
```

**Expert Options:**
- `--expert uci`: Train UCI Engine Expert (chess move generation)
- `--expert tutor`: Train Chess Tutor Expert (educational analysis)
- `--expert director`: Train Q&A Director Expert (reasoning and tactics)

**Training Options:**
- `--config auto`: Auto-select configuration based on expert
- `--max_steps_override N`: Override default max steps (default: 1000)
- `--disable_eval`: Skip evaluation during training
- `--resume_from_checkpoint PATH`: Resume from specific checkpoint

**Examples:**
```bash
# Train UCI Expert (recommended)
python src/training/train_lora_poc.py --expert uci --config auto --max_steps_override 1000 --disable_eval

# Train Tutor Expert
python src/training/train_lora_poc.py --expert tutor --config auto --max_steps_override 1000 --disable_eval

# Train Director Expert
python src/training/train_lora_poc.py --expert director --config auto --max_steps_override 1000 --disable_eval

# Resume training from checkpoint
python src/training/train_lora_poc.py --expert uci --config auto --resume_from_checkpoint checkpoints/lora_uci/checkpoint-600 --max_steps_override 1000 --disable_eval
```

#### Python API

```python
from src.training.train import main

# Run training programmatically
main(do_train=True, max_steps=1000)
```

### Configuration Files

#### `src/training/configs/lora.yaml`
Basic LoRA configuration template.

```yaml
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

#### `src/training/configs/lora_full.yaml`
Complete training configuration.

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
```

#### Dataset Mixer (optional)
Provide a weighted mixture instead of a single dataset path:

```yaml
datasets:
  - path: "data/finetune/chess_finetune_refined.jsonl"
    weight: 0.3
  - path: "data/datasets/lichess_puzzles_1000_2000.jsonl"
    weight: 0.7
```

#### Curriculum Phases (optional)
Sequential phases with per-phase steps and mixes:

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

## Inference API

### `src/inference/inference.py`

Main inference engine for chess Q&A with dual-mode prompting (engine/tutor).

#### `ChessGemmaInference` Class

```python
class ChessGemmaInference:
    def __init__(self, model_path: str = None, adapter_path: str = None):
        """Initialize ChessGemma inference.
        
        Args:
            model_path: Path to base model (default: auto-detect)
            adapter_path: Path to LoRA adapter (default: latest checkpoint)
        """
    
    def load_model(self) -> bool:
        """Load the model and tokenizer.
        
        Returns:
            bool: True if successful, False otherwise
        """
    
    def generate_response(
        self,
        question: str,
        context: Optional[str] = None,
        mode: str = "tutor",
        max_length: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Dict[str, Any]:
        """Generate a response to a chess question.
        
        Args:
            question: The chess question to answer
            max_length: Maximum response length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Dictionary with response, confidence, and metadata. In `mode="engine"`, the response is post-processed to a single legal UCI move when possible (adds `postprocessed: true`).
        """
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
```

#### Convenience Functions

```python
def run_inference(question: str, **kwargs) -> Dict[str, Any]:
    """Convenience function to run inference with a question.
    
    Args:
        question: The chess question to answer
        **kwargs: Additional arguments for generate_response
        
    Returns:
        Response dictionary
    """

def load_model() -> bool:
    """Load the model if not already loaded.
    
    Returns:
        bool: True if successful, False otherwise
    """

def get_model_info() -> Dict[str, Any]:
    """Get model information.
    
    Returns:
        Dictionary with model information
    """
```

#### Usage Examples

```python
from src.inference.inference import ChessGemmaInference, run_inference

# Method 1: Using class directly
inference = ChessGemmaInference()
if inference.load_model():
    response = inference.generate_response("What is the best opening move for White?")
    print(response['response'])

# Method 2: Using convenience functions
response = run_inference("Explain castling in chess", max_length=150)
print(f"Response: {response['response']}")
print(f"Confidence: {response['confidence']:.2f}")
```

### `src/inference/chess_engine.py`

Chess engine integration with Stockfish.

#### `ChessEngineManager` Class

```python
class ChessEngineManager:
    def __init__(self, engine_path: str = "/opt/homebrew/bin/stockfish", debug: bool = False):
        """Initialize chess engine with comprehensive error handling.
        
        Args:
            engine_path: Path to Stockfish executable
            debug: Enable debug logging
        """
    
    def validate_move(self, fen: str, move: str) -> MoveAnalysis:
        """Validate a move and provide comprehensive analysis.
        
        Args:
            fen: FEN position string
            move: Move in UCI notation
            
        Returns:
            MoveAnalysis object with validation results
        """
    
    def analyze_position(self, fen: str, depth: int = 15, time_limit: float = 1.0) -> PositionAnalysis:
        """Provide comprehensive position analysis.
        
        Args:
            fen: FEN position string
            depth: Analysis depth
            time_limit: Time limit in seconds
            
        Returns:
            PositionAnalysis object with analysis results
        """
    
    def cleanup(self) -> None:
        """Clean up engine resources."""
```

#### Data Classes

```python
@dataclass
class MoveAnalysis:
    """Comprehensive move analysis result."""
    move: str
    is_legal: bool
    is_best: bool
    centipawn_score: Optional[int] = None
    mate_in: Optional[int] = None
    principal_variation: List[str] = field(default_factory=list)
    depth: int = 0
    time_taken: float = 0.0
    nodes_searched: int = 0
    engine_score: Optional[float] = None
    move_quality: str = "unknown"
    explanation: str = ""

@dataclass
class PositionAnalysis:
    """Complete position analysis."""
    fen: str
    best_move: Optional[str] = None
    best_score: Optional[int] = None
    mate_in: Optional[int] = None
    evaluation: Dict[str, Any] = field(default_factory=dict)
    top_moves: List[MoveAnalysis] = field(default_factory=list)
    threats: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)
    position_type: str = "middle_game"
```

#### Convenience Functions

```python
def validate_chess_move(fen: str, move: str) -> MoveAnalysis:
    """Convenience function for single move validation."""

def analyze_chess_position(fen: str) -> PositionAnalysis:
    """Convenience function for position analysis."""

def generate_position_explanation(fen: str) -> str:
    """Convenience function for natural language analysis."""
```

#### Usage Examples

```python
from src.inference.chess_engine import ChessEngineManager, validate_chess_move

# Method 1: Using context manager
with ChessEngineManager() as engine:
    analysis = engine.validate_move("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2e4")
    print(f"Move: {analysis.move}")
    print(f"Legal: {analysis.is_legal}")
    print(f"Quality: {analysis.move_quality}")

# Method 2: Using convenience function
analysis = validate_chess_move("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2e4")
print(analysis.explanation)
```

## Evaluation API

### `src/evaluation/chess_evaluation.py`

Chess-specific evaluation framework.

#### `ChessEvaluator` Class

```python
class ChessEvaluator:
    def __init__(self, model_path: str, adapter_path: str = None):
        """Initialize evaluator with model and optional adapter.
        
        Args:
            model_path: Path to base model
            adapter_path: Path to LoRA adapter (optional)
        """
    
    def evaluate_position(self, prompt: str, max_tokens: int = 100) -> Dict[str, Any]:
        """Evaluate a single chess position/question.
        
        Args:
            prompt: Chess question or position
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with evaluation metrics
        """
    
    def evaluate_test_set(self, test_questions: List[str], output_file: str = None) -> Dict[str, Any]:
        """Evaluate a set of chess questions.
        
        Args:
            test_questions: List of chess questions
            output_file: Optional output file for results
            
        Returns:
            Dictionary with aggregated evaluation results
        """
    
    def cleanup(self):
        """Clean up resources."""
```

#### Usage Examples

```python
from src.evaluation.chess_evaluation import ChessEvaluator

# Initialize evaluator
evaluator = ChessEvaluator("models/gemma-3-270m", "checkpoints/lora_full/checkpoint-1000")

# Evaluate single question
result = evaluator.evaluate_position("What is the best opening move for White?")
print(f"Chess relevance: {result['chess_relevance_score']:.2f}")

# Evaluate test set
test_questions = [
    "What is the best opening move for White?",
    "Explain castling in chess",
    "What is a fork in chess?"
]
results = evaluator.evaluate_test_set(test_questions, "evaluation_results.json")
print(f"Average relevance: {results['average_chess_relevance']:.2f}")

# Cleanup
evaluator.cleanup()
```

## Web Interface API

### `src/web/app.py`

Comprehensive Flask web application with training controls, evaluation tools, and interactive chess analysis.

**Base URL**: http://localhost:5001

### Core Chess Analysis Endpoints

#### `GET /`
Main chess analysis interface with interactive board.

**Response**: HTML page with full chess analysis interface

#### `POST /api/game/analyze`
Analyze a chess position for move suggestions.

**Request Body**:
```json
{
    "square": "c2",
    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
}
```

**Response**:
```json
{
    "piece": "White Pawn",
    "legal_moves": ["c2c3", "c2c4"],
    "advice": ["The White pawn on c2 can move forward", "This pawn can move one or two squares forward"],
    "success": true
}
```

#### `POST /api/game/move`
Execute a chess move on the board.

**Request Body**:
```json
{
    "move": "c2c4",
    "current_fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
}
```

**Response**:
```json
{
    "success": true,
    "new_fen": "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq - 0 1",
    "current_player": "black"
}
```

#### `POST /api/game/ai_move`
Get AI move suggestion for current position.

**Request Body**:
```json
{
    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "expert": "uci"
}
```

**Response**:
```json
{
    "move": "e2e4",
    "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "success": true
}
```

#### `GET /api/game/state`
Get current game state.

**Response**:
```json
{
    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "current_player": "white",
    "legal_moves": ["g1h3", "g1f3", "b1c3", "b1a3", "h2h3", "g2g3", "f2f3", "e2e3", "d2d3", "c2c3", "b2b3", "a2a3", "h2h4", "g2g4", "f2f4", "e2e4", "d2d4", "c2c4", "b2b4", "a2a4"],
    "game_status": "active"
}
```

### Training API Endpoints

#### `POST /api/train/start`
Start expert training session.

**Request Body**:
```json
{
    "expert": "uci",
    "max_steps": 1000,
    "instruction_collator": false,
    "disable_eval": true
}
```

**Response**:
```json
{
    "success": true,
    "message": "Training started for UCI expert",
    "checkpoint_dir": "checkpoints/lora_uci"
}
```

#### `GET /api/train/status`
Get training progress and status.

**Response**:
```json
{
    "running": true,
    "current_step": 450,
    "max_steps": 1000,
    "elapsed_time": "15m 32s",
    "checkpoint_dir": "checkpoints/lora_uci",
    "loss": 1.85,
    "learning_rate": 8.5e-5,
    "cpu_percent": 25.0,
    "memory_usage": "4.2GB"
}
```

#### `POST /api/train/stop`
Stop current training session.

**Response**:
```json
{
    "success": true,
    "message": "Training stopped successfully"
}
```

### Evaluation API Endpoints

#### `POST /api/eval/stockfish`
Run Stockfish match evaluation.

**Request Body**:
```json
{
    "depth": 12,
    "games": 100
}
```

**Response**:
```json
{
    "success": true,
    "message": "Stockfish evaluation started",
    "output_file": "stockfish_match_after.json"
}
```

#### `POST /api/eval/puzzles`
Run puzzle accuracy evaluation.

**Request Body**:
```json
{
    "limit": 200,
    "expert": "uci"
}
```

**Response**:
```json
{
    "success": true,
    "message": "Puzzle evaluation started",
    "output_file": "eval_report_after.json"
}
```

#### `GET /api/eval/status`
Get evaluation progress.

**Response**:
```json
{
    "running": true,
    "progress": 65,
    "current_game": 65,
    "total_games": 100,
    "accuracy": 0.02
}
```

### Dataset Management Endpoints

#### `POST /api/data/clean`
Start dataset cleaning and validation.

**Request Body**:
```json
{
    "mode": "uci",
    "relabel_with_stockfish": true
}
```

**Response**:
```json
{
    "success": true,
    "message": "Dataset cleaning started",
    "output_file": "data/processed/uci_clean.jsonl"
}
```

#### `GET /api/data/status`
Get dataset processing status.

**Response**:
```json
{
    "running": false,
    "progress": 100,
    "processed": 50000,
    "valid": 49850,
    "invalid": 150
}
```

### System Status Endpoints

#### `GET /api/health`
System health check.

**Response**:
```json
{
    "status": "healthy",
    "model_loaded": true,
    "experts_available": ["uci", "tutor", "director"],
    "timestamp": "2025-09-11T00:45:23.456789Z"
}
```

#### `GET /api/stats`
Get system performance statistics.

**Response**:
```json
{
    "cpu_percent": 15.2,
    "memory_used": "4.1GB",
    "memory_total": "16GB",
    "disk_used": "45GB",
    "uptime": "2h 15m",
    "active_processes": 8
}
```

#### `GET /api/examples`
Get example chess questions and positions.

**Response**:
```json
{
    "examples": [
        "What is the best opening move for White?",
        "Explain castling in chess",
        "Analyze this position: r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    ]
}
```

#### Runtime Metrics
Training logs include CPU and RAM usage per step for quick monitoring on macOS MPS.

#### Usage Examples

```python
import requests

# Start web server
# python src/web/app.py

# Ask a question
response = requests.post('http://localhost:5000/api/ask', json={
    'question': 'What is the best opening move for White?'
})
result = response.json()
print(result['response'])

# Check health
health = requests.get('http://localhost:5000/api/health')
print(health.json())

# Get examples
examples = requests.get('http://localhost:5000/api/examples')
print(examples.json()['examples'])
```

## Data Processing API

### Dataset Preparation

#### `data/prepare_dataset.py`

```python
def convert_and_save(output_dir: str, full: bool = False):
    """Convert ChessInstruct dataset to chat format.
    
    Args:
        output_dir: Output directory for processed data
        full: Process full dataset (default: False, uses sample)
    """
```

#### `create_finetune_dataset.py`

```python
# Creates fine-tuning dataset from initial Q&A
# Output: data/datasets/chess_finetune.jsonl
```

## Utility Scripts

### `scripts/compare_chess_qa.py`

Compare model responses before and after training.

```bash
python scripts/compare_chess_qa.py
```

### `scripts/adapter_integrity_test.py`

Test LoRA adapter integrity.

```bash
python scripts/adapter_integrity_test.py
```

## Error Handling

### Common Exceptions

```python
class ChessGemmaError(Exception):
    """Base exception for ChessGemma errors."""
    pass

class ModelLoadError(ChessGemmaError):
    """Raised when model loading fails."""
    pass

class ChessEngineError(ChessGemmaError):
    """Raised when chess engine operations fail."""
    pass

class InferenceError(ChessGemmaError):
    """Raised when inference fails."""
    pass
```

### Error Response Format

```json
{
    "error": "Error message",
    "response": "Fallback response",
    "confidence": 0.0
}
```

## Configuration Reference

### Environment Variables

- `PYTORCH_ENABLE_MPS_FALLBACK=1`: Enable CPU fallback for MPS
- `STOCKFISH_PATH=/path/to/stockfish`: Custom Stockfish path
- `MODEL_CACHE_DIR=/path/to/cache`: Model cache directory

### Model Configuration

```python
MODEL_CONFIG = {
    "model_name": "unsloth/gemma-3-270m-it",
    "max_seq_length": 2048,
    "dtype": "float16",
    "load_in_4bit": False,
    "full_finetuning": False
}
```

### LoRA Configuration

```python
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "bias": "none",
    "use_gradient_checkpointing": "unsloth"
}
```

## Performance Considerations

### Memory Usage
- Base model: ~2GB
- LoRA adapter: ~50MB
- Training: ~4-6GB VRAM
- Inference: ~2-3GB VRAM

### Speed Benchmarks
- Training: ~2-3 steps/second on M3 MacBook Pro
- Inference: ~100-200 tokens/second
- Move validation: ~50-100 moves/second

### Optimization Tips
- Use model caching for repeated inference
- Enable gradient checkpointing for training
- Use appropriate batch sizes for your hardware
- Consider quantization for deployment
