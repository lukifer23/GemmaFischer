# ChessGemma API Reference

## Overview

This document provides comprehensive API reference for all ChessGemma components, including training, inference, evaluation, and web interface modules.

## Training API

### `src/training/train.py`

Main training script for LoRA fine-tuning.

#### Command Line Interface

```bash
python src/training/train.py [OPTIONS]
```

**Options:**
- `--do_train`: Enable training mode (default: False)
- `--max_steps`: Maximum training steps (default: 10)
- `--config`: Path to configuration file
- `--resume_from_checkpoint`: Resume from specific checkpoint
- `--output_dir`: Output directory for checkpoints
- `--learning_rate`: Learning rate for training
- `--batch_size`: Training batch size

**Examples:**
```bash
# Smoke test
python src/training/train.py --do_train --max_steps 10

# Full training with config
python src/training/train.py --config src/training/configs/lora_full.yaml

# Resume training
python src/training/train.py --resume_from_checkpoint checkpoints/lora_full/checkpoint-1000
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

## Inference API

### `src/inference/inference.py`

Main inference engine for chess Q&A.

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
            Dictionary with response, confidence, and metadata
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

Flask web application for chess Q&A.

#### Routes

##### `GET /`
Main page with chess Q&A interface.

**Response**: HTML page with chess interface

##### `POST /api/ask`
API endpoint for chess questions.

**Request Body**:
```json
{
    "question": "What is the best opening move for White?",
    "context": "Optional context for the question"
}
```

**Response**:
```json
{
    "response": "The best opening move for White is e4...",
    "confidence": 0.85,
    "model_loaded": true,
    "question": "What is the best opening move for White?"
}
```

##### `GET /api/health`
Health check endpoint.

**Response**:
```json
{
    "status": "healthy",
    "model_status": "loaded",
    "timestamp": 1640995200.0
}
```

##### `GET /api/examples`
Get example chess questions.

**Response**:
```json
{
    "examples": [
        "What is the best opening move for White and why?",
        "Explain the concept of castling in chess.",
        "How should I evaluate material versus initiative in the middlegame?"
    ]
}
```

##### `GET /api/model_info`
Get information about the loaded model.

**Response**:
```json
{
    "model_type": "ChessGemma (Gemma-3 270M fine-tuned)",
    "fine_tuned_for": "Chess Q&A and analysis",
    "capabilities": [
        "Opening analysis",
        "Tactical explanations",
        "Strategic concepts"
    ],
    "limitations": [
        "No real-time engine analysis",
        "Limited to text-based responses"
    ]
}
```

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
